"""Locate / Navigate / Extract tools for MCP context records.

This module hosts the read-only navigation tools that let an agent work with
stored context the way it works with a codebase: locate exact text, then read
only the relevant span.

- ``grep_context``: server-side grep -- literal/regex, line-oriented, UNRANKED
  pattern matching over ``text_content``. It is the complement of the full-text
  and semantic search tools, not a replacement: FTS matches stemmed/ranked
  lexemes and returns whole entries, while grep matches raw characters/regex and
  returns match LOCATIONS (line number + code-point offsets). All matching runs
  in Python (``re`` for literals, the ``regex`` engine for user regex) for
  byte-identical SQLite/PostgreSQL results.
- ``read_context_range``: partial read of one entry by character or line range,
  slicing the full stored ``text_content`` directly (works for every entry).

The two tools share ONE Unicode code-point offset contract: ``grep_context``
``content`` rows carry ``match_start``/``match_end`` that feed straight into
``read_context_range`` ``start_char``/``end_char`` -- locate, then extract.
"""

import asyncio
import logging
import re
import time
from typing import Annotated
from typing import Any
from typing import Literal
from typing import cast

import regex
from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from app.errors import format_exception_message
from app.ids import resolve_or_normalize_id
from app.repositories.index_node_repository import StoredNodeSummaries
from app.services.grep_service import GrepEntryResult
from app.services.grep_service import compile_pattern
from app.services.grep_service import extract_ascii_literal
from app.services.grep_service import match_entry
from app.services.outline_service import ROOT_NODE_ID
from app.services.outline_service import OutlineNode
from app.services.outline_service import count_nodes
from app.services.outline_service import parse_outline
from app.services.outline_service import resolve_node_span
from app.services.text_lines import line_index_for_offset
from app.services.text_lines import split_lines_with_offsets
from app.settings import get_settings
from app.startup import ensure_repositories
from app.tools._shared import reject_unstorable_input
from app.tools.search import MAX_FILTER_TAGS
from app.tools.search import tags_filter_cap_error
from app.types import GrepContentMatchDict
from app.types import GrepContextResultDict
from app.types import GrepCountDict
from app.types import GrepFileDict
from app.types import NavigateContextResultDict
from app.types import OutlineNodeDict
from app.types import ReadContextRangeDict

logger = logging.getLogger(__name__)
settings = get_settings()

# Outline parsing and line splitting are CPU-bound and O(text), and the stored
# entry text is unbounded, so for a large entry the work is offloaded to a worker
# thread -- otherwise a single multi-megabyte parse would pin the event loop and
# starve every other concurrent MCP request. Mirrors the same discipline the
# sibling grep matcher applies (grep_service._OFFLOAD_MIN_CHARS); small entries
# stay inline to avoid a per-call thread hop. Unicode code points, not bytes.
_OFFLOAD_MIN_CHARS = 1_000_000


def _shape_grep_output(
    output_mode: str,
    entry_results: list[GrepEntryResult],
    total_matches: int,
    truncated: bool,
    timed_out_ids: list[str] | None = None,
) -> GrepContextResultDict:
    """Build the grep response dict for the requested output mode."""
    if output_mode == 'content':
        content_rows: list[GrepContentMatchDict] = [
            {
                'context_id': entry.context_id,
                'line_number': match.line_number,
                'line': match.line,
                'match_start': match.char_start,
                'match_end': match.char_end,
                'before': list(match.before),
                'after': list(match.after),
            }
            for entry in entry_results
            for match in entry.matches
        ]
        results: list[Any] = content_rows
    elif output_mode == 'count':
        count_rows: list[GrepCountDict] = [
            {'context_id': entry.context_id, 'count': entry.match_count} for entry in entry_results
        ]
        results = count_rows
    else:  # files_with_matches (default)
        file_rows: list[GrepFileDict] = [
            {'context_id': entry.context_id, 'match_count': entry.match_count} for entry in entry_results
        ]
        results = file_rows

    response = cast(
        GrepContextResultDict,
        {
            'mode': output_mode,
            'total_matches': total_matches,
            'truncated': truncated,
            'results': results,
        },
    )
    # Surface entries skipped because their regex match timed out, so the caller
    # knows the result is incomplete for them (present only when some entry did).
    if timed_out_ids:
        response['timed_out_context_ids'] = timed_out_ids
    return response


def _trim_entry_results(
    entry_results: list[GrepEntryResult],
    limit: int,
) -> tuple[list[GrepEntryResult], int]:
    """Trim collected matches to exactly ``limit`` total, rebuilding the boundary entry.

    The grep loop collects one match past ``max_matches`` to detect genuine
    truncation truthfully; this drops the surplus and marks the entry straddling
    the cap as ``capped`` so the response stays internally consistent.

    Returns:
        The trimmed entry list and the new total match count (== ``limit``).
    """
    trimmed: list[GrepEntryResult] = []
    total = 0
    for entry in entry_results:
        if total >= limit:
            break
        room = limit - total
        if entry.match_count <= room:
            trimmed.append(entry)
            total += entry.match_count
            continue
        kept = entry.matches[:room]
        trimmed.append(
            GrepEntryResult(
                context_id=entry.context_id,
                matches=kept,
                match_count=len(kept),
                capped=True,
                timed_out=entry.timed_out,
            ),
        )
        total += len(kept)
        break
    return trimmed, total


async def grep_context(
    pattern: Annotated[
        str,
        Field(min_length=1, description='Literal substring (default) or regular expression to match, raw and unstemmed'),
    ],
    thread_id: Annotated[
        str | None,
        Field(description='Scope to one thread (recommended: bounds the scan to the indexed thread)'),
    ] = None,
    source: Annotated[
        Literal['user', 'agent'] | None,
        Field(description="Filter by entry source ('user' or 'agent')"),
    ] = None,
    tags: Annotated[
        list[str] | None,
        Field(max_length=MAX_FILTER_TAGS, description=f'Filter by tags (OR logic; at most {MAX_FILTER_TAGS})'),
    ] = None,
    metadata_filters: Annotated[
        list[dict[str, Any]] | None,
        Field(description='Advanced metadata filters with operators (gt, lt, contains, exists, etc.)'),
    ] = None,
    content_type: Annotated[
        Literal['text', 'multimodal'] | None,
        Field(description='Filter by content type'),
    ] = None,
    is_regex: Annotated[
        bool,
        Field(description='Treat pattern as a Python regular expression. Default False: literal substring (auto-escaped)'),
    ] = False,
    case_sensitive: Annotated[
        bool,
        Field(description='Match case-sensitively. Default False (Unicode-aware case-insensitive).'),
    ] = False,
    output_mode: Annotated[
        Literal['files_with_matches', 'content', 'count'],
        Field(
            description="'files_with_matches' (default: context_ids + match_count), "
                        "'content' (each match with line + offsets + context), 'count' (per-entry tally)",
        ),
    ] = 'files_with_matches',
    context_lines: Annotated[
        int,
        Field(ge=0, le=100, description='Surrounding lines before/after each match in content mode (like grep -C)'),
    ] = 0,
    max_matches: Annotated[
        int,
        Field(ge=1, le=10000, description='Maximum total matches to return (bounded output; clamped to the server cap)'),
    ] = 100,
    max_entries_scanned: Annotated[
        int,
        Field(ge=1, le=1000000, description='Maximum entries the scan visits (clamped to the server cap)'),
    ] = 1000,
    ctx: Context | None = None,
) -> GrepContextResultDict:
    """Server-side grep over stored context: literal/regex, line-oriented, unranked.

    Use this to find EXACT text -- identifiers, IDs, dates, code, error strings,
    punctuation -- that linguistic search cannot pin down. Unlike
    ``fts_search_context`` (stemmed, ranked, whole-entry) and
    ``semantic_search_context`` (meaning-based), grep matches raw characters and
    returns precise LOCATIONS so you can then read just that span with
    ``read_context_range``.

    Matching runs in Python (identical on SQLite and PostgreSQL): literals use the
    stdlib ``re`` engine; ``is_regex=True`` uses the ``regex`` engine with ``^``/``$``
    anchored per line (line-oriented: matching runs per logical line split on LF/CRLF,
    so a pattern never spans a line break and ``^``/``$`` anchor at CRLF as well as LF
    boundaries, like ripgrep with ``--crlf``), bounded by both a real per-entry
    wall-clock timeout AND an aggregate scan budget: a catastrophic pattern is
    preempted and that entry is skipped, and once the aggregate budget is exhausted
    the scan stops with ``truncated`` True -- rather than freezing the server. Output
    is always bounded by ``max_matches``; ``truncated`` is True only when matches or
    the scan were genuinely capped. Scope with ``thread_id`` whenever possible -- an
    unscoped pattern scans entries sequentially. The ``tags`` filter accepts at most
    100 tags per request; an oversized list (like an invalid metadata filter) returns
    a structured validation-error response with ``error`` and ``validation_errors``.

    Returns:
        GrepContextResultDict with ``mode``, ``total_matches``, ``truncated`` and
        ``results``. In ``content`` mode each result carries ``context_id``,
        ``line_number``, ``line``, ``match_start``/``match_end`` (Unicode
        code-point offsets into text_content), and ``before``/``after`` context
        lines. In ``files_with_matches`` mode each result is ``context_id`` +
        ``match_count``; in ``count`` mode ``context_id`` + ``count``.
        ``timed_out_context_ids`` lists entries skipped because their regex match
        exceeded the timeout (present only when some entry timed out -- the result
        is then incomplete for those entries).

    Raises:
        ToolError: If the regular expression is invalid or the scan fails.
    """
    try:
        repos = await ensure_repositories()

        # Reject an embedded NUL or unpaired UTF-16 surrogate in thread_id or a tag before
        # it reaches the PostgreSQL bind, where asyncpg would raise a non-ControlFlowError
        # that charges the circuit breaker (SQLite binds it silently -- a divergence). Both
        # feed grep_scan_text_contents below (thread_id and tags both parameterize the scan
        # query), so both must be screened, matching the sibling search tools.
        # A NUL in the grep pattern is handled separately by extract_ascii_literal, which
        # returns None for it so the pattern never reaches the LIKE/ILIKE pre-filter bind.
        reject_unstorable_input(thread_id=thread_id, tags=tags)

        # Boundary cap re-check behind the wire-schema max_length: each tag binds one
        # SQL placeholder in the scan query, so an oversized list is rejected as a
        # structured validation error before any SQL executes (never charging the
        # circuit breaker), matching the sibling search tools.
        tags_error = tags_filter_cap_error(tags)
        if tags_error is not None:
            return cast(
                GrepContextResultDict,
                {
                    'mode': output_mode,
                    'total_matches': 0,
                    'truncated': False,
                    'results': [],
                    'error': tags_error,
                    'validation_errors': [tags_error],
                },
            )

        grep_settings = settings.grep_context
        max_matches = min(max_matches, grep_settings.max_matches_cap)
        context_lines = min(context_lines, grep_settings.max_context_lines)
        max_entries_scanned = min(max_entries_scanned, grep_settings.max_entries_scanned)

        try:
            compiled = compile_pattern(pattern, is_regex=is_regex, case_sensitive=case_sensitive)
        except (re.error, regex.error) as exc:
            raise ToolError(f'Invalid regular expression: {exc}') from exc

        ascii_literal = extract_ascii_literal(pattern, is_regex=is_regex, case_sensitive=case_sensitive)

        if ctx:
            await ctx.info(f'grep_context ({"regex" if is_regex else "literal"}): {pattern[:50]!r}')

        rows, scan_stats = await repos.context.grep_scan_text_contents(
            ascii_literal=ascii_literal,
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            tags=tags,
            metadata_filters=metadata_filters,
            max_entries_scanned=max_entries_scanned,
            aggregate_bytes_budget=grep_settings.aggregate_bytes_budget,
        )

        scan_validation_errors = scan_stats.get('validation_errors')
        if scan_validation_errors:
            return cast(
                GrepContextResultDict,
                {
                    'mode': output_mode,
                    'total_matches': 0,
                    'truncated': False,
                    'results': [],
                    'error': 'Metadata filter validation failed',
                    'validation_errors': cast(list[str], scan_validation_errors),
                },
            )

        entry_results: list[GrepEntryResult] = []
        collected = 0
        match_truncated = False
        timed_out_ids: list[str] = []
        # Collect up to max_matches + 1 so a genuine extra match is detectable
        # (truthful ``truncated``): a trailing candidate that yields no match must
        # NOT flip the flag. The +1 overflow is trimmed back before shaping.
        overflow_cap = max_matches + 1
        # Aggregate wall-clock budget for a regex scan: the per-entry
        # regex_timeout_s bounds ONE entry, but an unscoped regex can visit up to
        # max_entries_scanned rows, so without a cumulative cap a catastrophic
        # pattern could hold one request for max_entries_scanned * regex_timeout_s.
        # The deadline applies only to is_regex (literals are linear-time); when it
        # fires the scan stops and returns the matches collected so far, flagged
        # truncated -- never aborting the read-only tool.
        regex_deadline = (
            time.monotonic() + grep_settings.regex_total_timeout_s if is_regex else None
        )
        scan_deadline_exceeded = False
        for context_id, text in rows:
            if collected >= overflow_cap:
                break
            if regex_deadline is not None and time.monotonic() >= regex_deadline:
                scan_deadline_exceeded = True
                break
            result = await match_entry(
                context_id,
                text,
                compiled,
                context_lines=context_lines,
                max_matches=overflow_cap - collected,
                is_regex=is_regex,
                regex_timeout_s=grep_settings.regex_timeout_s,
            )
            if result.timed_out:
                timed_out_ids.append(context_id)
            if result.capped:
                match_truncated = True
            if result.match_count == 0:
                continue
            entry_results.append(result)
            collected += result.match_count

        if collected > max_matches:
            # A real match beyond the cap exists -> genuine truncation; trim the
            # collected matches back to exactly max_matches before shaping.
            match_truncated = True
            entry_results, collected = _trim_entry_results(entry_results, max_matches)

        truncated = bool(scan_stats.get('truncated')) or match_truncated or scan_deadline_exceeded
        return _shape_grep_output(output_mode, entry_results, collected, truncated, timed_out_ids)
    except ToolError:
        raise
    except Exception as exc:
        logger.error(f'Error in grep_context: {exc}')
        raise ToolError(f'Failed to grep context: {format_exception_message(exc)}') from exc


def _outline_to_dict(node: OutlineNode, stored_nodes: StoredNodeSummaries) -> OutlineNodeDict:
    """Convert an OutlineNode (and its children) to a response dict.

    Each node's ``summary`` is looked up in the stored per-node LLM summaries
    (empty when that layer is off): primarily by ``node_id``, then by exact
    ``(char_start, char_end)`` span. A ``node_id`` hit is trusted ONLY when
    the stored row's span equals the computed node's span: node ids are
    algorithm-versioned, so a row written by an OLDER slug algorithm can
    collide with a DIFFERENT section's current id (a pre-cap sibling whose
    short title equals the capped prefix of a longer sibling's slug) and
    would otherwise attach that sibling's summary here. The span fallback
    then re-attaches rows whose node_id the current parse no longer computes
    for the SAME section -- spans derive from the text rather than the slug,
    and node rows are replaced on every text change, so an exact-span match
    against the same revision is deterministic. The caller overrides the
    root node's summary with the entry summary afterwards.

    Returns:
        The node as an OutlineNodeDict, children converted recursively.
    """
    stored = stored_nodes.by_node_id.get(node.node_id)
    summary = (
        stored.summary
        if stored is not None and (stored.char_start, stored.char_end) == (node.char_start, node.char_end)
        else None
    )
    if summary is None and node.node_id != ROOT_NODE_ID:
        # The synthetic root is excluded from the span fallback: its summary
        # mirrors the entry summary by reference (assigned by the caller), and
        # a document-spanning first heading shares the root's span, so the
        # fallback would leak that section's summary onto the root.
        summary = stored_nodes.by_span.get((node.char_start, node.char_end))
    return {
        'node_id': node.node_id,
        'level': node.level,
        'ordinal': node.ordinal,
        'title': node.title,
        'char_start': node.char_start,
        'char_end': node.char_end,
        'summary': summary,
        'children': [_outline_to_dict(child, stored_nodes) for child in node.children],
    }


async def navigate_context(
    context_id: Annotated[
        str,
        Field(min_length=8, description='Context entry id (32/36-char UUID or 8-31 char hex prefix)'),
    ],
    max_depth: Annotated[
        int,
        Field(ge=1, le=6, description='Deepest Markdown heading level to include (deeper headings fold into their section)'),
    ] = 6,
    include_node_summaries: Annotated[
        bool,
        Field(description='Attach stored per-node LLM summaries to descendant nodes (when that layer is enabled)'),
    ] = False,
    ctx: Context | None = None,
) -> NavigateContextResultDict:
    """Build a navigable Markdown outline (index_tree) for one context entry.

    Parses the entry's headings on demand into a hierarchical table of contents
    so an agent can orient in a long record before reading. Each node carries
    ``char_start``/``char_end`` (Unicode code-point offsets) that feed straight
    into ``read_context_range`` (by offset or by ``node_id``). The synthetic root
    spans the whole document and its ``summary`` mirrors the entry's stored
    summary by reference; an entry with no headings yields a childless root.

    The tree itself is code-derived (no LLM), computed fresh each call, so it is
    never stale and works for every entry. When ``include_node_summaries`` is True
    and the per-node summary layer is enabled, descendant nodes are enriched with
    their stored LLM section abstracts.

    Returns:
        NavigateContextResultDict with ``context_id``, ``total_chars``,
        ``node_count`` (headings, excluding the root), and ``root`` -- a recursive
        node carrying ``node_id``, ``level``, ``ordinal``, ``title``, ``char_start``,
        ``char_end``, ``summary`` and ``children``.

    Raises:
        ToolError: If the id is invalid or the entry is not found.
    """
    try:
        repos = await ensure_repositories()
        try:
            resolved_id = await resolve_or_normalize_id(context_id, repos.context)
        except ValueError as exc:
            raise ToolError(f'Invalid context ID: {exc}') from exc

        # The entry row and the stored node summaries live in different tables and
        # are read on separate connections (no shared snapshot). A text-change
        # update commits new text AND replacement node rows atomically, so a commit
        # landing between the two reads would pair the OLD text's outline with the
        # NEW node summaries -- mis-attaching a post-edit section summary to a
        # pre-edit span via a retained heading slug. Bracket both reads with probes
        # of the monotonic version token (bumped by exactly the writes that can
        # replace node rows) and re-take the whole snapshot (bounded) when the
        # token moved inside the window; the race is milliseconds-wide, so one
        # retry virtually always converges. updated_at is NOT a usable probe:
        # SQLite CURRENT_TIMESTAMP has second granularity, so two commits inside
        # the same wall-clock second compare equal and a torn pairing would be
        # accepted.
        text = ''
        root_summary: str | None = None
        stored_nodes = StoredNodeSummaries({}, {})
        want_node_summaries = include_node_summaries and settings.index_tree.node_summaries_enabled
        for _snapshot_attempt in range(3):
            version_before: int | None = None
            if want_node_summaries:
                _exists, _probe_source, version_before = await repos.context.check_entry_exists(resolved_id)
            rows = await repos.context.get_by_ids([resolved_id])
            if not rows:
                raise ToolError(f'Context entry not found: {context_id}')
            row = rows[0]
            text_value = row['text_content']
            text = text_value if text_value is not None else ''

            # Root summary mirrors the live summary column by reference; an empty or
            # whitespace-only stored summary normalizes to None per the optional-field
            # convention (OutlineNodeDict.summary is str | None), which is distinct from
            # the search tools' '' (empty string) for an absent summary. The flat
            # summary column is never overloaded.
            summary_value = row['summary']
            root_summary = summary_value if isinstance(summary_value, str) and summary_value.strip() else None

            if not want_node_summaries:
                break

            # Stored per-node summaries are fetched by id and do NOT depend on the
            # parse, so they are loaded before the CPU-bound parse + serialize block.
            stored_nodes = await repos.index_nodes.get_nodes_for_context(resolved_id)
            _exists, _probe_source, version_after = await repos.context.check_entry_exists(resolved_id)
            if version_after == version_before:
                break
            logger.debug(
                'navigate_context %s: entry changed between text and node-summary '
                'reads; retaking snapshot', resolved_id,
            )
        else:
            logger.warning(
                'navigate_context %s: snapshot retries exhausted under sustained '
                'concurrent updates; the returned outline and node summaries may '
                'pair text and section abstracts from different revisions',
                resolved_id,
            )

        def _build_outline() -> tuple[OutlineNodeDict, int]:
            built = parse_outline(text, max_depth=max_depth)
            return _outline_to_dict(built, stored_nodes), count_nodes(built)

        # parse_outline + serialization are O(text) CPU work over unbounded entry
        # text; a large entry is offloaded to a worker thread so it cannot pin the
        # event loop (see _OFFLOAD_MIN_CHARS).
        if len(text) > _OFFLOAD_MIN_CHARS:
            root_dict, node_count = await asyncio.to_thread(_build_outline)
        else:
            root_dict, node_count = _build_outline()

        if ctx:
            await ctx.info(f'navigate_context {resolved_id}: {node_count} nodes')

        # The root mirrors the entry summary by reference; the stored node
        # summaries never contain the synthetic root id, and the span fallback
        # excludes the root explicitly.
        root_dict['summary'] = root_summary

        return {
            'context_id': resolved_id,
            'total_chars': len(text),
            'node_count': node_count,
            'root': root_dict,
        }
    except ToolError:
        raise
    except Exception as exc:
        logger.error(f'Error in navigate_context: {exc}')
        raise ToolError(f'Failed to navigate context: {format_exception_message(exc)}') from exc


async def read_context_range(
    context_id: Annotated[
        str,
        Field(min_length=8, description='Context entry id (32/36-char UUID or 8-31 char hex prefix)'),
    ],
    start_char: Annotated[
        int | None,
        Field(ge=0, description='Start of a character range (Unicode code-point offset, inclusive)'),
    ] = None,
    end_char: Annotated[
        int | None,
        Field(ge=0, description='End of a character range (code-point offset, exclusive)'),
    ] = None,
    start_line: Annotated[
        int | None,
        Field(ge=1, description='Start of a line range (1-based, inclusive)'),
    ] = None,
    end_line: Annotated[
        int | None,
        Field(ge=1, description='End of a line range (1-based, inclusive)'),
    ] = None,
    node_id: Annotated[
        str | None,
        Field(description="Outline node id from navigate_context (e.g. 'setup/install' or 'root')"),
    ] = None,
    ctx: Context | None = None,
) -> ReadContextRangeDict:
    """Read part of one context entry by character range, line range, or outline node.

    Slices the full stored ``text_content`` directly, so it works for EVERY
    entry regardless of embeddings. Provide exactly ONE addressing mode: a
    character range (``start_char``/``end_char``), a line range
    (``start_line``/``end_line``), or a ``node_id`` from ``navigate_context``.
    Pair it with ``grep_context`` (content mode) by feeding a match's
    ``match_start``/``match_end`` into ``start_char``/``end_char``, or with
    ``navigate_context`` by passing a section's ``node_id``.

    Out-of-range offsets are CLAMPED to ``[0, len(text)]`` and the response
    ECHOES the resolved span, so an offset or node_id captured in a prior turn
    degrades gracefully if the entry was edited since.

    Returns:
        ReadContextRangeDict with ``context_id``, the resolved
        ``start_char``/``end_char`` and ``start_line``/``end_line`` (1-based), and
        the ``text`` of the slice. Offsets are Unicode code-point indices.

    Raises:
        ToolError: If the id is invalid, not exactly one addressing mode is
            given, the entry is not found, or the node_id does not resolve.
    """
    try:
        repos = await ensure_repositories()
        try:
            resolved_id = await resolve_or_normalize_id(context_id, repos.context)
        except ValueError as exc:
            raise ToolError(f'Invalid context ID: {exc}') from exc

        char_mode = start_char is not None or end_char is not None
        line_mode = start_line is not None or end_line is not None
        node_mode = node_id is not None
        if sum((char_mode, line_mode, node_mode)) != 1:
            raise ToolError(
                'Provide exactly one addressing mode: a character range (start_char/end_char), '
                'a line range (start_line/end_line), or a node_id.',
            )

        rows = await repos.context.get_by_ids([resolved_id])
        if not rows:
            raise ToolError(f'Context entry not found: {context_id}')
        text_value = rows[0]['text_content']
        text = text_value if text_value is not None else ''
        length = len(text)
        # Line splitting is O(text); offload a large entry so it cannot pin the
        # event loop (see _OFFLOAD_MIN_CHARS).
        if length > _OFFLOAD_MIN_CHARS:
            lines, line_starts = await asyncio.to_thread(split_lines_with_offsets, text)
        else:
            lines, line_starts = split_lines_with_offsets(text)

        # In line mode the clamped line numbers ARE the resolved span; echo them
        # verbatim. Recomputing from the exclusive end offset would under-report by
        # one when the requested range ends on the trailing empty line (its content
        # start equals the document length, whose preceding code point belongs to
        # the prior line).
        resolved_lines: tuple[int, int] | None = None
        if char_mode:
            start = start_char if start_char is not None else 0
            end = end_char if end_char is not None else length
        elif line_mode:
            line_count = len(lines)
            first_line = start_line if start_line is not None else 1
            last_line = end_line if end_line is not None else line_count
            first_line = max(1, min(first_line, line_count))
            last_line = max(1, min(last_line, line_count))
            if last_line < first_line:
                last_line = first_line
            start = line_starts[first_line - 1]
            end = line_starts[last_line - 1] + len(lines[last_line - 1])
            resolved_lines = (first_line, last_line)
        else:
            # node_id mode: resolve against the on-demand outline (node_id is not
            # None here because exactly one mode is active). resolve_node_span
            # re-parses the outline (O(text)); offload a large entry so the parse
            # cannot pin the event loop (see _OFFLOAD_MIN_CHARS).
            if length > _OFFLOAD_MIN_CHARS:
                span = await asyncio.to_thread(resolve_node_span, text, node_id or '')
            else:
                span = resolve_node_span(text, node_id or '')
            if span is None:
                raise ToolError(
                    f'node_id {node_id!r} did not resolve to a section in this entry '
                    f'(it may have changed since navigate_context was called).',
                )
            start, end = span

        # Mandatory clamp so stale grep/prior-turn offsets degrade gracefully.
        start = max(0, min(start, length))
        end = max(0, min(end, length))
        if end < start:
            end = start

        snippet = text[start:end]
        if resolved_lines is not None:
            start_line_out, end_line_out = resolved_lines
        else:
            start_line_out = line_index_for_offset(line_starts, start) + 1
            end_anchor = end - 1 if end > start else start
            end_line_out = line_index_for_offset(line_starts, end_anchor) + 1

        if ctx:
            await ctx.info(f'read_context_range {resolved_id} [{start}:{end}]')

        return {
            'context_id': resolved_id,
            'start_char': start,
            'end_char': end,
            'start_line': start_line_out,
            'end_line': end_line_out,
            'text': snippet,
        }
    except ToolError:
        raise
    except Exception as exc:
        logger.error(f'Error in read_context_range: {exc}')
        raise ToolError(f'Failed to read context range: {format_exception_message(exc)}') from exc
