"""Pure-CPU server-side grep matcher over context text.

All precise pattern matching for ``grep_context`` runs here in Python rather than
in backend SQL: literals use the stdlib ``re`` engine, user-supplied regular
expressions use the third-party ``regex`` engine (whose ``timeout`` is the only
way to bound catastrophic backtracking -- see :func:`compile_pattern` and
:func:`match_entry`). SQLite has no built-in REGEXP function (the project
registers none) and PostgreSQL's POSIX regex dialect differs from Python's, so
Python matching is the only way SQLite and PostgreSQL return byte-identical
results. SQL is used elsewhere only as an optional coarse substring pre-filter;
correctness comes from this module.

Offsets are Unicode code-point indices (both engines, matching over a ``str``,
report code-point positions), matching the rest of the codebase's offset
discipline. The matcher deliberately does not stem, rank, or normalize lexemes
-- that is full-text search; grep is the literal/regex, line-oriented, unranked
complement.
"""

import asyncio
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from typing import cast

import regex

from app.services.text_lines import line_index_for_offset
from app.services.text_lines import split_lines_with_offsets

logger = logging.getLogger(__name__)

# User-supplied regex is matched by the third-party ``regex`` engine, whose
# ``timeout`` is checked DURING matching -- a true wall-clock bound the stdlib
# ``re`` engine cannot give (re's C matcher holds the GIL through a whole call,
# so a thread-level timeout only reports late, it cannot interrupt). Literals
# stay on ``re`` (linear-time after re.escape). Compiled patterns are typed
# ``re.Pattern[str]``; the ``regex``-engine object is cast at the boundary.

# Outer asyncio.wait_for backstop margin over the regex-level per-entry deadline.
_REGEX_WAIT_MARGIN_S = 1.0

# A literal scan over a text longer than this is offloaded to a worker thread so
# its (linear but O(text)) CPU cannot pin the event loop; smaller entries stay
# inline to avoid a thread hop per scanned entry. The regex path always offloads
# (it also needs a wall-clock timeout). Unicode code points, not bytes.
_OFFLOAD_MIN_CHARS = 1_000_000


@dataclass(frozen=True, slots=True)
class GrepMatch:
    """A single pattern match located within one context entry.

    All offsets are Unicode code-point indices into the entry's ``text_content``.
    """

    line_number: int  # 1-based line on which the match starts
    char_start: int  # code-point offset of the match start
    char_end: int  # code-point offset of the match end (exclusive)
    line: str  # full text of the matched line (terminator excluded)
    before: tuple[str, ...]  # up to context_lines preceding lines
    after: tuple[str, ...]  # up to context_lines following lines


@dataclass(frozen=True, slots=True)
class GrepEntryResult:
    """All matches found within one context entry.

    ``capped`` is True when matching stopped at ``max_matches`` and at least one
    further match existed (so the caller can surface a ``truncated`` signal).
    ``timed_out`` is True when a regex match exceeded the per-entry timeout and
    the entry was skipped (bounded failure, never a store abort).
    """

    context_id: str
    matches: tuple[GrepMatch, ...]
    match_count: int
    capped: bool = False
    timed_out: bool = False


def compile_pattern(pattern: str, *, is_regex: bool, case_sensitive: bool) -> re.Pattern[str]:
    """Compile a grep pattern.

    For a literal search (``is_regex=False``) the pattern is escaped so regex
    metacharacters match literally and compiled with the stdlib ``re`` engine
    (linear-time after escaping, no backtracking risk). For ``is_regex=True`` the
    pattern is compiled with the third-party ``regex`` engine, whose ``timeout``
    is honored DURING matching -- the only way to bound catastrophic backtracking
    on a user-supplied pattern (CPython ``re`` holds the GIL through a whole match,
    so a thread-level timeout cannot interrupt it, only report late). The compiled
    object is typed ``re.Pattern[str]`` (the shared ``finditer`` surface both
    engines expose); the ``regex`` object is cast at this boundary. ``IGNORECASE``
    is applied unless ``case_sensitive`` is requested (Unicode-aware casefolding,
    which the ASCII-only SQLite ``LIKE`` cannot do). A user regex is also compiled
    with ``MULTILINE`` so ``^`` and ``$`` anchor at every line boundary, matching
    this tool's line-oriented / ripgrep model; ``MULTILINE`` affects only ``^``/``$``
    (not ``.``), so an ordinary pattern still matches within a single line. An
    invalid ``is_regex`` pattern propagates ``regex.error`` for the caller to map
    to a clean tool error.

    Args:
        pattern: The raw pattern text.
        is_regex: Treat ``pattern`` as a regular expression when True, otherwise
            as a literal substring.
        case_sensitive: Match case-sensitively when True.

    Returns:
        A compiled pattern (``re`` engine for literals, ``regex`` engine for user
        regular expressions), typed as ``re.Pattern[str]``.
    """
    if is_regex:
        regex_flags = regex.MULTILINE if case_sensitive else regex.IGNORECASE | regex.MULTILINE
        return cast(re.Pattern[str], regex.compile(pattern, regex_flags))
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(re.escape(pattern), flags)


def extract_ascii_literal(pattern: str, *, is_regex: bool, case_sensitive: bool) -> str | None:
    """Return a pure-ASCII literal usable as an SQL substring pre-filter, or None.

    The pre-filter MUST be a guaranteed superset of the authoritative Python
    match, otherwise ``grep_scan_text_contents`` silently drops real matches. A
    literal qualifies only when ALL of the following hold:

    * it is not a regular expression (``is_regex`` is False); and
    * it is pure ASCII -- a non-ASCII literal is rejected because SQLite ``LIKE``
      is ASCII-only case-insensitive and could exclude true Cyrillic/accented
      matches before Python ``re`` runs; and
    * EITHER the match is case-sensitive, OR the literal contains no ASCII
      letter. Under case-insensitive matching Python ``re.IGNORECASE`` folds
      certain non-ASCII code points onto ASCII letters (U+017F LATIN SMALL
      LETTER LONG S -> ``s``, U+212A KELVIN SIGN -> ``k``, and similar), so text
      containing those code points DOES match an ASCII-letter pattern in Python
      but NOT in ASCII-only SQL ``LIKE``/``ILIKE``. Returning the literal in that
      case would make the pre-filter a strict subset and lose rows, so we return
      None (forcing a full scan, the same safe path used for non-ASCII patterns).
      A letter-free literal (digits/punctuation/symbols) case-folds to itself, so
      the ASCII pre-filter stays a valid superset even case-insensitively.

    Args:
        pattern: The raw pattern text.
        is_regex: Whether the pattern is a regular expression.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        The literal to pre-filter on, or None when no safe pre-filter exists.
    """
    if is_regex:
        return None
    if not pattern or not pattern.isascii():
        return None
    if not case_sensitive and any(char.isalpha() for char in pattern):
        return None
    return pattern


def _match_entry_sync(
    context_id: str,
    text: str,
    compiled: re.Pattern[str],
    context_lines: int,
    max_matches: int,
    timeout: float,
) -> GrepEntryResult:
    """Find up to ``max_matches`` REGEX matches in one entry, PER LOGICAL LINE.

    The user-regex path only. Matching runs per logical line through the shared
    line model (:func:`app.services.text_lines.split_lines_with_offsets`) -- the
    single source of truth that also drives line numbering and the outline parser
    -- rather than over the whole text. Matching each line's content (terminator
    excluded) keeps the matcher's notion of a line identical to the line model's:
    ``^``/``$`` anchor at the same ``\\n``/``\\r\\n`` boundaries the model splits
    on, so ``word$`` matches a line ending ``word\\r\\n`` (a whole-text ``$`` would
    miss it, anchoring only before a bare ``\\n``); a pattern can never silently
    span a line break (line-oriented, like ripgrep); and every reported offset
    lands on real content, never inside a ``\\r\\n`` pair. The literal path takes
    none of this per-line cost -- it has no anchors, so
    :func:`_match_entry_literal_sync` matches the whole text in one pass.

    ``compiled`` is the ``regex`` engine; ONE per-entry wall-clock deadline
    (``timeout`` seconds) is shared across the lines (each line's ``finditer`` gets
    the REMAINING budget), so a catastrophic pattern is preempted and the
    whole-entry bound holds regardless of line count; an exhausted budget raises
    ``TimeoutError`` (the caller maps it to a skipped, ``timed_out`` entry).

    Raises:
        TimeoutError: When the shared per-entry deadline is exhausted.

    Returns:
        The entry's match result.
    """
    lines, line_starts = split_lines_with_offsets(text)
    matches: list[GrepMatch] = []
    capped = False
    deadline = time.monotonic() + timeout
    for idx, line in enumerate(lines):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Per-entry budget exhausted mid-scan: surface as a bounded timeout.
            raise TimeoutError
        # The regex engine's finditer takes the deadline kwarg (cast to Any: the
        # stdlib re.finditer stub lacks it).
        iterator = cast(Any, compiled).finditer(line, timeout=remaining)
        line_start = line_starts[idx]
        if _extend_line_matches(matches, max_matches, iterator, line, idx, lines, line_start, context_lines):
            capped = True
            break
    return GrepEntryResult(
        context_id=context_id,
        matches=tuple(matches),
        match_count=len(matches),
        capped=capped,
    )


def _extend_line_matches(
    matches: list[GrepMatch],
    max_matches: int,
    iterator: Iterator[re.Match[str]],
    line: str,
    idx: int,
    lines: list[str],
    line_start: int,
    context_lines: int,
) -> bool:
    """Append one line's matches to ``matches``, stopping at ``max_matches``.

    Offsets are made absolute via ``line_start``; context windows are sliced
    from the shared ``lines`` list.

    Returns:
        True if a further match existed beyond ``max_matches`` (the caller
        signals truncation); False otherwise.
    """
    for match in iterator:
        if len(matches) >= max_matches:
            return True
        before_start = max(0, idx - context_lines)
        after_end = min(len(lines), idx + 1 + context_lines)
        matches.append(
            GrepMatch(
                line_number=idx + 1,
                char_start=line_start + match.start(),
                char_end=line_start + match.end(),
                line=line,
                before=tuple(lines[before_start:idx]),
                after=tuple(lines[idx + 1:after_end]),
            ),
        )
    return False


def _match_entry_literal_sync(
    context_id: str,
    text: str,
    compiled: re.Pattern[str],
    context_lines: int,
    max_matches: int,
) -> GrepEntryResult:
    """Find up to ``max_matches`` LITERAL matches in one entry (whole-text, fast).

    A literal pattern is ``re.escape``-d, so it has no ``^``/``$`` anchors and the
    CRLF line-anchor concern that makes the regex path match per logical line does
    not apply: a single C-level ``finditer`` over the whole text is both correct
    and far cheaper than a Python-level ``finditer`` per line on newline-dense
    content. Line attribution (number / text / context window) is derived LAZILY
    through the shared line model only once a match is found, so a NON-matching
    scan over a very large entry never pays the ``split_lines_with_offsets`` cost.

    Returns:
        The entry's match result.
    """
    matches: list[GrepMatch] = []
    capped = False
    line_data: tuple[list[str], list[int]] | None = None
    for match in compiled.finditer(text):
        if len(matches) >= max_matches:
            # A further match exists beyond the budget: signal truncation.
            capped = True
            break
        if line_data is None:
            line_data = split_lines_with_offsets(text)
        lines, line_starts = line_data
        idx = line_index_for_offset(line_starts, match.start())
        before_start = max(0, idx - context_lines)
        after_end = min(len(lines), idx + 1 + context_lines)
        matches.append(
            GrepMatch(
                line_number=idx + 1,
                char_start=match.start(),
                char_end=match.end(),
                line=lines[idx],
                before=tuple(lines[before_start:idx]),
                after=tuple(lines[idx + 1:after_end]),
            ),
        )
    return GrepEntryResult(
        context_id=context_id,
        matches=tuple(matches),
        match_count=len(matches),
        capped=capped,
    )


async def match_entry(
    context_id: str,
    text: str,
    compiled: re.Pattern[str],
    *,
    context_lines: int,
    max_matches: int,
    is_regex: bool,
    regex_timeout_s: float,
) -> GrepEntryResult:
    """Match one entry, keeping a large scan off the event loop.

    A literal pattern (``re.escape``) is linear-time with no backtracking risk and
    is matched whole-text by :func:`_match_entry_literal_sync`; an entry larger
    than ``_OFFLOAD_MIN_CHARS`` is run under ``asyncio.to_thread`` so its O(text)
    scan cannot pin the event loop, while a small entry stays inline to avoid a
    thread hop per scanned entry. A user-supplied regex is matched per logical line
    by the ``regex`` engine whose ``timeout`` is checked DURING matching, so a
    pathological pattern is preempted at ``regex_timeout_s`` -- a TRUE per-entry
    wall-clock bound, unlike the stdlib ``re`` engine which holds the GIL through a
    whole match and can only be reported late. The bounded regex match runs in a
    worker thread under an outer ``asyncio.wait_for`` backstop (``regex_timeout_s``
    + a small margin) so even a deadline overshoot cannot pin the loop
    indefinitely; on timeout the entry yields no matches with ``timed_out=True``
    (bounded failure -- the tool is read-only and never aborts a store).

    Args:
        context_id: Canonical id of the entry being matched.
        text: The entry's full ``text_content``.
        compiled: Pre-compiled pattern from :func:`compile_pattern`.
        context_lines: Lines of surrounding context to attach to each match.
        max_matches: Maximum matches to collect from this entry.
        is_regex: Whether the pattern is a user-supplied regular expression.
        regex_timeout_s: Per-entry wall-clock deadline applied to regex matching.

    Returns:
        The entry's match result (``timed_out=True`` when the deadline fired).
    """
    if max_matches <= 0:
        return GrepEntryResult(context_id=context_id, matches=(), match_count=0)
    if not is_regex:
        if len(text) > _OFFLOAD_MIN_CHARS:
            return await asyncio.to_thread(
                _match_entry_literal_sync, context_id, text, compiled, context_lines, max_matches,
            )
        return _match_entry_literal_sync(context_id, text, compiled, context_lines, max_matches)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _match_entry_sync, context_id, text, compiled, context_lines, max_matches, regex_timeout_s,
            ),
            timeout=regex_timeout_s + _REGEX_WAIT_MARGIN_S,
        )
    except TimeoutError:
        logger.warning(
            'grep_context regex match exceeded %.1fs for context %s (entry skipped)',
            regex_timeout_s,
            context_id,
        )
        return GrepEntryResult(context_id=context_id, matches=(), match_count=0, timed_out=True)
