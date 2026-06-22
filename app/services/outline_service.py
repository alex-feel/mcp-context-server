"""Code-derived Markdown outline (index_tree) parser.

Builds a per-record table-of-contents tree from Markdown heading structure --
the structure agents are already instructed to write ("Prefer storing context in
pure Markdown format"). Pure CPU, no LLM, computed on demand so it is never stale
and can never abort a store. Each node carries Unicode code-point offsets
(``char_start``/``char_end``) into the original text, sharing the offset contract
used by grep_context and read_context_range.

Heading detection is fenced-code-block aware: a ``#`` inside a properly closed
``` or ~~~ fence is ignored. An UNCLOSED fence is treated leniently -- the opener
is not allowed to swallow every subsequent heading -- because a stray unterminated
fence should not blind the whole navigation tree.
"""

import bisect
import re
from dataclasses import dataclass
from dataclasses import field

from app.services.text_lines import split_lines_with_offsets

# ATX heading: up to 3 leading spaces, 1-6 '#', then (whitespace + content)?.
# A space is required before content, so '#foo' is NOT a heading (CommonMark).
_ATX_RE = re.compile(r'^[ ]{0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$')
# A trailing closing run of '#'s on an ATX line is decorative and stripped.
_ATX_TRAILING_RE = re.compile(r'[ \t]+#+[ \t]*$')
# Opening code fence (3+ backticks or tildes) with an optional info string.
_FENCE_OPEN_RE = re.compile(r'^[ ]{0,3}(`{3,}|~{3,})(.*)$')
# A bare fence line (a run of one fence char, <=3 indent, only trailing whitespace)
# -- the only lines that can CLOSE a fence. Precomputed once so a run of unclosable
# openers does not rescan to EOF each (the source of an O(n^2) parse blowup).
_FENCE_CLOSE_RE = re.compile(r'^[ ]{0,3}(`{3,}|~{3,})[ \t]*$')
# Setext underlines: a run of '=' (level 1) or '-' (level 2) on its own line.
_SETEXT_H1_RE = re.compile(r'^[ ]{0,3}=+[ \t]*$')
_SETEXT_H2_RE = re.compile(r'^[ ]{0,3}-+[ \t]*$')

ROOT_NODE_ID = 'root'

# The canonical heading depth (Markdown's deepest ATX level). Node ids are always
# assigned over the full level-1..6 heading set so an id is independent of the
# max_depth a caller folds the displayed tree to -- see _assign_canonical_ids.
_CANONICAL_HEADING_DEPTH = 6


@dataclass(frozen=True, slots=True)
class OutlineNode:
    """One node in a record's Markdown outline tree.

    ``char_start``/``char_end`` are Unicode code-point offsets into the original
    text; the section runs from its heading line to the next heading of the same
    or shallower level. The synthetic root spans the whole document.
    """

    node_id: str
    level: int
    ordinal: int  # 1-based disambiguation index, mirrored in the node_id suffix
    title: str
    char_start: int
    char_end: int
    children: tuple['OutlineNode', ...]


@dataclass
class _BuildNode:
    """Mutable node used while constructing the tree (frozen via _freeze)."""

    node_id: str
    level: int
    ordinal: int
    title: str
    char_start: int
    char_end: int
    children: list['_BuildNode'] = field(default_factory=list)
    slugs: dict[str, int] = field(default_factory=dict)
    assigned: set[str] = field(default_factory=set)  # segments already used by children


def slugify(title: str) -> str:
    """Convert a heading title to a node-id slug.

    Lowercases, collapses runs of non-word characters to single hyphens, and
    trims hyphens. Unicode letters are preserved (so Cyrillic headings get
    Cyrillic slugs). An empty result falls back to ``'section'``.

    Args:
        title: The heading title text.

    Returns:
        A non-empty slug suitable as a node-id segment.
    """
    slug = re.sub(r'[^\w]+', '-', title.strip().lower(), flags=re.UNICODE).strip('-')
    return slug or 'section'


def _match_fence_open(line: str) -> tuple[str, int] | None:
    """Return (fence_char, length) if the line opens a code fence, else None."""
    match = _FENCE_OPEN_RE.match(line)
    if match is None:
        return None
    marker = match.group(1)
    info = match.group(2)
    # A backtick fence's info string may not contain backticks (CommonMark).
    if marker[0] == '`' and '`' in info:
        return None
    return marker[0], len(marker)


def _collect_fence_candidates(lines: list[str]) -> dict[str, tuple[list[int], list[int]]]:
    """Map each fence char to the ascending line indices and run lengths of every
    bare fence line (a potential CLOSING fence).

    Built in one O(n) pass so the heading scan resolves a fence's closer without
    rescanning the document per opener.

    Returns:
        ``{char: (indices, runs)}`` with ``indices`` ascending and ``runs[i]`` the
        run length of the bare fence line at ``indices[i]``.
    """
    cands: dict[str, tuple[list[int], list[int]]] = {}
    for index, line in enumerate(lines):
        match = _FENCE_CLOSE_RE.match(line)
        if match is None:
            continue
        run = match.group(1)
        indices, runs = cands.setdefault(run[0], ([], []))
        indices.append(index)
        runs.append(len(run))
    return cands


def _fence_closer_index(
    cands: dict[str, tuple[list[int], list[int]]],
    no_closer_len: dict[str, int],
    char: str,
    length: int,
    opener_index: int,
) -> int | None:
    """Index of the first bare fence line after ``opener_index`` that closes a
    ``(char, length)`` fence (same char, run >= ``length``), or None if unclosed.

    Matches the exact per-opener semantics of the original linear scan: the first
    bare fence of the same char with a long-enough run, searched strictly after the
    opener. ``no_closer_len`` memoizes, per char, the smallest opener length already
    proven to have no closer ahead; since the heading scan advances monotonically,
    once an opener of length L has no closer to EOF, every later opener of length
    >= L also has none -- so a run of unclosable openers costs O(1) each instead of
    rescanning, removing the O(n^2) blowup.

    Returns:
        The closing fence line index, or None when the fence is unclosed.
    """
    proven = no_closer_len.get(char)
    if proven is not None and length >= proven:
        return None
    entry = cands.get(char)
    if entry is not None:
        indices, runs = entry
        start = bisect.bisect_right(indices, opener_index)
        for pos in range(start, len(indices)):
            if runs[pos] >= length:
                return indices[pos]
    no_closer_len[char] = length
    return None


def _setext_level(line: str) -> int | None:
    """Return 1 for a '=' underline, 2 for a '-' underline, else None."""
    if _SETEXT_H1_RE.match(line):
        return 1
    if _SETEXT_H2_RE.match(line):
        return 2
    return None


def _collect_headings(lines: list[str], line_starts: list[int], max_depth: int) -> list[tuple[int, str, int]]:
    """Collect (level, title, char_start) for each heading, fenced-code aware."""
    headings: list[tuple[int, str, int]] = []
    fence_candidates = _collect_fence_candidates(lines)
    no_closer_len: dict[str, int] = {}
    index = 0
    total = len(lines)
    while index < total:
        line = lines[index]

        fence = _match_fence_open(line)
        if fence is not None:
            char, length = fence
            close_index = _fence_closer_index(fence_candidates, no_closer_len, char, length, index)
            if close_index is not None:
                index = close_index + 1  # skip the whole closed fence
                continue
            # Unclosed fence: treat the opener as an ordinary line so subsequent
            # headings remain visible rather than being swallowed to EOF.
            index += 1
            continue

        atx = _ATX_RE.match(line)
        if atx is not None:
            level = len(atx.group(1))
            raw_content = (atx.group(2) or '').strip()
            title = _ATX_TRAILING_RE.sub('', raw_content).strip()
            # A heading whose entire content is a closing '#' run (e.g. '## ###')
            # is EMPTY per CommonMark. _ATX_TRAILING_RE only strips a closing run
            # preceded by whitespace; when the run directly follows the opening
            # marker the captured content is all '#', so collapse it to empty.
            if raw_content and set(raw_content) == {'#'}:
                title = ''
            if level <= max_depth:
                headings.append((level, title, line_starts[index]))
            index += 1
            continue

        # Setext: a non-blank text line underlined by '=' (H1) or '-' (H2).
        if line.strip() and index + 1 < total:
            level = _setext_level(lines[index + 1]) or 0
            if level and level <= max_depth:
                headings.append((level, line.strip(), line_starts[index]))
                index += 2
                continue

        index += 1
    return headings


def _section_end(headings: list[tuple[int, str, int]], k: int, doc_len: int) -> int:
    """End offset of heading k: the next same-or-shallower heading start, else EOF."""
    level_k = headings[k][0]
    for j in range(k + 1, len(headings)):
        if headings[j][0] <= level_k:
            return headings[j][2]
    return doc_len


def _freeze(node: _BuildNode) -> OutlineNode:
    """Recursively convert a mutable build node into a frozen OutlineNode."""
    return OutlineNode(
        node_id=node.node_id,
        level=node.level,
        ordinal=node.ordinal,
        title=node.title,
        char_start=node.char_start,
        char_end=node.char_end,
        children=tuple(_freeze(child) for child in node.children),
    )


def _assign_canonical_ids(headings: list[tuple[int, str, int]]) -> dict[int, tuple[str, int]]:
    """Assign a depth-invariant ``(node_id, ordinal)`` to every heading.

    Walks the FULL heading set (every level present, not a ``max_depth``-filtered
    view) with the same parent-path slug disambiguation :func:`parse_outline`
    uses, and returns a map keyed by each heading's ``char_start`` (unique per
    heading). A heading's id therefore depends only on the document, never on the
    ``max_depth`` a caller later folds the tree to, so node ids stay stable across
    ``navigate_context`` depths and match the canonical depth at which per-node
    summaries are generated and ``read_context_range`` resolves.

    Returns:
        A map from each heading's ``char_start`` to its ``(node_id, ordinal)``.
    """
    root = _BuildNode(node_id=ROOT_NODE_ID, level=0, ordinal=1, title='', char_start=0, char_end=0)
    # Reserve the synthetic root's id so a top-level heading whose title slugifies
    # to ROOT_NODE_ID (e.g. "# Root") cannot claim it: the disambiguation loop below
    # then renames such a heading to 'root-2', keeping every node_id unique so
    # resolve_node_span's ROOT short-circuit addresses only the true synthetic root
    # while the heading's own section still round-trips through read_context_range.
    root.assigned.add(ROOT_NODE_ID)
    root.slugs[ROOT_NODE_ID] = 1
    stack: list[_BuildNode] = [root]
    ids: dict[int, tuple[str, int]] = {}

    for level, title, start in headings:
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1]
        base = slugify(title)
        count = parent.slugs.get(base, 0) + 1
        segment = base if count == 1 else f'{base}-{count}'
        # A generated suffix (e.g. 'notes-2' for the 2nd "Notes") can collide with
        # a literal sibling heading whose own slug is already 'notes-2' ("Notes 2").
        # Bump until the segment is unique among THIS parent's children so every
        # node_id is unambiguous and read_context_range resolves the intended span.
        while segment in parent.assigned:
            count += 1
            segment = f'{base}-{count}'
        parent.slugs[base] = count
        parent.assigned.add(segment)
        parent_path = '' if parent.node_id == ROOT_NODE_ID else parent.node_id
        node_id = f'{parent_path}/{segment}' if parent_path else segment
        ids[start] = (node_id, count)
        node = _BuildNode(
            node_id=node_id,
            level=level,
            ordinal=count,
            title=title,
            char_start=start,
            char_end=start,
        )
        parent.children.append(node)
        stack.append(node)

    return ids


def parse_outline(text: str, *, max_depth: int = 6) -> OutlineNode:
    """Parse ``text`` into a Markdown outline tree rooted at a synthetic document node.

    The returned root spans the whole document (node_id ``'root'``); its
    descendants are the headings (<= ``max_depth``) nested by level. A document
    with no headings yields a childless root. Node ids are heading-path slugs
    (e.g. ``setup/install``) with a sibling ordinal disambiguating duplicate
    slugs (``notes``, ``notes-2``).

    Node ids and ordinals are assigned over the FULL canonical heading set (levels
    1-6), so a given heading's id is INDEPENDENT of ``max_depth``: a folded
    (lower-``max_depth``) view reuses exactly the ids the canonical parse produces.
    This is load-bearing -- per-node summaries are generated, and
    ``read_context_range`` resolves, at the canonical depth, so a
    ``navigate_context`` call with a lower ``max_depth`` must surface the same ids
    for stored summaries to attach to the right sections and for a returned
    ``node_id`` to round-trip. Ids are still regenerated wholesale on every parse
    and are NOT stable across heading inserts or renames.

    Args:
        text: The record's full text.
        max_depth: Deepest heading level to include (1-6); deeper headings fold
            into the nearest enclosing section (whose span extends over them). This
            only affects which nodes appear and their spans -- never the node ids.

    Returns:
        The root OutlineNode of the tree.
    """
    doc_len = len(text)
    lines, line_starts = split_lines_with_offsets(text)
    # Assign ids/ordinals over the full canonical heading set so they are
    # max_depth-invariant (see _assign_canonical_ids), then build the (possibly
    # folded) display tree from the headings within max_depth, reusing those ids.
    canonical_headings = _collect_headings(lines, line_starts, _CANONICAL_HEADING_DEPTH)
    canonical_ids = _assign_canonical_ids(canonical_headings)
    headings = [h for h in canonical_headings if h[0] <= max_depth]

    root = _BuildNode(node_id=ROOT_NODE_ID, level=0, ordinal=1, title='', char_start=0, char_end=doc_len)
    stack: list[_BuildNode] = [root]

    for k, (level, title, start) in enumerate(headings):
        end = _section_end(headings, k, doc_len)
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1]
        node_id, ordinal = canonical_ids[start]
        node = _BuildNode(
            node_id=node_id,
            level=level,
            ordinal=ordinal,
            title=title,
            char_start=start,
            char_end=end,
        )
        parent.children.append(node)
        stack.append(node)

    return _freeze(root)


def count_nodes(root: OutlineNode) -> int:
    """Return the number of heading nodes in the tree (excluding the root)."""
    return sum(1 + count_nodes(child) for child in root.children)


def resolve_node_span(text: str, node_id: str, *, max_depth: int = 6) -> tuple[int, int] | None:
    """Resolve a node id to its ``(char_start, char_end)`` span, or None if absent.

    Re-parses ``text`` (the parse is cheap and keeps the tree never-stale) and
    walks it for ``node_id``. The synthetic root id resolves to the whole
    document.

    Args:
        text: The record's full text.
        node_id: A node id from a prior :func:`parse_outline` / navigate_context.
        max_depth: Tree depth to resolve against. Node ids are depth-invariant
            (assigned over the full canonical heading set), so the default
            canonical depth resolves ids produced by navigate_context at ANY
            max_depth.

    Returns:
        The node's code-point span, or None when no node matches.
    """
    root = parse_outline(text, max_depth=max_depth)
    if node_id == ROOT_NODE_ID:
        return root.char_start, root.char_end

    stack = list(root.children)
    while stack:
        node = stack.pop()
        if node.node_id == node_id:
            return node.char_start, node.char_end
        stack.extend(node.children)
    return None
