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
# Setext underlines: a run of '=' (level 1) or '-' (level 2) on its own line.
_SETEXT_H1_RE = re.compile(r'^[ ]{0,3}=+[ \t]*$')
_SETEXT_H2_RE = re.compile(r'^[ ]{0,3}-+[ \t]*$')

ROOT_NODE_ID = 'root'


@dataclass(frozen=True, slots=True)
class OutlineNode:
    """One node in a record's Markdown outline tree.

    ``char_start``/``char_end`` are Unicode code-point offsets into the original
    text; the section runs from its heading line to the next heading of the same
    or shallower level. The synthetic root spans the whole document.
    """

    node_id: str
    level: int
    ordinal: int  # 1-based index among siblings sharing the same slug
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


def _find_fence_close(lines: list[str], start: int, fence: tuple[str, int]) -> int | None:
    """Return the index of the closing fence at or after ``start``, else None."""
    char, length = fence
    close_re = re.compile(r'^[ ]{0,3}(' + re.escape(char) + r'{' + str(length) + r',})[ \t]*$')
    for index in range(start, len(lines)):
        if close_re.match(lines[index]):
            return index
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
    index = 0
    total = len(lines)
    while index < total:
        line = lines[index]

        fence = _match_fence_open(line)
        if fence is not None:
            close_index = _find_fence_close(lines, index + 1, fence)
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
            title = _ATX_TRAILING_RE.sub('', (atx.group(2) or '').strip()).strip()
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


def parse_outline(text: str, *, max_depth: int = 6) -> OutlineNode:
    """Parse ``text`` into a Markdown outline tree rooted at a synthetic document node.

    The returned root spans the whole document (node_id ``'root'``); its
    descendants are the headings (<= ``max_depth``) nested by level. A document
    with no headings yields a childless root. Node ids are heading-path slugs
    (e.g. ``setup/install``) with a sibling ordinal disambiguating duplicate
    slugs (``notes``, ``notes-2``); they are regenerated wholesale on every parse
    and are not stable across heading inserts or renames.

    Args:
        text: The record's full text.
        max_depth: Deepest heading level to include (1-6); deeper headings fold
            into the nearest enclosing section.

    Returns:
        The root OutlineNode of the tree.
    """
    doc_len = len(text)
    lines, line_starts = split_lines_with_offsets(text)
    headings = _collect_headings(lines, line_starts, max_depth)

    root = _BuildNode(node_id=ROOT_NODE_ID, level=0, ordinal=1, title='', char_start=0, char_end=doc_len)
    stack: list[_BuildNode] = [root]

    for k, (level, title, start) in enumerate(headings):
        end = _section_end(headings, k, doc_len)
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1]
        base = slugify(title)
        count = parent.slugs.get(base, 0) + 1
        segment = base if count == 1 else f'{base}-{count}'
        # A generated suffix (e.g. 'notes-2' for the 2nd "Notes") can collide with a
        # literal sibling heading whose own slug is already 'notes-2' ("Notes 2").
        # Bump until the segment is unique among THIS parent's children so every
        # node_id is unambiguous and read_context_range resolves the intended span.
        while segment in parent.assigned:
            count += 1
            segment = f'{base}-{count}'
        parent.slugs[base] = count
        parent.assigned.add(segment)
        parent_path = '' if parent.node_id == ROOT_NODE_ID else parent.node_id
        node_id = f'{parent_path}/{segment}' if parent_path else segment
        node = _BuildNode(
            node_id=node_id,
            level=level,
            ordinal=count,
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
        max_depth: Must match the depth used when the id was produced.

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
