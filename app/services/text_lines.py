"""Shared newline-splitting utilities with code-point offsets.

Single source of truth for line splitting, used by the grep matcher
(:mod:`app.services.grep_service`) and the Markdown outline parser
(:mod:`app.services.outline_service`), so the line numbers and character offsets
computed in one place can never drift from the other. ``should_offload_line_scan``
decides when a line-oriented scan over stored text (the line split or the outline
parse) costs enough to run in a worker thread instead of on the event loop.

Lines are split ONLY on LF (``\\n``) and CRLF (``\\r\\n``). Python's
``str.splitlines()`` additionally breaks on vertical tab, form feed, file
separator, NEL (U+0085), and the Unicode line/paragraph separators
(U+2028/U+2029); using it would desync line numbers and offsets for any content
that legitimately contains those characters. Every offset returned here is a
Unicode CODE-POINT index (``len``/slice semantics), matching the
``start_index``/``end_index`` discipline of
:class:`app.services.chunking_service.TextChunk`.
"""

import bisect
import re

# A single LF or CRLF line terminator. CRLF is matched first so the pair is
# consumed as one terminator rather than as two separate breaks.
_NEWLINE_RE = re.compile(r'\r\n|\n')


def split_lines_with_offsets(text: str) -> tuple[list[str], list[int]]:
    """Split ``text`` into lines, returning each line's starting code-point offset.

    A line is the run of characters between line terminators; the terminator
    itself is not included in the line text. Text is always at least one line: an
    empty string yields ``([''], [0])`` and a trailing newline yields a final
    empty line (so ``'a\\nb'`` -> ``(['a', 'b'], [0, 2])`` and ``'a\\n'`` ->
    ``(['a', ''], [0, 2])``).

    Args:
        text: The source text.

    Returns:
        A ``(lines, line_starts)`` pair where ``lines[i]`` is the i-th line's
        text and ``line_starts[i]`` is the code-point offset at which it begins.
        Both lists have the same length and at least one element.
    """
    lines: list[str] = []
    line_starts: list[int] = []
    pos = 0
    for match in _NEWLINE_RE.finditer(text):
        line_starts.append(pos)
        lines.append(text[pos:match.start()])
        pos = match.end()
    line_starts.append(pos)
    lines.append(text[pos:])
    return lines, line_starts


def line_index_for_offset(line_starts: list[int], offset: int) -> int:
    """Return the 0-based line index containing ``offset``.

    Args:
        line_starts: Ascending line-start offsets from
            :func:`split_lines_with_offsets`.
        offset: A code-point offset into the original text.

    Returns:
        The index ``i`` such that ``line_starts[i] <= offset`` and either ``i``
        is the last line or ``offset < line_starts[i + 1]``. Offsets at or below
        zero map to the first line.
    """
    if offset <= 0:
        return 0
    # Largest index i with line_starts[i] <= offset.
    return max(0, bisect.bisect_right(line_starts, offset) - 1)


# Size half of the line-scan offload predicate below: a text this large is offloaded
# regardless of how few lines it has, because even the cheapest per-character pass
# over it is no longer negligible. The line-count half carries the rest of the
# decision. Small entries stay inline to avoid a per-call thread hop. Unicode code
# points, not bytes. Mirrors grep_service._OFFLOAD_MIN_CHARS, where size genuinely
# IS the whole cost driver.
_OFFLOAD_MIN_CHARS = 1_000_000

# Line-oriented scans over stored text -- outline parsing (parse_outline /
# resolve_node_span, several regexes per line) and line splitting
# (split_lines_with_offsets, one slice and one list append per line) -- cost time
# proportional to LINE COUNT, not character count. A million characters on ONE line
# parses in about two milliseconds; the same million characters split into short
# heading lines takes about two seconds to parse and about a tenth of a second to
# split. A size-only threshold is blind to that three-order-of-magnitude spread,
# which is how a dense sub-threshold entry ends up processed inline and pins the
# event loop on every navigate/read call. Counting newlines is a single C-level scan
# (microseconds even for megabytes), so the extra signal is effectively free. At
# roughly five microseconds per heading line for the costlier of the two workloads,
# this bound keeps an inline pass in the single-digit-millisecond range.
_OFFLOAD_MIN_LINES = 1_000


def should_offload_line_scan(text: str) -> bool:
    """Whether a line-oriented scan over ``text`` must run in a worker thread.

    Covers every CPU-bound pass whose cost tracks line count: the outline parse
    (``parse_outline`` / ``resolve_node_span``) and the offset-preserving line
    split (``split_lines_with_offsets``).

    Args:
        text: The entry text about to be scanned.

    Returns:
        True when the text is large enough OR line-dense enough that scanning it
        inline would block the event loop noticeably.
    """
    return len(text) > _OFFLOAD_MIN_CHARS or text.count('\n') >= _OFFLOAD_MIN_LINES
