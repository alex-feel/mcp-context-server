"""Unit tests for the shared newline-splitting utilities.

These guarantees are load-bearing for grep line numbers, grep/match char
offsets, and read_context_range slicing staying byte-identical across backends:
splitting only on LF/CRLF (never the wider Unicode set ``str.splitlines`` uses)
and reporting code-point offsets.
"""

from app.services.text_lines import line_index_for_offset
from app.services.text_lines import split_lines_with_offsets


class TestSplitLinesWithOffsets:
    """split_lines_with_offsets returns lines plus their code-point starts."""

    def test_simple_lf(self) -> None:
        lines, starts = split_lines_with_offsets('a\nb\nc')
        assert lines == ['a', 'b', 'c']
        assert starts == [0, 2, 4]

    def test_trailing_newline_yields_final_empty_line(self) -> None:
        lines, starts = split_lines_with_offsets('a\n')
        assert lines == ['a', '']
        assert starts == [0, 2]

    def test_empty_string_is_single_empty_line(self) -> None:
        lines, starts = split_lines_with_offsets('')
        assert lines == ['']
        assert starts == [0]

    def test_crlf_consumed_as_one_terminator(self) -> None:
        lines, starts = split_lines_with_offsets('a\r\nb')
        assert lines == ['a', 'b']
        # 'a' + CRLF (2 chars) -> second line starts at offset 3
        assert starts == [0, 3]

    def test_unicode_line_separators_are_not_breaks(self) -> None:
        # U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR, U+0085 NEL, vertical
        # tab and form feed all split under str.splitlines() but MUST NOT split
        # here (they would desync offsets across backend-stored text). Built via
        # chr() so the source stays plain ASCII.
        for code in (0x2028, 0x2029, 0x85, 0x0B, 0x0C):
            text = f'a{chr(code)}b'
            lines, starts = split_lines_with_offsets(text)
            assert lines == [text]
            assert starts == [0]

    def test_offsets_are_code_points_not_bytes(self) -> None:
        # A Cyrillic letter is one code point but two UTF-8 bytes; the second
        # line must start at code-point offset 2, not byte offset 3.
        cyrillic = chr(0x0451)  # CYRILLIC SMALL LETTER IO
        lines, starts = split_lines_with_offsets(f'{cyrillic}\nx')
        assert lines == [cyrillic, 'x']
        assert starts == [0, 2]


class TestLineIndexForOffset:
    """line_index_for_offset maps a code-point offset to its 0-based line."""

    def test_maps_offsets_to_lines(self) -> None:
        _lines, starts = split_lines_with_offsets('a\nbb\nc')
        # starts == [0, 2, 5]
        assert line_index_for_offset(starts, 0) == 0
        assert line_index_for_offset(starts, 1) == 0
        assert line_index_for_offset(starts, 2) == 1
        assert line_index_for_offset(starts, 4) == 1
        assert line_index_for_offset(starts, 5) == 2
        assert line_index_for_offset(starts, 99) == 2

    def test_negative_offset_maps_to_first_line(self) -> None:
        _lines, starts = split_lines_with_offsets('a\nb')
        assert line_index_for_offset(starts, -5) == 0
