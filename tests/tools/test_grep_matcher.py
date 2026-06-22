"""Unit tests for the pure-CPU grep matcher (app.services.grep_service).

Covers the backend-parity-critical behaviors: literal auto-escaping,
Unicode-aware case-insensitivity (which the ASCII-only SQLite LIKE cannot do),
code-point match offsets, context-line windows, the max_matches cap signalling
truncation, and the per-entry regex timeout (ReDoS guard).
"""

import asyncio
import time

import pytest

from app.services.grep_service import compile_pattern
from app.services.grep_service import extract_ascii_literal
from app.services.grep_service import match_entry


class TestCompilePattern:
    """compile_pattern escapes literals and honors case sensitivity."""

    def test_literal_metacharacters_are_escaped(self) -> None:
        compiled = compile_pattern('a.b', is_regex=False, case_sensitive=True)
        assert compiled.search('a.b') is not None
        assert compiled.search('axb') is None

    def test_regex_metacharacters_are_active(self) -> None:
        compiled = compile_pattern('a.b', is_regex=True, case_sensitive=True)
        assert compiled.search('axb') is not None

    def test_case_insensitive_is_default(self) -> None:
        compiled = compile_pattern('ABC', is_regex=False, case_sensitive=False)
        assert compiled.search('abc') is not None

    def test_case_sensitive_when_requested(self) -> None:
        compiled = compile_pattern('ABC', is_regex=False, case_sensitive=True)
        assert compiled.search('abc') is None

    def test_unicode_case_folding_for_cyrillic(self) -> None:
        # The reason matching lives in Python: SQLite LIKE is ASCII-only CI and
        # would miss this; Python re.IGNORECASE folds Cyrillic case.
        upper = ''.join(chr(c) for c in (0x041F, 0x0440, 0x0438, 0x0432, 0x0435, 0x0442))  # PRIVET upper
        lower = upper.lower()
        compiled = compile_pattern(upper, is_regex=False, case_sensitive=False)
        assert compiled.search(lower) is not None

    def test_regex_caret_anchors_per_line_multiline(self) -> None:
        # ^ anchors at every line boundary (MULTILINE), matching the line-oriented
        # / ripgrep model -- NOT only at the document start. Without MULTILINE this
        # would return just [0].
        compiled = compile_pattern('^TODO', is_regex=True, case_sensitive=True)
        text = 'TODO one\nTODO two\nTODO three'
        assert [m.start() for m in compiled.finditer(text)] == [0, 9, 18]

    def test_regex_dollar_anchors_per_line_multiline(self) -> None:
        # $ anchors at every line end, not only the document end.
        compiled = compile_pattern(r'end$', is_regex=True, case_sensitive=True)
        text = 'the end\nanother end\nlast end'
        assert len(list(compiled.finditer(text))) == 3

    def test_regex_dot_does_not_cross_newline(self) -> None:
        # MULTILINE governs ^/$ only; '.' must still NOT match a newline, so a
        # pattern cannot silently span lines.
        compiled = compile_pattern('a.b', is_regex=True, case_sensitive=True)
        assert compiled.search('a\nb') is None


class TestExtractAsciiLiteral:
    """extract_ascii_literal gates the optional SQL substring pre-narrow.

    The pre-narrow MUST be a superset of the authoritative Python match. Under
    case-insensitive matching an ASCII-letter literal is NOT a safe pre-narrow
    because Python ``re.IGNORECASE`` folds non-ASCII code points (U+017F LONG S
    -> ``s``, U+212A KELVIN SIGN -> ``k``) that ASCII-only SQL ``LIKE``/``ILIKE``
    cannot match -- returning the literal would silently drop those rows.
    """

    def test_case_sensitive_ascii_literal_returned(self) -> None:
        assert extract_ascii_literal('hello', is_regex=False, case_sensitive=True) == 'hello'

    def test_non_ascii_literal_rejected(self) -> None:
        cyrillic = ''.join(chr(c) for c in (0x043F, 0x0440, 0x0438))
        assert extract_ascii_literal(cyrillic, is_regex=False, case_sensitive=True) is None

    def test_regex_rejected(self) -> None:
        assert extract_ascii_literal('a.*b', is_regex=True, case_sensitive=True) is None

    def test_empty_rejected(self) -> None:
        assert extract_ascii_literal('', is_regex=False, case_sensitive=True) is None

    def test_case_insensitive_letter_literal_rejected(self) -> None:
        # An ASCII letter under case-insensitive matching could miss non-ASCII
        # case-folds, so the pre-narrow must fall back to a full scan (None).
        assert extract_ascii_literal('s', is_regex=False, case_sensitive=False) is None
        assert extract_ascii_literal('hello', is_regex=False, case_sensitive=False) is None

    def test_case_insensitive_letter_free_literal_returned(self) -> None:
        # Digits/punctuation do not case-fold; the ASCII pre-narrow stays a valid
        # superset even case-insensitively.
        assert extract_ascii_literal('1234', is_regex=False, case_sensitive=False) == '1234'
        assert extract_ascii_literal('-=>', is_regex=False, case_sensitive=False) == '-=>'

    def test_case_insensitive_fold_target_would_be_lost(self) -> None:
        # Regression for the silent-miss bug: 's' DOES match U+017F under Python
        # re.IGNORECASE, so an ASCII LIKE pre-narrow on 's' would drop that row.
        # The gate returns None to force the parity-safe full scan instead.
        compiled = compile_pattern('s', is_regex=False, case_sensitive=False)
        assert compiled.search('ſ') is not None
        assert extract_ascii_literal('s', is_regex=False, case_sensitive=False) is None


class TestMatchEntry:
    """match_entry locates matches with code-point offsets and context lines."""

    @pytest.mark.asyncio
    async def test_literal_matches_with_line_numbers_and_offsets(self) -> None:
        text = 'alpha\nbeta error here\ngamma\nerror again\n'
        compiled = compile_pattern('error', is_regex=False, case_sensitive=False)
        result = await match_entry(
            'cid', text, compiled, context_lines=0, max_matches=100, is_regex=False, regex_timeout_s=5.0,
        )
        assert result.match_count == 2
        first = result.matches[0]
        assert first.line_number == 2
        assert first.line == 'beta error here'
        # 'error' starts at code-point offset 5 within the line 'beta error...'
        # which itself starts at offset 6 -> absolute 11.
        assert text[first.char_start:first.char_end] == 'error'
        assert first.char_start == 11
        second = result.matches[1]
        assert second.line_number == 4

    @pytest.mark.asyncio
    async def test_context_lines_window(self) -> None:
        text = 'l1\nl2\nNEEDLE\nl4\nl5'
        compiled = compile_pattern('NEEDLE', is_regex=False, case_sensitive=True)
        result = await match_entry(
            'cid', text, compiled, context_lines=1, max_matches=10, is_regex=False, regex_timeout_s=5.0,
        )
        match = result.matches[0]
        assert match.before == ('l2',)
        assert match.after == ('l4',)

    @pytest.mark.asyncio
    async def test_code_point_offsets_after_multibyte(self) -> None:
        prefix = chr(0x0451) * 3  # three two-byte Cyrillic letters
        text = f'{prefix}TARGET'
        compiled = compile_pattern('TARGET', is_regex=False, case_sensitive=True)
        result = await match_entry(
            'cid', text, compiled, context_lines=0, max_matches=10, is_regex=False, regex_timeout_s=5.0,
        )
        match = result.matches[0]
        # Code-point offset is 3 (one per Cyrillic letter), not 6 (bytes).
        assert match.char_start == 3
        assert text[match.char_start:match.char_end] == 'TARGET'

    @pytest.mark.asyncio
    async def test_max_matches_caps_and_signals_truncation(self) -> None:
        text = 'x\n' * 10  # 10 matching lines
        compiled = compile_pattern('x', is_regex=False, case_sensitive=True)
        result = await match_entry(
            'cid', text, compiled, context_lines=0, max_matches=3, is_regex=False, regex_timeout_s=5.0,
        )
        assert result.match_count == 3
        assert result.capped is True

    @pytest.mark.asyncio
    async def test_exact_match_count_is_not_capped(self) -> None:
        text = 'x\nx\nx'
        compiled = compile_pattern('x', is_regex=False, case_sensitive=True)
        result = await match_entry(
            'cid', text, compiled, context_lines=0, max_matches=3, is_regex=False, regex_timeout_s=5.0,
        )
        assert result.match_count == 3
        assert result.capped is False

    @pytest.mark.asyncio
    async def test_regex_timeout_skips_entry(self) -> None:
        # A GENUINE catastrophic pattern the regex engine cannot optimize away
        # ((a|a)* over a non-matching tail): the engine's own preemptive timeout
        # fires mid-match and the entry is skipped with timed_out=True -- a bounded
        # failure, never a raise. (Note: the regex engine optimizes many classic
        # ReDoS shapes like (a+)+$ to linear time; (a|a)* still backtracks.)
        compiled = compile_pattern(r'(a|a)*$', is_regex=True, case_sensitive=True)
        text = 'a' * 40 + '!'
        result = await match_entry(
            'cid', text, compiled, context_lines=0, max_matches=10, is_regex=True, regex_timeout_s=0.3,
        )
        assert result.timed_out is True
        assert result.match_count == 0

    @pytest.mark.asyncio
    async def test_real_catastrophic_regex_is_wall_clock_bounded(self) -> None:
        # The regex engine's timeout is a TRUE wall-clock bound (the core of the
        # ReDoS fix): a catastrophic match returns shortly after regex_timeout_s
        # instead of running for the effectively unbounded backtracking time the
        # stdlib re engine would (which froze the event loop for tens of seconds).
        compiled = compile_pattern(r'(a|a)*$', is_regex=True, case_sensitive=True)
        text = 'a' * 60 + '!'
        start = time.perf_counter()
        result = await match_entry(
            'cid', text, compiled, context_lines=0, max_matches=10, is_regex=True, regex_timeout_s=0.3,
        )
        elapsed = time.perf_counter() - start
        assert result.timed_out is True
        # Bounded to ~regex_timeout_s + the outer wait_for margin (1.0s) + overhead.
        assert elapsed < 3.0

    @pytest.mark.asyncio
    async def test_regex_dollar_matches_crlf_line_ends(self) -> None:
        # Matching runs per logical line through the shared LF/CRLF line model, so
        # $ anchors before a CRLF terminator (like rg --crlf): EVERY line ending in
        # 'end' matches, not only the final line that lacks a terminator. A
        # whole-text MULTILINE $ anchors only before a bare \n and would miss the
        # CRLF lines, reporting 1 instead of 3.
        text = 'the end\r\nanother end\r\nlast end'
        compiled = compile_pattern(r'end$', is_regex=True, case_sensitive=True)
        result = await match_entry(
            'cid', text, compiled, context_lines=0, max_matches=10, is_regex=True, regex_timeout_s=5.0,
        )
        assert result.match_count == 3
        for match in result.matches:
            # Offsets land on real content, never inside a \r\n pair.
            assert text[match.char_start:match.char_end] == 'end'
            assert '\r' not in match.line
            assert '\n' not in match.line

    @pytest.mark.asyncio
    async def test_regex_caret_matches_after_crlf(self) -> None:
        text = 'TODO one\r\nTODO two\r\nTODO three'
        compiled = compile_pattern(r'^TODO', is_regex=True, case_sensitive=True)
        result = await match_entry(
            'cid', text, compiled, context_lines=0, max_matches=10, is_regex=True, regex_timeout_s=5.0,
        )
        assert result.match_count == 3
        assert [m.line_number for m in result.matches] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_literal_match_offsets_correct_across_crlf(self) -> None:
        # Line numbering and code-point offsets stay correct across a CRLF
        # terminator (the \r\n is consumed as ONE terminator by the line model).
        text = 'alpha\r\nbeta target\r\ngamma'
        compiled = compile_pattern('target', is_regex=False, case_sensitive=True)
        result = await match_entry(
            'cid', text, compiled, context_lines=0, max_matches=10, is_regex=False, regex_timeout_s=5.0,
        )
        assert result.match_count == 1
        match = result.matches[0]
        assert match.line_number == 2
        assert match.line == 'beta target'
        assert text[match.char_start:match.char_end] == 'target'

    @pytest.mark.asyncio
    async def test_pattern_does_not_span_line_break(self) -> None:
        # Line-oriented: matching is per logical line, so a pattern containing an
        # explicit newline cannot match across a line break (like default ripgrep
        # without -U). This pins the line-oriented contract a whole-text matcher
        # would violate by matching 'foo\nbar' across the LF.
        text = 'foo\nbar'
        compiled = compile_pattern(r'foo\nbar', is_regex=True, case_sensitive=True)
        result = await match_entry(
            'cid', text, compiled, context_lines=0, max_matches=10, is_regex=True, regex_timeout_s=5.0,
        )
        assert result.match_count == 0

    @pytest.mark.asyncio
    async def test_large_literal_scan_is_offloaded_correct_and_non_blocking(self) -> None:
        # A literal scan over a >_OFFLOAD_MIN_CHARS entry is offloaded to a worker
        # thread, so it must NOT starve the event loop while remaining correct.
        # Pre-fix the per-line literal scan ran inline and pinned the loop for
        # seconds on newline-dense content. 'NEEDLE' sits after a large filler.
        filler = 'x\n' * 800_000  # ~1.6MB, exceeds the 1M-char offload threshold
        text = filler + 'NEEDLE\n'
        compiled = compile_pattern('NEEDLE', is_regex=False, case_sensitive=True)

        ticks = 0
        keep_ticking = True

        async def heartbeat() -> None:
            nonlocal ticks
            while keep_ticking:
                ticks += 1
                await asyncio.sleep(0.005)

        hb = asyncio.create_task(heartbeat())
        try:
            result = await match_entry(
                'cid', text, compiled, context_lines=0, max_matches=10, is_regex=False, regex_timeout_s=5.0,
            )
        finally:
            keep_ticking = False
            await hb

        # Correctness on the offloaded path: exact match, line number, code-point offset.
        assert result.match_count == 1
        match = result.matches[0]
        assert text[match.char_start:match.char_end] == 'NEEDLE'
        assert match.char_start == 1_600_000  # 800_000 'x\n' lines * 2 code points
        assert match.line_number == 800_001
        # The loop kept ticking DURING the offloaded scan (an inline scan would
        # have starved the heartbeat until the scan finished).
        assert ticks >= 2
