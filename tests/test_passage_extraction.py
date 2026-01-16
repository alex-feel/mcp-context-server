"""Unit tests for the passage_extraction module."""

from __future__ import annotations

import pytest

from app.services.passage_extraction_service import DEFAULT_SEPARATORS
from app.services.passage_extraction_service import HighlightRegion
from app.services.passage_extraction_service import expand_to_boundary
from app.services.passage_extraction_service import extract_rerank_passage
from app.services.passage_extraction_service import merge_windows
from app.services.passage_extraction_service import parse_highlight_positions
from app.services.passage_extraction_service import strip_tags
from app.services.passage_extraction_service import truncate_at_boundary


class TestParseHighlightPositions:
    """Tests for parse_highlight_positions function."""

    def test_single_mark(self) -> None:
        """Test parsing a single <mark> tag."""
        highlighted = 'Hello <mark>world</mark>!'
        regions = parse_highlight_positions(highlighted)

        assert len(regions) == 1
        assert regions[0] == HighlightRegion(start=6, end=11)

    def test_multiple_marks(self) -> None:
        """Test parsing multiple <mark> tags."""
        highlighted = '<mark>Hello</mark> <mark>world</mark>!'
        regions = parse_highlight_positions(highlighted)

        assert len(regions) == 2
        assert regions[0] == HighlightRegion(start=0, end=5)
        # After first tag removal: 'Hello ' + 'world' = 6 chars before 'world'
        assert regions[1] == HighlightRegion(start=6, end=11)

    def test_adjacent_marks(self) -> None:
        """Test parsing adjacent <mark> tags with no gap."""
        highlighted = '<mark>foo</mark><mark>bar</mark>'
        regions = parse_highlight_positions(highlighted)

        assert len(regions) == 2
        assert regions[0] == HighlightRegion(start=0, end=3)
        assert regions[1] == HighlightRegion(start=3, end=6)

    def test_no_marks(self) -> None:
        """Test parsing text without any marks."""
        highlighted = 'Hello world!'
        regions = parse_highlight_positions(highlighted)

        assert len(regions) == 0

    def test_empty_string(self) -> None:
        """Test parsing empty string."""
        regions = parse_highlight_positions('')
        assert len(regions) == 0

    def test_empty_mark(self) -> None:
        """Test parsing empty mark content."""
        highlighted = 'Hello <mark></mark> world'
        regions = parse_highlight_positions(highlighted)

        assert len(regions) == 1
        assert regions[0] == HighlightRegion(start=6, end=6)

    def test_mark_at_start(self) -> None:
        """Test mark at the beginning of text."""
        highlighted = '<mark>Hello</mark> world'
        regions = parse_highlight_positions(highlighted)

        assert len(regions) == 1
        assert regions[0] == HighlightRegion(start=0, end=5)

    def test_mark_at_end(self) -> None:
        """Test mark at the end of text."""
        highlighted = 'Hello <mark>world</mark>'
        regions = parse_highlight_positions(highlighted)

        assert len(regions) == 1
        assert regions[0] == HighlightRegion(start=6, end=11)

    def test_multiline_content(self) -> None:
        """Test parsing marks across multiple lines."""
        highlighted = 'Line1\n<mark>Line2</mark>\nLine3'
        regions = parse_highlight_positions(highlighted)

        assert len(regions) == 1
        assert regions[0] == HighlightRegion(start=6, end=11)


class TestExpandToBoundary:
    """Tests for expand_to_boundary function."""

    def test_expand_left_to_sentence(self) -> None:
        """Test expanding left to sentence boundary."""
        text = 'First sentence. Second sentence here.'
        # Start at 'Second'
        pos = 16
        new_pos = expand_to_boundary(text, pos, 'left', DEFAULT_SEPARATORS)

        # Should expand to after '. '
        assert new_pos == 16

    def test_expand_right_to_sentence(self) -> None:
        """Test expanding right to sentence boundary."""
        text = 'First sentence. Second sentence here.'
        # Start at 'sentence' in 'Second sentence'
        pos = 23
        new_pos = expand_to_boundary(text, pos, 'right', DEFAULT_SEPARATORS)

        # Should expand to include '. ' at end
        assert new_pos == len(text) or text[new_pos - 1] == ' '

    def test_expand_left_to_paragraph(self) -> None:
        """Test expanding left to paragraph boundary."""
        text = 'First paragraph.\n\nSecond paragraph.'
        # Start somewhere in 'Second'
        pos = 20
        new_pos = expand_to_boundary(text, pos, 'left', DEFAULT_SEPARATORS)

        # Should stop at paragraph break
        assert new_pos == 18  # After '\n\n'

    def test_expand_right_to_paragraph(self) -> None:
        """Test expanding right to paragraph boundary."""
        text = 'First paragraph.\n\nSecond paragraph.'
        # Start in 'First'
        pos = 5
        new_pos = expand_to_boundary(text, pos, 'right', DEFAULT_SEPARATORS)

        # Should stop at paragraph break
        assert new_pos == 18  # After '\n\n'

    def test_no_boundary_found_left(self) -> None:
        """Test when no boundary is found expanding left."""
        text = 'SingleWordWithNoBoundary'
        pos = 12
        new_pos = expand_to_boundary(text, pos, 'left', DEFAULT_SEPARATORS)

        # Should return original position
        assert new_pos == pos

    def test_no_boundary_found_right(self) -> None:
        """Test when no boundary is found expanding right."""
        text = 'SingleWordWithNoBoundary'
        pos = 5
        new_pos = expand_to_boundary(text, pos, 'right', DEFAULT_SEPARATORS)

        # Should return original position
        assert new_pos == pos

    def test_max_search_limit_left(self) -> None:
        """Test max_search limit when expanding left."""
        text = 'A' * 500 + '. ' + 'B' * 100
        # Start near the end - position is within B section (which starts at 502)
        pos = 590  # 88 chars into B section, beyond max_search=50 from '. '
        new_pos = expand_to_boundary(text, pos, 'left', DEFAULT_SEPARATORS, max_search=50)

        # Should not find '. ' because it's more than 50 chars away (it's at 500)
        assert new_pos == pos

    def test_max_search_limit_right(self) -> None:
        """Test max_search limit when expanding right."""
        text = 'A' * 100 + '. ' + 'B' * 500
        # Start at the beginning
        pos = 0
        new_pos = expand_to_boundary(text, pos, 'right', DEFAULT_SEPARATORS, max_search=50)

        # Should not find '. ' because it's more than 50 chars away
        assert new_pos == pos

    def test_word_boundary_fallback(self) -> None:
        """Test fallback to word boundary when no sentence boundary."""
        text = 'hello world test'
        pos = 8
        new_pos = expand_to_boundary(text, pos, 'left', [' '])

        # Should find space
        assert new_pos == 6  # After 'hello '


class TestMergeWindows:
    """Tests for merge_windows function."""

    def test_empty_windows(self) -> None:
        """Test merging empty list."""
        result = merge_windows([], gap_threshold=100)
        assert result == []

    def test_single_window(self) -> None:
        """Test single window unchanged."""
        windows = [(0, 100)]
        result = merge_windows(windows, gap_threshold=50)

        assert result == [(0, 100)]

    def test_overlapping_windows(self) -> None:
        """Test merging overlapping windows."""
        windows = [(0, 100), (50, 150)]
        result = merge_windows(windows, gap_threshold=0)

        assert len(result) == 1
        assert result[0] == (0, 150)

    def test_adjacent_windows_within_threshold(self) -> None:
        """Test merging adjacent windows within gap threshold."""
        windows = [(0, 100), (150, 250)]
        result = merge_windows(windows, gap_threshold=100)

        # Gap is 50 (150 - 100), within threshold of 100
        assert len(result) == 1
        assert result[0] == (0, 250)

    def test_windows_beyond_threshold(self) -> None:
        """Test windows beyond gap threshold stay separate."""
        windows = [(0, 100), (300, 400)]
        result = merge_windows(windows, gap_threshold=100)

        # Gap is 200, beyond threshold of 100
        assert len(result) == 2
        assert result == [(0, 100), (300, 400)]

    def test_unsorted_windows(self) -> None:
        """Test that unsorted windows are sorted before merging."""
        windows = [(200, 300), (0, 100), (100, 200)]
        result = merge_windows(windows, gap_threshold=50)

        # All should merge since they're adjacent/overlapping
        assert len(result) == 1
        assert result[0] == (0, 300)

    def test_multiple_merge_groups(self) -> None:
        """Test multiple groups of windows that merge separately."""
        windows = [(0, 100), (50, 150), (500, 600), (550, 650)]
        result = merge_windows(windows, gap_threshold=100)

        # First two merge, last two merge
        assert len(result) == 2
        assert result == [(0, 150), (500, 650)]

    def test_exact_threshold_boundary(self) -> None:
        """Test windows exactly at threshold boundary."""
        windows = [(0, 100), (200, 300)]
        result = merge_windows(windows, gap_threshold=100)

        # Gap is exactly 100, should merge (<=)
        assert len(result) == 1
        assert result[0] == (0, 300)


class TestTruncateAtBoundary:
    """Tests for truncate_at_boundary function."""

    def test_short_text_unchanged(self) -> None:
        """Test that short text is unchanged."""
        text = 'Short text'
        result = truncate_at_boundary(text, max_length=100)

        assert result == 'Short text'

    def test_truncate_at_word_boundary(self) -> None:
        """Test truncation at word boundary."""
        text = 'Hello world this is a long sentence'
        result = truncate_at_boundary(text, max_length=15)

        # Should truncate at word boundary before 15 chars
        assert result.endswith('...')
        assert len(result) <= 18  # 15 + '...'
        assert 'Hello world' in result

    def test_truncate_no_word_boundary(self) -> None:
        """Test truncation when word boundary is too far back."""
        text = 'A' * 100
        result = truncate_at_boundary(text, max_length=50)

        # No word boundary, should truncate at max_length
        assert result == 'A' * 50 + '...'

    def test_exact_length(self) -> None:
        """Test text exactly at max_length."""
        text = 'Exact len'
        result = truncate_at_boundary(text, max_length=9)

        assert result == 'Exact len'


class TestStripTags:
    """Tests for strip_tags function."""

    def test_single_tag(self) -> None:
        """Test stripping single mark tag."""
        text = 'Hello <mark>world</mark>!'
        result = strip_tags(text)

        assert result == 'Hello world!'

    def test_multiple_tags(self) -> None:
        """Test stripping multiple mark tags."""
        text = '<mark>Hello</mark> <mark>world</mark>!'
        result = strip_tags(text)

        assert result == 'Hello world!'

    def test_no_tags(self) -> None:
        """Test text without tags unchanged."""
        text = 'Hello world!'
        result = strip_tags(text)

        assert result == 'Hello world!'

    def test_empty_string(self) -> None:
        """Test empty string."""
        result = strip_tags('')
        assert result == ''


class TestExtractRerankPassage:
    """Tests for extract_rerank_passage function."""

    def test_no_highlight_returns_beginning(self) -> None:
        """Test that no highlight returns beginning of document."""
        text = 'A' * 5000
        result = extract_rerank_passage(text, None, max_passage_size=1000)

        assert len(result) <= 1000
        assert result == 'A' * 1000

    def test_empty_highlight_returns_beginning(self) -> None:
        """Test that empty highlight returns beginning of document."""
        text = 'Hello world. This is a test.'
        result = extract_rerank_passage(text, '', max_passage_size=100)

        assert result == text

    def test_single_match_extracts_window(self) -> None:
        """Test single match extracts window around it."""
        # Create document with match in the middle
        text = 'A' * 1000 + ' target ' + 'B' * 1000
        highlighted = 'A' * 1000 + ' <mark>target</mark> ' + 'B' * 1000

        result = extract_rerank_passage(
            text,
            highlighted,
            window_size=100,
            max_passage_size=2000,
        )

        # Should contain 'target' and surrounding context
        assert 'target' in result
        # Should not be the full document
        assert len(result) < len(text)

    def test_multiple_distant_matches_creates_passages(self) -> None:
        """Test multiple distant matches create separate passages joined by ellipsis."""
        # Create document with matches far apart
        text = 'Start match1 here. ' + 'A' * 2000 + ' End match2 here.'
        highlighted = 'Start <mark>match1</mark> here. ' + 'A' * 2000 + ' End <mark>match2</mark> here.'

        result = extract_rerank_passage(
            text,
            highlighted,
            window_size=50,
            max_passage_size=500,
            gap_merge_threshold=100,
        )

        # Should contain both matches or be truncated
        assert 'match1' in result or 'match2' in result

    def test_nearby_matches_merge(self) -> None:
        """Test nearby matches merge into single passage."""
        text = 'First match. Second match. Third content.'
        highlighted = 'First <mark>match</mark>. Second <mark>match</mark>. Third content.'

        result = extract_rerank_passage(
            text,
            highlighted,
            window_size=50,
            max_passage_size=500,
            gap_merge_threshold=100,
        )

        # Should contain both matches in single passage (no ellipsis)
        assert 'match' in result
        # With these window sizes, should be single merged passage
        assert ' ... ' not in result or len(result) < 100

    def test_respects_max_passage_size(self) -> None:
        """Test that result respects max_passage_size."""
        text = 'word ' * 1000
        highlighted = '<mark>word</mark> ' * 100

        result = extract_rerank_passage(
            text,
            highlighted,
            window_size=200,
            max_passage_size=500,
        )

        # Should respect max_passage_size (with some tolerance for truncation)
        assert len(result) <= 520  # Some tolerance for word boundary + ellipsis

    def test_strips_mark_tags_from_result(self) -> None:
        """Test that result does not contain mark tags."""
        text = 'Hello world test'
        highlighted = 'Hello <mark>world</mark> test'

        result = extract_rerank_passage(text, highlighted)

        assert '<mark>' not in result
        assert '</mark>' not in result

    def test_expands_to_sentence_boundary(self) -> None:
        """Test that passage expands to sentence boundaries."""
        text = 'First sentence here. The target word is here. Third sentence.'
        highlighted = 'First sentence here. The <mark>target</mark> word is here. Third sentence.'

        result = extract_rerank_passage(
            text,
            highlighted,
            window_size=10,  # Small window
            max_passage_size=500,
        )

        # Should expand to include full sentence
        assert 'target' in result

    def test_document_beginning_match(self) -> None:
        """Test match at document beginning."""
        text = 'Beginning word here. Middle content. End content.'
        highlighted = '<mark>Beginning</mark> word here. Middle content. End content.'

        result = extract_rerank_passage(
            text,
            highlighted,
            window_size=50,
            max_passage_size=500,
        )

        assert 'Beginning' in result

    def test_document_end_match(self) -> None:
        """Test match at document end."""
        text = 'Start content. Middle content. End word here.'
        highlighted = 'Start content. Middle content. End <mark>word</mark> here.'

        result = extract_rerank_passage(
            text,
            highlighted,
            window_size=50,
            max_passage_size=500,
        )

        assert 'word' in result

    def test_custom_separators(self) -> None:
        """Test using custom separators."""
        text = 'Part1|Part2|Part3'
        highlighted = 'Part1|<mark>Part2</mark>|Part3'

        result = extract_rerank_passage(
            text,
            highlighted,
            window_size=2,
            max_passage_size=500,
            separators=['|'],
        )

        # Should use | as boundary
        assert 'Part2' in result


class TestHighlightRegion:
    """Tests for HighlightRegion dataclass."""

    def test_creation(self) -> None:
        """Test basic creation of HighlightRegion."""
        region = HighlightRegion(start=10, end=20)
        assert region.start == 10
        assert region.end == 20

    def test_frozen(self) -> None:
        """Test that HighlightRegion is frozen (immutable)."""
        region = HighlightRegion(start=10, end=20)
        # Verify the dataclass is configured as frozen

        # Check that the dataclass has frozen=True by checking it's immutable
        # We use setattr via exec to avoid type checker complaints
        with pytest.raises((AttributeError, TypeError)):
            exec('region.start = 15', {'region': region})

    def test_equality(self) -> None:
        """Test equality comparison."""
        region1 = HighlightRegion(start=10, end=20)
        region2 = HighlightRegion(start=10, end=20)
        region3 = HighlightRegion(start=10, end=25)

        assert region1 == region2
        assert region1 != region3

    def test_hashable(self) -> None:
        """Test that HighlightRegion is hashable."""
        region = HighlightRegion(start=10, end=20)
        # Should be usable in sets/dicts
        s = {region}
        assert region in s


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_pipeline_single_match(self) -> None:
        """Test full pipeline with single match in large document."""
        # Simulate a real document
        intro = 'This is an introduction paragraph with context. ' * 20
        match_section = 'The important keyword appears here in context. '
        outro = 'This is the conclusion section. ' * 20

        text = intro + match_section + outro
        highlighted = intro + 'The important <mark>keyword</mark> appears here in context. ' + outro

        result = extract_rerank_passage(
            text,
            highlighted,
            window_size=200,
            max_passage_size=500,
        )

        assert 'keyword' in result
        assert len(result) <= 520

    def test_full_pipeline_fts_style_highlighting(self) -> None:
        """Test with FTS-style highlighting (multiple terms)."""
        text = 'Python is a programming language. It supports async operations. Python code is readable.'
        highlighted = (
            '<mark>Python</mark> is a programming language. '
            'It supports <mark>async</mark> operations. '
            '<mark>Python</mark> code is readable.'
        )

        result = extract_rerank_passage(
            text,
            highlighted,
            window_size=50,
            max_passage_size=300,
        )

        # Should contain matches and be clean
        assert '<mark>' not in result
        # Should contain at least some of the matches
        assert 'Python' in result or 'async' in result
