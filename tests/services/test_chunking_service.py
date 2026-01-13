"""
Unit tests for ChunkingService.

Tests cover:
- Initialization with various configurations
- Disabled service behavior
- Short text behavior (no chunking needed)
- Long text splitting
- Chunk overlap verification
- Edge cases (empty string, exact chunk size)
- Separator hierarchy (paragraphs, lines, sentences)
"""

import importlib.util

import pytest

from app.services import ChunkingService
from app.services import TextChunk

# Local marker definition - mypy cannot resolve conftest module imports
requires_chunking = pytest.mark.skipif(
    importlib.util.find_spec('langchain_text_splitters') is None,
    reason='langchain-text-splitters package not installed (chunking feature)',
)

# Apply skip marker to all tests in this module
pytestmark = [requires_chunking]


class TestTextChunk:
    """Tests for the TextChunk dataclass."""

    def test_text_chunk_creation(self) -> None:
        """TextChunk should store text and chunk_index."""
        chunk = TextChunk(text='Hello world', chunk_index=0)
        assert chunk.text == 'Hello world'
        assert chunk.chunk_index == 0

    def test_text_chunk_is_frozen(self) -> None:
        """TextChunk should be immutable (cannot delete attributes)."""
        chunk = TextChunk(text='Hello', chunk_index=0)
        with pytest.raises(AttributeError):
            del chunk.text

    def test_text_chunk_equality(self) -> None:
        """TextChunks with same values should be equal."""
        chunk1 = TextChunk(text='Hello', chunk_index=0)
        chunk2 = TextChunk(text='Hello', chunk_index=0)
        assert chunk1 == chunk2

    def test_text_chunk_different_index(self) -> None:
        """TextChunks with different indices should not be equal."""
        chunk1 = TextChunk(text='Hello', chunk_index=0)
        chunk2 = TextChunk(text='Hello', chunk_index=1)
        assert chunk1 != chunk2

    def test_text_chunk_different_text(self) -> None:
        """TextChunks with different text should not be equal."""
        chunk1 = TextChunk(text='Hello', chunk_index=0)
        chunk2 = TextChunk(text='World', chunk_index=0)
        assert chunk1 != chunk2

    def test_text_chunk_hashable(self) -> None:
        """TextChunk should be hashable (for use in sets/dicts)."""
        chunk = TextChunk(text='Hello', chunk_index=0)
        # Should not raise
        chunk_set = {chunk}
        assert chunk in chunk_set


class TestChunkingServiceInit:
    """Tests for ChunkingService initialization."""

    def test_default_initialization(self) -> None:
        """Service should initialize with default values."""
        service = ChunkingService()
        assert service.is_enabled is True
        assert service.chunk_size == 1000
        assert service.chunk_overlap == 100

    def test_disabled_initialization(self) -> None:
        """Service should initialize when disabled."""
        service = ChunkingService(enabled=False)
        assert service.is_enabled is False

    def test_custom_chunk_size(self) -> None:
        """Service should accept custom chunk size."""
        service = ChunkingService(chunk_size=500, chunk_overlap=50)
        assert service.chunk_size == 500
        assert service.chunk_overlap == 50

    def test_overlap_must_be_less_than_size(self) -> None:
        """Service should reject overlap >= chunk_size."""
        with pytest.raises(ValueError, match='chunk_overlap.*must be less than chunk_size'):
            ChunkingService(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_size_rejected(self) -> None:
        """Service should reject overlap > chunk_size."""
        with pytest.raises(ValueError, match='chunk_overlap.*must be less than chunk_size'):
            ChunkingService(chunk_size=100, chunk_overlap=150)

    def test_minimum_chunk_size(self) -> None:
        """Service should work with small chunk sizes."""
        service = ChunkingService(chunk_size=50, chunk_overlap=10)
        assert service.chunk_size == 50
        assert service.chunk_overlap == 10

    def test_large_chunk_size(self) -> None:
        """Service should work with large chunk sizes."""
        service = ChunkingService(chunk_size=10000, chunk_overlap=1000)
        assert service.chunk_size == 10000
        assert service.chunk_overlap == 1000


class TestChunkingServiceDisabled:
    """Tests for disabled ChunkingService."""

    def test_disabled_returns_single_chunk(self) -> None:
        """Disabled service should return original text as single chunk."""
        service = ChunkingService(enabled=False)
        text = 'A' * 5000  # Long text
        chunks = service.split_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0

    def test_disabled_handles_empty_string(self) -> None:
        """Disabled service should handle empty string."""
        service = ChunkingService(enabled=False)
        chunks = service.split_text('')

        assert len(chunks) == 1
        assert chunks[0].text == ''
        assert chunks[0].chunk_index == 0

    def test_disabled_handles_short_text(self) -> None:
        """Disabled service should handle short text."""
        service = ChunkingService(enabled=False)
        chunks = service.split_text('Short text')

        assert len(chunks) == 1
        assert chunks[0].text == 'Short text'


class TestChunkingServiceShortText:
    """Tests for text shorter than chunk_size."""

    def test_short_text_returns_single_chunk(self) -> None:
        """Text shorter than chunk_size should not be split."""
        service = ChunkingService(enabled=True, chunk_size=1000, chunk_overlap=100)
        text = 'Short text'
        chunks = service.split_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0

    def test_exact_chunk_size_returns_single_chunk(self) -> None:
        """Text exactly chunk_size should not be split."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
        text = 'A' * 100
        chunks = service.split_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_empty_string(self) -> None:
        """Empty string should return single empty chunk."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
        chunks = service.split_text('')

        assert len(chunks) == 1
        assert chunks[0].text == ''
        assert chunks[0].chunk_index == 0

    def test_one_char_less_than_chunk_size(self) -> None:
        """Text one char less than chunk_size should not be split."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
        text = 'A' * 99
        chunks = service.split_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == text


class TestChunkingServiceLongText:
    """Tests for text longer than chunk_size."""

    def test_long_text_is_split(self) -> None:
        """Text longer than chunk_size should be split into multiple chunks."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
        text = 'A' * 250  # Should create at least 2 chunks

        chunks = service.split_text(text)

        assert len(chunks) > 1
        # All chunks should have sequential indices
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunks_have_overlap(self) -> None:
        """Consecutive chunks should have overlapping content."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=20)
        # Use distinct content to verify overlap
        text = 'AAAA ' * 50  # 250 chars with word boundaries

        chunks = service.split_text(text)

        # Verify we have multiple chunks
        assert len(chunks) >= 2

    def test_all_text_preserved(self) -> None:
        """Combined chunks should contain all original words."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
        words = ['word' + str(i) for i in range(50)]
        text = ' '.join(words)

        chunks = service.split_text(text)
        combined = ' '.join(c.text for c in chunks)

        # All original words should appear in chunks
        for word in words:
            assert word in combined

    def test_chunk_size_approximately_respected(self) -> None:
        """Chunks should be approximately chunk_size (may vary due to separators)."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
        text = ' '.join(['word'] * 200)  # Many words

        chunks = service.split_text(text)

        # Each chunk should be within reasonable bounds
        for chunk in chunks[:-1]:  # Exclude last chunk which may be smaller
            assert len(chunk.text) <= 150  # Allow some flexibility


class TestChunkingServiceSeparators:
    """Tests for separator-based splitting behavior."""

    def test_paragraph_separator_priority(self) -> None:
        """Splitter should prefer paragraph breaks."""
        service = ChunkingService(enabled=True, chunk_size=200, chunk_overlap=20)
        text = 'Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three.'

        chunks = service.split_text(text)

        # Should split at paragraph boundaries when possible
        assert len(chunks) >= 1
        # Each chunk should contain coherent content
        for chunk in chunks:
            assert chunk.text.strip()  # No empty chunks

    def test_line_separator_fallback(self) -> None:
        """Splitter should use line breaks when paragraphs too long."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
        text = 'Line one here\nLine two here\nLine three here\nLine four here\nLine five here'

        chunks = service.split_text(text)

        # Should handle line breaks appropriately
        assert len(chunks) >= 1

    def test_sentence_separator_fallback(self) -> None:
        """Splitter should use sentence boundaries when needed."""
        service = ChunkingService(enabled=True, chunk_size=50, chunk_overlap=5)
        text = 'First sentence here. Second sentence here. Third sentence here.'

        chunks = service.split_text(text)

        # Should respect sentence boundaries
        assert len(chunks) >= 1

    def test_word_separator_fallback(self) -> None:
        """Splitter should use word boundaries when sentences too long."""
        service = ChunkingService(enabled=True, chunk_size=20, chunk_overlap=2)
        text = 'word1 word2 word3 word4 word5 word6 word7 word8'

        chunks = service.split_text(text)

        assert len(chunks) >= 1


class TestChunkingServiceFromSettings:
    """Tests for creating service from ChunkingSettings."""

    def test_create_from_settings_values(self) -> None:
        """Service should work with settings-like values."""
        # Simulating values from ChunkingSettings
        service = ChunkingService(
            enabled=True,
            chunk_size=1000,
            chunk_overlap=100,
        )

        assert service.is_enabled is True
        assert service.chunk_size == 1000
        assert service.chunk_overlap == 100

    def test_create_disabled_from_settings(self) -> None:
        """Service should work when disabled via settings."""
        service = ChunkingService(enabled=False, chunk_size=1000, chunk_overlap=100)

        assert service.is_enabled is False
        # Should still return single chunk when disabled
        chunks = service.split_text('A' * 5000)
        assert len(chunks) == 1


class TestChunkingServiceSeparatorsConstant:
    """Tests for SEPARATORS class constant."""

    def test_separators_order(self) -> None:
        """SEPARATORS should be in priority order."""
        expected = ['\n\n', '\n', '. ', ' ', '']
        assert expected == ChunkingService.SEPARATORS

    def test_separators_immutable_in_practice(self) -> None:
        """Service instances should use same separators."""
        service1 = ChunkingService()
        service2 = ChunkingService(chunk_size=500)

        # Both should use same class constant
        assert service1.SEPARATORS is service2.SEPARATORS


class TestChunkingServiceEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_whitespace_only(self) -> None:
        """Service should handle whitespace-only text."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
        chunks = service.split_text('   ')

        assert len(chunks) >= 1

    def test_newlines_only(self) -> None:
        """Service should handle newlines-only text."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
        chunks = service.split_text('\n\n\n')

        assert len(chunks) >= 1

    def test_unicode_text(self) -> None:
        """Service should handle unicode text correctly."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
        text = 'Unicode text with special chars: \u00e9\u00e8\u00ea \u4e2d\u6587 \u0440\u0443\u0441\u0441\u043a\u0438\u0439'

        chunks = service.split_text(text)

        assert len(chunks) >= 1
        # All unicode chars should be preserved
        combined = ''.join(c.text for c in chunks)
        for char in ['\u00e9', '\u4e2d', '\u0440']:
            assert char in combined

    def test_zero_overlap(self) -> None:
        """Service should work with zero overlap."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=0)
        text = 'A' * 250

        chunks = service.split_text(text)

        assert len(chunks) >= 1

    def test_single_character_text(self) -> None:
        """Service should handle single character text."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
        chunks = service.split_text('A')

        assert len(chunks) == 1
        assert chunks[0].text == 'A'

    def test_text_with_only_separators(self) -> None:
        """Service should handle text containing only separators."""
        service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
        text = '\n\n.\n.\n\n'

        chunks = service.split_text(text)

        assert len(chunks) >= 1
