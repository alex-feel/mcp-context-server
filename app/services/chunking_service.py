"""
Text chunking service for semantic search.

This module provides the ChunkingService class that splits long documents
into smaller chunks suitable for embedding generation. Chunking improves
semantic search quality for documents longer than ~500 tokens by ensuring
each chunk can be independently embedded and matched.

The service uses LangChain's RecursiveCharacterTextSplitter which attempts
to keep natural text units (paragraphs, sentences) intact while respecting
chunk size limits. Chunk boundaries are tracked for chunk-aware reranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TextChunk:
    """Immutable chunk of text with index and character boundaries.

    Attributes:
        text: The chunk content.
        chunk_index: Zero-based position in the original document.
        start_index: Character offset where chunk starts in original document.
        end_index: Character offset where chunk ends in original document.

    Example:
        >>> chunk = TextChunk(text='Hello world', chunk_index=0, start_index=0, end_index=11)
        >>> chunk.text
        'Hello world'
        >>> chunk.end_index - chunk.start_index == len(chunk.text)
        True
    """

    text: str
    chunk_index: int
    start_index: int
    end_index: int


class ChunkingService:
    """Service for splitting text into chunks for embedding with boundary tracking.

    The service wraps LangChain's RecursiveCharacterTextSplitter with
    configuration from ChunkingSettings. When disabled or when text is
    shorter than chunk_size, returns the original text as a single chunk.

    Boundary Tracking:
        Each chunk includes start_index and end_index fields that track
        the character offsets in the original document. This enables
        chunk-aware reranking where the reranker scores the matched chunk
        rather than a truncated document beginning.

    Lazy Initialization:
        The langchain_text_splitters package is only imported when the
        service is enabled, allowing the server to run without the
        dependency when chunking is disabled.

    Separators:
        Uses markdown-friendly separators: ['\\n\\n', '\\n', '. ', ' ', '']
        This prioritizes paragraph breaks, then line breaks, then sentence
        boundaries for natural chunk boundaries.

    Example:
        >>> service = ChunkingService(enabled=True, chunk_size=1000, chunk_overlap=100)
        >>> chunks = service.split_text('Long document...')
        >>> for chunk in chunks:
        ...     print(f'Chunk {chunk.chunk_index}: chars {chunk.start_index}-{chunk.end_index}')
    """

    # Markdown-friendly separators
    # Prioritizes: paragraph > line > sentence > word > character
    SEPARATORS: list[str] = ['\n\n', '\n', '. ', ' ', '']

    def __init__(
        self,
        *,
        enabled: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> None:
        """Initialize the chunking service.

        Args:
            enabled: Whether chunking is enabled. When False, split_text
                always returns a single chunk with the original text.
            chunk_size: Target chunk size in characters (default: 1000).
            chunk_overlap: Overlap between chunks in characters (default: 100).
                Must be less than chunk_size.

        Raises:
            ValueError: If chunk_overlap >= chunk_size.
            ImportError: If enabled=True and langchain-text-splitters is not installed.
        """
        if chunk_overlap >= chunk_size:
            msg = f'chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})'
            raise ValueError(msg)

        self._enabled = enabled
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter: Any = None

        if enabled:
            # Lazy import - only when chunking is enabled
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError as e:
                raise ImportError(
                    'langchain-text-splitters package required for chunking. '
                    'Install with: uv sync --extra embeddings-ollama',
                ) from e

            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=self.SEPARATORS,
                is_separator_regex=False,
                add_start_index=True,  # Enable boundary tracking for chunk-aware reranking
            )

    @property
    def is_enabled(self) -> bool:
        """Check if chunking is enabled.

        Returns:
            True if the service will split text into chunks,
            False if it will return text as a single chunk.
        """
        return self._enabled

    @property
    def chunk_size(self) -> int:
        """Get the configured chunk size in characters."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Get the configured overlap between chunks in characters."""
        return self._chunk_overlap

    def split_text(self, text: str) -> list[TextChunk]:
        """Split text into chunks with boundary tracking.

        If disabled or text is shorter than chunk_size, returns a single
        chunk containing the original text with boundaries [0, len(text)].
        Otherwise, splits using RecursiveCharacterTextSplitter with
        start_index metadata for boundary tracking.

        Args:
            text: The text to split.

        Returns:
            List of TextChunk objects with text, chunk_index, start_index, and end_index.
            Always returns at least one chunk (even for empty string).

        Example:
            >>> service = ChunkingService(enabled=True, chunk_size=100, chunk_overlap=10)
            >>> chunks = service.split_text('Short text')
            >>> len(chunks)
            1
            >>> chunks[0].start_index
            0
            >>> chunks[0].end_index
            10
        """
        logger.debug(f'[CHUNKING] Split request: enabled={self._enabled}, text_len={len(text)}, chunk_size={self._chunk_size}')

        if not self._enabled or len(text) <= self._chunk_size:
            logger.debug('[CHUNKING] Fast path: text <= chunk_size, returning 1 chunk')
            return [TextChunk(text=text, chunk_index=0, start_index=0, end_index=len(text))]

        logger.debug('[CHUNKING] Invoking text splitter with boundary tracking')

        # Use create_documents to get metadata with start_index
        assert self._splitter is not None  # Type narrowing for mypy
        split_docs = self._splitter.create_documents([text])

        # Key operational event: shows chunking produced results
        logger.info(f'[CHUNKING] Split complete: {len(split_docs)} chunks with boundaries')

        return [
            TextChunk(
                text=split_doc.page_content,
                chunk_index=idx,
                start_index=split_doc.metadata.get('start_index', 0),
                end_index=split_doc.metadata.get('start_index', 0) + len(split_doc.page_content),
            )
            for idx, split_doc in enumerate(split_docs)
        ]
