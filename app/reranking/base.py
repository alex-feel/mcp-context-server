"""
Reranking provider protocol for cross-encoder reranking.

This module defines the interface that all reranking providers must implement.
Following the EmbeddingProvider pattern from app/embeddings/base.py.
"""

from __future__ import annotations

from typing import Any
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class RerankingProvider(Protocol):
    """Protocol for reranking providers.

    All reranking providers must implement this interface to ensure
    consistent behavior across different backends (FlashRank, Cohere, etc.).

    The reranking pipeline:
    1. Receive query + list of search results
    2. Score each result against the query using cross-encoder
    3. Return results sorted by relevance score (descending)

    Environment Variables:
        ENABLE_RERANKING: Enable/disable reranking (default: true)
        RERANKING_PROVIDER: Provider name (default: flashrank)
        RERANKING_MODEL: Model name (default: ms-marco-MiniLM-L-12-v2)
        RERANKING_MAX_LENGTH: Max input length (default: 512)
        RERANKING_OVERFETCH: Overfetch multiplier (default: 4)
        RERANKING_CACHE_DIR: Model cache directory (default: system cache)
    """

    async def initialize(self) -> None:
        """Initialize the reranking provider.

        This method should:
        - Load model weights (lazy initialization on first use is acceptable)
        - Validate configuration
        - Prepare any necessary resources

        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If initialization fails
        """
        ...

    async def shutdown(self) -> None:
        """Release resources held by the provider.

        This method should:
        - Unload model from memory
        - Close any open connections
        - Clean up temporary files
        """
        ...

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank search results using cross-encoder scoring.

        Args:
            query: The search query to score against
            results: List of search results, each containing at minimum:
                - 'id': Unique identifier (int)
                - 'text': Text content to score
                - Other fields are preserved in output
            limit: Maximum number of results to return (None = all)

        Returns:
            List of results sorted by relevance score (descending).
            Each result dict includes original fields plus:
                - 'rerank_score': float score from cross-encoder (0.0-1.0)

        Raises:
            RuntimeError: If provider not initialized
            ValueError: If results list is empty or malformed
        """
        ...

    async def is_available(self) -> bool:
        """Check if the reranking provider is ready.

        Returns:
            True if provider can process rerank requests
        """
        ...

    @property
    def provider_name(self) -> str:
        """Return provider identifier (e.g., 'flashrank', 'cohere')."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model being used for reranking."""
        ...
