"""
FlashRank reranking provider implementation.

This provider uses FlashRank for fast, lightweight cross-encoder reranking.
FlashRank supports multiple models with different size/quality tradeoffs.
"""

from __future__ import annotations

import logging
import operator
from typing import Any

from app.settings import get_settings

logger = logging.getLogger(__name__)


class FlashRankProvider:
    """FlashRank reranking provider.

    Implements the RerankingProvider protocol using FlashRank library.
    Uses lazy initialization - model loaded on first rerank() call.

    Environment Variables:
        RERANKING_MODEL: Model name (default: ms-marco-MiniLM-L-12-v2)
        RERANKING_MAX_LENGTH: Max input length in tokens (default: 512)
        RERANKING_CACHE_DIR: Model cache directory (default: None = system cache)

    Available Models (from FlashRank documentation):
        - ms-marco-TinyBERT-L-2-v2: ~4MB, fastest, lower quality
        - ms-marco-MiniLM-L-12-v2: ~34MB, good balance (DEFAULT)
        - ms-marco-MultiBERT-L-12: ~140MB, multilingual support
        - rank-T5-flan: ~110MB, T5-based, highest quality
    """

    def __init__(self) -> None:
        """Initialize provider configuration from settings."""
        settings = get_settings()
        self._model_name = settings.reranking.model
        self._max_length = settings.reranking.max_length
        self._cache_dir = settings.reranking.cache_dir
        self._chars_per_token = settings.reranking.chars_per_token
        self._ranker: Any = None  # Lazy initialization

    async def initialize(self) -> None:
        """Validate FlashRank is available (model loaded lazily).

        Raises:
            ImportError: If flashrank package is not installed
        """
        try:
            from flashrank import Ranker

            # Validate import succeeded by checking the class exists
            _ = Ranker
        except ImportError as e:
            raise ImportError(
                'flashrank package required. '
                'Install with: uv sync --extra reranking',
            ) from e

        logger.info(
            f'FlashRank provider initialized: model={self._model_name}, '
            f'max_length={self._max_length}',
        )

    async def shutdown(self) -> None:
        """Release model resources."""
        self._ranker = None
        logger.info('FlashRank provider shut down')

    def _ensure_ranker(self) -> None:
        """Lazy load the FlashRank Ranker model.

        Sets self._ranker to an initialized Ranker instance if not already loaded.
        """
        if self._ranker is None:
            from flashrank import Ranker

            logger.info(f'Loading FlashRank model: {self._model_name}')

            # Build kwargs conditionally - FlashRank doesn't accept None for cache_dir
            ranker_kwargs: dict[str, str | int] = {
                'model_name': self._model_name,
                'max_length': self._max_length,
            }
            if self._cache_dir is not None:
                ranker_kwargs['cache_dir'] = self._cache_dir

            self._ranker = Ranker(**ranker_kwargs)
            logger.info(f'FlashRank model loaded: {self._model_name}')

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank results using FlashRank cross-encoder.

        Args:
            query: Search query to score against
            results: List of search results with 'id' and 'text' fields
            limit: Maximum results to return (None = all)

        Returns:
            Results sorted by relevance score with 'rerank_score' added

        Raises:
            ValueError: If results is empty or missing required fields
        """
        if not results:
            return []

        # Validate required fields
        for i, result in enumerate(results):
            if 'text' not in result:
                raise ValueError(f"Result {i} missing required 'text' field")

        # Prepare passages for FlashRank
        # FlashRank expects: [{"id": any, "text": str, "meta": dict}, ...]
        from flashrank import RerankRequest

        passages: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            passages.append({
                'id': result.get('id', i),
                'text': result['text'],
                'meta': {'original_index': i},
            })

        # Ensure ranker is loaded (lazy initialization)
        self._ensure_ranker()

        # Execute reranking (FlashRank is synchronous)
        request = RerankRequest(query=query, passages=passages)
        reranked = self._ranker.rerank(request)

        # Operational logging with token estimates
        passage_sizes = [len(r['text']) for r in results]
        token_estimates = [size / self._chars_per_token for size in passage_sizes]
        max_tokens = max(token_estimates) if token_estimates else 0

        query_preview = query[:50] + '...' if len(query) > 50 else query

        # Warn if any passage likely exceeds token limit
        if max_tokens > self._max_length:
            logger.warning(
                f'[RERANKING] Passage may exceed token limit: '
                f'~{int(max_tokens)} tokens estimated (limit: {self._max_length}). '
                f'Largest passage: {max(passage_sizes)} chars',
            )
        else:
            logger.info(
                f'[RERANKING] Reranked {len(results)} results: '
                f'{min(passage_sizes)}-{max(passage_sizes)} chars '
                f'(~{int(min(token_estimates))}-{int(max(token_estimates))} tokens, '
                f'limit: {self._max_length}), query="{query_preview}"',
            )

        # Map scores back to original results
        # FlashRank returns: [{"id": ..., "text": ..., "meta": ..., "score": float}, ...]
        score_map: dict[int, float] = {}
        for item in reranked:
            original_idx = item['meta']['original_index']
            score_map[original_idx] = float(item['score'])

        # Add rerank_score to original results and sort
        scored_results: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            result_copy = result.copy()
            result_copy['rerank_score'] = score_map.get(i, 0.0)
            scored_results.append(result_copy)

        # Sort by rerank_score descending
        scored_results.sort(key=operator.itemgetter('rerank_score'), reverse=True)

        # Apply limit if specified
        if limit is not None:
            scored_results = scored_results[:limit]

        return scored_results

    async def is_available(self) -> bool:
        """Check if FlashRank is available.

        Returns:
            True if flashrank package is installed
        """
        try:
            from flashrank import Ranker

            # Validate import succeeded by checking the class exists
            _ = Ranker
            return True
        except ImportError:
            return False

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return 'flashrank'

    @property
    def model_name(self) -> str:
        """Return the model being used."""
        return self._model_name
