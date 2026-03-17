"""
FlashRank reranking provider implementation.

This provider uses FlashRank for fast, lightweight cross-encoder reranking.
FlashRank supports multiple models with different size/quality tradeoffs.
"""

from __future__ import annotations

import asyncio
import logging
import operator
from typing import Any

from app.settings import get_settings

logger = logging.getLogger(__name__)


class FlashRankProvider:
    """FlashRank reranking provider.

    Implements the RerankingProvider protocol using FlashRank library.
    Eagerly loads model during initialization with constrained ONNX settings.
    Applies constrained ONNX SessionOptions to prevent thread explosion
    in containerized environments.

    Environment Variables:
        RERANKING_MODEL: Model name (default: ms-marco-MiniLM-L-12-v2)
        RERANKING_MAX_LENGTH: Max input length in tokens (default: 512)
        RERANKING_CACHE_DIR: Model cache directory (default: None = system cache)
        RERANKING_INTRA_OP_THREADS: ONNX intra-op threads (default: 0 = auto-detect)
        RERANKING_CPU_MEM_ARENA: ONNX CPU memory arena (default: false)

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
        self._intra_op_threads = settings.reranking.intra_op_threads
        self._cpu_mem_arena = settings.reranking.cpu_mem_arena
        self._batch_size = settings.reranking.batch_size
        self._ranker: Any = None
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize provider and eagerly load model.

        Downloads the model (if not cached) and creates an ONNX
        InferenceSession with constrained resource settings.
        Runs synchronous model loading in a worker thread to
        avoid blocking the event loop.

        Raises:
            ImportError: If flashrank package is not installed.
        """
        try:
            from flashrank import Ranker

            _ = Ranker
        except ImportError as e:
            raise ImportError(
                'flashrank package required',
            ) from e

        await self._ensure_ranker()

        logger.info(
            f'FlashRank provider initialized: model={self._model_name}, '
            f'max_length={self._max_length}',
        )

    async def shutdown(self) -> None:
        """Release model resources."""
        self._ranker = None
        logger.info('FlashRank provider shut down')

    def _load_ranker_sync(self) -> None:
        """Load the FlashRank Ranker model with constrained ONNX Runtime settings.

        Synchronous model loading that runs in a worker thread via asyncio.to_thread.
        Creates the FlashRank Ranker, then replaces its default ONNX InferenceSession
        with one configured for constrained resource usage:

        - **Thread limiting:** intra-op threads set to a fixed count (default 2)
          instead of ONNX Runtime's default of host physical core count, which
          causes thread explosion in containers with lower CPU quotas.
        - **Arena disabled:** CPU memory arena (``enable_cpu_mem_arena``) set to
          ``False`` by default to prevent ONNX Runtime from permanently retaining
          multi-GiB intermediate tensor buffers after inference.
        - **Spin-wait disabled:** Both intra-op and inter-op spin-wait disabled
          to avoid busy-waiting threads consuming CPU in async workloads.

        The Ranker constructor's default session is explicitly released before
        the replacement session is created, freeing its unconstrained resources.

        Raises:
            FileNotFoundError: If no .onnx model file exists in the model directory.
        """
        from typing import cast

        import onnxruntime
        from flashrank import Ranker

        ort: Any = onnxruntime

        logger.info(f'Loading FlashRank model: {self._model_name}')

        # Build kwargs conditionally - FlashRank doesn't accept None for cache_dir
        ranker_kwargs: dict[str, str | int] = {
            'model_name': self._model_name,
            'max_length': self._max_length,
        }
        if self._cache_dir is not None:
            ranker_kwargs['cache_dir'] = self._cache_dir

        ranker: Any = cast(Any, Ranker(**ranker_kwargs))

        # Release the unconstrained default session created by Ranker()
        # before replacing it with our arena-disabled, thread-limited session.
        if hasattr(ranker, 'session') and ranker.session is not None:
            del ranker.session
            ranker.session = None

        # Replace with a thread-limited, arena-disabled ONNX session.
        # FlashRank's Ranker.__init__() creates ort.InferenceSession without
        # SessionOptions, defaulting intra_op_num_threads to 0 (= all host cores).
        # Resolve the ONNX model file from the Ranker's model directory.
        # Each FlashRank model directory contains exactly one .onnx file.
        model_onnx_files = list(ranker.model_dir.glob('*.onnx'))
        if not model_onnx_files:
            msg = f'No .onnx file found in {ranker.model_dir}'
            raise FileNotFoundError(msg)
        model_path = str(model_onnx_files[0])

        sess_options: Any = ort.SessionOptions()
        sess_options.intra_op_num_threads = self._intra_op_threads
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_cpu_mem_arena = self._cpu_mem_arena
        sess_options.add_session_config_entry(
            'session.intra_op.allow_spinning', '0',
        )
        sess_options.add_session_config_entry(
            'session.inter_op.allow_spinning', '0',
        )

        ranker.session = ort.InferenceSession(
            model_path, sess_options=sess_options,
        )

        self._ranker = ranker

        intra_threads: int = sess_options.intra_op_num_threads
        inter_threads: int = sess_options.inter_op_num_threads
        logger.info(
            f'FlashRank model loaded: {self._model_name} '
            f'(intra_op_threads={intra_threads}, '
            f'inter_op_threads={inter_threads}, '
            f'cpu_mem_arena={self._cpu_mem_arena})',
        )

    async def _ensure_ranker(self) -> None:
        """Ensure the FlashRank Ranker model is loaded, with concurrency protection.

        Uses double-checked locking with asyncio.Lock to prevent concurrent
        coroutines from triggering duplicate model loading. The fast-path
        check outside the lock avoids lock acquisition overhead for the
        common case (model already loaded).

        With eager loading in initialize(), this serves as a defense-in-depth
        safety net.
        """
        if self._ranker is not None:
            return
        async with self._init_lock:
            if self._ranker is None:
                await asyncio.to_thread(self._load_ranker_sync)

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

        # Ensure ranker is loaded (defense-in-depth, normally loaded during initialize())
        await self._ensure_ranker()

        # Micro-batch to prevent OOM from unbounded ONNX Runtime inference.
        # Cross-encoder scores are absolute per-row sigmoid values, so
        # sub-batch results can be concatenated without normalization.
        # Each batch runs in a worker thread via asyncio.to_thread to
        # prevent blocking the event loop during ONNX Runtime inference.
        reranked: list[Any] = []
        for batch_start in range(0, len(passages), self._batch_size):
            batch_passages = passages[batch_start:batch_start + self._batch_size]
            request = RerankRequest(query=query, passages=batch_passages)
            batch_result = await asyncio.to_thread(self._ranker.rerank, request)
            reranked.extend(batch_result)

        # Operational logging with token estimates
        num_batches = (len(passages) + self._batch_size - 1) // self._batch_size
        passage_sizes = [len(r['text']) for r in results]
        token_estimates = [size / self._chars_per_token for size in passage_sizes]
        max_tokens = max(token_estimates) if token_estimates else 0

        query_preview = query[:50] + '...' if len(query) > 50 else query

        # Warn if any passage likely exceeds token limit
        if max_tokens > self._max_length:
            logger.warning(
                f'Passage may exceed token limit: '
                f'~{int(max_tokens)} tokens estimated (limit: {self._max_length}). '
                f'Largest passage: {max(passage_sizes)} chars, '
                f'batches={num_batches}',
            )
        else:
            logger.info(
                f'Reranked {len(results)} results: '
                f'{min(passage_sizes)}-{max(passage_sizes)} chars '
                f'(~{int(min(token_estimates))}-{int(max(token_estimates))} tokens, '
                f'limit: {self._max_length}), query="{query_preview}", '
                f'batches={num_batches}',
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
