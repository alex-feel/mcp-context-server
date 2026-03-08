"""Tests for reranking factory and FlashRank provider."""

from __future__ import annotations

import importlib.util

import pytest

from app.reranking import RerankingProvider
from app.reranking import create_reranking_provider

# Local marker definition - mypy cannot resolve conftest module imports
requires_flashrank = pytest.mark.skipif(
    importlib.util.find_spec('flashrank') is None,
    reason='flashrank package not installed (reranking feature)',
)

# Apply skip marker to all tests in this module
pytestmark = [requires_flashrank]


class TestRerankingFactory:
    """Tests for create_reranking_provider factory function."""

    def test_factory_creates_flashrank_by_default(self) -> None:
        """Default provider should be flashrank."""
        provider = create_reranking_provider()
        assert provider.provider_name == 'flashrank'

    def test_factory_with_explicit_flashrank(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit flashrank provider should work."""
        from app.settings import get_settings
        monkeypatch.setenv('RERANKING_PROVIDER', 'flashrank')
        get_settings.cache_clear()
        provider = create_reranking_provider()
        assert provider.provider_name == 'flashrank'

    def test_factory_unsupported_provider_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unsupported provider should raise ValueError."""
        from app.settings import get_settings
        monkeypatch.setenv('RERANKING_PROVIDER', 'nonexistent')
        get_settings.cache_clear()
        with pytest.raises(ValueError, match='Unsupported reranking provider'):
            create_reranking_provider()

    def test_provider_implements_protocol(self) -> None:
        """Provider should implement RerankingProvider protocol."""
        provider = create_reranking_provider()
        assert isinstance(provider, RerankingProvider)


class TestFlashRankProvider:
    """Tests for FlashRankProvider implementation."""

    @pytest.fixture
    def provider(self) -> RerankingProvider:
        """Create FlashRank provider instance."""
        return create_reranking_provider()

    @pytest.mark.asyncio
    async def test_initialize_succeeds(
        self, provider: RerankingProvider,
    ) -> None:
        """Initialize should succeed with flashrank installed."""
        await provider.initialize()
        # Shutdown to clean up
        await provider.shutdown()

    @pytest.mark.asyncio
    async def test_is_available_true(
        self, provider: RerankingProvider,
    ) -> None:
        """is_available should return True when flashrank installed."""
        result = await provider.is_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_rerank_empty_list_returns_empty(
        self, provider: RerankingProvider,
    ) -> None:
        """Reranking empty list should return empty list."""
        await provider.initialize()
        try:
            result = await provider.rerank('test query', [])
            assert result == []
        finally:
            await provider.shutdown()

    @pytest.mark.asyncio
    async def test_rerank_adds_score_field(
        self, provider: RerankingProvider,
    ) -> None:
        """Reranking should add rerank_score to results."""
        await provider.initialize()
        try:
            results = [
                {'id': 1, 'text': 'Python programming language'},
                {'id': 2, 'text': 'Java programming language'},
            ]
            reranked = await provider.rerank('Python tutorial', results)

            assert len(reranked) == 2
            for item in reranked:
                assert 'rerank_score' in item
                assert isinstance(item['rerank_score'], float)
                assert 0.0 <= item['rerank_score'] <= 1.0
        finally:
            await provider.shutdown()

    @pytest.mark.asyncio
    async def test_rerank_sorts_by_relevance(
        self, provider: RerankingProvider,
    ) -> None:
        """Results should be sorted by rerank_score descending."""
        await provider.initialize()
        try:
            results = [
                {'id': 1, 'text': 'Cooking recipes'},
                {'id': 2, 'text': 'Python programming tutorials'},
                {'id': 3, 'text': 'Machine learning with Python'},
            ]
            reranked = await provider.rerank('Python programming', results)

            # Verify sorted descending by score
            scores = [r['rerank_score'] for r in reranked]
            assert scores == sorted(scores, reverse=True)

            # Python-related results should score higher
            top_result = reranked[0]
            assert 'Python' in top_result['text']
        finally:
            await provider.shutdown()

    @pytest.mark.asyncio
    async def test_rerank_respects_limit(
        self, provider: RerankingProvider,
    ) -> None:
        """Limit parameter should cap result count."""
        await provider.initialize()
        try:
            results = [
                {'id': i, 'text': f'Document {i} content'}
                for i in range(10)
            ]
            reranked = await provider.rerank('content', results, limit=3)
            assert len(reranked) == 3
        finally:
            await provider.shutdown()

    @pytest.mark.asyncio
    async def test_rerank_preserves_original_fields(
        self, provider: RerankingProvider,
    ) -> None:
        """Original result fields should be preserved."""
        await provider.initialize()
        try:
            results = [
                {
                    'id': 1,
                    'text': 'Test content',
                    'custom_field': 'preserved',
                    'thread_id': 'thread-123',
                },
            ]
            reranked = await provider.rerank('test', results)

            assert len(reranked) == 1
            assert reranked[0]['custom_field'] == 'preserved'
            assert reranked[0]['thread_id'] == 'thread-123'
        finally:
            await provider.shutdown()

    @pytest.mark.asyncio
    async def test_rerank_missing_text_field_fails(
        self, provider: RerankingProvider,
    ) -> None:
        """Missing text field should raise ValueError."""
        await provider.initialize()
        try:
            results = [{'id': 1, 'title': 'No text field'}]
            with pytest.raises(ValueError, match="missing required 'text' field"):
                await provider.rerank('query', results)
        finally:
            await provider.shutdown()

    def test_provider_name(self, provider: RerankingProvider) -> None:
        """provider_name should return 'flashrank'."""
        assert provider.provider_name == 'flashrank'

    def test_model_name_default(self, provider: RerankingProvider) -> None:
        """Default model should be ms-marco-MiniLM-L-12-v2."""
        assert provider.model_name == 'ms-marco-MiniLM-L-12-v2'

    def test_model_name_from_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Model name should be configurable via environment."""
        from app.settings import get_settings
        monkeypatch.setenv('RERANKING_MODEL', 'ms-marco-TinyBERT-L-2-v2')
        get_settings.cache_clear()
        provider = create_reranking_provider()
        assert provider.model_name == 'ms-marco-TinyBERT-L-2-v2'

    @pytest.mark.asyncio
    async def test_ensure_ranker_applies_session_options(
        self, provider: RerankingProvider,
    ) -> None:
        """_ensure_ranker should replace ONNX session with thread-constrained one."""
        await provider.initialize()
        try:
            # Trigger lazy loading
            results = [{'id': 1, 'text': 'Test document content'}]
            await provider.rerank('test query', results)

            # Verify the ranker has been initialized (access internal attr via cast)
            from typing import Any
            from typing import cast
            concrete = cast(Any, provider)
            assert concrete._ranker is not None

            # Verify session has been replaced with constrained options
            session = concrete._ranker.session
            assert session is not None

            # Verify inter_op_num_threads=1 (hardcoded for sequential workload)
            opts = session.get_session_options()
            assert opts.inter_op_num_threads == 1

            # Verify CPU memory arena is disabled by default
            assert opts.enable_cpu_mem_arena is False
        finally:
            await provider.shutdown()

    @pytest.mark.asyncio
    async def test_intra_op_threads_from_settings(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """intra_op_threads should be configurable via RERANKING_INTRA_OP_THREADS."""
        from app.settings import get_settings
        monkeypatch.setenv('RERANKING_INTRA_OP_THREADS', '4')
        get_settings.cache_clear()
        provider = create_reranking_provider()

        await provider.initialize()
        try:
            results = [{'id': 1, 'text': 'Test document content'}]
            await provider.rerank('test query', results)

            from typing import Any
            from typing import cast
            concrete = cast(Any, provider)
            session = concrete._ranker.session
            opts = session.get_session_options()
            assert opts.intra_op_num_threads == 4
        finally:
            await provider.shutdown()
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_cpu_mem_arena_disabled_by_default(
        self, provider: RerankingProvider,
    ) -> None:
        """CPU memory arena should be disabled by default to prevent OOM."""
        await provider.initialize()
        try:
            results = [{'id': 1, 'text': 'Test document content'}]
            await provider.rerank('test query', results)

            from typing import Any
            from typing import cast
            concrete = cast(Any, provider)
            session = concrete._ranker.session
            opts = session.get_session_options()
            assert opts.enable_cpu_mem_arena is False
        finally:
            await provider.shutdown()

    @pytest.mark.asyncio
    async def test_cpu_mem_arena_configurable(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CPU memory arena should be configurable via RERANKING_CPU_MEM_ARENA."""
        from app.settings import get_settings
        monkeypatch.setenv('RERANKING_CPU_MEM_ARENA', 'true')
        get_settings.cache_clear()
        provider = create_reranking_provider()

        await provider.initialize()
        try:
            results = [{'id': 1, 'text': 'Test document content'}]
            await provider.rerank('test query', results)

            from typing import Any
            from typing import cast
            concrete = cast(Any, provider)
            session = concrete._ranker.session
            opts = session.get_session_options()
            assert opts.enable_cpu_mem_arena is True
        finally:
            await provider.shutdown()
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_ensure_ranker_cleans_up_default_session(
        self, provider: RerankingProvider,
    ) -> None:
        """_ensure_ranker should release the Ranker constructor's default ONNX session."""
        await provider.initialize()
        try:
            results = [{'id': 1, 'text': 'Test document content'}]
            await provider.rerank('test query', results)

            from typing import Any
            from typing import cast
            concrete = cast(Any, provider)

            # The session should be the replacement session (not None, proving cleanup + replacement worked)
            assert concrete._ranker is not None
            assert concrete._ranker.session is not None

            # Verify the session has constrained options (proving it's the replacement, not the default)
            opts = concrete._ranker.session.get_session_options()
            assert opts.inter_op_num_threads == 1
            assert opts.enable_cpu_mem_arena is False
        finally:
            await provider.shutdown()

    def test_batch_size_default(self) -> None:
        """Default batch_size should be 32."""
        from typing import Any
        from typing import cast
        provider = create_reranking_provider()
        concrete = cast(Any, provider)
        assert concrete._batch_size == 32

    def test_batch_size_from_settings(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """batch_size should be configurable via RERANKING_BATCH_SIZE."""
        from app.settings import get_settings
        monkeypatch.setenv('RERANKING_BATCH_SIZE', '64')
        get_settings.cache_clear()
        from typing import Any
        from typing import cast
        provider = create_reranking_provider()
        concrete = cast(Any, provider)
        assert concrete._batch_size == 64

    @pytest.mark.asyncio
    async def test_rerank_micro_batching_splits_passages(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Micro-batching with batch_size=2 should correctly process 5 passages."""
        from app.settings import get_settings
        monkeypatch.setenv('RERANKING_BATCH_SIZE', '2')
        get_settings.cache_clear()
        provider = create_reranking_provider()

        await provider.initialize()
        try:
            results = [
                {'id': i, 'text': f'Document about topic {i}'}
                for i in range(5)
            ]
            reranked = await provider.rerank('topic', results)

            # All 5 passages should be scored and returned
            assert len(reranked) == 5
            for item in reranked:
                assert 'rerank_score' in item
                assert isinstance(item['rerank_score'], float)
                assert 0.0 <= item['rerank_score'] <= 1.0
        finally:
            await provider.shutdown()
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_rerank_single_batch_no_change(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """batch_size=100 with 10 passages should process in a single batch."""
        from app.settings import get_settings
        monkeypatch.setenv('RERANKING_BATCH_SIZE', '100')
        get_settings.cache_clear()
        provider = create_reranking_provider()

        await provider.initialize()
        try:
            results = [
                {'id': i, 'text': f'Document {i} about Python programming'}
                for i in range(10)
            ]
            reranked = await provider.rerank('Python', results)

            assert len(reranked) == 10
            for item in reranked:
                assert 'rerank_score' in item
                assert isinstance(item['rerank_score'], float)

            # Scores should be sorted descending
            scores = [r['rerank_score'] for r in reranked]
            assert scores == sorted(scores, reverse=True)
        finally:
            await provider.shutdown()
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_rerank_exact_batch_boundary(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """batch_size=3 with 6 passages should split into exactly 2 batches."""
        from app.settings import get_settings
        monkeypatch.setenv('RERANKING_BATCH_SIZE', '3')
        get_settings.cache_clear()
        provider = create_reranking_provider()

        await provider.initialize()
        try:
            results = [
                {'id': i, 'text': f'Document {i} about search engines'}
                for i in range(6)
            ]
            reranked = await provider.rerank('search', results)

            # All 6 passages should be scored correctly
            assert len(reranked) == 6
            for item in reranked:
                assert 'rerank_score' in item
                assert isinstance(item['rerank_score'], float)
                assert 0.0 <= item['rerank_score'] <= 1.0

            # Scores should be sorted descending
            scores = [r['rerank_score'] for r in reranked]
            assert scores == sorted(scores, reverse=True)
        finally:
            await provider.shutdown()
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_rerank_uses_asyncio_to_thread(
        self, provider: RerankingProvider, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """rerank() should offload ONNX inference via asyncio.to_thread."""
        import asyncio

        await provider.initialize()
        try:
            original_to_thread = asyncio.to_thread
            to_thread_calls: list[tuple[object, ...]] = []

            async def tracking_to_thread(func, *args, **kwargs):
                to_thread_calls.append((func, args, kwargs))
                return await original_to_thread(func, *args, **kwargs)

            monkeypatch.setattr(asyncio, 'to_thread', tracking_to_thread)

            results = [
                {'id': 0, 'text': 'Test document about Python'},
                {'id': 1, 'text': 'Another document about testing'},
            ]
            reranked = await provider.rerank('Python testing', results)

            # asyncio.to_thread was called at least once per batch
            assert len(to_thread_calls) >= 1
            # Each call invoked a callable (the ranker's rerank method)
            for call_func, _call_args, _ in to_thread_calls:
                assert callable(call_func)
            # Results are still correct
            assert len(reranked) == 2
            for item in reranked:
                assert 'rerank_score' in item
        finally:
            await provider.shutdown()

    @pytest.mark.asyncio
    async def test_rerank_micro_batching_calls_to_thread_per_batch(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each micro-batch should use a separate asyncio.to_thread call."""
        import asyncio

        from app.settings import get_settings

        monkeypatch.setenv('RERANKING_BATCH_SIZE', '2')
        get_settings.cache_clear()
        provider = create_reranking_provider()

        await provider.initialize()
        try:
            original_to_thread = asyncio.to_thread
            to_thread_call_count = 0

            async def counting_to_thread(func, *args, **kwargs):
                nonlocal to_thread_call_count
                to_thread_call_count += 1
                return await original_to_thread(func, *args, **kwargs)

            monkeypatch.setattr(asyncio, 'to_thread', counting_to_thread)

            results = [
                {'id': i, 'text': f'Document about topic {i}'}
                for i in range(5)
            ]
            reranked = await provider.rerank('topic', results)

            # 5 passages / batch_size=2 = 3 batches = 3 asyncio.to_thread calls
            assert to_thread_call_count == 3
            assert len(reranked) == 5
        finally:
            await provider.shutdown()
            get_settings.cache_clear()
