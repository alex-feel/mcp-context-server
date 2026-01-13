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
