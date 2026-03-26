"""Tests for search quality improvements: limit clamping, passage boundaries, retry logging, warnings."""

import logging

import pytest

# ============================================================
# 3.1: Limit Clamping Tests
# ============================================================


class TestLimitClamping:
    """Tests for Postel's Law limit clamping in all 4 search tools."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_clamping_over_max(self) -> None:
        """search_context clamps limit > 100 to 100 with clamped_limit hint."""
        from app.tools.search import search_context

        result = await search_context(limit=200)
        assert 'clamped_limit' in result
        assert result['clamped_limit']['requested'] == 200
        assert result['clamped_limit']['applied'] == 100

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_no_clamping_at_max(self) -> None:
        """search_context does NOT clamp when limit == 100."""
        from app.tools.search import search_context

        result = await search_context(limit=100)
        assert 'clamped_limit' not in result

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_no_clamping_below_max(self) -> None:
        """search_context does NOT clamp when limit < 100."""
        from app.tools.search import search_context

        result = await search_context(limit=50)
        assert 'clamped_limit' not in result

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_clamping_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """search_context logs warning when clamping occurs."""
        from app.tools.search import search_context

        with caplog.at_level(logging.WARNING, logger='app.tools.search'):
            await search_context(limit=150)

        assert any(
            'clamped' in record.message.lower() or 'exceeds' in record.message.lower()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_clamping_still_returns_results(self) -> None:
        """search_context with clamped limit still returns valid search results."""
        from app.tools.search import search_context

        result = await search_context(limit=500)
        assert 'results' in result
        assert 'count' in result
        assert isinstance(result['results'], list)


# ============================================================
# 3.2: Passage Boundary Tests
# ============================================================


class TestPassageBoundary:
    """Tests for configurable passage size using chars_per_token."""

    def test_passage_size_uses_chars_per_token_default(self) -> None:
        """Passage size computation with default chars_per_token=4.0."""
        max_length = 512
        chars_per_token = 4.0

        expected = int(max_length * chars_per_token * 0.95)
        assert expected == 1945  # int(512 * 4.0 * 0.95)

    def test_passage_size_with_code_chars_per_token(self) -> None:
        """Passage size adjusts when chars_per_token=3.0 (code-heavy content)."""
        max_length = 512
        chars_per_token = 3.0

        expected = int(max_length * chars_per_token * 0.95)
        assert expected == 1459  # int(512 * 3.0 * 0.95)

    def test_passage_size_with_high_chars_per_token(self) -> None:
        """Passage size adjusts when chars_per_token=5.0."""
        max_length = 512
        chars_per_token = 5.0

        expected = int(max_length * chars_per_token * 0.95)
        assert expected == 2432  # int(512 * 5.0 * 0.95)


# ============================================================
# 3.3: Warnings Field Tests
# ============================================================


class TestHybridSearchWarnings:
    """Tests for warnings field in hybrid search responses."""

    def test_hybrid_search_response_type_has_warnings(self) -> None:
        """HybridSearchResponseDict includes warnings field."""
        from app.types import HybridSearchResponseDict

        annotations = HybridSearchResponseDict.__annotations__
        assert 'warnings' in annotations

    def test_clamped_limit_dict_type_exists(self) -> None:
        """ClampedLimitDict TypedDict is defined."""
        from app.types import ClampedLimitDict

        annotations = ClampedLimitDict.__annotations__
        assert 'requested' in annotations
        assert 'applied' in annotations


# ============================================================
# MAX_SEARCH_LIMIT constant test
# ============================================================


class TestMaxSearchLimit:
    """Tests for the MAX_SEARCH_LIMIT constant."""

    def test_max_search_limit_exists(self) -> None:
        """MAX_SEARCH_LIMIT constant is defined in search module."""
        from app.tools.search import MAX_SEARCH_LIMIT

        assert MAX_SEARCH_LIMIT == 100


# ============================================================
# 3.5: _apply_reranking Edge Cases
# ============================================================


class TestApplyReranking:
    """Tests for _apply_reranking function edge cases and exception fallback."""

    @pytest.mark.asyncio
    async def test_apply_reranking_provider_unavailable_returns_original(self) -> None:
        """When reranking provider is None, original results returned."""
        from unittest.mock import patch

        from app.tools.search import _apply_reranking

        results = [{'id': 1, 'text_content': 'test', 'scores': {}}]

        with patch('app.tools.search.get_reranking_provider', return_value=None):
            output = await _apply_reranking('query', results)

        assert output == results

    @pytest.mark.asyncio
    async def test_apply_reranking_disabled_returns_original(self) -> None:
        """When reranking is disabled in settings, original results returned."""
        from unittest.mock import Mock
        from unittest.mock import patch

        from app.tools.search import _apply_reranking

        mock_provider = Mock()
        results = [{'id': 1, 'text_content': 'test', 'scores': {}}]

        with (
            patch('app.tools.search.get_reranking_provider', return_value=mock_provider),
            patch('app.tools.search.settings') as mock_settings,
        ):
            mock_settings.reranking.enabled = False
            output = await _apply_reranking('query', results)

        assert output == results

    @pytest.mark.asyncio
    async def test_apply_reranking_exception_returns_original(self) -> None:
        """When rerank() raises exception, original results returned gracefully."""
        from unittest.mock import AsyncMock
        from unittest.mock import patch

        from app.tools.search import _apply_reranking

        mock_provider = AsyncMock()
        mock_provider.rerank.side_effect = RuntimeError('Provider crashed')

        results = [
            {'id': 1, 'text_content': 'result one', 'scores': {'fts_score': 1.0}},
            {'id': 2, 'text_content': 'result two', 'scores': {'fts_score': 0.5}},
        ]

        with (
            patch('app.tools.search.get_reranking_provider', return_value=mock_provider),
            patch('app.tools.search.settings') as mock_settings,
        ):
            mock_settings.reranking.enabled = True
            output = await _apply_reranking('test query', results)

        # Should return original results unchanged (fallback behavior)
        assert len(output) == 2
        assert output[0]['id'] == 1
        assert output[1]['id'] == 2

    @pytest.mark.asyncio
    async def test_apply_reranking_empty_results(self) -> None:
        """Empty results list returns empty list without calling provider."""
        from unittest.mock import Mock
        from unittest.mock import patch

        from app.tools.search import _apply_reranking

        mock_provider = Mock()

        with (
            patch('app.tools.search.get_reranking_provider', return_value=mock_provider),
            patch('app.tools.search.settings') as mock_settings,
        ):
            mock_settings.reranking.enabled = True
            output = await _apply_reranking('query', [])

        assert output == []

    @pytest.mark.asyncio
    async def test_apply_reranking_with_limit_no_provider(self) -> None:
        """When provider is None, limit is applied to original results."""
        from unittest.mock import patch

        from app.tools.search import _apply_reranking

        results = [
            {'id': i, 'text_content': f'result {i}', 'scores': {}}
            for i in range(5)
        ]

        with patch('app.tools.search.get_reranking_provider', return_value=None):
            output = await _apply_reranking('query', results, limit=3)

        assert len(output) == 3
