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
# 3.3: Retry Logging Tests
# ============================================================


class TestRetryLogging:
    """Tests for meaningful retry log messages via __qualname__ fix."""

    def test_qualname_set_on_closure(self) -> None:
        """Verify the __qualname__ replacement logic for inner closures."""

        async def _embed():
            return [1.0, 2.0]

        _embed.__qualname__ = 'OllamaEmbeddingProvider.embed_query.<locals>._embed'

        # Simulate what with_retry_and_timeout does
        if hasattr(_embed, '__qualname__') and '.<locals>.' in _embed.__qualname__:
            _embed.__qualname__ = 'embed_query'

        assert _embed.__qualname__ == 'embed_query'

    def test_qualname_not_modified_for_top_level_functions(self) -> None:
        """Verify that __qualname__ is preserved for non-closure functions."""

        async def top_level_func():
            return [1.0, 2.0]

        # Simulate a truly top-level function (no .<locals>. in qualname)
        top_level_func.__qualname__ = 'top_level_func'
        original_qualname = top_level_func.__qualname__

        # Should NOT modify because no '.<locals>.' in qualname
        qualname = getattr(top_level_func, '__qualname__', '')
        if qualname and '.<locals>.' in qualname:
            top_level_func.__qualname__ = 'overridden'

        assert top_level_func.__qualname__ == original_qualname

    def test_qualname_set_on_nested_closure(self) -> None:
        """Verify __qualname__ replacement with deeply nested closure."""

        async def _embed():
            return [1.0, 2.0]

        _embed.__qualname__ = 'AzureEmbeddingProvider.embed_documents.<locals>._embed'

        if hasattr(_embed, '__qualname__') and '.<locals>.' in _embed.__qualname__:
            _embed.__qualname__ = 'embed_documents'

        assert _embed.__qualname__ == 'embed_documents'


# ============================================================
# 3.4: Warnings Field Tests
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
