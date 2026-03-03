"""
Tests for embedding concurrency controls and dynamic timeout computation.

Tests verify:
- compute_embedding_total_timeout() produces correct values for various settings
- Semaphore limits concurrent embedding operations
- Total timeout wrapper raises ToolError on expiration
- FTS/semantic error logging in hybrid search graceful degradation
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# --- Tests for compute_embedding_total_timeout() ---


@pytest.fixture
def mock_retry_settings():
    """Mock settings for compute_embedding_total_timeout tests."""
    with patch('app.embeddings.retry.get_settings') as mock:
        yield mock


def test_compute_timeout_default_settings(mock_retry_settings: MagicMock) -> None:
    """Test timeout computation with default settings (30s, 3 attempts, 1.0s delay)."""
    mock_retry_settings.return_value.embedding.timeout_s = 30.0
    mock_retry_settings.return_value.embedding.retry_max_attempts = 3
    mock_retry_settings.return_value.embedding.retry_base_delay_s = 1.0

    from app.embeddings.retry import compute_embedding_total_timeout

    result = compute_embedding_total_timeout()

    # (3 * 30 + (min(1*1+1, 60) + min(1*2+1, 60))) * 1.1
    # = (90 + (2 + 3)) * 1.1 = 95 * 1.1 = 104.5
    assert result == pytest.approx(104.5, abs=0.1)


def test_compute_timeout_user_deployed_settings(mock_retry_settings: MagicMock) -> None:
    """Test timeout computation with user deployed settings (60s, 3 attempts, 1.0s delay)."""
    mock_retry_settings.return_value.embedding.timeout_s = 60.0
    mock_retry_settings.return_value.embedding.retry_max_attempts = 3
    mock_retry_settings.return_value.embedding.retry_base_delay_s = 1.0

    from app.embeddings.retry import compute_embedding_total_timeout

    result = compute_embedding_total_timeout()

    # (3 * 60 + (2 + 3)) * 1.1 = 185 * 1.1 = 203.5
    assert result == pytest.approx(203.5, abs=0.1)


def test_compute_timeout_single_attempt(mock_retry_settings: MagicMock) -> None:
    """Test timeout computation with single attempt (no backoff intervals)."""
    mock_retry_settings.return_value.embedding.timeout_s = 30.0
    mock_retry_settings.return_value.embedding.retry_max_attempts = 1
    mock_retry_settings.return_value.embedding.retry_base_delay_s = 1.0

    from app.embeddings.retry import compute_embedding_total_timeout

    result = compute_embedding_total_timeout()

    # (1 * 30 + 0) * 1.1 = 33.0 (no backoff with single attempt)
    assert result == pytest.approx(33.0, abs=0.1)


def test_compute_timeout_high_config(mock_retry_settings: MagicMock) -> None:
    """Test timeout computation with high configuration values."""
    mock_retry_settings.return_value.embedding.timeout_s = 120.0
    mock_retry_settings.return_value.embedding.retry_max_attempts = 5
    mock_retry_settings.return_value.embedding.retry_base_delay_s = 2.0

    from app.embeddings.retry import compute_embedding_total_timeout

    result = compute_embedding_total_timeout()

    # attempt_time = 5 * 120 = 600
    # backoffs: min(2*1+2,60)=4, min(2*2+2,60)=6, min(2*4+2,60)=10, min(2*8+2,60)=18
    # total_backoff = 4 + 6 + 10 + 18 = 38
    # (600 + 38) * 1.1 = 701.8
    assert result == pytest.approx(701.8, abs=0.1)


def test_compute_timeout_backoff_cap(mock_retry_settings: MagicMock) -> None:
    """Test that backoff intervals are capped at 60 seconds."""
    mock_retry_settings.return_value.embedding.timeout_s = 10.0
    mock_retry_settings.return_value.embedding.retry_max_attempts = 10
    mock_retry_settings.return_value.embedding.retry_base_delay_s = 10.0

    from app.embeddings.retry import compute_embedding_total_timeout

    result = compute_embedding_total_timeout()

    # attempt_time = 10 * 10 = 100
    # backoffs for i=0..8: min(10*(2^i)+10, 60)
    # i=0: min(20, 60)=20
    # i=1: min(30, 60)=30
    # i=2: min(50, 60)=50
    # i=3: min(90, 60)=60  (capped)
    # i=4..8: all capped at 60
    # total_backoff = 20+30+50+60*6 = 460
    # (100 + 460) * 1.1 = 616.0
    assert result == pytest.approx(616.0, abs=0.1)


# --- Tests for semaphore concurrency control ---


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency() -> None:
    """Verify that the semaphore limits concurrent embedding calls."""
    max_concurrent_seen = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def mock_generate(_text: str) -> list[MagicMock]:
        nonlocal max_concurrent_seen, current_concurrent
        async with lock:
            current_concurrent += 1
            if current_concurrent > max_concurrent_seen:
                max_concurrent_seen = current_concurrent
        await asyncio.sleep(0.05)
        async with lock:
            current_concurrent -= 1
        return [MagicMock()]

    max_concurrent_setting = 2

    with (
        patch('app.tools.context.get_embedding_provider', return_value=MagicMock()),
        patch('app.tools.context.compute_embedding_total_timeout', return_value=999.0),
        patch('app.tools.context._generate_embeddings_for_text', side_effect=mock_generate),
        patch('app.tools.context.settings') as mock_settings,
    ):
        mock_settings.embedding.max_concurrent = max_concurrent_setting

        # Reset the module-level semaphore to pick up our test setting
        import app.tools.context as ctx_module

        original_semaphore = ctx_module._embedding_semaphore
        ctx_module._embedding_semaphore = None

        try:
            sem = ctx_module._get_embedding_semaphore()

            # Launch 4 concurrent tasks through the semaphore
            async def run_with_semaphore():
                async with sem:
                    await mock_generate('test')

            tasks = [run_with_semaphore() for _ in range(4)]
            await asyncio.gather(*tasks)

            assert max_concurrent_seen <= max_concurrent_setting
        finally:
            ctx_module._embedding_semaphore = original_semaphore


@pytest.mark.asyncio
async def test_total_timeout_raises_tool_error() -> None:
    """Verify that exceeding total timeout raises ToolError."""
    from fastmcp.exceptions import ToolError

    async def slow_embedding(_text: str) -> list[MagicMock]:
        await asyncio.sleep(10.0)
        return [MagicMock()]

    with (
        patch('app.tools.context.get_embedding_provider', return_value=MagicMock()),
        patch('app.tools.context.compute_embedding_total_timeout', return_value=0.05),
        patch('app.tools.context._generate_embeddings_for_text', side_effect=slow_embedding),
        patch('app.tools.context.settings') as mock_settings,
        patch('app.tools.context.ensure_repositories', new_callable=AsyncMock),
    ):
        mock_settings.embedding.max_concurrent = 3

        import app.tools.context as ctx_module

        original_semaphore = ctx_module._embedding_semaphore
        ctx_module._embedding_semaphore = None

        try:
            from app.tools.context import store_context

            with pytest.raises(ToolError, match='total timeout'):
                await store_context(
                    thread_id='test-thread',
                    source='agent',
                    text='Test text for embedding timeout',
                )
        finally:
            ctx_module._embedding_semaphore = original_semaphore


@pytest.mark.asyncio
async def test_embedding_disabled_skips_semaphore() -> None:
    """Verify that when embedding provider is None, semaphore is not acquired."""
    with (
        patch('app.tools.context.get_embedding_provider', return_value=None),
        patch('app.tools.context._get_embedding_semaphore') as mock_sem,
        patch('app.tools.context.ensure_repositories', new_callable=AsyncMock) as mock_repos,
    ):
        mock_backend = MagicMock()
        mock_txn = MagicMock()
        mock_txn.__aenter__ = AsyncMock(return_value=mock_txn)
        mock_txn.__aexit__ = AsyncMock(return_value=False)
        mock_backend.begin_transaction.return_value = mock_txn
        mock_repos.return_value.context.backend = mock_backend
        mock_repos.return_value.context.store_with_deduplication = AsyncMock(return_value=(1, False))
        mock_repos.return_value.tags.store_tags = AsyncMock()
        mock_repos.return_value.images.store_images = AsyncMock()
        mock_repos.return_value.embeddings.store_chunked = AsyncMock()

        from app.tools.context import store_context

        result = await store_context(
            thread_id='test-thread',
            source='agent',
            text='Test text without embedding',
        )

        assert result['success'] is True
        mock_sem.assert_not_called()


# --- Tests for hybrid search FTS/semantic error logging ---


@pytest.mark.asyncio
async def test_hybrid_search_logs_fts_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Verify warning logged when FTS fails but semantic succeeds."""
    with (
        patch('app.tools.search.settings') as mock_settings,
        patch('app.tools.search.get_embedding_provider', return_value=MagicMock()),
        patch('app.tools.search.get_reranking_provider', return_value=None),
        patch('app.tools.search.ensure_repositories', new_callable=AsyncMock) as mock_repos,
        patch('app.tools.search._fts_search_raw', new_callable=AsyncMock) as mock_fts,
        patch('app.tools.search._semantic_search_raw', new_callable=AsyncMock) as mock_semantic,
        patch('app.tools.search._apply_reranking', new_callable=AsyncMock) as mock_rerank,
    ):
        mock_settings.hybrid_search.enabled = True
        mock_settings.hybrid_search.rrf_k = 60
        mock_settings.hybrid_search.rrf_overfetch = 2
        mock_settings.hybrid_search.fts_or_threshold = 4
        mock_settings.storage.backend_type = 'sqlite'
        mock_settings.fts.enabled = True
        mock_settings.semantic_search.enabled = True
        mock_settings.reranking.enabled = False
        mock_settings.embedding.provider = 'ollama'

        # FTS fails, semantic succeeds
        mock_fts.side_effect = Exception('FTS index corrupted')
        mock_semantic.return_value = (
            [{'id': 1, 'text_content': 'test', 'metadata': None, 'source': 'agent', 'content_type': 'text'}],
            None,
        )
        mock_rerank.return_value = [
            {'id': 1, 'text_content': 'test', 'metadata': None, 'source': 'agent', 'content_type': 'text',
             'scores': {'rrf': 0.5}},
        ]
        mock_repos.return_value.tags.get_tags_for_context = AsyncMock(return_value=[])
        mock_repos.return_value.images.get_images_for_context = AsyncMock(return_value=[])

        from app.tools.search import hybrid_search_context

        with caplog.at_level(logging.WARNING, logger='app.tools.search'):
            await hybrid_search_context(query='test query')

        assert any('FTS sub-search failed' in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_hybrid_search_logs_semantic_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Verify warning logged when semantic fails but FTS succeeds."""
    with (
        patch('app.tools.search.settings') as mock_settings,
        patch('app.tools.search.get_embedding_provider', return_value=MagicMock()),
        patch('app.tools.search.get_reranking_provider', return_value=None),
        patch('app.tools.search.ensure_repositories', new_callable=AsyncMock) as mock_repos,
        patch('app.tools.search._fts_search_raw', new_callable=AsyncMock) as mock_fts,
        patch('app.tools.search._semantic_search_raw', new_callable=AsyncMock) as mock_semantic,
        patch('app.tools.search._apply_reranking', new_callable=AsyncMock) as mock_rerank,
    ):
        mock_settings.hybrid_search.enabled = True
        mock_settings.hybrid_search.rrf_k = 60
        mock_settings.hybrid_search.rrf_overfetch = 2
        mock_settings.hybrid_search.fts_or_threshold = 4
        mock_settings.storage.backend_type = 'sqlite'
        mock_settings.fts.enabled = True
        mock_settings.semantic_search.enabled = True
        mock_settings.reranking.enabled = False
        mock_settings.embedding.provider = 'ollama'

        # FTS succeeds, semantic fails
        mock_fts.return_value = (
            [{'id': 1, 'text_content': 'test', 'metadata': '{"key": "val"}', 'source': 'agent',
              'content_type': 'text'}],
            None,
        )
        mock_semantic.side_effect = Exception('Ollama unreachable')
        mock_rerank.return_value = [
            {'id': 1, 'text_content': 'test', 'metadata': {'key': 'val'}, 'source': 'agent', 'content_type': 'text',
             'scores': {'rrf': 0.5}},
        ]
        mock_repos.return_value.tags.get_tags_for_context = AsyncMock(return_value=[])
        mock_repos.return_value.images.get_images_for_context = AsyncMock(return_value=[])

        from app.tools.search import hybrid_search_context

        with caplog.at_level(logging.WARNING, logger='app.tools.search'):
            await hybrid_search_context(query='test query')

        assert any('semantic sub-search failed' in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_hybrid_search_no_warning_on_success(caplog: pytest.LogCaptureFixture) -> None:
    """Verify no warning logged when both sub-searches succeed."""
    with (
        patch('app.tools.search.settings') as mock_settings,
        patch('app.tools.search.get_embedding_provider', return_value=MagicMock()),
        patch('app.tools.search.get_reranking_provider', return_value=None),
        patch('app.tools.search.ensure_repositories', new_callable=AsyncMock) as mock_repos,
        patch('app.tools.search._fts_search_raw', new_callable=AsyncMock) as mock_fts,
        patch('app.tools.search._semantic_search_raw', new_callable=AsyncMock) as mock_semantic,
        patch('app.tools.search._apply_reranking', new_callable=AsyncMock) as mock_rerank,
    ):
        mock_settings.hybrid_search.enabled = True
        mock_settings.hybrid_search.rrf_k = 60
        mock_settings.hybrid_search.rrf_overfetch = 2
        mock_settings.hybrid_search.fts_or_threshold = 4
        mock_settings.storage.backend_type = 'sqlite'
        mock_settings.fts.enabled = True
        mock_settings.semantic_search.enabled = True
        mock_settings.reranking.enabled = False
        mock_settings.embedding.provider = 'ollama'

        mock_fts.return_value = (
            [{'id': 1, 'text_content': 'test', 'metadata': None, 'source': 'agent', 'content_type': 'text'}],
            None,
        )
        mock_semantic.return_value = (
            [{'id': 1, 'text_content': 'test', 'metadata': None, 'source': 'agent', 'content_type': 'text'}],
            None,
        )
        mock_rerank.return_value = [
            {'id': 1, 'text_content': 'test', 'metadata': None, 'source': 'agent', 'content_type': 'text',
             'scores': {'rrf': 0.5}},
        ]
        mock_repos.return_value.tags.get_tags_for_context = AsyncMock(return_value=[])
        mock_repos.return_value.images.get_images_for_context = AsyncMock(return_value=[])

        from app.tools.search import hybrid_search_context

        with caplog.at_level(logging.WARNING, logger='app.tools.search'):
            await hybrid_search_context(query='test query')

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any('sub-search failed' in r.message for r in warning_records)


# --- Tests for EMBEDDING_MAX_CONCURRENT setting ---


def test_embedding_max_concurrent_setting_default() -> None:
    """Test that EMBEDDING_MAX_CONCURRENT has correct default value."""
    from app.settings import EmbeddingSettings

    settings = EmbeddingSettings()
    assert settings.max_concurrent == 3


def test_embedding_max_concurrent_setting_bounds() -> None:
    """Test that EMBEDDING_MAX_CONCURRENT enforces bounds."""
    from pydantic import ValidationError

    from app.settings import EmbeddingSettings

    with pytest.raises(ValidationError):
        EmbeddingSettings.model_validate({'EMBEDDING_MAX_CONCURRENT': 0})

    with pytest.raises(ValidationError):
        EmbeddingSettings.model_validate({'EMBEDDING_MAX_CONCURRENT': 21})

    # Valid boundary values
    settings_min = EmbeddingSettings.model_validate({'EMBEDDING_MAX_CONCURRENT': 1})
    assert settings_min.max_concurrent == 1

    settings_max = EmbeddingSettings.model_validate({'EMBEDDING_MAX_CONCURRENT': 20})
    assert settings_max.max_concurrent == 20


# --- Tests for _generate_embeddings_with_timeout helper ---


@pytest.mark.asyncio
async def test_generate_embeddings_with_timeout_returns_none_when_no_provider() -> None:
    """Verify helper returns None when embedding provider is not configured."""
    with patch('app.tools.context.get_embedding_provider', return_value=None):
        from app.tools.context import _generate_embeddings_with_timeout

        result = await _generate_embeddings_with_timeout('test text')
        assert result is None


@pytest.mark.asyncio
async def test_generate_embeddings_with_timeout_success() -> None:
    """Verify helper returns embeddings on success."""
    mock_embeddings = [MagicMock()]

    with (
        patch('app.tools.context.get_embedding_provider', return_value=MagicMock()),
        patch('app.tools.context.compute_embedding_total_timeout', return_value=999.0),
        patch('app.tools.context._generate_embeddings_for_text', new_callable=AsyncMock, return_value=mock_embeddings),
        patch('app.tools.context.settings') as mock_settings,
    ):
        mock_settings.embedding.max_concurrent = 3

        import app.tools.context as ctx_module

        original_semaphore = ctx_module._embedding_semaphore
        ctx_module._embedding_semaphore = None

        try:
            from app.tools.context import _generate_embeddings_with_timeout

            result = await _generate_embeddings_with_timeout('test text')
            assert result == mock_embeddings
        finally:
            ctx_module._embedding_semaphore = original_semaphore


@pytest.mark.asyncio
async def test_generate_embeddings_with_timeout_raises_on_timeout() -> None:
    """Verify helper raises ToolError when embedding generation times out."""
    from fastmcp.exceptions import ToolError

    async def slow_embedding(_text: str) -> list[MagicMock]:
        await asyncio.sleep(10.0)
        return [MagicMock()]

    with (
        patch('app.tools.context.get_embedding_provider', return_value=MagicMock()),
        patch('app.tools.context.compute_embedding_total_timeout', return_value=0.05),
        patch('app.tools.context._generate_embeddings_for_text', side_effect=slow_embedding),
        patch('app.tools.context.settings') as mock_settings,
    ):
        mock_settings.embedding.max_concurrent = 3

        import app.tools.context as ctx_module

        original_semaphore = ctx_module._embedding_semaphore
        ctx_module._embedding_semaphore = None

        try:
            from app.tools.context import _generate_embeddings_with_timeout

            with pytest.raises(ToolError, match='total timeout'):
                await _generate_embeddings_with_timeout('test text')
        finally:
            ctx_module._embedding_semaphore = original_semaphore
