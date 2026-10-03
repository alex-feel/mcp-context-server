"""Tests for summary generation in store_context_batch, in atomic and non-atomic modes."""

import asyncio
from collections.abc import Generator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools
from tests.tools.batch._mocks import create_mock_repositories
from tests.tools.batch._mocks import preserved_providers

store_context_batch = app.tools.store_context_batch


@pytest.fixture(autouse=True)
def reset_providers() -> Generator[None, None, None]:
    """Reset global provider state between tests."""
    with preserved_providers():
        yield


@pytest.mark.usefixtures('mock_server_dependencies')
class TestStoreContextBatchWithSummary:
    """Tests for summary generation in store_context_batch."""

    @pytest.mark.asyncio
    async def test_store_batch_with_summary_generated(self) -> None:
        """Generate summaries for all entries when provider is configured."""
        repos = create_mock_repositories()
        # Return unique IDs for each entry
        repos.context.store_with_deduplication = AsyncMock(
            side_effect=[(101, False), (102, False)],
        )

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Batch summary')

        entries = [
            {'thread_id': 'batch-sum-1', 'source': 'user', 'text': 'x' * 500},
            {'thread_id': 'batch-sum-1', 'source': 'agent', 'text': 'y' * 500},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        assert result['succeeded'] == 2
        assert '(summaries generated)' in result['message']
        assert mock_summary.summarize.await_count == 2

        # Verify summary was passed to store_with_deduplication for each entry
        for call in repos.context.store_with_deduplication.call_args_list:
            assert call.kwargs.get('summary') == 'Batch summary' or call[1].get('summary') == 'Batch summary'

    @pytest.mark.asyncio
    async def test_store_batch_summary_disabled(self) -> None:
        """Skip summary generation when provider is not configured."""
        repos = create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(
            side_effect=[(101, False), (102, False)],
        )

        entries = [
            {'thread_id': 'batch-no-sum', 'source': 'user', 'text': 'First entry'},
            {'thread_id': 'batch-no-sum', 'source': 'agent', 'text': 'Second entry'},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_summary_provider', return_value=None),
        ):
            result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        assert '(summaries generated)' not in result['message']

        # Verify summary=None was passed to store_with_deduplication
        for call in repos.context.store_with_deduplication.call_args_list:
            assert call.kwargs.get('summary') is None

    @pytest.mark.asyncio
    async def test_store_batch_atomic_summary_failure(self) -> None:
        """Fail entire atomic batch when summary generation fails."""
        repos = create_mock_repositories()

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(side_effect=RuntimeError('LLM unavailable'))

        entries = [
            {'thread_id': 'batch-fail-1', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
            pytest.raises(ToolError, match='Generation failed'),
        ):
            await store_context_batch(entries=entries, atomic=True)

        # No data should have been stored
        repos.context.store_with_deduplication.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_batch_non_atomic_partial_summary_failure(self) -> None:
        """Report per-entry errors in non-atomic mode when summary fails for some."""
        repos = create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(return_value=(101, False))

        call_count = 0

        async def selective_summary(_text: str, _source: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError('LLM overloaded')
            return 'Generated summary'

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(side_effect=selective_summary)

        entries = [
            {'thread_id': 'partial-sum', 'source': 'user', 'text': 'x' * 500},
            {'thread_id': 'partial-sum', 'source': 'agent', 'text': 'y' * 500},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await store_context_batch(entries=entries, atomic=False)

        assert result['succeeded'] == 1
        assert result['failed'] == 1

        failed_results = [r for r in result['results'] if not r['success']]
        assert len(failed_results) == 1
        assert failed_results[0]['error'] is not None
        assert 'Generation failed' in failed_results[0]['error']

    @pytest.mark.asyncio
    async def test_store_batch_dedup_preserves_summary(self) -> None:
        """Pass generated summary through to store_with_deduplication for dedup entries."""
        repos = create_mock_repositories()
        # Simulate deduplication: was_updated=True
        repos.context.store_with_deduplication = AsyncMock(return_value=(200, True))
        repos.embeddings.exists = AsyncMock(return_value=True)

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='New summary for dedup')

        entries = [
            {'thread_id': 'dedup-sum', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        repos.context.store_with_deduplication.assert_awaited_once()
        call_kwargs = repos.context.store_with_deduplication.call_args.kwargs
        assert call_kwargs['summary'] == 'New summary for dedup'

    @pytest.mark.asyncio
    async def test_store_batch_summary_timeout(self) -> None:
        """Fail atomic batch when summary generation times out."""
        repos = create_mock_repositories()

        async def slow_summary(_text: str) -> str:
            await asyncio.sleep(0.5)
            return 'Too late'

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(side_effect=slow_summary)

        entries = [
            {'thread_id': 'timeout-sum', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=0.01),
            pytest.raises(ToolError, match='Generation failed'),
        ):
            await store_context_batch(entries=entries, atomic=True)

        repos.context.store_with_deduplication.assert_not_awaited()


@pytest.mark.usefixtures('mock_server_dependencies')
class TestBatchSummaryEdgeCases:
    """Tests for edge cases in batch summary operations."""

    @pytest.mark.asyncio
    async def test_store_batch_both_embedding_and_summary(self) -> None:
        """Generate both embeddings and summaries when both providers configured."""
        repos = create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(return_value=(300, False))

        mock_embedding = MagicMock()
        mock_embedding.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Combined summary')

        entries = [
            {'thread_id': 'both-gen', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
            patch('app.startup.get_chunking_service', return_value=None),
            patch('app.tools._generation.get_chunking_service', return_value=None),
        ):
            result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        assert 'embeddings generated' in result['message']
        assert 'summaries generated' in result['message']
        mock_embedding.embed_query.assert_awaited_once()
        mock_summary.summarize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_batch_embedding_fails_summary_runs_non_atomic(self) -> None:
        """Both embedding and summary run in parallel via gather; entry fails on embedding error."""
        repos = create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(return_value=(400, False))

        mock_embedding = MagicMock()
        mock_embedding.embed_query = AsyncMock(side_effect=RuntimeError('Embedding failed'))

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Summary text')

        entries = [
            {'thread_id': 'emb-fail', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.store.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.store.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools.batch.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=1.0),
            patch('app.startup.get_chunking_service', return_value=None),
            patch('app.tools._generation.get_chunking_service', return_value=None),
        ):
            result = await store_context_batch(entries=entries, atomic=False)

        assert result['failed'] == 1
        # With parallel gather, summary IS called even when embedding fails
        mock_summary.summarize.assert_awaited_once()
