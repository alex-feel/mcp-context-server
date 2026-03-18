"""Tests for summary generation in batch operations.

Covers store_context_batch and update_context_batch summary generation
in both atomic and non-atomic modes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.server
import app.startup

store_context_batch = app.server.store_context_batch
update_context_batch = app.server.update_context_batch


def _create_mock_repositories() -> MagicMock:
    """Create mock repositories with transaction support for batch tool tests."""
    repos = MagicMock()

    mock_backend = MagicMock()

    @asynccontextmanager
    async def mock_begin_transaction():
        txn = MagicMock()
        txn.backend_type = 'sqlite'
        txn.connection = MagicMock()
        yield txn

    mock_backend.begin_transaction = mock_begin_transaction

    repos.context = MagicMock()
    repos.context.backend = mock_backend
    repos.context.store_with_deduplication = AsyncMock(return_value=(100, False))
    repos.context.check_entry_exists = AsyncMock(return_value=(True, 'agent'))
    repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content', 'summary']))
    repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
    repos.context.update_content_type = AsyncMock(return_value=True)

    repos.tags = MagicMock()
    repos.tags.store_tags = AsyncMock()
    repos.tags.replace_tags_for_context = AsyncMock()

    repos.images = MagicMock()
    repos.images.store_images = AsyncMock()
    repos.images.replace_images_for_context = AsyncMock()

    repos.embeddings = MagicMock()
    repos.embeddings.exists = AsyncMock(return_value=False)
    repos.embeddings.store_chunked = AsyncMock()
    repos.embeddings.delete_all_chunks = AsyncMock()

    return repos


@pytest.fixture(autouse=True)
def reset_providers() -> Generator[None, None, None]:
    """Reset global provider state between tests."""
    original_summary = app.startup.get_summary_provider()
    original_embedding = app.startup.get_embedding_provider()
    try:
        yield
    finally:
        app.startup.set_summary_provider(original_summary)
        app.startup.set_embedding_provider(original_embedding)


@pytest.mark.usefixtures('mock_server_dependencies')
class TestStoreContextBatchWithSummary:
    """Tests for summary generation in store_context_batch."""

    @pytest.mark.asyncio
    async def test_store_batch_with_summary_generated(self) -> None:
        """Generate summaries for all entries when provider is configured."""
        repos = _create_mock_repositories()
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
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
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
        repos = _create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(
            side_effect=[(101, False), (102, False)],
        )

        entries = [
            {'thread_id': 'batch-no-sum', 'source': 'user', 'text': 'First entry'},
            {'thread_id': 'batch-no-sum', 'source': 'agent', 'text': 'Second entry'},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.tools.context.get_summary_provider', return_value=None),
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
        repos = _create_mock_repositories()

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(side_effect=RuntimeError('LLM unavailable'))

        entries = [
            {'thread_id': 'batch-fail-1', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
            pytest.raises(ToolError, match='Generation failed'),
        ):
            await store_context_batch(entries=entries, atomic=True)

        # No data should have been stored
        repos.context.store_with_deduplication.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_batch_non_atomic_partial_summary_failure(self) -> None:
        """Report per-entry errors in non-atomic mode when summary fails for some."""
        repos = _create_mock_repositories()
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
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
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
        repos = _create_mock_repositories()
        # Simulate deduplication: was_updated=True
        repos.context.store_with_deduplication = AsyncMock(return_value=(200, True))
        repos.embeddings.exists = AsyncMock(return_value=True)

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='New summary for dedup')

        entries = [
            {'thread_id': 'dedup-sum', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        repos.context.store_with_deduplication.assert_awaited_once()
        call_kwargs = repos.context.store_with_deduplication.call_args.kwargs
        assert call_kwargs['summary'] == 'New summary for dedup'

    @pytest.mark.asyncio
    async def test_store_batch_summary_timeout(self) -> None:
        """Fail atomic batch when summary generation times out."""
        repos = _create_mock_repositories()

        async def slow_summary(_text: str) -> str:
            await asyncio.sleep(0.5)
            return 'Too late'

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(side_effect=slow_summary)

        entries = [
            {'thread_id': 'timeout-sum', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=0.01),
            pytest.raises(ToolError, match='Generation failed'),
        ):
            await store_context_batch(entries=entries, atomic=True)

        repos.context.store_with_deduplication.assert_not_awaited()


@pytest.mark.usefixtures('mock_server_dependencies')
class TestUpdateContextBatchWithSummary:
    """Tests for summary generation in update_context_batch."""

    @pytest.mark.asyncio
    async def test_update_batch_text_change_regenerates_summary(self) -> None:
        """Generate new summary when text is changed in batch update."""
        repos = _create_mock_repositories()

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Updated batch summary')

        updates = [
            {'context_id': 1, 'text': 'x' * 500},
            {'context_id': 2, 'text': 'y' * 500},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        assert result['succeeded'] == 2
        assert '(summaries regenerated)' in result['message']
        assert mock_summary.summarize.await_count == 2

        # Verify summary passed to update_context_entry
        for call in repos.context.update_context_entry.call_args_list:
            assert call.kwargs.get('summary') == 'Updated batch summary'

    @pytest.mark.asyncio
    async def test_update_batch_no_text_change_skips_summary(self) -> None:
        """Skip summary generation when only metadata is updated."""
        repos = _create_mock_repositories()

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Should not appear')

        updates = [
            {'context_id': 1, 'metadata': {'key': 'value'}},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        mock_summary.summarize.assert_not_awaited()

        # Verify summary=None passed when no text change
        call_kwargs = repos.context.update_context_entry.call_args.kwargs
        assert call_kwargs.get('summary') is None

    @pytest.mark.asyncio
    async def test_update_batch_atomic_summary_failure(self) -> None:
        """Fail entire atomic batch when summary generation fails."""
        repos = _create_mock_repositories()

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(side_effect=RuntimeError('Provider down'))

        updates = [
            {'context_id': 1, 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
            pytest.raises(ToolError, match='Generation failed'),
        ):
            await update_context_batch(updates=updates, atomic=True)

        # No data should have been modified
        repos.context.update_context_entry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_batch_non_atomic_partial_summary_failure(self) -> None:
        """Report per-entry errors in non-atomic mode when summary fails for some."""
        repos = _create_mock_repositories()

        call_count = 0

        async def selective_summary(_text: str, _source: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError('LLM overloaded')
            return 'Generated summary'

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(side_effect=selective_summary)

        updates = [
            {'context_id': 1, 'text': 'x' * 500},
            {'context_id': 2, 'text': 'y' * 500},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await update_context_batch(updates=updates, atomic=False)

        assert result['succeeded'] == 1
        assert result['failed'] == 1

        failed_results = [r for r in result['results'] if not r['success']]
        assert len(failed_results) == 1
        assert failed_results[0]['error'] is not None
        assert 'Generation failed' in failed_results[0]['error']


@pytest.mark.usefixtures('mock_server_dependencies')
class TestBatchSummaryEdgeCases:
    """Tests for edge cases in batch summary operations."""

    @pytest.mark.asyncio
    async def test_store_batch_both_embedding_and_summary(self) -> None:
        """Generate both embeddings and summaries when both providers configured."""
        repos = _create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(return_value=(300, False))

        mock_embedding = MagicMock()
        mock_embedding.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Combined summary')

        entries = [
            {'thread_id': 'both-gen', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools.context.get_embedding_provider', return_value=mock_embedding),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
            patch('app.startup.get_chunking_service', return_value=None),
            patch('app.tools.context.get_chunking_service', return_value=None),
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
        repos = _create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(return_value=(400, False))

        mock_embedding = MagicMock()
        mock_embedding.embed_query = AsyncMock(side_effect=RuntimeError('Embedding failed'))

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Summary text')

        entries = [
            {'thread_id': 'emb-fail', 'source': 'user', 'text': 'x' * 500},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools.context.get_embedding_provider', return_value=mock_embedding),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
            patch('app.startup.get_chunking_service', return_value=None),
            patch('app.tools.context.get_chunking_service', return_value=None),
        ):
            result = await store_context_batch(entries=entries, atomic=False)

        assert result['failed'] == 1
        # With parallel gather, summary IS called even when embedding fails
        mock_summary.summarize.assert_awaited_once()


@pytest.mark.usefixtures('mock_server_dependencies')
class TestBatchMessageAccuracy:
    """Tests that batch message reflects actual generation, not provider availability."""

    @pytest.mark.asyncio
    async def test_store_batch_short_text_no_summary_message(self) -> None:
        """Message omits 'summaries generated' when all entries skip summary due to min_content_length."""
        repos = _create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(
            side_effect=[(101, False), (102, False)],
        )

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Should not be called')

        entries = [
            {'thread_id': 'batch-short', 'source': 'user', 'text': 'Short text'},
            {'thread_id': 'batch-short', 'source': 'agent', 'text': 'Also short'},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        assert 'summaries generated' not in result['message']
        mock_summary.summarize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_batch_short_text_no_summary_message(self) -> None:
        """Message omits 'summaries regenerated' when all entries skip summary due to min_content_length."""
        repos = _create_mock_repositories()
        repos.context.check_entry_exists = AsyncMock(return_value=(True, 'agent'))
        repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content']))

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Should not be called')

        updates = [
            {'context_id': 1, 'text': 'Short text'},
            {'context_id': 2, 'text': 'Also short'},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        assert 'summaries regenerated' not in result['message']
        assert 'summaries generated' not in result['message']
        mock_summary.summarize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_batch_no_text_change_no_regeneration_message(self) -> None:
        """Message omits generation info when only metadata is updated (no text changes)."""
        repos = _create_mock_repositories()
        repos.context.check_entry_exists = AsyncMock(return_value=(True, 'agent'))
        repos.context.update_context_entry = AsyncMock(return_value=(True, ['metadata']))

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Should not be called')

        mock_embedding = MagicMock()
        mock_embedding.embed_query = AsyncMock(return_value=[0.1, 0.2])

        updates = [
            {'context_id': 1, 'metadata': {'key': 'val1'}},
            {'context_id': 2, 'metadata': {'key': 'val2'}},
        ]

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=mock_embedding),
            patch('app.tools.context.get_embedding_provider', return_value=mock_embedding),
            patch('app.startup.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        assert 'embeddings regenerated' not in result['message']
        assert 'summaries regenerated' not in result['message']
        mock_summary.summarize.assert_not_awaited()
        mock_embedding.embed_query.assert_not_awaited()
