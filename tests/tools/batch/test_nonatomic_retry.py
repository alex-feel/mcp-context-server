"""Tests for the connection-error retry of non-atomic store_context_batch and update_context_batch."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.repositories.context_repository.records import EntryProbe


@pytest.mark.usefixtures('initialized_server')
class TestNonAtomicBatchRetry:
    """Connection retry in non-atomic batch operations."""

    @pytest.mark.asyncio
    async def test_store_batch_nonatomic_retries_on_connection_error(self):
        """Non-atomic store retries on ConnectionResetError."""
        from app.tools.batch.store import store_context_batch

        call_count = 0
        real_txn = MagicMock()
        real_txn.connection = MagicMock()
        real_txn.backend_type = 'sqlite'

        @asynccontextmanager
        async def failing_begin_transaction():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionResetError('Connection lost')
            yield real_txn

        with patch('app.tools.batch.store.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = failing_begin_transaction
            mock_repos.context.backend = mock_backend

            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.embeddings.exists = AsyncMock(return_value=False)

            result = await store_context_batch(
                entries=[{
                    'thread_id': 'test-retry',
                    'source': 'user',
                    'text': 'Retry test entry',
                }],
                atomic=False,
            )
            assert result['results'][0]['success'] is True
            assert call_count >= 2

    @pytest.mark.asyncio
    async def test_update_batch_nonatomic_retries_on_connection_error(self):
        """Non-atomic update retries on ConnectionResetError."""
        from app.tools.batch.store import store_context_batch
        from app.tools.batch.update import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-retry-update',
                'source': 'user',
                'text': 'Original text',
            }],
        )
        context_id = store_result['results'][0]['context_id']

        call_count = 0
        real_txn = MagicMock()
        real_txn.connection = MagicMock()
        real_txn.backend_type = 'sqlite'

        @asynccontextmanager
        async def failing_begin_transaction():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionResetError('Connection lost')
            yield real_txn

        with patch('app.tools.batch.update.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=EntryProbe(True, 'user', 0, 'local', True))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            mock_repos.context.update_context_entry = AsyncMock(return_value=(True, ['text']))
            mock_repos.images.count_images_for_context = AsyncMock(return_value=0)

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = failing_begin_transaction
            mock_repos.context.backend = mock_backend

            result = await update_context_batch(
                updates=[{
                    'context_id': context_id,
                    'text': 'Updated text',
                }],
                atomic=False,
            )
            assert result['results'][0]['success'] is True
            assert call_count >= 2
