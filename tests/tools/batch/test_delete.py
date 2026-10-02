"""Tests for delete_context_batch embedding cleanup and snapshot-constrained deletion by thread and age criteria."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.settings import get_settings


class _FakeDeleteTransaction:
    """Stand-in for TransactionContext exposing only what the delete path reads."""

    def __init__(self, backend_type: str) -> None:
        self.connection = object()
        self.backend_type = backend_type


class _FakeTransactionalBackend:
    """Backend stub whose begin_transaction yields a recording fake transaction."""

    def __init__(self, backend_type: str) -> None:
        self.backend_type = backend_type
        self.transactions: list[_FakeDeleteTransaction] = []

    @asynccontextmanager
    async def begin_transaction(self):
        """Yield a fake transaction and record it for assertions.

        Yields:
            The fake transaction context handed to the tool body.
        """
        txn = _FakeDeleteTransaction(self.backend_type)
        self.transactions.append(txn)
        yield txn


@pytest.fixture
def fp32_cleanup_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the fp32 layout, where the explicit per-entry cleanup is required.

    With compression enabled (the default) the compressed payload table cascades
    with the context row, so the explicit loop is correctly skipped; a test that
    asserts the loop RUNS therefore needs the fp32 layout.
    """
    import app.tools._delete_cleanup as delete_cleanup_module

    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    monkeypatch.setattr(delete_cleanup_module, 'settings', get_settings())


@pytest.mark.usefixtures('initialized_server')
class TestBatchDeleteEmbeddingCleanup:
    """Embedding cleanup and snapshot-constrained deletion by thread and age criteria."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('fp32_cleanup_mode')
    async def test_delete_batch_by_thread_cleans_embeddings_sqlite(self):
        """Verify the fp32 embedding cleanup runs when deleting by thread_ids on SQLite."""
        from app.tools.batch.delete import delete_context_batch

        with patch('app.tools.batch.delete.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = _FakeTransactionalBackend('sqlite')
            mock_repos.context.backend = mock_backend

            # The snapshot returns the ids the combined criteria match.
            snapshot = ['id-10', 'id-20', 'id-30']
            mock_repos.context.get_ids_matching_batch_criteria = AsyncMock(
                return_value=list(snapshot),
            )
            mock_repos.embeddings.embedding_tables_exist = AsyncMock(return_value=True)
            mock_repos.embeddings.delete_all_chunks_bulk = AsyncMock(return_value=0)
            mock_repos.context.delete_by_ids = AsyncMock(return_value=3)
            mock_repos.context.delete_contexts_batch = AsyncMock()

            result = await delete_context_batch(thread_ids=['thread-abc'])

            assert result['success'] is True
            assert result['deleted_count'] == 3
            assert result['criteria_used'] == ['thread_ids: 1 threads']

            # The cleanup covered the whole snapshot in ONE bulk call on the shared
            # transaction: a per-entry loop would cost one write round trip per matched
            # row and hold the single SQLite writer for the entire scan.
            txn = mock_backend.transactions[0]
            mock_repos.embeddings.delete_all_chunks_bulk.assert_awaited_once_with(snapshot, txn=txn)

            # Verify get_ids_matching_batch_criteria was called with correct args.
            mock_repos.context.get_ids_matching_batch_criteria.assert_called_once_with(
                context_ids=None,
                thread_ids=['thread-abc'],
                source=None,
                older_than_days=None,
            )

            # The destructive step deletes EXACTLY the snapshot ids, on the SAME
            # transaction; it never re-runs the criteria, which would sweep rows
            # committed after the cleanup snapshot and orphan their vec0 embeddings.
            mock_repos.context.delete_by_ids.assert_awaited_once_with(snapshot, txn=txn)
            mock_repos.context.delete_contexts_batch.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('fp32_cleanup_mode')
    async def test_delete_batch_by_older_than_cleans_embeddings_sqlite(self):
        """Verify the fp32 cleanup runs when deleting by older_than_days on SQLite."""
        from app.tools.batch.delete import delete_context_batch

        with patch('app.tools.batch.delete.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = _FakeTransactionalBackend('sqlite')
            mock_repos.context.backend = mock_backend

            snapshot = ['id-5', 'id-15']
            mock_repos.context.get_ids_matching_batch_criteria = AsyncMock(
                return_value=list(snapshot),
            )
            mock_repos.embeddings.embedding_tables_exist = AsyncMock(return_value=True)
            mock_repos.embeddings.delete_all_chunks_bulk = AsyncMock(return_value=0)
            mock_repos.context.delete_by_ids = AsyncMock(return_value=2)
            mock_repos.context.delete_contexts_batch = AsyncMock()

            # older_than_days alone is refused (it would reach the whole database), so
            # the age criterion is combined with a source filter here.
            result = await delete_context_batch(older_than_days=30, source='agent')

            assert result['success'] is True
            assert result['deleted_count'] == 2
            assert result['criteria_used'] == ['source: agent', 'older_than_days: 30']

            # The cleanup covered the whole snapshot in ONE bulk call.
            mock_repos.embeddings.delete_all_chunks_bulk.assert_awaited_once()
            assert mock_repos.embeddings.delete_all_chunks_bulk.await_args.args[0] == snapshot

            # The criteria (including the age boundary) are evaluated exactly ONCE,
            # in the snapshot SELECT; the destructive step deletes exactly the
            # snapshot ids, so no shared absolute cutoff between two predicate
            # evaluations is needed and a row crossing the age boundary between
            # the two statements cannot be deleted without its vec0 cleanup.
            mock_repos.context.get_ids_matching_batch_criteria.assert_called_once_with(
                context_ids=None,
                thread_ids=None,
                source='agent',
                older_than_days=30,
            )
            txn = mock_backend.transactions[0]
            mock_repos.context.delete_by_ids.assert_awaited_once_with(snapshot, txn=txn)
            mock_repos.context.delete_contexts_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_batch_postgresql_uses_atomic_criteria_delete(self):
        """On PostgreSQL the criteria delete stays a single atomic statement.

        Embedding rows cascade-delete with the context rows inside the SAME
        DELETE statement, so no snapshot or explicit cleanup is needed and the
        tool must route to delete_contexts_batch, not the SQLite snapshot flow.
        """
        from app.tools.batch.delete import delete_context_batch

        with patch('app.tools.batch.delete.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'postgresql'
            mock_repos.context.backend = mock_backend

            mock_repos.context.get_ids_matching_batch_criteria = AsyncMock()
            mock_repos.embeddings.embedding_tables_exist = AsyncMock()
            mock_repos.context.delete_by_ids = AsyncMock()
            mock_repos.context.delete_contexts_batch = AsyncMock(
                return_value=(2, ['thread_ids: 1 threads']),
            )

            result = await delete_context_batch(thread_ids=['thread-abc'])

            assert result['success'] is True
            assert result['deleted_count'] == 2
            assert result['criteria_used'] == ['thread_ids: 1 threads']

            mock_repos.context.delete_contexts_batch.assert_awaited_once_with(
                context_ids=None,
                thread_ids=['thread-abc'],
                source=None,
                older_than_days=None,
            )
            mock_repos.context.get_ids_matching_batch_criteria.assert_not_called()
            mock_repos.embeddings.delete_all_chunks_bulk.assert_not_called()
            mock_repos.context.delete_by_ids.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_batch_entry_inserted_after_snapshot_survives(self):
        """A row committed between the cleanup snapshot and the delete must survive.

        The destructive step deletes exactly the snapshotted ids instead of
        re-running the deletion criteria in a second transaction, so a store
        landing in the window between the embedding-cleanup snapshot and the
        delete is never swept without its vec0 cleanup (which would
        orphan the FK-less vec0 vectors permanently). The interleaved entry is
        neither cleaned nor deleted: it simply survives the operation.
        """
        from app.startup import ensure_repositories
        from app.tools.batch.delete import delete_context_batch
        from app.tools.context.retrieve import get_context_by_ids
        from app.tools.context.store import store_context

        repos = await ensure_repositories()
        thread = 'batch-del-snapshot-window'

        first = await store_context(thread_id=thread, source='user', text='Entry one')
        second = await store_context(thread_id=thread, source='user', text='Entry two')
        snapshot_ids = {first['context_id'], second['context_id']}

        interleaved: dict[str, str] = {}
        mock_embedding_delete = AsyncMock(return_value=0)

        async def snapshot_then_interleave(**_kwargs: object) -> list[str]:
            # The interleaving store must commit AFTER the snapshot SELECT and
            # BEFORE the cleanup + delete transaction opens -- exactly the window
            # a concurrent writer lands in. It cannot run once that transaction is
            # open: SQLite serializes writes onto one writer connection, so a
            # same-process store would wait on a lock the caller itself holds.
            ids = sorted(snapshot_ids)
            if 'id' not in interleaved:
                stored = await store_context(
                    thread_id=thread, source='user', text='Entry three (interleaved)',
                )
                interleaved['id'] = stored['context_id']
            return ids

        with (
            patch.object(repos.embeddings, 'delete_all_chunks_bulk', mock_embedding_delete),
            patch.object(
                repos.embeddings, 'embedding_tables_exist', AsyncMock(return_value=True),
            ),
            patch.object(
                repos.context,
                'get_ids_matching_batch_criteria',
                AsyncMock(side_effect=snapshot_then_interleave),
            ),
        ):
            result = await delete_context_batch(thread_ids=[thread])

        assert result['success'] is True
        # Only the two snapshotted entries are deleted.
        assert result['deleted_count'] == 2

        # The cleaned set and the deleted set are identical: exactly the snapshot.
        cleaned = {cid for call in mock_embedding_delete.await_args_list for cid in call.args[0]}
        assert cleaned == snapshot_ids

        # The snapshotted entries are gone; the interleaved entry survives.
        assert await get_context_by_ids(context_ids=sorted(snapshot_ids)) == []
        remaining = await get_context_by_ids(context_ids=[interleaved['id']])
        assert len(remaining) == 1
        assert remaining[0].get('text_content') == 'Entry three (interleaved)'

    @pytest.mark.asyncio
    async def test_delete_batch_thread_larger_than_parameter_limit(self):
        """A criteria match beyond SQLite's bound-variable ceiling deletes fully.

        The snapshot-constrained delete binds the snapshot ids into
        DELETE ... WHERE id IN (...) statements; delete_by_ids issues them in
        bounded chunks, so a match set larger than SQLITE_MAX_VARIABLE_NUMBER
        (historically as low as 999) cannot overflow a single statement's
        parameter list.
        """
        import sqlite3

        from app.ids import generate_id
        from app.startup import ensure_repositories
        from app.tools.batch.delete import delete_context_batch

        repos = await ensure_repositories()
        thread = 'batch-del-chunk-thread'
        total = 1005
        ids = [generate_id() for _ in range(total)]

        def _bulk_insert(conn: sqlite3.Connection) -> None:
            conn.executemany(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES (?, ?, ?, ?, ?, 'local')",
                [(cid, thread, 'user', 'text', f'chunk entry {i}') for i, cid in enumerate(ids)],
            )

        await repos.context.backend.execute_write(_bulk_insert)

        # Focus on the chunked delete: skip the per-id embedding cleanup loop.
        with patch.object(
            repos.embeddings, 'embedding_tables_exist', AsyncMock(return_value=False),
        ):
            result = await delete_context_batch(thread_ids=[thread])

        assert result['success'] is True
        assert result['deleted_count'] == total
        assert await repos.context.get_by_ids(ids[:5] + ids[-5:]) == []
