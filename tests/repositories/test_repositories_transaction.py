"""
Tests for repository transaction support via txn parameter.

This module tests the transaction context parameter (txn) added to repository
write methods, ensuring:
1. Backward compatibility when txn=None (uses execute_write)
2. Direct connection usage when txn is provided
3. Atomic multi-repository operations within transactions
4. Proper rollback on errors
"""


from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from app.backends import create_backend
from app.repositories import RepositoryContainer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from app.backends import StorageBackend


@pytest_asyncio.fixture
async def backend_with_repos(temp_db_path: Path) -> 'AsyncGenerator[tuple[StorageBackend, RepositoryContainer], None]':
    """Create backend and repository container for transaction tests."""
    # Initialize database schema first
    import sqlite3 as stdlib_sqlite3

    from app.schemas import load_schema

    conn = stdlib_sqlite3.connect(str(temp_db_path))
    try:
        schema_sql = load_schema('sqlite')
        conn.executescript(schema_sql)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.commit()
    finally:
        conn.close()

    # Create backend and initialize
    backend = create_backend(backend_type='sqlite', db_path=str(temp_db_path))
    await backend.initialize()

    repos = RepositoryContainer(backend)

    yield backend, repos

    await backend.shutdown()


class TestContextRepositoryTransaction:
    """Tests for ContextRepository transaction support."""

    @pytest.mark.asyncio
    async def test_store_with_deduplication_without_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test store_with_deduplication works without transaction (backward compat)."""
        backend, repos = backend_with_repos

        context_id, was_updated = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
            metadata=None,
            txn=None,  # Explicit None for backward compatibility
        )

        assert len(context_id) == 32
        assert was_updated is False

        # Verify data was stored
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1
        assert entries[0]['text_content'] == 'Test content'

    @pytest.mark.asyncio
    async def test_store_with_deduplication_with_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test store_with_deduplication works with transaction context."""
        backend, repos = backend_with_repos

        async with backend.begin_transaction() as txn:
            context_id, was_updated = await repos.context.store_with_deduplication(
                thread_id='test-thread',
                source='agent',
                content_type='text',
                text_content='Transaction content',
                metadata=None,
                txn=txn,
            )

            assert len(context_id) == 32
            assert was_updated is False

        # Transaction committed - verify data persisted
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1
        assert entries[0]['text_content'] == 'Transaction content'

    @pytest.mark.asyncio
    async def test_delete_by_ids_with_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test delete_by_ids works with transaction context."""
        backend, repos = backend_with_repos

        # First create an entry
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='To be deleted',
        )

        # Delete within transaction
        async with backend.begin_transaction() as txn:
            deleted_count = await repos.context.delete_by_ids([context_id], txn=txn)
            assert deleted_count == 1

        # Verify deletion persisted
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_update_context_entry_with_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test update_context_entry works with transaction context."""
        backend, repos = backend_with_repos

        # First create an entry
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Original content',
        )

        # Update within transaction
        async with backend.begin_transaction() as txn:
            success, updated_fields = await repos.context.update_context_entry(
                context_id=context_id,
                text_content='Updated content',
                txn=txn,
            )
            assert success is True
            assert 'text_content' in updated_fields

        # Verify update persisted
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1
        assert entries[0]['text_content'] == 'Updated content'


class TestTagRepositoryTransaction:
    """Tests for TagRepository transaction support."""

    @pytest.mark.asyncio
    async def test_store_tags_without_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test store_tags works without transaction (backward compat)."""
        backend, repos = backend_with_repos

        # Create context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
        )

        # Store tags without transaction
        await repos.tags.store_tags(context_id, ['tag1', 'tag2'], txn=None)

        # Verify tags were stored
        tags = await repos.tags.get_tags_for_context(context_id)
        assert set(tags) == {'tag1', 'tag2'}

    @pytest.mark.asyncio
    async def test_store_tags_with_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test store_tags works with transaction context."""
        backend, repos = backend_with_repos

        # Create context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
        )

        # Store tags within transaction
        async with backend.begin_transaction() as txn:
            await repos.tags.store_tags(context_id, ['txn-tag1', 'txn-tag2'], txn=txn)

        # Verify tags persisted after commit
        tags = await repos.tags.get_tags_for_context(context_id)
        assert set(tags) == {'txn-tag1', 'txn-tag2'}

    @pytest.mark.asyncio
    async def test_replace_tags_with_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test replace_tags_for_context works with transaction context."""
        backend, repos = backend_with_repos

        # Create context with initial tags
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
        )
        await repos.tags.store_tags(context_id, ['old-tag1', 'old-tag2'])

        # Replace tags within transaction
        async with backend.begin_transaction() as txn:
            await repos.tags.replace_tags_for_context(context_id, ['new-tag1', 'new-tag2'], txn=txn)

        # Verify replacement persisted
        tags = await repos.tags.get_tags_for_context(context_id)
        assert set(tags) == {'new-tag1', 'new-tag2'}


class TestImageRepositoryTransaction:
    """Tests for ImageRepository transaction support."""

    @pytest.mark.asyncio
    async def test_store_images_without_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
        sample_image_data: dict[str, str],
    ) -> None:
        """Test store_images works without transaction (backward compat)."""
        backend, repos = backend_with_repos

        # Create context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='multimodal',
            text_content='Image content',
        )

        # Store image without transaction
        await repos.images.store_images(context_id, [sample_image_data], txn=None)

        # Verify image was stored
        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 1
        assert images[0].get('mime_type') == 'image/png'

    @pytest.mark.asyncio
    async def test_store_images_with_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
        sample_image_data: dict[str, str],
    ) -> None:
        """Test store_images works with transaction context."""
        backend, repos = backend_with_repos

        # Create context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='multimodal',
            text_content='Image content',
        )

        # Store image within transaction
        async with backend.begin_transaction() as txn:
            await repos.images.store_images(context_id, [sample_image_data], txn=txn)

        # Verify image persisted after commit
        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 1


class TestMultiRepositoryTransaction:
    """Tests for atomic operations across multiple repositories."""

    @pytest.mark.asyncio
    async def test_atomic_context_with_tags_commit(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test atomic commit of context entry with tags."""
        backend, repos = backend_with_repos

        async with backend.begin_transaction() as txn:
            # Store context
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='atomic-test',
                source='agent',
                content_type='text',
                text_content='Atomic content',
                txn=txn,
            )

            # Store tags in same transaction
            await repos.tags.store_tags(context_id, ['atomic', 'test'], txn=txn)

        # Both should be committed
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1

        tags = await repos.tags.get_tags_for_context(context_id)
        assert set(tags) == {'atomic', 'test'}

    @pytest.mark.asyncio
    async def test_atomic_context_with_tags_and_images_commit(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
        sample_image_data: dict[str, str],
    ) -> None:
        """Test atomic commit of context entry with tags and images."""
        backend, repos = backend_with_repos

        async with backend.begin_transaction() as txn:
            # Store context
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='multimodal-atomic',
                source='user',
                content_type='multimodal',
                text_content='Content with image',
                txn=txn,
            )

            # Store tags
            await repos.tags.store_tags(context_id, ['multimodal', 'atomic'], txn=txn)

            # Store image
            await repos.images.store_images(context_id, [sample_image_data], txn=txn)

        # All should be committed
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1

        tags = await repos.tags.get_tags_for_context(context_id)
        assert set(tags) == {'multimodal', 'atomic'}

        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 1

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test that transaction rollback prevents partial writes."""
        backend, repos = backend_with_repos

        # Get initial count
        initial_count = 0

        try:
            async with backend.begin_transaction() as txn:
                # Store context - this should succeed
                context_id, _ = await repos.context.store_with_deduplication(
                    thread_id='rollback-test',
                    source='user',
                    content_type='text',
                    text_content='Should be rolled back',
                    txn=txn,
                )

                # Store tags - this should succeed
                await repos.tags.store_tags(context_id, ['rollback'], txn=txn)

                # Force an error to trigger rollback
                raise ValueError('Simulated error for rollback test')

        except ValueError:
            pass  # Expected error

        # Verify nothing was committed - search for the content
        entries, _ = await repos.context.search_contexts(thread_id='rollback-test')
        assert len(entries) == initial_count  # Should be 0 if this was the only test


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility when txn=None."""

    @pytest.mark.asyncio
    async def test_all_methods_work_without_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
        sample_image_data: dict[str, str],
    ) -> None:
        """Comprehensive test that all modified methods work without txn parameter."""
        backend, repos = backend_with_repos

        # ContextRepository methods
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='compat-test',
            source='user',
            content_type='text',
            text_content='Backward compat test',
        )

        success, fields = await repos.context.update_context_entry(
            context_id=context_id,
            text_content='Updated compat test',
        )
        assert success is True

        # TagRepository methods
        await repos.tags.store_tags(context_id, ['compat'])
        await repos.tags.replace_tags_for_context(context_id, ['replaced'])

        tags = await repos.tags.get_tags_for_context(context_id)
        assert tags == ['replaced']

        # ImageRepository methods (create multimodal entry for images)
        mm_id, _ = await repos.context.store_with_deduplication(
            thread_id='compat-test',
            source='user',
            content_type='multimodal',
            text_content='Multimodal compat test',
        )
        await repos.images.store_images(mm_id, [sample_image_data])
        await repos.images.replace_images_for_context(mm_id, [sample_image_data])

        images = await repos.images.get_images_for_context(mm_id)
        assert len(images) == 1

        # Cleanup
        await repos.context.delete_by_ids([context_id, mm_id])


class TestTxnAwareReadsUseTransactionConnection:
    """Transaction-internal reads run on the txn connection, not a 2nd pool conn.

    Regression: get_content_type / count_images_for_context / EmbeddingRepository.exists
    were called inside the store/update transaction WITHOUT ``txn=txn``, so on
    PostgreSQL they acquired a second pooled connection while the transaction
    connection was held -- a nested-pool-acquire starvation hazard under saturation.
    They now accept ``txn`` and run on the transaction's own connection.
    """

    @pytest.mark.asyncio
    async def test_reads_with_txn_avoid_second_connection(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        import sqlite3 as stdlib_sqlite3
        from unittest.mock import AsyncMock
        from unittest.mock import patch

        backend, repos = backend_with_repos

        # embedding_metadata is created by a migration, not the base schema; create
        # it (empty) so EmbeddingRepository.exists has a table to query.
        def _create_embedding_metadata(conn: stdlib_sqlite3.Connection) -> None:
            conn.execute(
                'CREATE TABLE IF NOT EXISTS embedding_metadata ('
                '  context_id TEXT NOT NULL PRIMARY KEY,'
                '  model_name TEXT NOT NULL,'
                '  dimensions INTEGER NOT NULL,'
                '  chunk_count INTEGER NOT NULL DEFAULT 1,'
                '  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,'
                '  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP'
                ')',
            )

        await backend.execute_write(_create_embedding_metadata)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='conc-1', source='user', content_type='text',
            text_content='txn read target', metadata=None,
        )

        async with backend.begin_transaction() as txn:
            # Any pool-acquiring read fails loudly; the txn-aware reads must use the
            # transaction connection instead, so execute_read is never called.
            guard = AsyncMock(side_effect=AssertionError('acquired a second connection'))
            with patch.object(backend, 'execute_read', new=guard):
                content_type = await repos.context.get_content_type(context_id, txn=txn)
                image_count = await repos.images.count_images_for_context(context_id, txn=txn)
                embedding_exists = await repos.embeddings.exists(context_id, txn=txn)

        assert content_type == 'text'
        assert image_count == 0
        assert embedding_exists is False
        guard.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reads_without_txn_use_pool_path(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        # Backward compatibility: omitting txn keeps the pooled-read path unchanged.
        backend, repos = backend_with_repos
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='conc-1b', source='user', content_type='text',
            text_content='pool read target', metadata=None,
        )
        assert await repos.context.get_content_type(context_id) == 'text'
        assert await repos.images.count_images_for_context(context_id) == 0


class TestTransactionExecutorOffload:
    """The SQLite txn branches run their sync closures OFF the event loop.

    A repository closure called directly on the loop blocks it for the full
    C-level sqlite3 call: a cross-process lock holder busy-waits inside SQLite
    for up to the resolved busy timeout, freezing every concurrent request and
    the /health endpoint. The txn branches must offload via the shared
    BaseRepository helper, matching the write-queue path.
    """

    @pytest.mark.asyncio
    async def test_run_sqlite_txn_executes_off_the_event_loop_thread(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """The helper runs the closure on an executor thread, not the loop thread."""
        import sqlite3
        import threading
        from typing import cast

        from app.repositories.base import BaseRepository

        backend, _repos = backend_with_repos
        loop_thread = threading.get_ident()
        seen: dict[str, int] = {}

        def _closure(conn: sqlite3.Connection) -> int:
            seen['thread'] = threading.get_ident()
            return int(conn.execute('SELECT 1').fetchone()[0])

        async with backend.begin_transaction() as txn:
            result = await BaseRepository._run_sqlite_txn(
                _closure, cast(sqlite3.Connection, txn.connection),
            )

        assert result == 1
        assert seen['thread'] != loop_thread

    @pytest.mark.asyncio
    async def test_txn_repository_write_routes_through_the_offload_helper(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """A txn-scoped repository write goes through _run_sqlite_txn."""
        import sqlite3
        from collections.abc import Callable
        from unittest.mock import patch

        from app.repositories.base import BaseRepository

        backend, repos = backend_with_repos
        original = BaseRepository._run_sqlite_txn
        calls: list[str] = []

        async def _spy(
            closure: Callable[[sqlite3.Connection], object],
            conn: sqlite3.Connection,
        ) -> object:
            calls.append(getattr(closure, '__name__', '?'))
            return await original(closure, conn)

        with patch.object(BaseRepository, '_run_sqlite_txn', staticmethod(_spy)):
            async with backend.begin_transaction() as txn:
                context_id, _ = await repos.context.store_with_deduplication(
                    thread_id='offload-1', source='user', content_type='text',
                    text_content='offloaded txn write', metadata=None, txn=txn,
                )
                await repos.tags.store_tags(context_id, ['alpha'], txn=txn)

        assert '_store_sqlite' in calls
        assert '_store_tags_sqlite' in calls
        assert await repos.context.get_content_type(context_id) == 'text'

    @pytest.mark.asyncio
    async def test_begin_transaction_rolls_back_on_cancellation(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """A BaseException unwind (task cancellation) still rolls the txn back.

        With the executor offload the transaction body awaits, so cancellation
        can unwind mid-transaction. An Exception-only handler would skip the
        rollback and leave an open partial transaction on the pooled writer
        connection, silently committed by the NEXT write. The breaker must not
        trip either: cancellation is not a database fault.
        """
        import asyncio

        backend, repos = backend_with_repos

        async def _cancelled_mid_transaction() -> None:
            """Open a transaction, write, then unwind with CancelledError.

            Raises:
                asyncio.CancelledError: Always, after the partial write, to
                    model task cancellation unwinding mid-transaction.
            """
            async with backend.begin_transaction() as txn:
                await repos.context.store_with_deduplication(
                    thread_id='cancel-1', source='user', content_type='text',
                    text_content='must roll back', metadata=None, txn=txn,
                )
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await _cancelled_mid_transaction()

        # The partial write was rolled back...
        results, _stats = await repos.context.search_contexts(thread_id='cancel-1')
        assert results == []
        # ...the breaker stayed closed, and the writer connection is clean:
        # a follow-up write commits normally.
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='cancel-2', source='user', content_type='text',
            text_content='post-cancel write', metadata=None,
        )
        assert await repos.context.get_content_type(context_id) == 'text'

    @pytest.mark.asyncio
    async def test_cancellation_mid_closure_drains_before_rollback(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Cancelling a task mid-closure joins the closure before unwinding.

        Cancellation cannot interrupt a closure already running on the
        executor thread; without the drain the CancelledError reaches
        begin_transaction's rollback while the closure keeps issuing
        statements on the same connection, and anything the zombie writes
        after that rollback silently rides the NEXT commit. The drain must
        hold the unwind open until the closure finishes, so its writes stay
        inside the rolled-back transaction.
        """
        import asyncio
        import sqlite3
        import threading
        from typing import cast

        from app.ids import generate_id
        from app.repositories.base import BaseRepository

        backend, repos = backend_with_repos
        started = threading.Event()
        release = threading.Event()
        zombie_id = generate_id()

        def _slow_write(conn: sqlite3.Connection) -> None:
            started.set()
            release.wait(timeout=10)
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                'VALUES (?, ?, ?, ?, ?)',
                (zombie_id, 'zombie-1', 'user', 'text', 'must never surface'),
            )

        async def _txn_task() -> None:
            async with backend.begin_transaction() as txn:
                await BaseRepository._run_sqlite_txn(
                    _slow_write, cast(sqlite3.Connection, txn.connection),
                )

        task = asyncio.create_task(_txn_task())
        await asyncio.to_thread(started.wait, 10)
        task.cancel()
        # The drain must hold the cancellation open while the closure runs.
        await asyncio.sleep(0.1)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The zombie write landed INSIDE the rolled-back transaction: a
        # follow-up commit must not resurrect it.
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='zombie-2', source='user', content_type='text',
            text_content='post-cancel commit', metadata=None,
        )
        assert await repos.context.get_content_type(context_id) == 'text'
        results, _stats = await repos.context.search_contexts(thread_id='zombie-1')
        assert results == []

    @pytest.mark.asyncio
    async def test_drain_helper_completes_callable_before_cancel_propagates(self) -> None:
        """The shared drain helper holds the unwind until the callable finishes.

        A cancellation landing on the offload cannot interrupt a callable
        already running on the executor thread; the helper must drain it to
        completion (side effect observable) before re-raising, and must not
        resolve the awaiting task early.
        """
        import asyncio
        import threading

        from app.backends._executor import run_in_executor_uninterruptible

        release = threading.Event()
        started = threading.Event()
        ran_to_completion = threading.Event()

        def _slow() -> str:
            started.set()
            release.wait(timeout=10)
            ran_to_completion.set()
            return 'done'

        async def _call() -> str:
            loop = asyncio.get_running_loop()
            return await run_in_executor_uninterruptible(loop, _slow)

        task = asyncio.create_task(_call())
        await asyncio.to_thread(started.wait, 10)
        task.cancel()
        # The drain holds the cancellation open while the callable runs.
        await asyncio.sleep(0.1)
        assert not task.done()
        assert not ran_to_completion.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ran_to_completion.is_set()

    @pytest.mark.asyncio
    async def test_begin_transaction_commit_and_rollback_route_through_drain(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """begin_transaction's own commit and rollback hops use the drain helper.

        A bare offload of writer.commit / writer.rollback would let a
        cancellation leave a zombie commit or skip the rollback on the shared
        writer connection; both boundary hops must route through the same
        drained helper the transaction body uses.
        """
        import asyncio
        from collections.abc import Callable
        from unittest.mock import patch

        from app.backends._executor import run_in_executor_uninterruptible as original

        backend, repos = backend_with_repos
        drained: list[str] = []

        async def _spy(loop: asyncio.AbstractEventLoop, func: Callable[..., object], *args: object) -> object:
            drained.append(getattr(func, '__name__', repr(func)))
            return await original(loop, func, *args)

        async def _failing_txn() -> None:
            """Open a transaction, write, then raise an ordinary Exception.

            Raises:
                RuntimeError: Always, to model a body failure that must roll back.
            """
            async with backend.begin_transaction() as txn:
                await repos.context.store_with_deduplication(
                    thread_id='drain-rollback', source='user', content_type='text',
                    text_content='rollback via drain', metadata=None, txn=txn,
                )
                raise RuntimeError('body failure')

        with patch('app.backends.sqlite_backend.run_in_executor_uninterruptible', _spy):
            # Success path -> commit routes through the drain.
            async with backend.begin_transaction() as txn:
                await repos.context.store_with_deduplication(
                    thread_id='drain-commit', source='user', content_type='text',
                    text_content='commit via drain', metadata=None, txn=txn,
                )
            assert 'commit' in drained

            # Failure path (ordinary Exception) -> rollback routes through the drain.
            drained.clear()
            with pytest.raises(RuntimeError):
                await _failing_txn()
            assert 'rollback' in drained

    @pytest.mark.asyncio
    async def test_reader_connection_not_leaked_on_cancellation(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """A read cancelled during connection creation leaks no connection.

        The worker registers the temporary connection before returning it, so a
        cancellation on the creation await must drain and then close+untrack the
        orphan -- the caller's finally never runs because its conn is unbound.
        """
        import asyncio
        import threading
        from typing import Any
        from typing import cast
        from unittest.mock import patch

        backend, _repos = backend_with_repos
        sqlite_backend = cast(Any, backend)
        real_create = sqlite_backend._create_connection
        gate = threading.Event()
        creating = threading.Event()

        def _blocking_create(*, readonly: bool = False) -> object:
            if readonly:
                creating.set()
                gate.wait(timeout=10)
            return real_create(readonly=readonly)

        before = len(sqlite_backend._temporary_connections)
        with patch.object(sqlite_backend, '_create_connection', _blocking_create):
            async def _read() -> None:
                async with backend.get_connection(readonly=True) as conn:
                    conn.execute('SELECT 1')

            task = asyncio.create_task(_read())
            await asyncio.to_thread(creating.wait, 10)
            task.cancel()
            gate.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        # The orphaned reader was closed and untracked; nothing leaked.
        assert len(sqlite_backend._temporary_connections) == before

    @pytest.mark.asyncio
    async def test_cancelled_read_drains_before_connection_close(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """A read cancelled mid-query drains it before get_connection closes the reader.

        execute_read offloads the query to an executor thread; a BARE offload lets a
        cancellation release get_connection's finally, which closes the temporary reader
        connection while the query is STILL running on it on the worker thread -- a
        use-after-free that stalls the event loop or crashes the interpreter. The drain
        must hold the unwind open until the query finishes on the live connection.
        """
        import asyncio
        import sqlite3
        import threading

        backend, _repos = backend_with_repos
        started = threading.Event()
        release = threading.Event()
        completed: dict[str, object] = {}

        def _slow_read(conn: sqlite3.Connection) -> int:
            started.set()
            release.wait(timeout=10)
            # Runs while the drain holds the cancellation open: the reader connection
            # must still be open here (a bare offload would already have closed it).
            value = int(conn.execute('SELECT 42').fetchone()[0])
            completed['value'] = value
            return value

        task = asyncio.create_task(backend.execute_read(_slow_read))
        await asyncio.to_thread(started.wait, 10)
        task.cancel()
        # The drain holds the cancellation open while the query runs.
        await asyncio.sleep(0.1)
        assert not task.done()
        assert 'value' not in completed
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The query completed on a live connection before the reader was closed.
        assert completed['value'] == 42
