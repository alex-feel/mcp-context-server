"""Tests that SQLite transaction and read paths run blocking sqlite3 calls on executor threads.

A running call drains before a cancellation unwinds.
"""

from typing import TYPE_CHECKING

import pytest

from tests.helpers import LOCAL_SCOPE

if TYPE_CHECKING:
    from app.backends import StorageBackend
    from app.repositories import RepositoryContainer


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
                    scope=LOCAL_SCOPE,
                    visibility='private',
                    thread_id='offload-1', source='user', content_type='text',
                    text_content='offloaded txn write', metadata=None, txn=txn,
                )
                await repos.tags.store_tags(context_id, ['alpha'], txn=txn)

        assert '_store_sqlite' in calls
        assert '_store_tags_sqlite' in calls
        assert await repos.context.get_content_type(context_id, scope=LOCAL_SCOPE) == 'text'

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
                    scope=LOCAL_SCOPE,
                    visibility='private',
                    thread_id='cancel-1', source='user', content_type='text',
                    text_content='must roll back', metadata=None, txn=txn,
                )
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await _cancelled_mid_transaction()

        # The partial write was rolled back...
        results, _stats = await repos.context.search_contexts(thread_id='cancel-1', scope=LOCAL_SCOPE)
        assert results == []
        # ...the breaker stayed closed, and the writer connection is clean:
        # a follow-up write commits normally.
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='cancel-2', source='user', content_type='text',
            text_content='post-cancel write', metadata=None,
        )
        assert await repos.context.get_content_type(context_id, scope=LOCAL_SCOPE) == 'text'

    @pytest.mark.asyncio
    async def test_cancellation_mid_closure_drains_before_rollback(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Canceling a task mid-closure joins the closure before unwinding.

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
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='zombie-2', source='user', content_type='text',
            text_content='post-cancel commit', metadata=None,
        )
        assert await repos.context.get_content_type(context_id, scope=LOCAL_SCOPE) == 'text'
        results, _stats = await repos.context.search_contexts(thread_id='zombie-1', scope=LOCAL_SCOPE)
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
                    scope=LOCAL_SCOPE,
                    visibility='private',
                    thread_id='drain-rollback', source='user', content_type='text',
                    text_content='rollback via drain', metadata=None, txn=txn,
                )
                raise RuntimeError('body failure')

        with patch('app.backends.sqlite_backend.transactions.run_in_executor_uninterruptible', _spy):
            # Success path -> commit routes through the drain.
            async with backend.begin_transaction() as txn:
                await repos.context.store_with_deduplication(
                    scope=LOCAL_SCOPE,
                    visibility='private',
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
        """A read canceled during connection creation leaks no connection.

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
        """A read canceled mid-query drains it before get_connection closes the reader.

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
