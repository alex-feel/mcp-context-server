"""SQLite write-queue processor lifecycle tests.

Every WriteRequest removed from the queue must either run or have its future
resolved: once dequeued, a request is invisible to the shutdown drain (which
only sees the queue), so abandoning it leaves the ``execute_write`` caller
awaiting a future nobody will ever resolve -- an MCP call that never returns
while the write silently never happens.

The dangerous interleaving is a request landing in the window right after the
processor's ``asyncio.wait`` has partitioned its waiters: the getter task
completes with a real request while the loop still believes the iteration was
an idle timeout.
"""

import asyncio
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from typing import cast

import pytest

from app.backends.sqlite_backend import SQLiteBackend
from app.backends.sqlite_backend import WriteRequest


async def _initialized_backend(db_path: Path) -> SQLiteBackend:
    """Create the base schema and return an initialized SQLite backend.

    Args:
        db_path: Location of the temporary database file.

    Returns:
        An initialized backend whose background tasks are running.
    """
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(schema_sql)
    backend = SQLiteBackend(db_path=str(db_path))
    await backend.initialize()
    return backend


def _coro_name(waiter: object) -> str:
    """Return the qualified name of the coroutine a waiter task is running.

    Args:
        waiter: A task (or any awaitable the backend passes to ``asyncio.wait``).

    Returns:
        The coroutine qualified name, or an empty string when unavailable.
    """
    get_coro = getattr(waiter, 'get_coro', None)
    if get_coro is None:
        return ''
    return str(getattr(get_coro(), '__qualname__', ''))


class TestWriteQueueHandoffWindow:
    """A write arriving in the processor's idle-timeout window is still serviced."""

    @pytest.mark.asyncio
    async def test_write_enqueued_after_wait_returns_is_executed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The dequeued request runs and resolves its caller's future."""
        backend = await _initialized_backend(tmp_path / 'write_queue_window.db')
        queue = backend._write_queue
        assert queue is not None

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        executed: list[str] = []

        def _operation(conn: sqlite3.Connection) -> str:
            conn.execute('SELECT 1')
            executed.append('ran')
            return 'written'

        request = WriteRequest(_operation, (), {}, future)

        real_wait = asyncio.wait
        injected = {'done': False}

        async def _wait_enqueuing_after_timeout(
            aws: Iterable[Any],
            **kwargs: Any,
        ) -> tuple[set[Any], Iterable[Any]]:
            done, pending = await real_wait(aws, **kwargs)
            waiters = list(aws)
            is_queue_wait = any(_coro_name(w) == 'Queue.get' for w in waiters)
            if not is_queue_wait or done or injected['done']:
                return done, pending
            injected['done'] = True
            # Land the write AFTER the wait has partitioned its waiters: the
            # getter resolves while the loop still treats the iteration as an
            # idle timeout.
            loop.call_soon(queue.put_nowait, request)
            # Hand back the still-pending waiters in a fixed order (the shutdown
            # waiter first) so the scenario is reproducible instead of depending
            # on set iteration order.
            ordered = sorted(pending, key=lambda w: _coro_name(w) != 'Event.wait')
            return done, ordered

        monkeypatch.setattr(asyncio, 'wait', _wait_enqueuing_after_timeout)
        try:
            result = await asyncio.wait_for(future, timeout=10)
            assert result == 'written'
            assert executed == ['ran']
        finally:
            monkeypatch.setattr(asyncio, 'wait', real_wait)
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_ordinary_write_still_completes(self, tmp_path: Path) -> None:
        """The unpatched path keeps servicing writes end to end."""
        backend = await _initialized_backend(tmp_path / 'write_queue_plain.db')
        try:

            def _operation(conn: sqlite3.Connection) -> int:
                cursor = conn.execute('SELECT 42')
                row = cursor.fetchone()
                return int(row[0])

            assert await backend.execute_write(_operation) == 42
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_idle_timeouts_reuse_one_queue_getter(self, tmp_path: Path) -> None:
        """Idle iterations do not churn a new ``Queue.get`` waiter each time.

        Recreating the getter every iteration is what allowed a completed getter
        to be cancelled-and-discarded; a getter that outlives the idle iteration
        cannot orphan a request by construction. The queue therefore holds at
        most one waiter no matter how many idle timeouts elapse.
        """
        backend = await _initialized_backend(tmp_path / 'write_queue_getters.db')
        queue = backend._write_queue
        assert queue is not None
        try:
            from app.settings import get_settings

            # Span several idle timeouts.
            await asyncio.sleep(get_settings().storage.queue_timeout_test_s * 3.5)
            pending_getters = cast(Any, queue)._getters
            assert len(pending_getters) <= 1
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_cancellation_resolves_a_request_the_getter_holds(
        self,
        tmp_path: Path,
    ) -> None:
        """A request the getter dequeued as the processor stops still resolves.

        Cancelling a getter that has ALREADY produced a request is a no-op, so
        the request would vanish with its future unresolved and its caller stuck
        forever. The loop's terminal backstop resolves it instead, exactly as
        shutdown's drain does for requests still sitting in the queue.
        """
        backend = await _initialized_backend(tmp_path / 'write_queue_cancel.db')
        queue = backend._write_queue
        assert queue is not None
        processor = backend._write_processor_task
        assert processor is not None

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        def _operation(_conn: sqlite3.Connection) -> str:
            return 'written'

        try:
            # Let the processor park in its wait, then hand the getter a request
            # and stop the processor in the SAME scheduling step: the getter
            # completes with the request while the loop is already unwinding.
            await asyncio.sleep(0.05)
            queue.put_nowait(WriteRequest(_operation, (), {}, future))
            processor.cancel()
            await asyncio.gather(processor, return_exceptions=True)

            assert future.done()
            assert future.cancelled()
        finally:
            await backend.shutdown()
