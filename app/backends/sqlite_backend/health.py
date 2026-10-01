"""Periodic health checks of the SQLite backend.

Writer liveness, idle-writer recycling, and the circuit-breaker recovery transition applied
on the health-check cadence.
"""

import asyncio
import logging
import time
from contextlib import suppress

from app.backends._executor import run_in_executor_uninterruptible
from app.backends.sqlite_backend.connections import SQLiteConnectionsMixin

logger = logging.getLogger(__name__)


class SQLiteHealthMixin(SQLiteConnectionsMixin):
    """Check the writer on an interval and recycle it once it sits idle."""

    async def _health_check_loop(self) -> None:
        """Periodic health check for connections."""
        logger.info('Health check loop started')

        assert self._shutdown_event is not None, 'Backend not initialized, call initialize() first'

        # One long-lived waiter kept across iterations, like the write-queue
        # processor: cancelling and recreating it every interval churns tasks and
        # discards the awaited result of a waiter that completed during the
        # cancellation window, so the shutdown signal is only noticed one
        # iteration later.
        shutdown_task: asyncio.Task[bool] | None = None

        try:
            while not self._shutdown:
                try:
                    # Use wait with timeout for interruptible sleep
                    if shutdown_task is None:
                        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
                    done, _pending = await asyncio.wait(
                        [shutdown_task],
                        timeout=self.pool_config.health_check_interval,
                    )

                    if shutdown_task in done:
                        break

                    await self._perform_health_check()
                except asyncio.CancelledError:
                    logger.info('Health check loop cancelled')
                    break
                except Exception as e:
                    logger.error(f'Health check error: {e}')
        finally:
            # Clean up any remaining tasks
            try:
                if shutdown_task and not shutdown_task.done():
                    shutdown_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await shutdown_task
            except RuntimeError:
                # Event loop may be closed, ignore
                pass

    async def _perform_health_check(self) -> None:
        """Perform health check on all connections."""
        loop = asyncio.get_running_loop()

        def _check() -> None:
            with self._pool_lock:
                # Writer
                if self._writer_conn:
                    try:
                        self._writer_conn.execute('SELECT 1')
                    except Exception:
                        logger.warning('Writer connection unhealthy, closing')
                        self._safe_close_connection(self._writer_conn)
                        self._writer_conn = None
                        self.metrics.failed_connections += 1

        await loop.run_in_executor(None, _check)

        await self._recycle_idle_writer_connection()

        # get_state() applies the FAILED -> DEGRADED recovery transition on the health-check cadence.
        self.circuit_breaker.get_state()

    async def _recycle_idle_writer_connection(self) -> None:
        """Close the writer connection once it has been idle past POOL_IDLE_TIMEOUT_S.

        The writer is created at initialize() and otherwise lives for the whole
        process, pinning the database file plus its -wal/-shm siblings even when
        no write has happened for hours; POOL_IDLE_TIMEOUT_S exists to bound that.
        Readers need no equivalent: they are per-use temporary connections that
        get_connection closes in its finally.

        Closing is safe because every writer user acquires the writer through
        ``_ensure_writer_connection`` while holding ``_writer_lock``, which this
        method also takes -- so no caller can be holding the connection object
        being closed -- and the next write recreates it lazily. The lock is taken
        only when it is already free (asyncio.Lock.acquire on a free lock does not
        yield, so the check and the acquisition are atomic on the event loop):
        waiting for it would mean a write is in flight, i.e. the writer is not
        idle at all.
        """
        assert self._writer_lock is not None, 'Backend not initialized, call initialize() first'

        idle_timeout = self.pool_config.idle_timeout
        if self._shutdown or self._writer_conn is None or self._writer_lock.locked():
            return
        if (time.monotonic() - self._writer_last_used) < idle_timeout:
            return

        loop = asyncio.get_running_loop()

        def _close_idle_writer() -> bool:
            with self._pool_lock:
                writer = self._writer_conn
                if writer is None:
                    return False
                # Closed while it is still the tracked writer so the helper's
                # PRAGMA optimize runs before the handle goes away.
                self._safe_close_connection(writer)
                self._writer_conn = None
                return True

        async with self._writer_lock:
            # Re-check under the lock: a write may have landed between the
            # unlocked pre-check above and the acquisition.
            if (time.monotonic() - self._writer_last_used) < idle_timeout:
                return
            if await run_in_executor_uninterruptible(loop, _close_idle_writer):
                logger.debug(f'Closed writer connection idle for more than {idle_timeout}s')
