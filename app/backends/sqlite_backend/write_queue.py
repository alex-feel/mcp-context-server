"""Queued writes of the SQLite backend.

Every ``execute_write`` call is queued and run by one background task on the shared writer,
which retries the self-clearing lock-contention family with exponential backoff.
"""

import asyncio
import logging
import random
import sqlite3
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from typing import TypeVar
from typing import cast
from typing import overload

from app.backends._executor import run_in_executor_uninterruptible
from app.backends.sqlite_backend.config import is_test_environment
from app.backends.sqlite_backend.connections import SQLiteConnectionsMixin
from app.backends.sqlite_backend.contention import is_sqlite_locked_error
from app.errors import ControlFlowError
from app.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# Type definitions
T = TypeVar('T')


@dataclass
class WriteRequest:
    """Encapsulates a write request for the queue."""

    operation: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    future: asyncio.Future[Any]


class SQLiteWriteQueueMixin(SQLiteConnectionsMixin):
    """Serialize writes through a queue processed by one background task."""

    async def _process_write_queue(self) -> None:
        """Background task to process write requests from the queue."""
        logger.info('Write queue processor started')

        assert self._write_queue is not None, 'Backend not initialized, call initialize() first'
        assert self._shutdown_event is not None, 'Backend not initialized, call initialize() first'

        # Use shorter timeout in test environment
        queue_timeout = settings.storage.queue_timeout_test_s if is_test_environment() else settings.storage.queue_timeout_s
        # Long-lived waiters, created once and kept ACROSS loop iterations. The
        # getter is recreated ONLY after it has actually yielded a request.
        # Cancelling and recreating the Queue.get() task every iteration would
        # lose writes: cancelling an already-completed getter is a no-op, so an
        # iteration that times out while a request lands in the same scheduling
        # window resolves the getter, sees a stale done/pending partition, and
        # drops the dequeued WriteRequest on the floor -- its execute_write
        # caller then awaits a future nobody ever resolves, and the entry is
        # never written. A getter that outlives the iteration cannot orphan a
        # request by construction: whatever it dequeues is still owned by the
        # task the next iteration inspects.
        wait_task: asyncio.Task[WriteRequest] | None = None
        shutdown_task: asyncio.Task[bool] | None = None
        # The request currently being serviced. Tracked at function scope so
        # the cancellation handler and the finally block can terminally
        # resolve its future: once dequeued, a request is invisible to the
        # shutdown drain (which only sees the queue), and an unresolved
        # future leaves its execute_write caller awaiting forever.
        current_request: WriteRequest | None = None

        try:
            while not self._shutdown:
                try:
                    # Wait for write request with timeout or shutdown
                    if wait_task is None:
                        wait_task = asyncio.create_task(self._write_queue.get())
                    if shutdown_task is None:
                        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
                    done, _pending = await asyncio.wait(
                        [wait_task, shutdown_task],
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=queue_timeout,
                    )

                    if wait_task not in done:
                        # Nothing was dequeued. Both waiters stay alive for the
                        # next iteration, so a request arriving in any suspension
                        # window remains owned by the live getter.
                        if shutdown_task in done:
                            break
                        # Idle timeout: no writes pending.
                        continue

                    # A WriteRequest was dequeued. Even when the shutdown signal
                    # fired in the same asyncio.wait batch it MUST be serviced
                    # first: once removed from the queue it is invisible to
                    # shutdown's drain, so abandoning it leaves the caller
                    # awaiting execute_write past shutdown. The loop condition
                    # honors the shutdown signal on the next iteration.
                    dequeued, wait_task = wait_task, None
                    request = dequeued.result()
                    current_request = request
                    self.metrics.write_queue_size = self._write_queue.qsize()

                    # Check circuit breaker. The done() guard matches every
                    # other resolution site in this block: the caller's task
                    # may have been cancelled while the request sat queued
                    # (client disconnect), and set_exception on a done future
                    # raises InvalidStateError -- which would skip the
                    # current_request reset and log a spurious processor error.
                    if self.circuit_breaker.is_open():
                        if not request.future.done():
                            request.future.set_exception(
                                Exception('Database circuit breaker is open, too many failures'),
                            )
                        current_request = None
                        continue

                    # Process write request with retry logic
                    try:
                        result = await self._execute_write_with_retry(request)
                        if not request.future.done():
                            request.future.set_result(result)
                        self.circuit_breaker.record_success()
                    except ControlFlowError as e:
                        # Control-flow signals (optimistic-concurrency
                        # VersionConflictError, post-dedup
                        # EmbeddingsReconcileRequiredError) escaping execute_write are
                        # normal write contention, NOT a database fault: propagate to
                        # the caller WITHOUT recording a breaker failure, mirroring
                        # begin_transaction's exemption on this backend and the
                        # PostgreSQL execute_write path. Recording them could open the
                        # breaker on routine contention and lock out writes.
                        if not request.future.done():
                            request.future.set_exception(e)
                    except Exception as e:
                        if not request.future.done():
                            request.future.set_exception(e)
                        self._record_charged_failure(e)
                    finally:
                        # Terminal guarantee: whatever path unwinds this
                        # block — including a cancellation delivered while
                        # the write ran in the executor thread, which
                        # cannot be interrupted and may still commit —
                        # the caller's future must resolve. An unresolved
                        # future leaves execute_write awaiting forever,
                        # invisible to the shutdown drain, which only
                        # cancels still-queued requests.
                        if not request.future.done():
                            request.future.cancel()
                        current_request = None

                except asyncio.CancelledError:
                    logger.info('Write queue processor cancelled')
                    break
                except Exception as e:
                    logger.error(f'Write queue processor error: {e}')
        finally:
            # Terminal backstop: never exit with a dequeued request left
            # unresolved, whatever path unwound the loop.
            if current_request is not None and not current_request.future.done():
                current_request.future.cancel()
            # Clean up any remaining tasks
            try:
                if wait_task is not None:
                    if not wait_task.done():
                        wait_task.cancel()
                    with suppress(asyncio.CancelledError):
                        orphan = await wait_task
                        # The getter had already dequeued a request when the loop
                        # unwound (shutdown or cancellation). Cancelling a
                        # completed task is a no-op, so its request would
                        # otherwise vanish with its future unresolved; resolve it
                        # terminally, exactly as shutdown's drain does for
                        # requests still sitting in the queue.
                        if not orphan.future.done():
                            orphan.future.cancel()
                if shutdown_task is not None and not shutdown_task.done():
                    shutdown_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await shutdown_task
            except RuntimeError:
                # Event loop may be closed, ignore
                pass

        logger.info('Write queue processor stopped')

    async def _execute_write_with_retry(self, request: WriteRequest) -> object:
        """Execute a write request with retry logic."""
        assert self._writer_lock is not None, 'Backend not initialized, call initialize() first'

        loop = asyncio.get_running_loop()
        last_error = None

        for attempt in range(self.retry_config.max_retries):
            try:
                # Acquire writer lock to ensure mutual exclusion with begin_transaction
                async with self._writer_lock:
                    writer = await self._ensure_writer_connection()

                    # Execute operation
                    def _execute(conn: sqlite3.Connection) -> object:
                        # Cast to sync callable since SQLiteBackend only uses sync operations
                        sync_operation = cast(Callable[..., object], request.operation)
                        try:
                            result = sync_operation(conn, *request.args, **request.kwargs)
                            conn.commit()
                        except BaseException:
                            # The writer connection is shared and persistent (DEFERRED
                            # isolation), so a failure after a partial multi-statement
                            # write would otherwise leave those rows in an open
                            # transaction that the NEXT execute_write commits. Roll back
                            # so a failed write leaves no state behind, mirroring the
                            # rollback contract begin_transaction already upholds.
                            with suppress(Exception):
                                conn.rollback()
                            raise
                        self.metrics.total_queries += 1
                        return result

                    # Drained offload: on shutdown the write-queue processor task
                    # is cancelled after a grace period, but the executor callable
                    # (sync_operation + conn.commit) cannot be interrupted mid-flight
                    # -- draining it to completion keeps _close_all_connections (which
                    # runs later in shutdown) from closing the shared writer while a
                    # statement is still executing on it.
                    return await run_in_executor_uninterruptible(loop, _execute, writer)

            except sqlite3.OperationalError as e:
                last_error = e
                if not is_sqlite_locked_error(e):
                    raise
                if attempt + 1 >= self.retry_config.max_retries:
                    # Final attempt failed: no retry follows, so a backoff sleep
                    # here would be pure dead latency that also delays the single
                    # exhaustion breaker charge in the caller. Fall straight
                    # through to the exhaustion raise below.
                    break
                # Calculate backoff delay
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.backoff_factor**attempt),
                    self.retry_config.max_delay,
                )

                # Add jitter if configured
                if self.retry_config.jitter:
                    delay += random.uniform(0, delay * 0.3)

                logger.warning(
                    f'Database locked on write, retrying in {delay:.2f}s '
                    f'(attempt {attempt + 1}/{self.retry_config.max_retries})',
                )
                await asyncio.sleep(delay)

        # Max retries exceeded
        raise last_error or Exception('Max retries exceeded for write operation')

    @overload
    async def execute_write(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...

    @overload
    async def execute_write(
        self,
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...

    async def execute_write(
        self,
        operation: Callable[..., T] | Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a write operation through the write queue.

        Args:
            operation: Sync callable to execute with connection as first argument.
                      Signature: operation(conn: sqlite3.Connection, *args, **kwargs) -> T
                      Note: Although protocol accepts sync or async, SQLiteBackend only uses sync.
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            RuntimeError: If connection manager is shutting down

        Note:
            SQLiteBackend expects SYNC callables (not async). The operation is executed
            synchronously in a thread executor to avoid blocking the event loop.
        """
        assert self._write_queue is not None, 'Backend not initialized, call initialize() first'

        if self._shutdown:
            raise RuntimeError('Connection manager is shutting down')

        # Create future for result
        future: asyncio.Future[T] = asyncio.Future()

        # Create and queue request
        request = WriteRequest(operation, args, kwargs, future)
        await self._write_queue.put(request)

        # Update metrics
        self.metrics.write_queue_size = self._write_queue.qsize()

        # Wait for result
        return await future
