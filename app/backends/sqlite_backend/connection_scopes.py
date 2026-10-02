"""Connection scopes and the read entry point of the SQLite backend.

``get_connection`` yields a per-use reader or the shared writer under circuit-breaker
accounting; ``execute_read`` runs a read callable on a fresh reader and retries the
self-clearing lock-contention family with exponential backoff.
"""

import asyncio
import logging
import random
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any
from typing import TypeVar
from typing import cast
from typing import overload

from app.backends._executor import run_in_executor_uninterruptible
from app.backends.sqlite_backend.connections import SQLiteConnectionsMixin
from app.backends.sqlite_backend.contention import is_sqlite_locked_error
from app.errors import ControlFlowError

logger = logging.getLogger(__name__)


# Type definitions
T = TypeVar('T')


class SQLiteConnectionScopeMixin(SQLiteConnectionsMixin):
    """Yield reader and writer connection scopes with breaker accounting, and run retried reads."""

    @asynccontextmanager
    async def get_connection(
        self,
        readonly: bool = False,
        allow_write: bool = False,
    ) -> AsyncGenerator[sqlite3.Connection, None]:
        """
        Get a database connection from the pool.

        Args:
            readonly: If True, get a reader connection, otherwise get writer
            allow_write: If True, allow direct writer connection, for migrations and schema

        Yields:
            Database connection

        Raises:
            RuntimeError: If connection manager is shutting down or circuit breaker is open
            ControlFlowError: Re-raised from the connection scope without recording a
                circuit-breaker failure (normal control flow, not a database fault)
        """
        assert self._reader_semaphore is not None, 'Backend not initialized, call initialize() first'
        assert self._writer_lock is not None, 'Backend not initialized, call initialize() first'

        if self._shutdown:
            raise RuntimeError('Connection manager is shutting down')

        # Check circuit breaker
        if self.circuit_breaker.is_open():
            raise RuntimeError(
                f'Database circuit breaker is open after {self.circuit_breaker.failures} failures',
            )

        if readonly:
            # Get reader connection. Wrap the creation await so a genuine
            # establishment fault (e.g. SQLITE_CANTOPEN when the database volume
            # detaches) charges the breaker instead of escaping above the use-time
            # recording try below with the breaker still reporting healthy.
            async with self._reader_semaphore:
                conn = await self._acquire_connection_charging_faults(self._get_reader_connection)
                # Check if it's a temporary connection that needs cleanup
                is_temporary = False
                with self._connection_lock:
                    is_temporary = conn in self._temporary_connections

                try:
                    yield conn
                    self.circuit_breaker.record_success()
                except ControlFlowError:
                    # Normal control flow (e.g. a client-input validation error raised
                    # inside a read callable), NOT a database fault: do not record a
                    # breaker failure, or a client repeatedly sending invalid input
                    # opens the breaker and rejects every caller's healthy requests.
                    raise
                except Exception as e:
                    if is_sqlite_locked_error(e):
                        # SQLITE_BUSY / SQLITE_LOCKED write contention surfacing on a
                        # read (typically a concurrent process holding an exclusive
                        # lock during VACUUM / backup / checkpoint truncate on a shared
                        # database file): self-clearing contention, NOT a database
                        # fault -- do not charge the breaker, mirroring the identical
                        # exemption on begin_transaction. Otherwise the process-global
                        # breaker would open on routine contention and reject every
                        # caller's requests, including the writes that are taught to
                        # ride out the same condition.
                        raise
                    # Charged with the failure metrics: this arm is the SINGLE
                    # accounting site for a read fault (the _execute_read_once
                    # wrapper deliberately records nothing), so an operator sees
                    # the failure count AND the message that caused it.
                    self._record_charged_failure(e)
                    raise
                finally:
                    # Clean up temporary connections after use
                    if is_temporary:
                        # Clean up synchronously to avoid race conditions with garbage collection
                        with self._connection_lock:
                            if conn in self._temporary_connections:
                                self._temporary_connections.remove(conn)
                        self._safe_close_connection(conn)

        elif allow_write:
            # Direct write connection with lock protection
            async with self._writer_lock:
                # Wrap the writer-recreation await so a genuine establishment fault
                # (e.g. SQLITE_CANTOPEN when the database volume detaches) charges the
                # breaker with its metrics, instead of escaping above the use-time
                # recording block below with the breaker still reporting healthy.
                # Mirrors begin_transaction, which routes the identical call this way.
                writer = await self._acquire_connection_charging_faults(self._ensure_writer_connection)
                loop = asyncio.get_running_loop()
                try:
                    yield writer
                    # Drained commit (see begin_transaction): a cancellation must
                    # not leave a zombie commit against the shared writer.
                    await run_in_executor_uninterruptible(loop, writer.commit)
                    # Count the committed scope as one completed operation, like
                    # begin_transaction: this arm's failure ladder moves
                    # failed_queries, so without the success-side increment the
                    # two connection_metrics counters would cover different
                    # populations here too.
                    self.metrics.total_queries += 1
                    self.circuit_breaker.record_success()
                except BaseException as e:
                    # Mirror begin_transaction: BaseException (cancellation) must
                    # still roll back the open transaction on the shared writer,
                    # or the NEXT write silently commits its partial state. Only a
                    # genuine fault trips the breaker.
                    #
                    # The rollback is GUARDED exactly as begin_transaction guards
                    # its identical rollback on the same shared writer: a rollback
                    # that itself fails (e.g. the health check closed the writer
                    # this arm already captured, so rollback raises 'Cannot operate
                    # on a closed database') must not replace the caller's real
                    # error, skip the classification ladder below, and leave the
                    # genuine fault uncharged -- and on a cancellation unwind it
                    # must not turn a cancelled task into a failed one.
                    try:
                        if isinstance(e, Exception):
                            await run_in_executor_uninterruptible(loop, writer.rollback)
                        else:
                            # Synchronous rollback on a BaseException unwind
                            # (interpreter shutdown): the executor may be gone.
                            writer.rollback()
                    except Exception as rollback_error:
                        logger.error(f'Rollback failed: {rollback_error}')

                    if isinstance(e, ControlFlowError):
                        # Normal control flow (a client-input validation error raised
                        # inside the connection scope), NOT a database fault: rolled
                        # back, but charging it would let a client repeatedly sending
                        # invalid input open the process-global breaker. Exempted on
                        # the readonly arm and in begin_transaction for the same
                        # reason.
                        raise
                    if not isinstance(e, Exception):
                        # Cancellation / interpreter shutdown: rolled back above,
                        # but not a database fault -- do not trip the breaker,
                        # exactly as begin_transaction treats the same unwind.
                        raise
                    if is_sqlite_locked_error(e):
                        # SQLITE_BUSY / SQLITE_LOCKED write contention (typically a
                        # concurrent process sharing the database file): rolled back,
                        # but self-clearing contention is NOT a database fault --
                        # charging it would open the breaker on the very condition the
                        # write paths are taught to ride out. Same exemption as the
                        # readonly arm and begin_transaction.
                        logger.warning(f'Direct write hit SQLite contention, rolled back (retry expected): {e}')
                        raise
                    self._record_charged_failure(e)
                    raise
        else:
            # Use write queue for normal write operations
            raise RuntimeError(
                'Direct write connections not allowed. Use execute_write() method or set allow_write=True.',
            )

    @overload
    async def execute_read(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...

    @overload
    async def execute_read(
        self,
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...

    async def execute_read(
        self,
        operation: Callable[..., T] | Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a read operation with a reader connection.

        Args:
            operation: Sync callable to execute with connection as first argument.
                      Signature: operation(conn: sqlite3.Connection, *args, **kwargs) -> T
                      Note: Although protocol accepts sync or async, SQLiteBackend only uses sync.
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            sqlite3.OperationalError: Re-raised for the SQLITE_BUSY / SQLITE_LOCKED
                family only after the bounded retry budget is exhausted, WITHOUT
                counting it in failed_queries or charging the breaker (self-clearing
                contention, not a database fault). A ControlFlowError or any other
                fault raised inside the read callable propagates unchanged from the
                single-attempt helper; the connection scope inside that helper does
                the accounting (ControlFlowError stays out of failed_queries, a
                genuine fault is charged there with last_error/last_error_time).

        Note:
            SQLiteBackend expects SYNC callables (not async). The operation is executed
            synchronously in a thread executor to avoid blocking the event loop.

            SQLITE_BUSY / SQLITE_LOCKED contention from a read is retried with the
            same bounded exponential backoff as the write path, since reads are
            idempotent (each attempt runs on a fresh reader connection). Contention
            never charges the failed_queries metric or the breaker; only a genuine
            fault does.
        """
        last_locked_error: sqlite3.OperationalError | None = None
        for attempt in range(self.retry_config.max_retries):
            try:
                return await self._execute_read_once(operation, *args, **kwargs)
            except sqlite3.OperationalError as e:
                if not is_sqlite_locked_error(e):
                    raise
                last_locked_error = e
                if attempt + 1 >= self.retry_config.max_retries:
                    # Final attempt failed: no retry follows, so a backoff sleep here
                    # would be pure dead latency. Fall through to the exhaustion raise.
                    break
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.backoff_factor**attempt),
                    self.retry_config.max_delay,
                )
                if self.retry_config.jitter:
                    delay += random.uniform(0, delay * 0.3)
                logger.warning(
                    f'Database locked on read, retrying in {delay:.2f}s '
                    f'(attempt {attempt + 1}/{self.retry_config.max_retries})',
                )
                await asyncio.sleep(delay)

        # Retries exhausted on self-clearing contention: re-raise the locked error
        # WITHOUT charging failed_queries or the breaker (mirrors begin_transaction's
        # contention exemption), so a long external lock cannot open the breaker.
        raise last_locked_error or sqlite3.OperationalError('database is locked')

    async def _execute_read_once(
        self,
        operation: Callable[..., T] | Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run one read attempt on a fresh reader connection.

        Args:
            operation: Sync callable invoked with the connection as first argument.
            *args: Additional positional arguments for the operation.
            **kwargs: Additional keyword arguments for the operation.

        Returns:
            The operation result.

        Note:
            Every fault raised by the read callable propagates unchanged; failure
            ACCOUNTING belongs to the enclosing ``get_connection(readonly=True)``
            arms, which own the whole classification -- ControlFlowError and the
            self-clearing SQLITE_BUSY / SQLITE_LOCKED family exempt, a genuine
            fault routed through ``_record_charged_failure`` so failed_queries,
            last_error and last_error_time move together with the breaker charge.
            Counting a fault here as well would double-count every read fault, and
            counting it here INSTEAD would report a failure count with no
            diagnostic message.
        """
        async with self.get_connection(readonly=True) as conn:
            loop = asyncio.get_running_loop()

            def _execute() -> T:
                # Cast to sync callable since SQLiteBackend only uses sync operations
                sync_operation = cast(Callable[..., T], operation)
                result = sync_operation(conn, *args, **kwargs)
                self.metrics.total_queries += 1
                return result

            # Drain the read like every other executor hop (commit, rollback,
            # the direct write, reader creation): a cancellation landing on a bare
            # await releases get_connection's finally, which CLOSES this temporary
            # reader connection while the query is still running on the worker
            # thread -- a use-after-free that stalls the event loop for the length
            # of the in-flight statement or crashes the interpreter. The drain lets
            # the query finish before the connection is closed.
            return await run_in_executor_uninterruptible(loop, _execute)
