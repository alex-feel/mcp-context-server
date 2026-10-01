"""Atomic multi-operation transactions of the SQLite backend."""

import asyncio
import logging
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from app.backends._executor import run_in_executor_uninterruptible
from app.backends.sqlite_backend.connections import SQLiteConnectionsMixin
from app.backends.sqlite_backend.contention import is_sqlite_locked_error
from app.errors import ControlFlowError

logger = logging.getLogger(__name__)


@dataclass
class SQLiteTransactionContext:
    """Transaction context for SQLite backend.

    Provides access to the writer connection within an active transaction.
    The transaction lifecycle is managed by SQLiteBackend.begin_transaction().

    Note: SQLite operations are SYNCHRONOUS. When using this context,
    wrap operations in asyncio.run_in_executor() for async compatibility.

    Attributes:
        _connection: The sqlite3.Connection for this transaction
    """

    _connection: sqlite3.Connection

    @property
    def connection(self) -> sqlite3.Connection:
        """Get the SQLite connection."""
        return self._connection

    @property
    def backend_type(self) -> str:
        """Get backend type identifier."""
        return 'sqlite'


class SQLiteTransactionMixin(SQLiteConnectionsMixin):
    """Run several operations atomically in one transaction on the shared writer."""

    @asynccontextmanager
    async def begin_transaction(self) -> AsyncGenerator[SQLiteTransactionContext, None]:
        """Begin an atomic transaction spanning multiple operations.

        This method bypasses the write queue and acquires the writer connection
        directly, providing exclusive access for the duration of the transaction.

        IMPORTANT: This method is intended for multi-operation atomic writes.
        For single operations, use execute_write() which is more efficient.

        Transaction semantics:
        - SQLite uses isolation_level='DEFERRED', transaction begins on first write
        - On successful context exit: COMMIT
        - On exception: ROLLBACK

        Yields:
            SQLiteTransactionContext with the writer connection

        Raises:
            RuntimeError: If backend is shutting down or circuit breaker is open

        Example:
            from app.ids import generate_id

            async with backend.begin_transaction() as txn:
                conn = txn.connection
                # All operations use the same connection and transaction
                context_id = generate_id()
                conn.execute(
                    'INSERT INTO context_entries (id, ...) VALUES (?, ...)',
                    (context_id, ...),
                )
                conn.execute(
                    'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                    (context_id, 'tag1'),
                )
                # COMMIT on exit
        """
        assert self._writer_lock is not None, 'Backend not initialized, call initialize() first'

        if self._shutdown:
            raise RuntimeError('Connection manager is shutting down')

        # Check circuit breaker
        if self.circuit_breaker.is_open():
            raise RuntimeError(
                f'Database circuit breaker is open after {self.circuit_breaker.failures} failures',
            )

        # Acquire writer lock to ensure exclusive access
        async with self._writer_lock:
            # Wrap the writer-recreation await so a genuine establishment fault
            # (e.g. SQLITE_CANTOPEN when the database volume detaches after the
            # health check closes the dead writer) charges the breaker instead of
            # escaping above the transaction-body recording block below with the
            # breaker still reporting healthy.
            writer = await self._acquire_connection_charging_faults(self._ensure_writer_connection)
            loop = asyncio.get_running_loop()

            # Create transaction context
            txn_context = SQLiteTransactionContext(_connection=writer)

            try:
                yield txn_context

                # Success: commit transaction. The commit hop is drained (not a
                # bare offload) so a cancellation landing on it cannot leave a
                # zombie commit callable to fire later against the shared writer
                # connection -- the drain runs it to completion, and the
                # cancellation then falls into the rollback arm below as a
                # harmless no-op (nothing is open once the commit succeeded).
                await run_in_executor_uninterruptible(loop, writer.commit)
                # Count the committed transaction as one completed operation, the
                # same unit execute_write/execute_read count. Without it
                # total_queries never moves for the transactional store/update/
                # delete path while its failure arm below still moves
                # failed_queries, so the two counters published side by side in
                # connection_metrics cover different populations and any
                # failure-rate computed from them is wrong.
                self.metrics.total_queries += 1
                self.circuit_breaker.record_success()
                logger.debug('Transaction committed successfully')

            except BaseException as e:
                # Roll back either way; only a genuine DB fault trips the breaker.
                # BaseException is caught deliberately: the repositories run their
                # transaction-body closures on the executor, so task cancellation
                # (CancelledError) can unwind through this block mid-transaction --
                # an Exception-only handler would skip the rollback and leave an
                # open partial transaction on the pooled writer connection that
                # the NEXT write on it would silently commit.
                try:
                    if isinstance(e, Exception):
                        # Drained offload, NOT a bare one: an ordinary-Exception
                        # body (e.g. a routine VersionConflictError) can still be
                        # cancelled while the rollback is queued/running on the
                        # shared executor, and a bare offload would skip the
                        # rollback -- reopening the exact silent-commit vector
                        # the drain closes for transaction bodies.
                        await run_in_executor_uninterruptible(loop, writer.rollback)
                    else:
                        # Synchronous rollback on a BaseException unwind (interpreter
                        # shutdown): the executor may be gone, so do not offload; the
                        # transaction body was already drained to completion, so the
                        # connection is quiescent and this rollback runs fast.
                        writer.rollback()
                except Exception as rollback_error:
                    logger.error(f'Rollback failed: {rollback_error}')

                if isinstance(e, ControlFlowError):
                    # Normal control flow (optimistic-concurrency conflict / post-dedup
                    # embedding reconciliation), NOT a database fault: rolled back, but
                    # do NOT record a circuit-breaker failure, so normal write
                    # contention cannot open the breaker and reject healthy writes.
                    raise

                if not isinstance(e, Exception):
                    # Cancellation / interpreter shutdown: rolled back above, but
                    # not a database fault -- do not trip the breaker.
                    raise

                if is_sqlite_locked_error(e):
                    # SQLITE_BUSY / SQLITE_LOCKED write contention (typically a
                    # concurrent process sharing the database file): rolled back,
                    # but self-clearing contention is NOT a database fault -- do
                    # not charge the breaker, mirroring the PostgreSQL class-40
                    # rollback exemption. The tool-layer retry loops classify it
                    # via is_connection_error and re-run the transaction with
                    # backoff.
                    logger.warning(f'Transaction hit SQLite write contention, rolled back (retry expected): {e}')
                    raise

                logger.warning(f'Transaction failed, rolling back: {e}')
                # Charged failure WITH metrics: begin_transaction is the main
                # store/update write path on this backend and no execute_read /
                # execute_write wrapper observes transactional flows, so this is
                # the only site that can count the fault into
                # failed_queries/last_error.
                self._record_charged_failure(e)
                raise
