"""Atomic multi-operation transactions of the PostgreSQL backend."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg

from app.backends.postgresql_backend.acquire_faults import ConnectionEstablishmentTimeoutError
from app.backends.postgresql_backend.acquire_faults import track_acquire
from app.backends.postgresql_backend.core import PostgreSQLBackendCore
from app.errors import ControlFlowError
from app.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class PostgreSQLTransactionContext:
    """Transaction context for PostgreSQL backend.

    Provides access to the asyncpg connection within an active transaction.
    The transaction lifecycle is managed by PostgreSQLBackend.begin_transaction().

    Note: PostgreSQL operations are ASYNCHRONOUS. All operations on the
    connection must be awaited.

    Attributes:
        _connection: The asyncpg connection proxy for this transaction
    """

    _connection: 'asyncpg.pool.PoolConnectionProxy[asyncpg.Record]'

    @property
    def connection(self) -> 'asyncpg.pool.PoolConnectionProxy[asyncpg.Record]':
        """Get the asyncpg connection proxy."""
        return self._connection

    @property
    def backend_type(self) -> str:
        """Get backend type identifier."""
        return 'postgresql'


class PostgreSQLTransactionMixin(PostgreSQLBackendCore):
    """Run several operations atomically in one transaction on one pooled connection."""

    @asynccontextmanager
    async def begin_transaction(self) -> AsyncGenerator[PostgreSQLTransactionContext, None]:
        """Begin an atomic transaction spanning multiple operations.

        This method acquires a connection from the pool and begins a transaction.
        All operations within the context share the same transaction.

        IMPORTANT: This method is intended for multi-operation atomic writes.
        For single operations, use execute_write() which is more efficient.

        Transaction semantics:
        - Uses asyncpg's native transaction context manager
        - Default isolation level: READ COMMITTED
        - On successful context exit: COMMIT
        - On exception: ROLLBACK

        Yields:
            PostgreSQLTransactionContext with the asyncpg connection

        Raises:
            RuntimeError: If backend is shutting down or circuit breaker is open
            ControlFlowError: Re-raised from the transaction scope without recording
                a circuit-breaker failure (normal control flow, not a database fault)
            ConnectionEstablishmentTimeoutError: If dialing a new connection for the
                acquire timed out (an unreachable database; charged so the outage
                can open the breaker)
            TimeoutError: If the pool stayed saturated past the acquire deadline (a
                capacity signal, re-raised uncharged), or if the acquire deadline
                cancelled an in-flight dial (an unreachable database, charged)

        Note:
            A RELEASE-phase failure after a COMMITTED transaction (asyncpg runs the
            pool's reset callback when the connection goes back to the pool and
            re-raises if it fails, e.g. against a connection killed by a failover)
            is charged, logged, and SWALLOWED: the write already landed, so
            propagating it would report a committed store as failed and send the
            caller into a retry that writes it again.

        Example:
            async with backend.begin_transaction() as txn:
                conn = txn.connection
                # All operations use same connection, same transaction
                row = await conn.fetchrow(
                    'INSERT INTO context_entries ... RETURNING id',
                    ...
                )
                context_id = row['id']
                await conn.execute('INSERT INTO tags ...', context_id, 'tag1')
                # COMMIT on exit
        """
        if self._shutdown:
            raise RuntimeError('PostgreSQL backend is shutting down')

        # Check circuit breaker
        if await self.circuit_breaker.is_open():
            raise RuntimeError(
                f'Database circuit breaker is open after {self.circuit_breaker.failures} failures',
            )

        assert self._pool is not None, 'Backend not initialized, call initialize() first'

        # Acquire the connection, then begin the transaction in an INNER context so we
        # observe the COMMIT result. asyncpg issues the COMMIT when the
        # `conn.transaction()` context EXITS; recording success before that exit (a single
        # `async with ..., conn.transaction():` form) would credit a circuit-breaker
        # SUCCESS to a transaction whose COMMIT could still fail -- and that COMMIT
        # failure, raised on the context exit, would land OUTSIDE the try and never be
        # recorded as a failure (it could even leave the breaker mislearning health).
        # Record success only AFTER the inner context exits cleanly (COMMIT succeeded); a
        # COMMIT failure flows through the same except as a body failure.
        acquired = False
        committed = False
        with track_acquire() as acquire_state:
            try:
                async with self._pool.acquire(
                    timeout=settings.storage.postgresql_pool_timeout_s,
                ) as conn:
                    acquired = True
                    # Create transaction context
                    txn_context = PostgreSQLTransactionContext(_connection=conn)

                    try:
                        async with conn.transaction():
                            yield txn_context
                            # Body succeeded; asyncpg COMMITs on exiting this inner context.
                        # Reaching here means the COMMIT itself also succeeded.
                        committed = True
                        # Count the committed transaction as one completed
                        # operation, the same unit execute_write/execute_read
                        # count. Without it total_queries never moves for the
                        # transactional store/update/delete path while the arm
                        # below still moves failed_queries, so the two counters
                        # published side by side in connection_metrics cover
                        # different populations and any failure-rate computed
                        # from them is wrong. Counted here rather than beside
                        # record_success() so a transaction whose RELEASE fails
                        # after a successful COMMIT is still counted -- its work
                        # did reach the database.
                        self.metrics.total_queries += 1

                    except Exception as e:
                        # Rolled back automatically on a body error, OR the COMMIT failed on exit.
                        if isinstance(e, ControlFlowError):
                            # Normal control flow (optimistic-concurrency conflict / post-dedup
                            # embedding reconciliation), NOT a database fault: rolled back
                            # automatically, but do NOT record a circuit-breaker failure, so
                            # normal write contention cannot open the breaker.
                            raise
                        if isinstance(e, asyncpg.exceptions.TransactionRollbackError):
                            # PostgreSQL aborted this transaction to break a deadlock or a
                            # serialization cycle (SQLSTATE class 40) -- routine, self-clearing
                            # write contention, not a database fault. The tool layer classifies
                            # these as retryable and re-runs the transaction, so charging the
                            # breaker per aborted attempt would let normal contention open it
                            # and reject every caller's healthy requests. Rolled back
                            # automatically; propagate uncharged for the caller to retry.
                            logger.warning(
                                f'Transaction rolled back by the server (deadlock/serialization), retry expected: {e}',
                            )
                            raise
                        logger.warning(f'Transaction failed or commit failed, rolling back: {e}')
                        # Charged failure with metrics: no execute_read/execute_write
                        # wrapper observes transactional flows, so this is the only
                        # site that can count the fault into failed_queries/last_error.
                        await self._record_charged_failure(e)
                        raise
                # Success is credited only HERE, after the pooled connection has
                # actually been released: asyncpg runs the reset callback during
                # release and re-raises on failure, so crediting inside the acquire
                # block would let a connection dying mid-request move the health
                # counter the WRONG way (a success, which also decrements accumulated
                # failures) while the caller still received an error.
                if committed:
                    await self.circuit_breaker.record_success()
                    logger.debug('Transaction committed successfully')
            except ControlFlowError:
                # Already exempted by the inner arm; never a database fault, so the
                # acquire-phase arms below must not observe it either.
                raise
            except ConnectionEstablishmentTimeoutError as e:
                # Dialing a NEW connection for this acquire timed out: an unreachable
                # (blackholed) database, a genuine fault that must charge the breaker
                # -- mirroring get_connection and execute_write, so transactional
                # writes can also open the breaker during a blackholed outage. Typed
                # by the pool's connect callable, so no elapsed-time inference is
                # needed.
                if not acquired:
                    await self._record_charged_failure(e)
                raise
            except TimeoutError as e:
                # A bare acquire TimeoutError is saturation ONLY when no dial was
                # interrupted: if the acquire deadline cancelled an in-flight dial,
                # asyncpg destroyed the typed establishment signal and this is an
                # unreachable database, which must charge or the outage never opens
                # the breaker. A release-phase timeout after a COMMITTED transaction
                # is a genuine fault too (and is swallowed below).
                if committed or (not acquired and acquire_state.interrupted):
                    await self._record_charged_failure(e)
                if committed:
                    self._log_swallowed_release_failure(e)
                    return
                raise
            except Exception as e:
                # Acquire-phase connection failure (refused port, DNS resolution,
                # connection reset, setup-callback failure): a genuine database fault
                # that charges the breaker, mirroring execute_write's generic arm. The
                # acquired guard excludes body failures the inner arms already
                # accounted for; a release-phase failure after a COMMITTED transaction
                # is charged here because no other arm can observe it.
                if not acquired or committed:
                    await self._record_charged_failure(e)
                if committed:
                    self._log_swallowed_release_failure(e)
                    return
                raise
