"""Pooled connection access and the read and write entry points of the PostgreSQL backend.

Every operation acquires through ``get_connection``, which applies the circuit breaker and
its fault accounting; ``execute_write`` adds the retry loop and the per-write transaction.
"""

import asyncio
import logging
import random
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any
from typing import TypeVar
from typing import cast
from typing import overload

import asyncpg

from app.backends.postgresql_backend.acquire_faults import ConnectionEstablishmentTimeoutError
from app.backends.postgresql_backend.acquire_faults import track_acquire
from app.backends.postgresql_backend.core import PostgreSQLBackendCore
from app.errors import ControlFlowError
from app.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# Type definitions
T = TypeVar('T')


class PostgreSQLOperationsMixin(PostgreSQLBackendCore):
    """Acquire pooled connections with breaker accounting, and run reads and retried writes."""

    @asynccontextmanager
    async def get_connection(
        self,
        readonly: bool = False,
        allow_write: bool = False,
        record_breaker: bool = True,
    ) -> AsyncGenerator[Any, None]:
        """Get a database connection from the pool.

        Args:
            readonly: Advisory flag (PostgreSQL handles via transactions)
            allow_write: Advisory flag (PostgreSQL handles via transactions)
            record_breaker: When True (default), record a circuit-breaker success
                on a clean exit and a charged failure on an exception. execute_write
                sets this False so its retry loop records at most ONE breaker
                outcome per logical write (matching the SQLite backend) instead of
                one per retry attempt. It does NOT cover the release phase: a
                release failure is swallowed here, so no caller could account for
                it, and it is therefore always charged (see the note below).

        Yields:
            asyncpg.Connection from the pool

        Raises:
            RuntimeError: If backend is shut down or circuit breaker is open
            ControlFlowError: Re-raised from the connection scope without recording a
                circuit-breaker failure (normal control flow, not a database fault)
            ConnectionEstablishmentTimeoutError: If dialing a new connection for the
                acquire timed out (an unreachable database; charged so the outage
                can open the breaker)
            TimeoutError: If the pool stayed saturated past the acquire deadline (a
                capacity signal, re-raised uncharged), or if the acquire deadline
                cancelled an in-flight dial (an unreachable database, charged)

        Note:
            A RELEASE-phase failure after a clean body (asyncpg runs the pool's
            reset callback when the connection goes back to the pool and re-raises
            if it fails, e.g. against a connection killed by a failover) is
            charged, logged, and SWALLOWED: the body's work already completed, so
            propagating it would report a finished operation -- including a
            COMMITTED write -- as failed and send the caller into a retry that
            re-runs it. Because the swallow returns normally, a caller running
            with ``record_breaker=False`` is told about it out of band through the
            acquire tracker's ``release_failed`` flag, so it does not credit a
            breaker success that would cancel this charge out.
        """
        # Parameters readonly and allow_write are part of StorageBackend protocol
        # but not used in PostgreSQL implementation (handled via transactions)
        _ = readonly
        _ = allow_write

        if self._shutdown:
            raise RuntimeError('PostgreSQL backend is shutting down')

        # Check circuit breaker
        if await self.circuit_breaker.is_open():
            raise RuntimeError(
                f'Database circuit breaker is open after {self.circuit_breaker.failures} failures',
            )

        assert self._pool is not None, 'Backend not initialized, call initialize() first'

        # Acquire connection from pool, bounded by the acquire-wait timeout so
        # pool exhaustion surfaces as a TimeoutError instead of an unbounded
        # hang (asyncpg's Pool.acquire waits forever with timeout=None).
        acquired = False
        body_ok = False
        with track_acquire() as acquire_state:
            try:
                async with self._pool.acquire(
                    timeout=settings.storage.postgresql_pool_timeout_s,
                ) as conn:
                    acquired = True
                    try:
                        yield conn
                        body_ok = True
                    except ControlFlowError:
                        # Normal control flow (e.g. a client-input validation error raised
                        # inside the connection scope), NOT a database fault: do not record
                        # a breaker failure, or a client repeatedly sending invalid input
                        # opens the breaker and rejects every caller's healthy requests.
                        raise
                    except Exception as e:
                        # The single accounting site for a body fault: execute_read
                        # deliberately records nothing itself, so counting the fault
                        # here keeps failed_queries, last_error and last_error_time
                        # moving together with the breaker charge instead of leaving
                        # a failure count with no diagnostic message.
                        if record_breaker:
                            await self._record_charged_failure(e)
                        raise
                # Success is credited only HERE, after the pooled connection has
                # actually been released: asyncpg runs the reset callback during
                # release and re-raises on failure, so crediting inside the acquire
                # block would let a connection dying mid-request move the health
                # counter the WRONG way (a success, which also decrements accumulated
                # failures) while the caller still received an error.
                if body_ok and record_breaker:
                    await self.circuit_breaker.record_success()
            except ControlFlowError:
                # Already exempted by the inner arm; never a database fault, so the
                # acquire-phase arms below must not observe it either.
                raise
            except ConnectionEstablishmentTimeoutError as e:
                # Dialing a NEW connection for this acquire timed out: an unreachable
                # (blackholed) database, a genuine fault that must charge the breaker --
                # or the outage never opens it and every request repeats the full
                # connect stall. Typed by the pool's connect callable, so no
                # elapsed-time inference is needed.
                if not acquired and record_breaker:
                    await self._record_charged_failure(e)
                raise
            except TimeoutError as e:
                # A bare acquire TimeoutError is saturation ONLY when no dial was
                # interrupted: if the acquire deadline cancelled an in-flight dial,
                # asyncpg destroyed the typed establishment signal and this is an
                # unreachable database, which must charge or the outage never opens
                # the breaker. A release-phase timeout after a clean body is a
                # genuine fault too (and is swallowed below).
                if body_ok or (not acquired and acquire_state.interrupted and record_breaker):
                    await self._record_charged_failure(e)
                if body_ok:
                    acquire_state.release_failed = True
                    self._log_swallowed_release_failure(e)
                    return
                raise
            except Exception as e:
                # Acquire-phase connection failure (refused port, DNS resolution,
                # connection reset, setup-callback failure): a genuine database fault
                # that charges the breaker, mirroring execute_write's generic arm.
                # record_breaker=False (the execute_write-driven acquire) leaves
                # acquire/body accounting to that caller so nothing double-charges.
                # A release-phase failure after a clean body is charged
                # UNCONDITIONALLY: the caller cannot account for it, because it is
                # swallowed here and never reaches the caller's arms.
                if body_ok or (not acquired and record_breaker):
                    await self._record_charged_failure(e)
                if body_ok:
                    acquire_state.release_failed = True
                    self._log_swallowed_release_failure(e)
                    return
                raise

    async def _validate_connection_state(self, conn: asyncpg.Connection) -> bool:
        """Validate connection is in healthy state before critical operations.

        Executes a lightweight query to verify the connection protocol state
        is synchronized with the server. This catches corrupted connections
        before they cause protocol errors in batch operations.

        Args:
            conn: The asyncpg connection to validate

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            # Lightweight query to verify protocol state
            await conn.fetchval('SELECT 1')
            return True
        except Exception as e:
            # Do not record a breaker failure here: execute_write (the only caller)
            # records exactly one breaker outcome per logical write on its final
            # result, so per-attempt accounting would over-count under retries.
            logger.warning(f'Connection validation failed: {e}')
            return False

    async def _backoff_before_next_attempt(self, attempt: int, log_prefix: str) -> None:
        """Sleep the exponential backoff between write retry attempts.

        No-op after the FINAL attempt: with no retry remaining, sleeping would
        only add up to max_delay plus jitter of dead latency before the
        exhaustion tail's single breaker charge and the caller's error, while
        logging a 'retrying' message for a retry that never happens.

        Args:
            attempt: Zero-based index of the attempt that just failed.
            log_prefix: Failure description for the retry warning log.
        """
        if attempt + 1 >= self.retry_config.max_retries:
            return
        delay = min(
            self.retry_config.base_delay * (self.retry_config.backoff_factor**attempt),
            self.retry_config.max_delay,
        )
        if self.retry_config.jitter:
            delay += random.uniform(0, delay * 0.3)
        logger.warning(
            f'{log_prefix}, retrying in {delay:.2f}s '
            f'(attempt {attempt + 1}/{self.retry_config.max_retries})',
        )
        await asyncio.sleep(delay)

    @overload
    async def execute_write(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        validate_connection: bool = False,
        **kwargs: Any,
    ) -> T: ...

    @overload
    async def execute_write(
        self,
        operation: Callable[..., T],
        *args: Any,
        validate_connection: bool = False,
        **kwargs: Any,
    ) -> T: ...

    async def execute_write(
        self,
        operation: Callable[..., T] | Callable[..., Awaitable[T]],
        *args: Any,
        validate_connection: bool = False,
        **kwargs: Any,
    ) -> T:
        """Execute a write operation with retry logic and transaction management.

        Args:
            operation: Async callable that performs the write operation.
                      Signature: async def operation(conn: asyncpg.Connection, *args, **kwargs) -> T
            *args: Positional arguments to pass to operation
            validate_connection: If True, validate connection state before operation.
                                Use for batch operations that are sensitive to protocol state.
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of the operation (type preserved via TypeVar)

        Raises:
            RuntimeError: If backend is shut down or circuit breaker is open
            asyncpg.exceptions.ConnectionDoesNotExistError: If connection validation fails
            ConnectionEstablishmentTimeoutError: If a new connection could not be
                established within the connect timeout during the acquire (charged --
                an unreachable database must be able to open the breaker)
            TimeoutError: If the connection pool stays saturated past the acquire
                deadline (re-raised uncharged), if the acquire deadline cancelled an
                in-flight dial (an unreachable database, charged), or if a statement
                exceeds the pool command_timeout (charged)

        Note:
            PostgreSQLBackend expects ASYNC callables (not sync). The operation is executed
            with await and wrapped in a transaction for consistency.
        """
        if self._shutdown:
            raise RuntimeError('PostgreSQL backend is shutting down')

        # Reject up-front when the breaker is already open (mirrors begin_transaction).
        # Doing this BEFORE the retry loop means a rejection is never seen by the loop's
        # generic handler, so it cannot record a spurious breaker failure that would
        # reset last_failure_time and perpetuate the open state (self-lockout).
        if await self.circuit_breaker.is_open():
            raise RuntimeError(
                f'Database circuit breaker is open after {self.circuit_breaker.failures} failures',
            )

        last_error: Exception | None = None

        for attempt in range(self.retry_config.max_retries):
            # False until we hold a live connection. It distinguishes an acquire-phase
            # TimeoutError (pool saturation) from an operation-level command_timeout (a
            # statement ran on a live connection and exceeded the pool's
            # command_timeout).
            acquired = False
            # Track the dials this attempt makes so a bare acquire TimeoutError
            # whose deadline cancelled an in-flight dial is charged as the
            # unreachable database it is, instead of being mistaken for pool
            # saturation. A fresh tracker per attempt keeps the observation
            # scoped to the attempt that raised.
            with track_acquire() as acquire_state:
                try:
                    async with self.get_connection(readonly=False, record_breaker=False) as conn:
                        acquired = True
                        # Validate connection state before critical operations
                        if validate_connection and not await self._validate_connection_state(conn):
                            raise asyncpg.exceptions.ConnectionDoesNotExistError(
                                'Connection validation failed - connection may be corrupted',
                            )

                        async with conn.transaction():
                            # Cast to async callable since PostgreSQLBackend only uses async operations
                            async_operation = cast(Callable[..., Awaitable[T]], operation)
                            result = await async_operation(conn, *args, **kwargs)
                            self.metrics.total_queries += 1
                    # Record exactly ONE breaker success per logical write, AFTER the
                    # transaction commits AND the pooled connection is released
                    # (record_breaker=False above suppresses the per-attempt accounting
                    # so a retried write is not counted N times). Crediting inside the
                    # connection scope would credit a success for a connection that
                    # then died on release; get_connection charges that release fault
                    # itself and swallows it, since the write already committed.
                    #
                    # A SWALLOWED release fault still returns normally here, so the
                    # tracker flag is the only way this arm can see it. Crediting a
                    # success for it would cancel get_connection's charge out (in the
                    # HEALTHY state record_success also decrements accumulated
                    # failures), leaving a net breaker delta of zero for a connection
                    # that just died -- the same wrong-direction accounting that crediting
                    # after the release, rather than inside the acquire block, prevents.
                    # The write itself still succeeded, so the result is returned either
                    # way.
                    if not acquire_state.release_failed:
                        await self.circuit_breaker.record_success()
                    return result

                except asyncpg.exceptions.TransactionRollbackError as e:
                    # Transaction-rollback failure (SQLSTATE class 40: serialization_failure
                    # 40001, deadlock_detected 40P01, and siblings). PostgreSQL aborted one
                    # transaction to break a serialization cycle or a deadlock; the loser is
                    # expected to retry, and the retry succeeds once the competing transaction
                    # commits. Catching the class-40 base (not only SerializationError) means a
                    # deadlock -- e.g. two atomic update batches locking the same rows in
                    # opposite order -- retries here instead of falling through to the generic
                    # arm and charging the breaker for a routine, self-clearing lock cycle.
                    last_error = e
                    await self._backoff_before_next_attempt(
                        attempt, 'Transaction rollback (serialization/deadlock) on write',
                    )

                except asyncpg.exceptions.ConnectionDoesNotExistError as e:
                    # Connection error (including a setup-callback timeout re-raised by
                    # setup_pool_connection) - retry
                    last_error = e
                    await self._backoff_before_next_attempt(attempt, 'Connection error on write')

                except asyncpg.exceptions.InternalClientError as e:
                    # Protocol state corruption - retry with fresh connection
                    last_error = e
                    await self._backoff_before_next_attempt(attempt, f'Protocol state error on write: {e}')

                except asyncpg.exceptions.QueryCanceledError as e:
                    # Statement / lock-wait timeout (SQLSTATE 57014): PostgreSQL
                    # cancelled the statement after it exceeded statement_timeout
                    # (~0.9 * POSTGRESQL_COMMAND_TIMEOUT_S, set in setup_pool_connection).
                    # Retry on a fresh connection with bounded backoff. This helps
                    # only a TRANSIENT lock-WAIT that has since cleared; a write
                    # fundamentally slower than the ceiling (e.g. fp32 in-transaction
                    # HNSW maintenance with ENABLE_EMBEDDING_COMPRESSION=false) needs
                    # a higher POSTGRESQL_COMMAND_TIMEOUT_S or compression left ON.
                    # Safe to retry: every write operation reaching execute_write is
                    # idempotent (deduplicating store / keyed update) and all
                    # generation completed outside the transaction.
                    last_error = e
                    await self._backoff_before_next_attempt(attempt, f'Statement timeout on write: {e}')

                except ConnectionEstablishmentTimeoutError as e:
                    # Dialing a NEW connection for the acquire timed out (typed by the
                    # pool's connect callable): an unreachable, blackholed database -- a
                    # genuine fault that must charge the breaker, or the outage never
                    # opens it and every request repeats the full connect stall. Fail
                    # fast with exactly one charged failure (get_connection left it
                    # uncharged under record_breaker=False).
                    await self._record_charged_failure(e)
                    raise

                except TimeoutError as e:
                    # Not acquired, no dial interrupted -> with establishment timeouts
                    # typed above, a bare acquire TimeoutError means the pool stayed
                    # saturated for POSTGRESQL_POOL_TIMEOUT_S: a capacity signal, NOT a
                    # database fault -- re-raise WITHOUT charging, matching
                    # get_connection/begin_transaction. Not acquired but a dial WAS
                    # interrupted -> the acquire deadline cancelled the dial, so asyncpg
                    # never constructed the typed establishment error: this is an
                    # unreachable database and must charge, or a total outage keeps the
                    # breaker closed and every request repeats the full stall. Acquired
                    # -> the statement exceeded the pool command_timeout (a genuine
                    # operation stall the server-side statement_timeout did not cancel
                    # first); charge exactly one breaker failure, as before.
                    if not acquired and not acquire_state.interrupted:
                        raise
                    await self._record_charged_failure(e)
                    raise

                except Exception as e:
                    # A circuit-breaker rejection raised by get_connection mid-loop (e.g.
                    # a concurrent writer opened the breaker after the up-front check) is
                    # NOT a write attempt. Recording a breaker failure for it would reset
                    # last_failure_time and perpetuate the open state (self-lockout), so
                    # re-raise that control-flow RuntimeError WITHOUT recording -- is_open()
                    # matches get_connection's own recovery-aware gate.
                    if isinstance(e, RuntimeError) and await self.circuit_breaker.is_open():
                        raise
                    # Control-flow signals (optimistic-concurrency VersionConflictError,
                    # post-dedup EmbeddingsReconcileRequiredError) are normal write contention,
                    # NOT a database fault: roll back and propagate WITHOUT tripping the breaker,
                    # mirroring begin_transaction's exemption (else routine contention could open
                    # the breaker and lock out writes).
                    if isinstance(e, ControlFlowError):
                        raise
                    # Non-retryable write failure -- record the single breaker failure for
                    # this logical write (get_connection's per-attempt accounting is off).
                    await self._record_charged_failure(e)
                    raise

        # Max retries exceeded -- record exactly one breaker failure for the write.
        final_error = last_error if last_error is not None else Exception('Max retries exceeded for write operation')
        await self._record_charged_failure(final_error)
        raise final_error

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
        """Execute a read operation with proper connection handling.

        Args:
            operation: Async callable that performs the read operation.
                      Signature: async def operation(conn: asyncpg.Connection, *args, **kwargs) -> T
                      Note: Although protocol accepts sync or async, PostgreSQLBackend only uses async.
            *args: Positional arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of the operation (type preserved via TypeVar)

        Note:
            PostgreSQLBackend expects ASYNC callables (not sync). The operation is executed
            with await.

            Every fault raised by the read callable propagates unchanged; failure
            ACCOUNTING belongs to the enclosing ``get_connection`` body arms, which
            exempt ControlFlowError (client-input validation is normal control flow,
            not a database fault) and route a genuine fault through
            ``_record_charged_failure`` so failed_queries, last_error and
            last_error_time move together with the breaker charge. Counting a fault
            here as well would double-count it, and counting it here INSTEAD would
            report a failure count with no diagnostic message.
        """
        async with self.get_connection(readonly=True) as conn:
            # Cast to async callable since PostgreSQLBackend only uses async operations
            async_operation = cast(Callable[..., Awaitable[T]], operation)
            result = await async_operation(conn, *args, **kwargs)
            self.metrics.total_queries += 1
            return result
