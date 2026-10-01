"""Tests for app/backends/postgresql_backend/operations.py: acquire and release fault accounting.

An acquire-phase fault is charged to the circuit breaker exactly once unless it
is control flow or pool saturation. A pooled connection that fails on RELEASE
(the pool's reset callback runs there and asyncpg re-raises after terminating
the connection) must not have already been credited a breaker SUCCESS, and
must not report an operation that already completed -- including a COMMITted
write -- as failed.
"""

import contextlib
import socket
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.postgresql_backend.acquire_faults import ConnectionEstablishmentTimeoutError
from app.errors import ControlFlowError
from tests.backends.postgresql_backend._builders import build_backend
from tests.backends.postgresql_backend._builders import fast_retries


def _backend_with_acquire_failure(error: Exception) -> PostgreSQLBackend:
    """Build a backend whose pool.acquire raises the given error on entry.

    Simulates an acquire-phase failure (the exception surfaces from the
    ``async with pool.acquire(...)`` entry, before any connection body runs),
    the shape asyncpg produces for saturation deadlines, typed establishment
    timeouts, refused ports, DNS failures, and setup-callback failures.

    Args:
        error: The exception pool.acquire raises on context entry.

    Returns:
        A backend wired to the failing fake pool.
    """
    backend = PostgreSQLBackend(
        connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
    )
    backend._shutdown = False

    class _FailOnEnter:
        async def __aenter__(self) -> object:
            raise error

        async def __aexit__(self, *_exc: object) -> bool:
            return False

    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=lambda **_kwargs: _FailOnEnter())
    backend._pool = pool
    return backend


def _backend_with_release_failure(error: Exception, conn: object) -> PostgreSQLBackend:
    """Build a backend whose pooled connection fails when it is released.

    asyncpg runs the pool's reset callback on release and re-raises when it
    fails (after terminating the connection), and ``PoolAcquireContext.__aexit__``
    awaits the release unconditionally -- so the fault escapes the acquire block
    AFTER the body already completed.

    Args:
        error: The exception the release raises.
        conn: Object handed to the caller of the acquire context.

    Returns:
        A backend wired to the failing fake pool.
    """
    backend = build_backend()

    class _FailOnExit:
        async def __aenter__(self) -> object:
            return conn

        async def __aexit__(self, *_exc: object) -> bool:
            raise error

    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=lambda **_kwargs: _FailOnExit())
    backend._pool = pool
    return backend


def _connection_with_transaction() -> MagicMock:
    """Build a connection mock whose transaction() is an async context manager.

    Returns:
        The configured connection mock.
    """

    @contextlib.asynccontextmanager
    async def _fake_transaction() -> AsyncIterator[None]:
        yield None

    conn = MagicMock()
    conn.transaction = MagicMock(side_effect=_fake_transaction)
    return conn


class TestAcquireTimeoutBreakerDiscrimination:
    """get_connection and begin_transaction separate acquire-phase timeouts by type.

    Pool saturation surfaces as the bare TimeoutError of the acquire deadline --
    a capacity signal that stays uncharged. An unreachable database surfaces as
    the typed ConnectionEstablishmentTimeoutError raised by the pool's connect
    callable and charges the breaker with the failure metrics, so read and
    transaction paths can also open it during a blackholed outage regardless of
    how the connect budget compares to the pool budget.
    """

    @pytest.mark.asyncio
    async def test_get_connection_establishment_timeout_charges(self) -> None:
        """The typed establishment timeout charges once with the failure metrics."""
        backend = _backend_with_acquire_failure(
            ConnectionEstablishmentTimeoutError(
                'timed out establishing a new database connection',
            ),
        )

        with pytest.raises(ConnectionEstablishmentTimeoutError):
            async with backend.get_connection():
                pass

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1
        assert backend.metrics.last_error is not None
        assert 'establishing' in backend.metrics.last_error
        assert backend.metrics.last_error_time is not None

    @pytest.mark.asyncio
    async def test_get_connection_saturation_timeout_uncharged(self) -> None:
        """A bare acquire TimeoutError (pool saturation) stays uncharged."""
        backend = _backend_with_acquire_failure(TimeoutError('pool acquire timed out'))

        with pytest.raises(TimeoutError):
            async with backend.get_connection():
                pass

        assert backend.circuit_breaker.failures == 0
        assert backend.metrics.failed_queries == 0

    @pytest.mark.asyncio
    async def test_get_connection_record_breaker_false_leaves_typed_timeout_uncharged(self) -> None:
        """record_breaker=False defers typed-timeout accounting to execute_write."""
        backend = _backend_with_acquire_failure(
            ConnectionEstablishmentTimeoutError(
                'timed out establishing a new database connection',
            ),
        )

        with pytest.raises(ConnectionEstablishmentTimeoutError):
            async with backend.get_connection(record_breaker=False):
                pass

        assert backend.circuit_breaker.failures == 0
        assert backend.metrics.failed_queries == 0

    @pytest.mark.asyncio
    async def test_begin_transaction_establishment_timeout_charges(self) -> None:
        """begin_transaction charges the typed establishment timeout with metrics."""
        backend = _backend_with_acquire_failure(
            ConnectionEstablishmentTimeoutError(
                'timed out establishing a new database connection',
            ),
        )

        with pytest.raises(ConnectionEstablishmentTimeoutError):
            async with backend.begin_transaction():
                pass

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1
        assert backend.metrics.last_error is not None
        assert backend.metrics.last_error_time is not None

    @pytest.mark.asyncio
    async def test_begin_transaction_saturation_timeout_uncharged(self) -> None:
        """A saturation timeout on the transaction acquire stays uncharged."""
        backend = _backend_with_acquire_failure(TimeoutError('pool acquire timed out'))

        with pytest.raises(TimeoutError):
            async with backend.begin_transaction():
                pass

        assert backend.circuit_breaker.failures == 0
        assert backend.metrics.failed_queries == 0


class TestAcquirePhaseConnectionFailureCharging:
    """Non-timeout acquire-phase connection failures charge the breaker with metrics.

    A refused port (the common container-crash shape), a DNS resolution failure,
    a connection reset, or a setup-callback failure raised while acquiring a
    pool connection is a genuine database fault: left uncharged, an outage whose
    connections fail instantly never opens the breaker even though every request
    fails, while the blackholed (timeout) form of the same outage does open it.
    ControlFlowError stays exempt, body failures are not double-charged, and the
    execute_write-driven acquire (record_breaker=False) leaves accounting to
    execute_write's own arms.
    """

    @pytest.mark.asyncio
    async def test_get_connection_refused_port_charges_with_metrics(self) -> None:
        """A refused connection at acquire time charges once with the metrics."""
        backend = _backend_with_acquire_failure(ConnectionRefusedError('connection refused'))

        with pytest.raises(ConnectionRefusedError):
            async with backend.get_connection():
                pass

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1
        assert backend.metrics.last_error is not None
        assert 'refused' in backend.metrics.last_error
        assert backend.metrics.last_error_time is not None

    @pytest.mark.asyncio
    async def test_get_connection_dns_failure_charges(self) -> None:
        """A DNS resolution failure at acquire time charges the breaker."""
        backend = _backend_with_acquire_failure(
            socket.gaierror(socket.EAI_AGAIN, 'temporary failure in name resolution'),
        )

        with pytest.raises(socket.gaierror):
            async with backend.get_connection():
                pass

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1

    @pytest.mark.asyncio
    async def test_get_connection_setup_callback_failure_charges(self) -> None:
        """A setup-callback failure re-raised from pool.acquire charges the breaker."""
        backend = _backend_with_acquire_failure(
            asyncpg.exceptions.ConnectionDoesNotExistError(
                'connection setup timed out; the pooled connection is unusable',
            ),
        )

        with pytest.raises(asyncpg.exceptions.ConnectionDoesNotExistError):
            async with backend.get_connection():
                pass

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1

    @pytest.mark.asyncio
    async def test_get_connection_record_breaker_false_leaves_refusal_uncharged(self) -> None:
        """record_breaker=False defers acquire-failure accounting to execute_write."""
        backend = _backend_with_acquire_failure(ConnectionRefusedError('connection refused'))

        with pytest.raises(ConnectionRefusedError):
            async with backend.get_connection(record_breaker=False):
                pass

        assert backend.circuit_breaker.failures == 0
        assert backend.metrics.failed_queries == 0

    @pytest.mark.asyncio
    async def test_get_connection_control_flow_error_at_acquire_uncharged(self) -> None:
        """A ControlFlowError surfacing from the acquire stays uncharged."""
        backend = _backend_with_acquire_failure(ControlFlowError('client input invalid'))

        with pytest.raises(ControlFlowError):
            async with backend.get_connection():
                pass

        assert backend.circuit_breaker.failures == 0
        assert backend.metrics.failed_queries == 0

    @pytest.mark.asyncio
    async def test_begin_transaction_refused_port_charges_with_metrics(self) -> None:
        """begin_transaction charges a refused connection at acquire time."""
        backend = _backend_with_acquire_failure(ConnectionRefusedError('connection refused'))

        with pytest.raises(ConnectionRefusedError):
            async with backend.begin_transaction():
                pass

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1
        assert backend.metrics.last_error is not None
        assert backend.metrics.last_error_time is not None

    @pytest.mark.asyncio
    async def test_get_connection_body_fault_charged_exactly_once(self) -> None:
        """A body fault is charged by the inner arm only, never re-charged outside."""
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        backend._shutdown = False
        mock_conn = MagicMock()

        @contextlib.asynccontextmanager
        async def _fake_acquire(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
            yield mock_conn

        pool = MagicMock()
        pool.acquire = MagicMock(side_effect=_fake_acquire)
        backend._pool = pool

        with pytest.raises(RuntimeError, match='db fault'):
            async with backend.get_connection():
                raise RuntimeError('db fault')

        assert backend.circuit_breaker.failures == 1


class TestExecuteReadControlFlowMetricsExemption:
    """execute_read keeps ControlFlowError out of the failed_queries metric.

    A client-input validation rejection raised inside a read callable (e.g. an
    invalid metadata filter) is normal control flow, not a database fault:
    counting it would let routine healthy rejections inflate the operator-facing
    health metric and misdiagnose database instability, inconsistent with
    execute_write's exemption.
    """

    @staticmethod
    def _backend_with_fake_pool() -> PostgreSQLBackend:
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        backend._shutdown = False
        mock_conn = MagicMock()

        @contextlib.asynccontextmanager
        async def _fake_acquire(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
            yield mock_conn

        pool = MagicMock()
        pool.acquire = MagicMock(side_effect=_fake_acquire)
        backend._pool = pool
        return backend

    @pytest.mark.asyncio
    async def test_control_flow_error_not_counted_as_failed_query(self) -> None:
        """A ControlFlowError from the read callable leaves failed_queries at zero."""
        backend = self._backend_with_fake_pool()

        async def operation(_conn: object) -> None:
            raise ControlFlowError('client input invalid')

        with pytest.raises(ControlFlowError):
            await backend.execute_read(operation)

        assert backend.metrics.failed_queries == 0

    @pytest.mark.asyncio
    async def test_genuine_read_fault_still_counted(self) -> None:
        """A genuine fault from the read callable still increments failed_queries."""
        backend = self._backend_with_fake_pool()

        async def operation(_conn: object) -> None:
            raise RuntimeError('db fault')

        with pytest.raises(RuntimeError, match='db fault'):
            await backend.execute_read(operation)

        assert backend.metrics.failed_queries == 1


class TestGetConnectionBreakerControlFlowExemption:
    """get_connection exempts ControlFlowError from circuit-breaker failure accounting.

    A client-input validation failure raised inside the connection scope is normal
    control flow, not a database fault: recording it as a breaker failure would let a
    client repeatedly sending invalid input open the breaker and reject every other
    caller's healthy requests on the process-wide backend singleton.
    """

    @staticmethod
    def _backend_with_fake_pool() -> PostgreSQLBackend:
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        backend._shutdown = False
        mock_conn = MagicMock()

        @contextlib.asynccontextmanager
        async def _fake_acquire(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
            yield mock_conn

        pool = MagicMock()
        pool.acquire = MagicMock(side_effect=_fake_acquire)
        backend._pool = pool
        return backend

    @pytest.mark.asyncio
    async def test_control_flow_error_does_not_record_breaker_failure(self) -> None:
        """A ControlFlowError escaping the connection scope leaves the breaker untouched."""
        backend = self._backend_with_fake_pool()
        with pytest.raises(ControlFlowError):
            async with backend.get_connection():
                raise ControlFlowError('client input invalid')
        assert backend.circuit_breaker.failures == 0

    @pytest.mark.asyncio
    async def test_ordinary_exception_still_records_breaker_failure(self) -> None:
        """A genuine fault escaping the connection scope still trips breaker accounting."""
        backend = self._backend_with_fake_pool()
        with pytest.raises(RuntimeError, match='db fault'):
            async with backend.get_connection():
                raise RuntimeError('db fault')
        assert backend.circuit_breaker.failures == 1


class TestReleaseFailureAccounting:
    """A connection that dies on release must not credit a breaker success.

    Crediting success inside the acquire block meant a connection dying
    mid-request (failover, restart, pg_terminate_backend, partition) recorded
    +1 success -- which in the HEALTHY state also DECREMENTS accumulated
    failures -- and zero failures, so successes credited by dying connections
    actively held the breaker closed during an outage. The release fault is
    charged instead, and swallowed, because the body's work already completed.
    """

    @pytest.mark.asyncio
    async def test_get_connection_charges_and_swallows_release_failure(self) -> None:
        """A clean body followed by a failing release charges, and does not raise."""
        backend = _backend_with_release_failure(
            asyncpg.exceptions.ConnectionDoesNotExistError('connection was terminated'),
            MagicMock(),
        )

        async with backend.get_connection():
            pass

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1
        assert backend.metrics.last_error is not None
        assert 'terminated' in backend.metrics.last_error
        assert backend.metrics.last_error_time is not None

    @pytest.mark.asyncio
    async def test_get_connection_release_timeout_is_charged_too(self) -> None:
        """A release that times out is a fault, not a saturation signal."""
        backend = _backend_with_release_failure(TimeoutError('reset timed out'), MagicMock())

        async with backend.get_connection():
            pass

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1

    @pytest.mark.asyncio
    async def test_execute_read_returns_its_result_despite_a_release_failure(self) -> None:
        """A completed read still returns its rows when the release then fails."""
        backend = _backend_with_release_failure(
            asyncpg.exceptions.ConnectionDoesNotExistError('connection was terminated'),
            MagicMock(),
        )

        async def _read(_conn: object) -> str:
            return 'rows'

        assert await backend.execute_read(_read) == 'rows'
        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1

    @pytest.mark.asyncio
    async def test_begin_transaction_reports_a_committed_write_as_success(self) -> None:
        """A COMMITted transaction is never reported as failed by a release fault."""
        backend = _backend_with_release_failure(
            asyncpg.exceptions.ConnectionDoesNotExistError('connection was terminated'),
            _connection_with_transaction(),
        )

        async with backend.begin_transaction() as txn:
            assert txn.backend_type == 'postgresql'

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1
        assert backend.metrics.last_error is not None

    @pytest.mark.asyncio
    async def test_execute_write_returns_its_result_despite_a_release_failure(self) -> None:
        """A committed write returns normally instead of driving the caller to retry."""
        backend = _backend_with_release_failure(
            asyncpg.exceptions.ConnectionDoesNotExistError('connection was terminated'),
            _connection_with_transaction(),
        )
        fast_retries(backend)

        calls = {'n': 0}

        async def operation(_conn: object) -> str:
            calls['n'] += 1
            return 'stored'

        assert await backend.execute_write(operation) == 'stored'
        assert calls['n'] == 1
        assert backend.metrics.failed_queries == 1

    @pytest.mark.asyncio
    async def test_execute_write_release_failure_leaves_the_breaker_charged(self) -> None:
        """A swallowed release fault must not be cancelled out by a success credit.

        execute_write suppresses get_connection's own accounting (record_breaker=False)
        so a retried write records exactly one outcome, then credits the success itself
        once the write committed. A release fault is charged and SWALLOWED inside
        get_connection, so the write returns normally -- and crediting a success for it
        cancels the charge out (in the HEALTHY state record_success also decrements
        accumulated failures), leaving a net breaker delta of zero for a connection
        that just died. The result still comes back; only the health accounting differs.
        """
        backend = _backend_with_release_failure(
            asyncpg.exceptions.ConnectionDoesNotExistError('connection was terminated'),
            _connection_with_transaction(),
        )
        fast_retries(backend)

        async def operation(_conn: object) -> str:
            return 'stored'

        assert await backend.execute_write(operation) == 'stored'
        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1
        assert backend.metrics.last_error is not None

    @pytest.mark.asyncio
    async def test_execute_write_clean_release_still_credits_success(self) -> None:
        """A clean write is still credited, so the guard cannot suppress every success."""
        backend = build_backend()
        fast_retries(backend)

        conn = _connection_with_transaction()

        class _CleanAcquire:
            async def __aenter__(self) -> object:
                return conn

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        pool = MagicMock()
        pool.acquire = MagicMock(side_effect=lambda **_kwargs: _CleanAcquire())
        backend._pool = pool
        backend.circuit_breaker.failures = 2

        async def operation(_conn: object) -> str:
            return 'stored'

        assert await backend.execute_write(operation) == 'stored'
        # record_success decrements accumulated failures in the HEALTHY state.
        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 0

    @pytest.mark.asyncio
    async def test_body_fault_with_release_failure_charges_once(self) -> None:
        """A failing body followed by a failing release is charged exactly once."""
        backend = _backend_with_release_failure(
            asyncpg.exceptions.ConnectionDoesNotExistError('connection was terminated'),
            MagicMock(),
        )

        with pytest.raises(asyncpg.exceptions.ConnectionDoesNotExistError):
            async with backend.get_connection():
                raise RuntimeError('body fault')

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1
