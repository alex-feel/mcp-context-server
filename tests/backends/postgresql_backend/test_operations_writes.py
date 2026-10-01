"""Tests for app/backends/postgresql_backend/operations.py: execute_write retries and fault accounting."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.postgresql_backend.acquire_faults import ConnectionEstablishmentTimeoutError


class TestExecuteWriteStatementTimeoutRetry:
    """execute_write retries QueryCanceledError (SQLSTATE 57014) then succeeds."""

    @staticmethod
    def _make_backend() -> PostgreSQLBackend:
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        # Tight, deterministic retry config: enough attempts, no real sleeping.
        backend.retry_config.max_retries = 3
        backend.retry_config.base_delay = 0.0
        backend.retry_config.max_delay = 0.0
        backend.retry_config.jitter = False
        return backend

    @pytest.mark.asyncio
    async def test_execute_write_retries_query_canceled_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A single QueryCanceledError is retried; the second attempt commits once.

        Asserts NO duplicate write: the operation callable is invoked exactly
        twice total (one cancelled attempt + one successful attempt), and the
        successful attempt returns its value exactly once.
        """
        backend = self._make_backend()
        backend._shutdown = False

        # Fake transaction context manager (no real DB).
        @contextlib.asynccontextmanager
        async def _fake_transaction() -> AsyncIterator[None]:
            yield None

        mock_conn = MagicMock()
        mock_conn.transaction = MagicMock(side_effect=_fake_transaction)

        # get_connection is an async context manager yielding mock_conn.
        @contextlib.asynccontextmanager
        async def _fake_get_connection(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
            yield mock_conn

        monkeypatch.setattr(backend, 'get_connection', _fake_get_connection)

        call_count = {'n': 0}

        async def operation(_conn: object, value: str) -> str:
            call_count['n'] += 1
            if call_count['n'] == 1:
                raise asyncpg.exceptions.QueryCanceledError(
                    'canceling statement due to statement timeout',
                )
            return value

        result = await backend.execute_write(operation, 'committed-once')

        assert result == 'committed-once'
        # Exactly two invocations: one cancelled, one successful. No third
        # invocation => no duplicate write.
        assert call_count['n'] == 2

    @pytest.mark.asyncio
    async def test_execute_write_query_canceled_exhausts_then_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Persistent QueryCanceledError exhausts retries and re-raises it."""
        backend = self._make_backend()
        backend.retry_config.max_retries = 2
        backend._shutdown = False

        @contextlib.asynccontextmanager
        async def _fake_transaction() -> AsyncIterator[None]:
            yield None

        mock_conn = MagicMock()
        mock_conn.transaction = MagicMock(side_effect=_fake_transaction)

        @contextlib.asynccontextmanager
        async def _fake_get_connection(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
            yield mock_conn

        monkeypatch.setattr(backend, 'get_connection', _fake_get_connection)

        async def operation(_conn: object) -> None:
            raise asyncpg.exceptions.QueryCanceledError('still timing out')

        with pytest.raises(asyncpg.exceptions.QueryCanceledError):
            await backend.execute_write(operation)


class TestExecuteWriteDeadlockRetry:
    """execute_write retries server-initiated rollbacks (SQLSTATE class 40) uncharged.

    PostgreSQL aborts one transaction to break a deadlock (40P01) or a
    serialization cycle (40001) and expects the loser to retry. Each aborted
    attempt is routine write contention, not a database fault, so no attempt may
    charge the circuit breaker; only retry exhaustion records the single failure
    for the logical write.
    """

    @staticmethod
    def _make_backend() -> PostgreSQLBackend:
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        backend.retry_config.max_retries = 3
        backend.retry_config.base_delay = 0.0
        backend.retry_config.max_delay = 0.0
        backend.retry_config.jitter = False
        backend._shutdown = False
        return backend

    @staticmethod
    def _install_fake_connection(
        backend: PostgreSQLBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        @contextlib.asynccontextmanager
        async def _fake_transaction() -> AsyncIterator[None]:
            yield None

        mock_conn = MagicMock()
        mock_conn.transaction = MagicMock(side_effect=_fake_transaction)

        @contextlib.asynccontextmanager
        async def _fake_get_connection(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
            yield mock_conn

        monkeypatch.setattr(backend, 'get_connection', _fake_get_connection)

    @pytest.mark.asyncio
    async def test_deadlock_is_retried_and_never_charges_breaker(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A deadlocked attempt retries and succeeds with zero breaker failures."""
        backend = self._make_backend()
        self._install_fake_connection(backend, monkeypatch)
        record_failure = AsyncMock()
        monkeypatch.setattr(backend.circuit_breaker, 'record_failure', record_failure)

        call_count = {'n': 0}

        async def operation(_conn: object, value: str) -> str:
            call_count['n'] += 1
            if call_count['n'] == 1:
                raise asyncpg.exceptions.DeadlockDetectedError('deadlock detected')
            return value

        result = await backend.execute_write(operation, 'committed-once')

        assert result == 'committed-once'
        assert call_count['n'] == 2
        record_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_persistent_deadlock_exhausts_with_single_breaker_failure(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Retry exhaustion re-raises the rollback and charges exactly one failure."""
        backend = self._make_backend()
        backend.retry_config.max_retries = 2
        self._install_fake_connection(backend, monkeypatch)
        record_failure = AsyncMock()
        monkeypatch.setattr(backend.circuit_breaker, 'record_failure', record_failure)

        async def operation(_conn: object) -> None:
            raise asyncpg.exceptions.DeadlockDetectedError('still deadlocked')

        with pytest.raises(asyncpg.exceptions.DeadlockDetectedError):
            await backend.execute_write(operation)

        record_failure.assert_awaited_once()


class TestExecuteWritePoolAcquireTimeout:
    """execute_write separates acquire-phase timeouts by their exception type.

    A saturated pool surfaces the bare TimeoutError of the acquire deadline --
    a capacity signal, not a database fault, so it stays uncharged. Dialing a
    new connection to an unreachable (blackholed) database instead surfaces the
    typed ConnectionEstablishmentTimeoutError raised by the pool's connect
    callable, which must charge the breaker (with the failure metrics), or the
    outage never opens it. A TimeoutError raised AFTER a live connection was
    obtained (statement exceeded the pool command_timeout) still charges
    exactly one failure.
    """

    @staticmethod
    def _make_backend() -> PostgreSQLBackend:
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        backend.retry_config.max_retries = 3
        backend.retry_config.base_delay = 0.0
        backend.retry_config.max_delay = 0.0
        backend.retry_config.jitter = False
        backend._shutdown = False
        return backend

    @staticmethod
    def _install_acquire_failure(
        backend: PostgreSQLBackend,
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
    ) -> None:
        """Make get_connection raise the given error on context entry.

        Mirrors the real flow where get_connection re-raises acquire-phase
        failures uncharged under record_breaker=False, leaving the accounting
        to execute_write's own arms.
        """

        class _FailOnEnter:
            async def __aenter__(self) -> object:
                raise error

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        def _acquire_fails(*_args: object, **_kwargs: object) -> _FailOnEnter:
            return _FailOnEnter()

        monkeypatch.setattr(backend, 'get_connection', _acquire_fails)

    @pytest.mark.asyncio
    async def test_saturation_timeout_propagates_uncharged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A bare acquire TimeoutError (pool saturation) stays uncharged."""
        backend = self._make_backend()
        record_failure = AsyncMock()
        monkeypatch.setattr(backend.circuit_breaker, 'record_failure', record_failure)

        self._install_acquire_failure(
            backend, monkeypatch, TimeoutError('pool acquire timed out'),
        )

        call_count = {'n': 0}

        async def operation(_conn: object) -> None:
            call_count['n'] += 1

        with pytest.raises(TimeoutError):
            await backend.execute_write(operation)

        assert call_count['n'] == 0
        record_failure.assert_not_called()
        assert backend.metrics.failed_queries == 0

    @pytest.mark.asyncio
    async def test_establishment_timeout_charges_once_with_metrics(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The typed establishment timeout charges the breaker and the metrics.

        An unreachable database surfaces as ConnectionEstablishmentTimeoutError
        from the pool's connect callable; leaving it uncharged would keep the
        breaker closed through the entire outage and repeat the full connect
        stall on every request, and skipping the metrics would leave dashboards
        keyed on failed_queries/last_error reporting a healthy database.
        """
        backend = self._make_backend()
        record_failure = AsyncMock()
        monkeypatch.setattr(backend.circuit_breaker, 'record_failure', record_failure)

        self._install_acquire_failure(
            backend,
            monkeypatch,
            ConnectionEstablishmentTimeoutError(
                'timed out establishing a new database connection',
            ),
        )

        call_count = {'n': 0}

        async def operation(_conn: object) -> None:
            call_count['n'] += 1

        with pytest.raises(ConnectionEstablishmentTimeoutError):
            await backend.execute_write(operation)

        assert call_count['n'] == 0
        record_failure.assert_awaited_once()
        assert backend.metrics.failed_queries == 1
        assert backend.metrics.last_error is not None
        assert 'establishing' in backend.metrics.last_error
        assert backend.metrics.last_error_time is not None

    @pytest.mark.asyncio
    async def test_statement_timeout_after_acquire_charges_once(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A TimeoutError on a live connection still charges exactly one failure."""
        backend = self._make_backend()
        record_failure = AsyncMock()
        monkeypatch.setattr(backend.circuit_breaker, 'record_failure', record_failure)

        @contextlib.asynccontextmanager
        async def _fake_transaction() -> AsyncIterator[None]:
            yield None

        mock_conn = MagicMock()
        mock_conn.transaction = MagicMock(side_effect=_fake_transaction)

        @contextlib.asynccontextmanager
        async def _fake_get_connection(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
            yield mock_conn

        monkeypatch.setattr(backend, 'get_connection', _fake_get_connection)

        async def operation(_conn: object) -> None:
            raise TimeoutError('statement exceeded pool command_timeout')

        with pytest.raises(TimeoutError):
            await backend.execute_write(operation)

        record_failure.assert_awaited_once()
        assert backend.metrics.failed_queries == 1


class TestExecuteWriteFinalAttemptBackoff:
    """The write retry arms do not sleep the backoff after the final attempt.

    With no retry remaining, sleeping only delays the exhaustion tail's single
    breaker charge and the caller's error by up to max_delay plus jitter, while
    logging a 'retrying' message for a retry that never happens.
    """

    @pytest.mark.asyncio
    async def test_no_backoff_sleep_after_final_attempt(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With two attempts, exactly one inter-attempt backoff sleep occurs."""
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        backend.retry_config.max_retries = 2
        backend.retry_config.base_delay = 0.01
        backend.retry_config.max_delay = 0.01
        backend.retry_config.jitter = False
        backend._shutdown = False

        @contextlib.asynccontextmanager
        async def _fake_transaction() -> AsyncIterator[None]:
            yield None

        mock_conn = MagicMock()
        mock_conn.transaction = MagicMock(side_effect=_fake_transaction)

        @contextlib.asynccontextmanager
        async def _fake_get_connection(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
            yield mock_conn

        monkeypatch.setattr(backend, 'get_connection', _fake_get_connection)

        sleep_mock = AsyncMock()
        monkeypatch.setattr(asyncio, 'sleep', sleep_mock)

        async def operation(_conn: object) -> None:
            raise asyncpg.exceptions.DeadlockDetectedError('still deadlocked')

        with pytest.raises(asyncpg.exceptions.DeadlockDetectedError):
            await backend.execute_write(operation)

        # One backoff between attempt one and attempt two; none after the final
        # attempt (the exhaustion tail raises immediately).
        assert sleep_mock.await_count == 1
