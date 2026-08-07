"""PostgreSQL acquire, release and boot fault-classification tests.

Three accounting boundaries the backend must not get wrong:

- A bare acquire TimeoutError whose deadline CANCELLED an in-flight dial is an
  unreachable database, not pool saturation. asyncpg wraps the queue wait and
  the connect callable in one deadline, so the cancelled dial never produces a
  typed establishment timeout and the two fault classes become identical by
  exception type alone.
- A pooled connection that fails on RELEASE (the pool's reset callback runs
  there and asyncpg re-raises after terminating the connection) must not have
  already been credited a breaker SUCCESS, and must not report an operation that
  already completed -- including a COMMITted write -- as failed.
- initialize() must prove reachability itself: with POSTGRESQL_POOL_MIN=0
  asyncpg pre-connects nothing, so create_pool() succeeds against an unreachable
  host or a wrong password, and a diagnostic probe swallows the evidence.
"""

import asyncio
import contextlib
import socket
import unittest.mock
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.backends.postgresql_backend import PostgreSQLBackend


async def _dial_cancelled_then_acquire_timeout() -> None:
    """Reproduce the shape asyncpg produces when an acquire deadline kills a dial.

    ``Pool._acquire`` wraps the queue wait AND the connect callable in ONE
    ``wait_for``, so an expiring acquire budget cancels the in-flight dial: the
    connect callable receives CancelledError (never a TimeoutError it could
    type), and the acquire surfaces a BARE TimeoutError indistinguishable from
    genuine pool saturation.

    Raises:
        TimeoutError: Always, after the dial has been cancelled.
    """
    from app.backends.postgresql_backend import _connect_pool_connection

    with (
        unittest.mock.patch(
            'asyncpg.connect',
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError(),
        ),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await _connect_pool_connection('postgresql://localhost:5432/testdb')
    raise TimeoutError('pool acquire timed out')


def _backend(connection_string: str = 'postgresql://postgres:postgres@localhost:5432/testdb') -> PostgreSQLBackend:
    """Build a non-shut-down backend without a pool.

    Args:
        connection_string: DSN handed to the backend.

    Returns:
        The constructed backend.
    """
    backend = PostgreSQLBackend(connection_string=connection_string)
    backend._shutdown = False
    return backend


def _backend_with_cancelled_dial() -> PostgreSQLBackend:
    """Build a backend whose acquire cancels its dial and then times out.

    Returns:
        A backend wired to a pool reproducing the cancelled-dial acquire.
    """
    backend = _backend()

    class _CancelDialThenTimeout:
        async def __aenter__(self) -> object:
            await _dial_cancelled_then_acquire_timeout()
            raise AssertionError('unreachable')

        async def __aexit__(self, *_exc: object) -> bool:
            return False

    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=lambda **_kwargs: _CancelDialThenTimeout())
    backend._pool = pool
    return backend


def _backend_with_saturation_timeout() -> PostgreSQLBackend:
    """Build a backend whose acquire times out without attempting any dial.

    Returns:
        A backend wired to a pool reproducing a saturated-pool acquire.
    """
    backend = _backend()

    class _FailOnEnter:
        async def __aenter__(self) -> object:
            raise TimeoutError('pool acquire timed out')

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
    backend = _backend()

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


def _fast_retries(backend: PostgreSQLBackend) -> None:
    """Make the write retry loop deterministic and sleepless.

    Args:
        backend: The backend whose retry configuration is tightened.
    """
    backend.retry_config.max_retries = 3
    backend.retry_config.base_delay = 0.0
    backend.retry_config.max_delay = 0.0
    backend.retry_config.jitter = False


class TestCancelledDialCharging:
    """A bare acquire TimeoutError whose deadline killed a dial is charged.

    The typed establishment timeout only covers the case where the CONNECT
    budget wins the race. When the ACQUIRE budget wins, asyncpg cancels the dial
    instead, so no typed error is ever constructed and an unreachable
    (blackholed) database is indistinguishable from a saturated pool by type
    alone -- leaving the breaker closed, failed_queries at zero and last_error
    null for the entire outage.
    """

    @pytest.mark.asyncio
    async def test_connect_wrapper_records_a_cancelled_dial(self) -> None:
        """The connect callable records the interruption and re-raises unchanged."""
        from app.backends.postgresql_backend import _connect_pool_connection
        from app.backends.postgresql_backend import _track_acquire

        with _track_acquire() as acquire_state:
            assert acquire_state.interrupted is False
            with (
                unittest.mock.patch(
                    'asyncpg.connect',
                    new_callable=AsyncMock,
                    side_effect=asyncio.CancelledError(),
                ),
                pytest.raises(asyncio.CancelledError),
            ):
                await _connect_pool_connection('postgresql://localhost:5432/testdb')
            assert acquire_state.interrupted is True

    @pytest.mark.asyncio
    async def test_successful_dial_records_no_interruption(self) -> None:
        """A dial that completes leaves the record untouched."""
        from app.backends.postgresql_backend import _connect_pool_connection
        from app.backends.postgresql_backend import _track_acquire

        with _track_acquire() as acquire_state:
            with unittest.mock.patch(
                'asyncpg.connect',
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ):
                await _connect_pool_connection('postgresql://localhost:5432/testdb')
            assert acquire_state.interrupted is False

    def test_nested_scopes_share_one_record(self) -> None:
        """An inner scope reuses the outer record instead of shadowing it.

        execute_write acquires through get_connection, so a record created by
        the inner scope would hide the interruption from the outer arm that has
        to charge it.
        """
        from app.backends.postgresql_backend import _track_acquire

        with _track_acquire() as outer, _track_acquire() as inner:
            assert inner is outer

    @pytest.mark.asyncio
    async def test_get_connection_charges_a_cancelled_dial(self) -> None:
        """get_connection charges the bare timeout that killed a dial."""
        backend = _backend_with_cancelled_dial()

        with pytest.raises(TimeoutError):
            async with backend.get_connection():
                pass

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1
        assert backend.metrics.last_error is not None
        assert backend.metrics.last_error_time is not None

    @pytest.mark.asyncio
    async def test_begin_transaction_charges_a_cancelled_dial(self) -> None:
        """begin_transaction charges the bare timeout that killed a dial."""
        backend = _backend_with_cancelled_dial()

        with pytest.raises(TimeoutError):
            async with backend.begin_transaction():
                pass

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1

    @pytest.mark.asyncio
    async def test_execute_write_charges_a_cancelled_dial(self) -> None:
        """execute_write charges the bare timeout that killed a dial."""
        backend = _backend_with_cancelled_dial()
        _fast_retries(backend)

        async def operation(_conn: object) -> None:
            raise AssertionError('the operation must never run')

        with pytest.raises(TimeoutError):
            await backend.execute_write(operation)

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1

    @pytest.mark.asyncio
    async def test_saturation_without_a_dial_stays_uncharged(self) -> None:
        """A bare timeout with no dial attempt remains an uncharged capacity signal."""
        backend = _backend_with_saturation_timeout()

        with pytest.raises(TimeoutError):
            async with backend.get_connection():
                pass

        assert backend.circuit_breaker.failures == 0
        assert backend.metrics.failed_queries == 0

    @pytest.mark.asyncio
    async def test_execute_write_saturation_stays_uncharged(self) -> None:
        """execute_write keeps treating a dial-free acquire timeout as capacity."""
        backend = _backend_with_saturation_timeout()
        _fast_retries(backend)

        async def operation(_conn: object) -> None:
            raise AssertionError('the operation must never run')

        with pytest.raises(TimeoutError):
            await backend.execute_write(operation)

        assert backend.circuit_breaker.failures == 0
        assert backend.metrics.failed_queries == 0

    @pytest.mark.asyncio
    async def test_blackholed_database_charges_through_a_real_pool(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A real asyncpg pool against a blackholed listener opens the accounting.

        The listener completes the TCP handshake and then never speaks the
        protocol, so the dial hangs until the (much shorter) acquire deadline
        cancels it -- the exact interleaving a firewalled or partitioned database
        produces.

        The budgets are installed directly on the module binding rather than
        through the environment: the settings boundary now REFUSES a connect budget
        at or above the acquire budget, and a sub-second acquire deadline is what
        makes the cancellation reproducible in a test. The code path itself is not
        limited to that ordering -- under the correctly ordered defaults an acquire
        that spends its budget queueing reaches the dial with the same result.
        """
        import app.backends.postgresql_backend as pg_module
        from app.backends.postgresql_backend import _connect_pool_connection
        from app.settings import get_settings

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(('127.0.0.1', 0))
        listener.listen(8)
        port = listener.getsockname()[1]

        get_settings.cache_clear()
        base = get_settings()
        squeezed = base.model_copy(
            update={
                'storage': base.storage.model_copy(
                    update={
                        'postgresql_pool_timeout_s': 0.5,
                        'postgresql_connect_timeout_s': 60.0,
                    },
                ),
            },
        )
        monkeypatch.setattr(pg_module, 'settings', squeezed)

        backend = _backend(f'postgresql://postgres:postgres@127.0.0.1:{port}/testdb')
        pool = await asyncpg.create_pool(
            backend.connection_string,
            min_size=0,
            max_size=1,
            connect=_connect_pool_connection,
            timeout=60,
        )
        backend._pool = pool
        try:
            with pytest.raises(TimeoutError):
                async with backend.get_connection():
                    pass

            assert backend.circuit_breaker.failures == 1
            assert backend.metrics.failed_queries == 1
            assert backend.metrics.last_error is not None
            assert backend.metrics.last_error_time is not None
        finally:
            pool.terminate()
            listener.close()
            get_settings.cache_clear()


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
        _fast_retries(backend)

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
        _fast_retries(backend)

        async def operation(_conn: object) -> str:
            return 'stored'

        assert await backend.execute_write(operation) == 'stored'
        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1
        assert backend.metrics.last_error is not None

    @pytest.mark.asyncio
    async def test_execute_write_clean_release_still_credits_success(self) -> None:
        """A clean write is still credited, so the guard cannot suppress every success."""
        backend = _backend()
        _fast_retries(backend)

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


class TestBootConnectivityVerification:
    """initialize() proves reachability itself instead of trusting a diagnostic probe.

    With POSTGRESQL_POOL_MIN=0 (an explicitly supported cold-pool choice)
    asyncpg pre-connects nothing, so create_pool() succeeds against an
    unreachable host or a wrong password. Without an unconditional classified
    dial, initialize() logged a WARNING (invisible at the default
    LOG_LEVEL=ERROR) and then reported success, leaving the authentication
    failure to surface later as a raw schema-statement error the supervisor
    restart-loops on.
    """

    @staticmethod
    def _backend_with_pool_acquire_error(
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
    ) -> PostgreSQLBackend:
        """Build a backend whose created pool fails on the first acquire.

        Args:
            monkeypatch: Fixture used to stub the vector-provisioning steps.
            error: The exception the pool's acquire raises.

        Returns:
            A backend ready for initialize().
        """
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:wrong@localhost:5432/testdb',
        )
        monkeypatch.setattr(backend, '_resolve_provision_vector', AsyncMock(return_value=False))
        monkeypatch.setattr(backend, '_ensure_pgvector_extension', AsyncMock())
        monkeypatch.setattr(backend, '_detect_session_mode_pooler', MagicMock())

        class _FailOnEnter:
            async def __aenter__(self) -> object:
                raise error

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        pool = MagicMock()
        pool.acquire = MagicMock(side_effect=lambda **_kwargs: _FailOnEnter())
        monkeypatch.setattr(
            'app.backends.postgresql_backend.asyncpg.create_pool',
            AsyncMock(return_value=pool),
        )
        return backend

    @pytest.mark.asyncio
    async def test_bad_password_on_a_cold_pool_is_a_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A wrong password reaches the exit-78 classifier instead of 'initialized'."""
        from app.errors import ConfigurationError

        backend = self._backend_with_pool_acquire_error(
            monkeypatch,
            asyncpg.exceptions.InvalidPasswordError('password authentication failed'),
        )

        with pytest.raises(ConfigurationError, match='authentication failed'):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_unreachable_host_on_a_cold_pool_is_a_dependency_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unreachable host reaches the retryable exit-69 classifier."""
        from app.errors import DependencyError

        backend = self._backend_with_pool_acquire_error(
            monkeypatch,
            ConnectionRefusedError('connection refused'),
        )

        with pytest.raises(DependencyError, match='PostgreSQL connection failed'):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_missing_database_on_a_cold_pool_is_a_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A nonexistent database reaches the exit-78 classifier."""
        from app.errors import ConfigurationError

        backend = self._backend_with_pool_acquire_error(
            monkeypatch,
            asyncpg.exceptions.InvalidCatalogNameError('database "testdb" does not exist'),
        )

        with pytest.raises(ConfigurationError, match='does not exist'):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_no_pg_hba_entry_is_a_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SQLSTATE 28000 ('no pg_hba.conf entry for host') is permanent, not retryable.

        It is a sibling of InvalidPasswordError under the same SQLSTATE class 28, and
        equally unfixable by restarting; matching only the password subclass sent it
        to the terminal handler as a retryable DependencyError, which is exactly the
        supervisor restart loop the classification ladder exists to prevent.
        """
        from app.errors import ConfigurationError

        backend = self._backend_with_pool_acquire_error(
            monkeypatch,
            asyncpg.exceptions.InvalidAuthorizationSpecificationError(
                'no pg_hba.conf entry for host "10.0.0.7"',
            ),
        )

        with pytest.raises(ConfigurationError, match='authentication failed'):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_revoked_connect_privilege_is_a_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SQLSTATE 42501 (revoked CONNECT) is permanent, not retryable."""
        from app.errors import ConfigurationError

        backend = self._backend_with_pool_acquire_error(
            monkeypatch,
            asyncpg.exceptions.InsufficientPrivilegeError(
                'permission denied for database "testdb"',
            ),
        )

        with pytest.raises(ConfigurationError, match='permission denied'):
            await backend.initialize()


class TestVectorProvisionProbeFaultScope:
    """The vector-provision probe answers a question; it never invents an answer.

    The probe decides whether the pgvector extension and vector codec must be
    provisioned. A connect fault means it never got to ask, so swallowing it and
    returning False would skip CREATE EXTENSION and the codec while a later
    ``CREATE TABLE ... vector(dim)`` still runs, failing unclassified. Letting the
    fault propagate puts it in front of initialize()'s classification ladder.
    """

    @pytest.mark.asyncio
    async def test_connect_fault_propagates_to_the_classifier(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A wrong password at probe time becomes exit 78, not a False answer."""
        import app.backends.postgresql_backend as pg_module
        from app.errors import ConfigurationError
        from app.settings import get_settings

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:wrong@localhost:5432/testdb',
        )
        monkeypatch.setattr(
            'app.backends.postgresql_backend.asyncpg.connect',
            AsyncMock(side_effect=asyncpg.exceptions.InvalidPasswordError(
                'password authentication failed',
            )),
        )
        # Force the probe path: it is skipped outright when generation is on and
        # compression is off. The settings models are frozen, so a copy is installed
        # on the module binding the backend reads.
        get_settings.cache_clear()
        base = get_settings()
        monkeypatch.setattr(
            pg_module,
            'settings',
            base.model_copy(
                update={'compression': base.compression.model_copy(update={'enabled': True})},
            ),
        )

        with pytest.raises(ConfigurationError, match='authentication failed'):
            await backend.initialize()


class TestPgpoolProbeFaultScope:
    """Only the Pgpool-II detection QUERY is diagnostic; its acquire is not."""

    @staticmethod
    def _backend_with_acquire(pool_acquire: object) -> PostgreSQLBackend:
        """Build a backend whose pool acquire is the supplied context factory.

        Args:
            pool_acquire: Callable returning the acquire context manager.

        Returns:
            A backend wired to the fake pool.
        """
        backend = _backend()
        pool = MagicMock()
        pool.acquire = MagicMock(side_effect=pool_acquire)
        backend._pool = pool
        return backend

    @pytest.mark.asyncio
    async def test_acquire_failure_propagates(self) -> None:
        """An establishment fault at acquire time is not swallowed as a probe failure."""

        class _FailOnEnter:
            async def __aenter__(self) -> object:
                raise asyncpg.exceptions.InvalidPasswordError('password authentication failed')

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        backend = self._backend_with_acquire(lambda **_kwargs: _FailOnEnter())

        with pytest.raises(asyncpg.exceptions.InvalidPasswordError):
            await backend._detect_pgpool_ii()

    @pytest.mark.asyncio
    async def test_query_failure_is_swallowed_and_names_the_exception_type(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A failing detection query only warns, naming the type for empty messages."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(side_effect=TimeoutError())

        acquire_ctx = AsyncMock()
        acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
        acquire_ctx.__aexit__ = AsyncMock(return_value=None)
        backend = self._backend_with_acquire(lambda **_kwargs: acquire_ctx)

        with caplog.at_level('WARNING'):
            await backend._detect_pgpool_ii()

        assert backend._pgpool_version is None
        assert 'TimeoutError' in caplog.text

    @pytest.mark.asyncio
    async def test_detected_version_is_still_reported(self) -> None:
        """A successful detection query still records the Pgpool-II version."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value='4.5.2 (firebrick)')

        acquire_ctx = AsyncMock()
        acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
        acquire_ctx.__aexit__ = AsyncMock(return_value=None)
        backend = self._backend_with_acquire(lambda **_kwargs: acquire_ctx)

        await backend._detect_pgpool_ii()

        assert backend._pgpool_version == '4.5.2 (firebrick)'
