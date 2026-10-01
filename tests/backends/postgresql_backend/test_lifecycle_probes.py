"""Tests for app/backends/postgresql_backend/lifecycle.py: the boot-time probes.

initialize() must prove reachability itself: with POSTGRESQL_POOL_MIN=0
asyncpg pre-connects nothing, so create_pool() succeeds against an unreachable
host or a wrong password, and a diagnostic probe swallows the evidence. The
Pgpool-II and session-mode pooler probes only diagnose the deployment.
"""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.backends.postgresql_backend import PostgreSQLBackend
from tests.backends.postgresql_backend._builders import build_backend


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
            'asyncpg.create_pool',
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
        backend = build_backend()
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


class TestPgpoolDetection:
    """Test Pgpool-II detection in PostgreSQLBackend."""

    @pytest.mark.asyncio
    async def test_pgpool_detected_when_show_pool_version_succeeds(self) -> None:
        """Pgpool-II should be detected when SHOW POOL_VERSION returns a value."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()

        # Mock connection that returns Pgpool-II version
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value='4.5.2 (firebrick)')

        mock_pool_acquire = AsyncMock()
        mock_pool_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool_acquire.__aexit__ = AsyncMock(return_value=None)
        backend._pool.acquire = MagicMock(return_value=mock_pool_acquire)

        await backend._detect_pgpool_ii()

        assert backend._pgpool_version == '4.5.2 (firebrick)'
        mock_conn.fetchval.assert_called_once_with('SHOW POOL_VERSION')

    @pytest.mark.asyncio
    async def test_direct_connection_when_undefined_object_error(self) -> None:
        """Direct PostgreSQL connection detected when SHOW POOL_VERSION raises UndefinedObjectError."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()

        # Mock connection that raises UndefinedObjectError (error code 42704)
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(
            side_effect=asyncpg.exceptions.UndefinedObjectError(
                'unrecognized configuration parameter "pool_version"',
            ),
        )

        mock_pool_acquire = AsyncMock()
        mock_pool_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool_acquire.__aexit__ = AsyncMock(return_value=None)
        backend._pool.acquire = MagicMock(return_value=mock_pool_acquire)

        await backend._detect_pgpool_ii()

        assert backend._pgpool_version is None

    @pytest.mark.asyncio
    async def test_detection_handles_empty_version_response(self) -> None:
        """Detection should handle empty version response gracefully."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()

        # Mock connection that returns empty/None
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)

        mock_pool_acquire = AsyncMock()
        mock_pool_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool_acquire.__aexit__ = AsyncMock(return_value=None)
        backend._pool.acquire = MagicMock(return_value=mock_pool_acquire)

        await backend._detect_pgpool_ii()

        assert backend._pgpool_version is None

    @pytest.mark.asyncio
    async def test_detection_handles_unexpected_error(self) -> None:
        """Detection should not fail initialization on unexpected errors."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()

        # Mock connection that raises unexpected error
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=RuntimeError('Unexpected error'))

        mock_pool_acquire = AsyncMock()
        mock_pool_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool_acquire.__aexit__ = AsyncMock(return_value=None)
        backend._pool.acquire = MagicMock(return_value=mock_pool_acquire)

        # Should not raise, just log and continue
        await backend._detect_pgpool_ii()

        assert backend._pgpool_version is None

    def test_metrics_include_pgpool_info_when_detected(self) -> None:
        """get_metrics() should include Pgpool-II detection results when detected."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()
        backend._pool.get_size = MagicMock(return_value=5)
        backend._pool.get_idle_size = MagicMock(return_value=3)
        backend._pool.get_min_size = MagicMock(return_value=2)
        backend._pool.get_max_size = MagicMock(return_value=10)
        backend._pgpool_version = '4.5.2 (firebrick)'

        metrics = backend.get_metrics()

        assert metrics['pgpool_detected'] is True
        assert metrics['pgpool_version'] == '4.5.2 (firebrick)'

    def test_metrics_include_pgpool_info_when_not_detected(self) -> None:
        """get_metrics() should include pgpool_detected=False when not behind Pgpool-II."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()
        backend._pool.get_size = MagicMock(return_value=5)
        backend._pool.get_idle_size = MagicMock(return_value=3)
        backend._pool.get_min_size = MagicMock(return_value=2)
        backend._pool.get_max_size = MagicMock(return_value=10)
        backend._pgpool_version = None

        metrics = backend.get_metrics()

        assert metrics['pgpool_detected'] is False
        assert metrics['pgpool_version'] is None

    def test_metrics_omit_pgpool_info_before_detection_runs(self) -> None:
        """get_metrics() should not include pgpool fields if detection never ran."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()
        backend._pool.get_size = MagicMock(return_value=5)
        backend._pool.get_idle_size = MagicMock(return_value=3)
        backend._pool.get_min_size = MagicMock(return_value=2)
        backend._pool.get_max_size = MagicMock(return_value=10)
        # _pgpool_version attribute not set (detection never ran)

        metrics = backend.get_metrics()

        assert 'pgpool_detected' not in metrics
        assert 'pgpool_version' not in metrics


class TestSessionModePoolerDetection:
    """Test session-mode pooler detection in PostgreSQLBackend."""

    def test_warns_for_session_pooler_with_high_pool_max(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """WARN and flag detection for *.pooler.supabase.com:5432 with high pool_max."""
        from app.backends.postgresql_backend import PostgreSQLBackend
        from app.backends.postgresql_backend import lifecycle

        monkeypatch.setattr(
            lifecycle.settings.storage, 'postgresql_pool_max', 20, raising=False,
        )

        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@aws-0-us-east-1.pooler.supabase.com:5432/postgres',
        )

        caplog.set_level(logging.WARNING)
        backend._detect_session_mode_pooler()

        assert backend._session_mode_pooler is True
        assert any('MaxClientsInSessionMode' in r.message for r in caplog.records)

    def test_detects_session_pooler_from_libpq_keyvalue_dsn(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A libpq key-value DSN (no URL scheme) is parsed for host/port too.

        ``urlsplit`` yields an empty hostname for the ``host=... port=...``
        spelling that asyncpg also accepts, so URL parsing alone would never fire
        the advisory for a real Supabase Session Pooler set via a key-value
        POSTGRESQL_CONNECTION_STRING.
        """
        from app.backends.postgresql_backend import PostgreSQLBackend
        from app.backends.postgresql_backend import lifecycle

        monkeypatch.setattr(
            lifecycle.settings.storage, 'postgresql_pool_max', 20, raising=False,
        )

        backend = PostgreSQLBackend(
            connection_string=(
                'host=aws-0-us-east-1.pooler.supabase.com port=5432 '
                'user=u password=p dbname=postgres'
            ),
        )

        caplog.set_level(logging.WARNING)
        backend._detect_session_mode_pooler()

        assert backend._session_mode_pooler is True
        assert any('MaxClientsInSessionMode' in r.message for r in caplog.records)

    def test_no_warn_for_session_pooler_within_cap(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Detected but no WARN when pool_max is within the default cap."""
        from app.backends.postgresql_backend import PostgreSQLBackend
        from app.backends.postgresql_backend import lifecycle

        monkeypatch.setattr(
            lifecycle.settings.storage, 'postgresql_pool_max', 10, raising=False,
        )

        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@aws-0-us-east-1.pooler.supabase.com:5432/postgres',
        )

        caplog.set_level(logging.WARNING)
        backend._detect_session_mode_pooler()

        assert backend._session_mode_pooler is True
        assert not any('MaxClientsInSessionMode' in r.message for r in caplog.records)

    def test_transaction_mode_port_not_flagged(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Transaction-mode pooler (port 6543) is not a session pooler."""
        from app.backends.postgresql_backend import PostgreSQLBackend
        from app.backends.postgresql_backend import lifecycle

        monkeypatch.setattr(
            lifecycle.settings.storage, 'postgresql_pool_max', 50, raising=False,
        )

        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@aws-0-us-east-1.pooler.supabase.com:6543/postgres',
        )

        caplog.set_level(logging.WARNING)
        backend._detect_session_mode_pooler()

        assert backend._session_mode_pooler is False
        assert not any('MaxClientsInSessionMode' in r.message for r in caplog.records)

    def test_direct_connection_not_flagged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A direct (non-Supabase) host is never flagged as a session pooler."""
        from app.backends.postgresql_backend import PostgreSQLBackend
        from app.backends.postgresql_backend import lifecycle

        monkeypatch.setattr(
            lifecycle.settings.storage, 'postgresql_pool_max', 100, raising=False,
        )

        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@localhost:5432/mcp_context',
        )

        backend._detect_session_mode_pooler()

        assert backend._session_mode_pooler is False

    def test_metrics_include_session_pooler_when_detected(self) -> None:
        """get_metrics() reports session_mode_pooler_detected after detection runs."""
        from unittest.mock import MagicMock

        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@aws-0-us-east-1.pooler.supabase.com:5432/postgres',
        )
        backend._pool = MagicMock()
        backend._pool.get_size = MagicMock(return_value=5)
        backend._pool.get_idle_size = MagicMock(return_value=3)
        backend._pool.get_min_size = MagicMock(return_value=2)
        backend._pool.get_max_size = MagicMock(return_value=20)
        backend._session_mode_pooler = True

        metrics = backend.get_metrics()

        assert metrics['session_mode_pooler_detected'] is True

    def test_metrics_omit_session_pooler_before_detection_runs(self) -> None:
        """get_metrics() omits the field if detection never ran."""
        from unittest.mock import MagicMock

        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@localhost:5432/mcp_context',
        )
        backend._pool = MagicMock()
        backend._pool.get_size = MagicMock(return_value=5)
        backend._pool.get_idle_size = MagicMock(return_value=3)
        backend._pool.get_min_size = MagicMock(return_value=2)
        backend._pool.get_max_size = MagicMock(return_value=10)
        # _session_mode_pooler attribute not set (detection never ran)

        metrics = backend.get_metrics()

        assert 'session_mode_pooler_detected' not in metrics
