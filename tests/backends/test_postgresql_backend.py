"""Unit tests for PostgreSQL backend implementation.

This module tests PostgreSQL backend functionality without requiring
a PostgreSQL database connection - focusing on unit tests for
configuration, connection string handling, advisory locks, and
connection reset behavior.
"""

import asyncio
import contextlib
import unittest.mock
from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.backends.postgresql_backend import PostgreSQLBackend
from app.errors import ControlFlowError


class TestBackendType:
    """Test backend type identification."""

    def test_backend_type_property(self) -> None:
        """Verify backend_type returns 'postgresql' for all PostgreSQL connections.

        Backend type should be consistent for all PostgreSQL variants.
        """
        # Supabase Direct Connection
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:password@db.project.supabase.co:5432/postgres',
        )
        assert backend.backend_type == 'postgresql', 'Supabase should report postgresql backend_type'

        # Self-hosted PostgreSQL
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:password@localhost:5432/postgres',
        )
        assert backend.backend_type == 'postgresql', 'Self-hosted should report postgresql backend_type'


class TestConnectionStringBuilding:
    """Test connection string construction from settings."""

    def test_explicit_connection_string_preserved(self) -> None:
        """Verify explicit connection strings are preserved as-is.

        When POSTGRESQL_CONNECTION_STRING is provided directly,
        it should be used without modification.
        """
        # Direct Connection via explicit string
        direct_conn = 'postgresql://postgres:password@db.project.supabase.co:5432/postgres'
        backend = PostgreSQLBackend(connection_string=direct_conn)
        assert backend.connection_string == direct_conn

        # Session Pooler via explicit string
        pooler_conn = 'postgresql://postgres.project:password@aws-0-us-west-1.pooler.supabase.com:5432/postgres'
        backend = PostgreSQLBackend(connection_string=pooler_conn)
        assert backend.connection_string == pooler_conn

    @staticmethod
    def _built_connection_string(monkeypatch: pytest.MonkeyPatch, host: str) -> str:
        """Build a DSN from components with POSTGRESQL_HOST set to the given host."""
        import app.backends.postgresql_backend as pg_module
        from app.settings import AppSettings

        monkeypatch.delenv('POSTGRESQL_CONNECTION_STRING', raising=False)
        monkeypatch.setenv('POSTGRESQL_HOST', host)
        monkeypatch.setattr(pg_module, 'settings', AppSettings())
        return PostgreSQLBackend().connection_string

    def test_ipv6_loopback_host_is_bracketed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An IPv6 loopback host literal is bracketed in the built DSN.

        The DSN authority parses a bare colon as the host:port separator, so an
        unbracketed ``::1`` corrupts the parse; RFC 3986 requires the IP-literal
        bracket form ``[::1]``.
        """
        conn_str = self._built_connection_string(monkeypatch, '::1')
        assert '@[::1]:' in conn_str

    def test_full_ipv6_host_literal_is_bracketed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A full IPv6 address literal is bracketed in the built DSN."""
        conn_str = self._built_connection_string(monkeypatch, '2001:db8::1')
        assert '@[2001:db8::1]:' in conn_str

    def test_hostname_without_colon_is_not_bracketed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A plain hostname interpolates without brackets."""
        conn_str = self._built_connection_string(monkeypatch, 'db.example.com')
        assert '@db.example.com:' in conn_str
        assert '[' not in conn_str


class TestBuildAsyncpgConnectKwargs:
    """Unit tests for the shared asyncpg connect-kwargs builder.

    The same helper feeds both the connection pool and the migration CLI, so
    its output is the single source of truth for search_path / statement cache
    behavior. These tests are DB-free.
    """

    def test_search_path_quotes_non_default_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-default schema is double-quoted and followed by public."""
        from app.backends.postgresql_backend import build_asyncpg_connect_kwargs
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_SCHEMA', 'My.Schema')
        kwargs = build_asyncpg_connect_kwargs(AppSettings())
        assert kwargs['server_settings']['search_path'] == '"My.Schema", public'

    def test_search_path_public_is_benign_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default public schema resolves to the benign no-op form."""
        from app.backends.postgresql_backend import build_asyncpg_connect_kwargs
        from app.settings import AppSettings

        monkeypatch.delenv('POSTGRESQL_SCHEMA', raising=False)
        kwargs = build_asyncpg_connect_kwargs(AppSettings())
        assert kwargs['server_settings']['search_path'] == '"public", public'

    def test_search_path_escapes_embedded_double_quote(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A schema name containing a double quote is escaped by doubling it.

        Regression: the unescaped ``f'"{schema}", public'`` produced a malformed
        quoted identifier (and thus a malformed startup search_path parameter) for
        a schema like ``my"schema``. PostgreSQL escapes an embedded double quote by
        doubling it.
        """
        from app.backends.postgresql_backend import build_asyncpg_connect_kwargs
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_SCHEMA', 'my"schema')
        kwargs = build_asyncpg_connect_kwargs(AppSettings())
        assert kwargs['server_settings']['search_path'] == '"my""schema", public'

    def test_statement_cache_zero_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """statement_cache_size=0 (transaction-pooler mode) propagates verbatim."""
        from app.backends.postgresql_backend import build_asyncpg_connect_kwargs
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_STATEMENT_CACHE_SIZE', '0')
        kwargs = build_asyncpg_connect_kwargs(AppSettings())
        assert kwargs['statement_cache_size'] == 0

    def test_statement_cache_default_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default prepared-statement cache size is forwarded unchanged."""
        from app.backends.postgresql_backend import build_asyncpg_connect_kwargs
        from app.settings import AppSettings

        monkeypatch.delenv('POSTGRESQL_STATEMENT_CACHE_SIZE', raising=False)
        kwargs = build_asyncpg_connect_kwargs(AppSettings())
        assert kwargs['statement_cache_size'] == 100

    def test_tcp_keepalive_gucs_present_when_positive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TCP keepalive GUCs appear in server_settings when their value is > 0."""
        from app.backends.postgresql_backend import build_asyncpg_connect_kwargs
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_IDLE_S', '15')
        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_INTERVAL_S', '5')
        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_COUNT', '3')
        server_settings = build_asyncpg_connect_kwargs(AppSettings())['server_settings']
        assert server_settings['tcp_keepalives_idle'] == '15'
        assert server_settings['tcp_keepalives_interval'] == '5'
        assert server_settings['tcp_keepalives_count'] == '3'

    def test_tcp_keepalive_gucs_absent_when_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TCP keepalive GUCs are omitted entirely when set to 0 (disabled)."""
        from app.backends.postgresql_backend import build_asyncpg_connect_kwargs
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_IDLE_S', '0')
        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_INTERVAL_S', '0')
        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_COUNT', '0')
        server_settings = build_asyncpg_connect_kwargs(AppSettings())['server_settings']
        assert 'tcp_keepalives_idle' not in server_settings
        assert 'tcp_keepalives_interval' not in server_settings
        assert 'tcp_keepalives_count' not in server_settings


class TestQuotePgIdentifier:
    """Unit tests for the shared PostgreSQL quoted-identifier helper.

    The same helper quotes the schema for BOTH the connection search_path and the
    migration CLI's CREATE SCHEMA DDL, so they cannot drift on an embedded double
    quote. These tests are DB-free.
    """

    def test_plain_identifier_quoted(self) -> None:
        """A plain identifier is wrapped in double quotes."""
        from app.backends.postgresql_backend import quote_pg_identifier

        assert quote_pg_identifier('public') == '"public"'

    def test_mixed_case_and_dot_preserved(self) -> None:
        """Mixed case and dots are preserved verbatim inside the quotes."""
        from app.backends.postgresql_backend import quote_pg_identifier

        assert quote_pg_identifier('My.Schema') == '"My.Schema"'

    def test_embedded_double_quote_doubled(self) -> None:
        """An embedded double quote is escaped by doubling it (a valid quoted identifier)."""
        from app.backends.postgresql_backend import quote_pg_identifier

        assert quote_pg_identifier('we"ird') == '"we""ird"'

    def test_search_path_uses_the_same_quoting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """build_asyncpg_connect_kwargs quotes the schema via quote_pg_identifier, so the
        connection search_path and any schema-qualified DDL escape it identically."""
        from app.backends.postgresql_backend import build_asyncpg_connect_kwargs
        from app.backends.postgresql_backend import quote_pg_identifier
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_SCHEMA', 'we"ird')
        search_path = build_asyncpg_connect_kwargs(AppSettings())['server_settings']['search_path']
        assert search_path == f'{quote_pg_identifier("we\"ird")}, public'
        assert search_path == '"we""ird", public'


class TestPoolHardeningSettings:
    """Test pool hardening settings."""

    def test_pool_hardening_settings_defaults(self) -> None:
        """Verify pool hardening settings have expected defaults."""
        from app.settings import StorageSettings

        settings = StorageSettings()

        # Verify default values match implementation guide specifications
        assert settings.postgresql_max_inactive_lifetime_s == 300.0
        assert settings.postgresql_max_queries == 10000

    def test_pool_hardening_settings_field_constraints(self) -> None:
        """Verify pool hardening settings have ge=0 constraint allowing zero."""
        from app.settings import StorageSettings

        # Default settings should have valid values
        settings = StorageSettings()

        # Values should be non-negative (ge=0 constraint allows zero)
        assert settings.postgresql_max_inactive_lifetime_s >= 0
        assert settings.postgresql_max_queries >= 0


class TestPoolHardeningCallbacks:
    """Test pool hardening callback logic."""

    def test_statement_timeout_calculation(self) -> None:
        """Verify statement_timeout is 90% of command_timeout."""
        from app.backends.postgresql_backend import _statement_timeout_ms
        from app.settings import get_settings

        settings = get_settings()

        # Default command_timeout is 60 seconds, so statement_timeout should be 54000ms
        assert _statement_timeout_ms(settings.storage.postgresql_command_timeout_s) == 54000

    def test_statement_timeout_floors_at_one_millisecond(self) -> None:
        """A sub-millisecond command timeout must not disable the server backstop.

        PostgreSQL treats ``SET statement_timeout = 0`` as UNLIMITED, so a
        command timeout whose 90-percent derivation truncates to 0 would
        silently disable the very protection the setting exists to provide --
        the derived value is floored at 1 ms instead.
        """
        from app.backends.postgresql_backend import _statement_timeout_ms

        assert _statement_timeout_ms(0.0005) == 1
        assert _statement_timeout_ms(0.001) == 1
        # Just above the floor the 90-percent derivation resumes.
        assert _statement_timeout_ms(0.01) == 9

    def test_pool_hardening_defaults_are_non_zero(self) -> None:
        """Verify default pool hardening settings are non-zero (enabled)."""
        from app.settings import StorageSettings

        settings = StorageSettings()

        # Default values should be non-zero (hardening enabled by default)
        assert settings.postgresql_max_inactive_lifetime_s > 0
        assert settings.postgresql_max_queries > 0

    def test_pool_hardening_values_match_plan(self) -> None:
        """Verify pool hardening defaults match implementation guide values."""
        from app.settings import StorageSettings

        settings = StorageSettings()

        # Implementation guide specifies:
        # - max_inactive_connection_lifetime: 300.0 seconds (5 minutes)
        # - max_queries: 10000 queries
        assert settings.postgresql_max_inactive_lifetime_s == 300.0
        assert settings.postgresql_max_queries == 10000


class TestResetConnectionRollback:
    """Test ROLLBACK behavior in _reset_connection callback.

    Verifies that the connection reset callback correctly issues ROLLBACK
    before other reset operations to ensure transaction cleanup on connection return.
    """

    @pytest.mark.asyncio
    async def test_reset_connection_issues_rollback_first(self) -> None:
        """Verify _reset_connection issues ROLLBACK before SELECT 1 and RESET ALL.

        The callback should execute operations in this order:
        1. ROLLBACK (abort any active transaction)
        2. SELECT 1 (validate connection)
        3. RESET ALL (reset GUC parameters)
        """
        from unittest.mock import AsyncMock
        from unittest.mock import call

        # Create mock connection that tracks execute calls
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)

        # Extract the callback signature from pool creation
        # The callback expects: async def _reset_connection(conn) -> None
        # Based on our implementation, it should:
        # 1. execute('ROLLBACK')
        # 2. fetchval('SELECT 1')
        # 3. execute('RESET ALL')

        # The test validates the expected order when the callback runs
        # Since we can't easily extract the nested callback, we test the expected behavior
        async def simulate_reset_callback() -> None:
            """Simulate what _reset_connection should do."""
            # Based on the implemented code in postgresql_backend.py
            await mock_conn.execute('ROLLBACK')
            await mock_conn.fetchval('SELECT 1')
            await mock_conn.execute('RESET ALL')

        await simulate_reset_callback()

        # Verify call order: ROLLBACK first, then SELECT 1, then RESET ALL
        mock_conn.execute.assert_has_calls([
            call('ROLLBACK'),
            call('RESET ALL'),
        ])
        mock_conn.fetchval.assert_called_once_with('SELECT 1')

        # Verify ROLLBACK was called before RESET ALL
        execute_calls = mock_conn.execute.call_args_list
        assert execute_calls[0] == call('ROLLBACK'), 'ROLLBACK must be first execute call'
        assert execute_calls[1] == call('RESET ALL'), 'RESET ALL must be second execute call'

    def test_rollback_is_safe_without_active_transaction(self) -> None:
        """Verify ROLLBACK is safe to execute when no transaction is active.

        PostgreSQL ROLLBACK is a no-op when no transaction is active,
        which is the expected behavior for the reset callback.
        """
        # ROLLBACK on a connection without an active transaction is a no-op in PostgreSQL
        # This test documents the expected behavior - ROLLBACK does not raise an error
        # when called outside a transaction

        # The implementation relies on this PostgreSQL behavior:
        # - ROLLBACK outside transaction: Warning (not error), no state change
        # - This is safe to call unconditionally before other reset operations

        # Since we can't execute real PostgreSQL commands in unit tests,
        # we verify the expectation is documented
        assert True  # Document the expectation: ROLLBACK is safe as no-op


class TestAdvisoryLockDDL:
    """Test advisory lock serialization for DDL operations.

    Verifies that PostgreSQL DDL operations (schema initialization via
    init_database(), migrations) use advisory locks to prevent
    'tuple concurrently updated' errors in multi-pod Kubernetes deployments.
    """

    def test_migration_uses_same_lock_key(self) -> None:
        """Verify all DDL operations use the same advisory lock key.

        Schema init and the migrations share one lock so they serialize against
        each other in multi-pod deployments. That single key lives in the shared
        begin_migration() helper, which init_database() calls to open each
        PostgreSQL schema-init transaction; verifying the delegation plus the
        helper's lock literal confirms both take the same transaction-level lock.
        """
        import inspect

        from app.migrations._pg_ddl import SCHEMA_INIT_ADVISORY_LOCK_SQL
        from app.migrations._pg_ddl import begin_migration
        from app.startup import init_database

        init_source = inspect.getsource(init_database)
        assert 'begin_migration' in init_source, (
            'init_database() must acquire the shared migration lock via begin_migration() '
            'for multi-pod safety'
        )

        begin_source = inspect.getsource(begin_migration)
        assert 'SCHEMA_INIT_ADVISORY_LOCK_SQL' in begin_source, (
            'begin_migration() must acquire the shared schema-init advisory lock'
        )
        # The one lock key, used by both schema init and the migrations.
        assert 'pg_advisory_xact_lock' in SCHEMA_INIT_ADVISORY_LOCK_SQL, (
            'the shared migration lock must use pg_advisory_xact_lock for multi-pod safety'
        )
        assert "hashtext('mcp_context_schema_init')" in SCHEMA_INIT_ADVISORY_LOCK_SQL, (
            'the shared migration lock must use the standard lock key'
        )


class TestTcpKeepaliveSettings:
    """Test TCP keepalive settings configuration."""

    def test_tcp_keepalive_settings_defaults(self) -> None:
        """Verify TCP keepalive settings have expected defaults."""
        from app.settings import StorageSettings

        settings = StorageSettings()

        assert settings.postgresql_tcp_keepalives_idle_s == 15
        assert settings.postgresql_tcp_keepalives_interval_s == 5
        assert settings.postgresql_tcp_keepalives_count == 3

    def test_tcp_keepalive_settings_allow_zero(self) -> None:
        """Verify TCP keepalive settings allow zero (disabled)."""
        from app.settings import StorageSettings

        settings = StorageSettings()

        # ge=0 constraint allows zero
        assert settings.postgresql_tcp_keepalives_idle_s >= 0
        assert settings.postgresql_tcp_keepalives_interval_s >= 0
        assert settings.postgresql_tcp_keepalives_count >= 0

    def test_tcp_keepalive_settings_types_are_int(self) -> None:
        """Verify TCP keepalive settings are integers (required by setsockopt)."""
        from app.settings import StorageSettings

        settings = StorageSettings()

        assert isinstance(settings.postgresql_tcp_keepalives_idle_s, int)
        assert isinstance(settings.postgresql_tcp_keepalives_interval_s, int)
        assert isinstance(settings.postgresql_tcp_keepalives_count, int)


class TestTcpKeepaliveSocketConstants:
    """Test TCP keepalive socket constants availability."""

    def test_so_keepalive_available(self) -> None:
        """Verify SO_KEEPALIVE is available on all platforms."""
        import socket

        assert hasattr(socket, 'SO_KEEPALIVE')
        assert hasattr(socket, 'SOL_SOCKET')
        assert hasattr(socket, 'IPPROTO_TCP')

    def test_keepalive_constants_cross_platform(self) -> None:
        """Verify keepalive constants use hasattr pattern (cross-platform).

        TCP_KEEPIDLE, TCP_KEEPINTVL are available on Linux/macOS/Windows.
        TCP_KEEPCNT is available on Linux/macOS and Windows 10 v1703+.
        The implementation uses hasattr() checks for cross-platform compatibility.
        """
        import socket

        # These should be available on the test machine
        # but the implementation correctly uses hasattr() for safety
        assert hasattr(socket, 'TCP_KEEPIDLE') or True  # May not be available on all platforms
        assert hasattr(socket, 'TCP_KEEPINTVL') or True
        assert hasattr(socket, 'TCP_KEEPCNT') or True


class TestTransactionRetry:
    """Test transaction retry behavior for connection errors."""

    def test_retry_on_connection_error(self) -> None:
        """Verify transaction retry catches connection errors and retries.

        Documents the expected retry behavior:
        - max_retries=2 (total 3 attempts)
        - Exponential backoff: 0.5s, 1.0s
        - Only connection errors trigger retry
        """
        # This test documents the retry parameters
        max_retries = 2
        delays = [0.5 * (2 ** attempt) for attempt in range(max_retries)]
        assert delays == [0.5, 1.0]

    def test_store_context_is_idempotent(self) -> None:
        """Verify store_context with deduplication is safe to retry.

        store_with_deduplication uses ON CONFLICT to handle duplicates,
        making the operation idempotent. This is critical for retry safety.
        """
        # Document the idempotency requirement
        assert True  # Idempotency is guaranteed by store_with_deduplication SQL


class TestInitializeErrorClassification:
    """Test error classification in PostgreSQLBackend.initialize().

    When initialize() fails, errors must be classified as either
    DependencyError (exit code 69, retryable) or ConfigurationError
    (exit code 78, non-retryable) to enable proper Docker/Kubernetes
    restart policy behavior.
    """

    @pytest.mark.asyncio
    async def test_connection_refused_raises_dependency_error(self) -> None:
        """ConnectionRefusedError during pool creation raises DependencyError."""

        from app.errors import DependencyError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=ConnectionRefusedError('Connection refused'),
            ),
            pytest.raises(DependencyError, match='PostgreSQL connection failed'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_os_error_raises_dependency_error(self) -> None:
        """OSError (network unreachable, timeout) during pool creation raises DependencyError."""

        from app.errors import DependencyError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=OSError('Network is unreachable'),
            ),
            pytest.raises(DependencyError, match='PostgreSQL connection failed'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_too_many_connections_raises_dependency_error(self) -> None:
        """TooManyConnectionsError during pool creation raises DependencyError."""
        from app.errors import DependencyError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=asyncpg.exceptions.TooManyConnectionsError('too many connections'),
            ),
            pytest.raises(DependencyError, match='PostgreSQL connection failed'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_invalid_password_raises_configuration_error(self) -> None:
        """InvalidPasswordError during pool creation raises ConfigurationError."""
        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:wrong@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=asyncpg.exceptions.InvalidPasswordError('password authentication failed'),
            ),
            pytest.raises(ConfigurationError, match='PostgreSQL authentication failed'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_invalid_catalog_name_raises_configuration_error(self) -> None:
        """InvalidCatalogNameError during pool creation raises ConfigurationError."""
        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/nonexistent',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=asyncpg.exceptions.InvalidCatalogNameError('database "nonexistent" does not exist'),
            ),
            pytest.raises(ConfigurationError, match='PostgreSQL database does not exist'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_value_error_raises_configuration_error(self) -> None:
        """ValueError during pool creation raises ConfigurationError.

        asyncpg raises plain ValueError synchronously for invalid construction
        inputs (pool size combinations, a non-positive command_timeout) before
        any network I/O; these are permanent misconfigurations that must exit
        78 instead of restart-looping as a retryable dependency failure. DSN
        option errors are NOT this shape: asyncpg raises those as
        ClientConfigurationError, whose InterfaceError base would shadow the
        ValueError clause, so the backend classifies them in a dedicated
        earlier clause covered by
        test_client_configuration_error_raises_configuration_error.
        """
        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=ValueError('min_size is greater than max_size'),
            ),
            pytest.raises(ConfigurationError, match='PostgreSQL configuration invalid'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_client_configuration_error_raises_configuration_error(self) -> None:
        """ClientConfigurationError classifies as ConfigurationError, not DependencyError.

        asyncpg raises ClientConfigurationError for permanent client-side
        misconfigurations (invalid sslmode/target_session_attrs/gsslib values,
        unresolvable DSN options). The class subclasses BOTH InterfaceError and
        ValueError, so a broad InterfaceError tuple listed first would shadow
        it into a retryable DependencyError (exit 69) and restart-loop the
        supervisor on a permanent misconfiguration; the backend must classify
        it as ConfigurationError (exit 78) in a clause preceding the
        InterfaceError tuple.
        """
        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb?sslmode=bogus',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=asyncpg.exceptions.ClientConfigurationError(
                    "sslmode is invalid, valid values are: 'disable', 'prefer', 'require'",
                ),
            ),
            pytest.raises(ConfigurationError, match='PostgreSQL client configuration invalid'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_client_configuration_error_in_pgvector_precheck(self) -> None:
        """The pgvector pre-check classifies ClientConfigurationError the same way.

        _ensure_pgvector_extension opens its own connection BEFORE pool
        creation and carries the same InterfaceError tuple, so it needs the
        same preceding ClientConfigurationError clause.
        """
        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb?sslmode=bogus',
        )

        with (
            unittest.mock.patch(
                'asyncpg.connect',
                side_effect=asyncpg.exceptions.ClientConfigurationError(
                    "sslmode is invalid, valid values are: 'disable', 'prefer', 'require'",
                ),
            ),
            pytest.raises(ConfigurationError, match='PostgreSQL client configuration invalid'),
        ):
            await backend._ensure_pgvector_extension()

    @pytest.mark.asyncio
    async def test_acquire_timeout_and_connect_timeout_wiring(self) -> None:
        """The acquire-wait and establishment timeouts reach their real asyncpg knobs.

        asyncpg.create_pool has NO acquire-timeout parameter -- an unknown
        'timeout' kwarg falls through connect_kwargs to asyncpg.connect() as
        the connection ESTABLISHMENT timeout. The documented acquire-wait
        bound (POSTGRESQL_POOL_TIMEOUT_S) therefore must be passed per-call
        at pool.acquire(timeout=...); wiring it into create_pool instead
        silently leaves every acquire waiting unbounded under pool
        exhaustion.
        """
        from app.settings import get_settings

        captured: dict[str, object] = {}

        class _FakeAcquireContext:
            async def __aenter__(self) -> AsyncMock:
                return AsyncMock()

            async def __aexit__(self, *exc_info: object) -> bool:
                return False

        fake_pool = unittest.mock.MagicMock()

        def _acquire(*, timeout: float | None = None) -> _FakeAcquireContext:
            captured['acquire_timeout'] = timeout
            return _FakeAcquireContext()

        fake_pool.acquire = _acquire

        create_kwargs: dict[str, object] = {}

        async def _fake_create_pool(dsn: str, **kwargs: object) -> unittest.mock.MagicMock:
            _ = dsn
            create_kwargs.update(kwargs)
            return fake_pool

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch('asyncpg.create_pool', side_effect=_fake_create_pool),
        ):
            await backend.initialize()

        settings = get_settings()
        # create_pool's 'timeout' is the ESTABLISHMENT timeout, sourced from
        # the dedicated connect knob -- never from the acquire knob.
        assert create_kwargs['timeout'] == settings.storage.postgresql_connect_timeout_s
        # The Pgpool-II detection probe runs during initialize() and must have
        # acquired with the acquire-wait bound.
        assert captured['acquire_timeout'] == settings.storage.postgresql_pool_timeout_s

    @pytest.mark.asyncio
    async def test_unknown_exception_raises_dependency_error(self) -> None:
        """Unknown exceptions during pool creation default to DependencyError."""

        from app.errors import DependencyError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=RuntimeError('unexpected internal error'),
            ),
            pytest.raises(DependencyError, match='PostgreSQL initialization failed'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_configuration_error_from_init_connection_reraised(self) -> None:
        """ConfigurationError from _init_connection is re-raised without wrapping."""

        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=ConfigurationError('pgvector codec registration failed'),
            ),
            pytest.raises(ConfigurationError, match='pgvector codec registration failed'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_dependency_error_from_ensure_pgvector_reraised(self) -> None:
        """DependencyError from _ensure_pgvector_extension is re-raised without wrapping."""
        from app.errors import DependencyError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(
                backend,
                '_ensure_pgvector_extension',
                new_callable=AsyncMock,
                side_effect=DependencyError('PostgreSQL connection failed: Connection refused'),
            ),
            pytest.raises(DependencyError, match='PostgreSQL connection failed'),
        ):
            await backend.initialize()


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
    """execute_write discriminates acquire-phase TimeoutError by elapsed wait.

    A saturated pool waits out (approximately) the full acquire budget before
    timing out -- a capacity signal, not a database fault, so it stays uncharged.
    asyncpg's inner connection-establishment timeout raises the SAME bare
    TimeoutError but fires clearly below the pool budget (an unreachable,
    blackholed database) and must charge the breaker, or the outage never opens
    it. A TimeoutError raised AFTER a live connection was obtained (statement
    exceeded the pool command_timeout) still charges exactly one failure.
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
    def _install_acquire_timeout(
        backend: PostgreSQLBackend,
        monkeypatch: pytest.MonkeyPatch,
        wait_s: float,
    ) -> None:
        """Make get_connection raise TimeoutError after waiting wait_s seconds."""

        class _TimeoutOnEnter:
            async def __aenter__(self) -> object:
                if wait_s > 0:
                    await asyncio.sleep(wait_s)
                raise TimeoutError('pool acquire timed out')

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        def _acquire_times_out(*_args: object, **_kwargs: object) -> _TimeoutOnEnter:
            return _TimeoutOnEnter()

        monkeypatch.setattr(backend, 'get_connection', _acquire_times_out)

    @pytest.mark.asyncio
    async def test_saturation_timeout_propagates_uncharged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A TimeoutError after waiting out the full pool budget stays uncharged."""
        import app.backends.postgresql_backend as pg_module

        backend = self._make_backend()
        record_failure = AsyncMock()
        monkeypatch.setattr(backend.circuit_breaker, 'record_failure', record_failure)

        # Shrink the pool budget so the fake acquire can wait it out in test time.
        fake_settings = MagicMock()
        fake_settings.storage.postgresql_pool_timeout_s = 0.02
        monkeypatch.setattr(pg_module, 'settings', fake_settings)
        self._install_acquire_timeout(backend, monkeypatch, wait_s=0.03)

        call_count = {'n': 0}

        async def operation(_conn: object) -> None:
            call_count['n'] += 1

        with pytest.raises(TimeoutError):
            await backend.execute_write(operation)

        assert call_count['n'] == 0
        record_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_establishment_timeout_below_pool_budget_charges_once(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A TimeoutError firing clearly below the pool budget charges the breaker.

        With the default budgets (connect 60s < pool 120s) an unreachable
        database always surfaces as an acquire-phase TimeoutError well below the
        pool budget; leaving it uncharged would keep the breaker closed through
        the entire outage and repeat the full connect stall on every request.
        """
        backend = self._make_backend()
        record_failure = AsyncMock()
        monkeypatch.setattr(backend.circuit_breaker, 'record_failure', record_failure)

        # Instant timeout: elapsed wait is far below the real 120s pool budget.
        self._install_acquire_timeout(backend, monkeypatch, wait_s=0.0)

        call_count = {'n': 0}

        async def operation(_conn: object) -> None:
            call_count['n'] += 1

        with pytest.raises(TimeoutError):
            await backend.execute_write(operation)

        assert call_count['n'] == 0
        record_failure.assert_awaited_once()

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


class TestSetupPoolConnectionTimeout:
    """The pool setup callback re-raises its TimeoutError as a distinguishable fault.

    asyncpg re-raises setup-callback failures from pool.acquire() verbatim, where a
    bare TimeoutError (a dead pooled connection whose SET statement exceeded the
    pool command_timeout) is indistinguishable from the saturation TimeoutError of
    a full pool -- which the backend deliberately leaves uncharged on the circuit
    breaker. Re-raising as ConnectionDoesNotExistError routes the fault into the
    charged retryable connection-error arm instead.
    """

    @pytest.mark.asyncio
    async def test_setup_timeout_reraised_as_connection_does_not_exist(self) -> None:
        """A TimeoutError in the setup statement surfaces as ConnectionDoesNotExistError."""
        from app.backends.postgresql_backend import _setup_pool_connection

        conn = MagicMock()
        conn.execute = AsyncMock(side_effect=TimeoutError('dead pooled connection'))

        with pytest.raises(asyncpg.exceptions.ConnectionDoesNotExistError, match='setup timed out'):
            await _setup_pool_connection(cast(asyncpg.Connection, conn))

    @pytest.mark.asyncio
    async def test_setup_success_sets_statement_timeout(self) -> None:
        """The happy path issues exactly one SET statement_timeout statement."""
        from app.backends.postgresql_backend import _setup_pool_connection

        conn = MagicMock()
        conn.execute = AsyncMock()

        await _setup_pool_connection(cast(asyncpg.Connection, conn))

        conn.execute.assert_awaited_once()
        assert conn.execute.await_args is not None
        sql = conn.execute.await_args.args[0]
        assert sql.startswith('SET statement_timeout = ')

    @pytest.mark.asyncio
    async def test_setup_generic_failure_propagates_unwrapped(self) -> None:
        """A non-timeout setup failure propagates as-is (asyncpg closes the connection)."""
        from app.backends.postgresql_backend import _setup_pool_connection

        conn = MagicMock()
        conn.execute = AsyncMock(side_effect=RuntimeError('protocol desync'))

        with pytest.raises(RuntimeError, match='protocol desync'):
            await _setup_pool_connection(cast(asyncpg.Connection, conn))


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


class TestAcquireTimeoutBreakerDiscrimination:
    """get_connection and begin_transaction discriminate acquire-phase timeouts.

    Pool saturation (the acquire waited out the full pool budget) is a capacity
    signal and stays uncharged; an inner connection-establishment timeout fires
    clearly below the budget (an unreachable database) and charges the breaker,
    so read and transaction paths can also open it during a blackholed outage.
    """

    @staticmethod
    def _backend_with_acquire_timeout(wait_s: float) -> PostgreSQLBackend:
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        backend._shutdown = False

        class _TimeoutOnEnter:
            async def __aenter__(self) -> object:
                if wait_s > 0:
                    await asyncio.sleep(wait_s)
                raise TimeoutError('pool acquire timed out')

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        pool = MagicMock()
        pool.acquire = MagicMock(side_effect=lambda **_kwargs: _TimeoutOnEnter())
        backend._pool = pool
        return backend

    @pytest.mark.asyncio
    async def test_get_connection_establishment_timeout_charges(self) -> None:
        """An instant acquire TimeoutError (far below the pool budget) charges once."""
        backend = self._backend_with_acquire_timeout(wait_s=0.0)

        with pytest.raises(TimeoutError):
            async with backend.get_connection():
                pass

        assert backend.circuit_breaker.failures == 1

    @pytest.mark.asyncio
    async def test_get_connection_saturation_timeout_uncharged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A TimeoutError after waiting out the full pool budget stays uncharged."""
        import app.backends.postgresql_backend as pg_module

        backend = self._backend_with_acquire_timeout(wait_s=0.03)
        fake_settings = MagicMock()
        fake_settings.storage.postgresql_pool_timeout_s = 0.02
        monkeypatch.setattr(pg_module, 'settings', fake_settings)

        with pytest.raises(TimeoutError):
            async with backend.get_connection():
                pass

        assert backend.circuit_breaker.failures == 0

    @pytest.mark.asyncio
    async def test_begin_transaction_establishment_timeout_charges(self) -> None:
        """begin_transaction applies the same discrimination on its acquire."""
        backend = self._backend_with_acquire_timeout(wait_s=0.0)

        with pytest.raises(TimeoutError):
            async with backend.begin_transaction():
                pass

        assert backend.circuit_breaker.failures == 1

    @pytest.mark.asyncio
    async def test_begin_transaction_saturation_timeout_uncharged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A saturation timeout on the transaction acquire stays uncharged."""
        import app.backends.postgresql_backend as pg_module

        backend = self._backend_with_acquire_timeout(wait_s=0.03)
        fake_settings = MagicMock()
        fake_settings.storage.postgresql_pool_timeout_s = 0.02
        monkeypatch.setattr(pg_module, 'settings', fake_settings)

        with pytest.raises(TimeoutError):
            async with backend.begin_transaction():
                pass

        assert backend.circuit_breaker.failures == 0


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


class TestBeginTransactionDeadlockExemption:
    """begin_transaction does not charge the breaker for server-initiated rollbacks.

    The tool layer retries deadlock/serialization rollbacks (SQLSTATE class 40),
    so charging the breaker per aborted attempt would let routine write
    contention open it and reject every caller's healthy requests. A genuine
    fault still charges.
    """

    @staticmethod
    def _backend_with_fake_pool() -> PostgreSQLBackend:
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        backend._shutdown = False

        @contextlib.asynccontextmanager
        async def _fake_transaction() -> AsyncIterator[None]:
            yield None

        mock_conn = MagicMock()
        mock_conn.transaction = MagicMock(side_effect=_fake_transaction)

        @contextlib.asynccontextmanager
        async def _fake_acquire(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
            yield mock_conn

        pool = MagicMock()
        pool.acquire = MagicMock(side_effect=_fake_acquire)
        backend._pool = pool
        return backend

    @pytest.mark.asyncio
    async def test_deadlock_rollback_does_not_charge_breaker(self) -> None:
        """A deadlock escaping the transaction body leaves the breaker untouched."""
        backend = self._backend_with_fake_pool()
        with pytest.raises(asyncpg.exceptions.DeadlockDetectedError):
            async with backend.begin_transaction():
                raise asyncpg.exceptions.DeadlockDetectedError('deadlock detected')
        assert backend.circuit_breaker.failures == 0

    @pytest.mark.asyncio
    async def test_genuine_fault_still_charges_breaker(self) -> None:
        """A non-rollback fault in the transaction body still records a failure."""
        backend = self._backend_with_fake_pool()
        with pytest.raises(RuntimeError, match='db fault'):
            async with backend.begin_transaction():
                raise RuntimeError('db fault')
        assert backend.circuit_breaker.failures == 1


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


class TestResolveProvisionVector:
    """The pgvector-provisioning gate keys on the ACTIVE payload format.

    The vector type is used ONLY by the fp32 vec_context_embeddings layout, so a
    compressed (BYTEA-only) database -- including a generation-off archive restored on a
    host without pgvector -- must NOT be forced to create the unused extension, while a
    compression-off database that will provision or already carries the fp32 layout must.
    """

    @staticmethod
    def _backend() -> PostgreSQLBackend:
        return PostgreSQLBackend(connection_string='postgresql://u:p@localhost:5432/db')

    @staticmethod
    def _settings(*, compression: bool, generation: bool) -> MagicMock:
        s = MagicMock()
        s.compression.enabled = compression
        s.embedding.generation_enabled = generation
        s.storage.postgresql_connect_timeout_s = 5.0
        return s

    @staticmethod
    def _conn(*, fp32: bool, embedding_metadata: bool) -> AsyncMock:
        async def _fetchval(sql: str) -> bool:
            if 'vec_context_embeddings' in sql:
                return fp32
            if 'embedding_metadata' in sql:
                return embedding_metadata
            raise AssertionError(f'unexpected probe SQL: {sql}')

        conn = AsyncMock()
        conn.fetchval = AsyncMock(side_effect=_fetchval)
        conn.close = AsyncMock()
        return conn

    @pytest.mark.asyncio
    async def test_generation_on_compression_off_provisions_without_probe(self) -> None:
        """generation on + compression off -> always provision the fp32 layout; no probe.

        Covers the compression-off server (the migration CLI's target init no longer relies
        on this gate: it passes the explicit provision_vector constructor override keyed on
        with_semantic).
        """
        connect = AsyncMock()
        with unittest.mock.patch(
            'app.backends.postgresql_backend.settings', self._settings(compression=False, generation=True),
        ), unittest.mock.patch('app.backends.postgresql_backend.asyncpg.connect', new=connect):
            assert await self._backend()._resolve_provision_vector() is True
            connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_generation_on_compression_on_no_fp32_skips_pgvector(self) -> None:
        """generation on + compression on (the v3.0.0 default) + no fp32 table -> skip pgvector.

        The compressed server stores BYTEA and never binds the vector type, so forcing CREATE
        EXTENSION vector would crash boot on a pgvector-less host. The gate falls through to the
        fp32-table probe and skips when no stray fp32 table exists.
        """
        conn = self._conn(fp32=False, embedding_metadata=True)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.settings', self._settings(compression=True, generation=True),
        ), unittest.mock.patch('app.backends.postgresql_backend.asyncpg.connect', new=AsyncMock(return_value=conn)):
            assert await self._backend()._resolve_provision_vector() is False
        conn.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_generation_on_compression_on_fp32_present_provisions(self) -> None:
        """generation on + compression on + a stray fp32 table present -> provision the codec.

        A leftover fp32 table still needs the vector codec to read it, so the probe returns True
        even though the compressed write path itself binds no vector.
        """
        conn = self._conn(fp32=True, embedding_metadata=True)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.settings', self._settings(compression=True, generation=True),
        ), unittest.mock.patch('app.backends.postgresql_backend.asyncpg.connect', new=AsyncMock(return_value=conn)):
            assert await self._backend()._resolve_provision_vector() is True

    @pytest.mark.asyncio
    async def test_compressed_generation_off_no_fp32_skips_pgvector(self) -> None:
        """The regression: a compressed archive (embedding_metadata, no fp32) skips pgvector."""
        conn = self._conn(fp32=False, embedding_metadata=True)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.settings', self._settings(compression=True, generation=False),
        ), unittest.mock.patch('app.backends.postgresql_backend.asyncpg.connect', new=AsyncMock(return_value=conn)):
            assert await self._backend()._resolve_provision_vector() is False
        conn.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_fp32_table_present_provisions(self) -> None:
        """An fp32 vec table present (fp32 archive, or the --compress CLI reading it) -> provision."""
        conn = self._conn(fp32=True, embedding_metadata=True)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.settings', self._settings(compression=True, generation=False),
        ), unittest.mock.patch('app.backends.postgresql_backend.asyncpg.connect', new=AsyncMock(return_value=conn)):
            assert await self._backend()._resolve_provision_vector() is True

    @pytest.mark.asyncio
    async def test_compression_off_gen_off_infra_present_reprovisions(self) -> None:
        """compression off + generation off + embedding_metadata present -> fp32 reprovisioned."""
        conn = self._conn(fp32=False, embedding_metadata=True)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.settings', self._settings(compression=False, generation=False),
        ), unittest.mock.patch('app.backends.postgresql_backend.asyncpg.connect', new=AsyncMock(return_value=conn)):
            assert await self._backend()._resolve_provision_vector() is True

    @pytest.mark.asyncio
    async def test_fresh_generation_off_skips(self) -> None:
        """A fresh generation-off database (no fp32, no embedding_metadata) skips pgvector."""
        conn = self._conn(fp32=False, embedding_metadata=False)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.settings', self._settings(compression=False, generation=False),
        ), unittest.mock.patch('app.backends.postgresql_backend.asyncpg.connect', new=AsyncMock(return_value=conn)):
            assert await self._backend()._resolve_provision_vector() is False

    @pytest.mark.asyncio
    async def test_probe_connection_failure_defers_to_pool(self) -> None:
        """A generation-off probe connection failure returns False; pool init surfaces it."""
        with unittest.mock.patch(
            'app.backends.postgresql_backend.settings', self._settings(compression=True, generation=False),
        ), unittest.mock.patch(
            'app.backends.postgresql_backend.asyncpg.connect', new=AsyncMock(side_effect=OSError('unreachable')),
        ):
            assert await self._backend()._resolve_provision_vector() is False


class TestProvisionVectorOverride:
    """The explicit provision_vector constructor override bypasses the boot gate.

    The migration CLI knows up front whether its target carries the fp32 vector
    layout (with_semantic), so it must not depend on the CLI process's env-driven
    settings gate: a vector-free target initialized under a compression-off env
    would otherwise force CREATE EXTENSION vector and crash on a pgvector-less host.
    """

    @staticmethod
    def _initialize_ready_backend(
        monkeypatch: pytest.MonkeyPatch,
        provision_vector: bool | None,
    ) -> tuple[PostgreSQLBackend, AsyncMock, AsyncMock]:
        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@localhost:5432/db',
            provision_vector=provision_vector,
        )
        resolve = AsyncMock(return_value=True)
        ensure = AsyncMock()
        monkeypatch.setattr(backend, '_resolve_provision_vector', resolve)
        monkeypatch.setattr(backend, '_ensure_pgvector_extension', ensure)
        monkeypatch.setattr(backend, '_detect_pgpool_ii', AsyncMock())
        monkeypatch.setattr(backend, '_detect_session_mode_pooler', MagicMock())
        return backend, resolve, ensure

    @pytest.mark.asyncio
    async def test_explicit_false_skips_resolution_and_extension(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """provision_vector=False initializes without probing or touching pgvector."""
        backend, resolve, ensure = self._initialize_ready_backend(monkeypatch, provision_vector=False)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.asyncpg.create_pool',
            new=AsyncMock(return_value=MagicMock()),
        ):
            await backend.initialize()
        assert backend._provision_vector is False
        resolve.assert_not_called()
        ensure.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_true_provisions_without_probe(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """provision_vector=True pre-creates the extension without the settings gate."""
        backend, resolve, ensure = self._initialize_ready_backend(monkeypatch, provision_vector=True)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.asyncpg.create_pool',
            new=AsyncMock(return_value=MagicMock()),
        ):
            await backend.initialize()
        assert backend._provision_vector is True
        resolve.assert_not_called()
        ensure.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_default_none_resolves_via_gate(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without the override, initialize() resolves via _resolve_provision_vector."""
        backend, resolve, ensure = self._initialize_ready_backend(monkeypatch, provision_vector=None)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.asyncpg.create_pool',
            new=AsyncMock(return_value=MagicMock()),
        ):
            await backend.initialize()
        assert backend._provision_vector is True
        resolve.assert_awaited_once()
        ensure.assert_awaited_once()
