"""Tests for app/backends/postgresql_backend/session.py: connection parameters.

Covers the asyncpg connect kwargs (an empty startup packet), the session GUCs
delivered as SET statements, identifier quoting, and the statement-timeout
derivation.
"""

import pytest


class TestBuildAsyncpgConnectKwargs:
    """Unit tests for the shared asyncpg connect-kwargs builder.

    The same helper feeds both the connection pool and the migration CLI, so
    its output is the single source of truth for the startup packet and the session
    parameters every connection applies. These tests are DB-free.
    """

    def test_search_path_quotes_non_default_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-default schema is double-quoted and followed by public."""
        from app.backends.postgresql_backend.session import session_guc_set_statements
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_SCHEMA', 'My.Schema')
        assert session_guc_set_statements(AppSettings())[0] == 'SET search_path = "My.Schema", public'

    def test_search_path_public_is_benign_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default public schema resolves to the benign no-op form."""
        from app.backends.postgresql_backend.session import session_guc_set_statements
        from app.settings import AppSettings

        monkeypatch.delenv('POSTGRESQL_SCHEMA', raising=False)
        assert session_guc_set_statements(AppSettings())[0] == 'SET search_path = "public", public'

    def test_search_path_escapes_embedded_double_quote(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A schema name containing a double quote is escaped by doubling it.

        An unescaped ``f'"{schema}", public'`` would produce a malformed quoted
        identifier (and thus a malformed search_path) for a schema like
        ``my"schema``. PostgreSQL escapes an embedded double quote by doubling it.
        """
        from app.backends.postgresql_backend.session import session_guc_set_statements
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_SCHEMA', 'my"schema')
        assert session_guc_set_statements(AppSettings())[0] == 'SET search_path = "my""schema", public'

    def test_statement_cache_zero_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """statement_cache_size=0 (transaction-pooler mode) propagates verbatim."""
        from app.backends.postgresql_backend.session import build_asyncpg_connect_kwargs
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_STATEMENT_CACHE_SIZE', '0')
        kwargs = build_asyncpg_connect_kwargs(AppSettings())
        assert kwargs['statement_cache_size'] == 0

    def test_statement_cache_default_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default prepared-statement cache size is forwarded unchanged."""
        from app.backends.postgresql_backend.session import build_asyncpg_connect_kwargs
        from app.settings import AppSettings

        monkeypatch.delenv('POSTGRESQL_STATEMENT_CACHE_SIZE', raising=False)
        kwargs = build_asyncpg_connect_kwargs(AppSettings())
        assert kwargs['statement_cache_size'] == 100

    def test_startup_packet_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No PostgreSQL startup parameter is sent at all.

        asyncpg delivers server_settings as startup-packet parameters, and PgBouncer
        REFUSES a connection carrying any parameter outside its small always-allowed
        set ('unsupported startup parameter: <name>'), so a single entry here makes
        the documented transaction-mode deployment unable to connect with a stock
        configuration -- and a pooler that strips instead of refusing silently drops
        search_path, resolving every query against the wrong schema.
        """
        from app.backends.postgresql_backend.session import build_asyncpg_connect_kwargs
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_SCHEMA', 'mcp')
        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_IDLE_S', '15')
        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_INTERVAL_S', '5')
        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_COUNT', '3')
        assert 'server_settings' not in build_asyncpg_connect_kwargs(AppSettings())

    def test_session_gucs_are_set_statements(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every session parameter is delivered as a SET statement."""
        from app.backends.postgresql_backend.session import session_guc_set_statements
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_SCHEMA', 'mcp')
        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_IDLE_S', '15')
        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_INTERVAL_S', '5')
        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_COUNT', '3')
        assert session_guc_set_statements(AppSettings()) == [
            'SET search_path = "mcp", public',
            'SET extra_float_digits = 1',
            'SET tcp_keepalives_idle = 15',
            'SET tcp_keepalives_interval = 5',
            'SET tcp_keepalives_count = 3',
        ]

    def test_tcp_keepalive_gucs_absent_when_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TCP keepalive GUCs are omitted entirely when set to 0 (disabled)."""
        from app.backends.postgresql_backend.session import session_guc_set_statements
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_IDLE_S', '0')
        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_INTERVAL_S', '0')
        monkeypatch.setenv('POSTGRESQL_TCP_KEEPALIVES_COUNT', '0')
        statements = session_guc_set_statements(AppSettings())
        assert not [statement for statement in statements if 'tcp_keepalives' in statement]

    @pytest.mark.asyncio
    async def test_connection_setup_applies_every_session_guc(self) -> None:
        """The pool's setup callback re-applies them on every acquire, after RESET ALL.

        This is what makes delivering search_path and extra_float_digits as statements
        safe: RESET ALL on release wipes them, and setup runs again before the next
        caller ever sees the connection.
        """
        from unittest.mock import AsyncMock

        from app.backends.postgresql_backend.pool_callbacks import setup_pool_connection

        conn = AsyncMock()
        await setup_pool_connection(conn)

        statement = conn.execute.await_args.args[0]
        assert 'SET statement_timeout' in statement
        assert 'SET search_path' in statement
        assert 'SET extra_float_digits' in statement
        assert 'SET tcp_keepalives_idle' in statement
        # One round trip: the whole session configuration goes out together.
        assert conn.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_one_off_connection_applies_the_same_session_gucs(self) -> None:
        """A connection opened outside the pool configures itself identically.

        The provisioning probes and both migration CLIs dial directly, so they never
        run the pool's setup callback; without this they would read and write through
        the server's default search_path instead of POSTGRESQL_SCHEMA.
        """
        from unittest.mock import AsyncMock

        from app.backends.postgresql_backend.session import apply_session_gucs

        conn = AsyncMock()
        await apply_session_gucs(conn)

        statement = conn.execute.await_args.args[0]
        assert 'SET search_path' in statement
        assert 'SET extra_float_digits' in statement
        assert conn.execute.await_count == 1


class TestQuotePgIdentifier:
    """Unit tests for the shared PostgreSQL quoted-identifier helper.

    The same helper quotes the schema for BOTH the connection search_path and the
    migration CLI's CREATE SCHEMA DDL, so they cannot drift on an embedded double
    quote. These tests are DB-free.
    """

    def test_plain_identifier_quoted(self) -> None:
        """A plain identifier is wrapped in double quotes."""
        from app.backends.postgresql_backend.session import quote_pg_identifier

        assert quote_pg_identifier('public') == '"public"'

    def test_mixed_case_and_dot_preserved(self) -> None:
        """Mixed case and dots are preserved verbatim inside the quotes."""
        from app.backends.postgresql_backend.session import quote_pg_identifier

        assert quote_pg_identifier('My.Schema') == '"My.Schema"'

    def test_embedded_double_quote_doubled(self) -> None:
        """An embedded double quote is escaped by doubling it (a valid quoted identifier)."""
        from app.backends.postgresql_backend.session import quote_pg_identifier

        assert quote_pg_identifier('we"ird') == '"we""ird"'

    def test_search_path_uses_the_same_quoting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """session_guc_set_statements quotes the schema via quote_pg_identifier, so the
        connection search_path and any schema-qualified DDL escape it identically."""
        from app.backends.postgresql_backend.session import quote_pg_identifier
        from app.backends.postgresql_backend.session import session_guc_set_statements
        from app.settings import AppSettings

        monkeypatch.setenv('POSTGRESQL_SCHEMA', 'we"ird')
        search_path = session_guc_set_statements(AppSettings())[0]
        assert search_path == f'SET search_path = {quote_pg_identifier("we\"ird")}, public'
        assert search_path == 'SET search_path = "we""ird", public'


class TestPoolHardeningCallbacks:
    """Test pool hardening callback logic."""

    def test_statement_timeout_calculation(self) -> None:
        """Verify statement_timeout is 90% of command_timeout."""
        from app.backends.postgresql_backend.session import statement_timeout_ms
        from app.settings import get_settings

        settings = get_settings()

        # Default command_timeout is 60 seconds, so statement_timeout should be 54000ms
        assert statement_timeout_ms(settings.storage.postgresql_command_timeout_s) == 54000

    def test_statement_timeout_floors_at_one_millisecond(self) -> None:
        """A sub-millisecond command timeout must not disable the server backstop.

        PostgreSQL treats ``SET statement_timeout = 0`` as UNLIMITED, so a
        command timeout whose 90-percent derivation truncates to 0 would
        silently disable the very protection the setting exists to provide --
        the derived value is floored at 1 ms instead.
        """
        from app.backends.postgresql_backend.session import statement_timeout_ms

        assert statement_timeout_ms(0.0005) == 1
        assert statement_timeout_ms(0.001) == 1
        # Just above the floor the 90-percent derivation resumes.
        assert statement_timeout_ms(0.01) == 9
