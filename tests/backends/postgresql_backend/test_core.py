"""Tests for app/backends/postgresql_backend/core.py: backend identity and connection-string building."""

import pytest

from app.backends.postgresql_backend import PostgreSQLBackend
from tests.helpers import rebind_package_settings


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
        rebind_package_settings(monkeypatch, pg_module, AppSettings())
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

    def test_already_bracketed_ipv6_host_is_not_double_bracketed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A host already in RFC 3986 bracket form is left as-is (idempotent bracketing).

        The colon-presence check alone also matches '[::1]', so wrapping again
        produced '[[::1]]', which asyncpg rejects at DSN parse with a plain
        ValueError that never names the bracket cause. The bracketed spelling is
        a natural copy-paste from URI-style examples and connected fine before
        automatic bracketing existed, so it must keep working.
        """
        conn_str = self._built_connection_string(monkeypatch, '[::1]')
        assert '@[::1]:' in conn_str
        assert '[[' not in conn_str

    def test_hostname_without_colon_is_not_bracketed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A plain hostname interpolates without brackets."""
        conn_str = self._built_connection_string(monkeypatch, 'db.example.com')
        assert '@db.example.com:' in conn_str
        assert '[' not in conn_str
