"""Unit tests for PostgreSQL backend implementation.

This module tests PostgreSQL backend functionality without requiring
a PostgreSQL database connection - focusing on unit tests for
configuration and connection string handling.
"""


from app.backends.postgresql_backend import PostgreSQLBackend


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
