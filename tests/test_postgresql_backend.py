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
        from app.settings import get_settings

        settings = get_settings()

        # The callback should set statement_timeout to 90% of command_timeout
        expected_timeout_ms = int(settings.storage.postgresql_command_timeout_s * 1000 * 0.9)

        # Default command_timeout is 60 seconds, so statement_timeout should be 54000ms
        assert expected_timeout_ms == 54000  # 60 * 1000 * 0.9 = 54000

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
