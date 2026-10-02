"""Tests for app/settings/storage.py: StorageSettings validation.

Ensures the storage validators fail fast with clear error messages
for invalid configuration values.
"""

import pytest
from pydantic import ValidationError

from tests.helpers import env_var


class TestStorageImageSizeLimits:
    """MAX_IMAGE_SIZE_MB / MAX_TOTAL_SIZE_MB must be at least 1 megabyte."""

    def test_defaults_are_positive(self) -> None:
        from app.settings.storage import StorageSettings

        settings = StorageSettings()
        assert settings.max_image_size_mb == 10
        assert settings.max_total_size_mb == 100

    def test_zero_image_size_rejected(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('MAX_IMAGE_SIZE_MB', '0'), pytest.raises(ValidationError):
            StorageSettings()

    def test_negative_total_size_rejected(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('MAX_TOTAL_SIZE_MB', '-5'), pytest.raises(ValidationError):
            StorageSettings()


class TestStoragePoolLimits:
    """POOL_MAX_READERS must be at least 1.

    It sizes an asyncio.Semaphore: a value of 0 would start it locked (every
    reader blocks forever, a silent deadlock) and a negative value raises an
    opaque ValueError deep in pool init, so both must be rejected cleanly at the
    configuration boundary like every peer concurrency cap.
    """

    def test_default_is_positive(self) -> None:
        from app.settings.storage import StorageSettings

        assert StorageSettings().pool_max_readers == 8

    def test_zero_readers_rejected(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('POOL_MAX_READERS', '0'), pytest.raises(ValidationError):
            StorageSettings()

    def test_negative_readers_rejected(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('POOL_MAX_READERS', '-1'), pytest.raises(ValidationError):
            StorageSettings()


class TestPostgresqlPortBounds:
    """POSTGRESQL_PORT is bounded to a valid TCP port range at the config boundary.

    A port typo (0, negative, >65535) that passes pydantic surfaces only at the
    socket layer as an OSError the backend classifies as a retryable
    DependencyError (exit 69, supervisor restart-loops forever) instead of a
    permanent ConfigurationError (exit 78). The ge/le bound rejects it up front,
    mirroring FASTMCP_PORT.
    """

    def test_default_port_is_valid(self) -> None:
        from app.settings.storage import StorageSettings

        assert StorageSettings().postgresql_port == 5432

    def test_zero_port_rejected(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('POSTGRESQL_PORT', '0'), pytest.raises(ValidationError):
            StorageSettings()

    def test_negative_port_rejected(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('POSTGRESQL_PORT', '-1'), pytest.raises(ValidationError):
            StorageSettings()

    def test_above_max_port_rejected(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('POSTGRESQL_PORT', '70000'), pytest.raises(ValidationError):
            StorageSettings()

    def test_boundary_ports_accepted(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('POSTGRESQL_PORT', '1'):
            assert StorageSettings().postgresql_port == 1
        with env_var('POSTGRESQL_PORT', '65535'):
            assert StorageSettings().postgresql_port == 65535


class TestPostgresqlPoolLimits:
    """POSTGRESQL_POOL_MIN / POSTGRESQL_POOL_MAX bounds at the config boundary.

    Any size combination asyncpg would reject (zero or negative max, negative
    min, min above max) passes pydantic without these guards but only fails
    later at asyncpg pool creation with a plain ValueError, which the
    backend's broad exception handler misclassifies as a retryable
    DependencyError (supervisor restart loop) instead of a permanent
    configuration error. min may be 0 (an empty warm pool is valid) but never
    negative, and never above max.
    """

    def test_defaults_are_valid(self) -> None:
        from app.settings.storage import StorageSettings

        settings = StorageSettings()
        assert settings.postgresql_pool_min == 2
        assert settings.postgresql_pool_max == 20

    def test_zero_pool_max_rejected(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('POSTGRESQL_POOL_MAX', '0'), pytest.raises(ValidationError):
            StorageSettings()

    def test_negative_pool_min_rejected(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('POSTGRESQL_POOL_MIN', '-1'), pytest.raises(ValidationError):
            StorageSettings()

    def test_zero_pool_min_accepted(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('POSTGRESQL_POOL_MIN', '0'):
            assert StorageSettings().postgresql_pool_min == 0

    def test_pool_min_above_max_rejected(self) -> None:
        """min above max would reach asyncpg as ValueError('min_size is greater than max_size')."""
        from app.settings.storage import StorageSettings

        with (
            env_var('POSTGRESQL_POOL_MIN', '2'),
            env_var('POSTGRESQL_POOL_MAX', '1'),
            pytest.raises(ValidationError, match='must not exceed'),
        ):
            StorageSettings()

    def test_pool_min_equal_max_accepted(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('POSTGRESQL_POOL_MIN', '5'), env_var('POSTGRESQL_POOL_MAX', '5'):
            settings = StorageSettings()
        assert settings.postgresql_pool_min == 5
        assert settings.postgresql_pool_max == 5

    def test_connect_timeout_defaults_to_asyncpg_default(self) -> None:
        """The establishment timeout is a separate knob from the acquire timeout."""
        from app.settings.storage import StorageSettings

        settings = StorageSettings()
        assert settings.postgresql_connect_timeout_s == 60.0
        assert settings.postgresql_pool_timeout_s == 120.0

    def test_connect_timeout_at_or_above_pool_timeout_rejected(self) -> None:
        """A connect budget the acquire deadline always pre-empts is a misconfiguration.

        asyncpg wraps the queue wait AND the connect callable in one
        wait_for(timeout=POSTGRESQL_POOL_TIMEOUT_S), so a connect budget at or above
        the acquire budget can never elapse on its own: the acquire deadline wins and
        CANCELS the dial, which destroys the typed establishment error the fault
        classification depends on. Nothing else rejects the ordering, so an operator
        could set it silently.
        """
        from app.settings.storage import StorageSettings

        with (
            env_var('POSTGRESQL_CONNECT_TIMEOUT_S', '180'),
            env_var('POSTGRESQL_POOL_TIMEOUT_S', '120'),
            pytest.raises(ValidationError, match='must be below'),
        ):
            StorageSettings()

        with (
            env_var('POSTGRESQL_CONNECT_TIMEOUT_S', '60'),
            env_var('POSTGRESQL_POOL_TIMEOUT_S', '60'),
            pytest.raises(ValidationError, match='must be below'),
        ):
            StorageSettings()

    def test_connect_timeout_below_pool_timeout_accepted(self) -> None:
        """A correctly ordered pair passes, including a tuned-down acquire budget."""
        from app.settings.storage import StorageSettings

        with (
            env_var('POSTGRESQL_CONNECT_TIMEOUT_S', '20'),
            env_var('POSTGRESQL_POOL_TIMEOUT_S', '60'),
        ):
            settings = StorageSettings()
        assert settings.postgresql_connect_timeout_s == 20.0
        assert settings.postgresql_pool_timeout_s == 60.0

    @pytest.mark.parametrize(
        ('env_name', 'value'),
        [
            ('POOL_CONNECTION_TIMEOUT_S', '0'),
            ('POOL_IDLE_TIMEOUT_S', '-1'),
            ('POOL_HEALTH_CHECK_INTERVAL_S', '0'),
            ('SHUTDOWN_TIMEOUT_S', '0'),
            ('SHUTDOWN_TIMEOUT_TEST_S', '-2'),
            ('QUEUE_TIMEOUT_S', '-1'),
            ('QUEUE_TIMEOUT_TEST_S', '0'),
            ('POSTGRESQL_POOL_TIMEOUT_S', '0'),
            ('POSTGRESQL_CONNECT_TIMEOUT_S', '-1'),
            ('POSTGRESQL_COMMAND_TIMEOUT_S', '0'),
            ('CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S', '-5'),
            ('CIRCUIT_BREAKER_FAILURE_THRESHOLD', '0'),
            ('CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS', '0'),
            ('RETRY_MAX_RETRIES', '-1'),
            # 0 is the one value that disables EVERY database write: both
            # backends run `for attempt in range(max_retries)`, so the loop
            # body never executes and the post-loop tail raises without a
            # single database attempt.
            ('RETRY_MAX_RETRIES', '0'),
            ('RETRY_BASE_DELAY_S', '-0.5'),
            ('RETRY_MAX_DELAY_S', '-1'),
            ('RETRY_BACKOFF_FACTOR', '0.5'),
            ('SQLITE_BUSY_TIMEOUT_MS', '-100'),
        ],
    )
    def test_non_positive_timeout_and_bound_values_rejected(self, env_name: str, value: str) -> None:
        """Out-of-bound timing values are rejected at the configuration boundary.

        A zero or negative timeout passes float parsing but produces a
        permanently broken runtime: QUEUE_TIMEOUT_S feeds asyncio.wait in the
        write-queue processor loop where a non-positive value busy-spins a
        core, and a non-positive asyncpg timeout raises an immediate
        TimeoutError classified as a retryable dependency failure -- the same
        restart-loop-on-permanent-misconfiguration class the pool-size bounds
        close.
        """
        from app.settings.storage import StorageSettings

        with env_var(env_name, value), pytest.raises(ValidationError):
            StorageSettings()

    @pytest.mark.parametrize(
        ('env_name', 'value'),
        [
            ('RETRY_MAX_RETRIES', '1'),
            ('RETRY_BASE_DELAY_S', '0'),
            ('RETRY_BACKOFF_FACTOR', '1'),
            ('SQLITE_BUSY_TIMEOUT_MS', '0'),
        ],
    )
    def test_boundary_timing_values_accepted(self, env_name: str, value: str) -> None:
        """Documented boundary values (single attempt, no delay, flat backoff) stay valid."""
        from app.settings.storage import StorageSettings

        with env_var(env_name, value):
            StorageSettings()


class TestSqlitePragmaValidation:
    """SQLITE_* pragma arguments are checked against what SQLite actually accepts.

    SQLite does not reject an unrecognized pragma argument: it silently keeps the
    current or default value, and the backend never reads the applied value back.
    So a typo produced a server that boots clean and reports healthy while running
    with different durability or concurrency than the operator configured --
    SQLITE_SYNCHRONOUS=FULLL runs at NORMAL (a power loss can discard transactions
    believed to be fsynced), and SQLITE_JOURNAL_MODE=wal-mode leaves a fresh
    database in DELETE mode (every write takes an exclusive lock).
    """

    @pytest.mark.parametrize(
        ('env_name', 'value'),
        [
            ('SQLITE_JOURNAL_MODE', 'WAL2'),
            ('SQLITE_JOURNAL_MODE', 'wal-mode'),
            ('SQLITE_SYNCHRONOUS', 'FULLL'),
            ('SQLITE_SYNCHRONOUS', 'typo'),
            ('SQLITE_TEMP_STORE', 'MEMORYY'),
            ('SQLITE_WAL_CHECKPOINT', 'PASIVE'),
        ],
    )
    def test_unrecognized_pragma_argument_rejected(self, env_name: str, value: str) -> None:
        """A value SQLite would silently ignore fails at the configuration boundary."""
        from app.settings.storage import StorageSettings

        with env_var(env_name, value), pytest.raises(ValidationError, match='SQLite'):
            StorageSettings()

    @pytest.mark.parametrize(
        ('env_name', 'value', 'expected'),
        [
            ('SQLITE_JOURNAL_MODE', 'wal', 'WAL'),
            ('SQLITE_JOURNAL_MODE', 'Delete', 'DELETE'),
            ('SQLITE_SYNCHRONOUS', 'full', 'FULL'),
            ('SQLITE_SYNCHRONOUS', '2', '2'),
            ('SQLITE_TEMP_STORE', 'memory', 'MEMORY'),
            ('SQLITE_TEMP_STORE', '0', '0'),
            ('SQLITE_WAL_CHECKPOINT', 'truncate', 'TRUNCATE'),
        ],
    )
    def test_recognized_pragma_argument_normalized(self, env_name: str, value: str, expected: str) -> None:
        """Every spelling SQLite accepts stays valid and is normalized to upper case."""
        from app.settings.storage import StorageSettings

        with env_var(env_name, value):
            settings = StorageSettings()
        assert getattr(settings, env_name.lower()) == expected

    def test_defaults_are_valid_pragma_arguments(self) -> None:
        """The shipped defaults pass their own validation."""
        from app.settings.storage import StorageSettings

        settings = StorageSettings()
        assert settings.sqlite_journal_mode == 'WAL'
        assert settings.sqlite_synchronous == 'NORMAL'
        assert settings.sqlite_temp_store == 'MEMORY'
        assert settings.sqlite_wal_checkpoint == 'PASSIVE'

    @pytest.mark.parametrize('value', ['5000', '256', '131072', '0'])
    def test_unsupported_page_size_rejected(self, value: str) -> None:
        """A page size SQLite would ignore is rejected instead of silently dropped."""
        from app.settings.storage import StorageSettings

        with env_var('SQLITE_PAGE_SIZE', value), pytest.raises(ValidationError, match='power of two'):
            StorageSettings()

    @pytest.mark.parametrize('value', ['512', '4096', '65536'])
    def test_supported_page_size_accepted(self, value: str) -> None:
        """Every power of two in SQLite's supported range stays valid."""
        from app.settings.storage import StorageSettings

        with env_var('SQLITE_PAGE_SIZE', value):
            assert StorageSettings().sqlite_page_size == int(value)


class TestBlankDbPath:
    """DB_PATH must not be blank when set.

    An empty DB_PATH coerces to Path('.') (a directory) and a whitespace-only value
    to an all-blank path name; either surfaces far from its cause when the SQLite
    backend later opens the file. A blank value almost always means the variable was
    set but left unfilled, so it is rejected at the configuration boundary.
    """

    def test_empty_db_path_rejected(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('DB_PATH', ''), pytest.raises(ValidationError, match='must not be empty'):
            StorageSettings()

    def test_whitespace_db_path_rejected(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('DB_PATH', '   '), pytest.raises(ValidationError, match='must not be empty'):
            StorageSettings()

    def test_valid_db_path_accepted(self) -> None:
        from pathlib import Path

        from app.settings.storage import StorageSettings

        with env_var('DB_PATH', '/tmp/context.db'):
            assert StorageSettings().db_path == Path('/tmp/context.db')

    def test_default_db_path_accepted(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('DB_PATH', None):
            assert StorageSettings().db_path is not None


class TestPoolHardeningSettings:
    """Test pool hardening settings."""

    def test_pool_hardening_settings_defaults(self) -> None:
        """Verify pool hardening settings have expected defaults."""
        from app.settings.storage import StorageSettings

        settings = StorageSettings()

        # Idle connections close after 5 minutes; each connection is replaced after 10000 queries
        assert settings.postgresql_max_inactive_lifetime_s == 300.0
        assert settings.postgresql_max_queries == 10000


class TestTcpKeepaliveSettings:
    """Test TCP keepalive settings configuration."""

    def test_tcp_keepalive_settings_defaults(self) -> None:
        """Verify TCP keepalive settings have expected defaults."""
        from app.settings.storage import StorageSettings

        settings = StorageSettings()

        assert settings.postgresql_tcp_keepalives_idle_s == 15
        assert settings.postgresql_tcp_keepalives_interval_s == 5
        assert settings.postgresql_tcp_keepalives_count == 3

    def test_tcp_keepalive_settings_types_are_int(self) -> None:
        """Verify TCP keepalive settings are integers (required by setsockopt)."""
        from app.settings.storage import StorageSettings

        settings = StorageSettings()

        assert isinstance(settings.postgresql_tcp_keepalives_idle_s, int)
        assert isinstance(settings.postgresql_tcp_keepalives_interval_s, int)
        assert isinstance(settings.postgresql_tcp_keepalives_count, int)
