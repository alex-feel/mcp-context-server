"""
Tests for application settings validation.

Ensures settings validators fail fast with clear error messages
for invalid configuration values.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from app.settings import AppSettings


@contextmanager
def env_var(key: str, value: str | None) -> Generator[None, None, None]:
    """Context manager for temporarily setting an environment variable."""
    original = os.environ.get(key)
    try:
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]
        yield
    finally:
        if original is not None:
            os.environ[key] = original
        elif key in os.environ:
            del os.environ[key]


class TestFtsLanguageValidation:
    """Test FTS_LANGUAGE setting validation."""

    def test_valid_languages_accepted(self) -> None:
        """Test that all valid PostgreSQL text search configurations are accepted."""
        valid_languages = [
            'simple',
            'arabic',
            'armenian',
            'basque',
            'catalan',
            'danish',
            'dutch',
            'english',
            'finnish',
            'french',
            'german',
            'greek',
            'hindi',
            'hungarian',
            'indonesian',
            'irish',
            'italian',
            'lithuanian',
            'nepali',
            'norwegian',
            'portuguese',
            'romanian',
            'russian',
            'serbian',
            'spanish',
            'swedish',
            'tamil',
            'turkish',
            'yiddish',
        ]

        for lang in valid_languages:
            with env_var('FTS_LANGUAGE', lang):
                settings = AppSettings()
                assert settings.fts.language == lang.lower(), f'Language {lang} should be accepted'

    def test_valid_languages_case_insensitive(self) -> None:
        """Test that language validation is case-insensitive."""
        case_variations = [
            ('english', 'english'),
            ('English', 'english'),
            ('ENGLISH', 'english'),
            ('EnGlIsH', 'english'),
            ('German', 'german'),
            ('FRENCH', 'french'),
            ('Russian', 'russian'),
        ]

        for input_lang, expected_output in case_variations:
            with env_var('FTS_LANGUAGE', input_lang):
                settings = AppSettings()
                assert settings.fts.language == expected_output, (
                    f'Language {input_lang} should be normalized to {expected_output}'
                )

    def test_invalid_language_raises_error(self) -> None:
        """Test that invalid languages raise ValueError with clear message."""
        invalid_languages = [
            'invalid',
            'nonsense',
            'foo',
            'bar',
            'unknown',
            'eng',
            'en',
            'de',
            'fr',
        ]

        for lang in invalid_languages:
            with env_var('FTS_LANGUAGE', lang):
                with pytest.raises(ValidationError) as exc_info:
                    AppSettings()

                # Check error message contains useful information
                error_str = str(exc_info.value)
                assert 'FTS_LANGUAGE' in error_str, f'Error should mention FTS_LANGUAGE for {lang}'
                assert 'valid options' in error_str.lower(), f'Error should mention valid options for {lang}'

    def test_invalid_language_error_shows_valid_options(self) -> None:
        """Test that the error message shows the list of valid options."""
        with env_var('FTS_LANGUAGE', 'invalid_language'):
            with pytest.raises(ValidationError) as exc_info:
                AppSettings()

            error_str = str(exc_info.value)
            # Check that at least some valid languages are mentioned in the error
            assert 'english' in error_str.lower(), 'Error should list english as a valid option'
            assert 'german' in error_str.lower(), 'Error should list german as a valid option'
            assert 'french' in error_str.lower(), 'Error should list french as a valid option'

    def test_default_language_is_english(self) -> None:
        """Test that the default FTS language is english."""
        # Ensure FTS_LANGUAGE is not set
        with env_var('FTS_LANGUAGE', None):
            settings = AppSettings()
            assert settings.fts.language == 'english'

    def test_fts_language_via_environment_variable(self) -> None:
        """Test that FTS_LANGUAGE can be set via environment variable."""
        # Test valid language via env var
        with env_var('FTS_LANGUAGE', 'german'):
            settings = AppSettings()
            assert settings.fts.language == 'german'

        # Test case normalization via env var
        with env_var('FTS_LANGUAGE', 'FRENCH'):
            settings = AppSettings()
            assert settings.fts.language == 'french'

    def test_invalid_language_via_environment_variable(self) -> None:
        """Test that invalid FTS_LANGUAGE via env var raises error."""
        with env_var('FTS_LANGUAGE', 'completely_invalid_language'):
            with pytest.raises(ValidationError) as exc_info:
                AppSettings()

            error_str = str(exc_info.value)
            assert 'FTS_LANGUAGE' in error_str

    def test_whitespace_language_raises_error(self) -> None:
        """Test that whitespace-only language raises error."""
        whitespace_variants = ['   ', '\t', '\n', ' \t\n ']

        for ws in whitespace_variants:
            with env_var('FTS_LANGUAGE', ws), pytest.raises(ValidationError):
                AppSettings()

    def test_all_29_valid_languages_count(self) -> None:
        """Test that exactly 29 valid languages are supported."""
        # This ensures we don't accidentally add or remove languages
        valid_languages = {
            'simple',
            'arabic',
            'armenian',
            'basque',
            'catalan',
            'danish',
            'dutch',
            'english',
            'finnish',
            'french',
            'german',
            'greek',
            'hindi',
            'hungarian',
            'indonesian',
            'irish',
            'italian',
            'lithuanian',
            'nepali',
            'norwegian',
            'portuguese',
            'romanian',
            'russian',
            'serbian',
            'spanish',
            'swedish',
            'tamil',
            'turkish',
            'yiddish',
        }
        assert len(valid_languages) == 29, 'Should have exactly 29 valid PostgreSQL text search configurations'

        # Verify all are accepted
        for lang in valid_languages:
            with env_var('FTS_LANGUAGE', lang):
                settings = AppSettings()
                assert settings.fts.language == lang


class TestAuthProviderSetting:
    """Tests for MCP_AUTH_PROVIDER setting in AuthSettings."""

    def test_auth_provider_default_is_none(self) -> None:
        """AuthSettings provider should default to 'none'."""
        from app.settings import AuthSettings

        settings = AuthSettings()
        assert settings.provider == 'none'

    def test_auth_provider_from_env(self) -> None:
        """AuthSettings should load MCP_AUTH_PROVIDER from environment."""
        from app.settings import AuthSettings

        with env_var('MCP_AUTH_PROVIDER', 'simple_token'):
            settings = AuthSettings()
            assert settings.provider == 'simple_token'

    def test_auth_provider_invalid_value(self) -> None:
        """AuthSettings should reject invalid provider values."""
        from app.settings import AuthSettings

        with env_var('MCP_AUTH_PROVIDER', 'invalid'), pytest.raises(ValidationError):
            AuthSettings()


class TestTransportStatelessHttp:
    """Tests for FASTMCP_STATELESS_HTTP setting in TransportSettings."""

    def test_stateless_http_default_is_true(self) -> None:
        """FASTMCP_STATELESS_HTTP should default to True."""
        from app.settings import TransportSettings

        settings = TransportSettings()
        assert settings.stateless_http is True

    def test_stateless_http_enabled_via_env(self) -> None:
        """FASTMCP_STATELESS_HTTP=true should enable stateless mode."""
        from app.settings import TransportSettings

        with env_var('FASTMCP_STATELESS_HTTP', 'true'):
            settings = TransportSettings()
            assert settings.stateless_http is True

    def test_stateless_http_disabled_via_env(self) -> None:
        """FASTMCP_STATELESS_HTTP=false should disable stateless mode."""
        from app.settings import TransportSettings

        with env_var('FASTMCP_STATELESS_HTTP', 'false'):
            settings = TransportSettings()
            assert settings.stateless_http is False


class TestStorageImageSizeLimits:
    """MAX_IMAGE_SIZE_MB / MAX_TOTAL_SIZE_MB must be at least 1 megabyte."""

    def test_defaults_are_positive(self) -> None:
        from app.settings import StorageSettings

        settings = StorageSettings()
        assert settings.max_image_size_mb == 10
        assert settings.max_total_size_mb == 100

    def test_zero_image_size_rejected(self) -> None:
        from app.settings import StorageSettings

        with env_var('MAX_IMAGE_SIZE_MB', '0'), pytest.raises(ValidationError):
            StorageSettings()

    def test_negative_total_size_rejected(self) -> None:
        from app.settings import StorageSettings

        with env_var('MAX_TOTAL_SIZE_MB', '-5'), pytest.raises(ValidationError):
            StorageSettings()


class TestStoragePoolLimits:
    """POOL_MAX_READERS / POOL_MAX_WRITERS must be at least 1.

    POOL_MAX_READERS sizes an asyncio.Semaphore: a value of 0 would start it
    locked (every reader blocks forever, a silent deadlock) and a negative value
    raises an opaque ValueError deep in pool init, so both must be rejected
    cleanly at the configuration boundary like every peer concurrency cap.
    """

    def test_defaults_are_positive(self) -> None:
        from app.settings import StorageSettings

        settings = StorageSettings()
        assert settings.pool_max_readers == 8
        assert settings.pool_max_writers == 1

    def test_zero_readers_rejected(self) -> None:
        from app.settings import StorageSettings

        with env_var('POOL_MAX_READERS', '0'), pytest.raises(ValidationError):
            StorageSettings()

    def test_negative_writers_rejected(self) -> None:
        from app.settings import StorageSettings

        with env_var('POOL_MAX_WRITERS', '-1'), pytest.raises(ValidationError):
            StorageSettings()


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
        from app.settings import StorageSettings

        settings = StorageSettings()
        assert settings.postgresql_pool_min == 2
        assert settings.postgresql_pool_max == 20

    def test_zero_pool_max_rejected(self) -> None:
        from app.settings import StorageSettings

        with env_var('POSTGRESQL_POOL_MAX', '0'), pytest.raises(ValidationError):
            StorageSettings()

    def test_negative_pool_min_rejected(self) -> None:
        from app.settings import StorageSettings

        with env_var('POSTGRESQL_POOL_MIN', '-1'), pytest.raises(ValidationError):
            StorageSettings()

    def test_zero_pool_min_accepted(self) -> None:
        from app.settings import StorageSettings

        with env_var('POSTGRESQL_POOL_MIN', '0'):
            assert StorageSettings().postgresql_pool_min == 0

    def test_pool_min_above_max_rejected(self) -> None:
        """min above max would reach asyncpg as ValueError('min_size is greater than max_size')."""
        from app.settings import StorageSettings

        with (
            env_var('POSTGRESQL_POOL_MIN', '2'),
            env_var('POSTGRESQL_POOL_MAX', '1'),
            pytest.raises(ValidationError, match='must not exceed'),
        ):
            StorageSettings()

    def test_pool_min_equal_max_accepted(self) -> None:
        from app.settings import StorageSettings

        with env_var('POSTGRESQL_POOL_MIN', '5'), env_var('POSTGRESQL_POOL_MAX', '5'):
            settings = StorageSettings()
        assert settings.postgresql_pool_min == 5
        assert settings.postgresql_pool_max == 5

    def test_connect_timeout_defaults_to_asyncpg_default(self) -> None:
        """The establishment timeout is a separate knob from the acquire timeout."""
        from app.settings import StorageSettings

        settings = StorageSettings()
        assert settings.postgresql_connect_timeout_s == 60.0
        assert settings.postgresql_pool_timeout_s == 120.0

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
        from app.settings import StorageSettings

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
        from app.settings import StorageSettings

        with env_var(env_name, value):
            StorageSettings()
