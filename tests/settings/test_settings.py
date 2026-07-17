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


class TestPostgresqlPortBounds:
    """POSTGRESQL_PORT is bounded to a valid TCP port range at the config boundary.

    A port typo (0, negative, >65535) that passes pydantic surfaces only at the
    socket layer as an OSError the backend classifies as a retryable
    DependencyError (exit 69, supervisor restart-loops forever) instead of a
    permanent ConfigurationError (exit 78). The ge/le bound rejects it up front,
    mirroring FASTMCP_PORT.
    """

    def test_default_port_is_valid(self) -> None:
        from app.settings import StorageSettings

        assert StorageSettings().postgresql_port == 5432

    def test_zero_port_rejected(self) -> None:
        from app.settings import StorageSettings

        with env_var('POSTGRESQL_PORT', '0'), pytest.raises(ValidationError):
            StorageSettings()

    def test_negative_port_rejected(self) -> None:
        from app.settings import StorageSettings

        with env_var('POSTGRESQL_PORT', '-1'), pytest.raises(ValidationError):
            StorageSettings()

    def test_above_max_port_rejected(self) -> None:
        from app.settings import StorageSettings

        with env_var('POSTGRESQL_PORT', '70000'), pytest.raises(ValidationError):
            StorageSettings()

    def test_boundary_ports_accepted(self) -> None:
        from app.settings import StorageSettings

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


class TestPgvectorDimensionLimit:
    """EMBEDDING_DIM against the pgvector index cap on the fp32 PostgreSQL path.

    pgvector caps HNSW/IVFFlat index dimensionality at 2000 for the vector type.
    The fp32 PostgreSQL path always builds an HNSW index over vector(dim), so a
    dimension above 2000 passes pydantic's le=4096 field bound yet crashes the
    semantic-search migration at CREATE INDEX. Enabling compression (BYTEA
    payloads, no pgvector index) or using SQLite (sqlite-vec has no such cap)
    removes the constraint, so the guard is scoped to PostgreSQL + fp32 + embedding
    generation on and rejects the misconfiguration at the settings boundary.
    """

    def test_fp32_postgresql_dim_above_limit_rejected(self) -> None:
        """PostgreSQL + compression off + dim above 2000 is rejected before boot."""
        with (
            env_var('STORAGE_BACKEND', 'postgresql'),
            env_var('ENABLE_EMBEDDING_COMPRESSION', 'false'),
            env_var('ENABLE_EMBEDDING_GENERATION', 'true'),
            env_var('EMBEDDING_DIM', '2500'),
            pytest.raises(ValidationError, match='pgvector index limit'),
        ):
            AppSettings()

    def test_fp32_postgresql_dim_at_limit_accepted(self) -> None:
        """The exact 2000-dimension boundary is a valid fp32 PostgreSQL configuration."""
        with (
            env_var('STORAGE_BACKEND', 'postgresql'),
            env_var('ENABLE_EMBEDDING_COMPRESSION', 'false'),
            env_var('ENABLE_EMBEDDING_GENERATION', 'true'),
            env_var('EMBEDDING_DIM', '2000'),
        ):
            assert AppSettings().embedding.dim == 2000

    def test_compressed_postgresql_dim_above_limit_accepted(self) -> None:
        """With compression on, the vector is stored as BYTEA and the dim cap does not apply."""
        with (
            env_var('STORAGE_BACKEND', 'postgresql'),
            env_var('ENABLE_EMBEDDING_COMPRESSION', 'true'),
            env_var('ENABLE_EMBEDDING_GENERATION', 'true'),
            env_var('EMBEDDING_DIM', '2500'),
        ):
            assert AppSettings().embedding.dim == 2500

    def test_sqlite_dim_above_limit_accepted(self) -> None:
        """SQLite's sqlite-vec has no per-dimension index cap, so the guard does not fire."""
        with (
            env_var('STORAGE_BACKEND', 'sqlite'),
            env_var('ENABLE_EMBEDDING_COMPRESSION', 'false'),
            env_var('ENABLE_EMBEDDING_GENERATION', 'true'),
            env_var('EMBEDDING_DIM', '2500'),
        ):
            assert AppSettings().embedding.dim == 2500

    def test_generation_off_postgresql_dim_above_limit_accepted(self) -> None:
        """With generation off, no fresh fp32 vector table is provisioned, so the guard defers."""
        with (
            env_var('STORAGE_BACKEND', 'postgresql'),
            env_var('ENABLE_EMBEDDING_COMPRESSION', 'false'),
            env_var('ENABLE_EMBEDDING_GENERATION', 'false'),
            env_var('EMBEDDING_DIM', '2500'),
        ):
            assert AppSettings().embedding.dim == 2500


class TestBlankDbPath:
    """DB_PATH must not be blank when set.

    An empty DB_PATH coerces to Path('.') (a directory) and a whitespace-only value
    to an all-blank path name; either surfaces far from its cause when the SQLite
    backend later opens the file. A blank value almost always means the variable was
    set but left unfilled, so it is rejected at the configuration boundary.
    """

    def test_empty_db_path_rejected(self) -> None:
        from app.settings import StorageSettings

        with env_var('DB_PATH', ''), pytest.raises(ValidationError, match='must not be empty'):
            StorageSettings()

    def test_whitespace_db_path_rejected(self) -> None:
        from app.settings import StorageSettings

        with env_var('DB_PATH', '   '), pytest.raises(ValidationError, match='must not be empty'):
            StorageSettings()

    def test_valid_db_path_accepted(self) -> None:
        from pathlib import Path

        from app.settings import StorageSettings

        with env_var('DB_PATH', '/tmp/context.db'):
            assert StorageSettings().db_path == Path('/tmp/context.db')

    def test_default_db_path_accepted(self) -> None:
        from app.settings import StorageSettings

        with env_var('DB_PATH', None):
            assert StorageSettings().db_path is not None


class TestMetadataIndexedFieldNames:
    """METADATA_INDEXED_FIELDS field-name validation at the configuration boundary.

    Each configured field name is interpolated verbatim into a generated
    idx_metadata_<field> index name and JSON path/key literal on both backends, so
    the validator refuses names that would either crash schema startup or leave
    metadata-index reconciliation permanently unable to converge. Three constraints
    are enforced: the plain-SQL-identifier grammar, a 50-character length cap (so the
    13-character idx_metadata_ prefix leaves the generated name within PostgreSQL's
    63-byte identifier limit, which otherwise truncates the catalog name and diverges
    the reconciliation diff from SQLite), and case uniqueness under case folding (two
    case-differing names collide on SQLite's case-insensitive CREATE INDEX IF NOT
    EXISTS while PostgreSQL keeps them distinct, a cross-backend divergence).
    """

    def test_default_fields_accepted(self) -> None:
        from app.settings import StorageSettings

        settings = StorageSettings()
        assert 'status' in settings.metadata_indexed_fields
        assert settings.metadata_indexed_fields['technologies'] == 'array'

    def test_invalid_grammar_field_rejected(self) -> None:
        from app.settings import StorageSettings

        with env_var('METADATA_INDEXED_FIELDS', 'bad-field'), pytest.raises(ValidationError, match='invalid field name'):
            StorageSettings()

    def test_field_over_fifty_characters_rejected(self) -> None:
        """A 51-character field truncates in PostgreSQL's catalog once prefixed, so it is refused."""
        from app.settings import StorageSettings

        field = 'a' * 51
        with env_var('METADATA_INDEXED_FIELDS', field), pytest.raises(ValidationError, match='at most 50'):
            StorageSettings()

    def test_field_at_fifty_characters_accepted(self) -> None:
        """The 50-character boundary keeps the generated idx_metadata_ name within 63 bytes."""
        from app.settings import StorageSettings

        field = 'a' * 50
        with env_var('METADATA_INDEXED_FIELDS', field):
            assert field in StorageSettings().metadata_indexed_fields

    def test_casefold_colliding_fields_rejected(self) -> None:
        """Two names differing only in case collide on SQLite, so the config is refused."""
        from app.settings import StorageSettings

        with (
            env_var('METADATA_INDEXED_FIELDS', 'Status,status'),
            pytest.raises(ValidationError, match='differ only in case'),
        ):
            StorageSettings()

    def test_exact_duplicate_field_rejected_with_duplicate_diagnostic(self) -> None:
        """An identical repeated name is refused with an accurate duplicate diagnostic.

        The rejection must NOT claim the two equal names 'differ only in case' --
        that message describes a nonexistent casing problem and misdirects the
        operator away from the actual duplicate entry.
        """
        from app.settings import StorageSettings

        with (
            env_var('METADATA_INDEXED_FIELDS', 'status,status'),
            pytest.raises(ValidationError, match='more than once') as exc_info,
        ):
            StorageSettings()
        assert 'differ only in case' not in str(exc_info.value)

    def test_duplicate_field_with_conflicting_type_hints_rejected(self) -> None:
        """A repeated name carrying conflicting type hints gets the duplicate diagnostic."""
        from app.settings import StorageSettings

        with (
            env_var('METADATA_INDEXED_FIELDS', 'status:string,status:integer'),
            pytest.raises(ValidationError, match='more than once'),
        ):
            StorageSettings()

    def test_distinct_fields_accepted(self) -> None:
        """Case-distinct-but-not-colliding names remain valid."""
        from app.settings import StorageSettings

        with env_var('METADATA_INDEXED_FIELDS', 'status,agent_name'):
            fields = StorageSettings().metadata_indexed_fields
        assert 'status' in fields
        assert 'agent_name' in fields
