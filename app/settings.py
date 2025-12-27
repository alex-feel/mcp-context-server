from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import find_dotenv
from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CommonSettings(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=True,
        env_file=find_dotenv(),
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
        populate_by_name=True,
    )


class TransportSettings(CommonSettings):
    """HTTP transport settings for Docker/remote deployments."""

    transport: Literal['stdio', 'http', 'streamable-http', 'sse'] = Field(
        default='stdio',
        alias='MCP_TRANSPORT',
        description='Transport mode: stdio for local, http for Docker/remote',
    )
    host: str = Field(
        default='0.0.0.0',
        alias='FASTMCP_HOST',
        description='HTTP bind address (use 0.0.0.0 for Docker)',
    )
    port: int = Field(
        default=8000,
        alias='FASTMCP_PORT',
        ge=1,
        le=65535,
        description='HTTP port number',
    )


class StorageSettings(BaseSettings):
    """Storage-related settings with environment variable mapping."""

    model_config = SettingsConfigDict(
        frozen=False,  # Allow property access
        env_file=find_dotenv(),
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
        populate_by_name=True,
    )
    # Backend selection
    backend_type: Literal['sqlite', 'postgresql'] = Field(
        default='sqlite',
        alias='STORAGE_BACKEND',
    )
    # General storage
    max_image_size_mb: int = Field(default=10, alias='MAX_IMAGE_SIZE_MB')
    max_total_size_mb: int = Field(default=100, alias='MAX_TOTAL_SIZE_MB')
    db_path: Path | None = Field(default_factory=lambda: Path.home() / '.mcp' / 'context_storage.db', alias='DB_PATH')

    # Connection pool settings for StorageBackend
    pool_max_readers: int = Field(default=8, alias='POOL_MAX_READERS')
    pool_max_writers: int = Field(default=1, alias='POOL_MAX_WRITERS')
    pool_connection_timeout_s: float = Field(default=10.0, alias='POOL_CONNECTION_TIMEOUT_S')
    pool_idle_timeout_s: float = Field(default=300.0, alias='POOL_IDLE_TIMEOUT_S')
    pool_health_check_interval_s: float = Field(default=30.0, alias='POOL_HEALTH_CHECK_INTERVAL_S')

    # Retry logic settings for StorageBackend
    retry_max_retries: int = Field(default=5, alias='RETRY_MAX_RETRIES')
    retry_base_delay_s: float = Field(default=0.5, alias='RETRY_BASE_DELAY_S')
    retry_max_delay_s: float = Field(default=10.0, alias='RETRY_MAX_DELAY_S')
    retry_jitter: bool = Field(default=True, alias='RETRY_JITTER')
    retry_backoff_factor: float = Field(default=2.0, alias='RETRY_BACKOFF_FACTOR')

    # SQLite PRAGMAs
    sqlite_foreign_keys: bool = Field(default=True, alias='SQLITE_FOREIGN_KEYS')
    sqlite_journal_mode: str = Field(default='WAL', alias='SQLITE_JOURNAL_MODE')
    sqlite_synchronous: str = Field(default='NORMAL', alias='SQLITE_SYNCHRONOUS')
    sqlite_temp_store: str = Field(default='MEMORY', alias='SQLITE_TEMP_STORE')
    sqlite_mmap_size: int = Field(default=268_435_456, alias='SQLITE_MMAP_SIZE')  # 256MB
    # SQLite expects negative value for KB; provide directive directly
    sqlite_cache_size: int = Field(default=-64_000, alias='SQLITE_CACHE_SIZE')  # -64000 => 64MB
    sqlite_page_size: int = Field(default=4096, alias='SQLITE_PAGE_SIZE')
    sqlite_wal_autocheckpoint: int = Field(default=1000, alias='SQLITE_WAL_AUTOCHECKPOINT')
    sqlite_busy_timeout_ms: int | None = Field(default=None, alias='SQLITE_BUSY_TIMEOUT_MS')
    sqlite_wal_checkpoint: str = Field(default='PASSIVE', alias='SQLITE_WAL_CHECKPOINT')

    # Circuit breaker settings for StorageBackend
    circuit_breaker_failure_threshold: int = Field(default=10, alias='CIRCUIT_BREAKER_FAILURE_THRESHOLD')
    circuit_breaker_recovery_timeout_s: float = Field(default=30.0, alias='CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S')
    circuit_breaker_half_open_max_calls: int = Field(default=5, alias='CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS')

    # Operation timeouts
    shutdown_timeout_s: float = Field(default=10.0, alias='SHUTDOWN_TIMEOUT_S')
    shutdown_timeout_test_s: float = Field(default=5.0, alias='SHUTDOWN_TIMEOUT_TEST_S')
    queue_timeout_s: float = Field(default=1.0, alias='QUEUE_TIMEOUT_S')
    queue_timeout_test_s: float = Field(default=0.1, alias='QUEUE_TIMEOUT_TEST_S')

    # PostgreSQL connection settings
    postgresql_connection_string: SecretStr | None = Field(default=None, alias='POSTGRESQL_CONNECTION_STRING')
    postgresql_host: str = Field(default='localhost', alias='POSTGRESQL_HOST')
    postgresql_port: int = Field(default=5432, alias='POSTGRESQL_PORT')
    postgresql_user: str = Field(default='postgres', alias='POSTGRESQL_USER')
    postgresql_password: SecretStr = Field(default=SecretStr('postgres'), alias='POSTGRESQL_PASSWORD')
    postgresql_database: str = Field(default='mcp_context', alias='POSTGRESQL_DATABASE')

    # PostgreSQL connection pool settings
    postgresql_pool_min: int = Field(default=2, alias='POSTGRESQL_POOL_MIN')
    postgresql_pool_max: int = Field(default=20, alias='POSTGRESQL_POOL_MAX')
    postgresql_pool_timeout_s: float = Field(default=10.0, alias='POSTGRESQL_POOL_TIMEOUT_S')
    postgresql_command_timeout_s: float = Field(default=60.0, alias='POSTGRESQL_COMMAND_TIMEOUT_S')

    # PostgreSQL SSL settings
    postgresql_ssl_mode: Literal['disable', 'allow', 'prefer', 'require', 'verify-ca', 'verify-full'] = Field(
        default='prefer',
        alias='POSTGRESQL_SSL_MODE',
    )

    @property
    def resolved_busy_timeout_ms(self) -> int:
        """Resolve busy timeout to a valid integer value for SQLite."""
        # Default to connection timeout in milliseconds if not specified
        if self.sqlite_busy_timeout_ms is not None:
            return self.sqlite_busy_timeout_ms
        # Convert connection timeout from seconds to milliseconds
        return int(self.pool_connection_timeout_s * 1000)


class AppSettings(CommonSettings):
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='ERROR',
        alias='LOG_LEVEL',
    )

    storage: StorageSettings = Field(default_factory=lambda: StorageSettings())

    # Tool disabling - stored as raw string to avoid pydantic-settings JSON parsing
    disabled_tools_raw: str = Field(
        default='',
        alias='DISABLED_TOOLS',
        description='Comma-separated list of tools to disable (e.g., delete_context,update_context)',
    )

    @property
    def disabled_tools(self) -> set[str]:
        """Parse comma-separated string into lowercase set of disabled tool names."""
        if not self.disabled_tools_raw or not self.disabled_tools_raw.strip():
            return set()
        return {t.lower().strip() for t in self.disabled_tools_raw.split(',') if t.strip()}

    # Semantic search settings
    enable_semantic_search: bool = Field(default=False, alias='ENABLE_SEMANTIC_SEARCH')
    ollama_host: str = Field(default='http://localhost:11434', alias='OLLAMA_HOST')
    embedding_model: str = Field(default='embeddinggemma:latest', alias='EMBEDDING_MODEL')
    embedding_dim: int = Field(default=768, alias='EMBEDDING_DIM', gt=0, le=4096)

    # Full-text search settings
    enable_fts: bool = Field(default=False, alias='ENABLE_FTS')
    fts_language: str = Field(
        default='english',
        alias='FTS_LANGUAGE',
        description='Language for FTS stemming (e.g., english, german, french)',
    )

    # Hybrid search settings
    enable_hybrid_search: bool = Field(default=False, alias='ENABLE_HYBRID_SEARCH')
    hybrid_rrf_k: int = Field(
        default=60,
        alias='HYBRID_RRF_K',
        ge=1,
        le=1000,
        description='RRF smoothing constant for hybrid search (default 60)',
    )

    # Transport settings
    transport: TransportSettings = Field(default_factory=lambda: TransportSettings())

    @field_validator('embedding_dim')
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        """Validate embedding dimension is reasonable and warn about non-standard values."""
        if v > 4096:
            raise ValueError(
                'EMBEDDING_DIM exceeds reasonable limit (4096). Most Ollama embedding models use dimensions between 128-1024.',
            )
        if v % 64 != 0:
            logger.warning(
                f'EMBEDDING_DIM={v} is not a multiple of 64. '
                f'Most Ollama embedding models use dimensions divisible by 64 (e.g., 128, 256, 384, 512, 768, 1024).',
            )
        return v

    @field_validator('fts_language')
    @classmethod
    def validate_fts_language(cls, v: str) -> str:
        """Validate FTS language is a known PostgreSQL text search configuration.

        PostgreSQL FTS requires a valid text search configuration. Invalid values
        cause runtime failures when applying migrations or executing queries.
        This validator fails fast at startup to prevent runtime errors.

        Returns:
            str: The validated language name normalized to lowercase.

        Raises:
            ValueError: If the language is not a valid PostgreSQL text search configuration.
        """
        # PostgreSQL built-in text search configurations
        # Full list: SELECT cfgname FROM pg_ts_config;
        valid_languages = {
            'simple', 'arabic', 'armenian', 'basque', 'catalan', 'danish', 'dutch',
            'english', 'finnish', 'french', 'german', 'greek', 'hindi', 'hungarian',
            'indonesian', 'irish', 'italian', 'lithuanian', 'nepali', 'norwegian',
            'portuguese', 'romanian', 'russian', 'serbian', 'spanish', 'swedish',
            'tamil', 'turkish', 'yiddish',
        }
        v_lower = v.lower()
        if v_lower not in valid_languages:
            raise ValueError(
                f"FTS_LANGUAGE='{v}' is not a valid PostgreSQL text search configuration. "
                f'Valid options: {", ".join(sorted(valid_languages))}',
            )
        return v_lower


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
