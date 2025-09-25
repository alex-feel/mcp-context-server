from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import find_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

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
    # General storage
    max_image_size_mb: int = Field(default=10, alias='MAX_IMAGE_SIZE_MB')
    max_total_size_mb: int = Field(default=100, alias='MAX_TOTAL_SIZE_MB')
    db_path: Path | None = Field(default_factory=lambda: Path.home() / '.mcp' / 'context_storage.db', alias='DB_PATH')

    # Connection pool (DatabaseConnectionManager.PoolConfig)
    pool_max_readers: int = Field(default=8, alias='POOL_MAX_READERS')
    pool_max_writers: int = Field(default=1, alias='POOL_MAX_WRITERS')
    pool_connection_timeout_s: float = Field(default=10.0, alias='POOL_CONNECTION_TIMEOUT_S')
    pool_idle_timeout_s: float = Field(default=300.0, alias='POOL_IDLE_TIMEOUT_S')
    pool_health_check_interval_s: float = Field(default=30.0, alias='POOL_HEALTH_CHECK_INTERVAL_S')

    # Retry (DatabaseConnectionManager.RetryConfig)
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

    # Circuit breaker (DatabaseConnectionManager.CircuitBreaker)
    circuit_breaker_failure_threshold: int = Field(default=10, alias='CIRCUIT_BREAKER_FAILURE_THRESHOLD')
    circuit_breaker_recovery_timeout_s: float = Field(default=30.0, alias='CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S')
    circuit_breaker_half_open_max_calls: int = Field(default=5, alias='CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS')

    # Operation timeouts
    shutdown_timeout_s: float = Field(default=10.0, alias='SHUTDOWN_TIMEOUT_S')
    shutdown_timeout_test_s: float = Field(default=5.0, alias='SHUTDOWN_TIMEOUT_TEST_S')
    queue_timeout_s: float = Field(default=1.0, alias='QUEUE_TIMEOUT_S')
    queue_timeout_test_s: float = Field(default=0.1, alias='QUEUE_TIMEOUT_TEST_S')

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


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
