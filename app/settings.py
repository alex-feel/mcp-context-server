from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from typing import Self

from dotenv import find_dotenv
from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator
from pydantic import model_validator
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


class LoggingSettings(CommonSettings):
    """Application logging configuration."""

    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='ERROR',
        alias='LOG_LEVEL',
        description='Application log level',
    )


class ToolManagementSettings(CommonSettings):
    """MCP tool availability configuration."""

    disabled_raw: str = Field(
        default='',
        alias='DISABLED_TOOLS',
        description='Comma-separated list of tools to disable (e.g., delete_context,update_context)',
    )

    @property
    def disabled(self) -> set[str]:
        """Parse comma-separated string into lowercase set of disabled tool names."""
        if not self.disabled_raw or not self.disabled_raw.strip():
            return set()
        return {t.lower().strip() for t in self.disabled_raw.split(',') if t.strip()}


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


class AuthSettings(CommonSettings):
    """Authentication settings for HTTP transport.

    Configures the authentication provider passed to FastMCP's auth= parameter.
    Supported providers: 'none' (default), 'simple_token'.
    """

    provider: Literal['none', 'simple_token'] = Field(
        default='none',
        alias='MCP_AUTH_PROVIDER',
        description=(
            'Authentication provider: '
            'none (no auth, default), '
            'simple_token (bearer token)'
        ),
    )
    auth_token: SecretStr | None = Field(
        default=None,
        alias='MCP_AUTH_TOKEN',
        description='Bearer token for HTTP authentication (required when MCP_AUTH_PROVIDER=simple_token)',
    )
    auth_client_id: str = Field(
        default='mcp-client',
        alias='MCP_AUTH_CLIENT_ID',
        description='Client ID to assign to authenticated requests (used with simple_token)',
    )


class EmbeddingSettings(CommonSettings):
    """Embedding provider settings following LangChain conventions.

    All environment variable names follow LangChain documentation conventions
    for maximum compatibility and user familiarity.
    """

    # Embedding generation toggle
    # CRITICAL: generation_enabled default=True is INTENTIONAL and MUST NOT be changed.
    #
    # Rationale:
    # 1. Embeddings are fundamental infrastructure - users should explicitly opt OUT, not opt IN
    # 2. Fail-fast semantics prevent silent embedding gaps in stored content
    # 3. Users who don't want embeddings MUST explicitly set ENABLE_EMBEDDING_GENERATION=false
    # 4. This ensures no surprises - if embeddings are missing, user explicitly disabled them
    # 5. ENABLE_SEMANTIC_SEARCH=true requires embeddings; if ENABLE_EMBEDDING_GENERATION=false
    #    and ENABLE_SEMANTIC_SEARCH=true, semantic_search_context tool will NOT be registered
    #
    # DO NOT change this default without understanding the full architectural implications.
    # This default=True is part of the breaking change in v1.0.0.
    generation_enabled: bool = Field(
        default=True,
        alias='ENABLE_EMBEDDING_GENERATION',
        description='Enable embedding generation for stored context entries. '
                    'If true and dependencies are not met, server will NOT start. '
                    'Set to false to disable embeddings entirely.',
    )

    # Provider selection
    provider: Literal['ollama', 'openai', 'azure', 'huggingface', 'voyage'] = Field(
        default='ollama',
        alias='EMBEDDING_PROVIDER',
        description='Embedding provider: ollama (default), openai, azure, huggingface, voyage',
    )

    # Common settings
    model: str = Field(
        default='qwen3-embedding:0.6b',
        alias='EMBEDDING_MODEL',
        description='Embedding model name',
    )
    dim: int = Field(
        default=1024,
        alias='EMBEDDING_DIM',
        gt=0,
        le=4096,
        description='Embedding vector dimensions',
    )

    # Timeout and retry settings
    timeout_s: float = Field(
        default=30.0,
        alias='EMBEDDING_TIMEOUT_S',
        gt=0,
        le=300,
        description='Timeout in seconds for embedding generation API calls',
    )
    retry_max_attempts: int = Field(
        default=3,
        alias='EMBEDDING_RETRY_MAX_ATTEMPTS',
        ge=1,
        le=10,
        description='Maximum number of retry attempts for embedding generation',
    )
    retry_base_delay_s: float = Field(
        default=1.0,
        alias='EMBEDDING_RETRY_BASE_DELAY_S',
        gt=0,
        le=30,
        description='Base delay in seconds between retry attempts (with exponential backoff)',
    )

    # Ollama-specific (matches OLLAMA_HOST convention)
    ollama_host: str = Field(
        default='http://localhost:11434',
        alias='OLLAMA_HOST',
        description='Ollama server URL',
    )
    ollama_num_ctx: int = Field(
        default=4096,
        alias='OLLAMA_NUM_CTX',
        ge=512,
        le=131072,
        description='Ollama context length in tokens. Default 4096. '
                    'Must match or exceed model capabilities.',
    )
    ollama_truncate: bool = Field(
        default=False,
        alias='OLLAMA_TRUNCATE',
        description='Control text truncation when exceeding context length. '
                    'False (default): Returns error on exceeded context. '
                    'True: Silently truncates input (may degrade embedding quality).',
    )

    # OpenAI-specific (matches LangChain docs: OPENAI_API_KEY)
    openai_api_key: SecretStr | None = Field(
        default=None,
        alias='OPENAI_API_KEY',
        description='OpenAI API key',
    )
    openai_api_base: str | None = Field(
        default=None,
        alias='OPENAI_API_BASE',
        description='Custom base URL for OpenAI-compatible APIs',
    )
    openai_organization: str | None = Field(
        default=None,
        alias='OPENAI_ORGANIZATION',
        description='OpenAI organization ID',
    )

    # Azure OpenAI-specific (matches LangChain docs)
    azure_openai_api_key: SecretStr | None = Field(
        default=None,
        alias='AZURE_OPENAI_API_KEY',
        description='Azure OpenAI API key',
    )
    azure_openai_endpoint: str | None = Field(
        default=None,
        alias='AZURE_OPENAI_ENDPOINT',
        description='Azure OpenAI endpoint URL',
    )
    azure_openai_api_version: str = Field(
        default='2024-02-01',
        alias='AZURE_OPENAI_API_VERSION',
        description='Azure OpenAI API version',
    )
    azure_openai_deployment_name: str | None = Field(
        default=None,
        alias='AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME',
        description='Azure OpenAI embedding deployment name',
    )

    # HuggingFace-specific (matches LangChain docs)
    huggingface_api_key: SecretStr | None = Field(
        default=None,
        alias='HUGGINGFACEHUB_API_TOKEN',
        description='HuggingFace Hub API token',
    )

    # Voyage AI-specific (matches LangChain docs: VOYAGE_API_KEY)
    voyage_api_key: SecretStr | None = Field(
        default=None,
        alias='VOYAGE_API_KEY',
        description='Voyage AI API key',
    )
    voyage_truncation: bool = Field(
        default=False,
        alias='VOYAGE_TRUNCATION',
        description='Control text truncation when exceeding context length. '
                    'False (default): Returns error on exceeded context. '
                    'True: Silently truncates input (may degrade embedding quality).',
    )
    voyage_batch_size: int = Field(
        default=7,
        alias='VOYAGE_BATCH_SIZE',
        ge=1,
        le=128,
        description='Number of texts per API call (default: 7)',
    )

    @field_validator('dim')
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        """Validate embedding dimension is reasonable and warn about non-standard values."""
        if v > 4096:
            raise ValueError(
                'EMBEDDING_DIM exceeds reasonable limit (4096). '
                'Most embedding models use dimensions between 128-4096.',
            )
        if v % 64 != 0:
            logger.warning(
                f'EMBEDDING_DIM={v} is not a multiple of 64. '
                f'Most embedding models use dimensions divisible by 64.',
            )
        return v


class LangSmithSettings(CommonSettings):
    """LangSmith tracing settings for cost tracking and observability."""

    tracing: bool = Field(
        default=False,
        alias='LANGSMITH_TRACING',
        description='Enable LangSmith tracing for cost tracking and observability',
    )
    api_key: SecretStr | None = Field(
        default=None,
        alias='LANGSMITH_API_KEY',
        description='LangSmith API key for tracing',
    )
    endpoint: str = Field(
        default='https://api.smith.langchain.com',
        alias='LANGSMITH_ENDPOINT',
        description='LangSmith API endpoint',
    )
    project: str = Field(
        default='mcp-context-server',
        alias='LANGSMITH_PROJECT',
        description='LangSmith project name for grouping traces',
    )


class ChunkingSettings(CommonSettings):
    """Text chunking settings for semantic search.

    Controls how long documents are split into smaller chunks for embedding.
    Chunking improves semantic search quality for documents longer than ~500 tokens.
    """

    enabled: bool = Field(
        default=True,
        alias='ENABLE_CHUNKING',
        description='Enable text chunking for embedding generation',
    )
    size: int = Field(
        default=1000,
        alias='CHUNK_SIZE',
        ge=100,
        le=10000,
        description='Target chunk size in characters (default: 1000)',
    )
    overlap: int = Field(
        default=100,
        alias='CHUNK_OVERLAP',
        ge=0,
        le=500,
        description='Overlap between chunks in characters (default: 100)',
    )
    aggregation: Literal['max'] = Field(
        default='max',
        alias='CHUNK_AGGREGATION',
        description='How to aggregate chunk scores (currently only max is supported; '
                    'avg and sum will be added in future releases)',
    )
    dedup_overfetch: int = Field(
        default=5,
        alias='CHUNK_DEDUP_OVERFETCH',
        ge=1,
        le=20,
        description='Multiplier for fetching extra chunks before deduplication (default: 5)',
    )

    @model_validator(mode='after')
    def validate_overlap_less_than_size(self) -> Self:
        """Ensure overlap is strictly less than chunk size."""
        if self.overlap >= self.size:
            raise ValueError(
                f'CHUNK_OVERLAP ({self.overlap}) must be less than CHUNK_SIZE ({self.size})',
            )
        return self


class RerankingSettings(CommonSettings):
    """Cross-encoder reranking settings.

    Reranking improves search precision by using a cross-encoder model
    to re-score and reorder initial search results.
    """

    enabled: bool = Field(
        default=True,
        alias='ENABLE_RERANKING',
        description='Enable cross-encoder reranking of search results',
    )
    provider: str = Field(
        default='flashrank',
        alias='RERANKING_PROVIDER',
        description='Reranking provider (default: flashrank)',
    )
    model: str = Field(
        default='ms-marco-MiniLM-L-12-v2',
        alias='RERANKING_MODEL',
        description='Reranking model name (default: ms-marco-MiniLM-L-12-v2, 34MB)',
    )
    max_length: int = Field(
        default=512,
        alias='RERANKING_MAX_LENGTH',
        ge=128,
        le=2048,
        description='Maximum input length for reranking (default: 512 tokens)',
    )
    overfetch: int = Field(
        default=4,
        alias='RERANKING_OVERFETCH',
        ge=1,
        le=20,
        description='Multiplier for over-fetching results before reranking (default: 4x)',
    )
    cache_dir: str | None = Field(
        default=None,
        alias='RERANKING_CACHE_DIR',
        description='Directory for caching reranking models (default: system cache)',
    )
    chars_per_token: float = Field(
        default=4.0,
        alias='RERANKING_CHARS_PER_TOKEN',
        ge=2.0,
        le=8.0,
        description='Estimated characters per token for passage size validation. '
                    'Default 4.0 for English. Use 3.0-3.5 for multilingual/code.',
    )


class FtsPassageSettings(CommonSettings):
    """FTS passage extraction settings for reranking.

    Controls how text passages are extracted from FTS results with highlighted
    matches for use in cross-encoder reranking. These settings affect the quality
    and size of passages sent to the reranker.
    """

    rerank_window_size: int = Field(
        default=750,
        alias='FTS_RERANK_WINDOW_SIZE',
        ge=100,
        le=2000,
        description='Characters of context around each FTS match for reranking passage extraction (default: 750)',
    )

    rerank_gap_merge: int = Field(
        default=100,
        alias='FTS_RERANK_GAP_MERGE',
        ge=0,
        le=500,
        description='Merge FTS match regions within this character distance (default: 100)',
    )


class SemanticSearchSettings(CommonSettings):
    """Semantic search feature configuration.

    Controls whether semantic_search_context tool is registered.
    Requires embedding provider to be available.
    """

    enabled: bool = Field(
        default=False,
        alias='ENABLE_SEMANTIC_SEARCH',
        description='Enable semantic search tool registration',
    )


class FtsSettings(CommonSettings):
    """Full-text search feature configuration.

    Controls FTS tool registration and language/tokenizer settings.
    """

    enabled: bool = Field(
        default=False,
        alias='ENABLE_FTS',
        description='Enable full-text search functionality',
    )

    language: str = Field(
        default='english',
        alias='FTS_LANGUAGE',
        description='Language for FTS stemming (e.g., english, german, french)',
    )

    @field_validator('language')
    @classmethod
    def validate_language(cls, v: str) -> str:
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


class HybridSearchSettings(CommonSettings):
    """Hybrid search configuration using Reciprocal Rank Fusion (RRF).

    Combines FTS and semantic search results for improved relevance.
    """

    enabled: bool = Field(
        default=False,
        alias='ENABLE_HYBRID_SEARCH',
        description='Enable hybrid search combining FTS and semantic search',
    )

    rrf_k: int = Field(
        default=60,
        alias='HYBRID_RRF_K',
        ge=1,
        le=1000,
        description='RRF smoothing constant for hybrid search (default 60)',
    )

    rrf_overfetch: int = Field(
        default=2,
        alias='HYBRID_RRF_OVERFETCH',
        ge=1,
        le=10,
        description='Multiplier for over-fetching results before RRF fusion (default: 2x)',
    )


class SearchSettings(CommonSettings):
    """General search behavior configuration.

    Settings that apply across all search types (FTS, semantic, hybrid).
    """

    default_sort_by: Literal['relevance'] = Field(
        default='relevance',
        alias='SEARCH_DEFAULT_SORT_BY',
        description='Default sort order for search results (currently only relevance is supported; '
                    'created_at and updated_at will be added in future releases)',
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
    postgresql_pool_timeout_s: float = Field(default=120.0, alias='POSTGRESQL_POOL_TIMEOUT_S')
    postgresql_command_timeout_s: float = Field(default=60.0, alias='POSTGRESQL_COMMAND_TIMEOUT_S')
    postgresql_migration_timeout_s: float = Field(
        default=300.0,
        alias='POSTGRESQL_MIGRATION_TIMEOUT_S',
        gt=0,
        le=3600,
        description='Timeout in seconds for PostgreSQL migration operations. '
                    'Migrations may run DDL operations (CREATE INDEX, ALTER TABLE) '
                    'that require longer timeouts than regular queries. '
                    'Default: 300 seconds (5 minutes).',
    )

    # PostgreSQL connection pool hardening settings
    postgresql_max_inactive_lifetime_s: float = Field(
        default=300.0,
        alias='POSTGRESQL_MAX_INACTIVE_LIFETIME_S',
        ge=0,
        description='Close idle connections after this many seconds (0 to disable)',
    )
    postgresql_max_queries: int = Field(
        default=10000,
        alias='POSTGRESQL_MAX_QUERIES',
        ge=0,
        description='Recycle connections after this many queries (0 to disable)',
    )

    # PostgreSQL TCP keepalive settings
    # Configures client-side TCP keepalive to prevent network intermediaries
    # (NAT, firewalls, proxies, Supavisor) from closing idle connections
    postgresql_tcp_keepalives_idle_s: int = Field(
        default=15,
        alias='POSTGRESQL_TCP_KEEPALIVES_IDLE_S',
        ge=0,
        description='Seconds of idle time before sending first TCP keepalive probe (0 to disable)',
    )
    postgresql_tcp_keepalives_interval_s: int = Field(
        default=5,
        alias='POSTGRESQL_TCP_KEEPALIVES_INTERVAL_S',
        ge=0,
        description='Seconds between subsequent TCP keepalive probes (0 to disable)',
    )
    postgresql_tcp_keepalives_count: int = Field(
        default=3,
        alias='POSTGRESQL_TCP_KEEPALIVES_COUNT',
        ge=0,
        description='Number of failed TCP keepalive probes before connection is considered dead (0 to disable)',
    )

    # PostgreSQL asyncpg prepared statement cache settings
    # For external pooler compatibility (PgBouncer transaction mode, Pgpool-II, etc.),
    # set POSTGRESQL_STATEMENT_CACHE_SIZE=0 to disable caching
    postgresql_statement_cache_size: int = Field(
        default=100,
        alias='POSTGRESQL_STATEMENT_CACHE_SIZE',
        ge=0,
        le=10000,
        description='asyncpg prepared statement cache size. '
                    'Default: 100 (asyncpg default). '
                    'Set to 0 when using external connection poolers '
                    '(PgBouncer transaction mode, Pgpool-II, etc.) to disable caching.',
    )
    postgresql_max_cached_statement_lifetime_s: int = Field(
        default=300,
        alias='POSTGRESQL_MAX_CACHED_STATEMENT_LIFETIME_S',
        ge=0,
        le=86400,
        description='Maximum lifetime of cached prepared statements in seconds. '
                    'Default: 300. Has no effect when statement_cache_size=0.',
    )
    postgresql_max_cacheable_statement_size: int = Field(
        default=15360,
        alias='POSTGRESQL_MAX_CACHEABLE_STATEMENT_SIZE',
        ge=0,
        le=1048576,
        description='Maximum size of statement to cache in bytes. '
                    'Default: 15360 (15KB). Has no effect when statement_cache_size=0.',
    )

    # PostgreSQL SSL settings
    postgresql_ssl_mode: Literal['disable', 'allow', 'prefer', 'require', 'verify-ca', 'verify-full'] = Field(
        default='prefer',
        alias='POSTGRESQL_SSL_MODE',
    )

    # PostgreSQL schema setting
    postgresql_schema: str = Field(
        default='public',
        alias='POSTGRESQL_SCHEMA',
        description='PostgreSQL schema name for table and index operations',
    )

    # Default metadata fields for indexing (based on context-preservation-protocol requirements)
    metadata_indexed_fields_raw: str = Field(
        default='status,agent_name,task_name,project,report_type,references:object,technologies:array',
        alias='METADATA_INDEXED_FIELDS',
        description='Comma-separated list of metadata fields to index with optional type hints (field:type format)',
    )

    metadata_index_sync_mode: Literal['strict', 'auto', 'warn', 'additive'] = Field(
        default='additive',
        alias='METADATA_INDEX_SYNC_MODE',
        description='How to handle index mismatches: strict (fail), auto (sync), warn (log), additive (add missing only)',
    )

    @property
    def metadata_indexed_fields(self) -> dict[str, str]:
        """Parse field:type pairs from METADATA_INDEXED_FIELDS into dict.

        Returns:
            Dictionary mapping field names to their type hints.
            Supported types: 'string' (default), 'integer', 'boolean', 'float', 'array', 'object'

        Example:
            'status,priority:integer,completed:boolean' -> {'status': 'string', 'priority': 'integer', 'completed': 'boolean'}
        """
        if not self.metadata_indexed_fields_raw or not self.metadata_indexed_fields_raw.strip():
            return {}

        result: dict[str, str] = {}
        valid_types = {'string', 'integer', 'boolean', 'float', 'array', 'object'}

        for item in self.metadata_indexed_fields_raw.split(','):
            item = item.strip()
            if not item:
                continue
            if ':' in item:
                field, type_hint = item.split(':', 1)
                field = field.strip()
                type_hint = type_hint.strip().lower()
                # Validate type hint
                if type_hint not in valid_types:
                    logger.warning(f'Invalid type hint "{type_hint}" for field "{field}", defaulting to string')
                    type_hint = 'string'
                result[field] = type_hint
            else:
                result[item] = 'string'
        return result

    @property
    def resolved_busy_timeout_ms(self) -> int:
        """Resolve busy timeout to a valid integer value for SQLite."""
        # Default to connection timeout in milliseconds if not specified
        if self.sqlite_busy_timeout_ms is not None:
            return self.sqlite_busy_timeout_ms
        # Convert connection timeout from seconds to milliseconds
        return int(self.pool_connection_timeout_s * 1000)


class AppSettings(CommonSettings):
    # Core settings
    logging: LoggingSettings = Field(default_factory=lambda: LoggingSettings())
    tools: ToolManagementSettings = Field(default_factory=lambda: ToolManagementSettings())
    storage: StorageSettings = Field(default_factory=lambda: StorageSettings())

    # Search-related settings
    search: SearchSettings = Field(default_factory=lambda: SearchSettings())
    semantic_search: SemanticSearchSettings = Field(default_factory=lambda: SemanticSearchSettings())
    fts: FtsSettings = Field(default_factory=lambda: FtsSettings())
    hybrid_search: HybridSearchSettings = Field(default_factory=lambda: HybridSearchSettings())
    fts_passage: FtsPassageSettings = Field(default_factory=lambda: FtsPassageSettings())

    # Embedding and processing settings
    embedding: EmbeddingSettings = Field(default_factory=lambda: EmbeddingSettings())
    chunking: ChunkingSettings = Field(default_factory=lambda: ChunkingSettings())
    reranking: RerankingSettings = Field(default_factory=lambda: RerankingSettings())

    # Infrastructure settings
    transport: TransportSettings = Field(default_factory=lambda: TransportSettings())
    auth: AuthSettings = Field(default_factory=lambda: AuthSettings())
    langsmith: LangSmithSettings = Field(default_factory=lambda: LangSmithSettings())

    @model_validator(mode='after')
    def validate_chunk_size_vs_context_limit(self) -> Self:
        """Validate CHUNK_SIZE against model context window from context_limits.py.

        This is a UNIVERSAL validator that works for ALL embedding providers.

        When ENABLE_CHUNKING=true:
            - Validates CHUNK_SIZE against model's max_tokens
            - Warns if chunk size may exceed context window

        When ENABLE_CHUNKING=false:
            - Warns about potential issues with large documents
            - Document sizes are unknown at startup, so only general warning is possible

        Returns:
            Self: The validated settings instance.
        """
        # Import here to avoid circular imports at module load time
        try:
            from app.embeddings.context_limits import get_model_spec
            from app.embeddings.context_limits import get_provider_default_context
        except ImportError:
            # context_limits module not available - skip validation
            return self

        # Get model specification from context_limits.py
        model_spec = get_model_spec(self.embedding.model)

        # Determine max_tokens for the model
        # truncation_behavior can be 'error', 'silent', 'configurable', or None (unknown)
        truncation_behavior: str | None
        if model_spec:
            max_tokens = model_spec.max_tokens
            truncation_behavior = model_spec.truncation_behavior
            source = f'model spec for {model_spec.model}'
        else:
            # Unknown model - use provider default
            max_tokens = get_provider_default_context(self.embedding.provider)
            truncation_behavior = None  # Unknown behavior
            source = f'provider default for {self.embedding.provider}'
            logger.warning(
                f'[EMBEDDING CONFIG] Model "{self.embedding.model}" not found in context_limits.py. '
                f'Using provider default context limit ({max_tokens} tokens). '
                f'Consider adding model spec to app/embeddings/context_limits.py for accurate validation.',
            )

        if not self.chunking.enabled:
            # ENABLE_CHUNKING=false - warn about potential issues
            if truncation_behavior == 'silent':
                logger.warning(
                    f'[EMBEDDING CONFIG] ENABLE_CHUNKING=false with provider "{self.embedding.provider}". '
                    f'Model "{self.embedding.model}" ALWAYS silently truncates (cannot be disabled). '
                    f'Documents exceeding {max_tokens} tokens ({source}) will be truncated without warning.',
                )
            elif truncation_behavior == 'configurable':
                # Determine current truncation setting for this provider
                truncation_enabled = self._get_truncation_setting_for_provider()
                if truncation_enabled:
                    logger.warning(
                        f'[EMBEDDING CONFIG] ENABLE_CHUNKING=false with truncation enabled. '
                        f'Large documents will be silently truncated to {max_tokens} tokens ({source}). '
                        f'Consider enabling chunking for better embedding quality.',
                    )
                else:
                    logger.warning(
                        f'[EMBEDDING CONFIG] ENABLE_CHUNKING=false with truncation disabled. '
                        f'Documents exceeding {max_tokens} tokens ({source}) will cause embedding errors. '
                        f'Consider enabling chunking to handle large documents.',
                    )
            elif truncation_behavior == 'error':
                logger.warning(
                    f'[EMBEDDING CONFIG] ENABLE_CHUNKING=false with provider "{self.embedding.provider}". '
                    f'Model "{self.embedding.model}" returns error on context exceed (no truncation). '
                    f'Documents exceeding {max_tokens} tokens ({source}) will fail embedding. '
                    f'Consider enabling chunking to handle large documents.',
                )
            else:
                # Unknown truncation behavior
                logger.warning(
                    f'[EMBEDDING CONFIG] ENABLE_CHUNKING=false. Document sizes unknown at startup. '
                    f'Documents exceeding {max_tokens} tokens ({source}) may cause issues. '
                    f'Consider enabling chunking for better reliability.',
                )
            return self

        # ENABLE_CHUNKING=true - validate CHUNK_SIZE against max_tokens
        # Heuristic: 1 token ~ 3-4 characters for English
        chunk_tokens_estimate = self.chunking.size / 3

        if chunk_tokens_estimate > max_tokens:
            # Determine consequence based on truncation behavior
            if truncation_behavior == 'silent':
                consequence = 'will be silently truncated (quality degradation)'
            elif truncation_behavior == 'configurable':
                truncation_enabled = self._get_truncation_setting_for_provider()
                consequence = 'will be silently truncated' if truncation_enabled else 'will cause embedding errors'
            elif truncation_behavior == 'error':
                consequence = 'will cause embedding errors'
            else:
                consequence = 'may cause issues'

            logger.warning(
                f'[EMBEDDING CONFIG] CHUNK_SIZE ({self.chunking.size} chars, '
                f'~{int(chunk_tokens_estimate)} tokens estimate) exceeds '
                f'model context limit ({max_tokens} tokens from {source}). '
                f'Chunks {consequence}. '
                f'Recommendation: Reduce CHUNK_SIZE to ~{int(max_tokens * 3 * 0.8)} chars '
                f'(80% of context window).',
            )

        return self

    @model_validator(mode='after')
    def validate_fts_passage_vs_reranking(self) -> Self:
        """Validate FTS passage settings against cross-encoder token limits.

        When reranking is enabled, validates that FTS passage extraction settings
        are configured appropriately for the cross-encoder's max_length limit.

        Uses configurable chars_per_token ratio for token estimation, allowing
        users to tune based on their content type (English prose ~4.5, code ~3.5).

        Returns:
            Self: The validated settings instance.
        """
        # Skip validation if reranking is disabled
        if not self.reranking.enabled:
            return self

        # Calculate estimated passage size for a single FTS match with context windows
        boundary_expansion = 400  # max_search * 2 from expand_to_boundary
        single_match_estimate = self.fts_passage.rerank_window_size * 2 + boundary_expansion

        # Estimate token usage
        estimated_tokens = single_match_estimate / self.reranking.chars_per_token

        if estimated_tokens > self.reranking.max_length:
            optimal_window = int(
                (self.reranking.max_length * self.reranking.chars_per_token - boundary_expansion) / 2,
            )
            logger.warning(
                f'[FTS PASSAGE CONFIG] Single FTS match may produce ~{int(estimated_tokens)} tokens '
                f'(using {self.reranking.chars_per_token} chars/token), exceeding RERANKING_MAX_LENGTH '
                f'({self.reranking.max_length} tokens). Cross-encoder will truncate. '
                f'Recommendations: '
                f'1. Reduce FTS_RERANK_WINDOW_SIZE to ~{optimal_window} chars, OR '
                f'2. Increase RERANKING_CHARS_PER_TOKEN if your content has longer words',
            )

        return self

    def _get_truncation_setting_for_provider(self) -> bool:
        """Get current truncation setting for the configured provider.

        Returns:
            bool: True if truncation is enabled, False otherwise
        """
        # Provider is Literal['ollama', 'openai', 'azure', 'huggingface', 'voyage']
        # All cases are exhaustively covered
        match self.embedding.provider:
            case 'ollama':
                return self.embedding.ollama_truncate
            case 'voyage':
                return self.embedding.voyage_truncation
            case 'openai' | 'azure':
                return False  # OpenAI/Azure always error on exceed
            case 'huggingface':
                return True  # HuggingFace always silently truncates


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
