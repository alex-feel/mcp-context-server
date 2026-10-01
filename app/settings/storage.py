"""Storage backend settings: backend selection, connection pools, SQLite PRAGMAs, PostgreSQL, and metadata indexing."""

import logging
import re
from pathlib import Path
from typing import Literal
from typing import Self

from dotenv import find_dotenv
from pydantic import Field
from pydantic import SecretStr
from pydantic import ValidationInfo
from pydantic import field_validator
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

logger = logging.getLogger(__name__)

# A metadata field name is safe to index only when it is a plain SQL identifier:
# it is interpolated verbatim into a generated ``idx_metadata_<field>`` index name
# and into a JSON path/key literal on both backends, so any character outside this
# grammar (a hyphen, dot, whitespace, or quote) either yields an invalid identifier
# that crashes schema startup or breaks out of the literal. Enforced at the settings
# boundary by ``StorageSettings._validate_metadata_indexed_field_names``.
_SAFE_METADATA_FIELD_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

# PostgreSQL truncates every identifier (quoted included) to NAMEDATALEN-1 = 63 bytes.
# The generated index name is ``idx_metadata_`` (13 characters) followed by the field
# name, so a field longer than 63 - 13 = 50 characters is stored truncated in the
# catalog while reconciliation diffs the full configured name against the truncated
# ``pg_indexes`` name -- the diff never converges (perpetual drop/recreate in auto,
# a permanent false positive in strict), and two names sharing the same first 50
# characters silently collapse to one index. The identifier grammar above is pure
# ASCII, so one character is one byte and the character count is the byte budget.
_MAX_METADATA_FIELD_NAME_LENGTH = 50

# Accepted keywords for the four string-valued SQLite PRAGMA settings, which are
# interpolated verbatim into ``PRAGMA <name> = <value>`` by the SQLite backend.
# SQLite does NOT reject an unrecognized pragma argument: it silently keeps the
# current or default value, and the backend never reads the applied value back.
# So SQLITE_SYNCHRONOUS=FULLL boots clean and runs at NORMAL, and
# SQLITE_JOURNAL_MODE=wal-mode leaves a fresh database in DELETE mode -- the
# server reports healthy while its durability and concurrency characteristics
# differ from what the operator configured. Validating here turns a typo into a
# startup failure, exactly like the numeric bounds on the sibling storage fields.
# Comparison is case-insensitive (values are normalized to upper case), and the
# numeric spellings SQLite itself accepts for synchronous and temp_store are
# allowed alongside the keywords.
_SQLITE_PRAGMA_CHOICES: dict[str, frozenset[str]] = {
    'sqlite_journal_mode': frozenset({'DELETE', 'TRUNCATE', 'PERSIST', 'MEMORY', 'WAL', 'OFF'}),
    'sqlite_synchronous': frozenset({'OFF', 'NORMAL', 'FULL', 'EXTRA', '0', '1', '2', '3'}),
    'sqlite_temp_store': frozenset({'DEFAULT', 'FILE', 'MEMORY', '0', '1', '2'}),
    'sqlite_wal_checkpoint': frozenset({'PASSIVE', 'FULL', 'RESTART', 'TRUNCATE'}),
}

# SQLite stores the page size in the database header as a power of two in this
# range; ``PRAGMA page_size`` silently ignores anything else, so an out-of-range
# value would leave a fresh database on the 4096-byte default while the operator
# believes it was applied.
_SQLITE_MIN_PAGE_SIZE = 512
_SQLITE_MAX_PAGE_SIZE = 65536


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
    max_image_size_mb: int = Field(default=10, alias='MAX_IMAGE_SIZE_MB', ge=1)
    max_total_size_mb: int = Field(default=100, alias='MAX_TOTAL_SIZE_MB', ge=1)
    db_path: Path | None = Field(default_factory=lambda: Path.home() / '.mcp' / 'context_storage.db', alias='DB_PATH')

    # Connection pool settings for StorageBackend. Timeouts and intervals carry
    # gt=0 bounds: a zero or negative value passes float parsing but produces a
    # permanently broken runtime (an asyncio wait that returns immediately
    # busy-spins its loop; a non-positive timeout misclassifies as a retryable
    # dependency failure), so it is rejected at the configuration boundary.
    pool_max_readers: int = Field(default=8, alias='POOL_MAX_READERS', ge=1)
    pool_connection_timeout_s: float = Field(default=10.0, alias='POOL_CONNECTION_TIMEOUT_S', gt=0)
    pool_idle_timeout_s: float = Field(default=300.0, alias='POOL_IDLE_TIMEOUT_S', gt=0)
    pool_health_check_interval_s: float = Field(default=30.0, alias='POOL_HEALTH_CHECK_INTERVAL_S', gt=0)

    # Retry logic settings for StorageBackend. retry_max_retries is the TOTAL
    # attempt budget -- both backends' write paths run
    # `for attempt in range(max_retries)` -- so 1 means a single attempt with
    # no retries, and 0 would disable every database write outright (the loop
    # body never runs and the post-loop tail raises without touching the
    # database); rejected at the configuration boundary like the other
    # broken-at-zero bounds in this class.
    retry_max_retries: int = Field(default=5, alias='RETRY_MAX_RETRIES', ge=1)
    retry_base_delay_s: float = Field(default=0.5, alias='RETRY_BASE_DELAY_S', ge=0)
    retry_max_delay_s: float = Field(default=10.0, alias='RETRY_MAX_DELAY_S', ge=0)
    retry_jitter: bool = Field(default=True, alias='RETRY_JITTER')
    retry_backoff_factor: float = Field(default=2.0, alias='RETRY_BACKOFF_FACTOR', ge=1)

    # SQLite PRAGMAs. The keyword-valued ones are constrained to the arguments
    # SQLite actually recognizes (see _SQLITE_PRAGMA_CHOICES): an unrecognized
    # argument is silently ignored by SQLite, so an unvalidated typo would run
    # the server with different durability or concurrency than configured.
    sqlite_foreign_keys: bool = Field(default=True, alias='SQLITE_FOREIGN_KEYS')
    sqlite_journal_mode: str = Field(default='WAL', alias='SQLITE_JOURNAL_MODE')
    sqlite_synchronous: str = Field(default='NORMAL', alias='SQLITE_SYNCHRONOUS')
    sqlite_temp_store: str = Field(default='MEMORY', alias='SQLITE_TEMP_STORE')
    sqlite_mmap_size: int = Field(default=268_435_456, alias='SQLITE_MMAP_SIZE')  # 256MB
    # SQLite expects negative value for KB; provide directive directly
    sqlite_cache_size: int = Field(default=-64_000, alias='SQLITE_CACHE_SIZE')  # -64000 => 64MB
    sqlite_page_size: int = Field(default=4096, alias='SQLITE_PAGE_SIZE')
    sqlite_wal_autocheckpoint: int = Field(default=1000, alias='SQLITE_WAL_AUTOCHECKPOINT')
    sqlite_busy_timeout_ms: int | None = Field(default=None, alias='SQLITE_BUSY_TIMEOUT_MS', ge=0)
    sqlite_wal_checkpoint: str = Field(default='PASSIVE', alias='SQLITE_WAL_CHECKPOINT')

    # Circuit breaker settings for StorageBackend
    circuit_breaker_failure_threshold: int = Field(default=10, alias='CIRCUIT_BREAKER_FAILURE_THRESHOLD', ge=1)
    circuit_breaker_recovery_timeout_s: float = Field(default=30.0, alias='CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S', gt=0)
    circuit_breaker_half_open_max_calls: int = Field(default=5, alias='CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS', ge=1)

    # Operation timeouts. QUEUE_TIMEOUT_S feeds asyncio.wait(timeout=...) in the
    # write-queue processor loop, where a non-positive value returns immediately
    # every iteration and busy-spins a core for the process lifetime.
    shutdown_timeout_s: float = Field(default=10.0, alias='SHUTDOWN_TIMEOUT_S', gt=0)
    shutdown_timeout_test_s: float = Field(default=5.0, alias='SHUTDOWN_TIMEOUT_TEST_S', gt=0)
    queue_timeout_s: float = Field(default=1.0, alias='QUEUE_TIMEOUT_S', gt=0)
    queue_timeout_test_s: float = Field(default=0.1, alias='QUEUE_TIMEOUT_TEST_S', gt=0)

    # PostgreSQL connection settings
    postgresql_connection_string: SecretStr | None = Field(default=None, alias='POSTGRESQL_CONNECTION_STRING')
    postgresql_host: str = Field(default='localhost', alias='POSTGRESQL_HOST')
    # Bounded to a valid TCP port range so a typo (0, negative, >65535) is
    # rejected at the configuration boundary as a ValidationError (exit 78,
    # never retried), mirroring FASTMCP_PORT. Without the bound a bad port
    # passes pydantic and surfaces only at the socket layer as an OSError that
    # the backend classifies as a retryable DependencyError (exit 69), so a
    # supervisor restart-loops forever on a permanent misconfiguration.
    postgresql_port: int = Field(default=5432, alias='POSTGRESQL_PORT', ge=1, le=65535)
    postgresql_user: str = Field(default='postgres', alias='POSTGRESQL_USER')
    postgresql_password: SecretStr = Field(default=SecretStr('postgres'), alias='POSTGRESQL_PASSWORD')
    postgresql_database: str = Field(default='mcp_context', alias='POSTGRESQL_DATABASE')

    # PostgreSQL connection pool settings. ge bounds mirror the SQLite pool
    # fields, and validate_pool_min_not_above_max below enforces min <= max:
    # any size asyncpg would reject passes pydantic but only fails later at
    # asyncpg pool creation with a plain ValueError, which the backend's broad
    # exception handler would misclassify as a retryable DependencyError
    # (exit 69, supervisor restart loop) instead of a permanent
    # ConfigurationError. min stays ge=0 (an empty warm pool is valid).
    postgresql_pool_min: int = Field(default=2, alias='POSTGRESQL_POOL_MIN', ge=0)
    postgresql_pool_max: int = Field(default=20, alias='POSTGRESQL_POOL_MAX', ge=1)
    postgresql_session_pooler_max_clients: int = Field(
        default=15,
        alias='POSTGRESQL_SESSION_POOLER_MAX_CLIENTS',
        ge=1,
        description='Per-session client cap of an external session-mode pooler '
        '(Supabase Session Pooler / Supavisor). Advisory only: when a Supabase '
        'session-pooler endpoint is detected at startup and POSTGRESQL_POOL_MAX '
        'exceeds this value, the server logs a WARNING about MaxClientsInSessionMode. '
        'Default 15 matches Supabase Free/Pro tiers; raise it on larger tiers to '
        'silence false advisories. Never clamps the pool.',
    )
    postgresql_pool_timeout_s: float = Field(
        default=120.0,
        alias='POSTGRESQL_POOL_TIMEOUT_S',
        gt=0,
        description='Pool acquire-wait timeout in seconds: how long a caller '
                    'waits for a free pooled connection when every connection '
                    'is busy. Passed per-call to pool.acquire(timeout=...); '
                    'asyncpg.create_pool has no acquire-timeout parameter, so '
                    'a pool-level kwarg would silently become the connection '
                    'ESTABLISHMENT timeout instead (see '
                    'POSTGRESQL_CONNECT_TIMEOUT_S).',
    )
    postgresql_connect_timeout_s: float = Field(
        default=60.0,
        alias='POSTGRESQL_CONNECT_TIMEOUT_S',
        gt=0,
        description='Connection ESTABLISHMENT timeout in seconds (TCP connect '
                    'plus PostgreSQL startup handshake) for each new '
                    'connection the pool or the pgvector pre-check opens. '
                    'Distinct from POSTGRESQL_POOL_TIMEOUT_S, which bounds '
                    'waiting for a free pooled connection. Default 60 matches '
                    'the asyncpg default.',
    )
    postgresql_command_timeout_s: float = Field(default=60.0, alias='POSTGRESQL_COMMAND_TIMEOUT_S', gt=0)
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

    @field_validator('metadata_indexed_fields_raw', mode='after')
    @classmethod
    def _validate_metadata_indexed_field_names(cls, value: str) -> str:
        """Reject metadata index field names that are not safe SQL identifiers.

        Each configured field name is interpolated verbatim into a generated
        CREATE INDEX statement -- into the ``idx_metadata_<field>`` index name and
        into the JSON path/key literal on both backends. Three constraints are
        enforced here so an unsafe or ambiguous name is refused at the configuration
        boundary rather than reaching the DDL generator, where it would either crash
        schema startup or leave metadata-index reconciliation permanently unable to
        converge:

        - Grammar: a name outside ``[A-Za-z_][A-Za-z0-9_]*`` (a hyphen, dot,
          whitespace, or quote) would produce an invalid identifier, and an embedded
          quote would break out of the single-quoted literal.
        - Length: PostgreSQL truncates every identifier to 63 bytes, so with the
          13-character ``idx_metadata_`` prefix a field longer than 50 characters is
          stored truncated in the catalog while reconciliation diffs the full
          configured name against the truncated ``pg_indexes`` name -- the diff never
          converges (SQLite has no length cap, so the same config would diverge across
          backends).
        - Case uniqueness: two field names that differ only in case collide on
          SQLite, where ``CREATE INDEX IF NOT EXISTS`` compares index names
          case-insensitively and silently no-ops the second create, so only one index
          exists and reconciliation reports the other permanently missing; PostgreSQL
          quotes the generated identifier and case-preserves it, so the same config
          would again diverge across backends. Rejecting a casefold collision refuses
          the config identically on both backends. An IDENTICAL repeated name is
          rejected too, with its own accurate diagnostic (repeated entries can carry
          conflicting type hints that the dict-collapsing parse would resolve
          silently).

        Only the field name is validated here; the type hint after ``:`` is checked
        when the property parses the value.

        Args:
            value: The raw METADATA_INDEXED_FIELDS string.

        Returns:
            The value unchanged when every field name is a safe, unique identifier.

        Raises:
            ValueError: If any field name is not a plain SQL identifier, exceeds the
                length limit, is listed more than once, or collides with another name
                under case folding.
        """
        if not value or not value.strip():
            return value
        seen_casefolded: dict[str, str] = {}
        for item in value.split(','):
            item = item.strip()
            if not item:
                continue
            field = item.split(':', 1)[0].strip() if ':' in item else item
            if not _SAFE_METADATA_FIELD_NAME.match(field):
                raise ValueError(
                    f'METADATA_INDEXED_FIELDS contains an invalid field name {field!r}: '
                    f'names must match [A-Za-z_][A-Za-z0-9_]* (a letter or underscore '
                    f'followed by letters, digits, or underscores)',
                )
            if len(field) > _MAX_METADATA_FIELD_NAME_LENGTH:
                raise ValueError(
                    f'METADATA_INDEXED_FIELDS contains a field name {field!r} of length '
                    f'{len(field)}: names must be at most {_MAX_METADATA_FIELD_NAME_LENGTH} '
                    f'characters (the generated idx_metadata_ index name must fit '
                    f"PostgreSQL's 63-byte identifier limit)",
                )
            folded = field.casefold()
            if folded in seen_casefolded:
                previous = seen_casefolded[folded]
                if previous == field:
                    # An identical repeat needs its own diagnostic: telling the
                    # operator that two equal names "differ only in case" points at
                    # a nonexistent casing problem. Rejecting the repeat itself is
                    # deliberate -- two entries for the same field can carry
                    # conflicting type hints (e.g. status:string,status:integer)
                    # that the dict-collapsing parse would resolve silently.
                    raise ValueError(
                        f'METADATA_INDEXED_FIELDS lists field name {field!r} more than '
                        f'once: remove the duplicate entry (repeated entries can carry '
                        f'conflicting type hints that would be resolved silently)',
                    )
                raise ValueError(
                    f'METADATA_INDEXED_FIELDS contains field names {previous!r} '
                    f'and {field!r} that differ only in case: field names must be unique under '
                    f'case folding (case-differing names collide on SQLite and diverge across '
                    f'backends)',
                )
            seen_casefolded[folded] = field
        return value

    @property
    def resolved_busy_timeout_ms(self) -> int:
        """Resolve busy timeout to a valid integer value for SQLite."""
        # Default to connection timeout in milliseconds if not specified
        if self.sqlite_busy_timeout_ms is not None:
            return self.sqlite_busy_timeout_ms
        # Convert connection timeout from seconds to milliseconds
        return int(self.pool_connection_timeout_s * 1000)

    @field_validator('db_path', mode='before')
    @classmethod
    def _reject_blank_db_path(cls, value: object) -> object:
        """Reject an empty or whitespace-only DB_PATH.

        An empty DB_PATH coerces to ``Path('.')`` (the current directory) and a
        whitespace-only value to an all-blank path name; both are silent
        misconfigurations that surface far from their cause when the SQLite backend
        later tries to open the file. A blank value almost always means the variable
        was set but left unfilled, so it is rejected at the configuration boundary.

        Args:
            value: The raw DB_PATH input (a string from the environment, an explicit
                Path, or None).

        Returns:
            The value unchanged when it is not a blank string.

        Raises:
            ValueError: If DB_PATH is a string that is empty or only whitespace.
        """
        if isinstance(value, str) and not value.strip():
            raise ValueError('DB_PATH must not be empty or whitespace-only when set')
        return value

    @field_validator('sqlite_journal_mode', 'sqlite_synchronous', 'sqlite_temp_store', 'sqlite_wal_checkpoint')
    @classmethod
    def validate_sqlite_pragma_keyword(cls, v: str, info: ValidationInfo) -> str:
        """Reject a pragma argument SQLite would silently ignore.

        Args:
            v: The configured pragma argument.
            info: Validation context carrying the field being validated.

        Returns:
            str: The argument normalized to upper case.

        Raises:
            ValueError: If the argument is not one SQLite recognizes for this pragma.
        """
        # The validator is bound to the four pragma fields above, and each of
        # those fields' env alias is its name upper-cased (SQLITE_JOURNAL_MODE,
        # SQLITE_SYNCHRONOUS, SQLITE_TEMP_STORE, SQLITE_WAL_CHECKPOINT).
        field_name = info.field_name or ''
        allowed = _SQLITE_PRAGMA_CHOICES[field_name]
        normalized = v.strip().upper()
        if normalized not in allowed:
            raise ValueError(
                f"{field_name.upper()}='{v}' is not a value SQLite accepts for "
                f'PRAGMA {field_name.removeprefix("sqlite_")}. SQLite ignores an unrecognized '
                f'argument silently, so the setting would have no effect. '
                f'Valid options: {", ".join(sorted(allowed))}',
            )
        return normalized

    @field_validator('sqlite_page_size')
    @classmethod
    def validate_sqlite_page_size(cls, v: int) -> int:
        """Reject a page size SQLite would silently ignore.

        Args:
            v: The configured page size in bytes.

        Returns:
            int: The validated page size.

        Raises:
            ValueError: If the value is not a power of two in SQLite's supported range.
        """
        if not (_SQLITE_MIN_PAGE_SIZE <= v <= _SQLITE_MAX_PAGE_SIZE) or (v & (v - 1)) != 0:
            raise ValueError(
                f'SQLITE_PAGE_SIZE={v} is not a power of two between '
                f'{_SQLITE_MIN_PAGE_SIZE} and {_SQLITE_MAX_PAGE_SIZE}. SQLite ignores an '
                f'unsupported page size silently, so the setting would have no effect.',
            )
        return v

    @model_validator(mode='after')
    def validate_pool_min_not_above_max(self) -> Self:
        """Reject a warm-pool floor above the pool ceiling.

        asyncpg raises a plain ValueError('min_size is greater than max_size')
        from Pool.__init__ for this combination; caught at the backend's broad
        handler it would be misclassified as a retryable DependencyError and
        send the supervisor into a restart loop for a permanent
        misconfiguration, so it is rejected at the configuration boundary.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If POSTGRESQL_POOL_MIN exceeds POSTGRESQL_POOL_MAX.
        """
        if self.postgresql_pool_min > self.postgresql_pool_max:
            raise ValueError(
                f'POSTGRESQL_POOL_MIN ({self.postgresql_pool_min}) must not exceed '
                f'POSTGRESQL_POOL_MAX ({self.postgresql_pool_max})',
            )
        return self

    @model_validator(mode='after')
    def validate_connect_timeout_below_pool_timeout(self) -> Self:
        """Reject a connection-establishment budget that the acquire deadline always pre-empts.

        asyncpg wraps the whole acquire -- the queue wait AND the connect
        callable -- in ONE ``wait_for(timeout=POSTGRESQL_POOL_TIMEOUT_S)``, so an
        establishment budget at or above the acquire budget can never expire on
        its own: the acquire deadline always wins and CANCELS the in-flight dial
        instead, destroying the typed establishment error that tells an
        unreachable database from a saturated pool. The backend still charges
        such a dial (the acquire tracker records the interruption out of band),
        but the ordering is a misconfiguration in its own right -- a new
        connection gets strictly less time than the operator asked for -- so it
        is rejected at the configuration boundary where the operator sees it,
        rather than silently degrading connection diagnostics at runtime.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If POSTGRESQL_CONNECT_TIMEOUT_S is not below
                POSTGRESQL_POOL_TIMEOUT_S.
        """
        if self.postgresql_connect_timeout_s >= self.postgresql_pool_timeout_s:
            raise ValueError(
                f'POSTGRESQL_CONNECT_TIMEOUT_S ({self.postgresql_connect_timeout_s}) must be '
                f'below POSTGRESQL_POOL_TIMEOUT_S ({self.postgresql_pool_timeout_s}): the pool '
                f'acquire deadline bounds the connection dial, so an equal or larger connect '
                f'budget can never elapse and the dial is cancelled instead of timing out',
            )
        return self
