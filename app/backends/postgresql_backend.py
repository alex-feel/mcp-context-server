"""PostgreSQL storage backend implementation.

This module provides a production-grade PostgreSQL backend implementing the StorageBackend
protocol with asyncpg connection pooling, circuit breaker pattern, retry logic, and health monitoring.
"""


import asyncio
import logging
import random
import socket
import time
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import TypeVar
from typing import cast
from typing import overload
from urllib.parse import quote
from urllib.parse import urlsplit

import asyncpg

from app.errors import ConfigurationError
from app.errors import ControlFlowError
from app.errors import DependencyError
from app.settings import AppSettings
from app.settings import get_settings

# Get settings (used for connection configuration)
settings = get_settings()
logger = logging.getLogger(__name__)


# Type definitions
T = TypeVar('T')


def quote_pg_identifier(name: str) -> str:
    """Return ``name`` as a PostgreSQL quoted identifier, doubling embedded quotes.

    The single source of truth for quoting a configured PostgreSQL identifier
    (currently the schema name) so the connection ``search_path`` and any
    schema-qualified DDL (e.g. the migration CLI's ``CREATE SCHEMA``) escape it
    IDENTICALLY and cannot drift: a name containing a double quote yields a valid
    quoted identifier rather than a malformed / identifier-injecting string.

    Args:
        name: The raw identifier (e.g. ``POSTGRESQL_SCHEMA``).

    Returns:
        The identifier wrapped in double quotes with embedded quotes doubled.
    """
    return '"' + name.replace('"', '""') + '"'


def build_asyncpg_connect_kwargs(app_settings: AppSettings | None = None) -> dict[str, Any]:
    """Build the asyncpg connection kwargs shared by the pool and the migration CLI.

    Returns a dict suitable for spreading into ``asyncpg.connect(dsn, **kwargs)``
    or merging into ``asyncpg.create_pool(dsn, **kwargs)``. This is the single
    source of truth for the two connection parameters that BOTH the long-lived
    server pool (``PostgreSQLBackend.initialize``) and the short-lived migration
    CLI (``app.cli.migrate``) must apply identically:

    - ``server_settings['search_path']``: always populated as
      ``"<POSTGRESQL_SCHEMA>", public`` (the schema double-quoted so mixed-case
      and reserved identifiers are safe) so bare table names resolve to the
      configured schema. With the default ``POSTGRESQL_SCHEMA=public`` this is
      the benign no-op ``"public", public``.
    - ``server_settings`` TCP keepalive GUCs: added only when their setting is
      > 0. These are SECONDARY (silently ignored by Supavisor/PgBouncer); the
      pool also installs the PRIMARY client-side ``setsockopt`` keepalive via its
      ``init`` callback, which short-lived CLI connections do not need.
    - ``statement_cache_size``: ``POSTGRESQL_STATEMENT_CACHE_SIZE`` (default 100;
      set 0 to disable prepared statements for transaction-mode poolers such as
      PgBouncer transaction mode, Pgpool-II, AWS RDS Proxy, or the Supabase
      Transaction Pooler).

    SSL is intentionally NOT included: asyncpg parses ``sslmode`` natively from
    the DSN query string, so SSL is carried by the connection URL itself.

    Args:
        app_settings: Resolved application settings. Defaults to
            ``get_settings()`` so callers in a different process (the CLI)
            pick up that process's environment.

    Returns:
        Mapping with ``server_settings`` and ``statement_cache_size`` keys.
    """
    resolved = app_settings if app_settings is not None else get_settings()
    storage = resolved.storage
    # Quote the schema via the shared quote_pg_identifier helper so the connection
    # search_path and any schema-qualified DDL (the migration CLI's CREATE SCHEMA)
    # escape it identically and cannot drift -- a schema name containing a double
    # quote yields a valid quoted identifier rather than a malformed search_path.
    server_settings: dict[str, str] = {
        'search_path': f'{quote_pg_identifier(storage.postgresql_schema)}, public',
        # Pin extra_float_digits to a shortest-round-trip setting so the numeric
        # metadata-filter discriminator (query_builder._pg_numeric_compare) sees
        # the Ryu shortest-repr float8 text it relies on: a cluster/role default
        # of 0 or negative reverts float8out to %.15g and would misclassify every
        # high-magnitude float-origin stored value, silently resurrecting the
        # eq-against-its-own-value divergence the discriminator exists to close.
        # Sent in the startup packet, so it survives the pool's RESET ALL (same
        # mechanism as search_path).
        'extra_float_digits': '1',
    }
    tcp_idle_guc = storage.postgresql_tcp_keepalives_idle_s
    tcp_interval_guc = storage.postgresql_tcp_keepalives_interval_s
    tcp_count_guc = storage.postgresql_tcp_keepalives_count
    if tcp_idle_guc > 0:
        server_settings['tcp_keepalives_idle'] = str(tcp_idle_guc)
    if tcp_interval_guc > 0:
        server_settings['tcp_keepalives_interval'] = str(tcp_interval_guc)
    if tcp_count_guc > 0:
        server_settings['tcp_keepalives_count'] = str(tcp_count_guc)
    return {
        'server_settings': server_settings,
        'statement_cache_size': storage.postgresql_statement_cache_size,
    }


class ConnectionState(Enum):
    """Connection health states for circuit breaker pattern."""

    HEALTHY = 'healthy'
    DEGRADED = 'degraded'
    FAILED = 'failed'


@dataclass
class ConnectionMetrics:
    """Metrics for monitoring connection health and performance."""

    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    total_queries: int = 0
    failed_queries: int = 0
    last_error: str | None = None
    last_error_time: float | None = None
    circuit_state: ConnectionState = ConnectionState.HEALTHY
    consecutive_failures: int = 0


@dataclass
class RetryConfig:
    """Configuration for retry logic with exponential backoff."""

    max_retries: int = 5
    base_delay: float = 0.5
    max_delay: float = 10.0
    jitter: bool = True
    backoff_factor: float = 2.0


def _statement_timeout_ms(command_timeout_s: float) -> int:
    """Derive the server-side statement_timeout from the client command timeout.

    Uses 90 percent of the client-side command timeout so the server-side
    backstop fires just before asyncpg's own cancellation, and floors the
    result at 1 ms: PostgreSQL treats ``SET statement_timeout = 0`` as
    UNLIMITED, so truncating a sub-millisecond command timeout to 0 would
    silently disable the very backstop the setting exists to provide.

    Args:
        command_timeout_s: ``POSTGRESQL_COMMAND_TIMEOUT_S`` in seconds.

    Returns:
        Millisecond value for ``SET statement_timeout``, at least 1.
    """
    return max(1, int(command_timeout_s * 1000 * 0.9))


@dataclass
class PostgreSQLTransactionContext:
    """Transaction context for PostgreSQL backend.

    Provides access to the asyncpg connection within an active transaction.
    The transaction lifecycle is managed by PostgreSQLBackend.begin_transaction().

    Note: PostgreSQL operations are ASYNCHRONOUS. All operations on the
    connection must be awaited.

    Attributes:
        _connection: The asyncpg connection proxy for this transaction
    """

    _connection: 'asyncpg.pool.PoolConnectionProxy[asyncpg.Record]'

    @property
    def connection(self) -> 'asyncpg.pool.PoolConnectionProxy[asyncpg.Record]':
        """Get the asyncpg connection proxy."""
        return self._connection

    @property
    def backend_type(self) -> str:
        """Get backend type identifier."""
        return 'postgresql'


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 10,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 5,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.failures = 0
        self.last_failure_time: float | None = None
        self.state = ConnectionState.HEALTHY
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def record_success(self) -> None:
        """Record a successful operation."""
        async with self._lock:
            if self.state == ConnectionState.DEGRADED:
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = ConnectionState.HEALTHY
                    self.failures = 0
                    self.half_open_calls = 0
                    logger.info('Circuit breaker recovered to HEALTHY state')
            elif self.state == ConnectionState.HEALTHY:
                self.failures = max(0, self.failures - 1)

    async def record_failure(self) -> None:
        """Record a failed operation."""
        async with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                self.state = ConnectionState.FAILED
                logger.warning(f'Circuit breaker tripped: {self.failures} consecutive failures')

    async def is_open(self) -> bool:
        """Check if circuit is open, meaning we should block calls."""
        async with self._lock:
            if self.state == ConnectionState.HEALTHY:
                return False

            if self.state == ConnectionState.FAILED:
                if self.last_failure_time:
                    elapsed = time.time() - self.last_failure_time
                    if elapsed > self.recovery_timeout:
                        self.state = ConnectionState.DEGRADED
                        self.half_open_calls = 0
                        logger.info('Circuit breaker entering DEGRADED state for recovery')
                        return False
                return True

            # DEGRADED state, allow limited calls
            return self.half_open_calls >= self.half_open_max_calls

    async def get_state(self) -> ConnectionState:
        """Get current circuit state."""
        async with self._lock:
            # Check if we should transition from FAILED to DEGRADED
            if self.state == ConnectionState.FAILED and self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed > self.recovery_timeout:
                    self.state = ConnectionState.DEGRADED
                    self.half_open_calls = 0
            return self.state

    def peek_state(self) -> ConnectionState:
        """Recovery-aware circuit state for synchronous, advisory reads.

        Mirrors the FAILED -> DEGRADED recovery transition that get_state() and
        is_open() apply once recovery_timeout elapses, WITHOUT acquiring the async
        lock or mutating state (a single-attribute read is safe for an advisory
        metric). Used by get_metrics() so the reported state matches live recovery
        behavior and stays consistent with the SQLite backend's reported state.

        Returns:
            DEGRADED when a FAILED breaker's recovery window has elapsed, else the
            current state.
        """
        if (
            self.state == ConnectionState.FAILED
            and self.last_failure_time
            and (time.time() - self.last_failure_time) > self.recovery_timeout
        ):
            return ConnectionState.DEGRADED
        return self.state


class PostgreSQLBackend:
    """Production-grade PostgreSQL storage backend implementing the StorageBackend protocol.

    Features:
    - asyncpg connection pooling with configurable min/max connections
    - Circuit breaker pattern for fault tolerance
    - Exponential backoff with jitter for transient errors
    - Explicit transaction management
    - Health checks and metrics
    - Automatic schema initialization

    Implements the StorageBackend protocol to enable database-agnostic repositories.
    """

    # Pgpool-II detection result (set during initialize() by _detect_pgpool_ii())
    _pgpool_version: str | None
    # Session-mode pooler detection result (set during initialize() by
    # _detect_session_mode_pooler()); True when a Supabase Session Pooler
    # endpoint (host *.pooler.supabase.com on port 5432) is in use.
    _session_mode_pooler: bool

    def __init__(
        self,
        connection_string: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        # Build connection string from settings if not provided
        if connection_string is None:
            connection_string = self._build_connection_string()

        self.connection_string = connection_string

        # Build retry config from settings if not supplied
        if retry_config is None:
            retry_config = RetryConfig(
                max_retries=settings.storage.retry_max_retries,
                base_delay=settings.storage.retry_base_delay_s,
                max_delay=settings.storage.retry_max_delay_s,
                jitter=settings.storage.retry_jitter,
                backoff_factor=settings.storage.retry_backoff_factor,
            )
        self.retry_config = retry_config

        # Connection pool
        self._pool: asyncpg.Pool | None = None
        # Whether the fp32 vector layout will be provisioned (so the pgvector
        # extension must exist and the vector codec must be registered):
        # generation on, OR a generation-off database that already carries
        # embedding infrastructure (the infra-present fallthrough the
        # semantic/chunking migrations use). Resolved in initialize().
        self._provision_vector: bool = settings.embedding.generation_enabled

        # Circuit breaker and metrics
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.storage.circuit_breaker_failure_threshold,
            recovery_timeout=settings.storage.circuit_breaker_recovery_timeout_s,
            half_open_max_calls=settings.storage.circuit_breaker_half_open_max_calls,
        )
        self.metrics = ConnectionMetrics()

        # Shutdown management
        self._shutdown = False

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier for PostgreSQL.

        Returns:
            str: Always returns 'postgresql' (includes Supabase via direct connection)
        """
        return 'postgresql'

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from settings.

        Supports both self-hosted PostgreSQL and Supabase via standard PostgreSQL settings.
        For Supabase, use POSTGRESQL_CONNECTION_STRING or individual settings with Supabase host.

        URL-encodes password to handle special characters like #, @, :, /, etc. that would
        otherwise break URL parsing in asyncpg connection strings.

        Returns:
            Connection string for asyncpg with properly URL-encoded credentials
        """
        # Use explicit connection string if provided
        if settings.storage.postgresql_connection_string:
            return settings.storage.postgresql_connection_string.get_secret_value()

        # Build from components (works for both self-hosted PostgreSQL and Supabase)
        host = settings.storage.postgresql_host
        port = settings.storage.postgresql_port
        user = settings.storage.postgresql_user
        password = settings.storage.postgresql_password.get_secret_value()
        database = settings.storage.postgresql_database

        # URL-encode password to handle special characters like #, @, :, /, etc.
        # safe='' ensures ALL special characters are encoded (e.g., # becomes %23)
        # asyncpg automatically URL-decodes the connection string
        encoded_password = quote(password, safe='')

        # Build connection string with encoded password
        conn_str = f'postgresql://{user}:{encoded_password}@{host}:{port}/{database}'

        # Add SSL mode if specified
        if settings.storage.postgresql_ssl_mode != 'prefer':
            conn_str += f'?sslmode={settings.storage.postgresql_ssl_mode}'

        return conn_str

    async def _ensure_pgvector_extension(self) -> None:
        """Ensure pgvector extension exists before pool creation.

        Creates a temporary connection to enable the pgvector extension,
        then immediately closes it. This allows pool connections to
        successfully register pgvector types during initialization.

        STRICT MODE: Fails fast if extension cannot be created.
        On Supabase: Enable via Dashboard → Extensions → vector (recommended).

        Raises:
            ConfigurationError: For permission/auth errors requiring human intervention
            DependencyError: For connection errors that may resolve with retry
        """
        try:
            conn = await asyncpg.connect(
                self.connection_string,
                timeout=settings.storage.postgresql_connect_timeout_s,
            )
            try:
                await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                logger.debug('pgvector extension ensured before pool creation')
            finally:
                await conn.close()

        except asyncpg.exceptions.InsufficientPrivilegeError as e:
            # Permission denied - common on managed services
            logger.error(
                'Cannot CREATE EXTENSION (insufficient privileges). '
                'Enable pgvector via database management interface: '
                'Supabase: Dashboard → Extensions → vector, '
                'AWS RDS: rds_superuser privileges required',
            )
            raise ConfigurationError(
                f'pgvector extension required but cannot be created (insufficient privileges): {e}',
            ) from e

        except asyncpg.exceptions.ClientConfigurationError as e:
            # Must precede the InterfaceError tuple below:
            # ClientConfigurationError subclasses InterfaceError, and asyncpg
            # raises it for permanent client-side misconfigurations (invalid
            # sslmode/target_session_attrs/gsslib values, unresolvable DSN
            # options) that a broad InterfaceError match would misclassify as
            # a retryable DependencyError and restart-loop on.
            logger.error(f'PostgreSQL client configuration invalid: {e}')
            raise ConfigurationError(
                f'PostgreSQL client configuration invalid: {e}. '
                'Check POSTGRESQL_CONNECTION_STRING and the POSTGRESQL_* '
                'connection options.',
            ) from e

        except (
            OSError,  # Includes ConnectionRefusedError, TimeoutError
            asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.exceptions.InterfaceError,
            asyncpg.exceptions.TooManyConnectionsError,
        ) as e:
            logger.error(f'Failed to connect to PostgreSQL: {e}')
            raise DependencyError(
                f'PostgreSQL connection failed: {e}. '
                'Ensure PostgreSQL is running and accessible.',
            ) from e

        except asyncpg.exceptions.InvalidPasswordError as e:
            logger.error(f'PostgreSQL authentication failed: {e}')
            raise ConfigurationError(
                f'PostgreSQL authentication failed: {e}. '
                'Check POSTGRESQL_USER and POSTGRESQL_PASSWORD.',
            ) from e

        except asyncpg.exceptions.InvalidCatalogNameError as e:
            logger.error(f'PostgreSQL database does not exist: {e}')
            raise ConfigurationError(
                f'PostgreSQL database does not exist: {e}. '
                'Create the database or check POSTGRESQL_DATABASE.',
            ) from e

        except Exception as e:
            logger.error(f'Failed to ensure pgvector extension: {e}')
            # Default to DependencyError for unknown errors (safer - allows retry)
            raise DependencyError(f'pgvector extension is required but could not be created: {e}') from e

    async def _resolve_provision_vector(self) -> bool:
        """Decide whether the pgvector extension and vector codec are needed at boot.

        The ``vector`` type is used ONLY by the fp32 ``vec_context_embeddings`` layout, so the
        extension and codec are needed exactly when that layout WILL be provisioned or ALREADY
        exists:

        - ``generation`` on + ``compression`` OFF: always provision, WITHOUT a probe (the common
          fast path). The compression-off server provisions the fp32 layout, so the extension must
          exist before its ``vector(dim)`` DDL runs.
        - ``generation`` on + ``compression`` ON (the v3.0.0 default): the server strips every fp32
          statement (``skip_fp32_vec``), stores payloads as BYTEA, and reads them in pure Python, so
          the ``vector`` type is never created or bound. Provisioning it anyway would force ``CREATE
          EXTENSION vector`` and CRASH boot on a host that lacks pgvector -- exactly the pgvector-free
          deployment the compression docs promote -- so fall through to the fp32-table probe and
          provision ONLY if a stray fp32 table actually exists (it needs the codec). The migration
          CLI's ``force=True`` init builds the full fp32 layout even under compression, but does NOT
          rely on this gate: ``initialize_target_postgresql`` creates the extension itself before its
          ``vector(dim)`` DDL, and its backend runs DDL only (no vector value I/O), while the paths
          that DO read fp32 vectors (``--compress``, a PG->PG copy) operate on a database where the
          fp32 table already exists, so the probe returns True for them.
        - the fp32 ``vec_context_embeddings`` table already exists: it is read via the vector codec
          (an fp32 archive, or the ``--compress`` CLI reading it before it replaces it with the
          compressed table).
        - ``compression`` off + ``generation`` off + ``embedding_metadata`` present: the
          infra-present fallthrough re-provisions the fp32 layout, which needs the type.

        The regression this closes: returning True for every generation-on boot forced a compressed
        (BYTEA-only) server -- the default configuration -- to create the unused pgvector extension,
        crashing boot on a pgvector-less PostgreSQL host. Keying on ``embedding_metadata`` presence
        alone could not tell the two payload formats apart (that table exists under both), which is
        why the fp32 table itself is probed. A connection failure returns False; the pool creation
        that follows surfaces the same error with proper classification.

        Returns:
            True when the pgvector extension and vector codec must be provisioned.
        """
        if settings.embedding.generation_enabled and not settings.compression.enabled:
            return True
        compression_enabled = settings.compression.enabled
        connect_kwargs = build_asyncpg_connect_kwargs()
        try:
            conn = await asyncpg.connect(
                self.connection_string,
                timeout=settings.storage.postgresql_connect_timeout_s,
                **connect_kwargs,
            )
        except Exception as e:
            logger.debug('vector-provision probe could not connect (deferring to pool init): %s', e)
            return False
        try:
            fp32_present = bool(
                await conn.fetchval("SELECT to_regclass('vec_context_embeddings') IS NOT NULL"),
            )
            if fp32_present:
                return True
            if compression_enabled:
                return False
            # compression off + generation off: the semantic/chunking infra-present
            # fallthrough re-provisions the fp32 vector layout iff embedding_metadata exists.
            return bool(await conn.fetchval("SELECT to_regclass('embedding_metadata') IS NOT NULL"))
        finally:
            await conn.close()

    async def initialize(self) -> None:
        """Initialize the PostgreSQL backend with connection pool."""
        logger.info(f'Initializing PostgreSQL backend: {self.backend_type}')

        try:
            # Resolve whether the pgvector extension and vector codec are needed. The
            # vector type is used ONLY by the fp32 vec_context_embeddings layout, so a
            # compressed (BYTEA-only) database -- a compression-on server, or a
            # generation-off archive restored on a host without pgvector -- must NOT be
            # forced to create the unused extension, while a compression-off database that
            # will provision or already carries the fp32 layout still must. See
            # _resolve_provision_vector.
            self._provision_vector = await self._resolve_provision_vector()

            # Pre-create pgvector extension when the vector layout will be
            # provisioned. This prevents "unknown type: public.vector" warnings
            # during pool initialization and lets the migration's vector DDL run.
            if self._provision_vector:
                await self._ensure_pgvector_extension()

            # Define connection initialization function for TCP keepalive and pgvector support
            async def _init_connection(conn: asyncpg.Connection) -> None:
                """Initialize each connection with TCP keepalive and pgvector type registration.

                Configures TCP keepalive on the client socket to prevent network intermediaries
                (NAT, firewalls, proxies, Supavisor) from closing idle connections.

                Also auto-detects the schema where pgvector extension is installed and registers
                the vector type codec for semantic search support.

                Raises:
                    ConfigurationError: If pgvector extension is not installed or codec registration fails
                """
                # === TCP Keepalive Configuration ===
                # Set keepalive on the client socket via setsockopt.
                # This is the PRIMARY mechanism for connections through Supavisor/PgBouncer,
                # which silently ignore server_settings GUC parameters.
                tcp_idle = settings.storage.postgresql_tcp_keepalives_idle_s
                tcp_interval = settings.storage.postgresql_tcp_keepalives_interval_s
                tcp_count = settings.storage.postgresql_tcp_keepalives_count

                if tcp_idle > 0 or tcp_interval > 0 or tcp_count > 0:
                    try:
                        transport = getattr(conn, '_transport', None)
                        raw_sock = transport.get_extra_info('socket') if transport is not None else None
                        if raw_sock is not None:
                            raw_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                            if tcp_idle > 0 and hasattr(socket, 'TCP_KEEPIDLE'):
                                raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, tcp_idle)
                            if tcp_interval > 0 and hasattr(socket, 'TCP_KEEPINTVL'):
                                raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, tcp_interval)
                            if tcp_count > 0 and hasattr(socket, 'TCP_KEEPCNT'):
                                raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, tcp_count)
                            logger.debug(
                                'TCP keepalive configured: idle=%ds, interval=%ds, count=%d',
                                tcp_idle, tcp_interval, tcp_count,
                            )
                        else:
                            logger.warning('Could not access socket for TCP keepalive configuration')
                    except Exception as e:
                        # TCP keepalive failure is non-fatal — log and continue
                        logger.warning('Failed to configure TCP keepalive: %s', e)

                # === pgvector Type Registration ===
                # Register the vector codec whenever the fp32 vector layout will be
                # provisioned (self._provision_vector: a compression-off generation-on
                # server, or any database that already carries the fp32 vec table). A
                # compressed server and a fresh generation-off database have no fp32 vec
                # table, so the codec is never needed and a missing pgvector extension must
                # NOT fail connection setup -- mirroring the same gate on
                # _ensure_pgvector_extension in initialize().
                # (The SQLite _load_sqlite_vec_extension load is NOT an analogue: it is
                # unconditional -- see its docstring -- because the vec0 module must stay
                # reachable for durable stale-embedding cleanup when generation is off.)
                if self._provision_vector:
                    try:
                        from pgvector.asyncpg import register_vector

                        # AUTO-DETECT: Query where pgvector extension is installed
                        # Works for ALL PostgreSQL variants (local, Supabase, AWS RDS, etc.)
                        result = await conn.fetchrow('''
                            SELECT n.nspname
                            FROM pg_extension e
                            JOIN pg_namespace n ON e.extnamespace = n.oid
                            WHERE e.extname = 'vector'
                        ''')

                        if not result:
                            # Extension not installed - fail fast with clear instructions
                            raise ConfigurationError(
                                'pgvector extension is not installed. '
                                'Enable it via: CREATE EXTENSION vector; (PostgreSQL) '
                                'or Dashboard → Extensions → vector (Supabase)',
                            )

                        schema = result['nspname']

                        # Register vector types using detected schema
                        # Note: Type stubs for register_vector are incomplete (missing schema parameter)
                        # Actual function signature: async def register_vector(conn, schema='public')
                        # Using cast to work around incomplete type stubs
                        register_func = cast(Callable[..., Awaitable[None]], register_vector)
                        await register_func(conn, schema)
                        logger.debug(f'Registered pgvector types from schema: {schema}')

                    except ImportError:
                        # ImportError is OK - semantic search is optional
                        logger.debug('pgvector not installed, skipping vector type registration')

                    except ConfigurationError:
                        # Re-raise ConfigurationError as-is (from "extension not installed" check above)
                        raise

                    except Exception as e:
                        # STRICT: All other errors are FATAL
                        logger.error(f'Failed to register pgvector type codec: {e}')
                        logger.error('Ensure pgvector extension is enabled and accessible')
                        raise ConfigurationError(f'pgvector codec registration failed: {e}') from e

                # === UUID Type Codec Registration ===
                # asyncpg's default codec maps the PostgreSQL ``uuid`` type to
                # ``asyncpg.pgproto.pgproto.UUID``, a fast subclass that
                # ``isinstance``-tests as ``uuid.UUID`` but is not a literal
                # ``str``. The repository layer exchanges identifiers as
                # canonical 32-char lowercase hex strings, so this codec
                # round-trips both directions through ``str``.
                #
                # Encoder: pass the string through verbatim. PostgreSQL's
                # native UUID input parser accepts both 32-char hex and
                # 36-char hyphenated forms (case-insensitive).
                # Decoder: normalize the 36-char hyphenated text returned by
                # PostgreSQL into the 32-char lowercase hex canonical form.
                from app.ids import normalize_id

                await conn.set_type_codec(
                    'uuid',
                    schema='pg_catalog',
                    encoder=lambda v: v,
                    decoder=normalize_id,
                    format='text',
                )
                logger.debug('Registered uuid->str type codec')

            async def _reset_connection(conn: asyncpg.Connection) -> None:
                """Validate and reset connection before returning to pool.

                Ensures clean state by:
                1. Rolling back any active transaction (safe no-op if none active)
                2. Validating connection health with lightweight query
                3. Resetting session GUC parameters

                If any step fails, connection is terminated and pool creates a new one.
                This catches corrupted connections before they cause protocol errors.

                Called BEFORE connection returns to pool. If it raises, connection
                is terminated (not returned to pool).
                """
                try:
                    # Abort any active transaction FIRST
                    # ROLLBACK is safe even if no transaction is active (no-op)
                    # This handles cases where request cancellation left uncommitted work
                    await conn.execute('ROLLBACK')
                    # Lightweight validation - catches protocol state mismatches
                    await conn.fetchval('SELECT 1')
                    # Reset session state (GUC parameters only, NOT transactions)
                    await conn.execute('RESET ALL')
                    logger.debug('Connection reset successful before pool return')
                except Exception as e:
                    logger.warning(f'Connection reset failed, connection will be terminated: {e}')
                    raise  # asyncpg will terminate connection and create new one

            async def _setup_connection(conn: asyncpg.Connection) -> None:
                """Configure session before connection is acquired.

                Sets statement_timeout to prevent queries from hanging indefinitely.
                This runs AFTER init but BEFORE the connection is returned to caller.
                """
                try:
                    # Set statement timeout to prevent infinite hangs
                    # Use slightly less than command_timeout for graceful handling
                    timeout_ms = _statement_timeout_ms(settings.storage.postgresql_command_timeout_s)
                    await conn.execute(f'SET statement_timeout = {timeout_ms}')
                    logger.debug(f'Connection setup: statement_timeout={timeout_ms}ms')
                except Exception as e:
                    logger.warning(f'Connection setup failed: {e}')
                    raise  # asyncpg will close connection and create new one

            # Prepare pool configuration with hardening parameters
            pool_kwargs: dict[str, Any] = {
                'min_size': settings.storage.postgresql_pool_min,
                'max_size': settings.storage.postgresql_pool_max,
                'command_timeout': settings.storage.postgresql_command_timeout_s,
                # asyncpg.create_pool has NO acquire-timeout parameter: unknown
                # kwargs fall through **connect_kwargs to asyncpg.connect(), so
                # 'timeout' here is the per-connection ESTABLISHMENT timeout
                # (TCP connect + startup handshake). The acquire-wait bound
                # (POSTGRESQL_POOL_TIMEOUT_S) is passed per-call at every
                # pool.acquire(timeout=...) site instead.
                'timeout': settings.storage.postgresql_connect_timeout_s,
                # statement_cache_size and server_settings (search_path + TCP
                # keepalive GUCs) are merged below via
                # build_asyncpg_connect_kwargs() so the pool and the migration
                # CLI share one source of truth.
                'max_cached_statement_lifetime': settings.storage.postgresql_max_cached_statement_lifetime_s,
                'max_cacheable_statement_size': settings.storage.postgresql_max_cacheable_statement_size,
                # Connection lifecycle callbacks
                'init': _init_connection,  # TCP keepalive + type registration
                'setup': _setup_connection,  # Session config before acquire
                'reset': _reset_connection,  # Health check before pool return
            }

            # Merge the shared connection kwargs (statement_cache_size +
            # server_settings) sent in the PostgreSQL startup packet so
            # session-level parameters are set in a single round-trip and
            # persist for the connection's lifetime (asyncpg-recommended over
            # per-connection ``SET`` callbacks).
            #
            # build_asyncpg_connect_kwargs() is the single source of truth shared
            # with the migration CLI: it always populates search_path
            # (``"POSTGRESQL_SCHEMA", public``, a benign no-op for the default
            # ``public``; double-quoted for mixed-case / reserved identifiers),
            # adds the SECONDARY TCP keepalive GUCs when > 0 (the PRIMARY
            # mechanism is the client-side setsockopt in _init_connection above,
            # since Supavisor/PgBouncer ignore these GUCs), and forwards
            # statement_cache_size (0 disables prepared statements for
            # transaction-mode poolers).
            pool_kwargs.update(build_asyncpg_connect_kwargs(settings))

            # Add connection recycling settings if configured (0 means disabled)
            if settings.storage.postgresql_max_inactive_lifetime_s > 0:
                pool_kwargs['max_inactive_connection_lifetime'] = (
                    settings.storage.postgresql_max_inactive_lifetime_s
                )
            if settings.storage.postgresql_max_queries > 0:
                pool_kwargs['max_queries'] = settings.storage.postgresql_max_queries

            # Create connection pool with hardening configuration
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                **pool_kwargs,
            )

            # Detect Pgpool-II and log result
            await self._detect_pgpool_ii()

            # Detect Supabase Session Pooler endpoint and warn on oversized pool
            self._detect_session_mode_pooler()

            logger.info('PostgreSQL backend initialized successfully')

        except asyncpg.exceptions.ClientConfigurationError as e:
            # Must precede the InterfaceError tuple below:
            # ClientConfigurationError subclasses BOTH InterfaceError and
            # ValueError, and asyncpg raises it for permanent client-side
            # misconfigurations (invalid sslmode/target_session_attrs/
            # gsslib values, unresolvable DSN options) that a broad
            # InterfaceError match would misclassify as a retryable
            # DependencyError and restart-loop on.
            logger.error(f'PostgreSQL client configuration invalid: {e}')
            await self.circuit_breaker.record_failure()
            raise ConfigurationError(
                f'PostgreSQL client configuration invalid: {e}. '
                'Check POSTGRESQL_CONNECTION_STRING and the POSTGRESQL_* '
                'connection options.',
            ) from e

        except (
            OSError,  # Includes ConnectionRefusedError, TimeoutError
            asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.exceptions.InterfaceError,
            asyncpg.exceptions.TooManyConnectionsError,
        ) as e:
            logger.error(f'Failed to initialize PostgreSQL backend: {e}')
            await self.circuit_breaker.record_failure()
            raise DependencyError(
                f'PostgreSQL connection failed: {e}. '
                'Ensure PostgreSQL is running and accessible.',
            ) from e

        except asyncpg.exceptions.InvalidPasswordError as e:
            logger.error(f'PostgreSQL authentication failed: {e}')
            await self.circuit_breaker.record_failure()
            raise ConfigurationError(
                f'PostgreSQL authentication failed: {e}. '
                'Check POSTGRESQL_USER and POSTGRESQL_PASSWORD.',
            ) from e

        except asyncpg.exceptions.InvalidCatalogNameError as e:
            logger.error(f'PostgreSQL database does not exist: {e}')
            await self.circuit_breaker.record_failure()
            raise ConfigurationError(
                f'PostgreSQL database does not exist: {e}. '
                'Create the database or check POSTGRESQL_DATABASE.',
            ) from e

        except ConfigurationError:
            raise  # Re-raise already-classified errors (from _init_connection)

        except DependencyError:
            raise  # Re-raise already-classified errors (from _ensure_pgvector_extension)

        except ValueError as e:
            # asyncpg raises plain ValueError for invalid construction inputs
            # (pool size combinations, non-positive command_timeout) before
            # any network I/O. DSN option errors instead surface as
            # ClientConfigurationError and are classified above -- its
            # InterfaceError base would shadow this clause. These are
            # permanent misconfigurations: exit 78 so the supervisor
            # does not restart-loop on them.
            logger.error(f'PostgreSQL configuration invalid: {e}')
            await self.circuit_breaker.record_failure()
            raise ConfigurationError(
                f'PostgreSQL configuration invalid: {e}. '
                'Check POSTGRESQL_POOL_* values and the connection string.',
            ) from e

        except Exception as e:
            logger.error(f'Failed to initialize PostgreSQL backend: {e}')
            await self.circuit_breaker.record_failure()
            # Default to DependencyError for unknown errors (safer - allows retry)
            raise DependencyError(
                f'PostgreSQL initialization failed: {e}. '
                'Ensure PostgreSQL is running and accessible.',
            ) from e

    async def _detect_pgpool_ii(self) -> None:
        """Detect if connected through Pgpool-II and log result.

        Uses SHOW POOL_VERSION command which is Pgpool-II specific.
        On direct PostgreSQL connections, this command raises UndefinedObjectError
        (error code 42704: unrecognized configuration parameter).
        """
        assert self._pool is not None, 'Pool not initialized'

        try:
            async with self._pool.acquire(
                timeout=settings.storage.postgresql_pool_timeout_s,
            ) as conn:
                version = await conn.fetchval('SHOW POOL_VERSION')
                if version:
                    logger.warning(
                        f'Pgpool-II detected: version {version}. '
                        f'Recommended to set POSTGRESQL_STATEMENT_CACHE_SIZE=0.',
                    )
                    self._pgpool_version = str(version)
                else:
                    logger.info('Looks like direct PostgreSQL connection (at least, no Pgpool-II)')
                    self._pgpool_version = None
        except asyncpg.exceptions.UndefinedObjectError:
            # Expected when not behind Pgpool-II - pool_version is not a known parameter
            logger.info('Looks like direct PostgreSQL connection (at least, no Pgpool-II)')
            self._pgpool_version = None
        except Exception as e:
            # Log but do not fail initialization
            logger.warning(f'Pgpool-II detection check failed: {e}')
            self._pgpool_version = None

    def _detect_session_mode_pooler(self) -> None:
        """Detect a Supabase Session Pooler endpoint and warn if pool too large.

        Inspects the connection string host/port (the authoritative endpoint the
        pool actually dials, including the POSTGRESQL_CONNECTION_STRING form).
        When a Supabase Session Pooler (host contains ``pooler.supabase.com`` on
        port 5432) is detected AND POSTGRESQL_POOL_MAX exceeds the conservative
        default per-session client cap, logs a targeted WARNING naming the
        symptom (MaxClientsInSessionMode) and the fix.

        Defense-in-depth only: does NOT modify pool size. Non-fatal -- any
        parsing failure is logged and treated as "not a session pooler".

        Mirrors _detect_pgpool_ii() in level, message shape, and resilience.
        """
        from app.startup.validation import is_supabase_session_pooler

        try:
            parsed = urlsplit(self.connection_string)
            host = (parsed.hostname or '').lower()
            port = parsed.port if parsed.port is not None else 5432
            if not host and '://' not in self.connection_string:
                # libpq key-value DSN form ("host=... port=..."), which asyncpg
                # also accepts: urlsplit yields no hostname, so parse the
                # whitespace-separated key=value tokens directly so the advisory
                # fires for this spelling too.
                kv = dict(
                    token.split('=', 1)
                    for token in self.connection_string.split()
                    if '=' in token
                )
                host = kv.get('host', '').lower()
                port = int(kv['port']) if kv.get('port') else 5432
        except Exception as e:
            # Malformed connection string is non-fatal for this advisory;
            # the pool creation above already succeeded with this string.
            logger.warning(f'Session-mode pooler detection check failed: {e}')
            self._session_mode_pooler = False
            return

        if not is_supabase_session_pooler(host, port):
            self._session_mode_pooler = False
            return

        self._session_mode_pooler = True

        pool_max = settings.storage.postgresql_pool_max
        cap = settings.storage.postgresql_session_pooler_max_clients
        if pool_max > cap:
            logger.warning(
                f'Supabase Session Pooler detected ({host}:{port}) with '
                f'POSTGRESQL_POOL_MAX={pool_max}, which exceeds '
                f'POSTGRESQL_SESSION_POOLER_MAX_CLIENTS={cap} (the session-mode '
                f'per-session client cap; default 15 on Supabase Free/Pro tiers). '
                f'This can intermittently fail with "MaxClientsInSessionMode: '
                f'max clients reached - in Session mode max clients are limited '
                f'to pool_size". Lower POSTGRESQL_POOL_MAX to fit your pooler '
                f'capacity (or raise POSTGRESQL_SESSION_POOLER_MAX_CLIENTS if your '
                f'tier allows more), or use the Transaction-mode pooler (port 6543) '
                f'or a Direct Connection. See docs/database-backends.md '
                f'"Session Pooler Connection Limits".',
            )

    async def shutdown(self) -> None:
        """Gracefully shut down the PostgreSQL backend."""
        logger.info('Shutting down PostgreSQL backend')

        self._shutdown = True

        try:
            # Close connection pool
            if self._pool:
                await self._pool.close()
                self._pool = None

            logger.info('PostgreSQL backend shutdown complete')

        except Exception as e:
            logger.error(f'Error during PostgreSQL backend shutdown: {e}')
            raise

    @asynccontextmanager
    async def get_connection(
        self,
        readonly: bool = False,
        allow_write: bool = False,
        record_breaker: bool = True,
    ) -> AsyncGenerator[Any, None]:
        """Get a database connection from the pool.

        Args:
            readonly: Advisory flag (PostgreSQL handles via transactions)
            allow_write: Advisory flag (PostgreSQL handles via transactions)
            record_breaker: When True (default), record a circuit-breaker success
                on a clean exit and a failure on an exception. execute_write sets
                this False so its retry loop records at most ONE breaker outcome
                per logical write (matching the SQLite backend) instead of one per
                retry attempt.

        Yields:
            asyncpg.Connection from the pool

        Raises:
            RuntimeError: If backend is shut down or circuit breaker is open
            ControlFlowError: Re-raised from the connection scope without recording a
                circuit-breaker failure (normal control flow, not a database fault)
        """
        # Parameters readonly and allow_write are part of StorageBackend protocol
        # but not used in PostgreSQL implementation (handled via transactions)
        _ = readonly
        _ = allow_write

        if self._shutdown:
            raise RuntimeError('PostgreSQL backend is shutting down')

        # Check circuit breaker
        if await self.circuit_breaker.is_open():
            raise RuntimeError(
                f'Database circuit breaker is open after {self.circuit_breaker.failures} failures',
            )

        assert self._pool is not None, 'Backend not initialized, call initialize() first'

        # Acquire connection from pool, bounded by the acquire-wait timeout so
        # pool exhaustion surfaces as a TimeoutError instead of an unbounded
        # hang (asyncpg's Pool.acquire waits forever with timeout=None).
        async with self._pool.acquire(
            timeout=settings.storage.postgresql_pool_timeout_s,
        ) as conn:
            try:
                yield conn
                if record_breaker:
                    await self.circuit_breaker.record_success()
            except ControlFlowError:
                # Normal control flow (e.g. a client-input validation error raised
                # inside the connection scope), NOT a database fault: do not record
                # a breaker failure, or a client repeatedly sending invalid input
                # opens the breaker and rejects every caller's healthy requests.
                raise
            except Exception:
                if record_breaker:
                    await self.circuit_breaker.record_failure()
                raise

    async def _validate_connection_state(self, conn: asyncpg.Connection) -> bool:
        """Validate connection is in healthy state before critical operations.

        Executes a lightweight query to verify the connection protocol state
        is synchronized with the server. This catches corrupted connections
        before they cause protocol errors in batch operations.

        Args:
            conn: The asyncpg connection to validate

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            # Lightweight query to verify protocol state
            await conn.fetchval('SELECT 1')
            return True
        except Exception as e:
            # Do not record a breaker failure here: execute_write (the only caller)
            # records exactly one breaker outcome per logical write on its final
            # result, so per-attempt accounting would over-count under retries.
            logger.warning(f'Connection validation failed: {e}')
            return False

    @overload
    async def execute_write(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        validate_connection: bool = False,
        **kwargs: Any,
    ) -> T: ...

    @overload
    async def execute_write(
        self,
        operation: Callable[..., T],
        *args: Any,
        validate_connection: bool = False,
        **kwargs: Any,
    ) -> T: ...

    async def execute_write(
        self,
        operation: Callable[..., T] | Callable[..., Awaitable[T]],
        *args: Any,
        validate_connection: bool = False,
        **kwargs: Any,
    ) -> T:
        """Execute a write operation with retry logic and transaction management.

        Args:
            operation: Async callable that performs the write operation.
                      Signature: async def operation(conn: asyncpg.Connection, *args, **kwargs) -> T
            *args: Positional arguments to pass to operation
            validate_connection: If True, validate connection state before operation.
                                Use for batch operations that are sensitive to protocol state.
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of the operation (type preserved via TypeVar)

        Raises:
            RuntimeError: If backend is shut down or circuit breaker is open
            asyncpg.exceptions.ConnectionDoesNotExistError: If connection validation fails

        Note:
            PostgreSQLBackend expects ASYNC callables (not sync). The operation is executed
            with await and wrapped in a transaction for consistency.
        """
        if self._shutdown:
            raise RuntimeError('PostgreSQL backend is shutting down')

        # Reject up-front when the breaker is already open (mirrors begin_transaction).
        # Doing this BEFORE the retry loop means a rejection is never seen by the loop's
        # generic handler, so it cannot record a spurious breaker failure that would
        # reset last_failure_time and perpetuate the open state (self-lockout).
        if await self.circuit_breaker.is_open():
            raise RuntimeError(
                f'Database circuit breaker is open after {self.circuit_breaker.failures} failures',
            )

        last_error: Exception | None = None

        for attempt in range(self.retry_config.max_retries):
            try:
                async with self.get_connection(readonly=False, record_breaker=False) as conn:
                    # Validate connection state before critical operations
                    if validate_connection and not await self._validate_connection_state(conn):
                        raise asyncpg.exceptions.ConnectionDoesNotExistError(
                            'Connection validation failed - connection may be corrupted',
                        )

                    async with conn.transaction():
                        # Cast to async callable since PostgreSQLBackend only uses async operations
                        async_operation = cast(Callable[..., Awaitable[T]], operation)
                        result = await async_operation(conn, *args, **kwargs)
                        self.metrics.total_queries += 1
                    # Record exactly ONE breaker success per logical write, AFTER the
                    # transaction commits (record_breaker=False above suppresses the
                    # per-attempt accounting so a retried write is not counted N times).
                    await self.circuit_breaker.record_success()
                    return result

            except asyncpg.exceptions.SerializationError as e:
                # Transient error - retry with backoff
                last_error = e
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.backoff_factor**attempt),
                    self.retry_config.max_delay,
                )

                if self.retry_config.jitter:
                    delay += random.uniform(0, delay * 0.3)

                logger.warning(
                    f'Serialization error on write, retrying in {delay:.2f}s '
                    f'(attempt {attempt + 1}/{self.retry_config.max_retries})',
                )
                await asyncio.sleep(delay)

            except asyncpg.exceptions.ConnectionDoesNotExistError as e:
                # Connection error - retry
                last_error = e
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.backoff_factor**attempt),
                    self.retry_config.max_delay,
                )

                if self.retry_config.jitter:
                    delay += random.uniform(0, delay * 0.3)

                logger.warning(
                    f'Connection error on write, retrying in {delay:.2f}s '
                    f'(attempt {attempt + 1}/{self.retry_config.max_retries})',
                )
                await asyncio.sleep(delay)

            except asyncpg.exceptions.InternalClientError as e:
                # Protocol state corruption - retry with fresh connection
                last_error = e
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.backoff_factor**attempt),
                    self.retry_config.max_delay,
                )

                if self.retry_config.jitter:
                    delay += random.uniform(0, delay * 0.3)

                logger.warning(
                    f'Protocol state error on write, retrying in {delay:.2f}s '
                    f'(attempt {attempt + 1}/{self.retry_config.max_retries}): {e}',
                )
                await asyncio.sleep(delay)

            except asyncpg.exceptions.QueryCanceledError as e:
                # Statement / lock-wait timeout (SQLSTATE 57014): PostgreSQL
                # cancelled the statement after it exceeded statement_timeout
                # (~0.9 * POSTGRESQL_COMMAND_TIMEOUT_S, set in _setup_connection).
                # Retry on a fresh connection with bounded backoff. This helps
                # only a TRANSIENT lock-WAIT that has since cleared; a write
                # fundamentally slower than the ceiling (e.g. fp32 in-transaction
                # HNSW maintenance with ENABLE_EMBEDDING_COMPRESSION=false) needs
                # a higher POSTGRESQL_COMMAND_TIMEOUT_S or compression left ON.
                # Safe to retry: every write operation reaching execute_write is
                # idempotent (deduplicating store / keyed update) and all
                # generation completed outside the transaction.
                last_error = e
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.backoff_factor**attempt),
                    self.retry_config.max_delay,
                )

                if self.retry_config.jitter:
                    delay += random.uniform(0, delay * 0.3)

                logger.warning(
                    f'Statement timeout on write, retrying in {delay:.2f}s '
                    f'(attempt {attempt + 1}/{self.retry_config.max_retries}): {e}',
                )
                await asyncio.sleep(delay)

            except Exception as e:
                # A circuit-breaker rejection raised by get_connection mid-loop (e.g.
                # a concurrent writer opened the breaker after the up-front check) is
                # NOT a write attempt. Recording a breaker failure for it would reset
                # last_failure_time and perpetuate the open state (self-lockout), so
                # re-raise that control-flow RuntimeError WITHOUT recording -- is_open()
                # matches get_connection's own recovery-aware gate.
                if isinstance(e, RuntimeError) and await self.circuit_breaker.is_open():
                    raise
                # Control-flow signals (optimistic-concurrency VersionConflictError,
                # post-dedup EmbeddingsReconcileRequiredError) are normal write contention,
                # NOT a database fault: roll back and propagate WITHOUT tripping the breaker,
                # mirroring begin_transaction's exemption (else routine contention could open
                # the breaker and lock out writes).
                if isinstance(e, ControlFlowError):
                    raise
                # Non-retryable write failure -- record the single breaker failure for
                # this logical write (get_connection's per-attempt accounting is off).
                self.metrics.failed_queries += 1
                self.metrics.last_error = str(e)
                self.metrics.last_error_time = time.time()
                await self.circuit_breaker.record_failure()
                raise

        # Max retries exceeded -- record exactly one breaker failure for the write.
        self.metrics.failed_queries += 1
        self.metrics.last_error = str(last_error)
        self.metrics.last_error_time = time.time()
        await self.circuit_breaker.record_failure()
        raise last_error or Exception('Max retries exceeded for write operation')

    @overload
    async def execute_read(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...

    @overload
    async def execute_read(
        self,
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...

    async def execute_read(
        self,
        operation: Callable[..., T] | Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a read operation with proper connection handling.

        Args:
            operation: Async callable that performs the read operation.
                      Signature: async def operation(conn: asyncpg.Connection, *args, **kwargs) -> T
                      Note: Although protocol accepts sync or async, PostgreSQLBackend only uses async.
            *args: Positional arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of the operation (type preserved via TypeVar)

        Note:
            PostgreSQLBackend expects ASYNC callables (not sync). The operation is executed
            with await.
        """
        async with self.get_connection(readonly=True) as conn:
            try:
                # Cast to async callable since PostgreSQLBackend only uses async operations
                async_operation = cast(Callable[..., Awaitable[T]], operation)
                result = await async_operation(conn, *args, **kwargs)
                self.metrics.total_queries += 1
                return result
            except Exception:
                self.metrics.failed_queries += 1
                raise

    @asynccontextmanager
    async def begin_transaction(self) -> AsyncGenerator[PostgreSQLTransactionContext, None]:
        """Begin an atomic transaction spanning multiple operations.

        This method acquires a connection from the pool and begins a transaction.
        All operations within the context share the same transaction.

        IMPORTANT: This method is intended for multi-operation atomic writes.
        For single operations, use execute_write() which is more efficient.

        Transaction semantics:
        - Uses asyncpg's native transaction context manager
        - Default isolation level: READ COMMITTED
        - On successful context exit: COMMIT
        - On exception: ROLLBACK

        Yields:
            PostgreSQLTransactionContext with the asyncpg connection

        Raises:
            RuntimeError: If backend is shutting down or circuit breaker is open

        Example:
            async with backend.begin_transaction() as txn:
                conn = txn.connection
                # All operations use same connection, same transaction
                row = await conn.fetchrow(
                    'INSERT INTO context_entries ... RETURNING id',
                    ...
                )
                context_id = row['id']
                await conn.execute('INSERT INTO tags ...', context_id, 'tag1')
                # COMMIT on exit
        """
        if self._shutdown:
            raise RuntimeError('PostgreSQL backend is shutting down')

        # Check circuit breaker
        if await self.circuit_breaker.is_open():
            raise RuntimeError(
                f'Database circuit breaker is open after {self.circuit_breaker.failures} failures',
            )

        assert self._pool is not None, 'Backend not initialized, call initialize() first'

        # Acquire the connection, then begin the transaction in an INNER context so we
        # observe the COMMIT result. asyncpg issues the COMMIT when the
        # `conn.transaction()` context EXITS; recording success before that exit (the old
        # `async with ..., conn.transaction():` form) credited a circuit-breaker SUCCESS
        # to a transaction whose COMMIT could still fail -- and that COMMIT failure,
        # raised on the context exit, landed OUTSIDE the try and was never recorded as a
        # failure (it could even leave the breaker mislearning health). Record success
        # only AFTER the inner context exits cleanly (COMMIT succeeded); a COMMIT failure
        # now flows through the same except as a body failure.
        async with self._pool.acquire(
            timeout=settings.storage.postgresql_pool_timeout_s,
        ) as conn:
            # Create transaction context
            txn_context = PostgreSQLTransactionContext(_connection=conn)

            try:
                async with conn.transaction():
                    yield txn_context
                    # Body succeeded; asyncpg COMMITs on exiting this inner context.
                # Reaching here means the COMMIT itself also succeeded.
                await self.circuit_breaker.record_success()
                logger.debug('Transaction committed successfully')

            except Exception as e:
                # Rolled back automatically on a body error, OR the COMMIT failed on exit.
                if isinstance(e, ControlFlowError):
                    # Normal control flow (optimistic-concurrency conflict / post-dedup
                    # embedding reconciliation), NOT a database fault: rolled back
                    # automatically, but do NOT record a circuit-breaker failure, so
                    # normal write contention cannot open the breaker.
                    raise
                logger.warning(f'Transaction failed or commit failed, rolling back: {e}')
                await self.circuit_breaker.record_failure()
                raise

    def get_metrics(self) -> dict[str, Any]:
        """Get backend health metrics and statistics.

        Returns:
            Dictionary with metrics including pool stats and circuit breaker state
        """
        pool_metrics: dict[str, Any] = {
            'backend_type': self.backend_type,
            'total_queries': self.metrics.total_queries,
            'failed_queries': self.metrics.failed_queries,
            'last_error': self.metrics.last_error,
            'last_error_time': self.metrics.last_error_time,
        }

        # Add pool metrics if pool exists
        if self._pool:
            pool_metrics.update({
                'pool_size': self._pool.get_size(),
                'pool_idle': self._pool.get_idle_size(),
                'pool_min_size': self._pool.get_min_size(),
                'pool_max_size': self._pool.get_max_size(),
            })

        # Add circuit breaker state. peek_state() is a sync, recovery-aware read
        # (applies the FAILED->DEGRADED transition once recovery_timeout elapses,
        # like get_state/is_open) so the reported state matches live behavior and
        # the SQLite backend; self.metrics.circuit_state is never updated here.
        pool_metrics['circuit_state'] = self.circuit_breaker.peek_state().value
        pool_metrics['consecutive_failures'] = self.circuit_breaker.failures

        # Add Pgpool-II detection info (only if detection has run)
        if hasattr(self, '_pgpool_version'):
            pool_metrics['pgpool_detected'] = self._pgpool_version is not None
            pool_metrics['pgpool_version'] = self._pgpool_version

        # Add session-mode pooler detection info (only if detection has run)
        if hasattr(self, '_session_mode_pooler'):
            pool_metrics['session_mode_pooler_detected'] = self._session_mode_pooler

        return pool_metrics
