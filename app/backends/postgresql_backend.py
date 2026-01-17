"""PostgreSQL storage backend implementation.

This module provides a production-grade PostgreSQL backend implementing the StorageBackend
protocol with asyncpg connection pooling, circuit breaker pattern, retry logic, and health monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import random
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
from urllib.parse import quote

import asyncpg

from app.schemas import load_schema
from app.settings import get_settings

# Get settings (used for connection configuration)
settings = get_settings()
logger = logging.getLogger(__name__)


# Type definitions
T = TypeVar('T')


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

    _connection: asyncpg.pool.PoolConnectionProxy[asyncpg.Record]

    @property
    def connection(self) -> asyncpg.pool.PoolConnectionProxy[asyncpg.Record]:
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
            RuntimeError: If pgvector extension cannot be created or is inaccessible
        """
        try:
            conn = await asyncpg.connect(
                self.connection_string,
                timeout=settings.storage.postgresql_pool_timeout_s,
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
            raise RuntimeError(
                f'pgvector extension required but cannot be created (insufficient privileges): {e}',
            ) from e

        except Exception as e:
            logger.error(f'Failed to ensure pgvector extension: {e}')
            raise RuntimeError(f'pgvector extension is required but could not be created: {e}') from e

    async def initialize(self) -> None:
        """Initialize the PostgreSQL backend with connection pool and schema."""
        logger.info(f'Initializing PostgreSQL backend: {self.backend_type}')

        try:
            # Pre-create pgvector extension if semantic search enabled
            # This prevents "unknown type: public.vector" warnings during pool initialization
            if settings.semantic_search.enabled:
                await self._ensure_pgvector_extension()

            # Define connection initialization function for pgvector support
            async def _init_connection(conn: asyncpg.Connection) -> None:
                """Initialize each connection with pgvector type registration.

                Auto-detects the schema where pgvector extension is installed and registers
                the vector type codec. Works universally for all PostgreSQL variants:
                - Local PostgreSQL: pgvector in 'public' schema (default)
                - Supabase: pgvector in 'extensions' schema (managed)
                - AWS RDS / custom: pgvector in any schema

                Raises:
                    RuntimeError: If pgvector extension is not installed or codec registration fails
                """
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
                        raise RuntimeError(
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

                except Exception as e:
                    # STRICT: All other errors are FATAL
                    logger.error(f'Failed to register pgvector type codec: {e}')
                    logger.error('Ensure pgvector extension is enabled and accessible')
                    raise RuntimeError(f'pgvector codec registration failed: {e}') from e

            async def _reset_connection(conn: asyncpg.Connection) -> None:
                """Validate and reset connection before returning to pool.

                If validation fails, connection is terminated and pool creates a new one.
                This catches corrupted connections before they cause protocol errors.

                Called BEFORE connection returns to pool. If it raises, connection
                is terminated (not returned to pool).
                """
                try:
                    # Lightweight validation - catches protocol state mismatches
                    await conn.fetchval('SELECT 1')
                    # Reset session state (releases locks, closes cursors, etc.)
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
                    timeout_ms = int(settings.storage.postgresql_command_timeout_s * 1000 * 0.9)
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
                'timeout': settings.storage.postgresql_pool_timeout_s,
                # Prepared statement cache configuration
                # Set POSTGRESQL_STATEMENT_CACHE_SIZE=0 for external pooler compatibility
                # (PgBouncer transaction mode, Pgpool-II, AWS RDS Proxy, etc.)
                'statement_cache_size': settings.storage.postgresql_statement_cache_size,
                'max_cached_statement_lifetime': settings.storage.postgresql_max_cached_statement_lifetime_s,
                'max_cacheable_statement_size': settings.storage.postgresql_max_cacheable_statement_size,
                # Connection lifecycle callbacks
                'init': _init_connection,  # Type registration (existing)
                'setup': _setup_connection,  # Session config before acquire
                'reset': _reset_connection,  # Health check before pool return
            }

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

            # Verify connection and apply schema
            await self._initialize_schema()

            logger.info('PostgreSQL backend initialized successfully')

        except Exception as e:
            logger.error(f'Failed to initialize PostgreSQL backend: {e}')
            await self.circuit_breaker.record_failure()
            raise

    async def _initialize_schema(self) -> None:
        """Initialize database schema if tables don't exist."""
        schema_sql_template = load_schema(self.backend_type)

        # Template the schema SQL with configured schema name
        # This replaces {SCHEMA} placeholders with the actual schema (default: 'public')
        schema_sql = schema_sql_template.replace('{SCHEMA}', settings.storage.postgresql_schema)

        # Split schema into individual statements
        statements: list[str] = []
        current_statement: list[str] = []
        in_function = False

        for line in schema_sql.split('\n'):
            stripped = line.strip()

            # Skip comment-only lines
            if stripped.startswith('--'):
                continue

            # Track dollar-quoted strings (function bodies use $$)
            if '$$' in stripped:
                in_function = not in_function

            if stripped:
                current_statement.append(line)

            # End of statement: semicolon when not in dollar quotes
            if stripped.endswith(';') and not in_function:
                statements.append('\n'.join(current_statement))
                current_statement = []

        # Add any remaining statement
        if current_statement:
            statements.append('\n'.join(current_statement))

        # Execute each statement
        assert self._pool is not None, 'Pool not initialized'
        async with self._pool.acquire() as conn:
            for i, stmt in enumerate(statements):
                stmt = stmt.strip()
                if stmt and not stmt.startswith('--'):
                    try:
                        await conn.execute(stmt)
                        logger.debug(f'Schema statement {i + 1}/{len(statements)}: SUCCESS')
                    except Exception as e:
                        logger.error(f'Schema statement {i + 1} FAILED: {e}')
                        logger.error(f'Statement: {stmt[:200]}...')
                        raise

        logger.info('Database schema initialized')

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
    ) -> AsyncGenerator[Any, None]:
        """Get a database connection from the pool.

        Args:
            readonly: Advisory flag (PostgreSQL handles via transactions)
            allow_write: Advisory flag (PostgreSQL handles via transactions)

        Yields:
            asyncpg.Connection from the pool

        Raises:
            RuntimeError: If backend is shut down or circuit breaker is open
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

        # Acquire connection from pool
        async with self._pool.acquire() as conn:
            try:
                yield conn
                await self.circuit_breaker.record_success()
            except Exception:
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
            logger.warning(f'Connection validation failed: {e}')
            await self.circuit_breaker.record_failure()
            return False

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

        last_error: Exception | None = None

        for attempt in range(self.retry_config.max_retries):
            try:
                async with self.get_connection(readonly=False) as conn:
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

            except Exception as e:
                # Non-retryable error
                self.metrics.failed_queries += 1
                self.metrics.last_error = str(e)
                self.metrics.last_error_time = time.time()
                raise

        # Max retries exceeded
        self.metrics.failed_queries += 1
        self.metrics.last_error = str(last_error)
        self.metrics.last_error_time = time.time()
        raise last_error or Exception('Max retries exceeded for write operation')

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

        # Acquire connection from pool and begin transaction
        async with self._pool.acquire() as conn, conn.transaction():
            # Create transaction context
            txn_context = PostgreSQLTransactionContext(_connection=conn)

            try:
                yield txn_context
                # Transaction commits automatically on successful exit
                await self.circuit_breaker.record_success()
                logger.debug('Transaction committed successfully')

            except Exception as e:
                # Transaction rolls back automatically on exception
                logger.warning(f'Transaction failed, rolling back: {e}')
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

        # Add circuit breaker state
        # Note: get_state() is async but we need sync here
        # We'll use the last known state from metrics
        pool_metrics['circuit_state'] = self.metrics.circuit_state.value
        pool_metrics['consecutive_failures'] = self.circuit_breaker.failures

        return pool_metrics
