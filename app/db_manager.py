"""
Database connection manager with pooling and concurrency control.

This module provides a robust connection management system for SQLite that solves
concurrency issues while maintaining ACID properties and high performance.
"""

import asyncio
import contextlib
import logging
import os
import random
import sqlite3
import sys
import time
from collections.abc import AsyncGenerator
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from threading import RLock
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
from typing import override

from app.logger_config import config_logger
from app.settings import get_settings

if TYPE_CHECKING:
    pass
else:
    pass

# Get setting
settings = get_settings()
# Configure logging
config_logger(settings.log_level)
logger = logging.getLogger(__name__)


# A connection subclass that guarantees best-effort finalization.
# If a connection object reaches GC while still open, this class
# will close it in __del__, which prevents ResourceWarning.
class ManagedConnection(sqlite3.Connection):
    _closed: bool = False

    @override
    def close(self) -> None:
        # Idempotent close, safe for __del__
        if self._closed:
            return
        try:
            super().close()
        finally:
            self._closed = True

    def __del__(self) -> None:
        # Never raise from __del__
        with contextlib.suppress(Exception):
            self.close()


# Type definitions
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')


def is_test_environment() -> bool:
    """Detect if running in test environment.

    Returns:
        bool: True if running in test environment, False otherwise
    """
    return any([
        'pytest' in sys.modules,
        os.environ.get('PYTEST_CURRENT_TEST'),
        os.environ.get('CI') == 'true',
    ])


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
    write_queue_size: int = 0
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
class PoolConfig:
    """Configuration for connection pooling."""

    max_readers: int = 8  # Increased for better concurrency
    max_writers: int = 1  # SQLite only supports one writer
    connection_timeout: float = 10.0  # Reduced for faster timeout
    idle_timeout: float = 300.0  # Close idle connections after 5 minutes
    health_check_interval: float = 30.0  # More frequent health checks

    def __post_init__(self) -> None:
        """Adjust settings based on environment."""
        if is_test_environment():
            # Optimize for test environment
            self.connection_timeout = 1.0  # Fast timeout in tests
            self.health_check_interval = 5.0  # More frequent health checks in tests


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 10,  # Increased threshold for less sensitivity
        recovery_timeout: float = 30.0,  # Faster recovery attempts
        half_open_max_calls: int = 5,  # More calls in half-open state
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.failures = 0
        self.last_failure_time: float | None = None
        self.state = ConnectionState.HEALTHY
        self.half_open_calls = 0
        self._lock = RLock()

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            if self.state == ConnectionState.DEGRADED:
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = ConnectionState.HEALTHY
                    self.failures = 0
                    self.half_open_calls = 0
                    logger.info('Circuit breaker recovered to HEALTHY state')
            elif self.state == ConnectionState.HEALTHY:
                self.failures = max(0, self.failures - 1)

    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                self.state = ConnectionState.FAILED
                logger.warning(f'Circuit breaker tripped: {self.failures} consecutive failures')

    def is_open(self) -> bool:
        """Check if circuit is open, meaning we should block calls."""
        with self._lock:
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

    def get_state(self) -> ConnectionState:
        """Get current circuit state."""
        with self._lock:
            # Check if we should transition from FAILED to DEGRADED
            if self.state == ConnectionState.FAILED and self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed > self.recovery_timeout:
                    self.state = ConnectionState.DEGRADED
                    self.half_open_calls = 0
            return self.state


class WriteRequest:
    """Encapsulates a write request for the queue."""

    def __init__(
        self,
        operation: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        future: asyncio.Future[Any],
    ) -> None:
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        self.future = future
        self.retry_count = 0
        self.created_at = time.time()


class DatabaseConnectionManager:
    """
    Production-grade connection manager with pooling and concurrency control.

    Features:
    - Connection pooling with separate reader and writer pools
    - Write queue for serializing write operations
    - Circuit breaker pattern for fault tolerance
    - Exponential backoff with jitter
    - Health checks and metrics
    - Automatic reconnection
    - Enhanced task lifecycle management for clean shutdown
    """

    def __init__(
        self,
        db_path: Path | str,
        pool_config: PoolConfig | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        # Build configs from settings if not supplied
        if pool_config is None:
            pool_config = PoolConfig(
                max_readers=settings.storage.pool_max_readers,
                max_writers=settings.storage.pool_max_writers,
                connection_timeout=settings.storage.pool_connection_timeout_s,
                idle_timeout=settings.storage.pool_idle_timeout_s,
                health_check_interval=settings.storage.pool_health_check_interval_s,
            )
        if retry_config is None:
            retry_config = RetryConfig(
                max_retries=settings.storage.retry_max_retries,
                base_delay=settings.storage.retry_base_delay_s,
                max_delay=settings.storage.retry_max_delay_s,
                jitter=settings.storage.retry_jitter,
                backoff_factor=settings.storage.retry_backoff_factor,
            )
        self.pool_config = pool_config
        self.retry_config = retry_config

        # Connection pools
        self._writer_conn: sqlite3.Connection | None = None
        self._reader_pool: list[sqlite3.Connection] = []
        self._reader_semaphore = asyncio.Semaphore(self.pool_config.max_readers)

        # Write queue for serialization
        self._write_queue: asyncio.Queue[WriteRequest] = asyncio.Queue()
        self._write_processor_task: asyncio.Task[None] | None = None

        # Synchronization primitives
        self._writer_lock = asyncio.Lock()
        self._pool_lock = Lock()

        # Comprehensive connection tracking for cleanup
        self._all_connections: set[sqlite3.Connection] = set()
        self._temporary_connections: set[sqlite3.Connection] = set()
        self._connection_lock = RLock()
        # Track connection IDs for debugging
        self._connection_ids: dict[int, str] = {}

        # Circuit breaker and metrics
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.storage.circuit_breaker_failure_threshold,
            recovery_timeout=settings.storage.circuit_breaker_recovery_timeout_s,
            half_open_max_calls=settings.storage.circuit_breaker_half_open_max_calls,
        )
        self.metrics = ConnectionMetrics()

        # Health check task
        self._health_check_task: asyncio.Task[None] | None = None

        # Enhanced task tracking for proper cleanup
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Shutdown management with complete signal
        self._shutdown = False
        self._shutdown_event = asyncio.Event()
        self._shutdown_complete = asyncio.Event()

    # Internal helpers

    def _safe_close_connection(self, conn: sqlite3.Connection) -> None:
        """Close and untrack a connection, idempotent and exception-safe.

        Always use this helper instead of calling conn.close() directly.
        """
        try:
            if conn is self._writer_conn:
                with contextlib.suppress(Exception):
                    conn.execute('PRAGMA optimize')
            conn.close()
        except sqlite3.ProgrammingError:
            pass  # Already closed
        except Exception:
            pass
        finally:
            with self._connection_lock:
                self._all_connections.discard(conn)
                self._temporary_connections.discard(conn)
                self._connection_ids.pop(id(conn), None)
            self.metrics.active_connections = max(0, self.metrics.active_connections - 1)

    async def initialize(self) -> None:
        """Initialize the connection manager and start background tasks."""
        logger.info(f'Initializing connection manager for {self.db_path}')

        # Create database directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize writer connection
        await self._ensure_writer_connection()

        # Start background tasks with proper tracking
        if not self._write_processor_task:
            task = asyncio.create_task(self._process_write_queue())
            self._write_processor_task = task
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        if not self._health_check_task:
            task = asyncio.create_task(self._health_check_loop())
            self._health_check_task = task
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        logger.info('Connection manager initialized successfully')

    async def shutdown(self) -> None:
        """Gracefully shutdown the connection manager with enhanced task cleanup."""
        logger.info('Shutting down connection manager')

        # Signal shutdown to all background tasks
        self._shutdown = True
        self._shutdown_event.set()

        # Drain write queue, cancel pending futures
        while not self._write_queue.empty():
            try:
                request = self._write_queue.get_nowait()
                if not request.future.done():
                    request.future.cancel()
            except asyncio.QueueEmpty:
                break
            except Exception:
                pass

        # Cancel and await all background tasks
        tasks_to_cancel = list(self._background_tasks)

        # Also include specific task references if they exist
        if self._write_processor_task and not self._write_processor_task.done():
            tasks_to_cancel.append(self._write_processor_task)
        if self._health_check_task and not self._health_check_task.done():
            tasks_to_cancel.append(self._health_check_task)

        if tasks_to_cancel:
            logger.debug(f'Cancelling {len(tasks_to_cancel)} background tasks')
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete with timeout
            try:
                # Determine timeout based on environment
                shutdown_timeout = (
                    settings.storage.shutdown_timeout_test_s
                    if is_test_environment()
                    else settings.storage.shutdown_timeout_s
                )
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=shutdown_timeout,
                )
            except TimeoutError:
                logger.warning('Some tasks did not complete within timeout')
                # Force-cancel any remaining tasks
                for task in tasks_to_cancel:
                    if not task.done():
                        task.cancel()
                        # Give tasks a brief moment to handle cancellation
                        with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                            await asyncio.wait_for(task, timeout=0.1)

        # Clear task references before closing connections
        self._background_tasks.clear()
        self._write_processor_task = None
        self._health_check_task = None

        # Small delay to ensure all async operations are settled
        await asyncio.sleep(0.01)

        # Close all connections
        await self._close_all_connections()

        # Signal shutdown complete
        self._shutdown_complete.set()
        logger.info('Connection manager shutdown complete')

    def _create_connection(self, readonly: bool = False) -> sqlite3.Connection:
        """Create a new SQLite connection with optimal settings."""
        # Do not create new connections during shutdown
        if self._shutdown:
            logger.error(f'Attempted to create connection during shutdown! readonly={readonly}')
            raise RuntimeError('Cannot create new connections during shutdown')

        # Create database file if it does not exist, for write connections
        if not readonly and not self.db_path.exists():
            # Create an initial connection to create the database file
            # Use with statement to ensure closure
            with sqlite3.connect(str(self.db_path)):
                pass

        # Use URI mode for better control
        uri = f"file:{self.db_path}?mode={'ro' if readonly else 'rw'}"
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=self.pool_config.connection_timeout,
            check_same_thread=False,  # Thread safety handled at a higher level
            isolation_level='DEFERRED',  # Better for concurrent access
            factory=ManagedConnection,  # Ensure finalizer-based safety on GC
        )

        try:
            conn.row_factory = sqlite3.Row

            # Apply optimized PRAGMAs for production
            pragmas = [
                ('foreign_keys', 'ON' if settings.storage.sqlite_foreign_keys else 'OFF'),
                ('journal_mode', settings.storage.sqlite_journal_mode),
                ('synchronous', settings.storage.sqlite_synchronous),
                ('temp_store', settings.storage.sqlite_temp_store),
                ('mmap_size', str(settings.storage.sqlite_mmap_size)),
                ('cache_size', str(settings.storage.sqlite_cache_size)),
                ('page_size', str(settings.storage.sqlite_page_size)),
                ('wal_autocheckpoint', str(settings.storage.sqlite_wal_autocheckpoint)),
                ('busy_timeout', str(settings.storage.resolved_busy_timeout_ms)),
            ]

            if not readonly:
                # Writer-specific optimizations
                pragmas.append(('wal_checkpoint', settings.storage.sqlite_wal_checkpoint))

            for pragma, value in pragmas:
                conn.execute(f'PRAGMA {pragma} = {value}')

            self.metrics.total_connections += 1
            self.metrics.active_connections += 1

            # Track all created connections for cleanup
            with self._connection_lock:
                self._all_connections.add(conn)
                self._connection_ids[id(conn)] = f'readonly={readonly}'

            logger.debug(f'Created connection: {id(conn)} (readonly={readonly}), total: {len(self._all_connections)}')

            return conn
        except Exception:
            # Close connection on any error during setup
            with contextlib.suppress(Exception):
                conn.close()
            with self._connection_lock:
                self._all_connections.discard(conn)
            raise

    async def _ensure_writer_connection(self) -> sqlite3.Connection:
        """Ensure writer connection exists and is healthy."""
        loop = asyncio.get_event_loop()

        def _get_writer() -> sqlite3.Connection:
            with self._pool_lock:
                if not self._writer_conn:
                    self._writer_conn = self._create_connection(readonly=False)
                    logger.debug('Created new writer connection')
                return self._writer_conn

        return await loop.run_in_executor(None, _get_writer)

    async def _get_reader_connection(self) -> sqlite3.Connection:
        """Get a reader connection from the pool."""
        loop = asyncio.get_event_loop()

        def _get_reader() -> sqlite3.Connection:
            with self._pool_lock:
                # Rebuild pool of healthy readers, pick one after the scan
                healthy_connections: list[sqlite3.Connection] = []
                chosen: sqlite3.Connection | None = None

                for conn in self._reader_pool[:]:
                    try:
                        # Quick health check
                        conn.execute('SELECT 1')
                        healthy_connections.append(conn)
                        if chosen is None:
                            chosen = conn
                    except Exception:
                        # Close unhealthy connection and remove from pool
                        self._reader_pool.remove(conn)
                        self._safe_close_connection(conn)

                # Update pool with only healthy connections
                self._reader_pool = healthy_connections

                if chosen is not None:
                    return chosen

                # Create new reader if pool not full
                if len(self._reader_pool) < self.pool_config.max_readers:
                    conn = self._create_connection(readonly=True)
                    self._reader_pool.append(conn)
                    logger.debug(f'Created new reader connection (pool size: {len(self._reader_pool)})')
                    return conn

                # Pool full, return writer connection as fallback
                if self._writer_conn:
                    return self._writer_conn

                # Create a temporary connection, track it for cleanup
                temp_conn = self._create_connection(readonly=False)
                with self._connection_lock:
                    self._temporary_connections.add(temp_conn)
                logger.debug('Created temporary reader connection (pool full)')
                return temp_conn

        return await loop.run_in_executor(None, _get_reader)

    async def _close_all_connections(self) -> None:
        """Close all database connections with comprehensive tracking."""
        # Close connections synchronously to avoid race conditions with garbage collection
        # Need both locks to access pools and tracking sets
        with self._pool_lock, self._connection_lock:
            # Create master list of ALL connections to close
            all_conns_to_close: set[sqlite3.Connection] = set()

            # Add all tracked connections
            logger.debug(f'Tracked connections: {len(self._all_connections)}')
            all_conns_to_close.update(self._all_connections)

            logger.debug(f'Temporary connections: {len(self._temporary_connections)}')
            all_conns_to_close.update(self._temporary_connections)

            # Add writer connection
            if self._writer_conn:
                logger.debug(f'Writer connection: {id(self._writer_conn)}')
                all_conns_to_close.add(self._writer_conn)

            # Add all reader pool connections
            logger.debug(f'Reader pool connections: {len(self._reader_pool)}')
            all_conns_to_close.update(self._reader_pool)

            logger.debug(f'Total connections to close: {len(all_conns_to_close)}')
            for conn in all_conns_to_close:
                logger.debug(f'  Connection to close: {id(conn)}')

            # Close ALL connections - don't check if already closed, just close them
            closed_count = 0
            for conn in all_conns_to_close:
                self._safe_close_connection(conn)
                closed_count += 1
                logger.debug(f'Closed connection: {id(conn)}')

            logger.debug(f'Closed {closed_count} connections out of {len(all_conns_to_close)} total')

            # Log any connection IDs that weren't closed
            remaining_ids = set(self._connection_ids.keys())
            remaining_ids.difference_update(id(conn) for conn in all_conns_to_close)
            if remaining_ids:
                logger.warning(f'Connection IDs not in close list: {remaining_ids}')

            # Clear all connection references completely
            self._all_connections.clear()
            self._temporary_connections.clear()
            self._writer_conn = None
            self._reader_pool.clear()

    async def _process_write_queue(self) -> None:
        """Background task to process write requests from the queue."""
        logger.info('Write queue processor started')

        # Use shorter timeout in test environment
        queue_timeout = settings.storage.queue_timeout_test_s if is_test_environment() else settings.storage.queue_timeout_s
        wait_task = None
        shutdown_task = None

        try:
            while not self._shutdown:
                try:
                    # Wait for write request with timeout or shutdown
                    wait_task = asyncio.create_task(self._write_queue.get())
                    shutdown_task = asyncio.create_task(self._shutdown_event.wait())
                    done, pending = await asyncio.wait(
                        [wait_task, shutdown_task],
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=queue_timeout,
                    )

                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        # Ensure cancellation is processed
                        with suppress(asyncio.CancelledError):
                            await task

                    if shutdown_task in done:
                        break

                    if wait_task in done:
                        try:
                            request = await wait_task
                        except asyncio.CancelledError:
                            continue
                        else:
                            # This else block only executes if no exception was raised
                            # At this point, request is guaranteed to be a WriteRequest
                            self.metrics.write_queue_size = self._write_queue.qsize()

                            # Check circuit breaker
                            if self.circuit_breaker.is_open():
                                request.future.set_exception(
                                    Exception('Database circuit breaker is open, too many failures'),
                                )
                                continue

                            # Process write request with retry logic
                            try:
                                result = await self._execute_write_with_retry(request)
                                if not request.future.done():
                                    request.future.set_result(result)
                                self.circuit_breaker.record_success()
                            except Exception as e:
                                if not request.future.done():
                                    request.future.set_exception(e)
                                self.circuit_breaker.record_failure()
                                self.metrics.failed_queries += 1
                                self.metrics.last_error = str(e)
                                self.metrics.last_error_time = time.time()
                    # If we get here and wait_task is not in done, it's a timeout - no writes pending

                except asyncio.CancelledError:
                    logger.info('Write queue processor cancelled')
                    break
                except Exception as e:
                    logger.error(f'Write queue processor error: {e}')
        finally:
            # Clean up any remaining tasks
            try:
                if wait_task and not wait_task.done():
                    wait_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await wait_task
                if shutdown_task and not shutdown_task.done():
                    shutdown_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await shutdown_task
            except RuntimeError:
                # Event loop may be closed, ignore
                pass

        logger.info('Write queue processor stopped')

    async def _execute_write_with_retry(self, request: WriteRequest) -> object:
        """Execute a write request with retry logic."""
        loop = asyncio.get_event_loop()
        last_error = None

        for attempt in range(self.retry_config.max_retries):
            try:
                # Ensure writer connection exists
                writer = await self._ensure_writer_connection()

                # Execute operation
                def _execute(conn: sqlite3.Connection) -> object:
                    result = request.operation(conn, *request.args, **request.kwargs)
                    conn.commit()
                    self.metrics.total_queries += 1
                    return result

                return await loop.run_in_executor(None, _execute, writer)

            except sqlite3.OperationalError as e:
                last_error = e
                if 'database is locked' in str(e):
                    # Calculate backoff delay
                    delay = min(
                        self.retry_config.base_delay * (self.retry_config.backoff_factor**attempt),
                        self.retry_config.max_delay,
                    )

                    # Add jitter if configured
                    if self.retry_config.jitter:
                        delay += random.uniform(0, delay * 0.3)

                    logger.warning(
                        f'Database locked on write, retrying in {delay:.2f}s '
                        f'(attempt {attempt + 1}/{self.retry_config.max_retries})',
                    )
                    await asyncio.sleep(delay)
                    request.retry_count += 1
                else:
                    raise
            except Exception:
                raise

        # Max retries exceeded
        raise last_error or Exception('Max retries exceeded for write operation')

    async def _health_check_loop(self) -> None:
        """Periodic health check for connections."""
        logger.info('Health check loop started')
        shutdown_task = None

        try:
            while not self._shutdown:
                try:
                    # Use wait with timeout for interruptible sleep
                    shutdown_task = asyncio.create_task(self._shutdown_event.wait())
                    done, pending = await asyncio.wait([shutdown_task], timeout=self.pool_config.health_check_interval)

                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        # Ensure cancellation is processed
                        with suppress(asyncio.CancelledError):
                            await task

                    if shutdown_task in done:
                        break

                    await self._perform_health_check()
                except asyncio.CancelledError:
                    logger.info('Health check loop cancelled')
                    break
                except Exception as e:
                    logger.error(f'Health check error: {e}')
        finally:
            # Clean up any remaining tasks
            try:
                if shutdown_task and not shutdown_task.done():
                    shutdown_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await shutdown_task
            except RuntimeError:
                # Event loop may be closed, ignore
                pass

    async def _perform_health_check(self) -> None:
        """Perform health check on all connections."""
        loop = asyncio.get_event_loop()

        def _check() -> None:
            with self._pool_lock:
                # Writer
                if self._writer_conn:
                    try:
                        self._writer_conn.execute('SELECT 1')
                    except Exception:
                        logger.warning('Writer connection unhealthy, closing')
                        self._safe_close_connection(self._writer_conn)
                        self._writer_conn = None
                        self.metrics.failed_connections += 1

                # Readers
                healthy_readers: list[sqlite3.Connection] = []
                for conn in self._reader_pool:
                    try:
                        conn.execute('SELECT 1')
                        healthy_readers.append(conn)
                    except Exception:
                        logger.warning('Reader connection unhealthy, closing')
                        self._safe_close_connection(conn)
                        self.metrics.failed_connections += 1

                self._reader_pool = healthy_readers

        await loop.run_in_executor(None, _check)

        # Update circuit breaker state in metrics
        self.metrics.circuit_state = self.circuit_breaker.get_state()
        self.metrics.consecutive_failures = self.circuit_breaker.failures

    @asynccontextmanager
    async def get_connection(
        self,
        readonly: bool = False,
        allow_write: bool = False,
    ) -> AsyncGenerator[sqlite3.Connection, None]:
        """
        Get a database connection from the pool.

        Args:
            readonly: If True, get a reader connection, otherwise get writer
            allow_write: If True, allow direct writer connection, for migrations and schema

        Yields:
            Database connection

        Raises:
            RuntimeError: If connection manager is shutting down or circuit breaker is open
        """
        if self._shutdown:
            raise RuntimeError('Connection manager is shutting down')

        # Check circuit breaker
        if self.circuit_breaker.is_open():
            raise RuntimeError(
                f'Database circuit breaker is open after {self.circuit_breaker.failures} failures',
            )

        if readonly:
            # Get reader connection
            async with self._reader_semaphore:
                conn = await self._get_reader_connection()
                # Check if it's a temporary connection that needs cleanup
                is_temporary = False
                with self._connection_lock:
                    is_temporary = conn in self._temporary_connections

                try:
                    yield conn
                    self.circuit_breaker.record_success()
                except Exception:
                    self.circuit_breaker.record_failure()
                    raise
                finally:
                    # Clean up temporary connections after use
                    if is_temporary:
                        # Clean up synchronously to avoid race conditions with garbage collection
                        with self._connection_lock:
                            if conn in self._temporary_connections:
                                self._temporary_connections.remove(conn)
                        self._safe_close_connection(conn)

        elif allow_write:
            # Direct write connection with lock protection
            async with self._writer_lock:
                writer = await self._ensure_writer_connection()
                try:
                    yield writer
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, writer.commit)
                    self.circuit_breaker.record_success()
                except Exception:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, writer.rollback)
                    self.circuit_breaker.record_failure()
                    raise
        else:
            # Use write queue for normal write operations
            raise RuntimeError(
                'Direct write connections not allowed. Use execute_write() method or set allow_write=True.',
            )

    async def execute_write(
        self,
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a write operation through the write queue.

        Args:
            operation: Function to execute with connection as first argument
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            RuntimeError: If connection manager is shutting down
        """
        if self._shutdown:
            raise RuntimeError('Connection manager is shutting down')

        # Create future for result
        future: asyncio.Future[T] = asyncio.Future()

        # Create and queue request
        request = WriteRequest(operation, args, kwargs, future)
        await self._write_queue.put(request)

        # Update metrics
        self.metrics.write_queue_size = self._write_queue.qsize()

        # Wait for result
        return await future

    async def execute_read(
        self,
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a read operation with a reader connection.

        Args:
            operation: Function to execute with connection as first argument
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the operation
        """
        async with self.get_connection(readonly=True) as conn:
            loop = asyncio.get_event_loop()

            def _execute() -> T:
                result = operation(conn, *args, **kwargs)
                self.metrics.total_queries += 1
                return result

            try:
                return await loop.run_in_executor(None, _execute)
            except Exception:
                self.metrics.failed_queries += 1
                raise

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics for monitoring."""
        return {
            'total_connections': self.metrics.total_connections,
            'active_connections': self.metrics.active_connections,
            'failed_connections': self.metrics.failed_connections,
            'total_queries': self.metrics.total_queries,
            'failed_queries': self.metrics.failed_queries,
            'write_queue_size': self.metrics.write_queue_size,
            'circuit_state': self.metrics.circuit_state.value,
            'consecutive_failures': self.metrics.consecutive_failures,
            'last_error': self.metrics.last_error,
            'last_error_time': self.metrics.last_error_time,
        }

    # Best-effort cleanup if shutdown was not called

    def _close_all_connections_sync(self) -> None:
        """Synchronous cleanup used by __del__, safe in interpreter shutdown."""
        with self._pool_lock, self._connection_lock:
            conns: set[sqlite3.Connection] = set(self._all_connections)
            if self._writer_conn:
                conns.add(self._writer_conn)
            conns.update(self._temporary_connections)
            conns.update(self._reader_pool)

            for conn in conns:
                self._safe_close_connection(conn)

            self._all_connections.clear()
            self._temporary_connections.clear()
            self._reader_pool.clear()
            self._writer_conn = None

    def __del__(self) -> None:
        # Last safety net, if user code forgot to call shutdown
        with contextlib.suppress(Exception):
            if not getattr(self, '_shutdown_complete', None) or not self._shutdown_complete.is_set():
                self._close_all_connections_sync()


# Global instance for singleton pattern
_manager_instance: DatabaseConnectionManager | None = None
_manager_lock = Lock()


def get_connection_manager(
    db_path: Path | str | None = None,
    pool_config: PoolConfig | None = None,
    retry_config: RetryConfig | None = None,
    force_new: bool = False,
) -> DatabaseConnectionManager:
    """
    Get or create the global connection manager instance.

    Args:
        db_path: Database file path, required on first call
        pool_config: Connection pool configuration
        retry_config: Retry logic configuration
        force_new: If True, return a brand-new instance, not stored globally

    Returns:
        Connection manager instance

    Raises:
        ValueError: If a required `db_path` is not provided on first call
            or when creating a new instance.
    """
    global _manager_instance
    with _manager_lock:
        # In test environments, always return a fresh instance for isolation
        if force_new or is_test_environment():
            if db_path is None:
                raise ValueError('db_path required when creating new instance')
            return DatabaseConnectionManager(
                db_path=db_path,
                pool_config=pool_config,
                retry_config=retry_config,
            )

        if _manager_instance is None:
            if db_path is None:
                raise ValueError('db_path required for first initialization')
            _manager_instance = DatabaseConnectionManager(
                db_path=db_path,
                pool_config=pool_config,
                retry_config=retry_config,
            )
        return _manager_instance


def reset_connection_manager() -> None:
    """
    Reset the global connection manager instance.

    This is primarily used for testing to ensure clean state between tests.
    Should not be used in production code.
    """
    global _manager_instance
    with _manager_lock:
        _manager_instance = None
        logger.debug('Connection manager instance reset')
