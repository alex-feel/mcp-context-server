"""Instance state and shared accounting of the SQLite backend.

Configuration, connection tracking, the write queue and shutdown primitives, the circuit
breaker and metrics, the failure bookkeeping shared by the read, write and transaction paths,
and the published health metrics.
"""

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from threading import RLock
from typing import TYPE_CHECKING
from typing import Any

from app.backends.sqlite_backend.config import PoolConfig
from app.backends.sqlite_backend.resilience import CircuitBreaker
from app.backends.sqlite_backend.resilience import RetryConfig
from app.settings import get_settings

if TYPE_CHECKING:
    # write_queue imports this module, so the queue's element type is resolved for type checkers only.
    from app.backends.sqlite_backend.write_queue import WriteRequest

settings = get_settings()


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


class SQLiteBackendCore:
    """Instance state and shared accounting of the SQLite backend.

    Owns the database path, pool and retry configuration, connection tracking, write queue,
    synchronization primitives, circuit breaker, metrics and shutdown events that the
    connection, write-queue, health, connection-scope, transaction and lifecycle mixins
    share, plus the failure bookkeeping and the published metrics.
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
        # Monotonic timestamp of the last writer acquisition, driving the
        # POOL_IDLE_TIMEOUT_S recycling in the health-check loop. Readers are
        # per-use temporary connections closed in get_connection's finally, so
        # only the process-lifetime writer can sit idle.
        self._writer_last_used: float = time.monotonic()
        self._reader_semaphore: asyncio.Semaphore | None = None

        # Write queue for serialization
        self._write_queue: asyncio.Queue[WriteRequest] | None = None
        self._write_processor_task: asyncio.Task[None] | None = None

        # Synchronization primitives
        self._writer_lock: asyncio.Lock | None = None
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
        self._shutdown_event: asyncio.Event | None = None
        self._shutdown_complete: asyncio.Event | None = None

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier for SQLite.

        Returns:
            str: 'sqlite' identifying this as a SQLite backend
        """
        return 'sqlite'

    def _record_query_failure(self, error: BaseException) -> None:
        """Record the failure METRICS for a database fault, without the breaker charge.

        Counts the fault in ``metrics.failed_queries`` and stores its text and
        timestamp in ``metrics.last_error`` / ``metrics.last_error_time`` -- the
        three fields ``get_metrics()`` publishes as ``connection_metrics``. A
        fault that moves the failure count but leaves ``last_error`` untouched
        gives an operator a count with no diagnostic text, or (worse) an OLD
        message beside fresh failures, actively misdirecting diagnosis.

        Args:
            error: The fault being recorded; its string form becomes
                ``metrics.last_error``.
        """
        self.metrics.failed_queries += 1
        self.metrics.last_error = str(error)
        self.metrics.last_error_time = time.time()

    def _record_charged_failure(self, error: BaseException) -> None:
        """Record one charged failure: the failure metrics plus the breaker charge.

        The single bookkeeping site for a charged database fault, shared by the
        write-queue arm, the connection-establishment wrapper, both arms of
        ``get_connection`` and ``begin_transaction`` so the sites cannot drift.
        The PostgreSQL backend carries the same helper under the same name: a
        charged fault that moves the circuit breaker but skips
        ``metrics.failed_queries`` / ``last_error`` / ``last_error_time`` leaves
        dashboards keyed on those fields reporting a healthy, error-free database
        while the breaker is counting the outage.

        Args:
            error: The exception being charged; its string form becomes
                ``metrics.last_error``.
        """
        self._record_query_failure(error)
        self.circuit_breaker.record_failure()

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics for monitoring.

        Returns:
            Mapping published to clients as ``connection_metrics``. ``backend_type``
            and ``pool_size`` are the two keys the shared cross-backend contract
            guarantees, so a client can identify the backend and read one pool
            bound without branching on backend-specific keys; the SQLite pool-size
            analogue is the reader-pool bound (SQLite serializes writes onto a
            single writer connection, so the concurrency bound IS the reader
            count). The remaining keys are backend-specific.
        """
        return {
            'backend_type': self.backend_type,
            'pool_size': self.pool_config.max_readers,
            'total_connections': self.metrics.total_connections,
            'active_connections': self.metrics.active_connections,
            'failed_connections': self.metrics.failed_connections,
            'total_queries': self.metrics.total_queries,
            'failed_queries': self.metrics.failed_queries,
            'write_queue_size': self.metrics.write_queue_size,
            'circuit_state': self.circuit_breaker.peek_state().value,
            'consecutive_failures': self.circuit_breaker.failures,
            'last_error': self.metrics.last_error,
            'last_error_time': self.metrics.last_error_time,
        }
