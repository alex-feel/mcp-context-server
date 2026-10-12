"""Instance state and shared accounting of the PostgreSQL backend.

Configuration, pool, circuit breaker and metrics, the charged-failure bookkeeping and the log
record of a refused statement shared by the acquire, read, write and transaction paths, and the
published health metrics.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import asyncpg

from app.backends.postgresql_backend.resilience import CircuitBreaker
from app.backends.postgresql_backend.resilience import RetryConfig
from app.backends.postgresql_backend.session import CredentialOverride
from app.backends.postgresql_backend.session import request_connection_string
from app.backends.postgresql_backend.session_profiles import SessionProfile
from app.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def operation_name(operation: object) -> str:
    """Return the name a log uses for an operation callable.

    Args:
        operation: The callable an entry point runs.

    Returns:
        Its qualified name, or its type's name when it has none (a ``functools.partial``).
    """
    qualified = getattr(operation, '__qualname__', None)
    return qualified if isinstance(qualified, str) else type(operation).__name__


@dataclass
class ConnectionMetrics:
    """Metrics for monitoring connection health and performance."""

    total_queries: int = 0
    failed_queries: int = 0
    last_error: str | None = None
    last_error_time: float | None = None


class PostgreSQLBackendCore:
    """Instance state and shared accounting of the PostgreSQL backend.

    Owns the connection string, credential override, retry configuration, session profile,
    pool, circuit breaker and metrics that the provisioning, lifecycle, operations and
    transaction mixins share, plus the charged-failure and refused-statement bookkeeping and
    the published metrics.
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
        provision_vector: bool | None = None,
        *,
        profile: SessionProfile,
        credentials: CredentialOverride,
    ) -> None:
        # Without an explicit connection string the backend dials the request connection settings
        if connection_string is None:
            connection_string = request_connection_string(settings)

        self.connection_string = connection_string

        # The login every connection uses in place of the connection string's own, field by
        # field: the pool and the one-off provisioning connections pass it to asyncpg, whose
        # explicit user and password win over the DSN's.
        self._credentials = credentials

        # How every transaction presents the caller's identity to the database, which
        # relation each plain read names and which session statements each connection runs:
        # the entry points hand each transaction's access scope to profile.bind,
        # readable_relation asks the profile, and every connection path appends its
        # session statements to the server's own.
        self._profile = profile

        # Explicit pgvector-provisioning override. The migration CLI knows
        # up front whether the target will carry the fp32 vector layout
        # (with_semantic), so it bypasses the settings-and-probe resolution
        # in initialize() -- which reads the CLI process's env, not the
        # target's actual needs -- with a deterministic decision. None means
        # resolve normally at initialize() time.
        self._provision_vector_override = provision_vector

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

    @staticmethod
    def _log_swallowed_release_failure(error: BaseException) -> None:
        """Log a release-phase failure that follows a completed operation.

        Returning the pooled connection runs the pool's reset callback, and
        asyncpg terminates the connection and re-raises when that fails (a
        failover, a server restart, ``pg_terminate_backend``, a partition). The
        body's work has ALREADY completed at that point -- including a COMMITted
        transaction -- so the exception is charged and swallowed rather than
        propagated: reporting a landed write as failed sends the caller into a
        retry that re-runs it.

        Args:
            error: The release-phase failure being swallowed.
        """
        logger.warning(f'Releasing the pooled connection failed after the operation completed: {error}')

    @staticmethod
    def _log_refused_statement(error: asyncpg.exceptions.InsufficientPrivilegeError, operation: str) -> None:
        """Log a statement the database refused under its privileges, a policy or a trigger.

        Startup verifies the privileges, so a refusal at run time means the application
        and the policy disagree. It is not a fault of the database, so the entry points
        neither retry it nor charge the circuit breaker, which would otherwise reject every
        caller over one disagreement; this ERROR record is how it surfaces. It carries the
        SQLSTATE, the operation's name and the server's primary message only: the DETAIL
        and HINT fields, which can describe the refused row, are never logged.

        Args:
            error: The refusal.
            operation: The name of the refused operation (:func:`operation_name`), or of
                the entry point when no single callable ran the statement.
        """
        message = error.args[0] if error.args else ''
        logger.error(f'PostgreSQL refused a statement in {operation} (SQLSTATE {error.sqlstate}): {message}')

    async def _record_charged_failure(self, error: BaseException) -> None:
        """Record one charged failure: the breaker charge plus the failure metrics.

        The single bookkeeping site for a charged database fault, shared by the
        acquire-phase arms of ``get_connection`` and ``begin_transaction`` and
        the write-path arms of ``execute_write`` so the sites cannot drift: a
        charged fault that skips ``metrics.failed_queries``, ``last_error``, and
        ``last_error_time`` leaves dashboards keyed on those fields reporting a
        healthy, error-free database while the circuit breaker is counting the
        outage.

        Args:
            error: The exception being charged; its string form becomes
                ``metrics.last_error``.
        """
        self.metrics.failed_queries += 1
        self.metrics.last_error = str(error)
        self.metrics.last_error_time = time.time()
        await self.circuit_breaker.record_failure()

    def get_metrics(self) -> dict[str, Any]:
        """Get backend health metrics and statistics.

        Returns:
            Mapping published to clients as ``connection_metrics``. ``backend_type``
            and ``pool_size`` are the two keys the shared cross-backend contract
            guarantees; here ``pool_size`` is the pool's CURRENT size and is
            therefore present only once the pool exists (before initialize(), and
            after shutdown, there is no pool to size). ``pool_idle``,
            ``pool_min_size`` and ``pool_max_size`` share that condition, and the
            pgpool / session-pooler keys appear only once their detection ran. The
            rest are always present. Every key is declared in
            ``ConnectionMetricsDict``, which drives the advertised outputSchema.
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
        # the SQLite backend.
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
