"""Connection health states, retry configuration and the circuit breaker of the SQLite backend."""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from threading import RLock

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection health states for circuit breaker pattern."""

    HEALTHY = 'healthy'
    DEGRADED = 'degraded'
    FAILED = 'failed'


@dataclass
class RetryConfig:
    """Configuration for retry logic with exponential backoff."""

    max_retries: int = 5
    base_delay: float = 0.5
    max_delay: float = 10.0
    jitter: bool = True
    backoff_factor: float = 2.0


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
        # Half-open bookkeeping, split into two counters because ADMISSION and
        # OUTCOME are different events: half_open_admissions is advanced by
        # is_open() for every probe it lets through and bounds how many probes
        # reach a database that may still be dead, while half_open_successes is
        # advanced by record_success() and drives promotion back to HEALTHY. One
        # counter cannot do both -- incrementing it only on success (and zeroing
        # it at the promotion threshold) keeps it permanently below the gate, so
        # the gate never closes and every waiting caller stampedes the dead
        # database on each recovery window.
        self.half_open_admissions = 0
        self.half_open_successes = 0
        self.half_open_started_at: float | None = None
        self._lock = RLock()

    def _open_half_open_window(self, now: float) -> None:
        """Enter DEGRADED with a fresh probe budget. The caller holds the lock.

        Args:
            now: Current ``time.time()`` value, used as the window start.
        """
        self.state = ConnectionState.DEGRADED
        self.half_open_admissions = 0
        self.half_open_successes = 0
        self.half_open_started_at = now

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            if self.state == ConnectionState.DEGRADED:
                self.half_open_successes += 1
                if self.half_open_successes >= self.half_open_max_calls:
                    self.state = ConnectionState.HEALTHY
                    self.failures = 0
                    self.half_open_admissions = 0
                    self.half_open_successes = 0
                    self.half_open_started_at = None
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
        """Check if circuit is open, meaning we should block calls.

        While DEGRADED (half-open) at most ``half_open_max_calls`` calls are
        ADMITTED per recovery window, so a still-dead database receives a handful
        of probes instead of every request that piled up during the outage. A
        window that elapses without a verdict (probes that neither succeeded nor
        failed, e.g. calls exempted from breaker accounting) re-arms the budget
        rather than blocking forever.

        Returns:
            True when the call must be rejected.
        """
        with self._lock:
            if self.state == ConnectionState.HEALTHY:
                return False

            now = time.time()
            if self.state == ConnectionState.FAILED:
                if self.last_failure_time is None or (now - self.last_failure_time) <= self.recovery_timeout:
                    return True
                self._open_half_open_window(now)
                logger.info('Circuit breaker entering DEGRADED state for recovery')

            # DEGRADED state, allow a bounded number of probe calls per window
            if self.half_open_admissions >= self.half_open_max_calls:
                started_at = self.half_open_started_at
                if started_at is not None and (now - started_at) <= self.recovery_timeout:
                    return True
                # The probe budget was spent without any probe reporting an
                # outcome; start a new window instead of rejecting forever.
                self.half_open_admissions = 0
                self.half_open_started_at = now
            self.half_open_admissions += 1
            return False

    def get_state(self) -> ConnectionState:
        """Get current circuit state."""
        with self._lock:
            # Check if we should transition from FAILED to DEGRADED
            if self.state == ConnectionState.FAILED and self.last_failure_time:
                now = time.time()
                if (now - self.last_failure_time) > self.recovery_timeout:
                    self._open_half_open_window(now)
            return self.state

    def peek_state(self) -> ConnectionState:
        """Recovery-aware circuit state for synchronous, advisory reads.

        Mirrors the FAILED -> DEGRADED recovery transition that get_state() and
        is_open() apply once recovery_timeout elapses, WITHOUT mutating state, so
        get_metrics() reports live recovery behavior instead of a value that only
        refreshes on the periodic health check.

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
