"""Unit tests for CircuitBreaker.peek_state() on both backends.

get_metrics() reports the breaker state synchronously, so it uses peek_state(),
which must apply the same lazy FAILED -> DEGRADED recovery transition that the
get_state()/is_open() paths apply once recovery_timeout elapses -- WITHOUT
mutating state -- so the reported state matches live behavior and the two backends
stay consistent (rather than reporting a stale 'healthy'/'failed').

Each backend module defines its own CircuitBreaker and ConnectionState, so the
two tests are concrete per backend.
"""

import time

from app.backends.postgresql_backend import CircuitBreaker as PgBreaker
from app.backends.postgresql_backend import ConnectionState as PgState
from app.backends.sqlite_backend import CircuitBreaker as SqBreaker
from app.backends.sqlite_backend import ConnectionState as SqState


def test_pg_peek_state_recovery_aware() -> None:
    cb = PgBreaker(recovery_timeout=10.0)
    assert cb.peek_state() == PgState.HEALTHY
    cb.state = PgState.FAILED
    cb.last_failure_time = time.time()
    assert cb.peek_state() == PgState.FAILED  # within recovery window
    cb.last_failure_time = time.time() - 100  # recovery window elapsed
    assert cb.peek_state() == PgState.DEGRADED
    assert cb.state == PgState.FAILED  # peek_state must not mutate


def test_sqlite_peek_state_recovery_aware() -> None:
    cb = SqBreaker(recovery_timeout=10.0)
    assert cb.peek_state() == SqState.HEALTHY
    cb.state = SqState.FAILED
    cb.last_failure_time = time.time()
    assert cb.peek_state() == SqState.FAILED  # within recovery window
    cb.last_failure_time = time.time() - 100  # recovery window elapsed
    assert cb.peek_state() == SqState.DEGRADED
    assert cb.state == SqState.FAILED  # peek_state must not mutate
