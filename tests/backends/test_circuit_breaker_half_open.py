"""Half-open admission control on both backends' circuit breakers.

While the breaker is half-open (DEGRADED) it must let only a handful of probe
calls through per recovery window. Otherwise every request that piled up during
the outage is released at the dead database the moment the recovery timeout
elapses -- on SQLite each opening a fresh reader connection, on PostgreSQL each
stalling for the full pool-acquire budget -- and the stampede repeats on every
window for the whole outage, which is precisely what the half-open state exists
to prevent.

Admission and outcome are different events, so they need different counters: a
single counter advanced only by record_success() (and zeroed at the promotion
threshold) is provably always below the gate, so the gate never closes.

Each backend module defines its own CircuitBreaker and ConnectionState, so the
cases are concrete per backend.
"""

import time

import pytest

from app.backends.postgresql_backend import CircuitBreaker as PgBreaker
from app.backends.postgresql_backend import ConnectionState as PgState
from app.backends.sqlite_backend import CircuitBreaker as SqBreaker
from app.backends.sqlite_backend import ConnectionState as SqState


def _tripped_sqlite_breaker(max_calls: int = 3, recovery_timeout: float = 10.0) -> SqBreaker:
    """Build a SQLite breaker that is FAILED with its recovery window elapsed.

    Args:
        max_calls: Half-open admission budget.
        recovery_timeout: Recovery window length in seconds.

    Returns:
        The tripped breaker, ready for its first half-open admission.
    """
    breaker = SqBreaker(failure_threshold=2, recovery_timeout=recovery_timeout, half_open_max_calls=max_calls)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == SqState.FAILED
    breaker.last_failure_time = time.time() - (recovery_timeout + 1)
    return breaker


async def _tripped_pg_breaker(max_calls: int = 3, recovery_timeout: float = 10.0) -> PgBreaker:
    """Build a PostgreSQL breaker that is FAILED with its recovery window elapsed.

    Args:
        max_calls: Half-open admission budget.
        recovery_timeout: Recovery window length in seconds.

    Returns:
        The tripped breaker, ready for its first half-open admission.
    """
    breaker = PgBreaker(failure_threshold=2, recovery_timeout=recovery_timeout, half_open_max_calls=max_calls)
    await breaker.record_failure()
    await breaker.record_failure()
    assert breaker.state == PgState.FAILED
    breaker.last_failure_time = time.time() - (recovery_timeout + 1)
    return breaker


class TestSqliteHalfOpenAdmission:
    """The SQLite breaker admits at most half_open_max_calls probes per window."""

    def test_admissions_are_capped_while_half_open(self) -> None:
        """The first max_calls calls are admitted; the next one is refused."""
        breaker = _tripped_sqlite_breaker(max_calls=3)

        admitted = [not breaker.is_open() for _ in range(3)]

        assert admitted == [True, True, True]
        assert breaker.state == SqState.DEGRADED
        # The whole point: the fourth caller in the same window is turned away
        # instead of being sent into a database that has not answered yet.
        assert breaker.is_open() is True
        assert breaker.is_open() is True

    def test_successful_probes_close_the_circuit(self) -> None:
        """max_calls successful probes promote the breaker back to HEALTHY."""
        breaker = _tripped_sqlite_breaker(max_calls=3)

        for _ in range(3):
            assert breaker.is_open() is False
            breaker.record_success()

        assert breaker.state == SqState.HEALTHY
        assert breaker.failures == 0
        assert breaker.is_open() is False

    def test_a_failed_probe_reopens_the_circuit(self) -> None:
        """A probe that fails sends the breaker straight back to FAILED."""
        breaker = _tripped_sqlite_breaker(max_calls=3)

        assert breaker.is_open() is False
        breaker.record_failure()

        assert breaker.state == SqState.FAILED
        assert breaker.is_open() is True

    def test_a_window_without_a_verdict_re_arms_instead_of_locking_out(self) -> None:
        """Probes that report no outcome cannot block the breaker forever.

        Several call paths are deliberately exempt from breaker accounting
        (normal control flow, self-clearing lock contention, cancellation), so a
        window's whole budget can be spent without a single success or failure
        being recorded. That must not leave the breaker permanently shut.
        """
        breaker = _tripped_sqlite_breaker(max_calls=2, recovery_timeout=10.0)

        assert breaker.is_open() is False
        assert breaker.is_open() is False
        assert breaker.is_open() is True

        breaker.half_open_started_at = time.time() - 11.0

        assert breaker.is_open() is False
        assert breaker.state == SqState.DEGRADED

    def test_recovery_transition_starts_a_fresh_budget(self) -> None:
        """get_state()'s FAILED -> DEGRADED transition resets both counters."""
        breaker = _tripped_sqlite_breaker(max_calls=2)
        breaker.half_open_admissions = 99
        breaker.half_open_successes = 99

        assert breaker.get_state() == SqState.DEGRADED
        assert breaker.half_open_admissions == 0
        assert breaker.half_open_successes == 0
        assert breaker.is_open() is False


class TestPostgresqlHalfOpenAdmission:
    """The PostgreSQL breaker carries the identical admission control."""

    @pytest.mark.asyncio
    async def test_admissions_are_capped_while_half_open(self) -> None:
        """The first max_calls calls are admitted; the next one is refused."""
        breaker = await _tripped_pg_breaker(max_calls=3)

        admitted = [not await breaker.is_open() for _ in range(3)]

        assert admitted == [True, True, True]
        assert breaker.state == PgState.DEGRADED
        assert await breaker.is_open() is True
        assert await breaker.is_open() is True

    @pytest.mark.asyncio
    async def test_successful_probes_close_the_circuit(self) -> None:
        """max_calls successful probes promote the breaker back to HEALTHY."""
        breaker = await _tripped_pg_breaker(max_calls=3)

        for _ in range(3):
            assert await breaker.is_open() is False
            await breaker.record_success()

        assert breaker.state == PgState.HEALTHY
        assert breaker.failures == 0
        assert await breaker.is_open() is False

    @pytest.mark.asyncio
    async def test_a_failed_probe_reopens_the_circuit(self) -> None:
        """A probe that fails sends the breaker straight back to FAILED."""
        breaker = await _tripped_pg_breaker(max_calls=3)

        assert await breaker.is_open() is False
        await breaker.record_failure()

        assert breaker.state == PgState.FAILED
        assert await breaker.is_open() is True

    @pytest.mark.asyncio
    async def test_a_window_without_a_verdict_re_arms_instead_of_locking_out(self) -> None:
        """Probes that report no outcome cannot block the breaker forever."""
        breaker = await _tripped_pg_breaker(max_calls=2, recovery_timeout=10.0)

        assert await breaker.is_open() is False
        assert await breaker.is_open() is False
        assert await breaker.is_open() is True

        breaker.half_open_started_at = time.time() - 11.0

        assert await breaker.is_open() is False
        assert breaker.state == PgState.DEGRADED

    @pytest.mark.asyncio
    async def test_recovery_transition_starts_a_fresh_budget(self) -> None:
        """get_state()'s FAILED -> DEGRADED transition resets both counters."""
        breaker = await _tripped_pg_breaker(max_calls=2)
        breaker.half_open_admissions = 99
        breaker.half_open_successes = 99

        assert await breaker.get_state() == PgState.DEGRADED
        assert breaker.half_open_admissions == 0
        assert breaker.half_open_successes == 0
        assert await breaker.is_open() is False
