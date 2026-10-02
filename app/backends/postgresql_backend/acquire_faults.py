"""Fault typing for pool acquires.

The typed establishment timeout raised by the pool's connect callable, and the per-acquire
tracker that records interrupted connection preparation and swallowed release faults, so a
bare acquire TimeoutError can be told apart from pool saturation.
"""

import asyncio
import contextvars
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import asyncpg


class ConnectionEstablishmentTimeoutError(TimeoutError):
    """Establishing a NEW database connection for the pool timed out.

    Raised by ``connect_pool_connection`` (the pool's ``connect`` callable) in
    place of the bare TimeoutError ``asyncpg.connect`` raises when dialing a new
    connection exceeds POSTGRESQL_CONNECT_TIMEOUT_S. ``pool.acquire`` re-raises
    connect failures verbatim, where a bare TimeoutError is indistinguishable
    from the acquire-deadline TimeoutError of a saturated pool -- a capacity
    signal the backend deliberately leaves uncharged on the circuit breaker.
    Typing the establishment timeout keeps the two fault classes separable
    WITHOUT inferring them from elapsed wall time (which misclassifies whenever
    the connect budget is configured close to the pool budget). Subclasses
    TimeoutError so broad timeout handling still sees it.

    It covers only the case where the CONNECT budget wins the race. When the
    ACQUIRE budget wins first, asyncpg cancels the in-flight dial instead
    (``Pool._acquire`` wraps the queue wait AND the connect callable in ONE
    ``wait_for``), so no establishment TimeoutError is ever constructed and the
    acquire raises a BARE TimeoutError. That case is covered by the
    ``_AcquireTracker`` recorded out of band around the dial, which the
    bare-TimeoutError arms consult so an unreachable database still charges the
    breaker under every budget combination.
    """


@dataclass
class _AcquireTracker:
    """Out-of-band record of what one pool acquire's connection preparation did.

    Published through the ``_acquire_tracker`` ContextVar by ``track_acquire()``
    around a ``pool.acquire()``, and written by all three pool callbacks asyncpg
    invokes INLINE in the acquiring task (so each reads the acquirer's own
    tracker, never a concurrent acquirer's): the ``connect`` dial, the ``init``
    callback and the ``setup`` callback.

    The acquire-phase handlers cannot infer the fault class from the exception
    alone: when the acquire deadline expires while any of those phases is in
    flight, asyncpg cancels it, so the phase receives CancelledError rather than
    TimeoutError and the acquire surfaces a BARE TimeoutError identical in type
    to genuine pool saturation. Recording the interruption here keeps the two
    apart: a bare TimeoutError with ``interrupted`` set means the acquire was
    talking to an unresponsive database (charged), and one without it is real
    saturation (uncharged capacity signal). Covering ``init``/``setup`` and not
    just the dial matters because a WARM pool blackholed mid-flight never dials
    at all -- it reuses a holder and hangs in one of those two callbacks.

    The same tracker also carries the RELEASE outcome back to a caller that
    suppressed ``get_connection``'s own breaker accounting (``execute_write``
    passes ``record_breaker=False`` so a retried write records exactly one
    outcome). ``get_connection`` charges a release fault and then SWALLOWS it,
    because the body's work -- including a COMMITted write -- already completed;
    without this flag the caller could not tell a clean release from a swallowed
    one and would credit a breaker SUCCESS that cancels the charge out (in the
    HEALTHY state ``record_success`` also decrements accumulated failures), so a
    connection dying on release would move the health counter the wrong way.

    Attributes:
        interrupted: True once connection preparation for this acquire (the dial,
            the init callback or the setup callback) was cancelled before the
            connection could be handed to the caller.
        release_failed: True once a release-phase fault after a clean body was
            charged and swallowed, so the caller must not credit a success for
            this acquire.
    """

    interrupted: bool = False
    release_failed: bool = False


_acquire_tracker: contextvars.ContextVar[_AcquireTracker | None] = contextvars.ContextVar(
    'mcp_context_server_acquire_tracker',
    default=None,
)


def record_preparation_interrupted() -> None:
    """Record on the current acquire's tracker that connection preparation was cut off.

    Called from every pool callback that runs INSIDE ``pool.acquire()`` and can be
    cancelled by the acquire deadline: the ``connect`` dial, the ``init`` callback
    (whose pgvector probe and codec registration are real server round-trips) and
    the ``setup`` callback (whose ``SET statement_timeout`` is one more). All three
    mean the acquire was talking to the database rather than waiting in the queue,
    so the resulting BARE TimeoutError is an unreachable database (charged), not
    pool saturation (uncharged capacity signal).
    """
    tracker = _acquire_tracker.get()
    if tracker is not None:
        tracker.interrupted = True


def charge_cancelled_preparation(
    callback: 'Callable[[asyncpg.Connection], Awaitable[None]]',
) -> 'Callable[[asyncpg.Connection], Awaitable[None]]':
    """Wrap a pool connection callback so a cancellation is recorded before it re-raises.

    The CancelledError itself is re-raised UNCHANGED: swallowing it would corrupt
    the cancellation bookkeeping ``asyncio.wait_for`` relies on to convert it into
    the acquire's TimeoutError.

    Args:
        callback: The pool ``init``/``setup`` callable to wrap.

    Returns:
        A callable with identical behavior that also records the interruption.
    """

    async def _wrapped(conn: 'asyncpg.Connection') -> None:
        try:
            await callback(conn)
        except asyncio.CancelledError:
            record_preparation_interrupted()
            raise

    return _wrapped


@contextmanager
def track_acquire() -> Iterator[_AcquireTracker]:
    """Publish a ``_AcquireTracker`` for the acquire performed inside the block.

    An already-published tracker is REUSED rather than shadowed, so an outer
    scope (``execute_write``, which acquires through ``get_connection``) and the
    inner scope observe the same dial instead of the inner one hiding the
    interruption from the arm that must charge it.

    Yields:
        The tracker covering dials made inside the block.
    """
    existing = _acquire_tracker.get()
    if existing is not None:
        yield existing
        return
    tracker = _AcquireTracker()
    token = _acquire_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _acquire_tracker.reset(token)
