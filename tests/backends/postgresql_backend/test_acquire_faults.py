"""Tests for app/backends/postgresql_backend/acquire_faults.py: charging a cancelled dial.

A bare acquire TimeoutError whose deadline CANCELLED an in-flight dial is an
unreachable database, not pool saturation. asyncpg wraps the queue wait and
the connect callable in one deadline, so the cancelled dial never produces a
typed establishment timeout and the two fault classes become identical by
exception type alone.
"""

import asyncio
import contextlib
import socket
import unittest.mock
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.backends.postgresql_backend import PostgreSQLBackend
from tests.backends.postgresql_backend._builders import build_backend
from tests.backends.postgresql_backend._builders import fast_retries
from tests.helpers import rebind_package_settings


async def _dial_cancelled_then_acquire_timeout() -> None:
    """Reproduce the shape asyncpg produces when an acquire deadline kills a dial.

    ``Pool._acquire`` wraps the queue wait AND the connect callable in ONE
    ``wait_for``, so an expiring acquire budget cancels the in-flight dial: the
    connect callable receives CancelledError (never a TimeoutError it could
    type), and the acquire surfaces a BARE TimeoutError indistinguishable from
    genuine pool saturation.

    Raises:
        TimeoutError: Always, after the dial has been cancelled.
    """
    from app.backends.postgresql_backend.pool_callbacks import connect_pool_connection

    with (
        unittest.mock.patch(
            'asyncpg.connect',
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError(),
        ),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await connect_pool_connection('postgresql://localhost:5432/testdb')
    raise TimeoutError('pool acquire timed out')


def _backend_with_cancelled_dial() -> PostgreSQLBackend:
    """Build a backend whose acquire cancels its dial and then times out.

    Returns:
        A backend wired to a pool reproducing the cancelled-dial acquire.
    """
    backend = build_backend()

    class _CancelDialThenTimeout:
        async def __aenter__(self) -> object:
            await _dial_cancelled_then_acquire_timeout()
            raise AssertionError('unreachable')

        async def __aexit__(self, *_exc: object) -> bool:
            return False

    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=lambda **_kwargs: _CancelDialThenTimeout())
    backend._pool = pool
    return backend


def _backend_with_saturation_timeout() -> PostgreSQLBackend:
    """Build a backend whose acquire times out without attempting any dial.

    Returns:
        A backend wired to a pool reproducing a saturated-pool acquire.
    """
    backend = build_backend()

    class _FailOnEnter:
        async def __aenter__(self) -> object:
            raise TimeoutError('pool acquire timed out')

        async def __aexit__(self, *_exc: object) -> bool:
            return False

    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=lambda **_kwargs: _FailOnEnter())
    backend._pool = pool
    return backend


class TestCancelledDialCharging:
    """A bare acquire TimeoutError whose deadline killed a dial is charged.

    The typed establishment timeout only covers the case where the CONNECT
    budget wins the race. When the ACQUIRE budget wins, asyncpg cancels the dial
    instead, so no typed error is ever constructed and an unreachable
    (blackholed) database is indistinguishable from a saturated pool by type
    alone -- leaving the breaker closed, failed_queries at zero and last_error
    null for the entire outage.
    """

    @pytest.mark.asyncio
    async def test_connect_wrapper_records_a_cancelled_dial(self) -> None:
        """The connect callable records the interruption and re-raises unchanged."""
        from app.backends.postgresql_backend.acquire_faults import track_acquire
        from app.backends.postgresql_backend.pool_callbacks import connect_pool_connection

        with track_acquire() as acquire_state:
            assert acquire_state.interrupted is False
            with (
                unittest.mock.patch(
                    'asyncpg.connect',
                    new_callable=AsyncMock,
                    side_effect=asyncio.CancelledError(),
                ),
                pytest.raises(asyncio.CancelledError),
            ):
                await connect_pool_connection('postgresql://localhost:5432/testdb')
            assert acquire_state.interrupted is True

    @pytest.mark.asyncio
    async def test_successful_dial_records_no_interruption(self) -> None:
        """A dial that completes leaves the record untouched."""
        from app.backends.postgresql_backend.acquire_faults import track_acquire
        from app.backends.postgresql_backend.pool_callbacks import connect_pool_connection

        with track_acquire() as acquire_state:
            with unittest.mock.patch(
                'asyncpg.connect',
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ):
                await connect_pool_connection('postgresql://localhost:5432/testdb')
            assert acquire_state.interrupted is False

    def test_nested_scopes_share_one_record(self) -> None:
        """An inner scope reuses the outer record instead of shadowing it.

        execute_write acquires through get_connection, so a record created by
        the inner scope would hide the interruption from the outer arm that has
        to charge it.
        """
        from app.backends.postgresql_backend.acquire_faults import track_acquire

        with track_acquire() as outer, track_acquire() as inner:
            assert inner is outer

    @pytest.mark.asyncio
    async def test_get_connection_charges_a_cancelled_dial(self) -> None:
        """get_connection charges the bare timeout that killed a dial."""
        backend = _backend_with_cancelled_dial()

        with pytest.raises(TimeoutError):
            async with backend.get_connection():
                pass

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1
        assert backend.metrics.last_error is not None
        assert backend.metrics.last_error_time is not None

    @pytest.mark.asyncio
    async def test_begin_transaction_charges_a_cancelled_dial(self) -> None:
        """begin_transaction charges the bare timeout that killed a dial."""
        backend = _backend_with_cancelled_dial()

        with pytest.raises(TimeoutError):
            async with backend.begin_transaction():
                pass

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1

    @pytest.mark.asyncio
    async def test_execute_write_charges_a_cancelled_dial(self) -> None:
        """execute_write charges the bare timeout that killed a dial."""
        backend = _backend_with_cancelled_dial()
        fast_retries(backend)

        async def operation(_conn: object) -> None:
            raise AssertionError('the operation must never run')

        with pytest.raises(TimeoutError):
            await backend.execute_write(operation)

        assert backend.circuit_breaker.failures == 1
        assert backend.metrics.failed_queries == 1

    @pytest.mark.asyncio
    async def test_saturation_without_a_dial_stays_uncharged(self) -> None:
        """A bare timeout with no dial attempt remains an uncharged capacity signal."""
        backend = _backend_with_saturation_timeout()

        with pytest.raises(TimeoutError):
            async with backend.get_connection():
                pass

        assert backend.circuit_breaker.failures == 0
        assert backend.metrics.failed_queries == 0

    @pytest.mark.asyncio
    async def test_execute_write_saturation_stays_uncharged(self) -> None:
        """execute_write keeps treating a dial-free acquire timeout as capacity."""
        backend = _backend_with_saturation_timeout()
        fast_retries(backend)

        async def operation(_conn: object) -> None:
            raise AssertionError('the operation must never run')

        with pytest.raises(TimeoutError):
            await backend.execute_write(operation)

        assert backend.circuit_breaker.failures == 0
        assert backend.metrics.failed_queries == 0

    @pytest.mark.asyncio
    async def test_blackholed_database_charges_through_a_real_pool(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A real asyncpg pool against a blackholed listener opens the accounting.

        The listener completes the TCP handshake and then never speaks the
        protocol, so the dial hangs until the (much shorter) acquire deadline
        cancels it -- the exact interleaving a firewalled or partitioned database
        produces.

        The budgets are installed directly on the module bindings rather than
        through the environment: the settings boundary REFUSES a connect budget
        at or above the acquire budget, and a sub-second acquire deadline is what
        makes the cancellation reproducible in a test. The code path itself is not
        limited to that ordering -- under the correctly ordered defaults an acquire
        that spends its budget queueing reaches the dial with the same result.
        """
        import app.backends.postgresql_backend as pg_module
        from app.backends.postgresql_backend.pool_callbacks import connect_pool_connection
        from app.settings import get_settings

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(('127.0.0.1', 0))
        listener.listen(8)
        port = listener.getsockname()[1]

        get_settings.cache_clear()
        base = get_settings()
        squeezed = base.model_copy(
            update={
                'storage': base.storage.model_copy(
                    update={
                        'postgresql_pool_timeout_s': 0.5,
                        'postgresql_connect_timeout_s': 60.0,
                    },
                ),
            },
        )
        rebind_package_settings(monkeypatch, pg_module, squeezed)

        backend = build_backend(f'postgresql://postgres:postgres@127.0.0.1:{port}/testdb')
        pool = await asyncpg.create_pool(
            backend.connection_string,
            min_size=0,
            max_size=1,
            connect=connect_pool_connection,
            timeout=60,
        )
        backend._pool = pool
        try:
            with pytest.raises(TimeoutError):
                async with backend.get_connection():
                    pass

            assert backend.circuit_breaker.failures == 1
            assert backend.metrics.failed_queries == 1
            assert backend.metrics.last_error is not None
            assert backend.metrics.last_error_time is not None
        finally:
            pool.terminate()
            listener.close()
            get_settings.cache_clear()
