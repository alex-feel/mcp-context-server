"""Tests for app/backends/postgresql_backend/transactions.py: begin_transaction breaker accounting."""

import contextlib
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.backends.postgresql_backend import PostgreSQLBackend


class TestBeginTransactionDeadlockExemption:
    """begin_transaction does not charge the breaker for server-initiated rollbacks.

    The tool layer retries deadlock/serialization rollbacks (SQLSTATE class 40),
    so charging the breaker per aborted attempt would let routine write
    contention open it and reject every caller's healthy requests. A genuine
    fault still charges.
    """

    @staticmethod
    def _backend_with_fake_pool() -> PostgreSQLBackend:
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        backend._shutdown = False

        @contextlib.asynccontextmanager
        async def _fake_transaction() -> AsyncIterator[None]:
            yield None

        mock_conn = MagicMock()
        mock_conn.transaction = MagicMock(side_effect=_fake_transaction)

        @contextlib.asynccontextmanager
        async def _fake_acquire(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
            yield mock_conn

        pool = MagicMock()
        pool.acquire = MagicMock(side_effect=_fake_acquire)
        backend._pool = pool
        return backend

    @pytest.mark.asyncio
    async def test_deadlock_rollback_does_not_charge_breaker(self) -> None:
        """A deadlock escaping the transaction body leaves the breaker untouched."""
        backend = self._backend_with_fake_pool()
        with pytest.raises(asyncpg.exceptions.DeadlockDetectedError):
            async with backend.begin_transaction():
                raise asyncpg.exceptions.DeadlockDetectedError('deadlock detected')
        assert backend.circuit_breaker.failures == 0

    @pytest.mark.asyncio
    async def test_genuine_fault_still_charges_breaker(self) -> None:
        """A non-rollback fault in the transaction body still records a failure."""
        backend = self._backend_with_fake_pool()
        with pytest.raises(RuntimeError, match='db fault'):
            async with backend.begin_transaction():
                raise RuntimeError('db fault')
        assert backend.circuit_breaker.failures == 1
