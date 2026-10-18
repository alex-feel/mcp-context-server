"""Tests for the SQLSTATE 42501 arms of the PostgreSQL entry points (operations.py and transactions.py).

A statement the database refuses under its privileges, a row-level security check or a
trigger raises ``InsufficientPrivilegeError`` inside the operation. Startup verifies the
privileges, so such a refusal means the application and the policy disagree: it is
neither retried nor charged to the circuit breaker, and it is logged at ERROR with its
SQLSTATE and the operation name but without the row the server may describe.
"""

import contextlib
import logging
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.backends.base import MAINTENANCE
from app.backends.postgresql_backend import PostgreSQLBackend
from tests.backends.postgresql_backend._builders import build_backend
from tests.backends.postgresql_backend._builders import fast_retries

DENIED = 'new row violates row-level security policy for table "context_entries"'
ROW_DETAIL = 'Failing row contains (secret-row-value)'


def _denial() -> asyncpg.exceptions.InsufficientPrivilegeError:
    """Build the refusal a policy check raises, with a row description the log must not carry."""
    error = asyncpg.exceptions.InsufficientPrivilegeError(DENIED)
    error.detail = ROW_DETAIL
    return error


def _pooled_backend() -> PostgreSQLBackend:
    """Build a backend whose pool hands out one fake connection, so the real entry points run."""

    @contextlib.asynccontextmanager
    async def _transaction() -> AsyncGenerator[None, None]:
        yield None

    conn = MagicMock()
    conn.transaction = MagicMock(side_effect=_transaction)
    conn.execute = AsyncMock()

    @contextlib.asynccontextmanager
    async def _acquire(*_args: object, **_kwargs: object) -> AsyncGenerator[object, None]:
        yield conn

    backend = build_backend()
    fast_retries(backend)
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=_acquire)
    backend._pool = pool
    return backend


def _assert_uncharged(backend: PostgreSQLBackend) -> None:
    """Assert the refusal reached neither the breaker nor the failure metrics."""
    assert backend.circuit_breaker.failures == 0
    assert backend.metrics.failed_queries == 0


def _assert_logged(caplog: pytest.LogCaptureFixture, operation: str) -> None:
    """Assert one ERROR record names the SQLSTATE and the operation, without the row description."""
    records = [record for record in caplog.records if record.levelno == logging.ERROR and '42501' in record.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert operation in message
    assert DENIED in message
    assert 'secret-row-value' not in message


class TestExecuteWrite:
    """A refused write is raised from the first attempt and charges nothing."""

    @pytest.mark.asyncio
    async def test_refusal_is_not_retried_charged_or_logged_with_the_row(self, caplog: pytest.LogCaptureFixture) -> None:
        backend = _pooled_backend()
        attempts = {'n': 0}

        async def store_refused_row(_conn: object) -> None:
            attempts['n'] += 1
            raise _denial()

        with caplog.at_level(logging.ERROR), pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await backend.execute_write(store_refused_row, access_scope=MAINTENANCE)

        assert attempts['n'] == 1
        _assert_uncharged(backend)
        _assert_logged(caplog, 'store_refused_row')

    @pytest.mark.asyncio
    async def test_another_database_error_is_still_charged(self) -> None:
        backend = _pooled_backend()

        async def write_missing_table(_conn: object) -> None:
            raise asyncpg.exceptions.UndefinedTableError('relation "missing" does not exist')

        with pytest.raises(asyncpg.exceptions.UndefinedTableError):
            await backend.execute_write(write_missing_table, access_scope=MAINTENANCE)

        assert backend.circuit_breaker.failures == 1


class TestBeginTransaction:
    """A refusal inside a transaction body rolls back and charges nothing."""

    @pytest.mark.asyncio
    async def test_refusal_in_the_body_is_uncharged_and_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        backend = _pooled_backend()

        with caplog.at_level(logging.ERROR), pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            async with backend.begin_transaction(access_scope=MAINTENANCE):
                raise _denial()

        _assert_uncharged(backend)
        _assert_logged(caplog, 'begin_transaction')

    @pytest.mark.asyncio
    async def test_another_database_error_is_still_charged(self) -> None:
        backend = _pooled_backend()

        with pytest.raises(asyncpg.exceptions.UndefinedTableError):
            async with backend.begin_transaction(access_scope=MAINTENANCE):
                raise asyncpg.exceptions.UndefinedTableError('relation "missing" does not exist')

        assert backend.circuit_breaker.failures == 1


class TestConnectionBody:
    """A refusal inside a pooled connection's body, the read path's accounting site, charges nothing."""

    @pytest.mark.asyncio
    async def test_refusal_in_a_connection_body_is_uncharged(self) -> None:
        backend = _pooled_backend()

        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            async with backend.get_connection():
                raise _denial()

        _assert_uncharged(backend)

    @pytest.mark.asyncio
    async def test_refused_read_is_uncharged_and_logged_with_its_operation(self, caplog: pytest.LogCaptureFixture) -> None:
        backend = _pooled_backend()

        async def read_refused_view(_conn: object) -> None:
            raise _denial()

        with caplog.at_level(logging.ERROR), pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await backend.execute_read(read_refused_view, access_scope=MAINTENANCE)

        _assert_uncharged(backend)
        _assert_logged(caplog, 'read_refused_view')

    @pytest.mark.asyncio
    async def test_another_database_error_in_a_read_is_still_charged(self) -> None:
        backend = _pooled_backend()

        async def read_missing_table(_conn: object) -> None:
            raise asyncpg.exceptions.UndefinedTableError('relation "missing" does not exist')

        with pytest.raises(asyncpg.exceptions.UndefinedTableError):
            await backend.execute_read(read_missing_table, access_scope=MAINTENANCE)

        assert backend.circuit_breaker.failures == 1
