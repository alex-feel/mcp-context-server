"""The transactional write path counts into total_queries on both backends.

``connection_metrics`` publishes ``total_queries`` and ``failed_queries`` side by
side, so an operator (or an alerting rule) reads them as one population: the
failure rate is failed/total. ``begin_transaction`` -- the path every
store/update/delete tool takes -- charged its faults into ``failed_queries`` but
never counted its successes, so the denominator ignored the operation entirely: a
process whose only traffic is failing deletes divides a climbing failure count by
zero.
"""

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.sqlite_backend import SQLiteBackend


async def _sqlite_backend(db_path: Path) -> SQLiteBackend:
    """Create the base schema and return an initialized SQLite backend.

    Args:
        db_path: Location of the temporary database file.

    Returns:
        An initialized SQLite backend.
    """
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(schema_sql)
    backend = SQLiteBackend(db_path=str(db_path))
    await backend.initialize()
    return backend


class TestSqliteTransactionCounters:
    """A committed SQLite transaction moves the same counter a read/write does."""

    @pytest.mark.asyncio
    async def test_committed_transaction_counts_one_query(self, tmp_path: Path) -> None:
        """Each committed transaction adds exactly one to total_queries."""
        backend = await _sqlite_backend(tmp_path / 'txn_counter.db')
        try:
            before = backend.metrics.total_queries
            for _ in range(3):
                async with backend.begin_transaction() as txn:
                    txn.connection.execute('SELECT 1')
            assert backend.metrics.total_queries == before + 3
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_failed_transaction_moves_both_counters_consistently(self, tmp_path: Path) -> None:
        """A failing transaction is counted in failed_queries against a real denominator."""
        backend = await _sqlite_backend(tmp_path / 'txn_counter_fail.db')
        try:
            async with backend.begin_transaction() as txn:
                txn.connection.execute('SELECT 1')

            with pytest.raises(sqlite3.OperationalError, match='disk I/O error'):
                async with backend.begin_transaction():
                    raise sqlite3.OperationalError('disk I/O error')

            assert backend.metrics.failed_queries == 1
            # The failure rate is computable because the successful sibling was
            # counted too, instead of leaving the denominator at zero.
            assert backend.metrics.total_queries >= 1
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_direct_writer_scope_counts_one_query(self, tmp_path: Path) -> None:
        """The allow_write scope counts its committed work like its siblings."""
        backend = await _sqlite_backend(tmp_path / 'writer_counter.db')
        try:
            before = backend.metrics.total_queries
            async with backend.get_connection(allow_write=True) as conn:
                conn.execute('SELECT 1')
            assert backend.metrics.total_queries == before + 1
        finally:
            await backend.shutdown()


class TestPostgresqlTransactionCounters:
    """The PostgreSQL transactional path counts the same unit."""

    @staticmethod
    def _backend_with_connection(conn: object) -> PostgreSQLBackend:
        """Build a backend whose pool yields the given connection.

        Args:
            conn: Object handed to the caller of the acquire context.

        Returns:
            A backend wired to the fake pool.
        """
        import contextlib
        from collections.abc import AsyncIterator

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        backend._shutdown = False

        @contextlib.asynccontextmanager
        async def _acquire(*_args: object, **_kwargs: object) -> AsyncIterator[Any]:
            yield conn

        pool = MagicMock()
        pool.acquire = MagicMock(side_effect=_acquire)
        backend._pool = pool
        return backend

    @staticmethod
    def _connection_with_transaction() -> MagicMock:
        """Build a connection mock whose transaction() is an async context manager.

        Returns:
            The configured connection mock.
        """
        import contextlib
        from collections.abc import AsyncIterator

        @contextlib.asynccontextmanager
        async def _fake_transaction() -> AsyncIterator[None]:
            yield None

        conn = MagicMock()
        conn.transaction = MagicMock(side_effect=_fake_transaction)
        return conn

    @pytest.mark.asyncio
    async def test_committed_transaction_counts_one_query(self) -> None:
        """Each committed transaction adds exactly one to total_queries."""
        backend = self._backend_with_connection(self._connection_with_transaction())

        for _ in range(2):
            async with backend.begin_transaction():
                pass

        assert backend.metrics.total_queries == 2

    @pytest.mark.asyncio
    async def test_rolled_back_transaction_counts_nothing(self) -> None:
        """A transaction that never commits is not counted as completed work."""
        backend = self._backend_with_connection(self._connection_with_transaction())

        with pytest.raises(RuntimeError, match='relation does not exist'):
            async with backend.begin_transaction():
                raise RuntimeError('relation does not exist')

        assert backend.metrics.total_queries == 0
        assert backend.metrics.failed_queries == 1
