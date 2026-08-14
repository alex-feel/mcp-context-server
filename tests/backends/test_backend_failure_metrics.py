"""Charged-failure bookkeeping parity across the storage backends.

A database fault that charges the circuit breaker must ALSO move the three
operator-facing failure metrics ``get_metrics()`` publishes as
``connection_metrics``: ``failed_queries``, ``last_error`` and
``last_error_time``. A charge without them leaves a dashboard reporting a
healthy, error-free database while the breaker counts an outage; a count
without the message leaves an operator a number and no diagnosis -- or, worse,
an OLD message beside fresh failures.

The same fault must be counted exactly ONCE: the connection scope owns the
bookkeeping, and the read wrapper above it records nothing.
"""

import asyncio
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.sqlite_backend import SQLiteBackend
from app.errors import ControlFlowError


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


def _assert_single_charged_failure(
    *,
    failures: int,
    failed_queries: int,
    last_error: str | None,
    last_error_time: float | None,
    needle: str,
) -> None:
    """Assert exactly one charged failure was recorded with its diagnostics.

    Args:
        failures: The circuit breaker's accumulated failure count.
        failed_queries: The failed-query metric.
        last_error: The recorded failure message.
        last_error_time: The recorded failure timestamp.
        needle: Substring expected in the recorded failure message.
    """
    assert failures == 1
    assert failed_queries == 1
    assert last_error is not None
    assert needle in last_error
    assert last_error_time is not None


def _charged(backend: SQLiteBackend | PostgreSQLBackend, needle: str) -> None:
    """Assert the backend recorded exactly one charged failure naming ``needle``.

    Args:
        backend: The backend under test.
        needle: Substring expected in the recorded failure message.
    """
    _assert_single_charged_failure(
        failures=backend.circuit_breaker.failures,
        failed_queries=backend.metrics.failed_queries,
        last_error=backend.metrics.last_error,
        last_error_time=backend.metrics.last_error_time,
        needle=needle,
    )


class TestSqliteChargedFailureMetrics:
    """Every charged SQLite site records the failure metrics alongside the charge."""

    @pytest.mark.asyncio
    async def test_transaction_fault_records_metrics(self, tmp_path: Path) -> None:
        """A transaction body fault (the main write path) records all four fields."""
        backend = await _sqlite_backend(tmp_path / 'txn_metrics.db')
        try:
            with pytest.raises(sqlite3.OperationalError, match='disk I/O error'):
                async with backend.begin_transaction():
                    raise sqlite3.OperationalError('disk I/O error')
            _charged(backend, 'disk I/O error')
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_direct_writer_fault_records_metrics(self, tmp_path: Path) -> None:
        """A fault on the allow_write connection records all four fields."""
        backend = await _sqlite_backend(tmp_path / 'writer_metrics.db')
        try:
            with pytest.raises(RuntimeError, match='schema step failed'):
                async with backend.get_connection(allow_write=True):
                    raise RuntimeError('schema step failed')
            _charged(backend, 'schema step failed')
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_direct_writer_failing_rollback_keeps_the_original_fault(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rollback that itself fails must not displace the caller's error."""
        # The rollback runs on the shared writer, which the periodic health check
        # may have closed underneath this scope. An unguarded rollback then raised
        # 'Cannot operate on a closed database' in place of the real fault: the
        # caller lost the actual error, and the classification ladder below the
        # rollback -- including the breaker charge -- was skipped entirely.
        # begin_transaction guards the identical rollback the same way.
        backend = await _sqlite_backend(tmp_path / 'writer_rollback_metrics.db')
        try:
            writer = backend._writer_conn
            assert writer is not None

            def _rollback_fails() -> None:
                raise sqlite3.ProgrammingError('Cannot operate on a closed database')

            monkeypatch.setattr(writer, 'rollback', _rollback_fails)

            with pytest.raises(sqlite3.OperationalError, match='schema step failed'):
                async with backend.get_connection(allow_write=True):
                    raise sqlite3.OperationalError('schema step failed')
            _charged(backend, 'schema step failed')
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_read_fault_records_metrics_exactly_once(self, tmp_path: Path) -> None:
        """A read fault is counted once, with its message, not counted twice."""
        backend = await _sqlite_backend(tmp_path / 'read_metrics.db')
        try:

            def _fail(_conn: sqlite3.Connection) -> None:
                raise ValueError('permission denied for table context_entries')

            with pytest.raises(ValueError, match='permission denied'):
                await backend.execute_read(_fail)
            _charged(backend, 'permission denied')
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_connection_creation_fault_records_metrics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An establishment fault records the metrics, not only the breaker charge."""
        backend = await _sqlite_backend(tmp_path / 'create_metrics.db')

        error = sqlite3.OperationalError('unable to open database file')
        error.sqlite_errorcode = sqlite3.SQLITE_CANTOPEN

        async def _fail_to_create() -> sqlite3.Connection:
            raise error

        try:
            monkeypatch.setattr(backend, '_get_reader_connection', _fail_to_create)
            with pytest.raises(sqlite3.OperationalError, match='unable to open database file'):
                async with backend.get_connection(readonly=True):
                    pass
            _charged(backend, 'unable to open database file')
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_queued_write_fault_records_metrics(self, tmp_path: Path) -> None:
        """The write-queue arm keeps recording all four fields."""
        backend = await _sqlite_backend(tmp_path / 'queue_metrics.db')
        try:

            def _fail(_conn: sqlite3.Connection) -> None:
                raise ValueError('constraint machinery exploded')

            with pytest.raises(ValueError, match='constraint machinery exploded'):
                await backend.execute_write(_fail)
            _charged(backend, 'constraint machinery exploded')
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_control_flow_error_records_nothing(self, tmp_path: Path) -> None:
        """Normal control flow stays out of both the breaker and the metrics."""
        backend = await _sqlite_backend(tmp_path / 'cf_metrics.db')
        try:
            with pytest.raises(ControlFlowError):
                async with backend.begin_transaction():
                    raise ControlFlowError('optimistic concurrency conflict')
            assert backend.circuit_breaker.failures == 0
            assert backend.metrics.failed_queries == 0
            assert backend.metrics.last_error is None
            assert backend.metrics.last_error_time is None
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_direct_writer_control_flow_error_records_nothing(self, tmp_path: Path) -> None:
        """The allow_write arm exempts normal control flow, like its two siblings."""
        # A client repeatedly sending input a validation guard rejects inside the
        # connection scope would otherwise open the process-global breaker and start
        # rejecting every other caller's healthy requests.
        backend = await _sqlite_backend(tmp_path / 'writer_cf_metrics.db')
        try:
            with pytest.raises(ControlFlowError):
                async with backend.get_connection(allow_write=True):
                    raise ControlFlowError('optimistic concurrency conflict')
            assert backend.circuit_breaker.failures == 0
            assert backend.metrics.failed_queries == 0
            assert backend.metrics.last_error is None
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_direct_writer_lock_contention_records_nothing(self, tmp_path: Path) -> None:
        """The allow_write arm exempts SQLITE_BUSY/SQLITE_LOCKED write contention.

        Contention from a concurrent process sharing the database file is
        self-clearing and the write paths are taught to ride it out, so charging it
        would open the breaker on exactly the condition retries exist for. The
        readonly arm and begin_transaction already exempt it.
        """
        backend = await _sqlite_backend(tmp_path / 'writer_busy_metrics.db')
        error = sqlite3.OperationalError('database is locked')
        error.sqlite_errorcode = sqlite3.SQLITE_BUSY
        try:
            with pytest.raises(sqlite3.OperationalError, match='database is locked'):
                async with backend.get_connection(allow_write=True):
                    raise error
            assert backend.circuit_breaker.failures == 0
            assert backend.metrics.failed_queries == 0
            assert backend.metrics.last_error is None
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_direct_writer_cancellation_records_nothing(self, tmp_path: Path) -> None:
        """A cancellation unwinding the allow_write scope is rolled back, not charged."""
        # Cancellation is not a database fault, and begin_transaction exempts the
        # same unwind; the scope must still roll back so the next write on the
        # shared writer cannot silently commit this one's partial state.
        backend = await _sqlite_backend(tmp_path / 'writer_cancel_metrics.db')
        try:
            with pytest.raises(asyncio.CancelledError):
                async with backend.get_connection(allow_write=True):
                    raise asyncio.CancelledError
            assert backend.circuit_breaker.failures == 0
            assert backend.metrics.failed_queries == 0
            assert backend.metrics.last_error is None
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_direct_writer_establishment_fault_records_metrics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An establishment fault on the allow_write arm charges with its metrics.

        The writer acquire ran bare here, so a detached database volume escaped
        above the use-time recording block with the breaker still reporting healthy.
        begin_transaction routes the identical call through the charging helper.
        """
        backend = await _sqlite_backend(tmp_path / 'writer_create_metrics.db')

        error = sqlite3.OperationalError('unable to open database file')
        error.sqlite_errorcode = sqlite3.SQLITE_CANTOPEN

        async def _fail_to_create() -> sqlite3.Connection:
            raise error

        try:
            monkeypatch.setattr(backend, '_ensure_writer_connection', _fail_to_create)
            with pytest.raises(sqlite3.OperationalError, match='unable to open database file'):
                async with backend.get_connection(allow_write=True):
                    pass
            _charged(backend, 'unable to open database file')
        finally:
            await backend.shutdown()


class TestPostgresqlChargedFailureMetrics:
    """The PostgreSQL body-fault arm records the metrics, without double counting."""

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

    @pytest.mark.asyncio
    async def test_connection_body_fault_records_metrics(self) -> None:
        """A body fault inside get_connection records all four fields."""
        backend = self._backend_with_connection(MagicMock())

        with pytest.raises(RuntimeError, match='relation does not exist'):
            async with backend.get_connection():
                raise RuntimeError('relation does not exist')

        _charged(backend, 'relation does not exist')

    @pytest.mark.asyncio
    async def test_read_fault_counted_exactly_once(self) -> None:
        """execute_read leaves the accounting to the connection scope."""
        backend = self._backend_with_connection(MagicMock())

        async def _fail(_conn: object) -> None:
            raise RuntimeError('permission denied for table context_entries')

        with pytest.raises(RuntimeError, match='permission denied'):
            await backend.execute_read(_fail)

        _charged(backend, 'permission denied')

    @pytest.mark.asyncio
    async def test_read_control_flow_error_records_nothing(self) -> None:
        """A client-input rejection inside a read stays out of both records."""
        backend = self._backend_with_connection(MagicMock())

        async def _reject(_conn: object) -> None:
            raise ControlFlowError('invalid metadata filter')

        with pytest.raises(ControlFlowError):
            await backend.execute_read(_reject)

        assert backend.circuit_breaker.failures == 0
        assert backend.metrics.failed_queries == 0
        assert backend.metrics.last_error is None


class TestCrossBackendMetricsContract:
    """Both backends publish the keys the shared connection-metrics contract names."""

    @pytest.mark.asyncio
    async def test_sqlite_metrics_expose_backend_type_and_pool_size(self, tmp_path: Path) -> None:
        """SQLite reports its backend type and the reader-pool bound."""
        backend = await _sqlite_backend(tmp_path / 'metrics_contract.db')
        try:
            metrics = backend.get_metrics()
            assert metrics['backend_type'] == 'sqlite'
            assert metrics['pool_size'] == backend.pool_config.max_readers
            # The backend-specific keys stay put.
            for key in (
                'total_connections',
                'active_connections',
                'failed_connections',
                'total_queries',
                'failed_queries',
                'write_queue_size',
                'circuit_state',
                'consecutive_failures',
                'last_error',
                'last_error_time',
            ):
                assert key in metrics
        finally:
            await backend.shutdown()

    def test_postgresql_metrics_expose_backend_type_and_pool_size(self) -> None:
        """PostgreSQL reports the same two contract keys."""
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        pool = MagicMock()
        pool.get_size = MagicMock(return_value=4)
        pool.get_idle_size = MagicMock(return_value=2)
        pool.get_min_size = MagicMock(return_value=0)
        pool.get_max_size = MagicMock(return_value=20)
        backend._pool = pool

        metrics = backend.get_metrics()
        assert metrics['backend_type'] == 'postgresql'
        assert metrics['pool_size'] == 4

    @pytest.mark.asyncio
    async def test_both_backends_publish_the_guaranteed_keys(self, tmp_path: Path) -> None:
        """A client can read the guaranteed keys without branching on the backend."""
        guaranteed = {'backend_type', 'pool_size'}

        sqlite_backend = await _sqlite_backend(tmp_path / 'metrics_shared.db')
        try:
            assert guaranteed <= set(sqlite_backend.get_metrics())
        finally:
            await sqlite_backend.shutdown()

        postgres_backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        pool = MagicMock()
        pool.get_size = MagicMock(return_value=1)
        pool.get_idle_size = MagicMock(return_value=1)
        pool.get_min_size = MagicMock(return_value=0)
        pool.get_max_size = MagicMock(return_value=20)
        postgres_backend._pool = pool
        assert guaranteed <= set(postgres_backend.get_metrics())
