"""SQLite write-contention classification and failure-accounting tests.

Cross-process SQLite write contention (SQLITE_BUSY / SQLITE_LOCKED, surfacing
as ``sqlite3.OperationalError('database is locked')``) is routine, self-clearing
contention -- NOT a database fault. These tests pin the contention contract on
the SQLite backend, mirroring the PostgreSQL class-40 rollback treatment:

- ``is_sqlite_locked_error`` classifies the locked/busy family by SQLite result
  code, with a message fallback for instances constructed in Python.
- ``begin_transaction`` rolls back and re-raises contention WITHOUT charging the
  circuit breaker, while a genuine fault still charges it.
- The ``execute_write`` retry loop performs no dead backoff sleep after the
  final failed attempt and still surfaces a single exhaustion breaker charge.
- ``execute_read`` re-raises ``ControlFlowError`` without counting it in the
  ``failed_queries`` metric, matching the write path's exemption.
- The readonly branch of ``get_connection`` re-raises the locked/busy family
  WITHOUT charging the breaker (mirroring ``begin_transaction``), and
  ``execute_read`` retries the family with bounded backoff on fresh reader
  connections, since reads are idempotent.
- Connection ESTABLISHMENT faults (e.g. SQLITE_CANTOPEN when the database volume
  detaches) charge the breaker on both the readonly ``get_connection`` arm and
  ``begin_transaction``, while an establishment-phase locked/busy handshake stays
  uncharged.
"""

import asyncio
import sqlite3
from pathlib import Path

import pytest

from app.backends.sqlite_backend import RetryConfig
from app.backends.sqlite_backend import SQLiteBackend
from app.backends.sqlite_backend import is_sqlite_locked_error
from app.errors import ControlFlowError


async def _initialized_backend(db_path: Path, retry_config: RetryConfig | None = None) -> SQLiteBackend:
    """Create the base schema and return an initialized SQLite backend."""
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(schema_sql)
    backend = SQLiteBackend(db_path=str(db_path), retry_config=retry_config)
    await backend.initialize()
    return backend


class TestIsSqliteLockedError:
    """Classification of the SQLITE_BUSY / SQLITE_LOCKED contention family."""

    def test_real_cross_connection_lock_collision_classified(self, tmp_path: Path) -> None:
        """A real lock collision carries SQLITE_BUSY and is classified as contention."""
        db_path = tmp_path / 'busy.db'
        conn_a = sqlite3.connect(str(db_path), timeout=0.05)
        conn_b = sqlite3.connect(str(db_path), timeout=0.05)
        try:
            conn_a.execute('CREATE TABLE t (a INTEGER)')
            conn_a.commit()
            conn_a.execute('BEGIN IMMEDIATE')
            with pytest.raises(sqlite3.OperationalError) as exc_info:
                conn_b.execute('BEGIN IMMEDIATE')
            assert exc_info.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
            assert is_sqlite_locked_error(exc_info.value)
        finally:
            conn_a.close()
            conn_b.close()

    def test_extended_busy_code_classified_via_primary_low_byte(self) -> None:
        """Extended result codes (SQLITE_BUSY_SNAPSHOT = 517) carry SQLITE_BUSY in the low byte."""
        error = sqlite3.OperationalError('database is locked')
        error.sqlite_errorcode = 517
        assert is_sqlite_locked_error(error)

    def test_result_code_is_authoritative_over_message(self) -> None:
        """When a result code is present, it decides classification, not the message text."""
        error = sqlite3.OperationalError('database is locked')
        error.sqlite_errorcode = 1  # SQLITE_ERROR: generic failure, not contention
        assert not is_sqlite_locked_error(error)

    def test_message_fallback_for_python_constructed_instances(self) -> None:
        """Instances constructed in Python lack sqlite_errorcode; the message decides."""
        assert is_sqlite_locked_error(sqlite3.OperationalError('database is locked'))
        assert is_sqlite_locked_error(sqlite3.OperationalError('database table is locked'))

    def test_non_contention_errors_not_classified(self) -> None:
        """Errors outside the locked/busy family are not classified as contention."""
        assert not is_sqlite_locked_error(sqlite3.OperationalError('no such table: context_entries'))
        assert not is_sqlite_locked_error(sqlite3.IntegrityError('FOREIGN KEY constraint failed'))
        assert not is_sqlite_locked_error(ValueError('database is locked'))


class TestBeginTransactionContentionExemption:
    """SQLite write contention inside begin_transaction must not charge the breaker.

    A cross-process lock collision (e.g. two MCP server processes sharing the
    default SQLite database file) self-clears when the competing writer commits,
    and the tool-layer retry loops re-run the transaction. Charging the breaker
    for each collision would open it on routine contention and reject every
    caller's healthy requests, including reads.
    """

    @pytest.mark.asyncio
    async def test_locked_error_propagates_without_breaker_charge(self, tmp_path: Path) -> None:
        backend = await _initialized_backend(tmp_path / 'txn_locked.db')
        try:
            # Far more than failure_threshold consecutive contention rollbacks.
            for _ in range(backend.circuit_breaker.failure_threshold + 5):
                with pytest.raises(sqlite3.OperationalError, match='database is locked'):
                    async with backend.begin_transaction():
                        raise sqlite3.OperationalError('database is locked')
            assert backend.circuit_breaker.failures == 0
            assert backend.circuit_breaker.is_open() is False
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_generic_operational_error_still_charges_breaker(self, tmp_path: Path) -> None:
        backend = await _initialized_backend(tmp_path / 'txn_fault.db')
        try:
            with pytest.raises(sqlite3.OperationalError, match='no such table'):
                async with backend.begin_transaction():
                    raise sqlite3.OperationalError('no such table: missing')
            assert backend.circuit_breaker.failures == 1
        finally:
            await backend.shutdown()


class TestExecuteWriteRetryNoFinalSleep:
    """The locked-retry arm must not sleep after the final failed attempt.

    A backoff sleep is only useful BETWEEN attempts; sleeping after the last one
    adds dead latency to every retry-exhausted write and delays the single
    exhaustion breaker charge recorded by the write-queue processor.
    """

    @pytest.mark.asyncio
    async def test_no_backoff_sleep_after_final_attempt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        retry_config = RetryConfig(max_retries=3, base_delay=0.001, max_delay=0.002, jitter=False)
        backend = await _initialized_backend(tmp_path / 'retry.db', retry_config=retry_config)
        sleep_calls: list[float] = []
        real_sleep = asyncio.sleep

        async def _recording_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await real_sleep(0)

        try:
            monkeypatch.setattr(asyncio, 'sleep', _recording_sleep)

            def _always_locked(_conn: sqlite3.Connection) -> None:
                raise sqlite3.OperationalError('database is locked')

            with pytest.raises(sqlite3.OperationalError, match='database is locked'):
                await backend.execute_write(_always_locked)

            # Restore before shutdown so its internal sleep is not recorded.
            monkeypatch.setattr(asyncio, 'sleep', real_sleep)

            # N attempts have N - 1 inter-attempt gaps; the exhausted final
            # attempt falls straight through to the raise, and the write-queue
            # processor charges the breaker exactly once.
            assert len(sleep_calls) == retry_config.max_retries - 1
            assert backend.circuit_breaker.failures == 1
        finally:
            await backend.shutdown()


class TestExecuteReadControlFlowAccounting:
    """execute_read must exempt ControlFlowError from the failed_queries metric.

    A client-input validation rejection raised inside a read callable (e.g. an
    invalid metadata filter) is normal control flow. Counting it as a failed
    query misreports healthy client-input rejection as database instability in
    the operator-facing connection metrics, inconsistent with the write path,
    which re-raises ControlFlowError before its failure accounting.
    """

    @pytest.mark.asyncio
    async def test_control_flow_error_not_counted_as_failed_query(self, tmp_path: Path) -> None:
        backend = await _initialized_backend(tmp_path / 'read_cf.db')
        try:

            def _reject(_conn: sqlite3.Connection) -> None:
                raise ControlFlowError('client-input validation rejection')

            with pytest.raises(ControlFlowError):
                await backend.execute_read(_reject)
            assert backend.metrics.failed_queries == 0
            assert backend.circuit_breaker.failures == 0
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_genuine_read_fault_still_counted(self, tmp_path: Path) -> None:
        backend = await _initialized_backend(tmp_path / 'read_fault.db')
        try:

            def _fail(_conn: sqlite3.Connection) -> None:
                raise ValueError('genuine read fault')

            with pytest.raises(ValueError, match='genuine read fault'):
                await backend.execute_read(_fail)
            assert backend.metrics.failed_queries == 1
        finally:
            await backend.shutdown()


def _locked_error() -> sqlite3.OperationalError:
    """Build a SQLITE_BUSY OperationalError carrying the authoritative result code."""
    error = sqlite3.OperationalError('database is locked')
    error.sqlite_errorcode = sqlite3.SQLITE_BUSY
    return error


def _cantopen_error() -> sqlite3.OperationalError:
    """Build a SQLITE_CANTOPEN OperationalError, the volume-detach creation fault."""
    error = sqlite3.OperationalError('unable to open database file')
    error.sqlite_errorcode = sqlite3.SQLITE_CANTOPEN
    return error


class TestReadonlyContentionExemption:
    """The readonly branch of get_connection must not charge the breaker for contention.

    SQLITE_BUSY / SQLITE_LOCKED surfacing on a read (e.g. an external process
    holding an exclusive lock during VACUUM / backup / checkpoint truncate on a
    shared database file) is self-clearing contention, exactly like the write
    path's contention that begin_transaction already exempts. Charging the
    process-global breaker for it would open the breaker on routine contention and
    reject every caller's requests, including the writes taught to ride it out.
    """

    @pytest.mark.asyncio
    async def test_locked_read_does_not_charge_breaker(self, tmp_path: Path) -> None:
        backend = await _initialized_backend(tmp_path / 'read_locked.db')
        try:
            for _ in range(backend.circuit_breaker.failure_threshold + 5):
                with pytest.raises(sqlite3.OperationalError, match='database is locked'):
                    async with backend.get_connection(readonly=True):
                        raise _locked_error()
            assert backend.circuit_breaker.failures == 0
            assert backend.circuit_breaker.is_open() is False
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_genuine_read_fault_still_charges_breaker(self, tmp_path: Path) -> None:
        backend = await _initialized_backend(tmp_path / 'read_fault_cb.db')
        error = sqlite3.OperationalError('no such table: missing')
        error.sqlite_errorcode = sqlite3.SQLITE_ERROR
        try:
            with pytest.raises(sqlite3.OperationalError, match='no such table'):
                async with backend.get_connection(readonly=True):
                    raise error
            assert backend.circuit_breaker.failures == 1
        finally:
            await backend.shutdown()


class TestExecuteReadLockedRetry:
    """execute_read retries the locked family with bounded backoff, then exempts it.

    Reads are idempotent and each attempt runs on a fresh reader connection, so a
    transient cross-process lock can be waited out. After the retry budget is
    exhausted the locked error is re-raised WITHOUT charging failed_queries or the
    breaker, so a long external lock cannot open the process-global breaker.
    """

    @pytest.mark.asyncio
    async def test_locked_read_retries_then_succeeds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        retry_config = RetryConfig(max_retries=4, base_delay=0.001, max_delay=0.002, jitter=False)
        backend = await _initialized_backend(tmp_path / 'read_retry_ok.db', retry_config=retry_config)
        sleep_calls: list[float] = []
        real_sleep = asyncio.sleep

        async def _recording_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await real_sleep(0)

        attempts = {'count': 0}

        def _locked_twice_then_ok(_conn: sqlite3.Connection) -> str:
            attempts['count'] += 1
            if attempts['count'] < 3:
                raise _locked_error()
            return 'ok'

        try:
            monkeypatch.setattr(asyncio, 'sleep', _recording_sleep)
            result = await backend.execute_read(_locked_twice_then_ok)
            monkeypatch.setattr(asyncio, 'sleep', real_sleep)

            assert result == 'ok'
            assert attempts['count'] == 3
            # Two contention failures produced two inter-attempt backoff sleeps.
            assert len(sleep_calls) == 2
            assert backend.metrics.failed_queries == 0
            assert backend.circuit_breaker.failures == 0
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_locked_read_exhausts_retries_without_breaker_charge(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        retry_config = RetryConfig(max_retries=3, base_delay=0.001, max_delay=0.002, jitter=False)
        backend = await _initialized_backend(tmp_path / 'read_retry_exhaust.db', retry_config=retry_config)
        sleep_calls: list[float] = []
        real_sleep = asyncio.sleep

        async def _recording_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await real_sleep(0)

        def _always_locked(_conn: sqlite3.Connection) -> None:
            raise _locked_error()

        try:
            monkeypatch.setattr(asyncio, 'sleep', _recording_sleep)
            with pytest.raises(sqlite3.OperationalError, match='database is locked'):
                await backend.execute_read(_always_locked)
            monkeypatch.setattr(asyncio, 'sleep', real_sleep)

            # N attempts have N - 1 inter-attempt gaps; the exhausted final attempt
            # falls straight through to the raise with no dead trailing sleep.
            assert len(sleep_calls) == retry_config.max_retries - 1
            # Self-clearing contention never charges the read metric or the breaker.
            assert backend.metrics.failed_queries == 0
            assert backend.circuit_breaker.failures == 0
        finally:
            await backend.shutdown()


class TestConnectionCreationFaultCharging:
    """Connection ESTABLISHMENT faults charge the breaker on read and store paths.

    Creating a reader and recreating the writer sit above the breaker-recording
    blocks that wrap connection USE, so a SQLITE_CANTOPEN 'unable to open database
    file' (the database volume detaches after the health check closes the dead
    writer) would otherwise escape with the breaker reporting healthy for the whole
    outage. Both establishment awaits now charge a genuine creation fault, while an
    establishment-phase locked/busy handshake stays uncharged for parity with the
    lock-retry exemption.
    """

    @pytest.mark.asyncio
    async def test_reader_creation_cantopen_charges_breaker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend = await _initialized_backend(tmp_path / 'reader_cantopen.db')

        async def _open_read() -> None:
            async with backend.get_connection(readonly=True):
                pass

        try:

            async def _fail_to_create() -> sqlite3.Connection:
                raise _cantopen_error()

            monkeypatch.setattr(backend, '_get_reader_connection', _fail_to_create)

            with pytest.raises(sqlite3.OperationalError, match='unable to open database file'):
                await _open_read()
            assert backend.circuit_breaker.failures == 1
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_reader_creation_locked_stays_uncharged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend = await _initialized_backend(tmp_path / 'reader_locked_create.db')

        async def _open_read() -> None:
            async with backend.get_connection(readonly=True):
                pass

        try:

            async def _fail_locked() -> sqlite3.Connection:
                raise _locked_error()

            monkeypatch.setattr(backend, '_get_reader_connection', _fail_locked)

            for _ in range(backend.circuit_breaker.failure_threshold + 2):
                with pytest.raises(sqlite3.OperationalError, match='database is locked'):
                    await _open_read()
            assert backend.circuit_breaker.failures == 0
            assert backend.circuit_breaker.is_open() is False
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_writer_recreation_cantopen_charges_breaker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend = await _initialized_backend(tmp_path / 'writer_cantopen.db')

        async def _open_txn() -> None:
            async with backend.begin_transaction():
                pass

        try:

            async def _fail_to_create() -> sqlite3.Connection:
                raise _cantopen_error()

            monkeypatch.setattr(backend, '_ensure_writer_connection', _fail_to_create)

            with pytest.raises(sqlite3.OperationalError, match='unable to open database file'):
                await _open_txn()
            assert backend.circuit_breaker.failures == 1
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_writer_recreation_locked_stays_uncharged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend = await _initialized_backend(tmp_path / 'writer_locked_create.db')

        async def _open_txn() -> None:
            async with backend.begin_transaction():
                pass

        try:

            async def _fail_locked() -> sqlite3.Connection:
                raise _locked_error()

            monkeypatch.setattr(backend, '_ensure_writer_connection', _fail_locked)

            for _ in range(backend.circuit_breaker.failure_threshold + 2):
                with pytest.raises(sqlite3.OperationalError, match='database is locked'):
                    await _open_txn()
            assert backend.circuit_breaker.failures == 0
            assert backend.circuit_breaker.is_open() is False
        finally:
            await backend.shutdown()
