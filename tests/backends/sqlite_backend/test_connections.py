"""Tests for app/backends/sqlite_backend/connections.py: reader PRAGMAs and the sqlite-vec load.

Reader PRAGMAs. A read-only SQLite connection cannot establish DATABASE-FILE
properties, so issuing one there is at best a no-op and at worst a hard error.
``PRAGMA journal_mode`` is the dangerous case: when the on-disk journal mode
differs from SQLITE_JOURNAL_MODE -- another process (a second server instance,
the migration CLI, a bare ``sqlite3`` shell) moved the shared file into WAL -- a
read-only connection answers with SQLITE_IOERR or SQLITE_READONLY. Neither
belongs to the self-clearing SQLITE_BUSY / SQLITE_LOCKED family the
creation-fault wrapper exempts, so every reader creation would charge the
process-global circuit breaker until it opened and rejected healthy writes too.

The sqlite-vec (vec0) extension load. Two properties are pinned here. First, the
extension loads independently of the embedding-generation toggle. Second, the
runtime extension-loading capability the load requires is withdrawn again even
when the load fails.

The fp32 ``vec_context_embeddings`` vec0 virtual table can physically persist
from an earlier session that ran with embedding generation enabled. The
delete/update stale-embedding cleanup paths gate on the durable
``embedding_tables_exist()`` table-presence signal, NOT on the runtime
``ENABLE_EMBEDDING_GENERATION`` toggle, so they attempt vec0 access whenever the
table exists. If the vec0 module is not loaded on the connection, that access
raises ``no such module: vec0`` -- which the delete path swallows (permanently
orphaning the FK-less vec0 rows once the embedding_chunks bridge cascades) and
the update path propagates (rolling back the whole text update). The extension
load must therefore be decoupled from the generation toggle.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

import app.backends.sqlite_backend as sqlite_backend_module
import app.backends.sqlite_backend.connections as connections_module
from app.backends.sqlite_backend import SQLiteBackend
from app.backends.sqlite_backend.connections import ManagedConnection
from app.settings import get_settings
from tests.conftest import requires_sqlite_vec
from tests.helpers import rebind_package_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """Drop the cached settings singleton around every test in this module.

    These tests override SQLITE_JOURNAL_MODE and ENABLE_EMBEDDING_GENERATION, and
    ``get_settings`` is a process-lifetime singleton, so the cache must be dropped
    both before (so the override is seen) and after (so the override does not leak
    into later tests).

    Yields:
        None, once the cache has been cleared for the test.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _use_journal_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Point the backend package at a settings singleton with the given journal mode.

    Args:
        monkeypatch: Fixture used to set the environment and rebind the
            module-level settings objects the backend submodules cache at import time.
        mode: Value for SQLITE_JOURNAL_MODE.
    """
    monkeypatch.setenv('SQLITE_JOURNAL_MODE', mode)
    get_settings.cache_clear()
    rebind_package_settings(monkeypatch, sqlite_backend_module, get_settings())


def _wal_database(db_path: Path) -> None:
    """Create a small database whose on-disk journal mode is WAL.

    Args:
        db_path: Location of the database file to create.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute('CREATE TABLE probe (a INTEGER)')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.commit()
    finally:
        conn.close()


class TestReaderJournalModeMismatch:
    """Readers open cleanly against a database whose journal mode differs."""

    def test_reader_opens_against_mismatched_on_disk_journal_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A DELETE-configured reader still opens a WAL database and can query it."""
        db_path = tmp_path / 'journal_mismatch.db'
        _wal_database(db_path)
        _use_journal_mode(monkeypatch, 'DELETE')

        backend = SQLiteBackend(db_path=str(db_path))
        conn = backend._create_connection(readonly=True)
        try:
            assert conn.execute('SELECT COUNT(*) FROM probe').fetchone()[0] == 0
        finally:
            backend._safe_close_connection(conn)
        assert backend.circuit_breaker.failures == 0

    def test_reader_leaves_the_on_disk_journal_mode_untouched(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Opening a reader never rewrites the file-level journal mode."""
        db_path = tmp_path / 'journal_untouched.db'
        _wal_database(db_path)
        _use_journal_mode(monkeypatch, 'DELETE')

        backend = SQLiteBackend(db_path=str(db_path))
        conn = backend._create_connection(readonly=True)
        backend._safe_close_connection(conn)

        probe = sqlite3.connect(str(db_path))
        try:
            assert probe.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        finally:
            probe.close()

    def test_writer_still_establishes_the_configured_journal_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The writer remains the connection that sets the file-level properties."""
        db_path = tmp_path / 'journal_writer.db'
        _wal_database(db_path)
        _use_journal_mode(monkeypatch, 'DELETE')

        backend = SQLiteBackend(db_path=str(db_path))
        conn = backend._create_connection(readonly=False)
        try:
            assert conn.execute('PRAGMA journal_mode').fetchone()[0] == 'delete'
        finally:
            backend._safe_close_connection(conn)


class TestReaderPathUnderJournalModeDrift:
    """The whole read path survives another process flipping the file into WAL."""

    @pytest.mark.asyncio
    async def test_reads_succeed_and_leave_the_breaker_uncharged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated reads after an external WAL flip never open the breaker."""
        from app.schemas import load_schema

        db_path = tmp_path / 'journal_drift.db'
        with sqlite3.connect(str(db_path)) as setup_conn:
            setup_conn.executescript(load_schema('sqlite'))

        _use_journal_mode(monkeypatch, 'DELETE')
        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()
        try:
            # Another process moves the shared file into WAL while the server
            # runs; this direction needs no exclusive lock, so it succeeds.
            flipper = sqlite3.connect(str(db_path))
            try:
                assert flipper.execute('PRAGMA journal_mode = WAL').fetchone()[0] == 'wal'
            finally:
                flipper.close()

            def _count(conn: sqlite3.Connection) -> int:
                row = conn.execute('SELECT COUNT(*) FROM context_entries').fetchone()
                return int(row[0])

            for _ in range(backend.circuit_breaker.failure_threshold + 2):
                assert await backend.execute_read(_count) == 0

            assert backend.circuit_breaker.failures == 0
            assert backend.metrics.failed_queries == 0
            assert backend.metrics.last_error is None
        finally:
            await backend.shutdown()


@requires_sqlite_vec
def test_vec_extension_loads_when_generation_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With generation disabled, a connection still loads vec0 so vec-table access works.

    A load gated on the generation toggle would leave the vec0 module absent:
    ``CREATE VIRTUAL TABLE ... USING vec0`` and the ``DELETE FROM
    vec_context_embeddings`` that ``delete_all_chunks`` runs would both raise
    ``no such module: vec0``.
    """
    # Force the module-level settings the backend reads to generation-disabled,
    # following the documented per-test settings-refresh pattern.
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    rebind_package_settings(monkeypatch, sqlite_backend_module, get_settings())
    assert connections_module.settings.embedding.generation_enabled is False

    backend = SQLiteBackend(db_path=str(tmp_path / 'vecgate.db'))
    # Use the production connection subclass (ManagedConnection): a bare
    # sqlite3.Connection has no __dict__ and cannot carry the _vec_loaded flag.
    conn = sqlite3.connect(':memory:', factory=ManagedConnection)
    try:
        backend._load_sqlite_vec_extension(conn)
        assert getattr(conn, '_vec_loaded', False) is True
        # The vec0 module must be usable even though generation is disabled.
        conn.execute('CREATE VIRTUAL TABLE v USING vec0(embedding float[4])')
        # The exact statement delete_all_chunks runs on the SQLite cleanup path.
        conn.execute('DELETE FROM v WHERE rowid = 1')
    finally:
        conn.close()
        get_settings.cache_clear()


@requires_sqlite_vec
def test_failed_vec_load_leaves_extension_loading_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A load that raises still leaves SQL-level load_extension() disabled.

    ``sqlite_vec.load`` is a plain dlopen, so it raises whenever the bundled
    shared object cannot be loaded on the host (a noexec mount, a musl base
    image, a distroless image missing the runtime deps). The failure is
    deliberately non-fatal, but the ``enable_load_extension(True)`` that preceded
    it must not survive it: otherwise every writer and reader connection the
    process ever opens accepts ``SELECT load_extension('...')`` from SQL for its
    whole lifetime, turning any future SQL-injection defect into native code
    execution instead of a 'not authorized' error.
    """
    import sqlite_vec

    def _fail_to_load(_conn: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError('The specified module could not be found')

    monkeypatch.setattr(sqlite_vec, 'load', _fail_to_load)

    backend = SQLiteBackend(db_path=str(tmp_path / 'vecfail.db'))
    conn = sqlite3.connect(':memory:', factory=ManagedConnection)
    try:
        # Graceful skip: the failure is logged, not raised.
        backend._load_sqlite_vec_extension(conn)
        assert getattr(conn, '_vec_loaded', False) is False

        with pytest.raises(sqlite3.OperationalError, match='not authorized'):
            conn.execute("SELECT load_extension('anything')")
    finally:
        conn.close()
