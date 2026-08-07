"""SQLite reader-connection PRAGMA tests.

A read-only SQLite connection cannot establish DATABASE-FILE properties, so
issuing one there is at best a no-op and at worst a hard error. ``PRAGMA
journal_mode`` is the dangerous case: when the on-disk journal mode differs from
SQLITE_JOURNAL_MODE -- another process (a second server instance, the migration
CLI, a bare ``sqlite3`` shell) moved the shared file into WAL -- a read-only
connection answers with SQLITE_IOERR or SQLITE_READONLY. Neither belongs to the
self-clearing SQLITE_BUSY / SQLITE_LOCKED family the creation-fault wrapper
exempts, so every reader creation would charge the process-global circuit
breaker until it opened and rejected healthy writes too.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

import app.backends.sqlite_backend as sqlite_backend_module
from app.backends.sqlite_backend import SQLiteBackend
from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """Drop the cached settings singleton around every test in this module.

    These tests override SQLITE_JOURNAL_MODE, and ``get_settings`` is a
    process-lifetime singleton, so the cache must be dropped both before (so the
    override is seen) and after (so the override does not leak into later tests).

    Yields:
        None, once the cache has been cleared for the test.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _use_journal_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Point the backend module at a settings singleton with the given journal mode.

    Args:
        monkeypatch: Fixture used to set the environment and rebind the module
            level settings object the backend caches at import time.
        mode: Value for SQLITE_JOURNAL_MODE.
    """
    monkeypatch.setenv('SQLITE_JOURNAL_MODE', mode)
    get_settings.cache_clear()
    monkeypatch.setattr(sqlite_backend_module, 'settings', get_settings())


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
