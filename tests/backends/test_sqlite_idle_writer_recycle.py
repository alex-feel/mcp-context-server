"""POOL_IDLE_TIMEOUT_S recycles the idle SQLite writer connection.

The writer is created at initialize() and is otherwise held for the whole process
lifetime, pinning the database file and its -wal/-shm siblings even when nothing
has been written for hours. POOL_IDLE_TIMEOUT_S is the documented bound on that,
so the health-check loop closes the writer once it has been idle that long and
the next write recreates it. Readers need no equivalent: they are per-use
connections closed right after each read.
"""

import sqlite3
from pathlib import Path

import pytest

from app.backends.sqlite_backend import PoolConfig
from app.backends.sqlite_backend import SQLiteBackend


async def _backend(db_path: Path, idle_timeout: float) -> SQLiteBackend:
    """Create the base schema and return an initialized backend.

    Args:
        db_path: Location of the temporary database file.
        idle_timeout: Value for the writer idle bound, in seconds.

    Returns:
        An initialized SQLite backend.
    """
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(schema_sql)

    # A long health-check interval keeps the background loop out of the way: the
    # tests drive _perform_health_check() directly.
    backend = SQLiteBackend(
        db_path=str(db_path),
        pool_config=PoolConfig(idle_timeout=idle_timeout, health_check_interval=3600.0),
    )
    await backend.initialize()
    return backend


@pytest.mark.asyncio
async def test_idle_writer_is_closed_by_the_health_check(tmp_path: Path) -> None:
    """A writer idle past the bound is closed instead of held for the process life."""
    backend = await _backend(tmp_path / 'idle_writer.db', idle_timeout=0.0)
    try:
        assert backend._writer_conn is not None

        await backend._perform_health_check()

        assert backend._writer_conn is None
    finally:
        await backend.shutdown()


@pytest.mark.asyncio
async def test_the_next_write_recreates_the_recycled_writer(tmp_path: Path) -> None:
    """Recycling is invisible to callers: the next write opens a fresh writer."""
    backend = await _backend(tmp_path / 'idle_writer_reuse.db', idle_timeout=0.0)
    try:
        await backend._perform_health_check()

        def _write(conn: sqlite3.Connection) -> int:
            row = conn.execute('SELECT 1').fetchone()
            return int(row[0])

        assert await backend.execute_write(_write) == 1
        assert backend._writer_conn is not None
        assert backend.circuit_breaker.failures == 0
    finally:
        await backend.shutdown()


@pytest.mark.asyncio
async def test_a_recently_used_writer_is_kept(tmp_path: Path) -> None:
    """A writer used within the bound is left alone."""
    backend = await _backend(tmp_path / 'busy_writer.db', idle_timeout=3600.0)
    try:
        writer_before = backend._writer_conn

        await backend._perform_health_check()

        assert backend._writer_conn is writer_before
    finally:
        await backend.shutdown()


@pytest.mark.asyncio
async def test_a_writer_in_use_is_never_closed_under_a_caller(tmp_path: Path) -> None:
    """An in-flight transaction holds the writer lock, so recycling stands down.

    Closing the connection object a running transaction already captured would
    surface as 'Cannot operate on a closed database' in the middle of a write.
    """
    backend = await _backend(tmp_path / 'inflight_writer.db', idle_timeout=0.0)
    try:
        async with backend.begin_transaction() as txn:
            await backend._perform_health_check()
            assert backend._writer_conn is not None
            # The captured connection is still usable inside the transaction.
            assert txn.connection.execute('SELECT 1').fetchone()[0] == 1
    finally:
        await backend.shutdown()
