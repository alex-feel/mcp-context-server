"""Tests for the fp32 guards of the embedding-compression migration.

Covers the refusal to drop a populated, never-compressed fp32 table (also with
embedding generation off), the provenance row that marks a leftover fp32 table
as droppable, and the fp32 row probe on an unloaded vec0 module and under lock
contention.
"""

import sqlite3
from collections.abc import Callable
from typing import Any
from typing import cast

import pytest

from app.backends import StorageBackend
from app.errors import ConfigurationError
from app.migrations.compression import apply_compression_migration
from app.settings import get_settings
from tests.helpers import enable_compression

pytestmark = pytest.mark.usefixtures('clear_settings_cache')


@pytest.mark.asyncio
async def test_sqlite_migration_refuses_first_time_with_populated_fp32(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-time application on a database with POPULATED fp32 embeddings refuses.

    A bare ENABLE_EMBEDDING_COMPRESSION=true flip on a deployment that stored
    fp32 embeddings while compression was off must NOT silently drop them: the
    migration raises ConfigurationError (exit 78) directing the operator to the
    --compress CLI, and the fp32 table survives untouched.
    """
    from app.errors import ConfigurationError

    enable_compression(monkeypatch)

    def _create_populated_legacy(conn: sqlite3.Connection) -> None:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS vec_context_embeddings '
            '(rowid INTEGER PRIMARY KEY, embedding BLOB)',
        )
        conn.execute(
            'INSERT INTO vec_context_embeddings (rowid, embedding) VALUES (1, ?)',
            (b'\x00\x01\x02\x03',),
        )

    await backend.execute_write(_create_populated_legacy)

    with pytest.raises(ConfigurationError, match='mcp-context-server-migrate'):
        await apply_compression_migration(backend=backend)

    def _survives(conn: sqlite3.Connection) -> int:
        return int(conn.execute('SELECT COUNT(*) FROM vec_context_embeddings').fetchone()[0])

    assert await backend.execute_read(_survives) == 1


@pytest.mark.asyncio
async def test_sqlite_migration_proceeds_with_populated_fp32_when_provenance_present(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provenance row marks the database as already-compressed, so a leftover
    populated fp32 table is a stray artifact and the migration proceeds
    (re-running the DROP) instead of refusing."""
    enable_compression(monkeypatch)

    # First application on a clean database, then simulate the validator's
    # bootstrap INSERT so the provenance row is present (the real post-first-
    # startup state).
    await apply_compression_migration(backend=backend)

    def _insert_provenance(conn: sqlite3.Connection) -> None:
        conn.execute(
            'INSERT INTO compression_metadata (id, provider, bits, variant, seed, dim) '
            'VALUES (1, ?, ?, ?, ?, ?)',
            ('turboquant', 4, 'ip', 42, 1024),
        )

    await backend.execute_write(_insert_provenance)

    def _create_populated_legacy(conn: sqlite3.Connection) -> None:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS vec_context_embeddings '
            '(rowid INTEGER PRIMARY KEY, embedding BLOB)',
        )
        conn.execute(
            'INSERT INTO vec_context_embeddings (rowid, embedding) VALUES (1, ?)',
            (b'\x00\x01\x02\x03',),
        )

    await backend.execute_write(_create_populated_legacy)

    # Must not raise: the provenance row proves the compressed table is the
    # authoritative store, so the stray fp32 table is dropped.
    await apply_compression_migration(backend=backend)

    def _exists(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='vec_context_embeddings'",
        )
        return cur.fetchone() is not None

    assert await backend.execute_read(_exists) is False


@pytest.mark.asyncio
async def test_sqlite_migration_refuses_generation_off_flip_over_populated_fp32(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A populated, never-compressed fp32 store refuses even with generation off.

    The enable-direction guard is evaluated BEFORE the generation-off skip: a
    bare ENABLE_EMBEDDING_COMPRESSION=true flip on an archive deployment that
    serves existing fp32 embeddings read-only (generation toggled off) must
    exit 78 directing the operator to --compress, not boot silently with
    every stored embedding invisible to search.
    """

    def _seed_fp32(conn: sqlite3.Connection) -> None:
        conn.execute(
            'CREATE TABLE vec_context_embeddings '
            '(id INTEGER PRIMARY KEY, context_id TEXT, embedding BLOB)',
        )
        conn.execute(
            'INSERT INTO vec_context_embeddings (context_id, embedding) '
            "VALUES ('abc', x'00')",
        )

    await backend.execute_write(_seed_fp32)

    enable_compression(monkeypatch)
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())

    with pytest.raises(ConfigurationError, match='--compress'):
        await apply_compression_migration(backend=backend)


class _FakeVecConnection:
    """SQLite connection stand-in whose vec0 table read raises a chosen error.

    ``sqlite_master`` reports the table as present, while the row probe against it
    fails -- the shape of a vec0 virtual table whose module is not loaded, and the
    shape of a lock held by another process, which are the two cases the probe's
    handler must tell apart.
    """

    def __init__(self, error: sqlite3.OperationalError) -> None:
        self._error = error

    def execute(self, sql: str) -> object:
        """Return a cursor-like object, or raise the configured error.

        Args:
            sql: The statement the probe issues.

        Returns:
            A cursor-like object for the sqlite_master lookup. The row probe against
            the vec0 table instead re-raises the configured error.
        """
        if 'FROM vec_context_embeddings' in sql:
            raise self._error

        class _Cursor:
            def fetchone(self) -> tuple[str]:
                return ('vec_context_embeddings',)

        return _Cursor()


class _FakeProbeBackend:
    """Minimal storage-backend stand-in running read callables on a fake connection."""

    backend_type = 'sqlite'

    def __init__(self, conn: _FakeVecConnection) -> None:
        self._conn = conn

    async def execute_read(self, operation: Callable[[Any], bool]) -> bool:
        """Run the probe callable against the fake connection.

        Args:
            operation: The probe closure.

        Returns:
            Whatever the closure returns.
        """
        return operation(self._conn)


@pytest.mark.asyncio
async def test_fp32_probe_treats_a_missing_vec0_module_as_populated() -> None:
    """An unreadable vec0 table refuses the drop: it may hold data once loadable.

    The table exists in sqlite_master but its module is not loaded, so the rows
    cannot be counted. Refusing is the fail-safe direction -- dropping it would
    destroy shadow-table data that becomes readable the moment the extension loads.
    """
    from app.migrations.compression import _fp32_table_has_rows

    backend = _FakeProbeBackend(_FakeVecConnection(sqlite3.OperationalError('no such module: vec0')))

    assert await _fp32_table_has_rows(cast('StorageBackend', backend)) is True


@pytest.mark.asyncio
async def test_fp32_probe_propagates_lock_contention() -> None:
    """SQLITE_BUSY must reach the bounded retry loop, not be read as 'fp32 populated'.

    Consumed inside the read callable, a self-clearing lock would become a permanent
    ConfigurationError telling the operator to run --compress on a database that may
    hold no fp32 rows at all.
    """
    from app.migrations.compression import _fp32_table_has_rows

    backend = _FakeProbeBackend(_FakeVecConnection(sqlite3.OperationalError('database is locked')))

    with pytest.raises(sqlite3.OperationalError, match='database is locked'):
        await _fp32_table_has_rows(cast('StorageBackend', backend))
