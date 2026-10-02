"""Tests for the embedding-compression schema migration.

Covers the SQLite branch in detail (always runnable in CI without docker).
PostgreSQL-side migration behavior is exercised by the docker-compose
integration tests under ``tests/integration/postgresql/`` and is intentionally
not duplicated here.

The tests use a dedicated SQLite database per test so the migration's
``DROP TABLE IF EXISTS vec_context_embeddings`` is harmless and the
``compression_metadata`` singleton starts empty.
"""

import sqlite3

import pytest

from app.backends import StorageBackend
from app.migrations.compression import apply_compression_migration
from tests.helpers import disable_compression
from tests.helpers import enable_compression

pytestmark = pytest.mark.usefixtures('clear_settings_cache')


@pytest.mark.asyncio
async def test_sqlite_migration_creates_tables(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When enabled, the migration creates the compressed-vector table and
    the singleton provenance table together with the supporting index."""
    enable_compression(monkeypatch)

    await apply_compression_migration(backend=backend)

    def _check(conn: sqlite3.Connection) -> dict[str, bool]:
        present: dict[str, bool] = {}
        for obj_type, name in [
            ('table', 'vec_context_embeddings_compressed'),
            ('table', 'compression_metadata'),
            ('index', 'idx_vec_compressed_context'),
        ]:
            cur = conn.execute(
                f"SELECT name FROM sqlite_master WHERE type='{obj_type}' AND name = ?",
                (name,),
            )
            present[name] = cur.fetchone() is not None
        return present

    found = await backend.execute_read(_check)
    assert found == {
        'vec_context_embeddings_compressed': True,
        'compression_metadata': True,
        'idx_vec_compressed_context': True,
    }


@pytest.mark.asyncio
async def test_sqlite_migration_skips_when_disabled(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the toggle is off the migration is a no-op (returns immediately
    without creating any tables)."""
    disable_compression(monkeypatch)

    await apply_compression_migration(backend=backend)

    def _check(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='compression_metadata'",
        )
        return cur.fetchone() is not None

    assert await backend.execute_read(_check) is False


@pytest.mark.asyncio
async def test_sqlite_migration_is_idempotent(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the migration twice produces the same schema with no errors."""
    enable_compression(monkeypatch)

    await apply_compression_migration(backend=backend)
    # Second run must not raise.
    await apply_compression_migration(backend=backend)

    def _count(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE name IN ('vec_context_embeddings_compressed', 'compression_metadata')",
        )
        return int(cur.fetchone()[0])

    assert await backend.execute_read(_count) == 2


@pytest.mark.asyncio
async def test_sqlite_singleton_check_enforced(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CHECK (id = 1) constraint rejects any row with id != 1."""
    enable_compression(monkeypatch)
    await apply_compression_migration(backend=backend)

    def _insert_second(conn: sqlite3.Connection) -> None:
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (2, 'turboquant', 4, 'ip', 42, 1024),
        )

    with pytest.raises(sqlite3.IntegrityError):
        await backend.execute_write(_insert_second)


@pytest.mark.asyncio
async def test_sqlite_singleton_unique_id(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inserting two rows with id=1 is rejected by the PRIMARY KEY."""
    enable_compression(monkeypatch)
    await apply_compression_migration(backend=backend)

    def _insert(conn: sqlite3.Connection) -> None:
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim) '
            'VALUES (1, ?, ?, ?, ?, ?)',
            ('turboquant', 4, 'ip', 42, 1024),
        )

    await backend.execute_write(_insert)
    with pytest.raises(sqlite3.IntegrityError):
        await backend.execute_write(_insert)


@pytest.mark.asyncio
async def test_sqlite_migration_drops_legacy_vec_table(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The migration drops any pre-existing fp32 vec_context_embeddings table.

    The standard schema does NOT create vec_context_embeddings (it's a vec0
    virtual table that requires sqlite-vec). We simulate the prior fp32 state
    by creating a stand-in table with the same name.
    """
    enable_compression(monkeypatch)

    def _create_legacy(conn: sqlite3.Connection) -> None:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS vec_context_embeddings '
            '(rowid INTEGER PRIMARY KEY, embedding BLOB)',
        )

    await backend.execute_write(_create_legacy)

    await apply_compression_migration(backend=backend)

    def _exists(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='vec_context_embeddings'",
        )
        return cur.fetchone() is not None

    assert await backend.execute_read(_exists) is False
