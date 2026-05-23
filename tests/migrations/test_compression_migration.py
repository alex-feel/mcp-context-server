"""Tests for the embedding-compression schema migration.

Covers the SQLite branch in detail (always runnable in CI without docker).
PostgreSQL-side migration behaviour is exercised by the docker-compose
integration tests under ``tests/integration/postgresql/`` and is intentionally
not duplicated here.

The tests use a dedicated SQLite database per test so the migration's
``DROP TABLE IF EXISTS vec_context_embeddings`` is harmless and the
``compression_metadata`` singleton starts empty.
"""

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Generator
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.backends import create_backend
from app.migrations.compression import apply_compression_migration
from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset ``get_settings`` cache before and after every test.

    Env-var monkeypatching for compression toggles would otherwise leak
    into unrelated tests because the settings singleton is process-global.

    Yields:
        Control to the test body; setup and teardown invalidate the cache.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def backend(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend with the standard schema pre-applied."""
    db_path = tmp_path / 'test_compression.db'

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    conn.executescript(schema_sql)
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    yield backend

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=5.0)


def _enable_compression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flip the compression toggle and refresh module-level settings caches."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    # COMPRESSION_SEED is required for runtime but not for the migration loader
    # which only inspects the enabled flag.
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())


def _disable_compression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset compression toggle to off."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.delenv('COMPRESSION_SEED', raising=False)
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())


@pytest.mark.asyncio
async def test_sqlite_migration_creates_tables(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When enabled, the migration creates the compressed-vector table and
    the singleton provenance table together with the supporting index."""
    _enable_compression(monkeypatch)

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
    _disable_compression(monkeypatch)

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
    _enable_compression(monkeypatch)

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
    _enable_compression(monkeypatch)
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
    _enable_compression(monkeypatch)
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
    _enable_compression(monkeypatch)

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
