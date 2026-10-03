"""Tests for the embedding-compression migration with embedding generation off.

Covers the skip on a fresh database, the maintained schema of a database that
already carries it, the schema provisioned for existing embedding
infrastructure, and the clean state after a zero-data decompress.
"""

import sqlite3

import pytest

from app.backends import StorageBackend
from app.migrations.compression import apply_compression_migration
from app.settings import get_settings
from tests.helpers import enable_compression

pytestmark = pytest.mark.usefixtures('clear_settings_cache')


@pytest.mark.asyncio
async def test_sqlite_migration_skips_when_generation_disabled_and_schema_absent(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With generation off on a FRESH database, the migration is a no-op.

    Embedding storage is provisioned from ENABLE_EMBEDDING_GENERATION, so a
    generation-off deployment with NO embedding infrastructure must not gain
    a compression schema it can never write to -- the validator would then
    seed a provenance row and a later ENABLE_EMBEDDING_COMPRESSION=false flip
    would wedge behind a --decompress run with no embedding infrastructure to
    operate on. A database that already carries embedding tables is
    provisioned instead (see the embedding-infra test below).
    """
    enable_compression(monkeypatch)
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())

    await apply_compression_migration(backend=backend)

    def _check(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name IN "
            "('compression_metadata', 'vec_context_embeddings_compressed')",
        )
        return cur.fetchone() is not None

    assert await backend.execute_read(_check) is False


@pytest.mark.asyncio
async def test_sqlite_migration_maintains_existing_schema_when_generation_disabled(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database that already carries the compression schema keeps it maintained.

    Data compressed while generation was on must stay decodable after the
    operator turns generation off, so the migration falls through for an
    existing schema instead of skipping it.
    """
    enable_compression(monkeypatch)
    await apply_compression_migration(backend=backend)

    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())

    await apply_compression_migration(backend=backend)

    def _check(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='vec_context_embeddings_compressed'",
        )
        return cur.fetchone() is not None

    assert await backend.execute_read(_check) is True


@pytest.mark.asyncio
async def test_sqlite_migration_provisions_schema_for_embedding_infra_generation_off(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding infrastructure without fp32 data still gets the compression schema.

    The delete/update cleanup paths gate on embedding_metadata presence and
    route through the compressed payload table whenever compression is
    enabled, so a database whose embedding tables were provisioned while
    generation was on must carry the compression schema even after generation
    is toggled off -- otherwise every text-carrying update would fail on the
    missing payload table inside its transaction.
    """

    def _seed_embedding_metadata(conn: sqlite3.Connection) -> None:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS embedding_metadata ('
            'context_id TEXT PRIMARY KEY, chunk_count INTEGER, dimensions INTEGER)',
        )

    await backend.execute_write(_seed_embedding_metadata)

    enable_compression(monkeypatch)
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())

    await apply_compression_migration(backend=backend)

    def _both_tables(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
            "AND name IN ('compression_metadata', 'vec_context_embeddings_compressed')",
        )
        row = cur.fetchone()
        return int(row[0])

    assert await backend.execute_read(_both_tables) == 2


@pytest.mark.asyncio
async def test_sqlite_migration_stays_clean_after_zero_data_decompress_generation_off(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The marker table alone does not re-create the payload table under generation off.

    The zero-data --decompress path drops the empty compressed table and
    deletes the provenance row but leaves the compression_metadata table
    itself. The SQLite applied probe requires BOTH tables (mirroring the
    PostgreSQL probe), so the next generation-off startup reports not-applied
    and keeps the database clean instead of re-provisioning the payload table
    the operator just removed.
    """
    enable_compression(monkeypatch)
    await apply_compression_migration(backend=backend)

    def _zero_data_decompress(conn: sqlite3.Connection) -> None:
        conn.execute('DROP TABLE IF EXISTS vec_context_embeddings_compressed')
        conn.execute('DELETE FROM compression_metadata WHERE id = 1')

    await backend.execute_write(_zero_data_decompress)

    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())

    await apply_compression_migration(backend=backend)

    def _payload_exists(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='vec_context_embeddings_compressed'",
        )
        return cur.fetchone() is not None

    assert await backend.execute_read(_payload_exists) is False
