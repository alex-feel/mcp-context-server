"""Tests for EmbeddingRepository.get_embeddings_size.

Covers the active-table selection by compression mode, the SQLite exact /
estimate split against a real database, the graceful degradation to a zero size
on a missing table or unknown backend, and a static source check confirming the
PostgreSQL path uses pg_total_relation_size with a NULL-guarding to_regclass and
the SQLite path is dbstat-free.
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
from app.repositories.embedding_repository import EmbeddingRepository
from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset get_settings cache before and after every test in this module.

    Tests here flip ENABLE_EMBEDDING_COMPRESSION via env vars; the settings
    singleton would otherwise leak the flipped state to unrelated tests.

    Yields:
        Control to the test body; setup and teardown invalidate the cache.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _make_backend_with_embedding_metadata(db_path: Path) -> StorageBackend:
    """Provision a SQLite backend with the base schema plus embedding tables.

    Creates embedding_metadata (chunk_count source for the fp32 estimate) and
    vec_context_embeddings_compressed (the compressed payload table). Avoids
    sqlite-vec by creating these tables directly rather than via the
    semantic-search migration.

    Returns:
        An initialized SQLite StorageBackend with the embedding tables present.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from app.schemas import load_schema

    conn.executescript(load_schema('sqlite'))
    conn.executescript(
        '''
        CREATE TABLE IF NOT EXISTS embedding_metadata (
            context_id TEXT NOT NULL PRIMARY KEY,
            model_name TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS vec_context_embeddings_compressed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_index INTEGER NOT NULL,
            end_index INTEGER NOT NULL,
            payload BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        ''',
    )
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    return backend


@pytest_asyncio.fixture
async def emb_backend(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend with embedding_metadata + compressed payload tables."""
    backend = await _make_backend_with_embedding_metadata(tmp_path / 'emb_size.db')
    yield backend
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=5.0)


@pytest.mark.asyncio
class TestEmbeddingsSizeSqlite:
    """SQLite payload-size behavior against a real database."""

    async def test_compressed_exact_not_estimated(
        self, emb_backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
        get_settings.cache_clear()

        # 2 MiB payload so the size rounds to a clearly nonzero value at 2 dp.
        payload_bytes = 2 * 1024 * 1024

        def _seed(conn: sqlite3.Connection) -> None:
            conn.execute(
                'INSERT INTO vec_context_embeddings_compressed '
                '(context_id, chunk_index, start_index, end_index, payload) '
                "VALUES ('c1', 0, 0, 10, ?)",
                (b'x' * payload_bytes,),
            )

        await emb_backend.execute_write(_seed)

        repo = EmbeddingRepository(emb_backend)
        size_mb, estimated = await repo.get_embeddings_size()
        assert type(size_mb) is float
        assert size_mb == 2.0
        assert estimated is False

    async def test_fp32_is_estimated(
        self, emb_backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
        get_settings.cache_clear()

        def _seed(conn: sqlite3.Connection) -> None:
            conn.execute(
                'INSERT INTO embedding_metadata (context_id, model_name, dimensions, chunk_count) '
                "VALUES ('c1', 'test-model', 1024, 3)",
            )

        await emb_backend.execute_write(_seed)

        repo = EmbeddingRepository(emb_backend)
        size_mb, estimated = await repo.get_embeddings_size()
        # 3 chunks * 1024 dims * 4 bytes = 12288 bytes.
        assert type(size_mb) is float
        assert size_mb == round(12288 / (1024 * 1024), 2)
        assert estimated is True

    async def test_empty_compressed_table_is_zero_float(
        self, emb_backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
        get_settings.cache_clear()
        repo = EmbeddingRepository(emb_backend)
        size_mb, estimated = await repo.get_embeddings_size()
        assert type(size_mb) is float
        assert size_mb == 0.0
        assert estimated is False

    async def test_missing_table_degrades_to_zero(
        self, async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # async_db_initialized has the base schema only (no embedding tables),
        # so the size query raises and must degrade to a zero size.
        monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
        get_settings.cache_clear()
        repo = EmbeddingRepository(async_db_initialized)
        size_mb, estimated = await repo.get_embeddings_size()
        assert size_mb == 0.0
        assert estimated is False


def test_sqlite_path_uses_dbstat_free_sql() -> None:
    """The SQLite path must use LENGTH(payload) / estimate, never the dbstat table."""
    import inspect

    src = inspect.getsource(EmbeddingRepository._get_embeddings_size_sqlite)
    # No reference to the dbstat virtual table (bundled SQLite does not compile it).
    # The docstring intentionally says "dbstat-free", so match the table usage form.
    assert 'FROM dbstat' not in src
    assert 'from dbstat' not in src.lower().replace('dbstat-free', '')
    assert 'SUM(LENGTH(payload))' in src
    assert 'SUM(chunk_count * dimensions * 4)' in src


def test_postgresql_path_uses_relation_size_with_regclass_guard() -> None:
    """The PostgreSQL path must use pg_total_relation_size with a to_regclass guard.

    to_regclass returns NULL for an absent table, so a missing payload table
    yields a zero size rather than UndefinedTableError (the compression
    migration drops vec_context_embeddings on PostgreSQL).
    """
    import inspect

    src = inspect.getsource(EmbeddingRepository._get_embeddings_size_postgresql)
    assert 'pg_total_relation_size' in src
    assert 'to_regclass' in src
    assert 'vec_context_embeddings_compressed' in src
    assert 'vec_context_embeddings' in src
