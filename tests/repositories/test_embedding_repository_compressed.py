"""Tests for the compressed branch in :class:`EmbeddingRepository`.

These tests exercise the storage code path that fires when
``ENABLE_EMBEDDING_COMPRESSION`` is true: ``store_chunked`` and
``delete_all_chunks`` route to ``_store_chunked_compressed`` and
``_delete_all_chunks_compressed`` respectively, writing to the
``vec_context_embeddings_compressed`` table.

The compressed write path is backend-agnostic and does NOT require
sqlite-vec. We provision a SQLite database with the standard schema +
the compression migration and exercise the repository directly with
synthetic payload bytes (compression encoding is provider-owned and is
covered by its own test suite).
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
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository import ChunkEmbedding
from app.repositories.embedding_repository import EmbeddingRepository
from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset ``get_settings`` cache before and after every test.

    The compressed-path tests flip ENABLE_EMBEDDING_COMPRESSION via env
    vars and the settings singleton would otherwise leak the flipped
    state to unrelated tests in the same pytest session.

    Yields:
        Control to the test body; setup and teardown invalidate the cache.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _enable_compression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flip the toggle and refresh module-level settings caches."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())


@pytest_asyncio.fixture
async def compressed_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend with the standard schema + compression migration applied."""
    _enable_compression(monkeypatch)

    db_path = tmp_path / 'test_compressed_repo.db'

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    conn.executescript(schema_sql)
    # The compressed write path INSERTs into embedding_metadata as the
    # source-of-truth for chunk_count + model. That table is normally
    # created by the semantic-search migration (which also creates the
    # vec0 virtual table and therefore requires sqlite-vec). Tests
    # exercise only the compressed branch, so we create just the
    # embedding_metadata table directly to avoid pulling in sqlite-vec.
    conn.executescript(
        '''
        CREATE TABLE IF NOT EXISTS embedding_metadata (
            context_id TEXT NOT NULL PRIMARY KEY,
            model_name TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
        )
        ''',
    )
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    await apply_compression_migration(backend=backend)

    yield backend

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=5.0)


def _make_chunk(idx: int, payload: bytes) -> ChunkEmbedding:
    """Build a synthetic ChunkEmbedding for compressed-path tests."""
    return ChunkEmbedding(
        embedding=[0.01 * (idx + 1)] * 16,
        start_index=idx * 100,
        end_index=(idx + 1) * 100,
        payload=payload,
    )


@pytest.mark.asyncio
async def test_store_chunked_routes_to_compressed_when_enabled(
    compressed_backend: StorageBackend,
) -> None:
    """With compression enabled, ``store_chunked`` writes BLOB rows to
    ``vec_context_embeddings_compressed`` instead of the fp32 vec table."""
    repos = RepositoryContainer(compressed_backend)
    repo = EmbeddingRepository(compressed_backend)

    cid, _ = await repos.context.store_with_deduplication(
        thread_id='t1', source='user', content_type='text',
        text_content='compressed-routing entry', metadata=None,
    )

    chunks = [_make_chunk(0, b'PAYLOAD-0'), _make_chunk(1, b'PAYLOAD-1')]
    await repo.store_chunked(cid, chunks, model='test-model')

    def _read(conn: sqlite3.Connection) -> list[tuple[int, int, int, bytes]]:
        cur = conn.execute(
            'SELECT chunk_index, start_index, end_index, payload '
            'FROM vec_context_embeddings_compressed '
            'WHERE context_id = ? ORDER BY chunk_index',
            (cid,),
        )
        return [
            (int(r[0]), int(r[1]), int(r[2]), bytes(r[3]))
            for r in cur.fetchall()
        ]

    rows = await compressed_backend.execute_read(_read)
    assert rows == [
        (0, 0, 100, b'PAYLOAD-0'),
        (1, 100, 200, b'PAYLOAD-1'),
    ]


@pytest.mark.asyncio
async def test_store_chunked_requires_payload_when_compressed(
    compressed_backend: StorageBackend,
) -> None:
    """The compressed path raises ValueError when a chunk lacks payload bytes."""
    repos = RepositoryContainer(compressed_backend)
    repo = EmbeddingRepository(compressed_backend)

    cid, _ = await repos.context.store_with_deduplication(
        thread_id='t-missing', source='user', content_type='text',
        text_content='missing payload', metadata=None,
    )

    chunks = [ChunkEmbedding(embedding=[0.1] * 16, start_index=0, end_index=10)]
    with pytest.raises(ValueError, match='payload bytes'):
        await repo.store_chunked(cid, chunks, model='test-model')


@pytest.mark.asyncio
async def test_delete_all_chunks_cleans_compressed_table_when_enabled(
    compressed_backend: StorageBackend,
) -> None:
    """Compressed-path delete removes rows from the compressed table and from
    ``embedding_metadata``; subsequent COUNT returns zero."""
    repos = RepositoryContainer(compressed_backend)
    repo = EmbeddingRepository(compressed_backend)

    cid, _ = await repos.context.store_with_deduplication(
        thread_id='t-delete', source='user', content_type='text',
        text_content='delete me', metadata=None,
    )

    chunks = [_make_chunk(0, b'AAA'), _make_chunk(1, b'BBB'), _make_chunk(2, b'CCC')]
    await repo.store_chunked(cid, chunks, model='test-model')

    deleted = await repo.delete_all_chunks(cid)
    assert deleted == 3

    def _counts(conn: sqlite3.Connection) -> tuple[int, int]:
        c1 = conn.execute(
            'SELECT COUNT(*) FROM vec_context_embeddings_compressed '
            'WHERE context_id = ?',
            (cid,),
        ).fetchone()[0]
        c2 = conn.execute(
            'SELECT COUNT(*) FROM embedding_metadata WHERE context_id = ?',
            (cid,),
        ).fetchone()[0]
        return (int(c1), int(c2))

    rows, meta = await compressed_backend.execute_read(_counts)
    assert rows == 0
    assert meta == 0


@pytest.mark.asyncio
async def test_compressed_round_trip_payload_bytes(
    compressed_backend: StorageBackend,
) -> None:
    """Payload bytes round-trip unchanged through the compressed write path."""
    repos = RepositoryContainer(compressed_backend)
    repo = EmbeddingRepository(compressed_backend)

    cid, _ = await repos.context.store_with_deduplication(
        thread_id='t-roundtrip', source='agent', content_type='text',
        text_content='round trip', metadata=None,
    )

    payload = bytes(range(256))
    await repo.store_chunked(
        cid, [_make_chunk(0, payload)], model='test-model',
    )

    def _read(conn: sqlite3.Connection) -> bytes:
        return bytes(conn.execute(
            'SELECT payload FROM vec_context_embeddings_compressed '
            'WHERE context_id = ? AND chunk_index = 0',
            (cid,),
        ).fetchone()[0])

    stored_payload = await compressed_backend.execute_read(_read)
    assert stored_payload == payload


@pytest.mark.asyncio
async def test_compressed_store_preserves_chunk_boundaries(
    compressed_backend: StorageBackend,
) -> None:
    """start_index/end_index survive the round trip for downstream reranking."""
    repos = RepositoryContainer(compressed_backend)
    repo = EmbeddingRepository(compressed_backend)

    cid, _ = await repos.context.store_with_deduplication(
        thread_id='t-boundaries', source='user', content_type='text',
        text_content='boundary check', metadata=None,
    )

    chunks = [
        ChunkEmbedding(
            embedding=[0.0] * 16,
            start_index=10,
            end_index=42,
            payload=b'first',
        ),
        ChunkEmbedding(
            embedding=[0.0] * 16,
            start_index=100,
            end_index=215,
            payload=b'second',
        ),
    ]
    await repo.store_chunked(cid, chunks, model='test-model')

    def _read(conn: sqlite3.Connection) -> list[tuple[int, int, int]]:
        cur = conn.execute(
            'SELECT chunk_index, start_index, end_index '
            'FROM vec_context_embeddings_compressed '
            'WHERE context_id = ? ORDER BY chunk_index',
            (cid,),
        )
        return [(int(r[0]), int(r[1]), int(r[2])) for r in cur.fetchall()]

    assert await compressed_backend.execute_read(_read) == [
        (0, 10, 42),
        (1, 100, 215),
    ]


@pytest.mark.asyncio
async def test_compressed_chunk_count_per_context(
    compressed_backend: StorageBackend,
) -> None:
    """A single context_id may have multiple compressed chunks with unique chunk_index."""
    repos = RepositoryContainer(compressed_backend)
    repo = EmbeddingRepository(compressed_backend)

    cid, _ = await repos.context.store_with_deduplication(
        thread_id='t-count', source='user', content_type='text',
        text_content='count check', metadata=None,
    )

    n = 5
    chunks = [_make_chunk(i, f'P{i}'.encode()) for i in range(n)]
    await repo.store_chunked(cid, chunks, model='test-model')

    def _read(conn: sqlite3.Connection) -> list[int]:
        cur = conn.execute(
            'SELECT chunk_index FROM vec_context_embeddings_compressed '
            'WHERE context_id = ? ORDER BY chunk_index',
            (cid,),
        )
        return [int(r[0]) for r in cur.fetchall()]

    assert await compressed_backend.execute_read(_read) == list(range(n))


@pytest.mark.asyncio
async def test_upsert_replaces_existing_compressed_chunks(
    compressed_backend: StorageBackend,
) -> None:
    """``upsert=True`` clears existing compressed rows before writing new ones."""
    repos = RepositoryContainer(compressed_backend)
    repo = EmbeddingRepository(compressed_backend)

    cid, _ = await repos.context.store_with_deduplication(
        thread_id='t-upsert', source='user', content_type='text',
        text_content='upsert', metadata=None,
    )

    await repo.store_chunked(
        cid, [_make_chunk(0, b'OLD0'), _make_chunk(1, b'OLD1')],
        model='test-model',
    )
    await repo.store_chunked(
        cid, [_make_chunk(0, b'NEW0')],
        model='test-model', upsert=True,
    )

    def _read(conn: sqlite3.Connection) -> list[bytes]:
        return [
            bytes(r[0]) for r in conn.execute(
                'SELECT payload FROM vec_context_embeddings_compressed '
                'WHERE context_id = ? ORDER BY chunk_index',
                (cid,),
            ).fetchall()
        ]

    assert await compressed_backend.execute_read(_read) == [b'NEW0']
