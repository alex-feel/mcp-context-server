"""
Tests for embedding repository.

Tests the EmbeddingRepository class with SQLite backend using sqlite-vec
for vector storage and search operations.
"""

import asyncio
import contextlib
import importlib.util
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.ids import generate_id
from app.migrations.compression import apply_compression_migration
from app.repositories.embedding_repository import ChunkEmbedding
from app.settings import get_settings

# Conditional skip marker for tests requiring sqlite-vec package
requires_sqlite_vec = pytest.mark.skipif(
    importlib.util.find_spec('sqlite_vec') is None,
    reason='sqlite-vec package not installed',
)


@pytest.mark.asyncio
class TestEmbeddingRepository:
    """Test EmbeddingRepository functionality."""

    @requires_sqlite_vec
    async def test_store_embedding(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Test storing embedding for context entry."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # First create a context entry
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test entry for embedding',
            metadata=None,
        )

        # Store embedding
        embedding = [0.1] * embedding_dim
        await embedding_repo.store(context_id=context_id, embedding=embedding, model='test-model')

        # Verify stored
        exists = await embedding_repo.exists(context_id)
        assert exists is True

    @requires_sqlite_vec
    async def test_store_embedding_with_model(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Test storing embedding with custom model name."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test entry',
            metadata=None,
        )

        # Store with custom model name
        await embedding_repo.store(
            context_id=context_id,
            embedding=[0.1] * embedding_dim,
            model='custom-model:latest',
        )

        exists = await embedding_repo.exists(context_id)
        assert exists is True

    @requires_sqlite_vec
    async def test_search_basic(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Test basic KNN search."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create multiple entries with embeddings
        for i in range(5):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'thread-{i}',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            # Create embeddings with varying values
            embedding = [0.1 * (i + 1)] * embedding_dim
            await embedding_repo.store(context_id, embedding, model='test-model')

        # Search for similar embeddings
        query_embedding = [0.1] * embedding_dim
        results, stats = await embedding_repo.search(
            query_embedding=query_embedding,
            limit=3,
        )

        assert len(results) == 3
        # Verify stats are returned
        assert 'execution_time_ms' in stats
        assert 'filters_applied' in stats
        assert 'rows_returned' in stats
        # Results should have distance field
        for result in results:
            assert 'distance' in result
            assert 'id' in result
            assert 'text_content' in result

    @requires_sqlite_vec
    async def test_search_with_thread_filter(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Test search with thread_id filter."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries in different threads
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='target-thread',
                source='user',
                content_type='text',
                text_content=f'Target entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1] * embedding_dim, model='test-model')

        for i in range(5):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'other-{i}',
                source='user',
                content_type='text',
                text_content=f'Other entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.2] * embedding_dim, model='test-model')

        # Search with thread filter
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            thread_id='target-thread',
        )

        assert len(results) == 3
        for result in results:
            assert result['thread_id'] == 'target-thread'

    @requires_sqlite_vec
    async def test_search_with_source_filter(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Test search with source filter."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with different sources
        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'user-thread-{i}',
                source='user',
                content_type='text',
                text_content=f'User entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1] * embedding_dim, model='test-model')

        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'agent-thread-{i}',
                source='agent',
                content_type='text',
                text_content=f'Agent entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.2] * embedding_dim, model='test-model')

        # Search with source filter
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            source='user',
        )

        assert len(results) == 2
        for result in results:
            assert result['source'] == 'user'

    @requires_sqlite_vec
    async def test_update_embedding(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Test updating an existing embedding."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import ChunkEmbedding
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entry and store initial embedding
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='update-test',
            source='user',
            content_type='text',
            text_content='Entry to update',
            metadata=None,
        )
        await embedding_repo.store(context_id, [0.1] * embedding_dim, model='test-model')

        # Update embedding using ChunkEmbedding
        new_embedding = [0.5] * embedding_dim
        chunk_emb = ChunkEmbedding(embedding=new_embedding, start_index=0, end_index=15)
        await embedding_repo.update(context_id, [chunk_emb], model='test-model')

        # Verify update by searching
        results, _ = await embedding_repo.search(
            query_embedding=[0.5] * embedding_dim,
            limit=1,
        )

        assert len(results) == 1
        assert results[0]['id'] == context_id

    @requires_sqlite_vec
    async def test_delete_embedding(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Test deleting an embedding."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entry and store embedding
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='delete-test',
            source='user',
            content_type='text',
            text_content='Entry to delete',
            metadata=None,
        )
        await embedding_repo.store(context_id, [0.1] * embedding_dim, model='test-model')

        # Verify exists
        assert await embedding_repo.exists(context_id) is True

        # Delete embedding
        await embedding_repo.delete(context_id)

        # Verify deleted
        assert await embedding_repo.exists(context_id) is False

    @requires_sqlite_vec
    async def test_exists_returns_false_for_nonexistent(
        self, async_db_with_embeddings: StorageBackend,
    ) -> None:
        """Test exists returns False for non-existent embedding."""
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        embedding_repo = EmbeddingRepository(backend)

        # Check non-existent ID
        exists = await embedding_repo.exists(generate_id())
        assert exists is False

    @requires_sqlite_vec
    async def test_get_statistics(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Test getting embedding statistics."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with embeddings
        for i in range(5):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='stats-thread',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim, model='test-model')

        # Create entries without embeddings
        for i in range(3):
            await repos.context.store_with_deduplication(
                thread_id='no-embedding-thread',
                source='user',
                content_type='text',
                text_content=f'No embedding {i}',
                metadata=None,
            )

        # Get statistics
        stats = await embedding_repo.get_statistics()

        assert stats['total_embeddings'] == 5
        assert stats['total_entries'] == 8
        assert 'coverage_percentage' in stats
        # Coverage should be 5/8 = 62.5%
        assert 60 <= stats['coverage_percentage'] <= 65

    @requires_sqlite_vec
    async def test_get_statistics_with_thread_filter(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Test getting statistics filtered by thread."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries in target thread with embeddings
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='target-stats',
                source='user',
                content_type='text',
                text_content=f'Target {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1] * embedding_dim, model='test-model')

        # Create entry in target thread without embedding
        await repos.context.store_with_deduplication(
            thread_id='target-stats',
            source='user',
            content_type='text',
            text_content='No embedding',
            metadata=None,
        )

        # Create entries in other thread
        for i in range(5):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='other-stats',
                source='user',
                content_type='text',
                text_content=f'Other {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.2] * embedding_dim, model='test-model')

        # Get statistics for target thread only
        stats = await embedding_repo.get_statistics(thread_id='target-stats')

        assert stats['total_embeddings'] == 3
        assert stats['total_entries'] == 4
        # Coverage should be 3/4 = 75%
        assert stats['coverage_percentage'] == 75.0

    @requires_sqlite_vec
    async def test_get_table_dimension(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Test getting table dimension."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entry and store embedding
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='dim-test',
            source='user',
            content_type='text',
            text_content='Entry for dimension',
            metadata=None,
        )
        await embedding_repo.store(context_id, [0.1] * embedding_dim, model='test-model')

        # Get dimension
        dimension = await embedding_repo.get_table_dimension()

        assert dimension == embedding_dim

    @requires_sqlite_vec
    async def test_get_table_dimension_empty(
        self, async_db_with_embeddings: StorageBackend,
    ) -> None:
        """Test getting table dimension when no embeddings exist."""
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        embedding_repo = EmbeddingRepository(backend)

        # Get dimension when no embeddings exist
        dimension = await embedding_repo.get_table_dimension()

        assert dimension is None

    @requires_sqlite_vec
    async def test_search_empty_database(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Test search returns empty list when no embeddings exist."""
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        embedding_repo = EmbeddingRepository(backend)

        # Search empty database
        results, stats = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
        )

        assert results == []
        assert stats['rows_returned'] == 0

    @requires_sqlite_vec
    async def test_get_statistics_empty_database(
        self, async_db_with_embeddings: StorageBackend,
    ) -> None:
        """Test statistics on empty database."""
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        embedding_repo = EmbeddingRepository(backend)

        stats = await embedding_repo.get_statistics()

        assert stats['total_embeddings'] == 0
        assert stats['total_entries'] == 0
        assert stats['coverage_percentage'] == 0.0


# get_statistics correctness under compression on SQLite.
# The compressed write path inserts into vec_context_embeddings_compressed
# and embedding_metadata only. embedding_chunks (SQLite) and
# vec_context_embeddings (PostgreSQL) are NOT populated on the compressed
# path -- vec_context_embeddings is dropped entirely on PostgreSQL by the
# compression migration. get_statistics therefore sources its chunk total
# from embedding_metadata.chunk_count, the single source-of-truth that
# every write path populates, so both backends report correct counts in
# every compression-mode combination.


def _enable_compression_for_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flip compression env vars and refresh migration module binding."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())


@pytest_asyncio.fixture
async def compressed_backend_for_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend with the standard schema + compression migration applied.

    Mirrors the fixture in test_embedding_repository_compressed.py: provisions
    a SQLite DB with the canonical schema plus a minimal embedding_metadata
    table (the semantic-search migration is skipped to avoid the sqlite-vec
    dependency). The compression migration creates the compressed payload
    table this test exercises.

    Yields:
        StorageBackend with compression migration applied and ready for use;
        teardown shuts the backend down and clears the settings singleton so
        the compression-enabled state does not leak into unrelated tests.
    """
    _enable_compression_for_stats(monkeypatch)
    db_path = tmp_path / 'test_get_statistics_compressed.db'

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from app.backends import create_backend
    from app.schemas import load_schema

    conn.executescript(load_schema('sqlite'))
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

    try:
        yield backend
    finally:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(backend.shutdown(), timeout=5.0)
        # Clear the singleton so the compression-enabled flip does not leak
        # into unrelated tests in the same pytest session.
        get_settings.cache_clear()


def _make_compressed_chunk(idx: int, payload: bytes) -> ChunkEmbedding:
    return ChunkEmbedding(
        embedding=[0.01 * (idx + 1)] * 16,
        start_index=idx * 100,
        end_index=(idx + 1) * 100,
        payload=payload,
    )


@pytest.mark.asyncio
async def test_get_statistics_with_compression_sqlite(
    compressed_backend_for_stats: StorageBackend,
) -> None:
    """Under compression, get_statistics MUST report nonzero chunk totals.

    Stores 3 context entries with 3 compressed chunks each (9 rows in
    embedding_metadata sum of chunk_count, 9 compressed payloads). The
    compressed write path does not populate embedding_chunks at all; the
    test asserts that get_statistics reads its chunk total from
    embedding_metadata.chunk_count instead.
    """
    from app.repositories import RepositoryContainer
    from app.repositories.embedding_repository import EmbeddingRepository

    backend = compressed_backend_for_stats
    repos = RepositoryContainer(backend)
    repo = EmbeddingRepository(backend)

    n_entries = 3
    chunks_per_entry = 3
    for i in range(n_entries):
        cid, _ = await repos.context.store_with_deduplication(
            thread_id='compressed-stats',
            source='user',
            content_type='text',
            text_content=f'compressed entry {i}',
            metadata=None,
        )
        chunks = [
            _make_compressed_chunk(j, f'P{i}-{j}'.encode())
            for j in range(chunks_per_entry)
        ]
        await repo.store_chunked(cid, chunks, model='test-model')

    stats = await repo.get_statistics()

    assert stats['total_embeddings'] == n_entries, (
        f'expected {n_entries} entries with embeddings, got '
        f"{stats['total_embeddings']}"
    )
    assert stats['total_chunks'] == n_entries * chunks_per_entry, (
        'Compressed path populates only embedding_metadata; querying '
        'embedding_chunks yields 0. Expected '
        f'{n_entries * chunks_per_entry} (sum of chunk_count); got '
        f"{stats['total_chunks']}"
    )
    assert stats['average_chunks_per_entry'] == float(chunks_per_entry)
    assert stats['coverage_percentage'] == 100.0
    assert stats['backend'] == 'sqlite'
