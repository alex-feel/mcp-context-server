"""
Tests for embedding repository.

Tests the EmbeddingRepository class with SQLite backend using sqlite-vec
for vector storage and search operations.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from app.backends import StorageBackend

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
        await embedding_repo.store(context_id=context_id, embedding=embedding)

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
            await embedding_repo.store(context_id, embedding)

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
            await embedding_repo.store(context_id, [0.1] * embedding_dim)

        for i in range(5):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'other-{i}',
                source='user',
                content_type='text',
                text_content=f'Other entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.2] * embedding_dim)

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
            await embedding_repo.store(context_id, [0.1] * embedding_dim)

        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'agent-thread-{i}',
                source='agent',
                content_type='text',
                text_content=f'Agent entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.2] * embedding_dim)

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
        await embedding_repo.store(context_id, [0.1] * embedding_dim)

        # Update embedding
        new_embedding = [0.5] * embedding_dim
        await embedding_repo.update(context_id, new_embedding)

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
        await embedding_repo.store(context_id, [0.1] * embedding_dim)

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
        exists = await embedding_repo.exists(99999)
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
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

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
            await embedding_repo.store(context_id, [0.1] * embedding_dim)

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
            await embedding_repo.store(context_id, [0.2] * embedding_dim)

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
        await embedding_repo.store(context_id, [0.1] * embedding_dim)

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
