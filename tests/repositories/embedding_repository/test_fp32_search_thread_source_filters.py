"""fp32 semantic search with thread_id and source filters.

On SQLite the search pre-filters entries in a CTE and ranks them with the vec_distance_l2() scalar function, so filters
apply before the result limit and a filtered search returns up to `limit` matching entries.
"""

import pytest

from app.backends import StorageBackend
from tests.conftest import requires_semantic_search
from tests.helpers import store_single_chunk_embedding


@pytest.mark.asyncio
class TestSemanticSearchFilters:
    """Test that thread_id and source filters apply before the result limit."""

    @requires_semantic_search
    async def test_thread_filter_returns_correct_count(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """thread_id filter returns correct number of results.

        With limit=3 and two entries in the filtered thread, both are returned:
        the thread filter applies before the result limit, so entries from other
        threads cannot consume it.
        """
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Store context entries in different threads
        # Create 2 entries in "test-thread"
        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id='test-thread',
                source='user',
                content_type='text',
                text_content=f'Test entry {i} in test-thread',
                metadata=None,
            )
            # Store mock embedding
            mock_embedding = [0.1 * (i + 1)] * embedding_dim
            await store_single_chunk_embedding(embedding_repo, context_id, mock_embedding)

        # Create 5 entries in other threads
        for i in range(5):
            context_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id=f'other-thread-{i}',
                source='user',
                content_type='text',
                text_content=f'Entry in other-thread-{i}',
                metadata=None,
            )
            mock_embedding = [0.2 * (i + 1)] * embedding_dim
            await store_single_chunk_embedding(embedding_repo, context_id, mock_embedding)

        # Perform search with thread filter
        query_embedding = [0.1] * embedding_dim
        results, _ = await embedding_repo.search(
            query_embedding=query_embedding,
            limit=3,
            thread_id='test-thread',
        )

        # Type guard: ensure results is a list (not error dict)
        assert isinstance(results, list)
        # Should return 2 results (all from "test-thread"), not fewer
        assert len(results) == 2
        for result in results:
            assert result['thread_id'] == 'test-thread'

    @requires_semantic_search
    async def test_source_filter_returns_correct_count(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """source filter returns correct number of results."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create 3 entries with source="user"
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id=f'thread-user-{i}',
                source='user',
                content_type='text',
                text_content=f'User entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Create 5 entries with source="agent"
        for i in range(5):
            context_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id=f'thread-agent-{i}',
                source='agent',
                content_type='text',
                text_content=f'Agent entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with source filter
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=5,
            source='user',
        )

        # Type guard: ensure results is a list (not error dict)
        assert isinstance(results, list)
        # Should return 3 results (all "user" entries)
        assert len(results) == 3
        for result in results:
            assert result['source'] == 'user'

    @requires_semantic_search
    async def test_combined_filters_return_correct_count(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Combined filters return correct number of results."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create 2 entries in "test-thread" with source="user"
        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id='test-thread',
                source='user',
                content_type='text',
                text_content=f'User entry {i} in test-thread',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Create entries in test-thread with source="agent"
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id='test-thread',
                source='agent',
                content_type='text',
                text_content=f'Agent entry {i} in test-thread',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with both filters
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=5,
            thread_id='test-thread',
            source='user',
        )

        # Type guard: ensure results is a list (not error dict)
        assert isinstance(results, list)
        # Should return 2 results (matching both filters)
        assert len(results) == 2
        for result in results:
            assert result['thread_id'] == 'test-thread'
            assert result['source'] == 'user'

    @requires_semantic_search
    async def test_no_filters_still_works_correctly(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Verify that search without filters still works correctly."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create 5 entries
        for i in range(5):
            context_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id=f'thread-{i}',
                source='user' if i % 2 == 0 else 'agent',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search without filters
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=3,
        )

        # Should return 3 results
        assert len(results) == 3

    @requires_semantic_search
    async def test_filter_returns_empty_when_no_matches(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test that filter returns empty list when no entries match."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries in thread-a
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id='thread-a',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with non-existent thread
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=5,
            thread_id='thread-b',  # Does not exist
        )

        # Should return empty list, not an error
        assert results == []

    @requires_semantic_search
    async def test_filter_returns_less_when_fewer_exist(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test that filter returns fewer results when fewer entries exist."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create only 2 entries in small-thread
        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id='small-thread',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for 10 but only 2 exist
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            thread_id='small-thread',
        )

        # Should return 2 results (all available)
        assert len(results) == 2


@pytest.mark.asyncio
class TestSemanticSearchEdgeCases:
    """Test edge cases for semantic search filtering."""

    @requires_semantic_search
    async def test_single_entry_thread_returns_one_result(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test filtering a thread with exactly one entry."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create 1 entry in single-thread
        context_id, _ = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='single-thread',
            source='user',
            content_type='text',
            text_content='Single entry',
            metadata=None,
        )
        await store_single_chunk_embedding(embedding_repo, context_id, [0.1] * embedding_dim)

        # Create entries in other threads
        for i in range(5):
            ctx_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id=f'other-{i}',
                source='user',
                content_type='text',
                text_content=f'Other {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, ctx_id, [0.2 * (i + 1)] * embedding_dim)

        # Search for single thread
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=5,
            thread_id='single-thread',
        )

        assert len(results) == 1
        assert results[0]['thread_id'] == 'single-thread'

    @requires_semantic_search
    async def test_all_entries_in_same_thread(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test when all entries are in the target thread."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create 10 entries all in "only-thread"
        for i in range(10):
            context_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id='only-thread',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for 5 from only-thread
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=5,
            thread_id='only-thread',
        )

        assert len(results) == 5
        for result in results:
            assert result['thread_id'] == 'only-thread'

    @requires_semantic_search
    async def test_null_thread_id_filter(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test that None thread_id doesn't filter (searches all threads)."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries in multiple threads
        for i in range(5):
            context_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id=f'thread-{i}',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with thread_id=None
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            thread_id=None,
        )

        # Should return results from all threads
        assert len(results) == 5
        thread_ids = {r['thread_id'] for r in results}
        assert len(thread_ids) == 5

    @requires_semantic_search
    async def test_null_source_filter(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test that None source doesn't filter (searches all sources)."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with both sources
        for i in range(4):
            context_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id=f'thread-{i}',
                source='user' if i % 2 == 0 else 'agent',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with source=None
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            source=None,
        )

        # Should return results from both sources
        assert len(results) == 4
        sources = {r['source'] for r in results}
        assert 'user' in sources
        assert 'agent' in sources
