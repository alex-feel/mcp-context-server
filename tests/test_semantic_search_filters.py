"""
Regression tests for semantic search filter bug.

Tests the fix for the bug where semantic_search_context returns fewer results
than requested when thread_id or source filters are applied.

Root cause: sqlite-vec's k parameter in MATCH clause limits results at
virtual table level BEFORE JOIN and WHERE filters are applied.

Solution: CTE-based pre-filtering with vec_distance_l2() scalar function.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from app.backends import StorageBackend

# Conditional skip marker for tests requiring semantic search dependencies
requires_semantic_search = pytest.mark.skipif(
    not all(
        importlib.util.find_spec(pkg) is not None
        for pkg in ['ollama', 'sqlite_vec', 'numpy']
    ),
    reason='Semantic search dependencies not available (ollama, sqlite_vec, numpy)',
)


@pytest.mark.asyncio
class TestSemanticSearchFilters:
    """Test semantic search filtering with regression tests."""

    @requires_semantic_search
    async def test_thread_filter_returns_correct_count(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Regression test: thread_id filter returns correct number of results.

        This test verifies the fix for the bug where requesting top_k=3 with
        thread_id filter returned only 1 result when 2 should be returned.

        The bug occurred because sqlite-vec's k parameter limited results
        at virtual table level BEFORE the thread_id filter was applied.
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
                thread_id='test-thread',
                source='user',
                content_type='text',
                text_content=f'Test entry {i} in test-thread',
                metadata=None,
            )
            # Store mock embedding
            mock_embedding = [0.1 * (i + 1)] * embedding_dim
            await embedding_repo.store(context_id, mock_embedding)

        # Create 5 entries in other threads
        for i in range(5):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'other-thread-{i}',
                source='user',
                content_type='text',
                text_content=f'Entry in other-thread-{i}',
                metadata=None,
            )
            mock_embedding = [0.2 * (i + 1)] * embedding_dim
            await embedding_repo.store(context_id, mock_embedding)

        # Perform search with thread filter
        query_embedding = [0.1] * embedding_dim
        results = await embedding_repo.search(
            query_embedding=query_embedding,
            limit=3,
            thread_id='test-thread',
        )

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
        """Regression test: source filter returns correct number of results."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create 3 entries with source="user"
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'thread-user-{i}',
                source='user',
                content_type='text',
                text_content=f'User entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Create 5 entries with source="agent"
        for i in range(5):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'thread-agent-{i}',
                source='agent',
                content_type='text',
                text_content=f'Agent entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with source filter
        results = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=5,
            source='user',
        )

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
        """Regression test: combined filters return correct number of results."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create 2 entries in "test-thread" with source="user"
        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='test-thread',
                source='user',
                content_type='text',
                text_content=f'User entry {i} in test-thread',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Create entries in test-thread with source="agent"
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='test-thread',
                source='agent',
                content_type='text',
                text_content=f'Agent entry {i} in test-thread',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with both filters
        results = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=5,
            thread_id='test-thread',
            source='user',
        )

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
                thread_id=f'thread-{i}',
                source='user' if i % 2 == 0 else 'agent',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search without filters
        results = await embedding_repo.search(
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
                thread_id='thread-a',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with non-existent thread
        results = await embedding_repo.search(
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
                thread_id='small-thread',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for 10 but only 2 exist
        results = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            thread_id='small-thread',
        )

        # Should return 2 results (all available)
        assert len(results) == 2


@pytest.mark.asyncio
class TestSemanticSearchDateFiltering:
    """Test date filtering in semantic search (start_date/end_date parameters)."""

    @requires_semantic_search
    async def test_start_date_filter_returns_correct_results(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test semantic search with start_date filter returns entries after the date."""
        from datetime import UTC
        from datetime import datetime
        from datetime import timedelta

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create test entries - all will have current timestamp
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='date-filter-thread',
                source='user',
                content_type='text',
                text_content=f'Date filter test entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with start_date in the past - should find all entries
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
        results = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            start_date=yesterday,
        )
        assert len(results) == 3

        # Search with start_date in the future - should find no entries
        future_date = (datetime.now(UTC) + timedelta(days=30)).strftime('%Y-%m-%d')
        results = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            start_date=future_date,
        )
        assert len(results) == 0

    @requires_semantic_search
    async def test_end_date_filter_returns_correct_results(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test semantic search with end_date filter returns entries before the date."""
        from datetime import UTC
        from datetime import datetime
        from datetime import timedelta

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create test entries
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='end-date-thread',
                source='agent',
                content_type='text',
                text_content=f'End date filter entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with end_date in the future - should find all entries
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
        results = await embedding_repo.search(
            query_embedding=[0.2] * embedding_dim,
            limit=10,
            end_date=tomorrow,
        )
        assert len(results) == 3

        # Search with end_date in the past - should find no entries
        past_date = (datetime.now(UTC) - timedelta(days=30)).strftime('%Y-%m-%d')
        results = await embedding_repo.search(
            query_embedding=[0.2] * embedding_dim,
            limit=10,
            end_date=past_date,
        )
        assert len(results) == 0

    @requires_semantic_search
    async def test_date_range_filter_returns_correct_results(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test semantic search with both start_date and end_date returns correct range."""
        from datetime import UTC
        from datetime import datetime
        from datetime import timedelta

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create test entries
        for i in range(5):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='range-thread',
                source='user',
                content_type='text',
                text_content=f'Date range entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.15 * (i + 1)] * embedding_dim)

        # Search with valid date range (yesterday to tomorrow) - should find all
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
        results = await embedding_repo.search(
            query_embedding=[0.15] * embedding_dim,
            limit=10,
            start_date=yesterday,
            end_date=tomorrow,
        )
        assert len(results) == 5

        # Search with date range in the past - should find none
        far_past = (datetime.now(UTC) - timedelta(days=60)).strftime('%Y-%m-%d')
        past = (datetime.now(UTC) - timedelta(days=30)).strftime('%Y-%m-%d')
        results = await embedding_repo.search(
            query_embedding=[0.15] * embedding_dim,
            limit=10,
            start_date=far_past,
            end_date=past,
        )
        assert len(results) == 0

    @requires_semantic_search
    async def test_date_filter_combined_with_thread_id(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test date filtering combined with thread_id filter."""
        from datetime import UTC
        from datetime import datetime
        from datetime import timedelta

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries in different threads
        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='target-date-thread',
                source='user',
                content_type='text',
                text_content=f'Target thread entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='other-date-thread',
                source='user',
                content_type='text',
                text_content=f'Other thread entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with date filter and thread_id - should find 2 entries from target thread
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
        results = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            thread_id='target-date-thread',
            start_date=yesterday,
            end_date=tomorrow,
        )
        assert len(results) == 2
        for result in results:
            assert result['thread_id'] == 'target-date-thread'

    @requires_semantic_search
    async def test_date_filter_combined_with_source(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test date filtering combined with source filter."""
        from datetime import UTC
        from datetime import datetime
        from datetime import timedelta

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with different sources
        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='mixed-source-thread',
                source='user',
                content_type='text',
                text_content=f'User entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='mixed-source-thread',
                source='agent',
                content_type='text',
                text_content=f'Agent entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with date filter and source - should find 2 user entries
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
        results = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            source='user',
            start_date=yesterday,
            end_date=tomorrow,
        )
        assert len(results) == 2
        for result in results:
            assert result['source'] == 'user'

    @requires_semantic_search
    async def test_date_filter_with_none_values(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test that None date values don't filter (searches all dates)."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create test entries
        for i in range(4):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='no-date-filter-thread',
                source='user',
                content_type='text',
                text_content=f'No date filter entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.25 * (i + 1)] * embedding_dim)

        # Search with None dates - should find all entries
        results = await embedding_repo.search(
            query_embedding=[0.25] * embedding_dim,
            limit=10,
            start_date=None,
            end_date=None,
        )
        assert len(results) == 4


@pytest.mark.asyncio
class TestSemanticSearchPerformance:
    """Test performance characteristics of CTE-based filtering."""

    @requires_semantic_search
    async def test_performance_with_small_filtered_set(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Verify acceptable performance with small filtered sets (<100 entries)."""
        import time

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create 50 entries in target thread
        for i in range(50):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='target-thread',
                source='user',
                content_type='text',
                text_content=f'Target entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * ((i % 10) + 1)] * embedding_dim)

        # Create 100 entries in other threads
        for i in range(100):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'other-thread-{i}',
                source='user',
                content_type='text',
                text_content=f'Other entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.2 * ((i % 10) + 1)] * embedding_dim)

        # Measure search time
        start_time = time.perf_counter()
        results = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            thread_id='target-thread',
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Query should complete in reasonable time (generous threshold for test env)
        assert elapsed_ms < 500  # 500ms threshold
        assert len(results) == 10

    @requires_semantic_search
    async def test_performance_with_medium_filtered_set(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Verify acceptable performance with medium filtered sets (100-500 entries)."""
        import time

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create 200 entries in target thread
        for i in range(200):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='medium-thread',
                source='user',
                content_type='text',
                text_content=f'Medium entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * ((i % 10) + 1)] * embedding_dim)

        # Measure search time
        start_time = time.perf_counter()
        results = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=20,
            thread_id='medium-thread',
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Query should complete in reasonable time
        assert elapsed_ms < 1000  # 1 second threshold
        assert len(results) == 20


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
            thread_id='single-thread',
            source='user',
            content_type='text',
            text_content='Single entry',
            metadata=None,
        )
        await embedding_repo.store(context_id, [0.1] * embedding_dim)

        # Create entries in other threads
        for i in range(5):
            ctx_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'other-{i}',
                source='user',
                content_type='text',
                text_content=f'Other {i}',
                metadata=None,
            )
            await embedding_repo.store(ctx_id, [0.2 * (i + 1)] * embedding_dim)

        # Search for single thread
        results = await embedding_repo.search(
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
                thread_id='only-thread',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for 5 from only-thread
        results = await embedding_repo.search(
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
                thread_id=f'thread-{i}',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with thread_id=None
        results = await embedding_repo.search(
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
                thread_id=f'thread-{i}',
                source='user' if i % 2 == 0 else 'agent',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with source=None
        results = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            source=None,
        )

        # Should return results from both sources
        assert len(results) == 4
        sources = {r['source'] for r in results}
        assert 'user' in sources
        assert 'agent' in sources
