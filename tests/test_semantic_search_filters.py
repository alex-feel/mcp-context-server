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

        This test verifies the fix for the bug where requesting limit=3 with
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
                thread_id=f'thread-{i}',
                source='user' if i % 2 == 0 else 'agent',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

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
                thread_id='thread-a',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

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
                thread_id='small-thread',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for 10 but only 2 exist
        results, _ = await embedding_repo.search(
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
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            start_date=yesterday,
        )
        assert len(results) == 3

        # Search with start_date in the future - should find no entries
        future_date = (datetime.now(UTC) + timedelta(days=30)).strftime('%Y-%m-%d')
        results, _ = await embedding_repo.search(
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
        results, _ = await embedding_repo.search(
            query_embedding=[0.2] * embedding_dim,
            limit=10,
            end_date=tomorrow,
        )
        assert len(results) == 3

        # Search with end_date in the past - should find no entries
        past_date = (datetime.now(UTC) - timedelta(days=30)).strftime('%Y-%m-%d')
        results, _ = await embedding_repo.search(
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
        results, _ = await embedding_repo.search(
            query_embedding=[0.15] * embedding_dim,
            limit=10,
            start_date=yesterday,
            end_date=tomorrow,
        )
        assert len(results) == 5

        # Search with date range in the past - should find none
        far_past = (datetime.now(UTC) - timedelta(days=60)).strftime('%Y-%m-%d')
        past = (datetime.now(UTC) - timedelta(days=30)).strftime('%Y-%m-%d')
        results, _ = await embedding_repo.search(
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
        results, _ = await embedding_repo.search(
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
        results, _ = await embedding_repo.search(
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
        results, _ = await embedding_repo.search(
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
        results, _ = await embedding_repo.search(
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
        results, _ = await embedding_repo.search(
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
                thread_id='only-thread',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

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
                thread_id=f'thread-{i}',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

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
                thread_id=f'thread-{i}',
                source='user' if i % 2 == 0 else 'agent',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

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


@pytest.mark.asyncio
class TestSemanticSearchMetadataFiltering:
    """Test metadata filtering in semantic search (metadata and metadata_filters parameters)."""

    @requires_semantic_search
    async def test_simple_metadata_filter(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test semantic search with simple metadata filter (key=value equality)."""
        import json

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with different status metadata
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='metadata-test-thread',
                source='agent',
                content_type='text',
                text_content=f'Completed task entry {i}',
                metadata=json.dumps({'status': 'completed', 'index': i}),
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='metadata-test-thread',
                source='agent',
                content_type='text',
                text_content=f'Pending task entry {i}',
                metadata=json.dumps({'status': 'pending', 'index': i}),
            )
            await embedding_repo.store(context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with simple metadata filter for status=completed
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata={'status': 'completed'},
        )

        # Should return only 3 entries with status=completed
        assert len(results) == 3
        for result in results:
            metadata = json.loads(result['metadata']) if result['metadata'] else {}
            assert metadata.get('status') == 'completed'

    @requires_semantic_search
    async def test_metadata_filter_with_gt_operator(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test semantic search with advanced metadata filter using gt operator."""
        import json

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with different priority values
        for priority in [1, 3, 5, 7, 9]:
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='priority-test-thread',
                source='user',
                content_type='text',
                text_content=f'Task with priority {priority}',
                metadata=json.dumps({'priority': priority}),
            )
            await embedding_repo.store(context_id, [0.1 * priority] * embedding_dim)

        # Search with metadata_filters for priority > 5
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata_filters=[{'key': 'priority', 'operator': 'gt', 'value': 5}],
        )

        # Should return entries with priority > 5 (7 and 9)
        assert len(results) == 2
        for result in results:
            metadata = json.loads(result['metadata']) if result['metadata'] else {}
            result_priority: int = metadata['priority']
            assert result_priority > 5

    @requires_semantic_search
    async def test_metadata_filter_with_contains_operator(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test semantic search with contains operator for string matching."""
        import json

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with different task_name values
        task_names = ['refactor_auth', 'refactor_database', 'implement_api', 'fix_bug']
        for i, name in enumerate(task_names):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='taskname-test-thread',
                source='agent',
                content_type='text',
                text_content=f'Working on {name}',
                metadata=json.dumps({'task_name': name}),
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for task_name containing 'refactor'
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata_filters=[{'key': 'task_name', 'operator': 'contains', 'value': 'refactor'}],
        )

        # Should return entries with task_name containing 'refactor'
        assert len(results) == 2
        for result in results:
            metadata = json.loads(result['metadata']) if result['metadata'] else {}
            assert 'refactor' in metadata.get('task_name', '')

    @requires_semantic_search
    async def test_metadata_filter_with_exists_operator(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test semantic search with exists operator."""
        import json

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries - some with 'important' flag, some without
        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='exists-test-thread',
                source='user',
                content_type='text',
                text_content=f'Important entry {i}',
                metadata=json.dumps({'important': True, 'index': i}),
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='exists-test-thread',
                source='user',
                content_type='text',
                text_content=f'Regular entry {i}',
                metadata=json.dumps({'index': i}),
            )
            await embedding_repo.store(context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search for entries where 'important' key exists
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata_filters=[{'key': 'important', 'operator': 'exists'}],
        )

        # Should return only 2 entries with 'important' field
        assert len(results) == 2
        for result in results:
            metadata = json.loads(result['metadata']) if result['metadata'] else {}
            assert 'important' in metadata

    @requires_semantic_search
    async def test_combined_metadata_and_other_filters(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test semantic search combining metadata with thread_id and source filters."""
        import json

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries in target thread with source=agent and status=completed
        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='combined-filter-thread',
                source='agent',
                content_type='text',
                text_content=f'Agent completed entry {i}',
                metadata=json.dumps({'status': 'completed'}),
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Create entries in target thread with source=user and status=completed
        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='combined-filter-thread',
                source='user',
                content_type='text',
                text_content=f'User completed entry {i}',
                metadata=json.dumps({'status': 'completed'}),
            )
            await embedding_repo.store(context_id, [0.15 * (i + 1)] * embedding_dim)

        # Create entries in target thread with source=agent and status=pending
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='combined-filter-thread',
                source='agent',
                content_type='text',
                text_content=f'Agent pending entry {i}',
                metadata=json.dumps({'status': 'pending'}),
            )
            await embedding_repo.store(context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with combined filters: thread_id + source + metadata
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            thread_id='combined-filter-thread',
            source='agent',
            metadata={'status': 'completed'},
        )

        # Should return only 2 entries matching all criteria
        assert len(results) == 2
        for result in results:
            assert result['thread_id'] == 'combined-filter-thread'
            assert result['source'] == 'agent'
            metadata = json.loads(result['metadata']) if result['metadata'] else {}
            assert metadata.get('status') == 'completed'

    @requires_semantic_search
    async def test_invalid_metadata_filter_raises_exception(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test that invalid metadata filters raise exception (unified with search_context)."""
        import json

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository
        from app.repositories.embedding_repository import MetadataFilterValidationError

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create test entries
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='invalid-filter-test-thread',
                source='user',
                content_type='text',
                text_content=f'Test entry {i}',
                metadata=json.dumps({'status': 'active'}),
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with invalid metadata_filters (invalid key with SQL injection attempt)
        # Should raise MetadataFilterValidationError, not skip filter silently
        with pytest.raises(MetadataFilterValidationError) as exc_info:
            await embedding_repo.search(
                query_embedding=[0.1] * embedding_dim,
                limit=10,
                metadata_filters=[
                    {'key': 'DROP TABLE;--', 'operator': 'eq', 'value': 'test'},  # Invalid key
                ],
            )

        # Verify exception contains proper error details
        assert exc_info.value.message == 'Metadata filter validation failed'
        assert len(exc_info.value.validation_errors) == 1
        assert 'DROP TABLE' in exc_info.value.validation_errors[0]

    @requires_semantic_search
    async def test_metadata_filter_returns_empty_when_no_matches(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test that metadata filter returns empty list when no entries match."""
        import json

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with status=active
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='no-match-thread',
                source='agent',
                content_type='text',
                text_content=f'Active entry {i}',
                metadata=json.dumps({'status': 'active'}),
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for non-existent status
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata={'status': 'archived'},  # No entries have this status
        )

        # Should return empty list
        assert results == []

    @requires_semantic_search
    async def test_metadata_filter_with_in_operator(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test semantic search with IN operator for list membership."""
        import json

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with different categories
        categories = ['backend', 'frontend', 'devops', 'testing', 'docs']
        for i, category in enumerate(categories):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='in-operator-thread',
                source='user',
                content_type='text',
                text_content=f'Entry in {category}',
                metadata=json.dumps({'category': category}),
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for entries in backend or frontend categories
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata_filters=[{'key': 'category', 'operator': 'in', 'value': ['backend', 'frontend']}],
        )

        # Should return 2 entries (backend and frontend)
        assert len(results) == 2
        for result in results:
            metadata = json.loads(result['metadata']) if result['metadata'] else {}
            assert metadata.get('category') in ['backend', 'frontend']

    @requires_semantic_search
    async def test_metadata_filter_with_in_operator_integer_array(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test semantic search with IN operator using integer array values.

        Regression test: Integer arrays caused silent failures on SQLite
        (type mismatch with json_extract TEXT result) and explicit errors
        on PostgreSQL (asyncpg type mismatch with TEXT cast).
        """
        import json

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with different priority values (integers stored in JSON)
        priorities = [1, 3, 5, 7, 9]
        for i, priority in enumerate(priorities):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='in-operator-int-thread',
                source='agent',
                content_type='text',
                text_content=f'Task with priority {priority}',
                metadata=json.dumps({'priority': priority}),
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for entries with priority IN [5, 9] - INTEGER array
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata_filters=[{'key': 'priority', 'operator': 'in', 'value': [5, 9]}],
        )

        # Should return 2 entries (priority 5 and 9)
        assert len(results) == 2
        for result in results:
            metadata = json.loads(result['metadata']) if result['metadata'] else {}
            assert metadata.get('priority') in [5, 9]

    @requires_semantic_search
    async def test_metadata_filter_with_not_in_operator_integer_array(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test semantic search with NOT IN operator using integer array values.

        Regression test: Integer arrays caused failures in NOT IN operator.
        """
        import json

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with different priority values
        priorities = [1, 2, 3, 4, 5]
        for i, priority in enumerate(priorities):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='not-in-operator-int-thread',
                source='user',
                content_type='text',
                text_content=f'Task with priority {priority}',
                metadata=json.dumps({'priority': priority}),
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for entries with priority NOT IN [1, 2, 3] - INTEGER array
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata_filters=[{'key': 'priority', 'operator': 'not_in', 'value': [1, 2, 3]}],
        )

        # Should return 2 entries (priority 4 and 5)
        assert len(results) == 2
        for result in results:
            metadata = json.loads(result['metadata']) if result['metadata'] else {}
            assert metadata.get('priority') in [4, 5]

    @requires_semantic_search
    async def test_metadata_filter_none_values_ignored(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test that None metadata and metadata_filters don't filter (search all)."""
        import json

        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create test entries with various metadata
        for i in range(4):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='none-filter-thread',
                source='agent',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=json.dumps({'index': i}),
            )
            await embedding_repo.store(context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with None metadata and metadata_filters
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata=None,
            metadata_filters=None,
        )

        # Should return all 4 entries
        assert len(results) == 4


@pytest.mark.asyncio
class TestSemanticSearchContentTypeFilter:
    """Test content_type filtering in semantic search - covers lines 203-206."""

    @requires_semantic_search
    async def test_content_type_filter_text(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test filtering by content_type='text'."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create text entry
        text_id, _ = await repos.context.store_with_deduplication(
            thread_id='content-type-test',
            source='user',
            content_type='text',
            text_content='Text only entry',
            metadata=None,
        )
        await embedding_repo.store(text_id, [0.1] * embedding_dim)

        # Create multimodal entry
        multi_id, _ = await repos.context.store_with_deduplication(
            thread_id='content-type-test',
            source='user',
            content_type='multimodal',
            text_content='Entry with image',
            metadata=None,
        )
        await embedding_repo.store(multi_id, [0.2] * embedding_dim)

        # Search with content_type filter for text
        results, _ = await embedding_repo.search(
            query_embedding=[0.15] * embedding_dim,
            limit=10,
            content_type='text',
        )

        assert len(results) == 1
        assert results[0]['content_type'] == 'text'

    @requires_semantic_search
    async def test_content_type_filter_multimodal(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test filtering by content_type='multimodal'."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create text entries
        for i in range(3):
            ctx_id, _ = await repos.context.store_with_deduplication(
                thread_id='content-type-multimodal-test',
                source='agent',
                content_type='text',
                text_content=f'Text entry {i}',
                metadata=None,
            )
            await embedding_repo.store(ctx_id, [0.1 * (i + 1)] * embedding_dim)

        # Create multimodal entry
        multi_id, _ = await repos.context.store_with_deduplication(
            thread_id='content-type-multimodal-test',
            source='agent',
            content_type='multimodal',
            text_content='Multimodal entry with image',
            metadata=None,
        )
        await embedding_repo.store(multi_id, [0.5] * embedding_dim)

        # Search with content_type filter for multimodal
        results, _ = await embedding_repo.search(
            query_embedding=[0.5] * embedding_dim,
            limit=10,
            content_type='multimodal',
        )

        assert len(results) == 1
        assert results[0]['content_type'] == 'multimodal'

    @requires_semantic_search
    async def test_content_type_none_returns_all(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test that None content_type returns all entries."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create mixed entries
        text_id, _ = await repos.context.store_with_deduplication(
            thread_id='content-type-none-test',
            source='user',
            content_type='text',
            text_content='Text entry',
            metadata=None,
        )
        await embedding_repo.store(text_id, [0.1] * embedding_dim)

        multi_id, _ = await repos.context.store_with_deduplication(
            thread_id='content-type-none-test',
            source='user',
            content_type='multimodal',
            text_content='Multimodal entry',
            metadata=None,
        )
        await embedding_repo.store(multi_id, [0.2] * embedding_dim)

        # Search without content_type filter
        results, _ = await embedding_repo.search(
            query_embedding=[0.15] * embedding_dim,
            limit=10,
            content_type=None,
        )

        assert len(results) == 2
        content_types = {r['content_type'] for r in results}
        assert 'text' in content_types
        assert 'multimodal' in content_types


@pytest.mark.asyncio
class TestSemanticSearchTagsFilter:
    """Test tags filtering in semantic search - covers lines 208-221."""

    @requires_semantic_search
    async def test_tags_filter_or_logic(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test that tags filter uses OR logic."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with different tags
        id1, _ = await repos.context.store_with_deduplication(
            thread_id='tags-test',
            source='user',
            content_type='text',
            text_content='Entry with python tag',
            metadata=None,
        )
        await repos.tags.store_tags(id1, ['python'])
        await embedding_repo.store(id1, [0.1] * embedding_dim)

        id2, _ = await repos.context.store_with_deduplication(
            thread_id='tags-test',
            source='user',
            content_type='text',
            text_content='Entry with javascript tag',
            metadata=None,
        )
        await repos.tags.store_tags(id2, ['javascript'])
        await embedding_repo.store(id2, [0.2] * embedding_dim)

        id3, _ = await repos.context.store_with_deduplication(
            thread_id='tags-test',
            source='user',
            content_type='text',
            text_content='Entry with no matching tags',
            metadata=None,
        )
        await repos.tags.store_tags(id3, ['rust'])
        await embedding_repo.store(id3, [0.3] * embedding_dim)

        # Search with tags filter (OR logic)
        results, _ = await embedding_repo.search(
            query_embedding=[0.15] * embedding_dim,
            limit=10,
            tags=['python', 'javascript'],
        )

        # Should return 2 entries (python OR javascript)
        assert len(results) == 2
        result_ids = {r['id'] for r in results}
        assert id1 in result_ids
        assert id2 in result_ids

    @requires_semantic_search
    async def test_tags_filter_single_tag(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test filtering by a single tag."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries with various tags
        for i in range(3):
            ctx_id, _ = await repos.context.store_with_deduplication(
                thread_id='single-tag-test',
                source='agent',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            tag = 'important' if i == 0 else f'other-{i}'
            await repos.tags.store_tags(ctx_id, [tag])
            await embedding_repo.store(ctx_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for only 'important' tagged entries
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            tags=['important'],
        )

        assert len(results) == 1

    @requires_semantic_search
    async def test_tags_filter_empty_list_returns_all(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test that empty tags list returns all entries."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries
        for i in range(3):
            ctx_id, _ = await repos.context.store_with_deduplication(
                thread_id='empty-tags-test',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await repos.tags.store_tags(ctx_id, [f'tag-{i}'])
            await embedding_repo.store(ctx_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with empty tags list
        results, _ = await embedding_repo.search(
            query_embedding=[0.2] * embedding_dim,
            limit=10,
            tags=[],
        )

        # Empty tags should not filter
        assert len(results) == 3

    @requires_semantic_search
    async def test_tags_filter_none_returns_all(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test that None tags returns all entries."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create entries
        for i in range(4):
            ctx_id, _ = await repos.context.store_with_deduplication(
                thread_id='none-tags-test',
                source='agent',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await repos.tags.store_tags(ctx_id, [f'tag-{i}'])
            await embedding_repo.store(ctx_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with tags=None
        results, _ = await embedding_repo.search(
            query_embedding=[0.25] * embedding_dim,
            limit=10,
            tags=None,
        )

        assert len(results) == 4

    @requires_semantic_search
    async def test_tags_combined_with_other_filters(
        self,
        async_db_with_embeddings: StorageBackend,
        embedding_dim: int,
    ) -> None:
        """Test tags filter combined with thread_id and source filters."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import EmbeddingRepository

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create target entry: in target thread, user source, python tag
        target_id, _ = await repos.context.store_with_deduplication(
            thread_id='combined-tags-thread',
            source='user',
            content_type='text',
            text_content='Python programming',
            metadata=None,
        )
        await repos.tags.store_tags(target_id, ['python'])
        await embedding_repo.store(target_id, [0.1] * embedding_dim)

        # Create non-matching: wrong source
        wrong_source_id, _ = await repos.context.store_with_deduplication(
            thread_id='combined-tags-thread',
            source='agent',  # Different source
            content_type='text',
            text_content='Python from agent',
            metadata=None,
        )
        await repos.tags.store_tags(wrong_source_id, ['python'])
        await embedding_repo.store(wrong_source_id, [0.2] * embedding_dim)

        # Create non-matching: wrong tag
        wrong_tag_id, _ = await repos.context.store_with_deduplication(
            thread_id='combined-tags-thread',
            source='user',
            content_type='text',
            text_content='JavaScript user entry',
            metadata=None,
        )
        await repos.tags.store_tags(wrong_tag_id, ['javascript'])
        await embedding_repo.store(wrong_tag_id, [0.3] * embedding_dim)

        # Search with combined filters
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            thread_id='combined-tags-thread',
            source='user',
            tags=['python'],
        )

        # Should only return the one matching all criteria
        assert len(results) == 1
        assert results[0]['id'] == target_id
