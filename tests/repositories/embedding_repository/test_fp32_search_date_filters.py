"""fp32 semantic search with start_date and end_date filters."""

import pytest

from app.backends import StorageBackend
from tests.conftest import requires_semantic_search
from tests.helpers import LOCAL_SCOPE
from tests.helpers import store_single_chunk_embedding


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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='date-filter-thread',
                source='user',
                content_type='text',
                text_content=f'Date filter test entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with start_date in the past - should find all entries
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            start_date=yesterday,
            scope=LOCAL_SCOPE,
        )
        assert len(results) == 3

        # Search with start_date in the future - should find no entries
        future_date = (datetime.now(UTC) + timedelta(days=30)).strftime('%Y-%m-%d')
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            start_date=future_date,
            scope=LOCAL_SCOPE,
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='end-date-thread',
                source='agent',
                content_type='text',
                text_content=f'End date filter entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with end_date in the future - should find all entries
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
        results, _ = await embedding_repo.search(
            query_embedding=[0.2] * embedding_dim,
            limit=10,
            end_date=tomorrow,
            scope=LOCAL_SCOPE,
        )
        assert len(results) == 3

        # Search with end_date in the past - should find no entries
        past_date = (datetime.now(UTC) - timedelta(days=30)).strftime('%Y-%m-%d')
        results, _ = await embedding_repo.search(
            query_embedding=[0.2] * embedding_dim,
            limit=10,
            end_date=past_date,
            scope=LOCAL_SCOPE,
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='range-thread',
                source='user',
                content_type='text',
                text_content=f'Date range entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.15 * (i + 1)] * embedding_dim)

        # Search with valid date range (yesterday to tomorrow) - should find all
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
        results, _ = await embedding_repo.search(
            query_embedding=[0.15] * embedding_dim,
            limit=10,
            start_date=yesterday,
            end_date=tomorrow,
            scope=LOCAL_SCOPE,
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
            scope=LOCAL_SCOPE,
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='target-date-thread',
                source='user',
                content_type='text',
                text_content=f'Target thread entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='other-date-thread',
                source='user',
                content_type='text',
                text_content=f'Other thread entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with date filter and thread_id - should find 2 entries from target thread
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            thread_id='target-date-thread',
            start_date=yesterday,
            end_date=tomorrow,
            scope=LOCAL_SCOPE,
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='mixed-source-thread',
                source='user',
                content_type='text',
                text_content=f'User entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='mixed-source-thread',
                source='agent',
                content_type='text',
                text_content=f'Agent entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with date filter and source - should find 2 user entries
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            source='user',
            start_date=yesterday,
            end_date=tomorrow,
            scope=LOCAL_SCOPE,
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='no-date-filter-thread',
                source='user',
                content_type='text',
                text_content=f'No date filter entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.25 * (i + 1)] * embedding_dim)

        # Search with None dates - should find all entries
        results, _ = await embedding_repo.search(
            query_embedding=[0.25] * embedding_dim,
            limit=10,
            start_date=None,
            end_date=None,
            scope=LOCAL_SCOPE,
        )
        assert len(results) == 4
