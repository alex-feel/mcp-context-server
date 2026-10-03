"""Timing of filtered fp32 semantic search over small and medium entry sets."""

import pytest

from app.backends import StorageBackend
from tests.conftest import requires_semantic_search
from tests.helpers import LOCAL_SCOPE
from tests.helpers import store_single_chunk_embedding


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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='target-thread',
                source='user',
                content_type='text',
                text_content=f'Target entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * ((i % 10) + 1)] * embedding_dim)

        # Create 100 entries in other threads
        for i in range(100):
            context_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id=f'other-thread-{i}',
                source='user',
                content_type='text',
                text_content=f'Other entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.2 * ((i % 10) + 1)] * embedding_dim)

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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='medium-thread',
                source='user',
                content_type='text',
                text_content=f'Medium entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * ((i % 10) + 1)] * embedding_dim)

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
