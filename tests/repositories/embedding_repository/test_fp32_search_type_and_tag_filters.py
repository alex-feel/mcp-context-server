"""fp32 semantic search with content_type and tags filters."""

import pytest

from app.backends import StorageBackend
from tests.conftest import requires_semantic_search
from tests.helpers import LOCAL_SCOPE
from tests.helpers import store_single_chunk_embedding


@pytest.mark.asyncio
class TestSemanticSearchContentTypeFilter:
    """Test content_type filtering in semantic search."""

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
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='content-type-test',
            source='user',
            content_type='text',
            text_content='Text only entry',
            metadata=None,
        )
        await store_single_chunk_embedding(embedding_repo, text_id, [0.1] * embedding_dim)

        # Create multimodal entry
        multi_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='content-type-test',
            source='user',
            content_type='multimodal',
            text_content='Entry with image',
            metadata=None,
        )
        await store_single_chunk_embedding(embedding_repo, multi_id, [0.2] * embedding_dim)

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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='content-type-multimodal-test',
                source='agent',
                content_type='text',
                text_content=f'Text entry {i}',
                metadata=None,
            )
            await store_single_chunk_embedding(embedding_repo, ctx_id, [0.1 * (i + 1)] * embedding_dim)

        # Create multimodal entry
        multi_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='content-type-multimodal-test',
            source='agent',
            content_type='multimodal',
            text_content='Multimodal entry with image',
            metadata=None,
        )
        await store_single_chunk_embedding(embedding_repo, multi_id, [0.5] * embedding_dim)

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
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='content-type-none-test',
            source='user',
            content_type='text',
            text_content='Text entry',
            metadata=None,
        )
        await store_single_chunk_embedding(embedding_repo, text_id, [0.1] * embedding_dim)

        multi_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='content-type-none-test',
            source='user',
            content_type='multimodal',
            text_content='Multimodal entry',
            metadata=None,
        )
        await store_single_chunk_embedding(embedding_repo, multi_id, [0.2] * embedding_dim)

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
    """Test tags filtering in semantic search."""

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
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='tags-test',
            source='user',
            content_type='text',
            text_content='Entry with python tag',
            metadata=None,
        )
        await repos.tags.store_tags(id1, ['python'])
        await store_single_chunk_embedding(embedding_repo, id1, [0.1] * embedding_dim)

        id2, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='tags-test',
            source='user',
            content_type='text',
            text_content='Entry with javascript tag',
            metadata=None,
        )
        await repos.tags.store_tags(id2, ['javascript'])
        await store_single_chunk_embedding(embedding_repo, id2, [0.2] * embedding_dim)

        id3, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='tags-test',
            source='user',
            content_type='text',
            text_content='Entry with no matching tags',
            metadata=None,
        )
        await repos.tags.store_tags(id3, ['rust'])
        await store_single_chunk_embedding(embedding_repo, id3, [0.3] * embedding_dim)

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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='single-tag-test',
                source='agent',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            tag = 'important' if i == 0 else f'other-{i}'
            await repos.tags.store_tags(ctx_id, [tag])
            await store_single_chunk_embedding(embedding_repo, ctx_id, [0.1 * (i + 1)] * embedding_dim)

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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='empty-tags-test',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await repos.tags.store_tags(ctx_id, [f'tag-{i}'])
            await store_single_chunk_embedding(embedding_repo, ctx_id, [0.1 * (i + 1)] * embedding_dim)

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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='none-tags-test',
                source='agent',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=None,
            )
            await repos.tags.store_tags(ctx_id, [f'tag-{i}'])
            await store_single_chunk_embedding(embedding_repo, ctx_id, [0.1 * (i + 1)] * embedding_dim)

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
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='combined-tags-thread',
            source='user',
            content_type='text',
            text_content='Python programming',
            metadata=None,
        )
        await repos.tags.store_tags(target_id, ['python'])
        await store_single_chunk_embedding(embedding_repo, target_id, [0.1] * embedding_dim)

        # Create non-matching: wrong source
        wrong_source_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='combined-tags-thread',
            source='agent',  # Different source
            content_type='text',
            text_content='Python from agent',
            metadata=None,
        )
        await repos.tags.store_tags(wrong_source_id, ['python'])
        await store_single_chunk_embedding(embedding_repo, wrong_source_id, [0.2] * embedding_dim)

        # Create non-matching: wrong tag
        wrong_tag_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='combined-tags-thread',
            source='user',
            content_type='text',
            text_content='JavaScript user entry',
            metadata=None,
        )
        await repos.tags.store_tags(wrong_tag_id, ['javascript'])
        await store_single_chunk_embedding(embedding_repo, wrong_tag_id, [0.3] * embedding_dim)

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
