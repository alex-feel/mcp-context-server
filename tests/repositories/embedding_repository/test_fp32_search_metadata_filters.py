"""fp32 semantic search with metadata and metadata_filters filters."""

import pytest

from app.backends import StorageBackend
from tests.conftest import requires_semantic_search
from tests.helpers import LOCAL_SCOPE
from tests.helpers import store_single_chunk_embedding


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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='metadata-test-thread',
                source='agent',
                content_type='text',
                text_content=f'Completed task entry {i}',
                metadata=json.dumps({'status': 'completed', 'index': i}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='metadata-test-thread',
                source='agent',
                content_type='text',
                text_content=f'Pending task entry {i}',
                metadata=json.dumps({'status': 'pending', 'index': i}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with simple metadata filter for status=completed
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata={'status': 'completed'},
            scope=LOCAL_SCOPE,
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='priority-test-thread',
                source='user',
                content_type='text',
                text_content=f'Task with priority {priority}',
                metadata=json.dumps({'priority': priority}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * priority] * embedding_dim)

        # Search with metadata_filters for priority > 5
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata_filters=[{'key': 'priority', 'operator': 'gt', 'value': 5}],
            scope=LOCAL_SCOPE,
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='taskname-test-thread',
                source='agent',
                content_type='text',
                text_content=f'Working on {name}',
                metadata=json.dumps({'task_name': name}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for task_name containing 'refactor'
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata_filters=[{'key': 'task_name', 'operator': 'contains', 'value': 'refactor'}],
            scope=LOCAL_SCOPE,
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='exists-test-thread',
                source='user',
                content_type='text',
                text_content=f'Important entry {i}',
                metadata=json.dumps({'important': True, 'index': i}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='exists-test-thread',
                source='user',
                content_type='text',
                text_content=f'Regular entry {i}',
                metadata=json.dumps({'index': i}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search for entries where 'important' key exists
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata_filters=[{'key': 'important', 'operator': 'exists'}],
            scope=LOCAL_SCOPE,
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='combined-filter-thread',
                source='agent',
                content_type='text',
                text_content=f'Agent completed entry {i}',
                metadata=json.dumps({'status': 'completed'}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Create entries in target thread with source=user and status=completed
        for i in range(2):
            context_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='combined-filter-thread',
                source='user',
                content_type='text',
                text_content=f'User completed entry {i}',
                metadata=json.dumps({'status': 'completed'}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.15 * (i + 1)] * embedding_dim)

        # Create entries in target thread with source=agent and status=pending
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='combined-filter-thread',
                source='agent',
                content_type='text',
                text_content=f'Agent pending entry {i}',
                metadata=json.dumps({'status': 'pending'}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.2 * (i + 1)] * embedding_dim)

        # Search with combined filters: thread_id + source + metadata
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            thread_id='combined-filter-thread',
            source='agent',
            metadata={'status': 'completed'},
            scope=LOCAL_SCOPE,
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
        from app.repositories.embedding_repository.records import MetadataFilterValidationError

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        embedding_repo = EmbeddingRepository(backend)

        # Create test entries
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='invalid-filter-test-thread',
                source='user',
                content_type='text',
                text_content=f'Test entry {i}',
                metadata=json.dumps({'status': 'active'}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with invalid metadata_filters (invalid key with SQL injection attempt)
        # Should raise MetadataFilterValidationError, not skip filter silently
        with pytest.raises(MetadataFilterValidationError) as exc_info:
            await embedding_repo.search(
                query_embedding=[0.1] * embedding_dim,
                limit=10,
                metadata_filters=[
                    {'key': 'DROP TABLE;--', 'operator': 'eq', 'value': 'test'},  # Invalid key
                ],
                scope=LOCAL_SCOPE,
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='no-match-thread',
                source='agent',
                content_type='text',
                text_content=f'Active entry {i}',
                metadata=json.dumps({'status': 'active'}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for non-existent status
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata={'status': 'archived'},  # No entries have this status
            scope=LOCAL_SCOPE,
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='in-operator-thread',
                source='user',
                content_type='text',
                text_content=f'Entry in {category}',
                metadata=json.dumps({'category': category}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for entries in backend or frontend categories
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata_filters=[{'key': 'category', 'operator': 'in', 'value': ['backend', 'frontend']}],
            scope=LOCAL_SCOPE,
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

        Integer array values must match integer metadata values despite the
        TEXT result of json_extract on SQLite and the TEXT cast on PostgreSQL.
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='in-operator-int-thread',
                source='agent',
                content_type='text',
                text_content=f'Task with priority {priority}',
                metadata=json.dumps({'priority': priority}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for entries with priority IN [5, 9] - INTEGER array
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata_filters=[{'key': 'priority', 'operator': 'in', 'value': [5, 9]}],
            scope=LOCAL_SCOPE,
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

        Integer array values must be excluded correctly by the NOT IN operator.
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='not-in-operator-int-thread',
                source='user',
                content_type='text',
                text_content=f'Task with priority {priority}',
                metadata=json.dumps({'priority': priority}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search for entries with priority NOT IN [1, 2, 3] - INTEGER array
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata_filters=[{'key': 'priority', 'operator': 'not_in', 'value': [1, 2, 3]}],
            scope=LOCAL_SCOPE,
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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='none-filter-thread',
                source='agent',
                content_type='text',
                text_content=f'Entry {i}',
                metadata=json.dumps({'index': i}),
            )
            await store_single_chunk_embedding(embedding_repo, context_id, [0.1 * (i + 1)] * embedding_dim)

        # Search with None metadata and metadata_filters
        results, _ = await embedding_repo.search(
            query_embedding=[0.1] * embedding_dim,
            limit=10,
            metadata=None,
            metadata_filters=None,
            scope=LOCAL_SCOPE,
        )

        # Should return all 4 entries
        assert len(results) == 4
