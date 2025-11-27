"""Extended tests for the context repository.

This module provides additional tests for ContextRepository to improve coverage
of search operations, edge cases, and error handling.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import create_backend
from app.backends.base import StorageBackend
from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from app.schemas import load_schema


@pytest_asyncio.fixture
async def context_test_db(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Create a test database for context repository testing."""
    db_path = tmp_path / 'context_test.db'

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    schema_sql = load_schema('sqlite')

    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(schema_sql)

    await backend.execute_write(_init_schema)

    yield backend

    await backend.shutdown()


@pytest_asyncio.fixture
async def context_repo(context_test_db: StorageBackend) -> ContextRepository:
    """Create a context repository for testing."""
    return ContextRepository(context_test_db)


@pytest_asyncio.fixture
async def repos(context_test_db: StorageBackend) -> RepositoryContainer:
    """Create full repository container."""
    return RepositoryContainer(context_test_db)


class TestContextRepositorySearch:
    """Test search functionality of ContextRepository."""

    @pytest.mark.asyncio
    async def test_search_empty_database(self, context_repo: ContextRepository) -> None:
        """Test searching empty database returns empty results."""
        rows, stats = await context_repo.search_contexts()

        assert rows == []
        assert 'execution_time_ms' in stats

    @pytest.mark.asyncio
    async def test_search_by_thread_id(
        self,
        context_repo: ContextRepository,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching by thread_id."""
        await repos.context.store_with_deduplication(
            thread_id='thread_a',
            source='user',
            content_type='text',
            text_content='Message A',
        )
        await repos.context.store_with_deduplication(
            thread_id='thread_b',
            source='user',
            content_type='text',
            text_content='Message B',
        )

        rows, stats = await context_repo.search_contexts(thread_id='thread_a')

        assert len(rows) == 1
        assert rows[0]['thread_id'] == 'thread_a'

    @pytest.mark.asyncio
    async def test_search_by_source(
        self,
        context_repo: ContextRepository,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching by source."""
        await repos.context.store_with_deduplication(
            thread_id='source_thread',
            source='user',
            content_type='text',
            text_content='User message',
        )
        await repos.context.store_with_deduplication(
            thread_id='source_thread',
            source='agent',
            content_type='text',
            text_content='Agent message',
        )

        rows, stats = await context_repo.search_contexts(source='agent')

        assert len(rows) == 1
        assert rows[0]['source'] == 'agent'

    @pytest.mark.asyncio
    async def test_search_by_content_type(
        self,
        context_repo: ContextRepository,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching by content_type."""
        await repos.context.store_with_deduplication(
            thread_id='type_thread',
            source='user',
            content_type='text',
            text_content='Text only',
        )
        await repos.context.store_with_deduplication(
            thread_id='type_thread',
            source='user',
            content_type='multimodal',
            text_content='With image',
        )

        rows, stats = await context_repo.search_contexts(content_type='multimodal')

        assert len(rows) == 1
        assert rows[0]['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    async def test_search_by_tags(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching by tags."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            thread_id='tag_thread',
            source='user',
            content_type='text',
            text_content='Tagged 1',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            thread_id='tag_thread',
            source='user',
            content_type='text',
            text_content='Tagged 2',
        )

        await repos.tags.store_tags(ctx_id1, ['important', 'review'])
        await repos.tags.store_tags(ctx_id2, ['other'])

        rows, stats = await repos.context.search_contexts(tags=['important'])

        assert len(rows) == 1
        assert rows[0]['id'] == ctx_id1

    @pytest.mark.asyncio
    async def test_search_with_limit(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with limit parameter."""
        for i in range(10):
            await repos.context.store_with_deduplication(
                thread_id='limit_thread',
                source='user',
                content_type='text',
                text_content=f'Message {i}',
            )

        rows, stats = await repos.context.search_contexts(
            thread_id='limit_thread',
            limit=5,
        )

        assert len(rows) == 5

    @pytest.mark.asyncio
    async def test_search_with_offset(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with offset parameter."""
        for i in range(10):
            await repos.context.store_with_deduplication(
                thread_id='offset_thread',
                source='user',
                content_type='text',
                text_content=f'Message {i}',
            )

        rows, stats = await repos.context.search_contexts(
            thread_id='offset_thread',
            limit=3,
            offset=5,
        )

        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_search_with_metadata_simple(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with simple metadata filter."""
        await repos.context.store_with_deduplication(
            thread_id='meta_thread',
            source='user',
            content_type='text',
            text_content='Priority 1',
            metadata=json.dumps({'priority': 1}),
        )
        await repos.context.store_with_deduplication(
            thread_id='meta_thread',
            source='user',
            content_type='text',
            text_content='Priority 2',
            metadata=json.dumps({'priority': 2}),
        )

        rows, stats = await repos.context.search_contexts(
            thread_id='meta_thread',
            metadata={'priority': 1},
        )

        assert len(rows) == 1
        # Parse metadata JSON and verify
        metadata = json.loads(rows[0]['metadata'])
        assert metadata['priority'] == 1

    @pytest.mark.asyncio
    async def test_search_with_explain_query(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with explain_query=True."""
        await repos.context.store_with_deduplication(
            thread_id='explain_thread',
            source='user',
            content_type='text',
            text_content='Test entry',
        )

        rows, stats = await repos.context.search_contexts(
            thread_id='explain_thread',
            explain_query=True,
        )

        assert len(rows) == 1
        assert 'execution_time_ms' in stats
        assert 'filters_applied' in stats

    @pytest.mark.asyncio
    async def test_search_combined_filters(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with multiple filters combined."""
        await repos.context.store_with_deduplication(
            thread_id='combo_thread',
            source='user',
            content_type='text',
            text_content='User text',
        )
        await repos.context.store_with_deduplication(
            thread_id='combo_thread',
            source='agent',
            content_type='text',
            text_content='Agent text',
        )
        await repos.context.store_with_deduplication(
            thread_id='other_thread',
            source='user',
            content_type='text',
            text_content='Other user',
        )

        rows, stats = await repos.context.search_contexts(
            thread_id='combo_thread',
            source='user',
            content_type='text',
        )

        assert len(rows) == 1
        assert rows[0]['thread_id'] == 'combo_thread'
        assert rows[0]['source'] == 'user'


class TestContextRepositoryDeduplication:
    """Test deduplication logic in ContextRepository."""

    @pytest.mark.asyncio
    async def test_deduplication_updates_timestamp(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that duplicate content updates timestamp instead of inserting."""
        ctx_id1, was_updated1 = await repos.context.store_with_deduplication(
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Same content',
        )
        assert was_updated1 is False  # First insert, not an update

        ctx_id2, was_updated2 = await repos.context.store_with_deduplication(
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Same content',
        )
        assert was_updated2 is True  # Second call with same content, should update
        assert ctx_id1 == ctx_id2  # Should be same ID

    @pytest.mark.asyncio
    async def test_deduplication_different_content(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that different content creates new entry."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Content A',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Content B',
        )

        assert ctx_id1 != ctx_id2  # Different content = different IDs

    @pytest.mark.asyncio
    async def test_deduplication_different_source(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that same content from different source creates new entry."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Same content',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            thread_id='dedup_thread',
            source='agent',  # Different source
            content_type='text',
            text_content='Same content',
        )

        assert ctx_id1 != ctx_id2  # Different source = different entry

    @pytest.mark.asyncio
    async def test_deduplication_different_thread(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that same content in different thread creates new entry."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            thread_id='thread_1',
            source='user',
            content_type='text',
            text_content='Same content',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            thread_id='thread_2',  # Different thread
            source='user',
            content_type='text',
            text_content='Same content',
        )

        assert ctx_id1 != ctx_id2  # Different thread = different entry


class TestContextRepositoryDelete:
    """Test delete operations in ContextRepository."""

    @pytest.mark.asyncio
    async def test_delete_by_thread_id(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test deleting entries by thread_id."""
        await repos.context.store_with_deduplication(
            thread_id='del_thread',
            source='user',
            content_type='text',
            text_content='To delete',
        )
        await repos.context.store_with_deduplication(
            thread_id='keep_thread',
            source='user',
            content_type='text',
            text_content='To keep',
        )

        deleted = await repos.context.delete_by_thread(thread_id='del_thread')

        assert deleted == 1

        # Verify deletion
        rows, _ = await repos.context.search_contexts(thread_id='del_thread')
        assert len(rows) == 0

        # Verify other thread kept
        rows, _ = await repos.context.search_contexts(thread_id='keep_thread')
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_delete_multiple_entries(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test deleting multiple entries from same thread."""
        await repos.context.store_with_deduplication(
            thread_id='multi_del_thread',
            source='user',
            content_type='text',
            text_content='Message 1',
        )
        await repos.context.store_with_deduplication(
            thread_id='multi_del_thread',
            source='agent',
            content_type='text',
            text_content='Message 2',
        )
        await repos.context.store_with_deduplication(
            thread_id='multi_del_thread',
            source='user',
            content_type='text',
            text_content='Message 3',
        )

        deleted = await repos.context.delete_by_thread(thread_id='multi_del_thread')

        assert deleted == 3

        # Verify all deleted
        rows, _ = await repos.context.search_contexts(thread_id='multi_del_thread')
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_thread(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test deleting from nonexistent thread returns 0."""
        deleted = await repos.context.delete_by_thread(thread_id='nonexistent')

        assert deleted == 0


class TestContextRepositoryGetById:
    """Test get_by_ids operations."""

    @pytest.mark.asyncio
    async def test_get_by_ids_single(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting single entry by ID."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='get_thread',
            source='user',
            content_type='text',
            text_content='Test entry',
        )

        rows = await repos.context.get_by_ids([ctx_id])

        assert len(rows) == 1
        assert rows[0]['id'] == ctx_id

    @pytest.mark.asyncio
    async def test_get_by_ids_multiple(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting multiple entries by IDs."""
        ids = []
        for i in range(3):
            ctx_id, _ = await repos.context.store_with_deduplication(
                thread_id='multi_get',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
            )
            ids.append(ctx_id)

        rows = await repos.context.get_by_ids(ids)

        assert len(rows) == 3
        returned_ids = {r['id'] for r in rows}
        assert returned_ids == set(ids)

    @pytest.mark.asyncio
    async def test_get_by_ids_empty_list(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting entries with empty ID list."""
        rows = await repos.context.get_by_ids([])

        assert rows == []

    @pytest.mark.asyncio
    async def test_get_by_ids_nonexistent(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting nonexistent IDs returns empty."""
        rows = await repos.context.get_by_ids([999998, 999999])

        assert rows == []

    @pytest.mark.asyncio
    async def test_get_by_ids_partial_match(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting mix of existing and nonexistent IDs."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='partial_get',
            source='user',
            content_type='text',
            text_content='Exists',
        )

        rows = await repos.context.get_by_ids([ctx_id, 999999])

        assert len(rows) == 1
        assert rows[0]['id'] == ctx_id


class TestContextRepositoryUpdate:
    """Test update operations in ContextRepository."""

    @pytest.mark.asyncio
    async def test_check_entry_exists(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test checking if entry exists."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='exists_thread',
            source='user',
            content_type='text',
            text_content='Exists',
        )

        exists = await repos.context.check_entry_exists(ctx_id)
        assert exists is True

        not_exists = await repos.context.check_entry_exists(999999)
        assert not_exists is False

    @pytest.mark.asyncio
    async def test_get_content_type(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting content type by ID."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='type_thread',
            source='user',
            content_type='text',
            text_content='Text content',
        )

        content_type = await repos.context.get_content_type(ctx_id)

        assert content_type == 'text'

    @pytest.mark.asyncio
    async def test_get_content_type_nonexistent(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting content type for nonexistent entry."""
        content_type = await repos.context.get_content_type(999999)

        assert content_type is None

    @pytest.mark.asyncio
    async def test_update_content_type(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test updating content type."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='update_type',
            source='user',
            content_type='text',
            text_content='Content',
        )

        await repos.context.update_content_type(ctx_id, 'multimodal')

        new_type = await repos.context.get_content_type(ctx_id)
        assert new_type == 'multimodal'
