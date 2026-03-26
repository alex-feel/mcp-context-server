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

    @pytest.mark.asyncio
    async def test_deduplication_updates_metadata_coalesce(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Deduplication COALESCE: new metadata replaces existing."""
        ctx_id1, was_updated1 = await repos.context.store_with_deduplication(
            thread_id='coalesce-thread',
            source='user',
            content_type='text',
            text_content='Coalesce test content',
            metadata=json.dumps({'key': 'original'}),
        )
        assert was_updated1 is False

        ctx_id2, was_updated2 = await repos.context.store_with_deduplication(
            thread_id='coalesce-thread',
            source='user',
            content_type='text',
            text_content='Coalesce test content',
            metadata=json.dumps({'key': 'updated'}),
        )
        assert was_updated2 is True
        assert ctx_id2 == ctx_id1

        rows, _ = await context_repo.search_contexts(thread_id='coalesce-thread')
        meta = json.loads(rows[0]['metadata'])
        assert meta['key'] == 'updated'

    @pytest.mark.asyncio
    async def test_deduplication_preserves_metadata_on_none(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Deduplication COALESCE(NULL, existing) preserves existing metadata."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            thread_id='preserve-meta-thread',
            source='user',
            content_type='text',
            text_content='Preserve meta content',
            metadata=json.dumps({'preserved': 'yes'}),
        )

        ctx_id2, was_updated = await repos.context.store_with_deduplication(
            thread_id='preserve-meta-thread',
            source='user',
            content_type='text',
            text_content='Preserve meta content',
            metadata=None,
        )
        assert was_updated is True
        assert ctx_id2 == ctx_id1

        rows, _ = await context_repo.search_contexts(thread_id='preserve-meta-thread')
        meta = json.loads(rows[0]['metadata'])
        assert meta['preserved'] == 'yes'

    @pytest.mark.asyncio
    async def test_deduplication_summary_coalesce(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Deduplication COALESCE preserves existing summary when new is None."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            thread_id='summary-coalesce-thread',
            source='user',
            content_type='text',
            text_content='Summary coalesce test',
            summary='Existing summary',
        )

        ctx_id2, was_updated = await repos.context.store_with_deduplication(
            thread_id='summary-coalesce-thread',
            source='user',
            content_type='text',
            text_content='Summary coalesce test',
            summary=None,
        )
        assert was_updated is True
        assert ctx_id2 == ctx_id1

        result = await context_repo.get_summary(ctx_id1)
        assert result == 'Existing summary'

    @pytest.mark.asyncio
    async def test_deduplication_content_hash_path(
        self, repos: RepositoryContainer,
    ) -> None:
        """Deduplication uses content_hash for fast comparison."""
        from app.repositories.context_repository import compute_content_hash

        text = 'Hash-based dedup test content'
        expected_hash = compute_content_hash(text)
        assert expected_hash is not None

        ctx_id1, _ = await repos.context.store_with_deduplication(
            thread_id='hash-dedup-thread',
            source='user',
            content_type='text',
            text_content=text,
        )

        ctx_id2, was_updated = await repos.context.store_with_deduplication(
            thread_id='hash-dedup-thread',
            source='user',
            content_type='text',
            text_content=text,
        )
        assert was_updated is True
        assert ctx_id2 == ctx_id1

    @pytest.mark.asyncio
    async def test_deduplication_empty_summary_normalized(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Deduplication normalizes empty/whitespace summary to None."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='norm-summary-thread',
            source='user',
            content_type='text',
            text_content='Normalized summary test',
            summary='   ',
        )
        result = await context_repo.get_summary(ctx_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_deduplication_check_empty_database(
        self, context_repo: ContextRepository,
    ) -> None:
        """check_latest_is_duplicate on empty DB returns None."""
        result = await context_repo.check_latest_is_duplicate(
            thread_id='empty-thread',
            source='user',
            text_content='Some content',
        )
        assert result is None


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

        exists, source = await repos.context.check_entry_exists(ctx_id)
        assert exists is True
        assert source == 'user'

        not_exists, no_source = await repos.context.check_entry_exists(999999)
        assert not_exists is False
        assert no_source is None

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


class TestContextRepositoryGetSummary:
    """Test get_summary method of ContextRepository."""

    @pytest.mark.asyncio
    async def test_get_summary_existing_entry(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """get_summary returns summary string for entry with summary."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='summary-thread',
            source='user',
            content_type='text',
            text_content='Entry with summary',
            summary='This is a test summary.',
        )
        result = await context_repo.get_summary(ctx_id)
        assert result == 'This is a test summary.'

    @pytest.mark.asyncio
    async def test_get_summary_no_summary(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """get_summary returns None for entry without summary."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='no-summary-thread',
            source='user',
            content_type='text',
            text_content='Entry without summary',
        )
        result = await context_repo.get_summary(ctx_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_summary_empty_string_normalized_to_none(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """get_summary normalizes empty/whitespace-only summary to None."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='empty-summary-thread',
            source='user',
            content_type='text',
            text_content='Entry with empty summary',
            summary='   ',
        )
        result = await context_repo.get_summary(ctx_id)
        assert result is None


class TestContextRepositoryCheckDuplicate:
    """Test check_latest_is_duplicate method of ContextRepository."""

    @pytest.mark.asyncio
    async def test_check_latest_is_duplicate_match(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Returns context_id when latest entry has identical content."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='dup-check-thread',
            source='user',
            content_type='text',
            text_content='Duplicate content check',
        )
        result = await context_repo.check_latest_is_duplicate(
            thread_id='dup-check-thread',
            source='user',
            text_content='Duplicate content check',
        )
        assert result == ctx_id

    @pytest.mark.asyncio
    async def test_check_latest_is_duplicate_no_match(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Returns None when latest entry has different content."""
        await repos.context.store_with_deduplication(
            thread_id='no-dup-thread',
            source='user',
            content_type='text',
            text_content='Original content',
        )
        result = await context_repo.check_latest_is_duplicate(
            thread_id='no-dup-thread',
            source='user',
            text_content='Different content',
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_check_latest_is_duplicate_different_thread(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Returns None when content matches but thread_id differs."""
        await repos.context.store_with_deduplication(
            thread_id='thread-a',
            source='user',
            content_type='text',
            text_content='Same content different thread',
        )
        result = await context_repo.check_latest_is_duplicate(
            thread_id='thread-b',
            source='user',
            text_content='Same content different thread',
        )
        assert result is None


class TestContextRepositoryPatchMetadata:
    """Test patch_metadata method of ContextRepository (RFC 7396)."""

    @pytest.mark.asyncio
    async def test_patch_metadata_adds_new_key(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Patching adds a new key to existing metadata."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='patch-add-thread',
            source='user',
            content_type='text',
            text_content='Patch test entry',
            metadata=json.dumps({'existing': 'value'}),
        )
        success, fields = await context_repo.patch_metadata(ctx_id, {'new_key': 'new_value'})
        assert success is True
        assert 'metadata' in fields

        rows, _ = await context_repo.search_contexts(thread_id='patch-add-thread')
        assert len(rows) == 1
        meta = json.loads(rows[0]['metadata'])
        assert meta['existing'] == 'value'
        assert meta['new_key'] == 'new_value'

    @pytest.mark.asyncio
    async def test_patch_metadata_updates_existing_key(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Patching updates an existing key's value."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='patch-update-thread',
            source='user',
            content_type='text',
            text_content='Patch update test',
            metadata=json.dumps({'status': 'pending'}),
        )
        success, fields = await context_repo.patch_metadata(ctx_id, {'status': 'done'})
        assert success is True

        rows, _ = await context_repo.search_contexts(thread_id='patch-update-thread')
        meta = json.loads(rows[0]['metadata'])
        assert meta['status'] == 'done'

    @pytest.mark.asyncio
    async def test_patch_metadata_deletes_key_with_null(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Patching with null value deletes the key (RFC 7396)."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='patch-delete-thread',
            source='user',
            content_type='text',
            text_content='Patch delete test',
            metadata=json.dumps({'keep': 'yes', 'remove': 'me'}),
        )
        success, _ = await context_repo.patch_metadata(ctx_id, {'remove': None})
        assert success is True

        rows, _ = await context_repo.search_contexts(thread_id='patch-delete-thread')
        meta = json.loads(rows[0]['metadata'])
        assert 'keep' in meta
        assert 'remove' not in meta

    @pytest.mark.asyncio
    async def test_patch_metadata_nonexistent_entry(
        self, context_repo: ContextRepository,
    ) -> None:
        """Patching nonexistent entry returns (False, [])."""
        success, fields = await context_repo.patch_metadata(999999, {'key': 'value'})
        assert success is False
        assert fields == []

    @pytest.mark.asyncio
    async def test_patch_metadata_empty_patch(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Empty patch is a no-op for data but updates timestamp."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='patch-empty-thread',
            source='user',
            content_type='text',
            text_content='Patch empty test',
            metadata=json.dumps({'unchanged': 'value'}),
        )
        success, fields = await context_repo.patch_metadata(ctx_id, {})
        assert success is True
        assert 'metadata' in fields

        rows, _ = await context_repo.search_contexts(thread_id='patch-empty-thread')
        meta = json.loads(rows[0]['metadata'])
        assert meta['unchanged'] == 'value'


class TestContextRepositoryBatchDelete:
    """Test delete_contexts_batch method of ContextRepository."""

    @pytest.mark.asyncio
    async def test_delete_contexts_batch(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Batch delete removes multiple entries."""
        ids = []
        for i in range(3):
            ctx_id, _ = await repos.context.store_with_deduplication(
                thread_id='batch-del-thread',
                source='user',
                content_type='text',
                text_content=f'Batch delete entry {i}',
            )
            ids.append(ctx_id)

        deleted_count, criteria = await context_repo.delete_contexts_batch(context_ids=ids)
        assert deleted_count == 3

        rows = await context_repo.get_by_ids(ids)
        assert rows == []

    @pytest.mark.asyncio
    async def test_delete_contexts_batch_partial_ids(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Batch delete with mix of existing and nonexistent IDs."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='partial-del-thread',
            source='user',
            content_type='text',
            text_content='Entry to delete partially',
        )
        deleted_count, _ = await context_repo.delete_contexts_batch(
            context_ids=[ctx_id, 999998, 999999],
        )
        assert deleted_count == 1

    @pytest.mark.asyncio
    async def test_delete_contexts_batch_empty_list(
        self, context_repo: ContextRepository,
    ) -> None:
        """Batch delete with empty list returns 0."""
        deleted_count, _ = await context_repo.delete_contexts_batch(context_ids=[])
        assert deleted_count == 0


class TestComputeContentHash:
    """Test the compute_content_hash utility function."""

    def test_compute_content_hash_consistency(self) -> None:
        """Same content produces same hash."""
        from app.repositories.context_repository import compute_content_hash

        hash1 = compute_content_hash('test content')
        hash2 = compute_content_hash('test content')
        assert hash1 == hash2

    def test_compute_content_hash_different_content(self) -> None:
        """Different content produces different hashes."""
        from app.repositories.context_repository import compute_content_hash

        hash1 = compute_content_hash('content A')
        hash2 = compute_content_hash('content B')
        assert hash1 != hash2
