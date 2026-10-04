"""Tests for `ContextRepository.store_with_deduplication`.

Covers when a store updates the latest identical entry instead of inserting, and how the update treats
metadata, tags, content_type and updated_at.
"""

import asyncio
import json
import time

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.repositories import RepositoryContainer
from tests.helpers import LOCAL_SCOPE


@pytest_asyncio.fixture
async def repos(backend: StorageBackend) -> RepositoryContainer:
    """Create a RepositoryContainer with the test database manager."""
    return RepositoryContainer(backend)


@pytest.mark.asyncio
class TestDeduplication:
    """Test suite for deduplication functionality."""

    async def test_identical_consecutive_entries_update(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that identical consecutive entries update the timestamp instead of inserting."""
        # Store first entry
        context_id1, was_updated1 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test message content',
            metadata=json.dumps({'key': 'value'}),
        )

        assert isinstance(context_id1, str)
        assert len(context_id1) == 32
        assert was_updated1 is False  # First entry should be inserted

        # Wait to ensure timestamp would differ (SQLite has second precision)
        # Using to_thread to run sync sleep and avoid Windows asyncio deadlock
        await asyncio.to_thread(time.sleep, 1.1)

        # Store identical entry
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test message content',
            metadata=json.dumps({'key': 'different'}),  # Different metadata should not affect dedup
        )

        assert context_id2 == context_id1  # Should return same ID
        assert was_updated2 is True  # Should indicate update

        # Verify only one row exists using repository
        all_entries, _ = await repos.context.search_contexts(
            thread_id='test-thread',
            source=None,
            content_type=None,
            tags=None,
            metadata_filters=None,
            limit=1000,
            offset=0,
            scope=LOCAL_SCOPE,
        )
        assert len(all_entries) == 1

        # Verify updated_at was actually updated
        entries = await repos.context.get_by_ids([context_id1], scope=LOCAL_SCOPE)
        assert len(entries) == 1
        entry = entries[0]
        assert entry['created_at'] != entry['updated_at']

    async def test_non_identical_entries_insert_normally(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that non-identical entries still insert as new rows."""
        # Store first entry
        context_id1, was_updated1 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='First message',
            metadata=None,
        )

        assert isinstance(context_id1, str)
        assert len(context_id1) == 32
        assert was_updated1 is False

        # Store different text content
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Different message',  # Different text
            metadata=None,
        )

        assert context_id2 != context_id1  # Should be different ID
        assert context_id2 > context_id1  # Should be newer
        assert was_updated2 is False  # Should be insertion

        # Store different source
        context_id3, was_updated3 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='agent',  # Different source
            content_type='text',
            text_content='Different message',  # Same text as id2
            metadata=None,
        )

        assert context_id3 != context_id2  # Should be different ID
        assert context_id3 > context_id2  # Should be newer
        assert was_updated3 is False  # Should be insertion

        # Store different thread
        context_id4, was_updated4 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='different-thread',  # Different thread
            source='agent',
            content_type='text',
            text_content='Different message',  # Same text as id2 and id3
            metadata=None,
        )

        assert context_id4 != context_id3  # Should be different ID
        assert context_id4 > context_id3  # Should be newer
        assert was_updated4 is False  # Should be insertion

        # Verify we have 4 entries - search across all threads
        entries_thread1, _ = await repos.context.search_contexts(thread_id='test-thread', limit=1000, scope=LOCAL_SCOPE)
        entries_thread2, _ = await repos.context.search_contexts(thread_id='different-thread', limit=1000, scope=LOCAL_SCOPE)
        assert len(entries_thread1) == 3  # 2 user + 1 agent
        assert len(entries_thread2) == 1  # 1 agent
        # Total: 4 entries

    async def test_only_latest_entry_checked(self, repos: RepositoryContainer) -> None:
        """Test that only the LATEST entry is checked for deduplication, not all entries."""
        # Store first entry
        context_id1, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Message A',
            metadata=None,
        )

        # Store different entry
        context_id2, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Message B',
            metadata=None,
        )

        # Store third different entry
        context_id3, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Message C',
            metadata=None,
        )

        # Now store duplicate of FIRST entry (should insert, not update)
        context_id4, was_updated4 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Message A',  # Same as first entry
            metadata=None,
        )

        assert context_id4 != context_id1  # Should NOT match first entry
        assert context_id4 > context_id3  # Should be a new entry
        assert was_updated4 is False  # Should be insertion, not update

        # Now store duplicate of LATEST entry (should update)
        context_id5, was_updated5 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Message A',  # Same as fourth entry (the latest)
            metadata=None,
        )

        assert context_id5 == context_id4  # Should match the latest entry
        assert was_updated5 is True  # Should be update

        # Verify we have 4 unique entries
        entries, _ = await repos.context.search_contexts(thread_id='test-thread', limit=1000, scope=LOCAL_SCOPE)
        assert len(entries) == 4

    async def test_return_values_correct(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that return values (context_id and was_updated flag) are correct."""
        # First entry: should insert
        context_id1, was_updated1 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='agent',
            content_type='text',
            text_content='Agent response',
            metadata=None,
        )

        assert isinstance(context_id1, str)
        assert len(context_id1) == 32
        assert isinstance(was_updated1, bool)
        assert was_updated1 is False

        # Duplicate: should update
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='agent',
            content_type='text',
            text_content='Agent response',
            metadata=None,
        )

        assert isinstance(context_id2, str)
        assert context_id2 == context_id1  # Same ID
        assert isinstance(was_updated2, bool)
        assert was_updated2 is True

        # Different content: should insert
        context_id3, was_updated3 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='agent',
            content_type='text',
            text_content='Different response',
            metadata=None,
        )

        assert isinstance(context_id3, str)
        assert context_id3 > context_id1  # New ID (UUIDv7 lex ordering)
        assert isinstance(was_updated3, bool)
        assert was_updated3 is False

    async def test_metadata_changes_do_not_affect_dedup(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that metadata changes don't affect deduplication logic."""
        # Store with metadata
        context_id1, was_updated1 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
            metadata=json.dumps({'version': 1, 'timestamp': '2024-01-01'}),
        )

        assert was_updated1 is False

        # Store same content with different metadata
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
            metadata=json.dumps({'version': 2, 'timestamp': '2024-01-02', 'extra': 'data'}),
        )

        assert context_id2 == context_id1  # Should deduplicate
        assert was_updated2 is True

        # Store same content with no metadata
        context_id3, was_updated3 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
            metadata=None,
        )

        assert context_id3 == context_id1  # Should deduplicate
        assert was_updated3 is True

        # Verify only one entry exists
        entries, _ = await repos.context.search_contexts(thread_id='test-thread', limit=1000, scope=LOCAL_SCOPE)
        assert len(entries) == 1

    async def test_content_type_does_not_affect_dedup(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that content_type field doesn't affect deduplication (only thread_id, source, text_content)."""
        # Store as text type
        context_id1, was_updated1 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Message content',
            metadata=None,
        )

        assert was_updated1 is False

        # Store same with multimodal type (should still deduplicate based on text)
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='multimodal',  # Different content type
            text_content='Message content',
            metadata=None,
        )

        assert context_id2 == context_id1  # Should deduplicate
        assert was_updated2 is True

        # Verify only one entry
        entries, _ = await repos.context.search_contexts(thread_id='test-thread', limit=1000, scope=LOCAL_SCOPE)
        assert len(entries) == 1

    async def test_rapid_successive_duplicates(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test handling of rapid successive duplicate entries."""
        # Store multiple duplicates in quick succession
        results = []
        for i in range(5):
            context_id, was_updated = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='rapid-thread',
                source='agent',
                content_type='text',
                text_content='Rapid message',
                metadata=json.dumps({'attempt': i}),
            )
            results.append((context_id, was_updated))

        # First should insert, rest should update
        assert results[0][1] is False  # First is insertion
        for i in range(1, 5):
            assert results[i][0] == results[0][0]  # Same ID
            assert results[i][1] is True  # Updates

        # Verify only one entry exists
        entries, _ = await repos.context.search_contexts(thread_id='rapid-thread', limit=1000, scope=LOCAL_SCOPE)
        assert len(entries) == 1

    async def test_empty_text_content_dedup(self, repos: RepositoryContainer) -> None:
        """Test deduplication with empty text content."""
        # Store empty content
        context_id1, was_updated1 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='',  # Empty
            metadata=None,
        )

        assert was_updated1 is False

        # Store duplicate empty content
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='',  # Empty again
            metadata=None,
        )

        assert context_id2 == context_id1  # Should deduplicate
        assert was_updated2 is True

        # Store non-empty content
        context_id3, was_updated3 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Not empty',
            metadata=None,
        )

        assert context_id3 != context_id1  # Should be different
        assert was_updated3 is False

        # Verify we have 2 entries
        entries, _ = await repos.context.search_contexts(thread_id='test-thread', limit=1000, scope=LOCAL_SCOPE)
        assert len(entries) == 2

    async def test_long_text_content_dedup(self, repos: RepositoryContainer) -> None:
        """Test deduplication with very long text content."""
        # Create long text
        long_text = 'A' * 10000 + ' middle content ' + 'B' * 10000

        # Store long content
        context_id1, was_updated1 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='agent',
            content_type='text',
            text_content=long_text,
            metadata=None,
        )

        assert was_updated1 is False

        # Store duplicate long content
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='agent',
            content_type='text',
            text_content=long_text,  # Exact same long text
            metadata=None,
        )

        assert context_id2 == context_id1  # Should deduplicate
        assert was_updated2 is True

        # Store slightly different long content
        different_long_text = 'A' * 10000 + ' DIFFERENT middle ' + 'B' * 10000
        context_id3, was_updated3 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='agent',
            content_type='text',
            text_content=different_long_text,
            metadata=None,
        )

        assert context_id3 != context_id1  # Should be different
        assert was_updated3 is False

        # Verify we have 2 entries
        entries, _ = await repos.context.search_contexts(thread_id='test-thread', limit=1000, scope=LOCAL_SCOPE)
        assert len(entries) == 2

    async def test_metadata_updated_during_dedup(self, repos: RepositoryContainer) -> None:
        """Metadata is updated via COALESCE during deduplication."""
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content',
            metadata=json.dumps({'key': 'original'}),
        )
        context_id2, was_updated = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content',
            metadata=json.dumps({'key': 'updated', 'new_key': 'value'}),
        )
        assert was_updated is True
        assert context_id2 == context_id
        entries = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        stored_metadata = json.loads(entries[0]['metadata'])
        assert stored_metadata == {'key': 'updated', 'new_key': 'value'}

    async def test_metadata_preserved_when_none_during_dedup(self, repos: RepositoryContainer) -> None:
        """Metadata is preserved via COALESCE when None is passed during deduplication."""
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content',
            metadata=json.dumps({'key': 'original'}),
        )
        context_id2, was_updated = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content',
            metadata=None,
        )
        assert was_updated is True
        entries = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        stored_metadata = json.loads(entries[0]['metadata'])
        assert stored_metadata == {'key': 'original'}

    async def test_content_type_updated_during_dedup(self, repos: RepositoryContainer) -> None:
        """Content type is updated during deduplication."""
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content', metadata=None,
        )
        context_id2, was_updated = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread', source='user', content_type='multimodal',
            text_content='Test content', metadata=None,
        )
        assert was_updated is True
        entries = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        assert entries[0]['content_type'] == 'multimodal'

    async def test_updated_at_changes_during_dedup(self, repos: RepositoryContainer) -> None:
        """updated_at timestamp changes after deduplication."""
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content', metadata=None,
        )
        await asyncio.to_thread(time.sleep, 1.1)  # SQLite has second precision
        context_id2, was_updated = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content', metadata=None,
        )
        assert was_updated is True
        entries = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        assert entries[0]['created_at'] != entries[0]['updated_at']

    async def test_tags_replaced_not_accumulated_during_dedup(self, repos: RepositoryContainer) -> None:
        """Tags are replaced (not accumulated) when replace_tags_for_context is used during dedup."""
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Tag test content', metadata=None,
        )
        await repos.tags.store_tags(context_id, ['a', 'b'])
        # Dedup with was_updated=True
        context_id2, was_updated = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Tag test content', metadata=None,
        )
        assert was_updated is True
        await repos.tags.replace_tags_for_context(context_id, ['c', 'd'])
        tags = await repos.tags.get_tags_for_context(context_id)
        assert sorted(tags) == ['c', 'd']

    async def test_tags_preserved_when_none_during_dedup(self, repos: RepositoryContainer) -> None:
        """Tags are preserved when not provided during deduplication."""
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Tag preserve test', metadata=None,
        )
        await repos.tags.store_tags(context_id, ['existing'])
        # Dedup fires but no tag operation called (simulates tags=None)
        context_id2, was_updated = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Tag preserve test', metadata=None,
        )
        assert was_updated is True
        # Don't call any tag function (simulating tags=None from tool layer)
        tags = await repos.tags.get_tags_for_context(context_id)
        assert tags == ['existing']
