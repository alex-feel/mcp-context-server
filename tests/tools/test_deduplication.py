"""
Comprehensive tests for the context deduplication functionality.

Tests the deduplication logic implemented in StorageBackend context storage.
Ensures that duplicate entries (same thread_id, source, text_content) update the timestamp
instead of creating new rows.
"""

import asyncio
import json
import sqlite3
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.backends import create_backend
from app.repositories import RepositoryContainer


@pytest_asyncio.fixture
async def backend(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Create a StorageBackend with a test database (SQLite only)."""
    db_path = tmp_path / 'test.db'

    # Initialize database with schema
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    conn.executescript(schema_sql)
    conn.close()

    # Create backend with db_path
    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    yield backend

    # Proper async cleanup with timeout protection to prevent hangs
    try:
        await asyncio.wait_for(backend.shutdown(), timeout=5.0)
    except TimeoutError:
        import logging
        logging.getLogger(__name__).warning('Backend shutdown timed out after 5 seconds')
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f'Error during backend shutdown: {e}')


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
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test message content',
            metadata=json.dumps({'key': 'value'}),
        )

        assert context_id1 > 0
        assert was_updated1 is False  # First entry should be inserted

        # Wait to ensure timestamp would differ (SQLite has second precision)
        # Using to_thread to run sync sleep and avoid Windows asyncio deadlock
        await asyncio.to_thread(time.sleep, 1.1)

        # Store identical entry
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
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
        )
        assert len(all_entries) == 1

        # Verify updated_at was actually updated
        entries = await repos.context.get_by_ids([context_id1])
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
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='First message',
            metadata=None,
        )

        assert context_id1 > 0
        assert was_updated1 is False

        # Store different text content
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
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
        entries_thread1, _ = await repos.context.search_contexts(thread_id='test-thread', limit=1000)
        entries_thread2, _ = await repos.context.search_contexts(thread_id='different-thread', limit=1000)
        assert len(entries_thread1) == 3  # 2 user + 1 agent
        assert len(entries_thread2) == 1  # 1 agent
        # Total: 4 entries

    async def test_only_latest_entry_checked(self, repos: RepositoryContainer) -> None:
        """Test that only the LATEST entry is checked for deduplication, not all entries."""
        # Store first entry
        context_id1, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Message A',
            metadata=None,
        )

        # Store different entry
        context_id2, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Message B',
            metadata=None,
        )

        # Store third different entry
        context_id3, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Message C',
            metadata=None,
        )

        # Now store duplicate of FIRST entry (should insert, not update)
        context_id4, was_updated4 = await repos.context.store_with_deduplication(
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
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Message A',  # Same as fourth entry (the latest)
            metadata=None,
        )

        assert context_id5 == context_id4  # Should match the latest entry
        assert was_updated5 is True  # Should be update

        # Verify we have 4 unique entries
        entries, _ = await repos.context.search_contexts(thread_id='test-thread', limit=1000)
        assert len(entries) == 4

    async def test_return_values_correct(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that return values (context_id and was_updated flag) are correct."""
        # First entry: should insert
        context_id1, was_updated1 = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='agent',
            content_type='text',
            text_content='Agent response',
            metadata=None,
        )

        assert isinstance(context_id1, int)
        assert context_id1 > 0
        assert isinstance(was_updated1, bool)
        assert was_updated1 is False

        # Duplicate: should update
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='agent',
            content_type='text',
            text_content='Agent response',
            metadata=None,
        )

        assert isinstance(context_id2, int)
        assert context_id2 == context_id1  # Same ID
        assert isinstance(was_updated2, bool)
        assert was_updated2 is True

        # Different content: should insert
        context_id3, was_updated3 = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='agent',
            content_type='text',
            text_content='Different response',
            metadata=None,
        )

        assert isinstance(context_id3, int)
        assert context_id3 > context_id1  # New ID
        assert isinstance(was_updated3, bool)
        assert was_updated3 is False

    async def test_metadata_changes_do_not_affect_dedup(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that metadata changes don't affect deduplication logic."""
        # Store with metadata
        context_id1, was_updated1 = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
            metadata=json.dumps({'version': 1, 'timestamp': '2024-01-01'}),
        )

        assert was_updated1 is False

        # Store same content with different metadata
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
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
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
            metadata=None,
        )

        assert context_id3 == context_id1  # Should deduplicate
        assert was_updated3 is True

        # Verify only one entry exists
        entries, _ = await repos.context.search_contexts(thread_id='test-thread', limit=1000)
        assert len(entries) == 1

    async def test_content_type_does_not_affect_dedup(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that content_type field doesn't affect deduplication (only thread_id, source, text_content)."""
        # Store as text type
        context_id1, was_updated1 = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Message content',
            metadata=None,
        )

        assert was_updated1 is False

        # Store same with multimodal type (should still deduplicate based on text)
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='multimodal',  # Different content type
            text_content='Message content',
            metadata=None,
        )

        assert context_id2 == context_id1  # Should deduplicate
        assert was_updated2 is True

        # Verify only one entry
        entries, _ = await repos.context.search_contexts(thread_id='test-thread', limit=1000)
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
        entries, _ = await repos.context.search_contexts(thread_id='rapid-thread', limit=1000)
        assert len(entries) == 1

    async def test_empty_text_content_dedup(self, repos: RepositoryContainer) -> None:
        """Test deduplication with empty text content."""
        # Store empty content
        context_id1, was_updated1 = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='',  # Empty
            metadata=None,
        )

        assert was_updated1 is False

        # Store duplicate empty content
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
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
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Not empty',
            metadata=None,
        )

        assert context_id3 != context_id1  # Should be different
        assert was_updated3 is False

        # Verify we have 2 entries
        entries, _ = await repos.context.search_contexts(thread_id='test-thread', limit=1000)
        assert len(entries) == 2

    async def test_long_text_content_dedup(self, repos: RepositoryContainer) -> None:
        """Test deduplication with very long text content."""
        # Create long text
        long_text = 'A' * 10000 + ' middle content ' + 'B' * 10000

        # Store long content
        context_id1, was_updated1 = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='agent',
            content_type='text',
            text_content=long_text,
            metadata=None,
        )

        assert was_updated1 is False

        # Store duplicate long content
        context_id2, was_updated2 = await repos.context.store_with_deduplication(
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
            thread_id='test-thread',
            source='agent',
            content_type='text',
            text_content=different_long_text,
            metadata=None,
        )

        assert context_id3 != context_id1  # Should be different
        assert was_updated3 is False

        # Verify we have 2 entries
        entries, _ = await repos.context.search_contexts(thread_id='test-thread', limit=1000)
        assert len(entries) == 2

    async def test_metadata_updated_during_dedup(self, repos: RepositoryContainer) -> None:
        """Metadata is updated via COALESCE during deduplication."""
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content',
            metadata=json.dumps({'key': 'original'}),
        )
        context_id2, was_updated = await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content',
            metadata=json.dumps({'key': 'updated', 'new_key': 'value'}),
        )
        assert was_updated is True
        assert context_id2 == context_id
        entries = await repos.context.get_by_ids([context_id])
        stored_metadata = json.loads(entries[0]['metadata'])
        assert stored_metadata == {'key': 'updated', 'new_key': 'value'}

    async def test_metadata_preserved_when_none_during_dedup(self, repos: RepositoryContainer) -> None:
        """Metadata is preserved via COALESCE when None is passed during deduplication."""
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content',
            metadata=json.dumps({'key': 'original'}),
        )
        context_id2, was_updated = await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content',
            metadata=None,
        )
        assert was_updated is True
        entries = await repos.context.get_by_ids([context_id])
        stored_metadata = json.loads(entries[0]['metadata'])
        assert stored_metadata == {'key': 'original'}

    async def test_content_type_updated_during_dedup(self, repos: RepositoryContainer) -> None:
        """Content type is updated during deduplication."""
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content', metadata=None,
        )
        context_id2, was_updated = await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='multimodal',
            text_content='Test content', metadata=None,
        )
        assert was_updated is True
        entries = await repos.context.get_by_ids([context_id])
        assert entries[0]['content_type'] == 'multimodal'

    async def test_updated_at_changes_during_dedup(self, repos: RepositoryContainer) -> None:
        """updated_at timestamp changes after deduplication."""
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content', metadata=None,
        )
        await asyncio.to_thread(time.sleep, 1.1)  # SQLite has second precision
        context_id2, was_updated = await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Test content', metadata=None,
        )
        assert was_updated is True
        entries = await repos.context.get_by_ids([context_id])
        assert entries[0]['created_at'] != entries[0]['updated_at']

    async def test_tags_replaced_not_accumulated_during_dedup(self, repos: RepositoryContainer) -> None:
        """Tags are replaced (not accumulated) when replace_tags_for_context is used during dedup."""
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Tag test content', metadata=None,
        )
        await repos.tags.store_tags(context_id, ['a', 'b'])
        # Dedup with was_updated=True
        context_id2, was_updated = await repos.context.store_with_deduplication(
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
            thread_id='test-thread', source='user', content_type='text',
            text_content='Tag preserve test', metadata=None,
        )
        await repos.tags.store_tags(context_id, ['existing'])
        # Dedup fires but no tag operation called (simulates tags=None)
        context_id2, was_updated = await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Tag preserve test', metadata=None,
        )
        assert was_updated is True
        # Don't call any tag function (simulating tags=None from tool layer)
        tags = await repos.tags.get_tags_for_context(context_id)
        assert tags == ['existing']


@pytest.mark.asyncio
class TestDuplicatePreCheck:
    """Tests for check_latest_is_duplicate pre-check method."""

    async def test_check_latest_is_duplicate_found(self, repos: RepositoryContainer) -> None:
        """Pre-check detects duplicate content."""
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Duplicate text', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='test-thread', source='user', text_content='Duplicate text',
        )
        assert result == context_id

    async def test_check_latest_is_duplicate_not_found(self, repos: RepositoryContainer) -> None:
        """Pre-check returns None for different content."""
        await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Original text', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='test-thread', source='user', text_content='Different text',
        )
        assert result is None

    async def test_check_latest_is_duplicate_empty_thread(self, repos: RepositoryContainer) -> None:
        """Pre-check returns None for empty thread."""
        result = await repos.context.check_latest_is_duplicate(
            thread_id='nonexistent', source='user', text_content='Any text',
        )
        assert result is None

    async def test_check_latest_different_source(self, repos: RepositoryContainer) -> None:
        """Pre-check returns None when source differs."""
        await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Same text', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='test-thread', source='agent', text_content='Same text',
        )
        assert result is None

    async def test_check_latest_only_checks_latest(self, repos: RepositoryContainer) -> None:
        """Pre-check only checks the latest entry, not older ones."""
        await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='Old matching text', metadata=None,
        )
        await repos.context.store_with_deduplication(
            thread_id='test-thread', source='user', content_type='text',
            text_content='New different text', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='test-thread', source='user', text_content='Old matching text',
        )
        assert result is None  # Latest is "New different text", not matching


@pytest.mark.asyncio
class TestDeduplicationInterleaving:
    """Tests for the interleaving check in deduplication logic.

    Validates that deduplication is suppressed when opposite-source entries
    exist between the candidate duplicate and the new entry, preserving
    chronological ordering for repeated identical messages in conversations.
    """

    async def test_dedup_blocked_when_opposite_source_interleaves(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Dedup is suppressed when an opposite-source entry exists after the candidate."""
        # User says "Proceed"
        id_a, _ = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Proceed', metadata=None,
        )
        # Agent responds
        await repos.context.store_with_deduplication(
            thread_id='t1', source='agent', content_type='text',
            text_content='Working on it', metadata=None,
        )
        # User says "Proceed" again (new conversational turn)
        id_c, was_updated = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Proceed', metadata=None,
        )
        assert id_c != id_a, 'Should create a new entry, not update the old one'
        assert id_c > id_a, 'New entry should have a higher id'
        assert was_updated is False, 'Should be an insertion, not an update'
        # Verify original entry still exists unchanged
        entries = await repos.context.get_by_ids([id_a])
        assert len(entries) == 1
        assert entries[0]['text_content'] == 'Proceed'

    async def test_dedup_allowed_when_no_interleaving(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Dedup proceeds normally for genuine rapid duplicates (retry protection)."""
        id_a, _ = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Hello', metadata=None,
        )
        id_b, was_updated = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Hello', metadata=None,
        )
        assert id_b == id_a, 'Should update the existing entry'
        assert was_updated is True

    async def test_dedup_blocked_symmetrically_for_agent(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Interleaving check works for agent source with user interleaving."""
        id_a, _ = await repos.context.store_with_deduplication(
            thread_id='t1', source='agent', content_type='text',
            text_content='status update', metadata=None,
        )
        # User interleaves
        await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='acknowledged', metadata=None,
        )
        # Agent sends identical text again
        id_c, was_updated = await repos.context.store_with_deduplication(
            thread_id='t1', source='agent', content_type='text',
            text_content='status update', metadata=None,
        )
        assert id_c != id_a
        assert was_updated is False

    async def test_dedup_precheck_respects_interleaving(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Pre-check method returns None when interleaving is detected."""
        id_a, _ = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Proceed', metadata=None,
        )
        await repos.context.store_with_deduplication(
            thread_id='t1', source='agent', content_type='text',
            text_content='Done', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='t1', source='user', text_content='Proceed',
        )
        assert result is None, (
            f'Expected None (interleaving suppresses dedup), got {result}'
        )

    async def test_dedup_single_source_thread(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Dedup works normally in a single-source thread (no opposite-source entries)."""
        id_a, _ = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Only users here', metadata=None,
        )
        await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Different message', metadata=None,
        )
        # Store duplicate of first when latest is different -- should insert (not latest)
        id_c, was_updated_c = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Different message', metadata=None,
        )
        # Latest matches, no opposite-source entries at all -- dedup should work
        assert was_updated_c is True

    async def test_chronological_ordering_preserved_after_fix(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """End-to-end ordering: alternating identical messages create distinct entries."""
        ids: list[int] = []
        sources = ['user', 'agent', 'user', 'agent', 'user']
        texts = ['Go', 'Done', 'Go', 'Done again', 'Go']
        for source, text in zip(sources, texts, strict=True):
            ctx_id, _ = await repos.context.store_with_deduplication(
                thread_id='t1', source=source, content_type='text',
                text_content=text, metadata=None,
            )
            ids.append(ctx_id)
        # All 5 should be distinct entries
        assert len(set(ids)) == 5, f'Expected 5 distinct ids, got {ids}'
        # Monotonically increasing
        for i in range(1, len(ids)):
            assert ids[i] > ids[i - 1], f'ids not monotonically increasing: {ids}'
        # Verify all entries exist
        entries, _ = await repos.context.search_contexts(thread_id='t1', limit=1000)
        assert len(entries) == 5


@pytest.mark.asyncio
class TestBatchPreCheckInterleaving:
    """Tests for the batch pre-check optimization and its interaction with interleaving.

    The batch pre-check calls check_latest_is_duplicate before generating
    embeddings/summaries. These tests verify the pre-check returns correct
    results in batch-relevant scenarios.
    """

    async def test_batch_precheck_returns_id_for_duplicate(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Pre-check returns existing ID when entry is a genuine duplicate (no interleaving).

        In batch context, this would cause the tool layer to skip embedding/summary
        generation for this entry, saving LLM API calls.
        """
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='batch-t1', source='user', content_type='text',
            text_content='Batch duplicate', metadata=None,
        )
        # Pre-check should identify this as a duplicate
        result = await repos.context.check_latest_is_duplicate(
            thread_id='batch-t1', source='user', text_content='Batch duplicate',
        )
        assert result == ctx_id, (
            'Pre-check should return existing ID for genuine duplicate'
        )

    async def test_batch_precheck_returns_none_when_interleaving(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Pre-check returns None when interleaving is detected.

        In batch context, this would cause the tool layer to generate new
        embeddings/summaries for this entry (new conversational turn).
        """
        # Store user entry, then agent entry (creates interleaving)
        await repos.context.store_with_deduplication(
            thread_id='batch-t1', source='user', content_type='text',
            text_content='Batch entry', metadata=None,
        )
        await repos.context.store_with_deduplication(
            thread_id='batch-t1', source='agent', content_type='text',
            text_content='Agent response', metadata=None,
        )
        # Pre-check for same user text should return None (interleaving detected)
        result = await repos.context.check_latest_is_duplicate(
            thread_id='batch-t1', source='user', text_content='Batch entry',
        )
        assert result is None, (
            'Pre-check should return None when interleaving detected '
            '(agent entry exists after user candidate)'
        )
