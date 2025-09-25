"""
Comprehensive tests for the context deduplication functionality.

Tests the deduplication logic implemented in DatabaseConnectionManager.store_context_entry().
Ensures that duplicate entries (same thread_id, source, text_content) update the timestamp
instead of creating new rows.
"""

import asyncio
import json
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from app.db_manager import DatabaseConnectionManager
from app.repositories import RepositoryContainer


@pytest_asyncio.fixture
async def db_manager(tmp_path: Path) -> AsyncGenerator[DatabaseConnectionManager, None]:
    """Create a DatabaseConnectionManager with a test database."""
    db_path = tmp_path / 'test.db'

    # Initialize database with schema
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    schema_path = Path(__file__).parent.parent / 'app' / 'schema.sql'
    conn.executescript(schema_path.read_text())
    conn.close()

    # Create manager with db_path
    manager = DatabaseConnectionManager(db_path)
    await manager.initialize()

    yield manager

    await manager.shutdown()


@pytest_asyncio.fixture
async def repos(db_manager: DatabaseConnectionManager) -> RepositoryContainer:
    """Create a RepositoryContainer with the test database manager."""
    return RepositoryContainer(db_manager)


@pytest.mark.asyncio
class TestDeduplication:
    """Test suite for deduplication functionality."""

    async def test_identical_consecutive_entries_update(
        self,
        repos: RepositoryContainer,
        db_manager: DatabaseConnectionManager,
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
        await asyncio.sleep(1.1)

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

        # Verify only one row exists
        def _count_entries(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM context_entries')
            result = cursor.fetchone()
            return result[0] if result else 0

        count = await db_manager.execute_read(_count_entries)
        assert count == 1

        # Verify updated_at was actually updated
        def _get_timestamps(conn: sqlite3.Connection) -> dict[str, Any]:
            cursor = conn.cursor()
            cursor.execute('SELECT created_at, updated_at FROM context_entries WHERE id = ?', (context_id1,))
            row = cursor.fetchone()
            return {'created_at': row[0], 'updated_at': row[1]} if row else {}

        timestamps = await db_manager.execute_read(_get_timestamps)
        assert timestamps['created_at'] != timestamps['updated_at']

    async def test_non_identical_entries_insert_normally(
        self,
        repos: RepositoryContainer,
        db_manager: DatabaseConnectionManager,
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

        # Verify we have 4 entries
        def _count_entries(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM context_entries')
            result = cursor.fetchone()
            return result[0] if result else 0

        count = await db_manager.execute_read(_count_entries)
        assert count == 4

    async def test_only_latest_entry_checked(self, repos: RepositoryContainer, db_manager: DatabaseConnectionManager) -> None:
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
        def _count_entries(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM context_entries')
            result = cursor.fetchone()
            return result[0] if result else 0

        count = await db_manager.execute_read(_count_entries)
        assert count == 4

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
        db_manager: DatabaseConnectionManager,
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
        def _count_entries(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM context_entries')
            result = cursor.fetchone()
            return result[0] if result else 0

        count = await db_manager.execute_read(_count_entries)
        assert count == 1

    async def test_content_type_does_not_affect_dedup(
        self,
        repos: RepositoryContainer,
        db_manager: DatabaseConnectionManager,
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
        def _count_entries(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM context_entries')
            result = cursor.fetchone()
            return result[0] if result else 0

        count = await db_manager.execute_read(_count_entries)
        assert count == 1

    async def test_rapid_successive_duplicates(
        self,
        repos: RepositoryContainer,
        db_manager: DatabaseConnectionManager,
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
        def _count_entries(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM context_entries')
            result = cursor.fetchone()
            return result[0] if result else 0

        count = await db_manager.execute_read(_count_entries)
        assert count == 1

    async def test_empty_text_content_dedup(self, repos: RepositoryContainer, db_manager: DatabaseConnectionManager) -> None:
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
        def _count_entries(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM context_entries')
            result = cursor.fetchone()
            return result[0] if result else 0

        count = await db_manager.execute_read(_count_entries)
        assert count == 2

    async def test_long_text_content_dedup(self, repos: RepositoryContainer, db_manager: DatabaseConnectionManager) -> None:
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
        def _count_entries(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM context_entries')
            result = cursor.fetchone()
            return result[0] if result else 0

        count = await db_manager.execute_read(_count_entries)
        assert count == 2
