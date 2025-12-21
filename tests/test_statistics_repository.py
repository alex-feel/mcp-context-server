"""Tests for the statistics repository.

This module tests the StatisticsRepository class which provides
database statistics and thread information retrieval.
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
from app.repositories.statistics_repository import StatisticsRepository
from app.schemas import load_schema


@pytest_asyncio.fixture
async def stats_test_db(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Create a test database with the statistics repository."""
    db_path = tmp_path / 'stats_test.db'

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    # Initialize schema
    schema_sql = load_schema('sqlite')

    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(schema_sql)

    await backend.execute_write(_init_schema)

    yield backend

    await backend.shutdown()


@pytest_asyncio.fixture
async def stats_repo(stats_test_db: StorageBackend) -> StatisticsRepository:
    """Create a statistics repository for testing."""
    return StatisticsRepository(stats_test_db)


@pytest_asyncio.fixture
async def repo_container(stats_test_db: StorageBackend) -> RepositoryContainer:
    """Create a full repository container for testing."""
    return RepositoryContainer(stats_test_db)


class TestStatisticsRepository:
    """Test the StatisticsRepository class."""

    @pytest.mark.asyncio
    async def test_get_thread_list_empty(self, stats_repo: StatisticsRepository) -> None:
        """Test getting thread list from empty database."""
        result = await stats_repo.get_thread_list()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_thread_list_with_data(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test getting thread list with data."""

        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            # Insert test data
            cursor.execute(
                '''INSERT INTO context_entries (thread_id, source, content_type, text_content)
                   VALUES ('thread1', 'user', 'text', 'Test 1')''',
            )
            cursor.execute(
                '''INSERT INTO context_entries (thread_id, source, content_type, text_content)
                   VALUES ('thread1', 'agent', 'text', 'Test 2')''',
            )
            cursor.execute(
                '''INSERT INTO context_entries (thread_id, source, content_type, text_content)
                   VALUES ('thread2', 'user', 'multimodal', 'Test 3')''',
            )

        await stats_test_db.execute_write(_insert_data)

        result = await stats_repo.get_thread_list()

        assert len(result) == 2
        # Results should be ordered by last entry
        thread_ids = [t['thread_id'] for t in result]
        assert 'thread1' in thread_ids
        assert 'thread2' in thread_ids

    @pytest.mark.asyncio
    async def test_get_database_statistics_empty(
        self,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test getting statistics from empty database."""
        result = await stats_repo.get_database_statistics()

        assert result['total_entries'] == 0
        assert result['by_source'] == {}
        assert result['by_content_type'] == {}
        assert result['total_images'] == 0
        assert result['unique_tags'] == 0

    @pytest.mark.asyncio
    async def test_get_database_statistics_with_data(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test getting statistics with data."""
        # Use repository container for proper data insertion
        repos = RepositoryContainer(stats_test_db)

        # Insert context entries via repository
        ctx_id1, _ = await repos.context.store_with_deduplication(
            thread_id='thread1',
            source='user',
            content_type='text',
            text_content='Test 1',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            thread_id='thread1',
            source='agent',
            content_type='text',
            text_content='Test 2',
        )
        ctx_id3, _ = await repos.context.store_with_deduplication(
            thread_id='thread1',
            source='user',
            content_type='multimodal',
            text_content='Test 3',
        )

        # Insert tags via repository
        await repos.tags.store_tags(ctx_id1, ['important', 'test'])
        await repos.tags.store_tags(ctx_id2, ['important'])

        # Insert image via repository
        await repos.images.store_images(ctx_id3, [{'data': 'iVBORw0KGgo=', 'mime_type': 'image/png'}])

        result = await stats_repo.get_database_statistics()

        assert result['total_entries'] == 3
        assert result['by_source'] == {'user': 2, 'agent': 1}
        assert result['by_content_type'] == {'text': 2, 'multimodal': 1}
        assert result['total_images'] == 1
        assert result['unique_tags'] == 2  # 'important' and 'test'

    @pytest.mark.asyncio
    async def test_get_thread_statistics_empty_thread(
        self,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test getting thread statistics for nonexistent thread."""
        result = await stats_repo.get_thread_statistics('nonexistent_thread')

        assert result['thread_id'] == 'nonexistent_thread'
        assert result['total_entries'] == 0

    @pytest.mark.asyncio
    async def test_get_thread_statistics_with_data(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test getting thread statistics with data."""
        # Use repository container for proper data insertion
        repos = RepositoryContainer(stats_test_db)

        # Thread 1: 2 entries, both sources, 1 multimodal
        ctx_id1, _ = await repos.context.store_with_deduplication(
            thread_id='thread1',
            source='user',
            content_type='text',
            text_content='Test 1',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            thread_id='thread1',
            source='agent',
            content_type='multimodal',
            text_content='Test 2',
        )

        # Add tags via repository
        await repos.tags.store_tags(ctx_id1, ['important'])
        await repos.tags.store_tags(ctx_id2, ['test'])

        # Add image via repository
        await repos.images.store_images(ctx_id2, [{'data': 'iVBORw0KGgo=', 'mime_type': 'image/png'}])

        result = await stats_repo.get_thread_statistics('thread1')

        assert result['thread_id'] == 'thread1'
        assert result['total_entries'] == 2
        assert result['source_types'] == 2  # Both user and agent
        assert result['text_count'] == 1
        assert result['multimodal_count'] == 1
        assert result['image_count'] == 1
        assert set(result['tags']) == {'important', 'test'}
        assert result['by_source'] == {'user': 1, 'agent': 1}

    @pytest.mark.asyncio
    async def test_get_tag_statistics_empty(
        self,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test getting tag statistics from empty database."""
        result = await stats_repo.get_tag_statistics()

        assert result['unique_tags'] == 0
        assert result['total_tag_uses'] == 0
        assert result['all_tags'] == []
        assert result['top_10_tags'] == []

    @pytest.mark.asyncio
    async def test_get_tag_statistics_with_data(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test getting tag statistics with data."""

        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            # Insert context entries
            cursor.execute(
                '''INSERT INTO context_entries (thread_id, source, content_type, text_content)
                   VALUES ('thread1', 'user', 'text', 'Test 1')''',
            )
            cursor.execute(
                '''INSERT INTO context_entries (thread_id, source, content_type, text_content)
                   VALUES ('thread1', 'agent', 'text', 'Test 2')''',
            )
            cursor.execute(
                '''INSERT INTO context_entries (thread_id, source, content_type, text_content)
                   VALUES ('thread2', 'user', 'text', 'Test 3')''',
            )
            # Insert tags - 'important' used 3 times, 'test' used 2 times, 'unique' used 1 time
            cursor.execute("INSERT INTO tags (context_entry_id, tag) VALUES (1, 'important')")
            cursor.execute("INSERT INTO tags (context_entry_id, tag) VALUES (1, 'test')")
            cursor.execute("INSERT INTO tags (context_entry_id, tag) VALUES (2, 'important')")
            cursor.execute("INSERT INTO tags (context_entry_id, tag) VALUES (2, 'test')")
            cursor.execute("INSERT INTO tags (context_entry_id, tag) VALUES (3, 'important')")
            cursor.execute("INSERT INTO tags (context_entry_id, tag) VALUES (3, 'unique')")

        await stats_test_db.execute_write(_insert_data)

        result = await stats_repo.get_tag_statistics()

        assert result['unique_tags'] == 3
        assert result['total_tag_uses'] == 6

        # Tags should be sorted by usage (descending)
        all_tags = result['all_tags']
        assert len(all_tags) == 3
        assert all_tags[0]['tag'] == 'important'
        assert all_tags[0]['count'] == 3
        assert all_tags[1]['tag'] == 'test'
        assert all_tags[1]['count'] == 2
        assert all_tags[2]['tag'] == 'unique'
        assert all_tags[2]['count'] == 1

        # top_10_tags should be the same since we have less than 10
        assert result['top_10_tags'] == all_tags

    @pytest.mark.asyncio
    async def test_get_tag_statistics_many_tags(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test getting tag statistics with many tags."""

        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            # Insert context entry
            cursor.execute(
                '''INSERT INTO context_entries (thread_id, source, content_type, text_content)
                   VALUES ('thread1', 'user', 'text', 'Test')''',
            )
            # Insert 15 tags to test top_10 filtering
            for i in range(15):
                cursor.execute(f"INSERT INTO tags (context_entry_id, tag) VALUES (1, 'tag{i:02d}')")

        await stats_test_db.execute_write(_insert_data)

        result = await stats_repo.get_tag_statistics()

        assert result['unique_tags'] == 15
        assert len(result['all_tags']) == 15
        assert len(result['top_10_tags']) == 10  # Only top 10

    @pytest.mark.asyncio
    async def test_get_database_statistics_with_path(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
        tmp_path: Path,
    ) -> None:
        """Test getting database statistics with db_path for size calculation."""
        db_path = tmp_path / 'stats_test.db'

        # Insert some data to make the database non-empty
        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO context_entries (thread_id, source, content_type, text_content)
                   VALUES ('thread1', 'user', 'text', 'Test')''',
            )

        await stats_test_db.execute_write(_insert_data)

        result = await stats_repo.get_database_statistics(db_path=db_path)

        assert 'database_size_mb' in result
        assert result['database_size_mb'] >= 0


class TestRepositoryContainerStatistics:
    """Test statistics through the RepositoryContainer."""

    @pytest.mark.asyncio
    async def test_full_statistics_workflow(
        self,
        repo_container: RepositoryContainer,
    ) -> None:
        """Test a full statistics workflow with all repository operations."""
        # Store some context entries using correct API
        context_id1, _ = await repo_container.context.store_with_deduplication(
            thread_id='workflow_thread',
            source='user',
            content_type='text',
            text_content='First entry',
            metadata=json.dumps({'priority': 1}),
        )
        assert context_id1 is not None

        context_id2, _ = await repo_container.context.store_with_deduplication(
            thread_id='workflow_thread',
            source='agent',
            content_type='text',
            text_content='Second entry',
            metadata=json.dumps({'priority': 2}),
        )
        assert context_id2 is not None

        # Add tags
        await repo_container.tags.store_tags(context_id1, ['workflow', 'test'])
        await repo_container.tags.store_tags(context_id2, ['workflow', 'response'])

        # Get database statistics
        stats = await repo_container.statistics.get_database_statistics()

        assert stats['total_entries'] == 2
        assert stats['by_source'] == {'user': 1, 'agent': 1}
        assert stats['unique_tags'] == 3  # workflow, test, response

        # Get thread statistics for specific thread
        thread_stats = await repo_container.statistics.get_thread_statistics('workflow_thread')

        assert thread_stats['thread_id'] == 'workflow_thread'
        assert thread_stats['total_entries'] == 2
        assert thread_stats['source_types'] == 2

        # Get thread list
        thread_list = await repo_container.statistics.get_thread_list()

        assert len(thread_list) == 1
        assert thread_list[0]['thread_id'] == 'workflow_thread'
        assert thread_list[0]['entry_count'] == 2

        # Get tag statistics
        tag_stats = await repo_container.statistics.get_tag_statistics()

        assert tag_stats['unique_tags'] == 3
        assert tag_stats['total_tag_uses'] == 4


class TestStatisticsBackendField:
    """Test that backend field is included in statistics output."""

    @pytest.mark.asyncio
    async def test_get_database_statistics_includes_backend(
        self,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test get_database_statistics includes backend identifier.

        Covers lines 157 and 207 in statistics_repository.py.
        """
        result = await stats_repo.get_database_statistics()

        # Should include backend field
        assert 'backend' in result
        # Since we're using SQLite backend in tests
        assert result['backend'] == 'sqlite'

    @pytest.mark.asyncio
    async def test_get_database_statistics_all_expected_fields(
        self,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test get_database_statistics returns all expected fields."""
        result = await stats_repo.get_database_statistics()

        expected_fields = [
            'total_entries',
            'by_source',
            'by_content_type',
            'total_images',
            'unique_tags',
            'total_threads',
            'avg_entries_per_thread',
            'most_active_threads',
            'top_tags',
            'backend',
        ]

        for field in expected_fields:
            assert field in result, f'Missing expected field: {field}'

    @pytest.mark.asyncio
    async def test_get_database_statistics_most_active_threads_format(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test that most_active_threads has correct format."""
        # Insert some data
        repos = RepositoryContainer(stats_test_db)

        for i in range(3):
            await repos.context.store_with_deduplication(
                thread_id='active-thread',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
            )

        await repos.context.store_with_deduplication(
            thread_id='less-active-thread',
            source='user',
            content_type='text',
            text_content='Single entry',
        )

        result = await stats_repo.get_database_statistics()

        assert 'most_active_threads' in result
        assert len(result['most_active_threads']) == 2

        # Most active should be first
        first_thread = result['most_active_threads'][0]
        assert first_thread['thread_id'] == 'active-thread'
        assert first_thread['count'] == 3

    @pytest.mark.asyncio
    async def test_get_database_statistics_top_tags_format(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test that top_tags has correct format."""
        repos = RepositoryContainer(stats_test_db)

        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='tags-thread',
            source='user',
            content_type='text',
            text_content='Tagged entry',
        )
        await repos.tags.store_tags(ctx_id, ['python', 'testing'])

        result = await stats_repo.get_database_statistics()

        assert 'top_tags' in result
        assert len(result['top_tags']) == 2

        # Each tag entry should have tag and count
        for tag_entry in result['top_tags']:
            assert 'tag' in tag_entry
            assert 'count' in tag_entry


class TestThreadStatisticsDetails:
    """Test detailed thread statistics fields."""

    @pytest.mark.asyncio
    async def test_thread_statistics_includes_timestamps(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test that thread statistics include first/last entry timestamps."""
        repos = RepositoryContainer(stats_test_db)

        await repos.context.store_with_deduplication(
            thread_id='timestamp-thread',
            source='user',
            content_type='text',
            text_content='First entry',
        )

        result = await stats_repo.get_thread_statistics('timestamp-thread')

        assert 'first_entry' in result
        assert 'last_entry' in result
        # Both should be set and equal for a single entry
        assert result['first_entry'] is not None
        assert result['last_entry'] is not None

    @pytest.mark.asyncio
    async def test_thread_statistics_by_source_breakdown(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test that thread statistics include source breakdown."""
        repos = RepositoryContainer(stats_test_db)

        await repos.context.store_with_deduplication(
            thread_id='source-breakdown-thread',
            source='user',
            content_type='text',
            text_content='User entry 1',
        )
        await repos.context.store_with_deduplication(
            thread_id='source-breakdown-thread',
            source='user',
            content_type='text',
            text_content='User entry 2',
        )
        await repos.context.store_with_deduplication(
            thread_id='source-breakdown-thread',
            source='agent',
            content_type='text',
            text_content='Agent entry',
        )

        result = await stats_repo.get_thread_statistics('source-breakdown-thread')

        assert 'by_source' in result
        assert result['by_source'] == {'user': 2, 'agent': 1}
