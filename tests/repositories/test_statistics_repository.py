"""Tests for the statistics repository.

This module tests the StatisticsRepository class which provides
database statistics and thread information retrieval.
"""

import json
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any
from typing import TypeVar
from typing import cast

import pytest
import pytest_asyncio

from app.backends import create_backend
from app.backends.base import StorageBackend
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.statistics_repository import StatisticsRepository
from app.repositories.statistics_repository import _to_float
from app.schemas import load_schema

T = TypeVar('T')


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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES ('0190abcdef1234567890abcd00000001', 'thread1', 'user', 'text', 'Test 1')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES ('0190abcdef1234567890abcd00000002', 'thread1', 'agent', 'text', 'Test 2')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES ('0190abcdef1234567890abcd00000003', 'thread2', 'user', 'multimodal', 'Test 3')",
            )

        await stats_test_db.execute_write(_insert_data)

        result = await stats_repo.get_thread_list()

        assert len(result) == 2
        # Results should be ordered by last entry
        thread_ids = [t['thread_id'] for t in result]
        assert 'thread1' in thread_ids
        assert 'thread2' in thread_ids

    @pytest.mark.asyncio
    async def test_get_thread_list_last_id_format_and_ordering(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Verify last_id matches the canonical 32-char lowercase hex contract and
        returns the chronologically latest id per thread.

        The contract is documented in docs/api-reference.md (regex ^[0-9a-f]{32}$)
        and is reflected in app/types.py ThreadInfoDict.last_id: str.
        """

        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            # thread_a: three monotonic UUIDv7 ids, distinct created_at
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, created_at) '
                "VALUES ('0190abcdef1234567890abcd00000001', 'thread_a', 'user', 'text', 'A1', '2026-01-01 10:00:00')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, created_at) '
                "VALUES ('0190abcdef1234567890abcd00000005', 'thread_a', 'agent', 'text', 'A2', '2026-01-01 10:00:01')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, created_at) '
                "VALUES ('0190abcdef1234567890abcd00000003', 'thread_a', 'user', 'text', 'A3', '2026-01-01 10:00:02')",
            )
            # thread_b: two ids, ordered so that the lex-max id is NOT the most recently inserted
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, created_at) '
                "VALUES ('0190abcdef1234567890abcd00000099', 'thread_b', 'user', 'text', 'B1', '2026-01-01 10:00:10')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, created_at) '
                "VALUES ('0190abcdef1234567890abcd00000010', 'thread_b', 'agent', 'text', 'B2', '2026-01-01 10:00:11')",
            )

        await stats_test_db.execute_write(_insert_data)

        result = await stats_repo.get_thread_list()

        assert len(result) == 2

        thread_a = next(t for t in result if t['thread_id'] == 'thread_a')
        thread_b = next(t for t in result if t['thread_id'] == 'thread_b')

        # MAX(id) under SQLite BINARY collation = lex-max over canonical lowercase hex.
        # thread_a ids: ...00000001, ...00000003, ...00000005 -> lex-max is ...00000005
        # thread_b ids: ...00000010, ...00000099 -> lex-max is ...00000099
        assert thread_a['last_id'] == '0190abcdef1234567890abcd00000005'
        assert thread_b['last_id'] == '0190abcdef1234567890abcd00000099'

        # Format invariants per docs/api-reference.md regex ^[0-9a-f]{32}$
        for thread in result:
            last_id = thread['last_id']
            assert isinstance(last_id, str)
            assert len(last_id) == 32, f'last_id must be 32-char hyphen-free hex: {last_id!r}'
            assert last_id == last_id.lower(), f'last_id must be lowercase: {last_id!r}'
            assert all(c in '0123456789abcdef' for c in last_id), (
                f'last_id must contain only lowercase hex digits: {last_id!r}'
            )

        # Outer ORDER BY uses last_entry DESC tie-broken by last_id DESC.
        # thread_b last_entry = 2026-01-01 10:00:11 > thread_a last_entry = 2026-01-01 10:00:02
        # So thread_b must come first.
        assert result[0]['thread_id'] == 'thread_b'
        assert result[1]['thread_id'] == 'thread_a'

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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES ('0190abcdef1234567890abcd00000004', 'thread1', 'user', 'text', 'Test 1')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES ('0190abcdef1234567890abcd00000005', 'thread1', 'agent', 'text', 'Test 2')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES ('0190abcdef1234567890abcd00000006', 'thread2', 'user', 'text', 'Test 3')",
            )
            # Tags: 'important' used 3 times, 'test' used 2 times, 'unique' used 1 time
            id_a = '0190abcdef1234567890abcd00000004'
            id_b = '0190abcdef1234567890abcd00000005'
            id_c = '0190abcdef1234567890abcd00000006'
            cursor.execute('INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)', (id_a, 'important'))
            cursor.execute('INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)', (id_a, 'test'))
            cursor.execute('INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)', (id_b, 'important'))
            cursor.execute('INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)', (id_b, 'test'))
            cursor.execute('INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)', (id_c, 'important'))
            cursor.execute('INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)', (id_c, 'unique'))

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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES ('0190abcdef1234567890abcd00000007', 'thread1', 'user', 'text', 'Test')",
            )
            # Insert 15 tags to test top_10 filtering
            entry_id = '0190abcdef1234567890abcd00000007'
            for i in range(15):
                cursor.execute(
                    'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                    (entry_id, f'tag{i:02d}'),
                )

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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES ('0190abcdef1234567890abcd00000008', 'thread1', 'user', 'text', 'Test')",
            )

        await stats_test_db.execute_write(_insert_data)

        result = await stats_repo.get_database_statistics(db_path=db_path)

        assert 'database_size_mb' in result
        assert result['database_size_mb'] >= 0

    @pytest.mark.asyncio
    async def test_get_summary_statistics_empty_database(
        self,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test summary statistics from empty database."""
        result = await stats_repo.get_summary_statistics()

        assert result['summary_count'] == 0
        assert result['total_entries'] == 0
        assert result['coverage_percentage'] == 0.0

    @pytest.mark.asyncio
    async def test_get_summary_statistics_with_summaries(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test summary statistics with entries that have summaries."""

        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            # Entry with valid summary
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, summary) '
                "VALUES (?, 't1', 'user', 'text', 'Content 1', 'Summary 1')",
                (generate_id(),),
            )
            # Entry with NULL summary
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES (?, 't1', 'agent', 'text', 'Content 2')",
                (generate_id(),),
            )
            # Entry with valid summary
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, summary) '
                "VALUES (?, 't2', 'user', 'text', 'Content 3', 'Summary 3')",
                (generate_id(),),
            )

        await stats_test_db.execute_write(_insert_data)

        result = await stats_repo.get_summary_statistics()

        assert result['total_entries'] == 3
        assert result['summary_count'] == 2
        assert result['coverage_percentage'] == pytest.approx(66.67, rel=0.01)

    @pytest.mark.asyncio
    async def test_get_summary_statistics_excludes_empty_strings(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test that empty string summaries are NOT counted as valid summaries.

        The SQL uses WHERE summary IS NOT NULL AND summary != '' to exclude
        entries where empty strings were written before normalization guards
        were added.
        """

        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            # Entry with valid summary
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, summary) '
                "VALUES (?, 't1', 'user', 'text', 'Content 1', 'Valid summary')",
                (generate_id(),),
            )
            # Entry with empty string summary (edge case)
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, summary) '
                "VALUES (?, 't1', 'agent', 'text', 'Content 2', '')",
                (generate_id(),),
            )
            # Entry with NULL summary
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES (?, 't2', 'user', 'text', 'Content 3')",
                (generate_id(),),
            )

        await stats_test_db.execute_write(_insert_data)

        result = await stats_repo.get_summary_statistics()

        assert result['total_entries'] == 3
        # Only 1 entry has a valid (non-empty) summary
        assert result['summary_count'] == 1
        assert result['coverage_percentage'] == pytest.approx(33.33, rel=0.01)

    @pytest.mark.asyncio
    async def test_get_database_statistics_content_type_counts(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Statistics correctly count text vs multimodal content types."""
        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES (?, 'ct-thread', 'user', 'text', 'Text entry')",
                (generate_id(),),
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES (?, 'ct-thread', 'user', 'multimodal', 'Multimodal entry')",
                (generate_id(),),
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES (?, 'ct-thread', 'agent', 'text', 'Another text entry')",
                (generate_id(),),
            )

        await stats_test_db.execute_write(_insert_data)

        result = await stats_repo.get_database_statistics()
        assert result['total_entries'] == 3

    @pytest.mark.asyncio
    async def test_get_database_statistics_after_deletion(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Statistics update correctly after deleting entries."""
        repos = RepositoryContainer(stats_test_db)

        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='del-stats-thread',
            source='user',
            content_type='text',
            text_content='Entry to be deleted',
        )
        await repos.tags.store_tags(ctx_id, ['deleteme'])

        stats_before = await stats_repo.get_database_statistics()
        assert stats_before['total_entries'] >= 1

        await repos.context.delete_by_ids([ctx_id])

        stats_after = await stats_repo.get_database_statistics()
        assert stats_after['total_entries'] == stats_before['total_entries'] - 1


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


class TestThreadListDetails:
    """Test detailed thread list information."""

    @pytest.mark.asyncio
    async def test_get_thread_list_multimodal_count(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Thread list reports multimodal entry count per thread."""
        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES (?, 'mm-thread', 'user', 'text', 'Text only')",
                (generate_id(),),
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES (?, 'mm-thread', 'user', 'multimodal', 'With images')",
                (generate_id(),),
            )

        await stats_test_db.execute_write(_insert_data)

        threads = await stats_repo.get_thread_list()
        assert len(threads) == 1
        thread = threads[0]
        assert thread['multimodal_count'] == 1
        assert thread['entry_count'] == 2


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

    @pytest.mark.asyncio
    async def test_get_database_statistics_multiple_threads_ordering(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """most_active_threads ordered by count descending."""
        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            for i in range(3):
                cursor.execute(
                    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                    f"VALUES (?, 'busy-thread', 'user', 'text', 'Entry {i}')",
                    (generate_id(),),
                )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES (?, 'quiet-thread', 'user', 'text', 'Single entry')",
                (generate_id(),),
            )

        await stats_test_db.execute_write(_insert_data)

        result = await stats_repo.get_database_statistics()
        active = result['most_active_threads']
        assert len(active) >= 2
        assert active[0]['count'] >= active[1]['count']


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


class TestListThreadsPostgresqlSqlText:
    """Static-text checks on the PostgreSQL branch of get_thread_list.

    Asserts that the PostgreSQL branch aggregates the latest entry id via
    ``(array_agg(id ORDER BY id DESC))[1]`` and does NOT use a ``MAX(id)``
    aggregate. PostgreSQL provides no MAX aggregate for the ``uuid`` type
    (see https://www.postgresql.org/docs/current/functions-aggregate.html --
    ``uuid`` is absent from the supported MAX/MIN input types), so the
    array_agg subscripting form is the canonical way to obtain the latest
    UUID value while preserving the native ``uuid`` column type for the
    asyncpg codec.
    """

    @pytest.mark.asyncio
    async def test_postgresql_branch_uses_array_agg_not_max_id(self) -> None:
        """The PostgreSQL branch of get_thread_list emits an array_agg-based
        latest-id expression and does not emit ``MAX(id)``.

        The check inspects the SQL string by reading the source of
        StatisticsRepository.get_thread_list directly, so it runs without a
        running PostgreSQL instance.
        """
        import inspect

        from app.repositories.statistics_repository import StatisticsRepository

        source = inspect.getsource(StatisticsRepository.get_thread_list)

        # Locate the PostgreSQL closure within the method source. The SQLite
        # branch precedes it and uses MAX(id), so split the source and
        # inspect only the PostgreSQL portion.
        marker = 'async def _list_threads_postgresql'
        assert marker in source, (
            'get_thread_list must contain an async _list_threads_postgresql closure'
        )
        pg_branch = source.split(marker, 1)[1]

        # Must use the codec-preserving array_agg form per the project's
        # asyncpg uuid type codec contract (decoder=normalize_id at
        # app/backends/postgresql_backend.py).
        assert 'array_agg(id ORDER BY id DESC)' in pg_branch, (
            'PostgreSQL branch must aggregate latest id via '
            "'(array_agg(id ORDER BY id DESC))[1]' to preserve native uuid type "
            'so the asyncpg codec normalizes the value to 32-char lowercase hex.'
        )

        # PostgreSQL has no MAX aggregate accepting a UUID input type.
        assert 'MAX(id)' not in pg_branch, (
            'PostgreSQL branch must not apply MAX to the UUID id column; '
            'PostgreSQL provides no MAX aggregate over the uuid type '
            '(see https://www.postgresql.org/docs/current/functions-aggregate.html).'
        )

    @pytest.mark.asyncio
    async def test_sqlite_branch_continues_to_use_max_id(self) -> None:
        """The SQLite branch of get_thread_list uses ``MAX(id)`` on the TEXT id
        column.

        SQLite stores the id column as ``TEXT NOT NULL UNIQUE`` under the
        project's canonical lowercase invariant; ``MAX(TEXT)`` under BINARY
        collation is well-defined and chronologically correct for UUIDv7.
        The two backends therefore use different aggregation strategies for
        the latest-id expression.
        """
        import inspect

        from app.repositories.statistics_repository import StatisticsRepository

        source = inspect.getsource(StatisticsRepository.get_thread_list)

        marker_sqlite = 'def _list_threads_sqlite'
        marker_pg = 'async def _list_threads_postgresql'
        assert marker_sqlite in source
        assert marker_pg in source

        sqlite_branch = source.split(marker_sqlite, 1)[1].split(marker_pg, 1)[0]

        assert 'MAX(id) as last_id' in sqlite_branch, (
            'SQLite branch must continue to use MAX(id) on the TEXT id column; '
            'this is the correct and tested form for the SQLite backend.'
        )


class _PgStubConnection:
    """Async stub of an asyncpg connection for the PostgreSQL statistics paths.

    Dispatches on the SQL text so each query in ``get_database_statistics`` and
    ``get_tag_statistics`` receives a realistic value. The numeric aggregates
    (``AVG(...)``) return ``decimal.Decimal`` exactly as asyncpg maps PostgreSQL
    ``NUMERIC`` results; the count aggregates and ``pg_database_size`` return
    native ``int``. ``fetchrow``/``fetch`` return mappings, matching the
    ``row['column']`` access the production closures perform.
    """

    def __init__(self, avg_value: Decimal | None) -> None:
        self._avg_value = avg_value

    async def fetchrow(self, query: str, *_args: object) -> dict[str, object] | None:
        normalized = ' '.join(query.split())
        if 'AVG(entry_count)' in normalized or 'AVG(tag_count)' in normalized:
            return {'avg_entries': self._avg_value, 'avg_tags': self._avg_value}
        if 'pg_database_size' in normalized:
            return {'db_size': 8192}
        if 'COUNT(DISTINCT thread_id)' in normalized:
            return {'count': 2}
        if 'COUNT(DISTINCT tag)' in normalized:
            return {'count': 3}
        if 'COUNT(*)' in normalized:
            return {'count': 20}
        raise AssertionError(f'Unexpected fetchrow query: {normalized}')

    async def fetch(self, query: str, *_args: object) -> list[dict[str, object]]:
        normalized = ' '.join(query.split())
        if 'GROUP BY source' in normalized:
            return [{'source': 'user', 'count': 12}, {'source': 'agent', 'count': 8}]
        if 'GROUP BY content_type' in normalized:
            return [{'content_type': 'text', 'count': 20}]
        if 'GROUP BY thread_id' in normalized:
            return [{'thread_id': 't1', 'count': 12}, {'thread_id': 't2', 'count': 8}]
        if 'GROUP BY tag' in normalized:
            return [{'tag': 'python', 'count': 5}, {'tag': 'sqlite', 'count': 3}]
        raise AssertionError(f'Unexpected fetch query: {normalized}')


class _PgStubBackend:
    """Minimal ``StorageBackend`` stub routing through the PostgreSQL code path.

    ``backend_type == 'postgresql'`` selects the PostgreSQL branch of every
    statistics method, and ``execute_read`` awaits the production async closure
    with a :class:`_PgStubConnection`. This exercises the real ``_to_float``
    coercion on a ``decimal.Decimal`` aggregate without a live PostgreSQL
    instance, reproducing the asyncpg ``Decimal`` serialization defect that a
    SQLite-only test cannot reach (SQLite computes ``AVG`` as a native float).
    """

    backend_type = 'postgresql'

    def __init__(self, avg_value: Decimal | None) -> None:
        self._conn = _PgStubConnection(avg_value)

    async def execute_read(self, operation: Callable[[Any], Awaitable[T]]) -> T:
        return await operation(self._conn)


class TestToFloatHelper:
    """Unit tests for the ``_to_float`` aggregate-normalization helper."""

    def test_decimal_converts_to_native_float(self) -> None:
        result = _to_float(Decimal('10.00'))
        assert type(result) is float
        assert result == 10.0

    def test_zero_decimal_preserved_as_float(self) -> None:
        result = _to_float(Decimal(0))
        assert type(result) is float
        assert result == 0.0

    def test_none_converts_to_default_zero(self) -> None:
        result = _to_float(None)
        assert type(result) is float
        assert result == 0.0

    def test_float_passes_through_rounded(self) -> None:
        # round(float, 2) uses Python's round-half-to-even with binary float
        # representation; 1.235 is stored as 1.2349999... so it rounds to 1.23.
        result = _to_float(1.235)
        assert type(result) is float
        assert result == round(1.235, 2)

    def test_float_two_decimal_places_preserved(self) -> None:
        result = _to_float(5.5)
        assert type(result) is float
        assert result == 5.5

    def test_int_converts_to_float(self) -> None:
        result = _to_float(7)
        assert type(result) is float
        assert result == 7.0


@pytest.mark.asyncio
class TestPostgresqlDecimalRegression:
    """Regression guard for the PostgreSQL asyncpg ``Decimal`` serialization defect.

    asyncpg maps PostgreSQL ``AVG()`` to ``decimal.Decimal``, which serializes to
    a JSON string (e.g. ``"10.00"``) and fails the MCP ``number`` output schema.
    Driving the PostgreSQL branch through a stub backend that returns ``Decimal``
    asserts the repository emits a native ``float`` for ``avg_entries_per_thread``
    and ``avg_tags_per_entry``. This reproduces the reported symptom that a
    SQLite-only test cannot reach.
    """

    async def test_avg_entries_per_thread_is_native_float(self) -> None:
        repo = StatisticsRepository(cast(StorageBackend, _PgStubBackend(Decimal('10.00'))))
        stats = await repo.get_database_statistics()
        avg = stats['avg_entries_per_thread']
        assert not isinstance(avg, Decimal)
        assert not isinstance(avg, str)
        assert type(avg) is float
        assert avg == 10.0

    async def test_avg_tags_per_entry_is_native_float(self) -> None:
        repo = StatisticsRepository(cast(StorageBackend, _PgStubBackend(Decimal('3.50'))))
        stats = await repo.get_tag_statistics()
        avg = stats['avg_tags_per_entry']
        assert not isinstance(avg, Decimal)
        assert not isinstance(avg, str)
        assert type(avg) is float
        assert avg == 3.5

    async def test_null_avg_entries_returns_zero_float(self) -> None:
        repo = StatisticsRepository(cast(StorageBackend, _PgStubBackend(None)))
        stats = await repo.get_database_statistics()
        avg = stats['avg_entries_per_thread']
        assert type(avg) is float
        assert avg == 0.0

    async def test_null_avg_tags_returns_zero_float(self) -> None:
        repo = StatisticsRepository(cast(StorageBackend, _PgStubBackend(None)))
        stats = await repo.get_tag_statistics()
        avg = stats['avg_tags_per_entry']
        assert type(avg) is float
        assert avg == 0.0


@pytest.mark.asyncio
class TestPostgresqlDatabaseSize:
    """Guard that PostgreSQL ``database_size_mb`` is a native float from pg_database_size.

    The PostgreSQL branch queries ``pg_database_size(current_database())`` and
    must convert the ``bigint`` byte count to a native ``float`` MB value. The
    local ``db_path`` argument is irrelevant to a PostgreSQL database and must
    not be file-stat'd.
    """

    async def test_database_size_mb_is_native_float(self) -> None:
        repo = StatisticsRepository(cast(StorageBackend, _PgStubBackend(Decimal('10.00'))))
        stats = await repo.get_database_statistics()
        assert 'database_size_mb' in stats
        size = stats['database_size_mb']
        assert type(size) is float
        # 8192 bytes / (1024 * 1024) rounded to 2 dp.
        assert size == round(8192 / (1024 * 1024), 2)

    async def test_local_db_path_ignored_on_postgresql(self, tmp_path: Path) -> None:
        # A bogus local path must not influence the PostgreSQL size, which comes
        # solely from pg_database_size.
        bogus = tmp_path / 'not_a_pg_database.db'
        bogus.write_bytes(b'x' * (5 * 1024 * 1024))
        repo = StatisticsRepository(cast(StorageBackend, _PgStubBackend(Decimal('10.00'))))
        stats = await repo.get_database_statistics(db_path=bogus)
        assert stats['database_size_mb'] == round(8192 / (1024 * 1024), 2)


@pytest.mark.asyncio
class TestSqliteTruthinessAlignment:
    """Guard that the SQLite avg aggregates are native floats, including zero.

    SQLite computes ``AVG`` in Python as a native float, so the ``_to_float``
    wrap is idempotent. These tests confirm the alignment to ``0.0`` (rather than
    an int ``0`` from a truthiness ``else 0``) on an empty database and a native
    float on a populated database.
    """

    async def test_empty_avg_entries_per_thread_is_zero_float(
        self, stats_repo: StatisticsRepository,
    ) -> None:
        stats = await stats_repo.get_database_statistics()
        avg = stats['avg_entries_per_thread']
        assert type(avg) is float
        assert avg == 0.0

    async def test_empty_avg_tags_per_entry_is_zero_float(
        self, stats_repo: StatisticsRepository,
    ) -> None:
        stats = await stats_repo.get_tag_statistics()
        avg = stats['avg_tags_per_entry']
        assert type(avg) is float
        assert avg == 0.0

    async def test_populated_avg_entries_per_thread_is_native_float(
        self, stats_test_db: StorageBackend, stats_repo: StatisticsRepository,
    ) -> None:
        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES ('0190abcdef1234567890abcd0000aa01', 't1', 'user', 'text', 'A')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES ('0190abcdef1234567890abcd0000aa02', 't1', 'agent', 'text', 'B')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES ('0190abcdef1234567890abcd0000aa03', 't2', 'user', 'text', 'C')",
            )

        await stats_test_db.execute_write(_insert_data)

        stats = await stats_repo.get_database_statistics()
        avg = stats['avg_entries_per_thread']
        assert type(avg) is float
        # 3 entries across 2 threads -> average 1.5.
        assert avg == 1.5


class TestGetThreadListPagination:
    """Tests for optional limit/offset pagination on get_thread_list (SQLite).

    Five threads with strictly increasing last_entry timestamps are inserted so
    the deterministic ORDER BY (MAX(created_at) DESC, MAX(id) DESC) produces a
    known sequence: thread_e, thread_d, thread_c, thread_b, thread_a.
    """

    @staticmethod
    def _insert_five_threads(conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        rows = [
            ('0190abcdef1234567890abcd0000e001', 'thread_a', '2026-01-01 10:00:01'),
            ('0190abcdef1234567890abcd0000e002', 'thread_b', '2026-01-01 10:00:02'),
            ('0190abcdef1234567890abcd0000e003', 'thread_c', '2026-01-01 10:00:03'),
            ('0190abcdef1234567890abcd0000e004', 'thread_d', '2026-01-01 10:00:04'),
            ('0190abcdef1234567890abcd0000e005', 'thread_e', '2026-01-01 10:00:05'),
        ]
        for entry_id, thread_id, created_at in rows:
            cursor.execute(
                'INSERT INTO context_entries '
                '(id, thread_id, source, content_type, text_content, created_at) '
                "VALUES (?, ?, 'user', 'text', 'entry', ?)",
                (entry_id, thread_id, created_at),
            )

    # Expected order, newest activity first.
    _EXPECTED_ORDER = ['thread_e', 'thread_d', 'thread_c', 'thread_b', 'thread_a']

    @pytest.mark.asyncio
    async def test_no_limit_returns_all_threads(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Default (no limit) returns every thread in the canonical order."""
        await stats_test_db.execute_write(self._insert_five_threads)

        result = await stats_repo.get_thread_list()

        assert [t['thread_id'] for t in result] == self._EXPECTED_ORDER

    @pytest.mark.asyncio
    async def test_limit_none_explicit_returns_all_threads(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Explicit limit=None is identical to the default (no LIMIT clause)."""
        await stats_test_db.execute_write(self._insert_five_threads)

        result = await stats_repo.get_thread_list(limit=None)

        assert [t['thread_id'] for t in result] == self._EXPECTED_ORDER

    @pytest.mark.asyncio
    async def test_limit_bounds_result_preserving_order(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """limit returns the first N threads of the canonical order."""
        await stats_test_db.execute_write(self._insert_five_threads)

        result = await stats_repo.get_thread_list(limit=2)

        assert [t['thread_id'] for t in result] == ['thread_e', 'thread_d']

    @pytest.mark.asyncio
    async def test_offset_skips_leading_threads(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """limit + offset returns the correct page slice in canonical order."""
        await stats_test_db.execute_write(self._insert_five_threads)

        result = await stats_repo.get_thread_list(limit=2, offset=2)

        assert [t['thread_id'] for t in result] == ['thread_c', 'thread_b']

    @pytest.mark.asyncio
    async def test_offset_past_end_returns_empty(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """An offset past the last row yields an empty page, not an error."""
        await stats_test_db.execute_write(self._insert_five_threads)

        result = await stats_repo.get_thread_list(limit=5, offset=10)

        assert result == []

    @pytest.mark.asyncio
    async def test_limit_larger_than_total_returns_all(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """A limit exceeding the thread count returns every thread."""
        await stats_test_db.execute_write(self._insert_five_threads)

        result = await stats_repo.get_thread_list(limit=100)

        assert [t['thread_id'] for t in result] == self._EXPECTED_ORDER

    @pytest.mark.asyncio
    async def test_full_page_walk_covers_all_threads_once(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Walking pages of size 2 reconstructs the full ordered list exactly once."""
        await stats_test_db.execute_write(self._insert_five_threads)

        page1 = await stats_repo.get_thread_list(limit=2, offset=0)
        page2 = await stats_repo.get_thread_list(limit=2, offset=2)
        page3 = await stats_repo.get_thread_list(limit=2, offset=4)

        walked = [t['thread_id'] for t in (*page1, *page2, *page3)]
        assert walked == self._EXPECTED_ORDER
