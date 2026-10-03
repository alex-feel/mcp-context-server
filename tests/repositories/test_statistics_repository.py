"""Tests for StatisticsRepository across its five read methods on SQLite, and through RepositoryContainer."""

import json
import sqlite3
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends.base import StorageBackend
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.statistics_repository import StatisticsRepository
from tests.helpers import LOCAL_SCOPE


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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000001', 'thread1', 'user', 'text', 'Test 1', 'local')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000002', 'thread1', 'agent', 'text', 'Test 2', 'local')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000003', 'thread2', 'user', 'multimodal', 'Test 3', 'local')",
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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, created_at, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000001', 'thread_a', 'user', 'text', 'A1', '2026-01-01 10:00:00', "
                "'local')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, created_at, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000005', 'thread_a', 'agent', 'text', 'A2', '2026-01-01 10:00:01', "
                "'local')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, created_at, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000003', 'thread_a', 'user', 'text', 'A3', '2026-01-01 10:00:02', "
                "'local')",
            )
            # thread_b: two ids, ordered so that the lex-max id is NOT the most recently inserted
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, created_at, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000099', 'thread_b', 'user', 'text', 'B1', '2026-01-01 10:00:10', "
                "'local')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, created_at, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000010', 'thread_b', 'agent', 'text', 'B2', '2026-01-01 10:00:11', "
                "'local')",
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
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='thread1',
            source='user',
            content_type='text',
            text_content='Test 1',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='thread1',
            source='agent',
            content_type='text',
            text_content='Test 2',
        )
        ctx_id3, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
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
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='thread1',
            source='user',
            content_type='text',
            text_content='Test 1',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000004', 'thread1', 'user', 'text', 'Test 1', 'local')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000005', 'thread1', 'agent', 'text', 'Test 2', 'local')",
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000006', 'thread2', 'user', 'text', 'Test 3', 'local')",
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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000007', 'thread1', 'user', 'text', 'Test', 'local')",
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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES ('0190abcdef1234567890abcd00000008', 'thread1', 'user', 'text', 'Test', 'local')",
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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, summary, owner_id) '
                "VALUES (?, 't1', 'user', 'text', 'Content 1', 'Summary 1', 'local')",
                (generate_id(),),
            )
            # Entry with NULL summary
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES (?, 't1', 'agent', 'text', 'Content 2', 'local')",
                (generate_id(),),
            )
            # Entry with valid summary
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, summary, owner_id) '
                "VALUES (?, 't2', 'user', 'text', 'Content 3', 'Summary 3', 'local')",
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

        The SQL uses WHERE summary IS NOT NULL AND summary != '' so an entry
        whose stored summary is an empty string does not count as summarized.
        """

        def _insert_data(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            # Entry with valid summary
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, summary, owner_id) '
                "VALUES (?, 't1', 'user', 'text', 'Content 1', 'Valid summary', 'local')",
                (generate_id(),),
            )
            # Entry with empty string summary (edge case)
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, summary, owner_id) '
                "VALUES (?, 't1', 'agent', 'text', 'Content 2', '', 'local')",
                (generate_id(),),
            )
            # Entry with NULL summary
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES (?, 't2', 'user', 'text', 'Content 3', 'local')",
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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES (?, 'ct-thread', 'user', 'text', 'Text entry', 'local')",
                (generate_id(),),
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES (?, 'ct-thread', 'user', 'multimodal', 'Multimodal entry', 'local')",
                (generate_id(),),
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES (?, 'ct-thread', 'agent', 'text', 'Another text entry', 'local')",
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
            scope=LOCAL_SCOPE,
            visibility='private',
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
        # Store some context entries
        context_id1, _ = await repo_container.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='workflow_thread',
            source='user',
            content_type='text',
            text_content='First entry',
            metadata=json.dumps({'priority': 1}),
        )
        assert context_id1 is not None

        context_id2, _ = await repo_container.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
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
