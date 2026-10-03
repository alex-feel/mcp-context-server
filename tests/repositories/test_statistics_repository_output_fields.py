"""Tests for the fields and formats returned by get_database_statistics."""

import sqlite3

import pytest

from app.backends.base import StorageBackend
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.statistics_repository import StatisticsRepository
from tests.helpers import LOCAL_SCOPE


class TestStatisticsBackendField:
    """Test that backend field is included in statistics output."""

    @pytest.mark.asyncio
    async def test_get_database_statistics_includes_backend(
        self,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Test get_database_statistics includes backend identifier."""
        result = await stats_repo.get_database_statistics(scope=LOCAL_SCOPE)

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
        result = await stats_repo.get_database_statistics(scope=LOCAL_SCOPE)

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
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='active-thread',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
            )

        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='less-active-thread',
            source='user',
            content_type='text',
            text_content='Single entry',
        )

        result = await stats_repo.get_database_statistics(scope=LOCAL_SCOPE)

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
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='tags-thread',
            source='user',
            content_type='text',
            text_content='Tagged entry',
        )
        await repos.tags.store_tags(ctx_id, ['python', 'testing'])

        result = await stats_repo.get_database_statistics(scope=LOCAL_SCOPE)

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
                    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                    f"VALUES (?, 'busy-thread', 'user', 'text', 'Entry {i}', 'local')",
                    (generate_id(),),
                )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES (?, 'quiet-thread', 'user', 'text', 'Single entry', 'local')",
                (generate_id(),),
            )

        await stats_test_db.execute_write(_insert_data)

        result = await stats_repo.get_database_statistics(scope=LOCAL_SCOPE)
        active = result['most_active_threads']
        assert len(active) >= 2
        assert active[0]['count'] >= active[1]['count']
