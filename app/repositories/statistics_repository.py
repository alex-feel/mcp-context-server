"""
Statistics repository for analytics and reporting.

This module handles all database operations related to statistics,
thread information, and database metrics.
"""

import sqlite3
from pathlib import Path
from typing import Any
from typing import cast

from app.db_manager import DatabaseConnectionManager
from app.repositories.base import BaseRepository
from app.types import ThreadInfoDict


class StatisticsRepository(BaseRepository):
    """Repository for statistics and analytics operations.

    Handles retrieval of thread information, database statistics,
    and usage metrics.
    """

    def __init__(self, db_manager: DatabaseConnectionManager) -> None:
        """Initialize statistics repository.

        Args:
            db_manager: Database connection manager for executing operations
        """
        super().__init__(db_manager)

    async def get_thread_list(self) -> list[ThreadInfoDict]:
        """Get list of all threads with statistics.

        Returns:
            List of thread information dictionaries
        """
        def _list_threads(conn: sqlite3.Connection) -> list[ThreadInfoDict]:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT
                    thread_id,
                    COUNT(*) as entry_count,
                    COUNT(DISTINCT source) as source_types,
                    SUM(CASE WHEN content_type = 'multimodal' THEN 1 ELSE 0 END) as multimodal_count,
                    MIN(created_at) as first_entry,
                    MAX(created_at) as last_entry,
                    MAX(id) as last_id
                FROM context_entries
                GROUP BY thread_id
                ORDER BY MAX(created_at) DESC, MAX(id) DESC
            ''')

            threads: list[ThreadInfoDict] = []
            for row in cursor.fetchall():
                thread = cast(ThreadInfoDict, dict(row))
                threads.append(thread)

            return threads

        return await self.db_manager.execute_read(_list_threads)

    async def get_database_statistics(self, db_path: Path | None = None) -> dict[str, Any]:
        """Get comprehensive database statistics.

        Args:
            db_path: Optional path to database file for size calculation

        Returns:
            Dictionary containing various database statistics
        """
        def _get_stats(conn: sqlite3.Connection) -> dict[str, Any]:
            cursor = conn.cursor()
            stats: dict[str, Any] = {}

            # Total entries
            cursor.execute('SELECT COUNT(*) as count FROM context_entries')
            stats['total_entries'] = cursor.fetchone()['count']

            # By source
            cursor.execute('''
                SELECT source, COUNT(*) as count
                FROM context_entries
                GROUP BY source
            ''')
            by_source: dict[str, int] = {}
            for row in cursor.fetchall():
                by_source[row['source']] = row['count']
            stats['by_source'] = by_source

            # By content type
            cursor.execute('''
                SELECT content_type, COUNT(*) as count
                FROM context_entries
                GROUP BY content_type
            ''')
            by_content_type: dict[str, int] = {}
            for row in cursor.fetchall():
                by_content_type[row['content_type']] = row['count']
            stats['by_content_type'] = by_content_type

            # Total images
            cursor.execute('SELECT COUNT(*) as count FROM image_attachments')
            stats['total_images'] = cursor.fetchone()['count']

            # Total unique tags
            cursor.execute('SELECT COUNT(DISTINCT tag) as count FROM tags')
            stats['unique_tags'] = cursor.fetchone()['count']

            # Thread count
            cursor.execute('SELECT COUNT(DISTINCT thread_id) as count FROM context_entries')
            stats['total_threads'] = cursor.fetchone()['count']

            # Average entries per thread
            cursor.execute('''
                SELECT AVG(entry_count) as avg_entries
                FROM (
                    SELECT thread_id, COUNT(*) as entry_count
                    FROM context_entries
                    GROUP BY thread_id
                )
            ''')
            result = cursor.fetchone()
            stats['avg_entries_per_thread'] = round(result['avg_entries'], 2) if result['avg_entries'] else 0

            # Most active threads
            cursor.execute('''
                SELECT thread_id, COUNT(*) as count
                FROM context_entries
                GROUP BY thread_id
                ORDER BY count DESC
                LIMIT 5
            ''')
            most_active: list[dict[str, Any]] = [
                {'thread_id': row['thread_id'], 'count': row['count']}
                for row in cursor.fetchall()
            ]
            stats['most_active_threads'] = most_active

            # Most used tags
            cursor.execute('''
                SELECT tag, COUNT(*) as count
                FROM tags
                GROUP BY tag
                ORDER BY count DESC
                LIMIT 10
            ''')
            top_tags: list[dict[str, Any]] = [
                {'tag': row['tag'], 'count': row['count']}
                for row in cursor.fetchall()
            ]
            stats['top_tags'] = top_tags

            return stats

        stats = await self.db_manager.execute_read(_get_stats)

        # Add database file size if path provided
        if db_path and db_path.exists():
            size_in_bytes: int = db_path.stat().st_size
            size_in_mb: float = size_in_bytes / (1024 * 1024)
            stats['database_size_mb'] = round(size_in_mb, 2)

        return stats

    async def get_thread_statistics(self, thread_id: str) -> dict[str, Any]:
        """Get statistics for a specific thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Dictionary containing thread-specific statistics
        """
        def _get_thread_stats(conn: sqlite3.Connection) -> dict[str, Any]:
            cursor = conn.cursor()
            stats: dict[str, Any] = {'thread_id': thread_id}

            # Basic counts
            cursor.execute('''
                SELECT
                    COUNT(*) as total_entries,
                    COUNT(DISTINCT source) as source_types,
                    SUM(CASE WHEN content_type = 'text' THEN 1 ELSE 0 END) as text_count,
                    SUM(CASE WHEN content_type = 'multimodal' THEN 1 ELSE 0 END) as multimodal_count,
                    MIN(created_at) as first_entry,
                    MAX(created_at) as last_entry
                FROM context_entries
                WHERE thread_id = ?
            ''', (thread_id,))

            row = cursor.fetchone()
            if row:
                stats.update(dict(row))

            # Entry breakdown by source
            cursor.execute('''
                SELECT source, COUNT(*) as count
                FROM context_entries
                WHERE thread_id = ?
                GROUP BY source
            ''', (thread_id,))

            by_source: dict[str, int] = {}
            for row in cursor.fetchall():
                by_source[row['source']] = row['count']
            stats['by_source'] = by_source

            # Tags used in this thread
            cursor.execute('''
                SELECT DISTINCT t.tag
                FROM tags t
                JOIN context_entries c ON t.context_entry_id = c.id
                WHERE c.thread_id = ?
                ORDER BY t.tag
            ''', (thread_id,))

            tags: list[str] = [row['tag'] for row in cursor.fetchall()]
            stats['tags'] = tags

            # Image count
            cursor.execute('''
                SELECT COUNT(*) as count
                FROM image_attachments i
                JOIN context_entries c ON i.context_entry_id = c.id
                WHERE c.thread_id = ?
            ''', (thread_id,))

            stats['image_count'] = cursor.fetchone()['count']

            return stats

        return await self.db_manager.execute_read(_get_thread_stats)

    async def get_tag_statistics(self) -> dict[str, Any]:
        """Get comprehensive tag usage statistics.

        Returns:
            Dictionary containing tag-related statistics
        """
        def _get_tag_stats(conn: sqlite3.Connection) -> dict[str, Any]:
            cursor = conn.cursor()
            stats: dict[str, Any] = {}

            # Total tags
            cursor.execute('SELECT COUNT(*) as count FROM tags')
            stats['total_tag_uses'] = cursor.fetchone()['count']

            # Unique tags
            cursor.execute('SELECT COUNT(DISTINCT tag) as count FROM tags')
            stats['unique_tags'] = cursor.fetchone()['count']

            # Tag frequency distribution
            cursor.execute('''
                SELECT tag, COUNT(*) as count
                FROM tags
                GROUP BY tag
                ORDER BY count DESC
            ''')

            all_tags: list[dict[str, Any]] = [
                {'tag': row['tag'], 'count': row['count']}
                for row in cursor.fetchall()
            ]

            stats['all_tags'] = all_tags
            stats['top_10_tags'] = all_tags[:10] if all_tags else []

            # Average tags per entry
            cursor.execute('''
                SELECT AVG(tag_count) as avg_tags
                FROM (
                    SELECT context_entry_id, COUNT(*) as tag_count
                    FROM tags
                    GROUP BY context_entry_id
                )
            ''')
            result = cursor.fetchone()
            stats['avg_tags_per_entry'] = round(result['avg_tags'], 2) if result['avg_tags'] else 0

            return stats

        return await self.db_manager.execute_read(_get_tag_stats)
