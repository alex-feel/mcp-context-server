"""
Statistics repository for analytics and reporting.

This module handles all database operations related to statistics,
thread information, and database metrics.
"""


import logging
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from anyio import Path as AsyncPath

from app.backends.base import StorageBackend
from app.ids import normalize_id
from app.repositories.base import BaseRepository
from app.types import ThreadInfoDict

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


def _to_float(value: float | Decimal | None, default: float = 0.0) -> float:
    """Coerce a database aggregate value to a native rounded float.

    The asyncpg driver maps PostgreSQL ``AVG()`` and other numeric aggregates to
    ``decimal.Decimal``. ``Decimal`` serializes to a JSON string, which fails the
    MCP output schema that expects a native ``number`` for fields such as
    ``avg_entries_per_thread``. This helper guarantees a native ``float`` for any
    numeric aggregate at response-assembly time, on every backend.

    A ``None`` value (empty result set) yields ``default``. A legitimate zero is
    preserved as ``0.0`` because the guard tests against ``None`` explicitly
    rather than truthiness.

    Args:
        value: Aggregate value from a query row (int, float, Decimal, or None).
        default: Value returned when ``value`` is ``None``.

    Returns:
        Native ``float`` rounded to two decimal places, or ``default``.
    """
    if value is None:
        return default
    return round(float(value), 2)


class StatisticsRepository(BaseRepository):
    """Repository for statistics and analytics operations.

    Handles retrieval of thread information, database statistics,
    and usage metrics.
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize statistics repository.

        Args:
            backend: Storage backend for executing database operations
        """
        super().__init__(backend)

    async def get_thread_list(self, limit: int | None = None, offset: int = 0) -> list[ThreadInfoDict]:
        """Get list of threads with statistics, optionally paginated.

        When ``limit`` is None (the default) ALL threads are returned and no
        LIMIT/OFFSET clause is emitted, preserving the historical unbounded
        behavior. When ``limit`` is provided, the result is bounded to ``limit``
        rows starting at ``offset``, applied AFTER the ORDER BY so pagination
        walks the most-recently-active threads first.

        Args:
            limit: Maximum number of threads to return. None returns all threads.
            offset: Number of leading threads to skip (only applied when ``limit``
                is provided).

        Returns:
            List of thread information dictionaries for the requested page.
        """
        # Bind LIMIT/OFFSET only when a limit is requested. The placeholders are
        # positions 1 and 2 because the GROUP BY listing queries carry no other
        # bound parameters; _placeholder yields '?' on SQLite and '$1'/'$2' on
        # PostgreSQL.
        pagination_clause = ''
        params: tuple[int, ...] = ()
        if limit is not None:
            pagination_clause = f'\n                    LIMIT {self._placeholder(1)} OFFSET {self._placeholder(2)}'
            params = (limit, offset)

        if self.backend.backend_type == 'sqlite':

            def _list_threads_sqlite(conn: sqlite3.Connection) -> list[ThreadInfoDict]:
                cursor = conn.cursor()
                cursor.execute(f'''
                    SELECT
                        thread_id,
                        COUNT(*) as entry_count,
                        COUNT(DISTINCT source) as source_types,
                        SUM(CASE WHEN content_type = 'multimodal' THEN 1 ELSE 0 END) as multimodal_count,
                        strftime('%Y-%m-%dT%H:%M:%SZ', MIN(created_at)) as first_entry,
                        strftime('%Y-%m-%dT%H:%M:%SZ', MAX(created_at)) as last_entry,
                        MAX(id) as last_id
                    FROM context_entries
                    GROUP BY thread_id
                    ORDER BY MAX(created_at) DESC, MAX(id) DESC{pagination_clause}
                ''', params)

                threads: list[ThreadInfoDict] = []
                for row in cursor.fetchall():
                    thread = cast(ThreadInfoDict, dict(row))
                    threads.append(thread)

                return threads

            return await self.backend.execute_read(_list_threads_sqlite)

        # postgresql

        async def _list_threads_postgresql(conn: 'asyncpg.Connection') -> list[ThreadInfoDict]:
            rows = await conn.fetch(f'''
                    SELECT
                        thread_id,
                        COUNT(*) as entry_count,
                        COUNT(DISTINCT source) as source_types,
                        SUM(CASE WHEN content_type = 'multimodal' THEN 1 ELSE 0 END) as multimodal_count,
                        to_char(MIN(created_at) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as first_entry,
                        to_char(MAX(created_at) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as last_entry,
                        (array_agg(id ORDER BY id DESC))[1] as last_id
                    FROM context_entries
                    GROUP BY thread_id
                    ORDER BY MAX(created_at) DESC, (array_agg(id ORDER BY id DESC))[1] DESC{pagination_clause}
                ''', *params)

            threads: list[ThreadInfoDict] = []
            for row in rows:
                d = dict(row)
                # asyncpg returns last_id as a native (hyphenated 36-char) UUID; normalize
                # to the canonical 32-char hyphen-free lowercase hex the SQLite branch
                # already emits, so list_threads' last_id is identical and contract-valid
                # (str id format) on both backends.
                if d.get('last_id') is not None:
                    d['last_id'] = normalize_id(str(d['last_id']))
                threads.append(cast(ThreadInfoDict, d))

            return threads

        return await self.backend.execute_read(_list_threads_postgresql)

    async def get_database_statistics(self, db_path: Path | None = None) -> dict[str, Any]:
        """Get comprehensive database statistics.

        Args:
            db_path: Optional path to database file for size calculation

        Returns:
            Dictionary containing various database statistics
        """
        if self.backend.backend_type == 'sqlite':

            def _get_stats_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
                cursor = conn.cursor()
                stats: dict[str, Any] = {}

                cursor.execute('SELECT COUNT(*) as count FROM context_entries')
                stats['total_entries'] = cursor.fetchone()['count']

                cursor.execute('SELECT source, COUNT(*) as count FROM context_entries GROUP BY source')
                by_source: dict[str, int] = {}
                for row in cursor.fetchall():
                    by_source[row['source']] = row['count']
                stats['by_source'] = by_source

                cursor.execute('SELECT content_type, COUNT(*) as count FROM context_entries GROUP BY content_type')
                by_content_type: dict[str, int] = {}
                for row in cursor.fetchall():
                    by_content_type[row['content_type']] = row['count']
                stats['by_content_type'] = by_content_type

                cursor.execute('SELECT COUNT(*) as count FROM image_attachments')
                stats['total_images'] = cursor.fetchone()['count']

                cursor.execute('SELECT COUNT(DISTINCT tag) as count FROM tags')
                stats['unique_tags'] = cursor.fetchone()['count']

                cursor.execute('SELECT COUNT(DISTINCT thread_id) as count FROM context_entries')
                stats['total_threads'] = cursor.fetchone()['count']

                cursor.execute('''
                    SELECT AVG(entry_count) as avg_entries
                    FROM (SELECT thread_id, COUNT(*) as entry_count FROM context_entries GROUP BY thread_id)
                ''')
                result = cursor.fetchone()
                stats['avg_entries_per_thread'] = _to_float(result['avg_entries'])

                cursor.execute('''
                    SELECT thread_id, COUNT(*) as count FROM context_entries
                    GROUP BY thread_id ORDER BY count DESC LIMIT 5
                ''')
                most_active: list[dict[str, Any]] = [
                    {'thread_id': row['thread_id'], 'count': row['count']} for row in cursor.fetchall()
                ]
                stats['most_active_threads'] = most_active

                cursor.execute('SELECT tag, COUNT(*) as count FROM tags GROUP BY tag ORDER BY count DESC LIMIT 10')
                top_tags: list[dict[str, Any]] = [{'tag': row['tag'], 'count': row['count']} for row in cursor.fetchall()]
                stats['top_tags'] = top_tags

                stats['backend'] = 'sqlite'
                return stats

            stats = await self.backend.execute_read(_get_stats_sqlite)
        else:  # postgresql

            async def _get_stats_postgresql(conn: 'asyncpg.Connection') -> dict[str, Any]:
                stats: dict[str, Any] = {}

                row = await conn.fetchrow('SELECT COUNT(*) as count FROM context_entries')
                stats['total_entries'] = row['count'] if row else 0

                rows = await conn.fetch('SELECT source, COUNT(*) as count FROM context_entries GROUP BY source')
                by_source: dict[str, int] = {}
                for row in rows:
                    by_source[row['source']] = row['count']
                stats['by_source'] = by_source

                rows = await conn.fetch('SELECT content_type, COUNT(*) as count FROM context_entries GROUP BY content_type')
                by_content_type: dict[str, int] = {}
                for row in rows:
                    by_content_type[row['content_type']] = row['count']
                stats['by_content_type'] = by_content_type

                row = await conn.fetchrow('SELECT COUNT(*) as count FROM image_attachments')
                stats['total_images'] = row['count'] if row else 0

                row = await conn.fetchrow('SELECT COUNT(DISTINCT tag) as count FROM tags')
                stats['unique_tags'] = row['count'] if row else 0

                row = await conn.fetchrow('SELECT COUNT(DISTINCT thread_id) as count FROM context_entries')
                stats['total_threads'] = row['count'] if row else 0

                row = await conn.fetchrow('''
                    SELECT AVG(entry_count) as avg_entries
                    FROM (SELECT thread_id, COUNT(*) as entry_count FROM context_entries GROUP BY thread_id) sub
                ''')
                stats['avg_entries_per_thread'] = _to_float(row['avg_entries'] if row else None)

                rows = await conn.fetch('''
                    SELECT thread_id, COUNT(*) as count FROM context_entries
                    GROUP BY thread_id ORDER BY count DESC LIMIT 5
                ''')
                most_active: list[dict[str, Any]] = [{'thread_id': row['thread_id'], 'count': row['count']} for row in rows]
                stats['most_active_threads'] = most_active

                rows = await conn.fetch('SELECT tag, COUNT(*) as count FROM tags GROUP BY tag ORDER BY count DESC LIMIT 10')
                top_tags: list[dict[str, Any]] = [{'tag': row['tag'], 'count': row['count']} for row in rows]
                stats['top_tags'] = top_tags

                stats['backend'] = 'postgresql'
                return stats

            stats = await self.backend.execute_read(_get_stats_postgresql)

        backend_type = self.backend.backend_type
        if backend_type == 'sqlite':
            # SQLite size is the on-disk size of the database file. This excludes
            # the -wal/-shm sidecars, so the figure can transiently under-report
            # under WAL mode. An in-memory or missing-file database leaves the
            # key absent (tolerated by the NotRequired schema).
            if db_path:
                async_path = AsyncPath(db_path)
                if await async_path.exists():
                    stat_result = await async_path.stat()
                    size_in_bytes: int = stat_result.st_size
                    size_in_mb: float = size_in_bytes / (1024 * 1024)
                    stats['database_size_mb'] = round(size_in_mb, 2)
        elif backend_type == 'postgresql':
            # PostgreSQL size is the whole database, queried server-side. The
            # local db_path is irrelevant to a remote PostgreSQL database, so it
            # is never file-stat'd here.
            async def _get_db_size_postgresql(conn: 'asyncpg.Connection') -> int | None:
                row = await conn.fetchrow('SELECT pg_database_size(current_database()) AS db_size')
                return row['db_size'] if row else None

            size_bytes = await self.backend.execute_read(_get_db_size_postgresql)
            if size_bytes is not None:
                stats['database_size_mb'] = round(float(size_bytes) / (1024 * 1024), 2)
        else:
            logger.warning('Unknown backend type %r; database_size_mb omitted', backend_type)

        return stats

    async def get_thread_statistics(self, thread_id: str) -> dict[str, Any]:
        """Get statistics for a specific thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Dictionary containing thread-specific statistics
        """
        if self.backend.backend_type == 'sqlite':

            def _get_thread_stats_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
                cursor = conn.cursor()
                stats: dict[str, Any] = {'thread_id': thread_id}

                query1 = f'''
                    SELECT
                        COUNT(*) as total_entries,
                        COUNT(DISTINCT source) as source_types,
                        SUM(CASE WHEN content_type = 'text' THEN 1 ELSE 0 END) as text_count,
                        SUM(CASE WHEN content_type = 'multimodal' THEN 1 ELSE 0 END) as multimodal_count,
                        strftime('%Y-%m-%dT%H:%M:%SZ', MIN(created_at)) as first_entry,
                        strftime('%Y-%m-%dT%H:%M:%SZ', MAX(created_at)) as last_entry
                    FROM context_entries
                    WHERE thread_id = {self._placeholder(1)}
                '''
                cursor.execute(query1, (thread_id,))
                row = cursor.fetchone()
                if row:
                    stats.update(dict(row))

                query2 = f'''
                    SELECT source, COUNT(*) as count
                    FROM context_entries
                    WHERE thread_id = {self._placeholder(1)}
                    GROUP BY source
                '''
                cursor.execute(query2, (thread_id,))
                by_source: dict[str, int] = {}
                for row in cursor.fetchall():
                    by_source[row['source']] = row['count']
                stats['by_source'] = by_source

                query3 = f'''
                    SELECT DISTINCT t.tag
                    FROM tags t
                    JOIN context_entries c ON t.context_entry_id = c.id
                    WHERE c.thread_id = {self._placeholder(1)}
                    ORDER BY t.tag
                '''
                cursor.execute(query3, (thread_id,))
                tags: list[str] = [row['tag'] for row in cursor.fetchall()]
                stats['tags'] = tags

                query4 = f'''
                    SELECT COUNT(*) as count
                    FROM image_attachments i
                    JOIN context_entries c ON i.context_entry_id = c.id
                    WHERE c.thread_id = {self._placeholder(1)}
                '''
                cursor.execute(query4, (thread_id,))
                stats['image_count'] = cursor.fetchone()['count']

                return stats

            return await self.backend.execute_read(_get_thread_stats_sqlite)

        # postgresql

        async def _get_thread_stats_postgresql(conn: 'asyncpg.Connection') -> dict[str, Any]:
            stats: dict[str, Any] = {'thread_id': thread_id}

            query1 = f'''
                    SELECT
                        COUNT(*) as total_entries,
                        COUNT(DISTINCT source) as source_types,
                        SUM(CASE WHEN content_type = 'text' THEN 1 ELSE 0 END) as text_count,
                        SUM(CASE WHEN content_type = 'multimodal' THEN 1 ELSE 0 END) as multimodal_count,
                        to_char(MIN(created_at) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as first_entry,
                        to_char(MAX(created_at) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as last_entry
                    FROM context_entries
                    WHERE thread_id = {self._placeholder(1)}
                '''
            row = await conn.fetchrow(query1, thread_id)
            if row:
                stats.update(dict(row))

            query2 = f'''
                    SELECT source, COUNT(*) as count
                    FROM context_entries
                    WHERE thread_id = {self._placeholder(1)}
                    GROUP BY source
                '''
            rows = await conn.fetch(query2, thread_id)
            by_source: dict[str, int] = {}
            for row in rows:
                by_source[row['source']] = row['count']
            stats['by_source'] = by_source

            query3 = f'''
                    SELECT DISTINCT t.tag
                    FROM tags t
                    JOIN context_entries c ON t.context_entry_id = c.id
                    WHERE c.thread_id = {self._placeholder(1)}
                    ORDER BY t.tag
                '''
            rows = await conn.fetch(query3, thread_id)
            tags: list[str] = [row['tag'] for row in rows]
            stats['tags'] = tags

            query4 = f'''
                    SELECT COUNT(*) as count
                    FROM image_attachments i
                    JOIN context_entries c ON i.context_entry_id = c.id
                    WHERE c.thread_id = {self._placeholder(1)}
                '''
            row = await conn.fetchrow(query4, thread_id)
            stats['image_count'] = row['count'] if row else 0

            return stats

        return await self.backend.execute_read(_get_thread_stats_postgresql)

    async def get_tag_statistics(self) -> dict[str, Any]:
        """Get comprehensive tag usage statistics.

        Returns:
            Dictionary containing tag-related statistics
        """
        if self.backend.backend_type == 'sqlite':

            def _get_tag_stats_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
                cursor = conn.cursor()
                stats: dict[str, Any] = {}

                cursor.execute('SELECT COUNT(*) as count FROM tags')
                stats['total_tag_uses'] = cursor.fetchone()['count']

                cursor.execute('SELECT COUNT(DISTINCT tag) as count FROM tags')
                stats['unique_tags'] = cursor.fetchone()['count']

                cursor.execute('SELECT tag, COUNT(*) as count FROM tags GROUP BY tag ORDER BY count DESC')
                all_tags: list[dict[str, Any]] = [{'tag': row['tag'], 'count': row['count']} for row in cursor.fetchall()]
                stats['all_tags'] = all_tags
                stats['top_10_tags'] = all_tags[:10] if all_tags else []

                cursor.execute('''
                    SELECT AVG(tag_count) as avg_tags
                    FROM (SELECT context_entry_id, COUNT(*) as tag_count FROM tags GROUP BY context_entry_id)
                ''')
                result = cursor.fetchone()
                stats['avg_tags_per_entry'] = _to_float(result['avg_tags'])

                return stats

            return await self.backend.execute_read(_get_tag_stats_sqlite)

        # postgresql

        async def _get_tag_stats_postgresql(conn: 'asyncpg.Connection') -> dict[str, Any]:
            stats: dict[str, Any] = {}

            row = await conn.fetchrow('SELECT COUNT(*) as count FROM tags')
            stats['total_tag_uses'] = row['count'] if row else 0

            row = await conn.fetchrow('SELECT COUNT(DISTINCT tag) as count FROM tags')
            stats['unique_tags'] = row['count'] if row else 0

            rows = await conn.fetch('SELECT tag, COUNT(*) as count FROM tags GROUP BY tag ORDER BY count DESC')
            all_tags: list[dict[str, Any]] = [{'tag': row['tag'], 'count': row['count']} for row in rows]
            stats['all_tags'] = all_tags
            stats['top_10_tags'] = all_tags[:10] if all_tags else []

            row = await conn.fetchrow('''
                    SELECT AVG(tag_count) as avg_tags
                    FROM (SELECT context_entry_id, COUNT(*) as tag_count FROM tags GROUP BY context_entry_id) sub
                ''')
            stats['avg_tags_per_entry'] = _to_float(row['avg_tags'] if row else None)

            return stats

        return await self.backend.execute_read(_get_tag_stats_postgresql)

    async def get_summary_statistics(self) -> dict[str, Any]:
        """Get summary generation statistics.

        Returns:
            Dictionary with summary_count, total_entries, and coverage_percentage
        """
        if self.backend.backend_type == 'sqlite':

            def _get_summary_stats_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
                cursor = conn.cursor()

                cursor.execute('SELECT COUNT(*) as count FROM context_entries')
                total_entries = cursor.fetchone()['count']

                cursor.execute(
                    "SELECT COUNT(*) as count FROM context_entries WHERE summary IS NOT NULL AND summary != ''",
                )
                summary_count = cursor.fetchone()['count']

                coverage_percentage = (
                    round(summary_count / total_entries * 100, 2)
                    if total_entries > 0
                    else 0.0
                )

                return {
                    'summary_count': summary_count,
                    'total_entries': total_entries,
                    'coverage_percentage': coverage_percentage,
                }

            return await self.backend.execute_read(_get_summary_stats_sqlite)

        # postgresql

        async def _get_summary_stats_postgresql(conn: 'asyncpg.Connection') -> dict[str, Any]:
            total_entries = await conn.fetchval('SELECT COUNT(*) FROM context_entries')

            summary_count = await conn.fetchval(
                "SELECT COUNT(*) FROM context_entries WHERE summary IS NOT NULL AND summary != ''",
            )

            coverage_percentage = (
                round(summary_count / total_entries * 100, 2)
                if total_entries > 0
                else 0.0
            )

            return {
                'summary_count': summary_count,
                'total_entries': total_entries,
                'coverage_percentage': coverage_percentage,
            }

        return await self.backend.execute_read(_get_summary_stats_postgresql)
