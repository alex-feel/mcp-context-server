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

from app.access_scope import AccessMode
from app.access_scope import Scope
from app.access_scope import build_access_predicate
from app.backends.base import StorageBackend
from app.ids import normalize_id
from app.repositories._collation import byte_ordered_text
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

    async def get_thread_list(
        self, limit: int | None = None, offset: int = 0, *, scope: Scope,
    ) -> list[ThreadInfoDict]:
        """Get the threads holding an entry the scope may read, with statistics, optionally paginated.

        The READ predicate filters rows before GROUP BY, so a thread whose entries the
        scope cannot read is absent, and every figure -- entry count, source count,
        multimodal count, first and last creation time and last id -- describes the
        readable rows alone. Threads are ordered by their latest readable activity.

        When ``limit`` is None (the default) every thread is returned and no
        LIMIT/OFFSET clause is emitted. When ``limit`` is provided, the result is
        bounded to ``limit`` rows starting at ``offset``, applied AFTER the ORDER BY
        so pagination walks the most-recently-active threads first.

        Args:
            limit: Maximum number of threads to return. None returns all threads.
            offset: Number of leading threads to skip (only applied when ``limit``
                is provided).
            scope: The caller's scope.

        Returns:
            List of thread information dictionaries for the requested page.
        """
        predicate = build_access_predicate(
            scope, mode=AccessMode.READ, backend_type=self.backend.backend_type, outer='context_entries',
        )
        # LIMIT/OFFSET bind after the predicate, so their placeholders follow its binds.
        pagination_clause = ''
        params: list[Any] = list(predicate.params)
        if limit is not None:
            first = predicate.bind_count + 1
            pagination_clause = f'\n                    LIMIT {self._placeholder(first)} OFFSET {self._placeholder(first + 1)}'
            params.extend((limit, offset))
        entry_filter = predicate.where_clause()

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
                    FROM context_entries{entry_filter}
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
                    FROM context_entries{entry_filter}
                    GROUP BY thread_id
                    ORDER BY MAX(created_at) DESC, (array_agg(id ORDER BY id DESC))[1] DESC{pagination_clause}
                ''', *params)

            threads: list[ThreadInfoDict] = []
            for row in rows:
                d = dict(row)
                # The pool's uuid->str codec (registered in init_pool_connection)
                # already decodes uuid columns -- including array elements, so
                # this array_agg pick too -- to the canonical 32-char lowercase
                # hex the SQLite branch emits. normalize_id(str(...)) is
                # idempotent defense-in-depth for codec-less connections,
                # keeping last_id contract-valid on both backends.
                if d.get('last_id') is not None:
                    d['last_id'] = normalize_id(str(d['last_id']))
                threads.append(cast(ThreadInfoDict, d))

            return threads

        return await self.backend.execute_read(_list_threads_postgresql)

    async def get_database_statistics(self, db_path: Path | None = None, *, scope: Scope) -> dict[str, Any]:
        """Get database statistics over the entries the scope may read.

        Every figure derived from stored entries -- the totals, the source and content-type
        breakdowns, the image, tag and thread counts and the two top-N lists -- counts only
        the rows the READ predicate admits. Image and tag figures count only the rows whose
        parent entry the predicate admits. The predicate filters rows before GROUP BY and LIMIT,
        so each top-N list holds the scope's own top items. ``database_size_mb`` is the size
        of the whole database, the same for every scope.

        Args:
            db_path: Path of the SQLite database file for the size figure; unused on PostgreSQL.
            scope: The caller's scope.

        Returns:
            Dictionary containing the database statistics.
        """
        backend_type = self.backend.backend_type
        entries = build_access_predicate(scope, mode=AccessMode.READ, backend_type=backend_type, outer='context_entries')
        parents = build_access_predicate(scope, mode=AccessMode.READ, backend_type=backend_type, outer='ce')
        entry_filter = entries.where_clause()
        parent_filter = parents.where_clause()

        # The grouping key is the unique secondary sort key of each top-N list: without it
        # a tie in `count` leaves the LIMIT window computed over an undefined ordering, so
        # which rows make the top N flaps under unrelated writes (PostgreSQL MVCC rewrites
        # heap order on every UPDATE). The byte-wise ordering makes that tiebreak decide
        # the LIMIT window the same way on both backends, instead of by the locale.
        thread_order = byte_ordered_text('thread_id', backend_type)
        tag_order = byte_ordered_text('t.tag', backend_type)
        total_sql = f'SELECT COUNT(*) AS count FROM context_entries{entry_filter}'
        by_source_sql = f'SELECT source, COUNT(*) AS count FROM context_entries{entry_filter} GROUP BY source'
        by_content_type_sql = (
            f'SELECT content_type, COUNT(*) AS count FROM context_entries{entry_filter} GROUP BY content_type'
        )
        if backend_type == 'sqlite':
            # SQLite looks a joined parent up through the UNIQUE index on id, which holds
            # neither owner_id nor visibility, so each image or tag row would read its
            # parent's table row and walk the text overflow pages stored before those
            # columns. The id set of the readable parents comes from one scan of the
            # covering idx_context_access_id instead, and image and tag rows are matched
            # against it.
            readable_parent_ids = f'SELECT ce.id FROM context_entries ce{parent_filter}'
            readable_images = f'image_attachments i WHERE i.context_entry_id IN ({readable_parent_ids})'
            readable_tags = f'tags t WHERE t.context_entry_id IN ({readable_parent_ids})'
        else:
            readable_images = f'image_attachments i JOIN context_entries ce ON ce.id = i.context_entry_id{parent_filter}'
            readable_tags = f'tags t JOIN context_entries ce ON ce.id = t.context_entry_id{parent_filter}'
        images_sql = f'SELECT COUNT(*) AS count FROM {readable_images}'
        unique_tags_sql = f'SELECT COUNT(DISTINCT t.tag) AS count FROM {readable_tags}'
        threads_sql = f'SELECT COUNT(DISTINCT thread_id) AS count FROM context_entries{entry_filter}'
        average_sql = (
            'SELECT AVG(entry_count) AS avg_entries FROM '
            f'(SELECT thread_id, COUNT(*) AS entry_count FROM context_entries{entry_filter} GROUP BY thread_id) sub'
        )
        most_active_sql = (
            f'SELECT thread_id, COUNT(*) AS count FROM context_entries{entry_filter} '
            f'GROUP BY thread_id ORDER BY count DESC, {thread_order} ASC LIMIT 5'
        )
        top_tags_sql = (
            f'SELECT t.tag AS tag, COUNT(*) AS count FROM {readable_tags} '
            f'GROUP BY t.tag ORDER BY count DESC, {tag_order} ASC LIMIT 10'
        )

        if backend_type == 'sqlite':

            def _get_stats_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
                cursor = conn.cursor()
                stats: dict[str, Any] = {}

                cursor.execute(total_sql, entries.params)
                stats['total_entries'] = cursor.fetchone()['count']

                cursor.execute(by_source_sql, entries.params)
                stats['by_source'] = {row['source']: row['count'] for row in cursor.fetchall()}

                cursor.execute(by_content_type_sql, entries.params)
                stats['by_content_type'] = {row['content_type']: row['count'] for row in cursor.fetchall()}

                cursor.execute(images_sql, parents.params)
                stats['total_images'] = cursor.fetchone()['count']

                cursor.execute(unique_tags_sql, parents.params)
                stats['unique_tags'] = cursor.fetchone()['count']

                cursor.execute(threads_sql, entries.params)
                stats['total_threads'] = cursor.fetchone()['count']

                cursor.execute(average_sql, entries.params)
                stats['avg_entries_per_thread'] = _to_float(cursor.fetchone()['avg_entries'])

                cursor.execute(most_active_sql, entries.params)
                stats['most_active_threads'] = [
                    {'thread_id': row['thread_id'], 'count': row['count']} for row in cursor.fetchall()
                ]

                cursor.execute(top_tags_sql, parents.params)
                stats['top_tags'] = [{'tag': row['tag'], 'count': row['count']} for row in cursor.fetchall()]

                stats['backend'] = 'sqlite'
                return stats

            stats = await self.backend.execute_read(_get_stats_sqlite)
        else:  # postgresql

            async def _get_stats_postgresql(conn: 'asyncpg.Connection') -> dict[str, Any]:
                stats: dict[str, Any] = {}

                stats['total_entries'] = await conn.fetchval(total_sql, *entries.params)

                rows = await conn.fetch(by_source_sql, *entries.params)
                stats['by_source'] = {row['source']: row['count'] for row in rows}

                rows = await conn.fetch(by_content_type_sql, *entries.params)
                stats['by_content_type'] = {row['content_type']: row['count'] for row in rows}

                stats['total_images'] = await conn.fetchval(images_sql, *parents.params)
                stats['unique_tags'] = await conn.fetchval(unique_tags_sql, *parents.params)
                stats['total_threads'] = await conn.fetchval(threads_sql, *entries.params)
                stats['avg_entries_per_thread'] = _to_float(await conn.fetchval(average_sql, *entries.params))

                rows = await conn.fetch(most_active_sql, *entries.params)
                stats['most_active_threads'] = [{'thread_id': row['thread_id'], 'count': row['count']} for row in rows]

                rows = await conn.fetch(top_tags_sql, *parents.params)
                stats['top_tags'] = [{'tag': row['tag'], 'count': row['count']} for row in rows]

                stats['backend'] = 'postgresql'
                return stats

            stats = await self.backend.execute_read(_get_stats_postgresql)

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

    async def get_summary_statistics(self, *, scope: Scope) -> dict[str, Any]:
        """Get summary generation statistics over the entries the scope may read.

        Args:
            scope: The caller's scope.

        Returns:
            Dictionary with summary_count, total_entries, and coverage_percentage
        """
        predicate = build_access_predicate(
            scope, mode=AccessMode.READ, backend_type=self.backend.backend_type, outer='context_entries',
        )
        total_sql = f'SELECT COUNT(*) AS count FROM context_entries{predicate.where_clause()}'
        summary_sql = (
            "SELECT COUNT(*) AS count FROM context_entries WHERE summary IS NOT NULL AND summary != ''"
            f'{predicate.and_clause()}'
        )

        def _figures(summary_count: int, total_entries: int) -> dict[str, Any]:
            coverage_percentage = round(summary_count / total_entries * 100, 2) if total_entries > 0 else 0.0
            return {
                'summary_count': summary_count,
                'total_entries': total_entries,
                'coverage_percentage': coverage_percentage,
            }

        if self.backend.backend_type == 'sqlite':

            def _get_summary_stats_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
                cursor = conn.cursor()
                total_entries = cursor.execute(total_sql, predicate.params).fetchone()['count']
                summary_count = cursor.execute(summary_sql, predicate.params).fetchone()['count']
                return _figures(summary_count, total_entries)

            return await self.backend.execute_read(_get_summary_stats_sqlite)

        # postgresql

        async def _get_summary_stats_postgresql(conn: 'asyncpg.Connection') -> dict[str, Any]:
            total_entries = await conn.fetchval(total_sql, *predicate.params)
            summary_count = await conn.fetchval(summary_sql, *predicate.params)
            return _figures(summary_count, total_entries)

        return await self.backend.execute_read(_get_summary_stats_postgresql)
