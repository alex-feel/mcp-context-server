"""Tests for StatisticsRepository.get_thread_list.

Covers per-thread counts, the per-backend latest-id SQL, and limit/offset pagination.
"""

import sqlite3

import pytest

from app.backends.base import StorageBackend
from app.ids import generate_id
from app.repositories.statistics_repository import StatisticsRepository
from tests.helpers import LOCAL_SCOPE


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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES (?, 'mm-thread', 'user', 'text', 'Text only', 'local')",
                (generate_id(),),
            )
            cursor.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES (?, 'mm-thread', 'user', 'multimodal', 'With images', 'local')",
                (generate_id(),),
            )

        await stats_test_db.execute_write(_insert_data)

        threads = await stats_repo.get_thread_list(scope=LOCAL_SCOPE)
        assert len(threads) == 1
        thread = threads[0]
        assert thread['multimodal_count'] == 1
        assert thread['entry_count'] == 2


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
        # app/backends/postgresql_backend/pool_callbacks.py).
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
            'SQLite branch must use MAX(id) on the TEXT id column; '
            'this is the correct form for the SQLite backend.'
        )


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
                '(id, thread_id, source, content_type, text_content, created_at, owner_id) '
                "VALUES (?, ?, 'user', 'text', 'entry', ?, 'local')",
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

        result = await stats_repo.get_thread_list(scope=LOCAL_SCOPE)

        assert [t['thread_id'] for t in result] == self._EXPECTED_ORDER

    @pytest.mark.asyncio
    async def test_limit_none_explicit_returns_all_threads(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Explicit limit=None is identical to the default (no LIMIT clause)."""
        await stats_test_db.execute_write(self._insert_five_threads)

        result = await stats_repo.get_thread_list(scope=LOCAL_SCOPE, limit=None)

        assert [t['thread_id'] for t in result] == self._EXPECTED_ORDER

    @pytest.mark.asyncio
    async def test_limit_bounds_result_preserving_order(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """limit returns the first N threads of the canonical order."""
        await stats_test_db.execute_write(self._insert_five_threads)

        result = await stats_repo.get_thread_list(scope=LOCAL_SCOPE, limit=2)

        assert [t['thread_id'] for t in result] == ['thread_e', 'thread_d']

    @pytest.mark.asyncio
    async def test_offset_skips_leading_threads(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """limit + offset returns the correct page slice in canonical order."""
        await stats_test_db.execute_write(self._insert_five_threads)

        result = await stats_repo.get_thread_list(scope=LOCAL_SCOPE, limit=2, offset=2)

        assert [t['thread_id'] for t in result] == ['thread_c', 'thread_b']

    @pytest.mark.asyncio
    async def test_offset_past_end_returns_empty(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """An offset past the last row yields an empty page, not an error."""
        await stats_test_db.execute_write(self._insert_five_threads)

        result = await stats_repo.get_thread_list(scope=LOCAL_SCOPE, limit=5, offset=10)

        assert result == []

    @pytest.mark.asyncio
    async def test_limit_larger_than_total_returns_all(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """A limit exceeding the thread count returns every thread."""
        await stats_test_db.execute_write(self._insert_five_threads)

        result = await stats_repo.get_thread_list(scope=LOCAL_SCOPE, limit=100)

        assert [t['thread_id'] for t in result] == self._EXPECTED_ORDER

    @pytest.mark.asyncio
    async def test_full_page_walk_covers_all_threads_once(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """Walking pages of size 2 reconstructs the full ordered list exactly once."""
        await stats_test_db.execute_write(self._insert_five_threads)

        page1 = await stats_repo.get_thread_list(scope=LOCAL_SCOPE, limit=2, offset=0)
        page2 = await stats_repo.get_thread_list(scope=LOCAL_SCOPE, limit=2, offset=2)
        page3 = await stats_repo.get_thread_list(scope=LOCAL_SCOPE, limit=2, offset=4)

        walked = [t['thread_id'] for t in (*page1, *page2, *page3)]
        assert walked == self._EXPECTED_ORDER
