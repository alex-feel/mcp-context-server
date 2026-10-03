"""Tests for deterministic top-N windows: the grouping-key tiebreak and byte-wise text collation on both backends."""

import sqlite3
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
from typing import TypeVar
from typing import cast

import pytest

from app.backends.base import StorageBackend
from app.repositories.statistics_repository import StatisticsRepository

T = TypeVar('T')


class TestTopNTiebreakDeterminism:
    """Ranked top-N windows carry a unique secondary sort key.

    ``most_active_threads`` (LIMIT 5) and ``top_tags`` (LIMIT 10) rank by a COUNT
    that ties easily; when a tie straddles the window boundary, an ordering without
    a unique key leaves which rows appear undefined, and unrelated writes can change
    them. ``get_tag_statistics`` slices the first ten of its full list, so the same
    holds there. The grouping key as a secondary sort makes each window a function
    of the data alone.
    """

    @staticmethod
    def _insert_tied_data(conn: sqlite3.Connection) -> None:
        """Seed threads and tags whose counts are entirely tied."""
        cursor = conn.cursor()
        entry_id = 0
        # Twelve threads, two entries each: every thread ties at count 2, so the
        # LIMIT 5 window is decided purely by the tiebreak.
        for thread_index in range(12):
            for _ in range(2):
                entry_id += 1
                cursor.execute(
                    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                    "VALUES (?, ?, ?, ?, ?, 'local')",
                    (
                        f'0190abcdef1234567890abcd{entry_id:08d}',
                        f'thread-{thread_index:02d}',
                        'user',
                        'text',
                        'tied entry',
                    ),
                )
                # Twelve tags, each used twice: the same full tie inside LIMIT 10.
                cursor.execute(
                    'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                    (f'0190abcdef1234567890abcd{entry_id:08d}', f'tag-{thread_index:02d}'),
                )

    @pytest.mark.asyncio
    async def test_top_windows_are_ordered_by_the_grouping_key_on_a_full_tie(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """A complete tie yields the lexicographically first rows, not an arbitrary set."""
        await stats_test_db.execute_write(self._insert_tied_data)

        stats = await stats_repo.get_database_statistics()

        assert [t['thread_id'] for t in stats['most_active_threads']] == [
            f'thread-{i:02d}' for i in range(5)
        ]
        assert [t['tag'] for t in stats['top_tags']] == [f'tag-{i:02d}' for i in range(10)]

    @pytest.mark.asyncio
    async def test_repeated_calls_return_the_same_window(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """The window is a function of the data: unrelated writes cannot reshuffle it."""
        await stats_test_db.execute_write(self._insert_tied_data)

        first = await stats_repo.get_database_statistics()

        def _unrelated_update(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE context_entries SET metadata = '{\"touched\": true}' "
                "WHERE thread_id = 'thread-00'",
            )

        await stats_test_db.execute_write(_unrelated_update)
        second = await stats_repo.get_database_statistics()

        assert second['most_active_threads'] == first['most_active_threads']
        assert second['top_tags'] == first['top_tags']

    @pytest.mark.asyncio
    async def test_tag_statistics_top_ten_slice_is_deterministic(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """top_10_tags slices a fully tied list, so the list itself must be ordered."""
        await stats_test_db.execute_write(self._insert_tied_data)

        stats = await stats_repo.get_tag_statistics()

        assert [t['tag'] for t in stats['top_10_tags']] == [f'tag-{i:02d}' for i in range(10)]


class _SqlRecordingPgConnection:
    """Async connection stub recording every statement the PostgreSQL paths emit."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def fetch(self, query: str, *_args: object) -> list[dict[str, object]]:
        """Record the statement and return no rows.

        Args:
            query: SQL text emitted by the repository.
            _args: Bound parameters (unused).

        Returns:
            An empty row list.
        """
        self.queries.append(' '.join(query.split()))
        return []

    async def fetchrow(self, query: str, *_args: object) -> dict[str, object] | None:
        """Record the statement and return no row.

        Args:
            query: SQL text emitted by the repository.
            _args: Bound parameters (unused).

        Returns:
            ``None``, which every caller tolerates as an empty result.
        """
        self.queries.append(' '.join(query.split()))
        return None

    async def fetchval(self, query: str, *_args: object) -> int:
        """Record the statement and return a zero scalar.

        Args:
            query: SQL text emitted by the repository.
            _args: Bound parameters (unused).

        Returns:
            Zero.
        """
        self.queries.append(' '.join(query.split()))
        return 0


class _SqlRecordingPgBackend:
    """Backend stub selecting the PostgreSQL branch and capturing its SQL."""

    backend_type = 'postgresql'

    def __init__(self) -> None:
        self.connection = _SqlRecordingPgConnection()

    async def execute_read(self, operation: Callable[[Any], Awaitable[T]]) -> T:
        """Run the repository's async closure against the recording connection.

        Args:
            operation: The async closure the repository passes to the backend.

        Returns:
            Whatever the closure returns.
        """
        return await operation(self.connection)


@pytest.mark.asyncio
class TestTextOrderingIsByteWiseOnBothBackends:
    """Text sort terms compare bytes on both backends, so top-N membership matches.

    SQLite orders TEXT byte-wise (BINARY); PostgreSQL orders it by the database
    locale collation, which ranks punctuation differently. Because these windows
    carry LIMIT 5 / LIMIT 10 / a ``[:10]`` slice, a tie in the count makes the
    collation decide WHICH rows are returned, not merely their order -- so the
    same data yields a different ``most_active_threads`` / ``top_tags`` list after
    a SQLite-to-PostgreSQL migration. PostgreSQL therefore needs an explicit
    byte-wise collation on every free-text sort term.
    """

    @staticmethod
    def _insert_collation_sensitive_data(conn: sqlite3.Connection) -> None:
        """Seed six equally-active threads and six equally-used tags.

        Half the names carry a hyphen before the final character. Byte ordering
        ranks '-' (0x2D) below every letter, while a locale collation that treats
        punctuation as secondary ranks them the other way, so the two orderings
        select DIFFERENT rows once the window is smaller than the tie.
        """
        cursor = conn.cursor()
        names = ['coll-a', 'coll-b', 'coll-c', 'colla', 'collb', 'collc']
        entry_id = 0
        for name in names:
            for _ in range(2):
                entry_id += 1
                context_id = f'0190abcdef1234567890abce{entry_id:08d}'
                cursor.execute(
                    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                    "VALUES (?, ?, ?, ?, ?, 'local')",
                    (context_id, name, 'user', 'text', 'collation entry'),
                )
                cursor.execute(
                    'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                    (context_id, name),
                )

    async def test_sqlite_top_threads_window_follows_byte_order(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """The LIMIT 5 window admits the byte-wise first five tied threads."""
        await stats_test_db.execute_write(self._insert_collation_sensitive_data)

        stats = await stats_repo.get_database_statistics()

        assert [t['thread_id'] for t in stats['most_active_threads']] == [
            'coll-a', 'coll-b', 'coll-c', 'colla', 'collb',
        ]

    async def test_sqlite_thread_tag_list_follows_byte_order(
        self,
        stats_test_db: StorageBackend,
        stats_repo: StatisticsRepository,
    ) -> None:
        """The per-thread tag list is byte-ordered, matching the tags array contract."""
        await stats_test_db.execute_write(self._insert_collation_sensitive_data)

        def _tag_both_names(conn: sqlite3.Connection) -> None:
            row = conn.execute(
                "SELECT id FROM context_entries WHERE thread_id = 'coll-a' LIMIT 1",
            ).fetchone()
            conn.execute(
                'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                (row[0], 'ra'),
            )
            conn.execute(
                'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                (row[0], 'r-z'),
            )

        await stats_test_db.execute_write(_tag_both_names)

        stats = await stats_repo.get_thread_statistics('coll-a')

        assert stats['tags'] == ['coll-a', 'r-z', 'ra']

    async def test_postgresql_top_windows_force_byte_collation(self) -> None:
        """Both LIMIT-carrying tiebreaks in get_database_statistics are byte-wise."""
        backend = _SqlRecordingPgBackend()
        repo = StatisticsRepository(cast(StorageBackend, backend))

        await repo.get_database_statistics()

        recorded = backend.connection.queries
        assert any(
            'ORDER BY count DESC, thread_id COLLATE "C" ASC LIMIT 5' in query
            for query in recorded
        ), recorded
        assert any(
            'ORDER BY count DESC, tag COLLATE "C" ASC LIMIT 10' in query
            for query in recorded
        ), recorded

    async def test_postgresql_tag_statistics_forces_byte_collation(self) -> None:
        """The full tag list feeding the top_10_tags slice is byte-ordered."""
        backend = _SqlRecordingPgBackend()
        repo = StatisticsRepository(cast(StorageBackend, backend))

        await repo.get_tag_statistics()

        assert any(
            'ORDER BY count DESC, tag COLLATE "C" ASC' in query
            for query in backend.connection.queries
        ), backend.connection.queries

    async def test_postgresql_thread_tag_list_forces_byte_collation(self) -> None:
        """The per-thread tag list is byte-ordered and avoids the DISTINCT restriction."""
        backend = _SqlRecordingPgBackend()
        repo = StatisticsRepository(cast(StorageBackend, backend))

        await repo.get_thread_statistics('some-thread')

        tag_queries = [q for q in backend.connection.queries if 'FROM tags t' in q]
        assert tag_queries, backend.connection.queries
        assert any('ORDER BY t.tag COLLATE "C"' in query for query in tag_queries)
        # PostgreSQL rejects an ORDER BY expression missing from a DISTINCT select
        # list, so the deduplication is expressed with GROUP BY instead.
        assert all('SELECT DISTINCT' not in query for query in tag_queries)
