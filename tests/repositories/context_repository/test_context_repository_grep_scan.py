"""Tests for ContextRepository.grep_scan_text_contents (the grep pre-filter scan).

The headline guarantee: the scan is EXHAUSTIVE. It must NOT be built on
search_contexts (which hard-caps at LIMIT 50), so a grep over more than 50
matching entries returns every one. Also covers the optional ASCII substring
pre-narrow, newest-first keyset ordering, the max_entries_scanned cap with a
truncated flag, canonical id output, and metadata-filter validation.
"""

import re
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
import pytest_asyncio

from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.backends import create_backend
from app.ids import generate_id_with_timestamp
from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from tests.helpers import LOCAL_SCOPE


@pytest_asyncio.fixture
async def backend_and_repos(
    tmp_path: Path,
) -> AsyncGenerator[tuple[StorageBackend, RepositoryContainer], None]:
    """SQLite backend + RepositoryContainer with the full schema applied."""
    from app.schemas import load_schema

    db_path = tmp_path / 'grep_scan.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(load_schema('sqlite'))
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    repos = RepositoryContainer(backend)
    try:
        yield backend, repos
    finally:
        await backend.shutdown()


async def _insert_entries(
    backend: StorageBackend,
    *,
    count: int,
    text_fn: Callable[[int], str],
    thread_id: str = 't',
) -> list[str]:
    """Insert ``count`` entries with strictly increasing ids; return the ids."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    ids: list[str] = []
    for i in range(count):
        cid = generate_id_with_timestamp(base + timedelta(seconds=i))
        ids.append(cid)
        text = text_fn(i)

        def _write(conn: sqlite3.Connection, cid: str = cid, text: str = text) -> None:
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES (?, ?, ?, ?, ?, 'local')",
                (cid, thread_id, 'agent', 'text', text),
            )

        await backend.execute_write(_write)
    return ids


async def _insert_foreign_private(backend: StorageBackend, offset_seconds: int) -> str:
    """Insert one private alice entry containing ``needle``, ``offset_seconds`` after the first ``_insert_entries`` row."""
    cid = generate_id_with_timestamp(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=offset_seconds))

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
            "VALUES (?, 't', 'agent', 'text', 'needle', 'alice')",
            (cid,),
        )

    await backend.execute_write(_write)
    return cid


class TestGrepScanExhaustiveness:
    """The scan returns every matching entry, not just the newest 50."""

    @pytest.mark.asyncio
    async def test_returns_all_matches_beyond_fifty(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=120, text_fn=lambda _i: 'contains needle here')
        rows, stats = await repos.context.grep_scan_text_contents(
            ascii_literal='needle', thread_id='t', max_entries_scanned=1000,
            scope=LOCAL_SCOPE,
        )
        assert len(rows) == 120
        assert stats['truncated'] is False

    @pytest.mark.asyncio
    async def test_pagination_crosses_pages(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=25, text_fn=lambda _i: 'needle')
        rows, _stats = await repos.context.grep_scan_text_contents(
            ascii_literal='needle', thread_id='t', page_size=10, max_entries_scanned=1000,
            scope=LOCAL_SCOPE,
        )
        assert len(rows) == 25


class TestGrepScanPreNarrow:
    """The optional ASCII pre-narrow filters at the SQL layer; None scans all."""

    @pytest.mark.asyncio
    async def test_ascii_literal_filters_candidates(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=40, text_fn=lambda i: 'needle' if i % 2 == 0 else 'haystack')
        rows, _stats = await repos.context.grep_scan_text_contents(ascii_literal='needle', thread_id='t', scope=LOCAL_SCOPE)
        assert len(rows) == 20
        assert all('needle' in text for _cid, text in rows)

    @pytest.mark.asyncio
    async def test_no_literal_scans_all_in_thread(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=40, text_fn=lambda i: 'needle' if i % 2 == 0 else 'haystack')
        rows, _stats = await repos.context.grep_scan_text_contents(ascii_literal=None, thread_id='t', scope=LOCAL_SCOPE)
        assert len(rows) == 40


class TestGrepScanBounds:
    """max_entries_scanned caps the scan and flags truncation; ids are canonical."""

    @pytest.mark.asyncio
    async def test_max_entries_scanned_truncates(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=30, text_fn=lambda _i: 'needle')
        rows, stats = await repos.context.grep_scan_text_contents(
            ascii_literal='needle', thread_id='t', max_entries_scanned=10,
            scope=LOCAL_SCOPE,
        )
        assert len(rows) == 10
        assert stats['truncated'] is True

    @pytest.mark.asyncio
    async def test_exact_fit_at_cap_is_not_truncated(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        # Regression: a matching set EXACTLY equal to max_entries_scanned (with a
        # full final page) must NOT be flagged truncated -- the single-row lookahead
        # finds no further candidate, so the scan was exhaustive.
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=10, text_fn=lambda _i: 'needle')
        rows, stats = await repos.context.grep_scan_text_contents(
            ascii_literal='needle', thread_id='t', max_entries_scanned=10, page_size=5,
            scope=LOCAL_SCOPE,
        )
        assert len(rows) == 10
        assert stats['truncated'] is False

    @pytest.mark.asyncio
    async def test_one_over_cap_is_truncated(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        # One more matching row than the cap -> the lookahead finds it -> truncated.
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=11, text_fn=lambda _i: 'needle')
        rows, stats = await repos.context.grep_scan_text_contents(
            ascii_literal='needle', thread_id='t', max_entries_scanned=10, page_size=5,
            scope=LOCAL_SCOPE,
        )
        assert len(rows) == 10
        assert stats['truncated'] is True

    @pytest.mark.asyncio
    async def test_budget_exact_fit_on_last_row_is_not_truncated(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        # Regression: when the aggregate byte budget is crossed EXACTLY on the
        # final matching row (no further matches remain), the scan was exhaustive
        # and must NOT be flagged truncated -- the budget break runs the same
        # single-row lookahead as the entry-cap break. Five 20-char entries with a
        # 100-code-point budget cross the budget on the oldest (last) row.
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=5, text_fn=lambda _i: 'x' * 20)
        rows, stats = await repos.context.grep_scan_text_contents(
            ascii_literal=None, thread_id='t', aggregate_bytes_budget=100, max_entries_scanned=1000,
            scope=LOCAL_SCOPE,
        )
        assert len(rows) == 5
        assert stats['truncated'] is False

    @pytest.mark.asyncio
    async def test_budget_with_more_rows_remaining_is_truncated(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        # One more matching row than the budget admits -> the lookahead finds it
        # -> truncated, and only the budgeted rows are returned.
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=6, text_fn=lambda _i: 'x' * 20)
        rows, stats = await repos.context.grep_scan_text_contents(
            ascii_literal=None, thread_id='t', aggregate_bytes_budget=100, max_entries_scanned=1000,
            scope=LOCAL_SCOPE,
        )
        assert len(rows) == 5
        assert stats['truncated'] is True

    @pytest.mark.asyncio
    async def test_exact_fit_without_pre_narrow_is_not_truncated(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        # Same exact-fit guarantee on the full-scan path (ascii_literal=None) so the
        # lookahead's no-LIKE clause branch is exercised too.
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=10, text_fn=lambda _i: 'anything')
        rows, stats = await repos.context.grep_scan_text_contents(
            ascii_literal=None, thread_id='t', max_entries_scanned=10, page_size=5,
            scope=LOCAL_SCOPE,
        )
        assert len(rows) == 10
        assert stats['truncated'] is False

    @pytest.mark.asyncio
    async def test_newest_first_ordering(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=5, text_fn=lambda i: f'item{i}')
        rows, _stats = await repos.context.grep_scan_text_contents(
            ascii_literal=None, thread_id='t', max_entries_scanned=3,
            scope=LOCAL_SCOPE,
        )
        # id DESC -> newest first: item4, item3, item2
        assert [text for _cid, text in rows] == ['item4', 'item3', 'item2']

    @pytest.mark.asyncio
    async def test_returned_ids_are_canonical_hex(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=3, text_fn=lambda _i: 'needle')
        rows, _stats = await repos.context.grep_scan_text_contents(ascii_literal='needle', thread_id='t', scope=LOCAL_SCOPE)
        for cid, _text in rows:
            assert len(cid) == 32
            assert all(c in '0123456789abcdef' for c in cid)


class TestGrepScanValidation:
    """An invalid metadata filter short-circuits with validation_errors."""

    @pytest.mark.asyncio
    async def test_invalid_metadata_filter_returns_errors(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=3, text_fn=lambda _i: 'needle')
        rows, stats = await repos.context.grep_scan_text_contents(
            ascii_literal='needle',
            thread_id='t',
            metadata_filters=[{'field': 'status', 'operator': 'NOT_A_REAL_OP', 'value': 'x'}],
            scope=LOCAL_SCOPE,
        )
        assert rows == []
        assert stats.get('validation_errors')


class _RecordingConnection:
    """A PostgreSQL stand-in that serves one page of one row and records every statement."""

    def __init__(self, row_id: str) -> None:
        self.row_id = row_id
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query: str, *args: object) -> list[dict[str, str]]:
        self.statements.append((query, args))
        return [{'id': self.row_id, 'text_content': 'needle'}]

    async def fetchval(self, query: str, *args: object) -> None:
        self.statements.append((query, args))


def _bound(query: str, args: tuple[object, ...], pattern: str) -> object:
    """Return the argument bound to the single ``$n`` placeholder that ``pattern`` captures."""
    match = re.search(pattern, query)
    assert match is not None, f'{pattern!r} not in {query!r}'
    return args[int(match.group(1)) - 1]


class TestGrepScanScope:
    """The scan visits only rows the caller may read, and so do its page and lookahead statements."""

    @pytest.mark.asyncio
    async def test_hidden_rows_never_take_a_slot_or_flag_truncation(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """Another principal's private rows on both sides of the caller's rows are neither scanned nor counted."""
        backend, repos = backend_and_repos
        own_ids = await _insert_entries(backend, count=2, text_fn=lambda _i: 'needle')
        hidden_ids = [await _insert_foreign_private(backend, offset) for offset in (-5, -4, 10, 11, 12)]

        rows, stats = await repos.context.grep_scan_text_contents(
            ascii_literal='needle', thread_id='t', max_entries_scanned=2, page_size=1, scope=LOCAL_SCOPE,
        )

        assert [cid for cid, _text in rows] == own_ids[::-1]
        assert stats['scanned'] == 2
        assert stats['truncated'] is False
        owner_rows, _ = await repos.context.grep_scan_text_contents(
            ascii_literal='needle', thread_id='t', scope=AccessScope('alice', frozenset()),
        )
        assert sorted(cid for cid, _text in owner_rows) == sorted(hidden_ids)

    @pytest.mark.asyncio
    async def test_postgresql_page_and_lookahead_bind_the_scope_in_order(self) -> None:
        """Every PostgreSQL statement numbers its placeholders contiguously with the scope after the filters."""
        row_id = 'a' * 32
        connection = _RecordingConnection(row_id)
        pg_backend = Mock()
        pg_backend.backend_type = 'postgresql'

        async def _execute_read(closure: Callable[[object], Awaitable[object]]) -> object:
            return await closure(connection)

        pg_backend.execute_read = _execute_read
        repo = ContextRepository(cast(StorageBackend, pg_backend))

        rows, stats = await repo.grep_scan_text_contents(
            ascii_literal='needle',
            thread_id='t',
            tags=['x'],
            metadata_filters=[{'key': 'author', 'operator': 'eq', 'value': 'alice'}],
            max_entries_scanned=1,
            page_size=1,
            scope=AccessScope('bob', frozenset({'team-x'})),
        )

        assert rows == [(row_id, 'needle')]
        assert stats == {'scanned': 1, 'truncated': False, 'backend': 'postgresql'}
        (page, page_args), (lookahead, lookahead_args) = connection.statements
        for query, args in connection.statements:
            assert sorted({int(number) for number in re.findall(r'\$(\d+)', query)}) == list(range(1, len(args) + 1))
            assert query.index('FROM tags') < query.index('context_entries.owner_id') < query.index('ILIKE')
            assert _bound(query, args, r'context_entries\.owner_id = \$(\d+)') == 'bob'
            assert _bound(query, args, r'= ANY\(\$(\d+)::text\[\]\)') == ['team-x']
            assert _bound(query, args, r'ILIKE \$(\d+)') == '%needle%'
        assert _bound(page, page_args, r'LIMIT \$(\d+)') == 1
        assert _bound(lookahead, lookahead_args, r'id < \$(\d+)') == row_id
