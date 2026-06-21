"""Tests for ContextRepository.grep_scan_text_contents (the grep pre-filter scan).

The headline guarantee: the scan is EXHAUSTIVE. It must NOT be built on
search_contexts (which hard-caps at LIMIT 50), so a grep over more than 50
matching entries returns every one. Also covers the optional ASCII substring
pre-narrow, newest-first keyset ordering, the max_entries_scanned cap with a
truncated flag, canonical id output, and metadata-filter validation.
"""

import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.backends import create_backend
from app.ids import generate_id_with_timestamp
from app.repositories import RepositoryContainer


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
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                'VALUES (?, ?, ?, ?, ?)',
                (cid, thread_id, 'agent', 'text', text),
            )

        await backend.execute_write(_write)
    return ids


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
        rows, _stats = await repos.context.grep_scan_text_contents(ascii_literal='needle', thread_id='t')
        assert len(rows) == 20
        assert all('needle' in text for _cid, text in rows)

    @pytest.mark.asyncio
    async def test_no_literal_scans_all_in_thread(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        backend, repos = backend_and_repos
        await _insert_entries(backend, count=40, text_fn=lambda i: 'needle' if i % 2 == 0 else 'haystack')
        rows, _stats = await repos.context.grep_scan_text_contents(ascii_literal=None, thread_id='t')
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
        )
        assert len(rows) == 10
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
        rows, _stats = await repos.context.grep_scan_text_contents(ascii_literal='needle', thread_id='t')
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
        )
        assert rows == []
        assert stats.get('validation_errors')
