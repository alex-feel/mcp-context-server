"""Tests for the whitespace-only tag-filter rejection across repository sites.

A non-empty ``tags`` filter that normalizes to empty (every tag blank after
trimming, e.g. ``['   ']``) must NOT silently drop the criterion and widen the
result set to the whole scope. Each tag-filter site now routes that case through
its structured, breaker-exempt validation-error channel -- the ContextRepository
clause builder returns ``validation_errors``, while FtsRepository and
EmbeddingRepository raise their ControlFlowError-derived validation exceptions.

The genuinely absent filter (``tags=None``) and the empty-list filter
(``tags=[]``, which never enters the ``if tags:`` branch) stay untouched: both
mean "no tag filter" and return the unfiltered scope.
"""

import sqlite3
from collections.abc import AsyncGenerator
from datetime import UTC
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.backends import create_backend
from app.ids import generate_id_with_timestamp
from app.migrations.fts import apply_fts_migration
from app.repositories import RepositoryContainer
from app.repositories.base import BaseRepository
from app.repositories.embedding_repository import MetadataFilterValidationError
from app.repositories.fts_repository import FtsValidationError
from tests.conftest import requires_sqlite_vec


@pytest_asyncio.fixture
async def backend_and_repos(
    tmp_path: Path,
) -> AsyncGenerator[tuple[StorageBackend, RepositoryContainer], None]:
    """SQLite backend + RepositoryContainer with the full schema and two entries."""
    from app.schemas import load_schema

    db_path = tmp_path / 'tag_filter.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(load_schema('sqlite'))
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    await apply_fts_migration(backend, force=True)
    repos = RepositoryContainer(backend)

    base = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(2):
        cid = generate_id_with_timestamp(base.replace(second=i))

        def _write(conn: sqlite3.Connection, cid: str = cid) -> None:
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                'VALUES (?, ?, ?, ?, ?)',
                (cid, 't', 'agent', 'text', 'needle body'),
            )

        await backend.execute_write(_write)

    try:
        yield backend, repos
    finally:
        await backend.shutdown()


class TestNormalizeTagFilterHelper:
    """The shared normalizer folds case/whitespace and rejects an emptied filter."""

    def test_normalizes_and_lowercases(self) -> None:
        assert BaseRepository.normalize_tag_filter([' Foo ', 'BAR']) == ['foo', 'bar']

    def test_drops_blank_members_but_keeps_usable_ones(self) -> None:
        assert BaseRepository.normalize_tag_filter(['   ', 'kept']) == ['kept']

    def test_all_blank_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match='at least one non-blank tag'):
            BaseRepository.normalize_tag_filter(['   ', '\t', ''])


class TestContextRepositoryTagValidation:
    """search_contexts and grep_scan_text_contents short-circuit with errors."""

    @pytest.mark.asyncio
    async def test_search_contexts_blank_tags_returns_validation_error(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        _backend, repos = backend_and_repos
        rows, stats = await repos.context.search_contexts(thread_id='t', tags=['   '])
        assert rows == []
        assert stats.get('validation_errors')

    @pytest.mark.asyncio
    async def test_search_contexts_none_tags_returns_all(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        _backend, repos = backend_and_repos
        rows, stats = await repos.context.search_contexts(thread_id='t', tags=None)
        assert stats.get('validation_errors') is None
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_search_contexts_empty_tags_returns_all(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        # tags=[] never enters the `if tags:` branch: it means "no tag filter".
        _backend, repos = backend_and_repos
        rows, stats = await repos.context.search_contexts(thread_id='t', tags=[])
        assert stats.get('validation_errors') is None
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_grep_scan_blank_tags_returns_validation_error(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        _backend, repos = backend_and_repos
        rows, stats = await repos.context.grep_scan_text_contents(
            ascii_literal='needle', thread_id='t', tags=['   '],
        )
        assert rows == []
        assert stats.get('validation_errors')

    @pytest.mark.asyncio
    async def test_grep_scan_none_tags_scans_all(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        _backend, repos = backend_and_repos
        rows, stats = await repos.context.grep_scan_text_contents(
            ascii_literal='needle', thread_id='t', tags=None,
        )
        assert stats.get('validation_errors') is None
        assert len(rows) == 2


class TestFtsRepositoryTagValidation:
    """FtsRepository.search raises FtsValidationError on an emptied tag filter."""

    @pytest.mark.asyncio
    async def test_blank_tags_raises(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        _backend, repos = backend_and_repos
        with pytest.raises(FtsValidationError):
            await repos.fts.search('needle', thread_id='t', tags=['   '])

    @pytest.mark.asyncio
    async def test_empty_tags_does_not_raise_and_returns_matches(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        # tags=[] means "no tag filter": the search runs unfiltered.
        _backend, repos = backend_and_repos
        rows, _stats = await repos.fts.search('needle', thread_id='t', tags=[])
        assert len(rows) == 2


@requires_sqlite_vec
class TestEmbeddingRepositoryTagValidation:
    """EmbeddingRepository.search raises on an emptied tag filter before vec work."""

    @pytest.mark.asyncio
    async def test_blank_tags_raises(
        self,
        backend_and_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        _backend, repos = backend_and_repos
        with pytest.raises(MetadataFilterValidationError):
            await repos.embeddings.search([0.0] * 4, thread_id='t', tags=['   '])
