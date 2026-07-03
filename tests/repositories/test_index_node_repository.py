"""Tests for IndexNodeRepository (context_index_nodes per-node summaries).

Covers wholesale replace, summary retrieval, count, ON DELETE CASCADE, and the
graceful empty/zero behavior when the table is absent (feature disabled).
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
from app.migrations.index_tree import apply_index_tree_migration
from app.repositories import RepositoryContainer
from app.repositories.index_node_repository import IndexNodeRow


def _row(node_id: str, summary: str, span: tuple[int, int] = (0, 10)) -> IndexNodeRow:
    return IndexNodeRow(
        node_id=node_id, level=1, ordinal=1, title=node_id.title(),
        node_summary=summary, char_start=span[0], char_end=span[1],
    )


@pytest_asyncio.fixture
async def repos_with_table(tmp_path: Path) -> AsyncGenerator[tuple[StorageBackend, RepositoryContainer, str], None]:
    """Backend + repos with the index_tree table and one context entry."""
    from app.schemas import load_schema

    db_path = tmp_path / 'index_nodes.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(load_schema('sqlite'))
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    await apply_index_tree_migration(backend, force=True)
    repos = RepositoryContainer(backend)

    cid = generate_id_with_timestamp(datetime(2024, 1, 1, tzinfo=UTC))

    def _insert(conn: sqlite3.Connection) -> None:
        conn.execute(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) VALUES (?, ?, ?, ?, ?)',
            (cid, 't', 'agent', 'text', '# A\nbody'),
        )

    await backend.execute_write(_insert)
    try:
        yield backend, repos, cid
    finally:
        await backend.shutdown()


class TestReplaceAndGet:
    @pytest.mark.asyncio
    async def test_replace_then_get(
        self,
        repos_with_table: tuple[StorageBackend, RepositoryContainer, str],
    ) -> None:
        _backend, repos, cid = repos_with_table
        await repos.index_nodes.replace_nodes_for_context(
            cid,
            [_row('intro', 'About intro', span=(0, 10)), _row('setup', 'About setup', span=(10, 25))],
        )
        summaries = await repos.index_nodes.get_nodes_for_context(cid)
        assert summaries.by_node_id == {'intro': 'About intro', 'setup': 'About setup'}
        # The span index carries the same summaries keyed by (char_start, char_end)
        # so rows written by an older slug algorithm can re-attach by position.
        assert summaries.by_span == {(0, 10): 'About intro', (10, 25): 'About setup'}

    @pytest.mark.asyncio
    async def test_replace_is_wholesale(
        self,
        repos_with_table: tuple[StorageBackend, RepositoryContainer, str],
    ) -> None:
        _backend, repos, cid = repos_with_table
        await repos.index_nodes.replace_nodes_for_context(cid, [_row('a', 'first'), _row('b', 'second')])
        await repos.index_nodes.replace_nodes_for_context(cid, [_row('a', 'updated')])
        summaries = await repos.index_nodes.get_nodes_for_context(cid)
        assert summaries.by_node_id == {'a': 'updated'}

    @pytest.mark.asyncio
    async def test_empty_replace_clears(
        self,
        repos_with_table: tuple[StorageBackend, RepositoryContainer, str],
    ) -> None:
        _backend, repos, cid = repos_with_table
        await repos.index_nodes.replace_nodes_for_context(cid, [_row('a', 'first')])
        await repos.index_nodes.replace_nodes_for_context(cid, [])
        assert (await repos.index_nodes.get_nodes_for_context(cid)).by_node_id == {}

    @pytest.mark.asyncio
    async def test_count_all_nodes(
        self,
        repos_with_table: tuple[StorageBackend, RepositoryContainer, str],
    ) -> None:
        _backend, repos, cid = repos_with_table
        await repos.index_nodes.replace_nodes_for_context(cid, [_row('a', 'x'), _row('b', 'y'), _row('c', 'z')])
        assert await repos.index_nodes.count_all_nodes() == 3


class TestCascadeDelete:
    @pytest.mark.asyncio
    async def test_nodes_removed_on_entry_delete(
        self,
        repos_with_table: tuple[StorageBackend, RepositoryContainer, str],
    ) -> None:
        _backend, repos, cid = repos_with_table
        await repos.index_nodes.replace_nodes_for_context(cid, [_row('a', 'x')])
        deleted = await repos.context.delete_by_ids([cid])
        assert deleted == 1
        assert (await repos.index_nodes.get_nodes_for_context(cid)).by_node_id == {}


class TestTableAbsent:
    @pytest.mark.asyncio
    async def test_get_and_count_degrade_gracefully(self, tmp_path: Path) -> None:
        from app.schemas import load_schema

        db_path = tmp_path / 'no_table.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(load_schema('sqlite'))
        conn.close()

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        repos = RepositoryContainer(backend)
        try:
            # No index_tree migration applied -> table absent.
            absent = await repos.index_nodes.get_nodes_for_context('0' * 32)
            assert absent.by_node_id == {}
            assert absent.by_span == {}
            assert await repos.index_nodes.count_all_nodes() == 0
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_replace_nodes_is_noop_when_table_absent(self, tmp_path: Path) -> None:
        # The store/update write path calls replace_nodes_for_context INSIDE the
        # atomic transaction that persists the entry. When the table is absent
        # (feature never enabled / not migrated) the WRITE must early-return as a
        # no-op so the missing table can never abort a store. Guards the SQLite
        # sqlite_master / PostgreSQL to_regclass pre-check against regression -- a
        # revert would raise "no such table" and roll back the whole store.
        from app.schemas import load_schema

        db_path = tmp_path / 'no_table_replace.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(load_schema('sqlite'))
        conn.close()

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        repos = RepositoryContainer(backend)
        try:
            # Writing nodes against the absent table must NOT raise; it is a no-op
            # and the read side still degrades to empty.
            await repos.index_nodes.replace_nodes_for_context('0' * 32, [_row('a', 'x')])
            assert (await repos.index_nodes.get_nodes_for_context('0' * 32)).by_node_id == {}
        finally:
            await backend.shutdown()
