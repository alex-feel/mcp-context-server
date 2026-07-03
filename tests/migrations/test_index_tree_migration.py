"""Tests for the index_tree (context_index_nodes) migration.

Covers SQLite table creation, idempotent re-apply, and toggle gating (the table
is provisioned only when per-node summaries are enabled, mirroring how the FTS
migration gates on its own toggle).
"""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

import app.migrations.index_tree as index_tree_module
from app.backends import StorageBackend
from app.backends import create_backend
from app.migrations.index_tree import apply_index_tree_migration
from app.settings import get_settings


@pytest_asyncio.fixture
async def backend(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend with the base schema applied."""
    from app.schemas import load_schema

    db_path = tmp_path / 'index_tree.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(load_schema('sqlite'))
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    try:
        yield backend
    finally:
        await backend.shutdown()


async def _table_exists(backend: StorageBackend) -> bool:
    def _check(conn: sqlite3.Connection) -> bool:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='context_index_nodes'",
        )
        return cursor.fetchone() is not None

    return await backend.execute_read(_check)


class TestIndexTreeMigration:
    @pytest.mark.asyncio
    async def test_force_creates_table(self, backend: StorageBackend) -> None:
        await apply_index_tree_migration(backend, force=True)
        assert await _table_exists(backend) is True

    @pytest.mark.asyncio
    async def test_idempotent_reapply(self, backend: StorageBackend) -> None:
        await apply_index_tree_migration(backend, force=True)
        # Second application must not raise (CREATE ... IF NOT EXISTS).
        await apply_index_tree_migration(backend, force=True)
        assert await _table_exists(backend) is True

    @pytest.mark.asyncio
    async def test_gated_off_skips_table(
        self,
        backend: StorageBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'false')
        get_settings.cache_clear()
        try:
            monkeypatch.setattr(index_tree_module, 'settings', get_settings())
            await apply_index_tree_migration(backend, force=False)
            assert await _table_exists(backend) is False
        finally:
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_table_has_expected_columns(self, backend: StorageBackend) -> None:
        await apply_index_tree_migration(backend, force=True)

        def _columns(conn: sqlite3.Connection) -> set[str]:
            cursor = conn.execute('PRAGMA table_info(context_index_nodes)')
            return {row[1] for row in cursor.fetchall()}

        columns = await backend.execute_read(_columns)
        assert {'context_id', 'node_id', 'level', 'ordinal', 'title', 'node_summary', 'char_start', 'char_end'} <= columns


class TestSqliteIndexTreeDdlAccessor:
    """The shared DDL accessor the migration CLI uses for its raw SQLite target init."""

    def test_ddl_creates_table_on_raw_connection(self, tmp_path: Path) -> None:
        # The migration CLI's initialize_target_sqlite provisions context_index_nodes
        # on its raw sqlite3 connection via sqlite_index_tree_ddl() -- the SAME DDL as
        # the server startup migration -- so a migrated SQLite target matches a
        # server-initialized DB (no drift). Verify the DDL applies and is idempotent.
        from app.migrations.index_tree import sqlite_index_tree_ddl
        from app.schemas import load_schema

        conn = sqlite3.connect(str(tmp_path / 'cli_target.db'))
        try:
            conn.executescript(load_schema('sqlite'))
            create_table_sql, create_index_sql = sqlite_index_tree_ddl()
            conn.execute(create_table_sql)
            conn.execute(create_index_sql)
            conn.commit()
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='context_index_nodes'",
            ).fetchone()
            assert row is not None
            # Idempotent (CREATE ... IF NOT EXISTS): a re-apply must not raise.
            conn.execute(create_table_sql)
            conn.execute(create_index_sql)
        finally:
            conn.close()
