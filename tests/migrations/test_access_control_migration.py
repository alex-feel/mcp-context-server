"""Tests for the access-control schema migration.

Covers the in-place upgrade path: a database created BEFORE ``owner_id``/
``visibility`` and ``context_entry_grants`` were added to the base schema gains
them via ``apply_access_control_migration``. Mirrors
``tests/migrations/test_version_migration.py``: a real temp SQLite backend built
from a hand-rolled ``CREATE TABLE`` in the pre-migration shape, then the
migration adds the columns (fail-closed backfill: owner = configured default
principal, visibility 'private'), provisions the grants table and lookup
indexes, and is idempotent.

PostgreSQL coverage rides on the dual-backend real-server harness plus the live
deploy-stack integration, matching the sibling column-migration tests; a
dedicated PostgreSQL fixture is intentionally NOT invented here.
"""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.backends import create_backend
from app.errors import ConfigurationError
from app.ids import generate_id
from app.migrations.access_control import apply_access_control_migration
from app.settings import get_settings

# The current context_entries columns MINUS owner_id/visibility -- the
# pre-migration shape of a database created before the access-control columns
# were added to the base schema.
_PRE_MIGRATION_CONTEXT_ENTRIES_DDL = '''
    CREATE TABLE context_entries (
        rowid_int INTEGER PRIMARY KEY AUTOINCREMENT,
        id TEXT NOT NULL UNIQUE,
        thread_id TEXT NOT NULL,
        source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
        content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
        text_content TEXT,
        metadata JSON,
        summary TEXT,
        content_hash TEXT,
        version INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
'''


async def _columns(backend: StorageBackend) -> list[str]:
    def _check(conn: sqlite3.Connection) -> list[str]:
        cursor = conn.execute('PRAGMA table_info(context_entries)')
        return [row[1] for row in cursor.fetchall()]

    return await backend.execute_read(_check)


async def _table_exists(backend: StorageBackend, table: str) -> bool:
    def _check(conn: sqlite3.Connection) -> bool:
        cursor = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,),
        )
        return cursor.fetchone() is not None

    return await backend.execute_read(_check)


async def _index_names(backend: StorageBackend) -> set[str]:
    def _check(conn: sqlite3.Connection) -> set[str]:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        return {row[0] for row in cursor.fetchall()}

    return await backend.execute_read(_check)


@pytest_asyncio.fixture
async def backend_pre_migration(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend whose schema predates the access-control columns and table."""
    db_path = tmp_path / 'test_access_control_pre_migration.db'

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(_PRE_MIGRATION_CONTEXT_ENTRIES_DDL)
        conn.execute('CREATE INDEX IF NOT EXISTS idx_thread_id ON context_entries(thread_id)')
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_thread_source ON context_entries(thread_id, source)',
        )

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    try:
        yield backend
    finally:
        await backend.shutdown()


class TestAccessControlMigration:
    """apply_access_control_migration provisions columns, grants table, indexes (SQLite)."""

    @pytest.mark.asyncio
    async def test_schema_absent_before_migration(self, backend_pre_migration: StorageBackend) -> None:
        """Precondition: the pre-migration DB has no access-control schema."""
        columns = await _columns(backend_pre_migration)
        assert 'owner_id' not in columns
        assert 'visibility' not in columns
        assert not await _table_exists(backend_pre_migration, 'context_entry_grants')

    @pytest.mark.asyncio
    async def test_migration_adds_columns_table_and_indexes(
        self, backend_pre_migration: StorageBackend,
    ) -> None:
        """The migration adds both columns, the grants table, and the lookup indexes."""
        await apply_access_control_migration(backend_pre_migration)

        columns = await _columns(backend_pre_migration)
        assert 'owner_id' in columns
        assert 'visibility' in columns
        assert await _table_exists(backend_pre_migration, 'context_entry_grants')

        indexes = await _index_names(backend_pre_migration)
        assert 'idx_grants_entry_principal' in indexes
        assert 'idx_grants_principal' in indexes
        assert 'idx_context_owner' in indexes
        assert 'idx_context_owner_thread' in indexes
        assert 'idx_context_public' in indexes

    @pytest.mark.asyncio
    async def test_preexisting_row_backfills_fail_closed(
        self, backend_pre_migration: StorageBackend,
    ) -> None:
        """A row present BEFORE the migration is backfilled to the configured
        default principal and 'private' visibility (the fail-closed decision)."""
        legacy_id = generate_id()

        def _insert_legacy(conn: sqlite3.Connection) -> None:
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES (?, 't1', 'user', 'text', 'legacy row')",
                (legacy_id,),
            )

        await backend_pre_migration.execute_write(_insert_legacy)

        await apply_access_control_migration(backend_pre_migration)

        def _read(conn: sqlite3.Connection) -> tuple[str, str] | None:
            cursor = conn.execute(
                'SELECT owner_id, visibility FROM context_entries WHERE id = ?', (legacy_id,),
            )
            row = cursor.fetchone()
            return (row[0], row[1]) if row else None

        stamped = await backend_pre_migration.execute_read(_read)
        assert stamped == (get_settings().access_control.default_principal, 'private')

    @pytest.mark.asyncio
    async def test_visibility_check_constraint_enforced(
        self, backend_pre_migration: StorageBackend,
    ) -> None:
        """The added visibility column carries the CHECK constraint."""
        await apply_access_control_migration(backend_pre_migration)

        def _insert_bad(conn: sqlite3.Connection) -> None:
            conn.execute(
                'INSERT INTO context_entries '
                '(id, thread_id, source, content_type, text_content, owner_id, visibility) '
                "VALUES (?, 't1', 'agent', 'text', 'x', 'local', 'everyone')",
                (generate_id(),),
            )

        with pytest.raises(sqlite3.IntegrityError):
            await backend_pre_migration.execute_write(_insert_bad)

    @pytest.mark.asyncio
    async def test_migration_idempotent(self, backend_pre_migration: StorageBackend) -> None:
        """Re-applying the migration does not raise and leaves exactly one of each column."""
        await apply_access_control_migration(backend_pre_migration)
        await apply_access_control_migration(backend_pre_migration)

        columns = await _columns(backend_pre_migration)
        assert columns.count('owner_id') == 1
        assert columns.count('visibility') == 1

    @pytest.mark.asyncio
    async def test_migration_noop_on_full_schema(self, tmp_path: Path) -> None:
        """On a database already built from the full base schema, the migration
        is a safe no-op."""
        from app.schemas import load_schema

        db_path = tmp_path / 'test_access_control_full_schema.db'
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(load_schema('sqlite'))

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            assert 'owner_id' in await _columns(backend)
            assert await _table_exists(backend, 'context_entry_grants')
            await apply_access_control_migration(backend)
            columns = await _columns(backend)
            assert columns.count('owner_id') == 1
            assert columns.count('visibility') == 1
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_base_schema_reapplies_over_pre_migration_database(
        self, backend_pre_migration: StorageBackend,
    ) -> None:
        """The base schema script survives an existing pre-access-control database.

        Server startup executes the full base schema against whatever database is
        already there BEFORE the column migrations run, so nothing in that script
        may reference owner_id/visibility on context_entries: the CREATE TABLE is
        an IF NOT EXISTS no-op against the old table, and an index on the not-yet-
        added columns would crash initialization before the migration could add
        them. The access indexes must therefore arrive via the migration only.
        """
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')

        def _reapply(conn: sqlite3.Connection) -> None:
            conn.executescript(schema_sql)

        await backend_pre_migration.execute_write(_reapply)

        # The old table is untouched (no access columns yet)...
        assert 'owner_id' not in await _columns(backend_pre_migration)
        # ...and the subsequent migration completes the upgrade, indexes included.
        await apply_access_control_migration(backend_pre_migration)
        assert 'owner_id' in await _columns(backend_pre_migration)
        indexes = await _index_names(backend_pre_migration)
        assert 'idx_context_owner' in indexes
        assert 'idx_context_public' in indexes

    @pytest.mark.asyncio
    async def test_unsafe_default_principal_refused_before_ddl(
        self,
        backend_pre_migration: StorageBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A default principal outside the safe charset raises ConfigurationError
        BEFORE any DDL interpolation (the defense-in-depth re-check)."""
        import app.migrations.access_control as access_control_module

        unsafe_settings = SimpleNamespace(
            access_control=SimpleNamespace(default_principal="bad'principal"),
            storage=SimpleNamespace(postgresql_migration_timeout_s=300),
        )
        monkeypatch.setattr(access_control_module, 'settings', unsafe_settings)

        with pytest.raises(ConfigurationError):
            await apply_access_control_migration(backend_pre_migration)

        columns = await _columns(backend_pre_migration)
        assert 'owner_id' not in columns
