"""Tests for the optimistic-concurrency ``version`` column migration.

Covers the in-place upgrade path: a database created BEFORE ``version`` was added
to the base schema gains the column via ``apply_version_migration``. Mirrors
``tests/migrations/test_content_hash.py`` and ``tests/migrations/test_summary_migration.py``:
a real temp SQLite backend built from a hand-rolled ``CREATE TABLE`` that OMITS the
``version`` column (the pre-migration shape), then the migration adds it,
backfills the ``NOT NULL DEFAULT 0`` value on existing rows, and is idempotent.

PostgreSQL coverage: there is no PostgreSQL migration-test fixture in this suite
(the content_hash / summary migration tests are SQLite-only too). The PostgreSQL
branch of ``apply_version_migration`` (``ADD COLUMN IF NOT EXISTS ... BIGINT NOT
NULL DEFAULT 0`` under the schema-init advisory lock) rides on the dual-backend
real-server harness (``tests/integration/_harness.py``, run against PostgreSQL via
the ``@requires_docker_postgres`` entry point) plus the live deploy-stack
integration, which exercise update_context / update_context_batch against a
PostgreSQL backend whose schema includes ``version``. A dedicated PostgreSQL
fixture is intentionally NOT invented here.
"""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.backends import create_backend
from app.ids import generate_id
from app.migrations.version import apply_version_migration

# The current context_entries columns MINUS ``version`` -- the pre-migration shape
# of a database created before the column was added to the base schema. Copied
# from app/schemas/sqlite_schema.sql (rowid_int, id, thread_id, source,
# content_type, text_content, metadata, summary, content_hash, created_at,
# updated_at) with the ``version INTEGER NOT NULL DEFAULT 0`` line removed.
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
'''


async def _columns(backend: StorageBackend) -> list[str]:
    def _check(conn: sqlite3.Connection) -> list[str]:
        cursor = conn.execute('PRAGMA table_info(context_entries)')
        return [row[1] for row in cursor.fetchall()]

    return await backend.execute_read(_check)


@pytest_asyncio.fixture
async def backend_pre_migration(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend whose context_entries table OMITS the ``version`` column."""
    db_path = tmp_path / 'test_version_pre_migration.db'

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(_PRE_MIGRATION_CONTEXT_ENTRIES_DDL)
        conn.execute('CREATE INDEX IF NOT EXISTS idx_thread_id ON context_entries(thread_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_source ON context_entries(source)')
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_thread_source ON context_entries(thread_id, source)',
        )

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    try:
        yield backend
    finally:
        await backend.shutdown()


class TestVersionMigration:
    """apply_version_migration adds the version column on an in-place upgrade (SQLite)."""

    @pytest.mark.asyncio
    async def test_column_absent_before_migration(self, backend_pre_migration: StorageBackend) -> None:
        """Precondition: the pre-migration table has NO version column."""
        columns = await _columns(backend_pre_migration)
        assert 'version' not in columns

    @pytest.mark.asyncio
    async def test_migration_adds_version_column(self, backend_pre_migration: StorageBackend) -> None:
        """The migration adds the version column to context_entries (SQLite)."""
        await apply_version_migration(backend_pre_migration)
        columns = await _columns(backend_pre_migration)
        assert 'version' in columns

    @pytest.mark.asyncio
    async def test_existing_row_backfills_default_zero(self, backend_pre_migration: StorageBackend) -> None:
        """A row inserted after the migration reads version 0 (the NOT NULL DEFAULT)."""
        await apply_version_migration(backend_pre_migration)

        new_id = generate_id()

        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES (?, 't1', 'agent', 'text', 'hello')",
                (new_id,),
            )

        await backend_pre_migration.execute_write(_insert)

        def _read_version(conn: sqlite3.Connection) -> int | None:
            cursor = conn.execute(
                'SELECT version FROM context_entries WHERE id = ?', (new_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

        assert await backend_pre_migration.execute_read(_read_version) == 0

    @pytest.mark.asyncio
    async def test_preexisting_row_backfills_default_zero(self, backend_pre_migration: StorageBackend) -> None:
        """A row present BEFORE the migration is backfilled to version 0.

        ``ADD COLUMN ... NOT NULL DEFAULT 0`` populates existing rows with the
        default, so a legacy row gains a valid optimistic-concurrency token rather
        than a NULL that would break the ``WHERE id = ? AND version = ?`` CAS.
        """
        legacy_id = generate_id()

        def _insert_legacy(conn: sqlite3.Connection) -> None:
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES (?, 't1', 'user', 'text', 'legacy row')",
                (legacy_id,),
            )

        await backend_pre_migration.execute_write(_insert_legacy)

        await apply_version_migration(backend_pre_migration)

        def _read_version(conn: sqlite3.Connection) -> int | None:
            cursor = conn.execute(
                'SELECT version FROM context_entries WHERE id = ?', (legacy_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

        assert await backend_pre_migration.execute_read(_read_version) == 0

    @pytest.mark.asyncio
    async def test_migration_idempotent(self, backend_pre_migration: StorageBackend) -> None:
        """Re-applying the migration does not raise and leaves exactly one version column."""
        await apply_version_migration(backend_pre_migration)
        # Second application must be a no-op (the PRAGMA guard finds the column).
        await apply_version_migration(backend_pre_migration)

        columns = await _columns(backend_pre_migration)
        assert columns.count('version') == 1

    @pytest.mark.asyncio
    async def test_migration_noop_on_full_schema(self, tmp_path: Path) -> None:
        """On a database already built from the full base schema (which includes
        version), the migration is a safe no-op."""
        from app.schemas import load_schema

        db_path = tmp_path / 'test_version_full_schema.db'
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(load_schema('sqlite'))

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            # Precondition: the full schema already has the column.
            assert 'version' in await _columns(backend)
            await apply_version_migration(backend)
            columns = await _columns(backend)
            assert columns.count('version') == 1
        finally:
            await backend.shutdown()
