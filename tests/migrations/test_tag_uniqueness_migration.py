"""Tests for the tag-uniqueness migration.

Covers the in-place upgrade path: a database created BEFORE the write path
deduplicated tags can hold the same label twice for one entry, which every reader
returns verbatim and the tag statistics count twice. ``apply_tag_uniqueness_migration``
removes the duplicate rows and installs the unique index that prevents new ones.

Mirrors ``tests/migrations/test_version_migration.py``: a real temp SQLite backend
built from a hand-rolled pre-migration ``CREATE TABLE`` that OMITS the unique index,
seeded with duplicates, then migrated.

PostgreSQL coverage: there is no PostgreSQL migration-test fixture in this suite (the
version / content_hash / summary migration tests are SQLite-only too). The PostgreSQL
branch runs the SAME two statements under the schema-init advisory lock and rides on
the dual-backend real-server harness plus the live deploy-stack integration.
"""

import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Callable
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.backends import create_backend
from app.ids import generate_id
from app.migrations.tag_uniqueness import apply_tag_uniqueness_migration

# The tags table WITHOUT the unique index -- the shape of a database created before it
# was added to the base schema. Copied from app/schemas/sqlite_schema.sql minus the
# CREATE UNIQUE INDEX line.
_PRE_MIGRATION_DDL = (
    '''
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
    ''',
    '''
    CREATE TABLE tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        context_entry_id TEXT NOT NULL,
        tag TEXT NOT NULL,
        FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
    )
    ''',
    'CREATE INDEX idx_tags_entry ON tags(context_entry_id)',
    'CREATE INDEX idx_tags_tag ON tags(tag)',
)

_ENTRY_ID = generate_id()
_OTHER_ENTRY_ID = generate_id()


async def _tag_rows(backend: StorageBackend, context_entry_id: str) -> list[str]:
    """Return the stored tag labels for one entry, in insertion order.

    Args:
        backend: The backend to read from.
        context_entry_id: The entry whose tag rows to read.

    Returns:
        One element per stored ROW, so a duplicate row is visible.
    """

    def _read(conn: sqlite3.Connection) -> list[str]:
        cursor = conn.execute(
            'SELECT tag FROM tags WHERE context_entry_id = ? ORDER BY id',
            (context_entry_id,),
        )
        return [str(row[0]) for row in cursor.fetchall()]

    return await backend.execute_read(_read)


@pytest_asyncio.fixture
async def backend_with_duplicate_tags(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend whose tags table has no unique index and holds duplicate rows."""
    db_path = tmp_path / 'test_tag_uniqueness_pre_migration.db'

    with sqlite3.connect(str(db_path)) as conn:
        for statement in _PRE_MIGRATION_DDL:
            conn.execute(statement)
        for entry_id in (_ENTRY_ID, _OTHER_ENTRY_ID):
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                "VALUES (?, 'legacy-thread', 'agent', 'text', 'legacy entry')",
                (entry_id,),
            )
        conn.executemany(
            'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
            [
                (_ENTRY_ID, 'python'),
                (_ENTRY_ID, 'python'),
                (_ENTRY_ID, 'testing'),
                (_ENTRY_ID, 'python'),
                (_OTHER_ENTRY_ID, 'python'),
            ],
        )

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    try:
        yield backend
    finally:
        await backend.shutdown()


class TestTagUniquenessMigration:
    """The migration repairs stored duplicates and prevents new ones (SQLite)."""

    @pytest.mark.asyncio
    async def test_duplicates_present_before_migration(
        self, backend_with_duplicate_tags: StorageBackend,
    ) -> None:
        """Precondition: the pre-migration table really does hold repeated labels."""
        assert await _tag_rows(backend_with_duplicate_tags, _ENTRY_ID) == [
            'python', 'python', 'testing', 'python',
        ]

    @pytest.mark.asyncio
    async def test_migration_collapses_duplicate_rows(
        self, backend_with_duplicate_tags: StorageBackend,
    ) -> None:
        """Every repeated label collapses to a single row, and the others are untouched.

        A write-path fix alone never reaches these rows: they are already stored, and
        every reader keeps returning them verbatim.
        """
        await apply_tag_uniqueness_migration(backend_with_duplicate_tags)

        assert await _tag_rows(backend_with_duplicate_tags, _ENTRY_ID) == ['python', 'testing']
        # The same label on a DIFFERENT entry is a different pair and must survive.
        assert await _tag_rows(backend_with_duplicate_tags, _OTHER_ENTRY_ID) == ['python']

    @pytest.mark.asyncio
    async def test_migration_blocks_a_new_duplicate(
        self, backend_with_duplicate_tags: StorageBackend,
    ) -> None:
        """After the migration the database itself refuses a second identical row."""
        await apply_tag_uniqueness_migration(backend_with_duplicate_tags)

        def _insert_duplicate(conn: sqlite3.Connection) -> None:
            conn.execute(
                'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                (_ENTRY_ID, 'python'),
            )

        with pytest.raises(sqlite3.IntegrityError):
            await backend_with_duplicate_tags.execute_write(_insert_duplicate)

    @pytest.mark.asyncio
    async def test_migration_is_idempotent(
        self, backend_with_duplicate_tags: StorageBackend,
    ) -> None:
        """Rerunning changes nothing: nothing left to delete, index already present."""
        await apply_tag_uniqueness_migration(backend_with_duplicate_tags)
        await apply_tag_uniqueness_migration(backend_with_duplicate_tags)

        assert await _tag_rows(backend_with_duplicate_tags, _ENTRY_ID) == ['python', 'testing']

    @pytest.mark.asyncio
    async def test_rerun_skips_the_write_path_entirely(
        self,
        backend_with_duplicate_tags: StorageBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Once the unique index exists, a rerun issues no write at all.

        The index is the completion marker; without this skip every startup would
        rescan the whole tags table with the repair delete, which is quadratic on
        PostgreSQL for a large table.
        """
        await apply_tag_uniqueness_migration(backend_with_duplicate_tags)

        write_calls: list[Callable[[sqlite3.Connection], object]] = []
        original_execute_write = backend_with_duplicate_tags.execute_write

        async def _spying_execute_write(operation: Callable[[sqlite3.Connection], object]) -> object:
            write_calls.append(operation)
            return await original_execute_write(operation)

        monkeypatch.setattr(backend_with_duplicate_tags, 'execute_write', _spying_execute_write)

        await apply_tag_uniqueness_migration(backend_with_duplicate_tags)

        assert write_calls == []
