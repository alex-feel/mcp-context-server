"""Tests for summary column migration and model updates.

This module tests:
- Summary column migration (idempotent, SQLite)
- ContextEntry model with summary field
- ContextEntryDict with summary field
- CONTEXT_ENTRY_COLUMNS includes summary
- store_with_deduplication with summary parameter
- update_context_entry with summary parameter
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.backends import StorageBackend
from app.backends import create_backend
from app.migrations.summary import apply_summary_migration
from app.repositories.context_repository import CONTEXT_ENTRY_COLUMNS


class TestApplySummaryMigration:
    """Test the apply_summary_migration function."""

    @pytest.mark.asyncio
    async def test_migration_adds_summary_column_sqlite(self, tmp_path: Path) -> None:
        """Test migration adds summary column to context_entries (SQLite)."""
        db_path = tmp_path / 'test_summary.db'

        # Create base schema WITHOUT summary column (simulating pre-migration state)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute('''
                CREATE TABLE context_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    text_content TEXT,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            await apply_summary_migration(backend=backend)

            # Verify column exists
            def _check(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.execute('PRAGMA table_info(context_entries)')
                return [row[1] for row in cursor.fetchall()]

            columns = await backend.execute_read(_check)
            assert 'summary' in columns
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_idempotent_sqlite(self, tmp_path: Path) -> None:
        """Test migration can be run twice without error."""
        db_path = tmp_path / 'test_idempotent.db'

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute('''
                CREATE TABLE context_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    text_content TEXT,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            # Run migration twice -- no exception should be raised
            await apply_summary_migration(backend=backend)
            await apply_summary_migration(backend=backend)

            def _check(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.execute('PRAGMA table_info(context_entries)')
                return [row[1] for row in cursor.fetchall()]

            columns = await backend.execute_read(_check)
            assert 'summary' in columns
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_summary_column_nullable(self, tmp_path: Path) -> None:
        """Test summary column accepts NULL values."""
        db_path = tmp_path / 'test_nullable.db'

        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            # Insert entry without summary
            def _insert(conn: sqlite3.Connection) -> int:
                cursor = conn.execute(
                    "INSERT INTO context_entries (thread_id, source, content_type, text_content) "
                    "VALUES ('t1', 'user', 'text', 'hello')",
                )
                return cursor.lastrowid or 0

            entry_id = await backend.execute_write(_insert)
            assert entry_id > 0

            # Verify summary is NULL
            def _check(conn: sqlite3.Connection) -> object:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT summary FROM context_entries WHERE id = ?', (entry_id,),
                )
                row = cursor.fetchone()
                return row['summary'] if row else 'NOT_FOUND'

            summary_value = await backend.execute_read(_check)
            assert summary_value is None
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_summary_column_accepts_text(self, tmp_path: Path) -> None:
        """Test summary column stores text correctly."""
        db_path = tmp_path / 'test_text.db'

        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            expected_summary = 'This is a test summary of the context entry.'

            def _insert(conn: sqlite3.Connection) -> int:
                cursor = conn.execute(
                    "INSERT INTO context_entries (thread_id, source, content_type, text_content, summary) "
                    "VALUES ('t1', 'user', 'text', 'hello world', ?)",
                    (expected_summary,),
                )
                return cursor.lastrowid or 0

            entry_id = await backend.execute_write(_insert)

            def _check(conn: sqlite3.Connection) -> str | None:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT summary FROM context_entries WHERE id = ?', (entry_id,),
                )
                row = cursor.fetchone()
                return row['summary'] if row else None

            actual_summary = await backend.execute_read(_check)
            assert actual_summary == expected_summary
        finally:
            await backend.shutdown()

    def test_new_database_has_summary_column(self, test_db: sqlite3.Connection) -> None:
        """Test fresh database schema includes summary column."""
        cursor = test_db.execute('PRAGMA table_info(context_entries)')
        columns = [row[1] for row in cursor.fetchall()]
        assert 'summary' in columns


class TestContextEntryColumnsConstant:
    """Test CONTEXT_ENTRY_COLUMNS includes summary."""

    def test_summary_in_columns(self) -> None:
        """Test that CONTEXT_ENTRY_COLUMNS includes summary."""
        assert 'summary' in CONTEXT_ENTRY_COLUMNS

    def test_summary_position_after_metadata(self) -> None:
        """Test that summary appears after metadata and before created_at."""
        columns = [c.strip() for c in CONTEXT_ENTRY_COLUMNS.split(',')]
        metadata_idx = columns.index('metadata')
        summary_idx = columns.index('summary')
        created_at_idx = columns.index('created_at')
        assert metadata_idx < summary_idx < created_at_idx


class TestContextEntryModel:
    """Test ContextEntry model with summary field."""

    def test_summary_field_defaults_to_none(self) -> None:
        """Test summary field defaults to None."""
        from app.models import ContextEntry

        entry = ContextEntry(
            thread_id='t1',
            source='user',
            text_content='hello',
        )
        assert entry.summary is None

    def test_summary_field_accepts_string(self) -> None:
        """Test summary field accepts a string value."""
        from app.models import ContextEntry

        entry = ContextEntry(
            thread_id='t1',
            source='user',
            text_content='hello',
            summary='A brief summary.',
        )
        assert entry.summary == 'A brief summary.'


class TestContextEntryDict:
    """Test ContextEntryDict includes summary field."""

    def test_summary_key_in_typed_dict(self) -> None:
        """Test ContextEntryDict has summary field."""
        from app.types import ContextEntryDict

        # ContextEntryDict with total=False means all keys are optional
        # Verify summary is in the annotations
        assert 'summary' in ContextEntryDict.__annotations__


class TestStoreWithDeduplicationSummary:
    """Test store_with_deduplication with summary parameter."""

    @pytest.mark.asyncio
    async def test_store_new_entry_with_summary(self, async_db_initialized: StorageBackend) -> None:
        """Test storing a new entry with summary."""
        _backend = async_db_initialized  # Fixture needed for DB initialization side effect
        from app.startup import ensure_repositories

        repos = await ensure_repositories()
        context_id, was_updated = await repos.context.store_with_deduplication(
            thread_id='test-summary-store',
            source='agent',
            content_type='text',
            text_content='Test content for summary',
            summary='Summary of test content',
        )
        assert context_id > 0
        assert was_updated is False

        # Verify summary was stored
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1
        assert entries[0]['summary'] == 'Summary of test content'

    @pytest.mark.asyncio
    async def test_store_new_entry_without_summary(self, async_db_initialized: StorageBackend) -> None:
        """Test storing a new entry without summary (None)."""
        _backend = async_db_initialized  # Fixture needed for DB initialization side effect
        from app.startup import ensure_repositories

        repos = await ensure_repositories()
        context_id, was_updated = await repos.context.store_with_deduplication(
            thread_id='test-summary-none',
            source='user',
            content_type='text',
            text_content='Content without summary',
        )
        assert context_id > 0
        assert was_updated is False

        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1
        assert entries[0]['summary'] is None

    @pytest.mark.asyncio
    async def test_dedup_preserves_summary_when_none(self, async_db_initialized: StorageBackend) -> None:
        """Test deduplication preserves existing summary when new summary is None."""
        _backend = async_db_initialized  # Fixture needed for DB initialization side effect
        from app.startup import ensure_repositories

        repos = await ensure_repositories()

        # Store entry with summary
        context_id, _was_updated = await repos.context.store_with_deduplication(
            thread_id='test-summary-dedup',
            source='agent',
            content_type='text',
            text_content='Dedup test content',
            summary='Original summary',
        )

        # Store same text (triggers dedup) without summary
        dedup_id, was_updated = await repos.context.store_with_deduplication(
            thread_id='test-summary-dedup',
            source='agent',
            content_type='text',
            text_content='Dedup test content',
            summary=None,
        )

        assert dedup_id == context_id
        assert was_updated is True

        # Verify original summary is preserved via COALESCE
        entries = await repos.context.get_by_ids([context_id])
        assert entries[0]['summary'] == 'Original summary'

    @pytest.mark.asyncio
    async def test_dedup_updates_summary_when_provided(self, async_db_initialized: StorageBackend) -> None:
        """Test deduplication updates summary when new summary is provided."""
        _backend = async_db_initialized  # Fixture needed for DB initialization side effect
        from app.startup import ensure_repositories

        repos = await ensure_repositories()

        # Store entry with summary
        context_id, _was_updated = await repos.context.store_with_deduplication(
            thread_id='test-summary-dedup-update',
            source='agent',
            content_type='text',
            text_content='Dedup update test',
            summary='Old summary',
        )

        # Store same text with new summary
        dedup_id, was_updated = await repos.context.store_with_deduplication(
            thread_id='test-summary-dedup-update',
            source='agent',
            content_type='text',
            text_content='Dedup update test',
            summary='New summary',
        )

        assert dedup_id == context_id
        assert was_updated is True

        entries = await repos.context.get_by_ids([context_id])
        assert entries[0]['summary'] == 'New summary'


class TestUpdateContextEntrySummary:
    """Test update_context_entry with summary parameter."""

    @pytest.mark.asyncio
    async def test_update_summary_only(self, async_db_initialized: StorageBackend) -> None:
        """Test updating only the summary field."""
        _backend = async_db_initialized  # Fixture needed for DB initialization side effect
        from app.startup import ensure_repositories

        repos = await ensure_repositories()

        # Store entry without summary
        context_id, _was_updated = await repos.context.store_with_deduplication(
            thread_id='test-update-summary',
            source='user',
            content_type='text',
            text_content='Content to update',
        )

        # Update only summary
        success, updated_fields = await repos.context.update_context_entry(
            context_id=context_id,
            summary='Generated summary',
        )

        assert success is True
        assert 'summary' in updated_fields

        entries = await repos.context.get_by_ids([context_id])
        assert entries[0]['summary'] == 'Generated summary'

    @pytest.mark.asyncio
    async def test_update_text_and_summary(self, async_db_initialized: StorageBackend) -> None:
        """Test updating both text_content and summary."""
        _backend = async_db_initialized  # Fixture needed for DB initialization side effect
        from app.startup import ensure_repositories

        repos = await ensure_repositories()

        context_id, _was_updated = await repos.context.store_with_deduplication(
            thread_id='test-update-both',
            source='agent',
            content_type='text',
            text_content='Original content',
        )

        success, updated_fields = await repos.context.update_context_entry(
            context_id=context_id,
            text_content='Updated content',
            summary='Updated summary',
        )

        assert success is True
        assert 'text_content' in updated_fields
        assert 'summary' in updated_fields

        entries = await repos.context.get_by_ids([context_id])
        assert entries[0]['text_content'] == 'Updated content'
        assert entries[0]['summary'] == 'Updated summary'

    @pytest.mark.asyncio
    async def test_update_without_summary_preserves_existing(self, async_db_initialized: StorageBackend) -> None:
        """Test that updating other fields does not clear existing summary."""
        _backend = async_db_initialized  # Fixture needed for DB initialization side effect
        from app.startup import ensure_repositories

        repos = await ensure_repositories()

        # Store with summary
        context_id, _was_updated = await repos.context.store_with_deduplication(
            thread_id='test-preserve-summary',
            source='user',
            content_type='text',
            text_content='Content with summary',
            summary='Existing summary',
        )

        # Update only metadata (not summary)
        import json
        success, _updated = await repos.context.update_context_entry(
            context_id=context_id,
            metadata=json.dumps({'key': 'value'}),
        )

        assert success is True

        entries = await repos.context.get_by_ids([context_id])
        assert entries[0]['summary'] == 'Existing summary'
