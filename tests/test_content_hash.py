"""Tests for content hash deduplication optimization.

Tests cover:
- compute_content_hash() utility function
- content_hash migration (idempotent, SQLite)
- Hash-based deduplication in store_with_deduplication()
- Hash-based pre-check in check_latest_is_duplicate()
- Hash recomputation in update_context_entry()
- Hash-based deduplication in store_contexts_batch()
- Backward compatibility with pre-migration rows (NULL hash fallback)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.backends import create_backend
from app.migrations.content_hash import apply_content_hash_migration
from app.repositories import RepositoryContainer
from app.repositories.context_repository import compute_content_hash

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def backend(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Create a StorageBackend with a test database (SQLite, full schema)."""
    db_path = tmp_path / 'test_hash.db'

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    conn.executescript(schema_sql)
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    yield backend

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=5.0)


@pytest_asyncio.fixture
async def repos(backend: StorageBackend) -> RepositoryContainer:
    """Create a RepositoryContainer with the test database."""
    return RepositoryContainer(backend)


@pytest_asyncio.fixture
async def backend_pre_migration(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Create a backend WITHOUT content_hash column (simulating pre-migration state)."""
    db_path = tmp_path / 'test_pre_migration.db'

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute('''
            CREATE TABLE context_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
                content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
                text_content TEXT,
                metadata JSON,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_thread_id ON context_entries(thread_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_source ON context_entries(source)')
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_thread_source ON context_entries(thread_id, source)',
        )
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_entry_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
            )
        ''')

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    yield backend

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=5.0)


@pytest_asyncio.fixture
async def repos_pre_migration(backend_pre_migration: StorageBackend) -> RepositoryContainer:
    """Create repos on a pre-migration database (no content_hash column)."""
    return RepositoryContainer(backend_pre_migration)


# ===========================================================================
# compute_content_hash tests
# ===========================================================================


class TestComputeContentHash:
    """Tests for the compute_content_hash utility function."""

    def test_consistent_results(self) -> None:
        """Same input always produces same hash."""
        text = 'Hello, World!'
        hash1 = compute_content_hash(text)
        hash2 = compute_content_hash(text)
        assert hash1 == hash2

    def test_different_inputs_produce_different_hashes(self) -> None:
        """Different inputs produce different hashes."""
        hash1 = compute_content_hash('text A')
        hash2 = compute_content_hash('text B')
        assert hash1 != hash2

    def test_returns_hex_string(self) -> None:
        """Hash is a 64-character hex string (SHA-256)."""
        result = compute_content_hash('test')
        assert len(result) == 64
        assert all(c in '0123456789abcdef' for c in result)

    def test_empty_string(self) -> None:
        """Empty string produces a valid hash."""
        result = compute_content_hash('')
        assert len(result) == 64
        # SHA-256 of empty string is well-known
        assert result == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'

    def test_unicode_content(self) -> None:
        """Unicode content is handled correctly."""
        hash1 = compute_content_hash('Привет мир')
        hash2 = compute_content_hash('Привет мир')
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_long_content(self) -> None:
        """Long content produces same-length hash."""
        long_text = 'A' * 100_000
        result = compute_content_hash(long_text)
        assert len(result) == 64


# ===========================================================================
# Migration tests
# ===========================================================================


class TestContentHashMigration:
    """Tests for apply_content_hash_migration."""

    @pytest.mark.asyncio
    async def test_migration_adds_column_sqlite(self, tmp_path: Path) -> None:
        """Migration adds content_hash column to context_entries (SQLite)."""
        db_path = tmp_path / 'test_migration.db'

        # Create schema WITHOUT content_hash
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute('''
                CREATE TABLE context_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    text_content TEXT,
                    metadata JSON,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            await apply_content_hash_migration(backend=backend)

            def _check(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.execute('PRAGMA table_info(context_entries)')
                return [row[1] for row in cursor.fetchall()]

            columns = await backend.execute_read(_check)
            assert 'content_hash' in columns
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_adds_index_sqlite(self, tmp_path: Path) -> None:
        """Migration creates the dedup composite index (SQLite)."""
        db_path = tmp_path / 'test_index.db'

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute('''
                CREATE TABLE context_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    text_content TEXT,
                    metadata JSON,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            await apply_content_hash_migration(backend=backend)

            def _check_index(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
                return [row[0] for row in cursor.fetchall()]

            indexes = await backend.execute_read(_check_index)
            assert 'idx_context_entries_dedup_hash' in indexes
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_idempotent_sqlite(self, tmp_path: Path) -> None:
        """Migration can be run twice without error."""
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
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            # Run twice -- should not raise
            await apply_content_hash_migration(backend=backend)
            await apply_content_hash_migration(backend=backend)

            def _check(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.execute('PRAGMA table_info(context_entries)')
                return [row[1] for row in cursor.fetchall()]

            columns = await backend.execute_read(_check)
            assert columns.count('content_hash') == 1
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_on_full_schema_sqlite(self, backend: StorageBackend) -> None:
        """Migration is safe on a database that already has content_hash (full schema)."""
        # Full schema already includes content_hash; migration should be a no-op
        await apply_content_hash_migration(backend=backend)

        def _check(conn: sqlite3.Connection) -> list[str]:
            cursor = conn.execute('PRAGMA table_info(context_entries)')
            return [row[1] for row in cursor.fetchall()]

        columns = await backend.execute_read(_check)
        assert 'content_hash' in columns


# ===========================================================================
# Hash stored on insert
# ===========================================================================


@pytest.mark.asyncio
class TestHashStoredOnInsert:
    """Verify that store_with_deduplication stores the hash alongside text_content."""

    async def test_hash_stored_on_new_entry(
        self, backend: StorageBackend, repos: RepositoryContainer,
    ) -> None:
        """New entries have content_hash populated."""
        text = 'Hello test content'
        context_id, was_updated = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content=text, metadata=None,
        )
        assert was_updated is False

        # Read content_hash directly from DB
        def _read_hash(conn: sqlite3.Connection) -> str | None:
            cursor = conn.execute(
                'SELECT content_hash FROM context_entries WHERE id = ?', (context_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

        stored_hash = await backend.execute_read(_read_hash)
        assert stored_hash == compute_content_hash(text)

    async def test_hash_updated_on_dedup(
        self, backend: StorageBackend, repos: RepositoryContainer,
    ) -> None:
        """When dedup fires, content_hash is refreshed (same value since text matches)."""
        text = 'Duplicate content'
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content=text, metadata=None,
        )
        # Store duplicate
        context_id2, was_updated = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content=text, metadata=None,
        )
        assert was_updated is True
        assert context_id2 == context_id

        def _read_hash(conn: sqlite3.Connection) -> str | None:
            cursor = conn.execute(
                'SELECT content_hash FROM context_entries WHERE id = ?', (context_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

        stored_hash = await backend.execute_read(_read_hash)
        assert stored_hash == compute_content_hash(text)


# ===========================================================================
# Hash-based dedup in check_latest_is_duplicate
# ===========================================================================


@pytest.mark.asyncio
class TestHashBasedPreCheck:
    """Tests for hash-based dedup in check_latest_is_duplicate."""

    async def test_hash_match_returns_id(self, repos: RepositoryContainer) -> None:
        """When content_hash matches, returns existing entry id."""
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Same text', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='t1', source='user', text_content='Same text',
        )
        assert result == context_id

    async def test_hash_mismatch_returns_none(self, repos: RepositoryContainer) -> None:
        """When content_hash does not match, returns None."""
        await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Original', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='t1', source='user', text_content='Different',
        )
        assert result is None

    async def test_empty_thread_returns_none(self, repos: RepositoryContainer) -> None:
        """Empty thread returns None."""
        result = await repos.context.check_latest_is_duplicate(
            thread_id='nonexistent', source='user', text_content='Any',
        )
        assert result is None


# ===========================================================================
# Backward compatibility: NULL hash fallback
# ===========================================================================


@pytest.mark.asyncio
class TestNullHashFallback:
    """Tests for backward compatibility with pre-migration rows (NULL content_hash)."""

    async def test_pre_migration_store_dedup_uses_text_fallback(
        self, backend_pre_migration: StorageBackend, repos_pre_migration: RepositoryContainer,
    ) -> None:
        """Pre-migration rows without content_hash still deduplicate via text comparison."""
        # Apply the migration so the column exists (needed for INSERT to include content_hash)
        await apply_content_hash_migration(backend=backend_pre_migration)

        # Manually insert a row WITHOUT content_hash (simulating pre-migration data)
        def _insert_without_hash(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "INSERT INTO context_entries (thread_id, source, content_type, text_content) "
                "VALUES ('t1', 'user', 'text', 'Legacy text')",
            )
            return cursor.lastrowid or 0

        legacy_id = await backend_pre_migration.execute_write(_insert_without_hash)
        assert legacy_id > 0

        # Verify the row has NULL content_hash
        def _check_null(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                'SELECT content_hash FROM context_entries WHERE id = ?', (legacy_id,),
            )
            row = cursor.fetchone()
            return row[0] is None

        assert await backend_pre_migration.execute_read(_check_null) is True

        # store_with_deduplication with same text should detect duplicate via fallback
        context_id, was_updated = await repos_pre_migration.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Legacy text', metadata=None,
        )
        assert was_updated is True
        assert context_id == legacy_id

    async def test_pre_migration_check_latest_uses_text_fallback(
        self, backend_pre_migration: StorageBackend, repos_pre_migration: RepositoryContainer,
    ) -> None:
        """check_latest_is_duplicate falls back to text comparison for NULL hash rows."""
        await apply_content_hash_migration(backend=backend_pre_migration)

        # Insert row without hash
        def _insert_without_hash(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "INSERT INTO context_entries (thread_id, source, content_type, text_content) "
                "VALUES ('t1', 'agent', 'text', 'Old agent text')",
            )
            return cursor.lastrowid or 0

        legacy_id = await backend_pre_migration.execute_write(_insert_without_hash)

        # check_latest_is_duplicate should find the match via text fallback
        result = await repos_pre_migration.context.check_latest_is_duplicate(
            thread_id='t1', source='agent', text_content='Old agent text',
        )
        assert result == legacy_id

    async def test_pre_migration_different_text_returns_none(
        self, backend_pre_migration: StorageBackend, repos_pre_migration: RepositoryContainer,
    ) -> None:
        """NULL hash row with different text correctly returns None."""
        await apply_content_hash_migration(backend=backend_pre_migration)

        def _insert_without_hash(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO context_entries (thread_id, source, content_type, text_content) "
                "VALUES ('t1', 'user', 'text', 'Original text')",
            )

        await backend_pre_migration.execute_write(_insert_without_hash)

        result = await repos_pre_migration.context.check_latest_is_duplicate(
            thread_id='t1', source='user', text_content='Different text',
        )
        assert result is None


# ===========================================================================
# Hash recomputation in update_context_entry
# ===========================================================================


@pytest.mark.asyncio
class TestHashRecomputationOnUpdate:
    """Tests for content_hash recomputation when text_content is updated."""

    async def test_hash_updated_when_text_changes(
        self, backend: StorageBackend, repos: RepositoryContainer,
    ) -> None:
        """Updating text_content recomputes the content_hash."""
        original_text = 'Original content'
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content=original_text, metadata=None,
        )

        new_text = 'Updated content'
        success, fields = await repos.context.update_context_entry(
            context_id=context_id, text_content=new_text,
        )
        assert success is True
        assert 'text_content' in fields

        # Verify hash was recomputed
        def _read_hash(conn: sqlite3.Connection) -> str | None:
            cursor = conn.execute(
                'SELECT content_hash FROM context_entries WHERE id = ?', (context_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

        stored_hash = await backend.execute_read(_read_hash)
        assert stored_hash == compute_content_hash(new_text)
        assert stored_hash != compute_content_hash(original_text)

    async def test_hash_unchanged_when_only_metadata_updated(
        self, backend: StorageBackend, repos: RepositoryContainer,
    ) -> None:
        """Updating only metadata does NOT change content_hash."""
        text = 'Stable content'
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='t1', source='agent', content_type='text',
            text_content=text, metadata=None,
        )

        expected_hash = compute_content_hash(text)

        success, fields = await repos.context.update_context_entry(
            context_id=context_id, metadata=json.dumps({'key': 'value'}),
        )
        assert success is True
        assert 'metadata' in fields
        assert 'text_content' not in fields

        def _read_hash(conn: sqlite3.Connection) -> str | None:
            cursor = conn.execute(
                'SELECT content_hash FROM context_entries WHERE id = ?', (context_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

        stored_hash = await backend.execute_read(_read_hash)
        assert stored_hash == expected_hash

    async def test_dedup_works_after_text_update(
        self, repos: RepositoryContainer,
    ) -> None:
        """After updating text_content, dedup correctly uses the new hash."""
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Version 1', metadata=None,
        )

        # Update text
        await repos.context.update_context_entry(
            context_id=context_id, text_content='Version 2',
        )

        # Dedup check for new text should match
        result = await repos.context.check_latest_is_duplicate(
            thread_id='t1', source='user', text_content='Version 2',
        )
        assert result == context_id

        # Dedup check for old text should NOT match
        result_old = await repos.context.check_latest_is_duplicate(
            thread_id='t1', source='user', text_content='Version 1',
        )
        assert result_old is None


# ===========================================================================
# Hash-based dedup in store_contexts_batch
# ===========================================================================


@pytest.mark.asyncio
class TestBatchHashDedup:
    """Tests for hash-based deduplication in store_contexts_batch."""

    async def test_batch_stores_hash(
        self, backend: StorageBackend, repos: RepositoryContainer,
    ) -> None:
        """Batch insert stores content_hash for each entry."""
        entries = [
            {
                'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                'text_content': 'Entry one', 'metadata': None,
            },
            {
                'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                'text_content': 'Entry two', 'metadata': None,
            },
        ]
        results = await repos.context.store_contexts_batch(entries)
        assert len(results) == 2

        # Return type is list[tuple[int, int | None, str | None]]
        # (index, context_id, error)
        expected_texts = ['Entry one', 'Entry two']
        for i in range(len(entries)):
            _idx, context_id, error = results[i]
            assert error is None
            assert context_id is not None
            cid = context_id  # local binding for closure

            def _read_hash(conn: sqlite3.Connection, _cid: int = cid) -> str | None:
                cursor = conn.execute(
                    'SELECT content_hash FROM context_entries WHERE id = ?', (_cid,),
                )
                row = cursor.fetchone()
                return row[0] if row else None

            stored_hash = await backend.execute_read(_read_hash)
            assert stored_hash == compute_content_hash(expected_texts[i])

    async def test_batch_dedup_uses_hash(
        self, repos: RepositoryContainer,
    ) -> None:
        """Batch dedup detects duplicates via hash comparison."""
        # Pre-store an entry
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Existing text', metadata=None,
        )

        # Batch with duplicate
        entries = [
            {
                'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                'text_content': 'Existing text', 'metadata': None,
            },
        ]
        results = await repos.context.store_contexts_batch(entries)
        assert len(results) == 1
        # Return: (index, context_id, error) -- same ID means dedup occurred
        _idx, batch_context_id, error = results[0]
        assert error is None
        assert batch_context_id == context_id

    async def test_batch_non_duplicate_inserts(
        self, repos: RepositoryContainer,
    ) -> None:
        """Batch correctly inserts non-duplicate entries."""
        # Pre-store an entry
        original_id, _ = await repos.context.store_with_deduplication(
            thread_id='t1', source='user', content_type='text',
            text_content='Original text', metadata=None,
        )

        # Batch with different text
        entries = [
            {
                'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                'text_content': 'New text', 'metadata': None,
            },
        ]
        results = await repos.context.store_contexts_batch(entries)
        assert len(results) == 1
        _idx, batch_context_id, error = results[0]
        assert error is None
        # New entry should have a different (higher) ID
        assert batch_context_id is not None
        assert batch_context_id != original_id


# ===========================================================================
# content_hash NOT in CONTEXT_ENTRY_COLUMNS (API boundary)
# ===========================================================================


class TestContentHashNotExposed:
    """Verify content_hash is internal-only and not exposed in API responses."""

    def test_context_entry_columns_excludes_content_hash(self) -> None:
        """CONTEXT_ENTRY_COLUMNS should NOT include content_hash."""
        from app.repositories.context_repository import CONTEXT_ENTRY_COLUMNS
        assert 'content_hash' not in CONTEXT_ENTRY_COLUMNS
