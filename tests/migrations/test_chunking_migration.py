"""Tests for apply_chunking_migration() on SQLite.

Covers the embedding_chunks mapping table (one context entry, many chunk
embeddings), the chunk_count column on embedding_metadata, and the
generation-off skip.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.ids import generate_id


class TestApplyChunkingMigration:
    """Tests for apply_chunking_migration()."""

    @pytest.mark.asyncio
    async def test_migration_skipped_when_embedding_generation_disabled(self) -> None:
        """Verify no-op when ENABLE_EMBEDDING_GENERATION=false on a FRESH database.

        The chunking migration provisions the 1:N embedding-storage layer from
        embedding GENERATION (settings.embedding.generation_enabled), not from the
        semantic-search TOOL toggle. With generation off it skips only when the
        database carries no embedding infrastructure (the infra-present
        fallthrough keeps an existing layout maintained); the only read allowed
        is that probe.
        """
        from unittest.mock import AsyncMock

        from app.migrations import apply_chunking_migration

        # Create mock backend; the infra probe reports a fresh database.
        mock_backend = MagicMock()
        mock_backend.backend_type = 'sqlite'
        mock_backend.execute_read = AsyncMock(return_value=False)

        # Mock settings with embedding generation disabled
        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = False

        with patch('app.migrations.chunking.settings', mock_settings):
            # Call should return early without doing anything
            await apply_chunking_migration(backend=mock_backend)

            # No writes; the single read is the embedding-infra probe.
            mock_backend.execute_write.assert_not_called()
            mock_backend.execute_read.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_migration_creates_embedding_chunks_table_sqlite(self, tmp_path: Path) -> None:
        """Verify embedding_chunks table created for SQLite."""
        db_path = tmp_path / 'test_chunking.db'

        # Create base schema first
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            # Create embedding_metadata table (prerequisite for chunking migration)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id INTEGER NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                )
            ''')
            conn.commit()

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'STORAGE_BACKEND': 'sqlite',
        }

        import os

        with patch.dict(os.environ, env, clear=False):
            from app.backends.sqlite_backend import SQLiteBackend

            backend = SQLiteBackend(db_path=str(db_path))
            await backend.initialize()

            try:
                from app.migrations import apply_chunking_migration

                await apply_chunking_migration(backend=backend)

                # Verify embedding_chunks table exists
                def _check_tables(conn: sqlite3.Connection) -> tuple[bool, bool, bool]:
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_chunks'",
                    )
                    chunks_table_exists = cursor.fetchone() is not None

                    # Check for indexes
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_embedding_chunks_context'",
                    )
                    context_index_exists = cursor.fetchone() is not None

                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_embedding_chunks_vec_rowid'",
                    )
                    vec_rowid_index_exists = cursor.fetchone() is not None

                    return chunks_table_exists, context_index_exists, vec_rowid_index_exists

                chunks_exists, ctx_idx, vec_idx = await backend.execute_read(_check_tables)
                assert chunks_exists, 'embedding_chunks table should exist'
                assert ctx_idx, 'idx_embedding_chunks_context index should exist'
                assert vec_idx, 'idx_embedding_chunks_vec_rowid index should exist'

            finally:
                await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_adds_chunk_count_column_sqlite(self, tmp_path: Path) -> None:
        """Verify chunk_count column added to embedding_metadata for SQLite."""
        db_path = tmp_path / 'test_chunk_count.db'

        # Create base schema with embedding_metadata (without chunk_count)
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id INTEGER NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                )
            ''')
            conn.commit()

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'STORAGE_BACKEND': 'sqlite',
        }

        import os

        with patch.dict(os.environ, env, clear=False):
            from app.backends.sqlite_backend import SQLiteBackend

            backend = SQLiteBackend(db_path=str(db_path))
            await backend.initialize()

            try:
                from app.migrations import apply_chunking_migration

                await apply_chunking_migration(backend=backend)

                # Verify chunk_count column exists
                def _check_chunk_count_column(conn: sqlite3.Connection) -> bool:
                    cursor = conn.execute('PRAGMA table_info(embedding_metadata)')
                    columns = [row[1] for row in cursor.fetchall()]
                    return 'chunk_count' in columns

                has_chunk_count = await backend.execute_read(_check_chunk_count_column)
                assert has_chunk_count, 'chunk_count column should exist in embedding_metadata'

            finally:
                await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_idempotent_sqlite(self, tmp_path: Path) -> None:
        """Verify running migration twice does not fail for SQLite."""
        db_path = tmp_path / 'test_idempotent.db'

        # Create base schema
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id INTEGER NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                )
            ''')
            conn.commit()

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'STORAGE_BACKEND': 'sqlite',
        }

        import os

        with patch.dict(os.environ, env, clear=False):
            from app.backends.sqlite_backend import SQLiteBackend

            backend = SQLiteBackend(db_path=str(db_path))
            await backend.initialize()

            try:
                from app.migrations import apply_chunking_migration

                # First run
                await apply_chunking_migration(backend=backend)

                # Second run should not fail
                await apply_chunking_migration(backend=backend)

                # Verify table still exists
                def _check_table(conn: sqlite3.Connection) -> bool:
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_chunks'",
                    )
                    return cursor.fetchone() is not None

                table_exists = await backend.execute_read(_check_table)
                assert table_exists, 'embedding_chunks table should still exist after second migration'

            finally:
                await backend.shutdown()

    @pytest.mark.asyncio
    async def test_no_chunk_index_column_sqlite(self, tmp_path: Path) -> None:
        """CRITICAL: Verify NO chunk_index column exists (user decision)."""
        db_path = tmp_path / 'test_no_chunk_index.db'

        # Create base schema
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id INTEGER NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                )
            ''')
            conn.commit()

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'STORAGE_BACKEND': 'sqlite',
        }

        import os

        with patch.dict(os.environ, env, clear=False):
            from app.backends.sqlite_backend import SQLiteBackend

            backend = SQLiteBackend(db_path=str(db_path))
            await backend.initialize()

            try:
                from app.migrations import apply_chunking_migration

                await apply_chunking_migration(backend=backend)

                # Verify NO chunk_index column in embedding_chunks
                def _check_no_chunk_index(conn: sqlite3.Connection) -> list[str]:
                    cursor = conn.execute('PRAGMA table_info(embedding_chunks)')
                    return [row[1] for row in cursor.fetchall()]

                columns = await backend.execute_read(_check_no_chunk_index)
                assert 'chunk_index' not in columns, 'chunk_index column should NOT exist'
                assert 'chunk_start' not in columns, 'chunk_start column should NOT exist'
                assert 'chunk_end' not in columns, 'chunk_end column should NOT exist'

                # Verify expected columns exist
                assert 'id' in columns, 'id column should exist'
                assert 'context_id' in columns, 'context_id column should exist'
                assert 'vec_rowid' in columns, 'vec_rowid column should exist'
                assert 'created_at' in columns, 'created_at column should exist'

            finally:
                await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_adds_boundary_columns_to_existing_table_sqlite(self, tmp_path: Path) -> None:
        """Verify boundary columns added to existing embedding_chunks table without them.

        This test simulates an upgrade from a schema whose embedding_chunks table
        has no start_index and end_index columns.
        """
        db_path = tmp_path / 'test_upgrade.db'

        # Create base schema with OLD embedding_chunks (no boundary columns)
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            # Create embedding_metadata (prerequisite)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id INTEGER NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                )
            ''')
            # Create OLD embedding_chunks WITHOUT boundary columns
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embedding_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id INTEGER NOT NULL,
                    vec_rowid INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                )
            ''')
            conn.commit()

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'STORAGE_BACKEND': 'sqlite',
        }

        import os

        with patch.dict(os.environ, env, clear=False):
            from app.backends.sqlite_backend import SQLiteBackend

            backend = SQLiteBackend(db_path=str(db_path))
            await backend.initialize()

            try:
                from app.migrations import apply_chunking_migration

                # Should NOT raise - should add missing columns gracefully
                await apply_chunking_migration(backend=backend)

                # Verify boundary columns now exist
                def _check_columns(conn: sqlite3.Connection) -> list[str]:
                    cursor = conn.execute('PRAGMA table_info(embedding_chunks)')
                    return [row[1] for row in cursor.fetchall()]

                columns = await backend.execute_read(_check_columns)
                assert 'start_index' in columns, 'start_index column should exist after migration'
                assert 'end_index' in columns, 'end_index column should exist after migration'

            finally:
                await backend.shutdown()

    @pytest.mark.asyncio
    async def test_chunking_migration_applies_cleanly_with_existing_metadata_sqlite(
        self, tmp_path: Path,
    ) -> None:
        """The chunking migration creates embedding_chunks and adds chunk_count without backfilling.

        Verifies the migration adds the embedding_chunks table and the chunk_count column
        on embedding_metadata, but does NOT copy existing rows: embedding_chunks stays
        empty until embeddings are written through the regular write path.
        """
        db_path = tmp_path / 'test_data_migration.db'

        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')

        ctx_id_a = generate_id()
        ctx_id_b = generate_id()

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

            conn.execute('''
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id TEXT NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                )
            ''')

            conn.execute(
                '''INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
                   VALUES (?, ?, ?, ?, ?, 'local')''',
                (ctx_id_a, 'thread-1', 'user', 'text', 'Test content 1'),
            )
            conn.execute(
                '''INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
                   VALUES (?, ?, ?, ?, ?, 'local')''',
                (ctx_id_b, 'thread-1', 'agent', 'text', 'Test content 2'),
            )

            conn.execute(
                '''INSERT INTO embedding_metadata (context_id, provider, model, dimensions)
                   VALUES (?, ?, ?, ?)''',
                (ctx_id_a, 'test-provider', 'test-model', 768),
            )
            conn.execute(
                '''INSERT INTO embedding_metadata (context_id, provider, model, dimensions)
                   VALUES (?, ?, ?, ?)''',
                (ctx_id_b, 'test-provider', 'test-model', 768),
            )
            conn.commit()

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'STORAGE_BACKEND': 'sqlite',
        }

        import os

        with patch.dict(os.environ, env, clear=False):
            from app.backends.sqlite_backend import SQLiteBackend

            backend = SQLiteBackend(db_path=str(db_path))
            await backend.initialize()

            try:
                from app.migrations import apply_chunking_migration

                await apply_chunking_migration(backend=backend)

                def _check_migration(
                    conn: sqlite3.Connection,
                ) -> tuple[list[tuple[str, int]], list[str]]:
                    cursor = conn.execute(
                        '''SELECT context_id, vec_rowid FROM embedding_chunks ORDER BY context_id''',
                    )
                    chunks = [(row[0], row[1]) for row in cursor.fetchall()]
                    cursor = conn.execute('PRAGMA table_info(embedding_metadata)')
                    metadata_columns = [row[1] for row in cursor.fetchall()]
                    return chunks, metadata_columns

                chunks, metadata_columns = await backend.execute_read(_check_migration)

                assert chunks == []
                assert 'chunk_count' in metadata_columns

            finally:
                await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_skipped_when_embedding_metadata_missing_sqlite(
        self, tmp_path: Path,
    ) -> None:
        """Verify migration skipped gracefully when embedding_metadata table doesn't exist."""
        db_path = tmp_path / 'test_no_metadata.db'

        # Create base schema WITHOUT embedding_metadata
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            conn.commit()

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'STORAGE_BACKEND': 'sqlite',
        }

        import os

        with patch.dict(os.environ, env, clear=False):
            from app.backends.sqlite_backend import SQLiteBackend

            backend = SQLiteBackend(db_path=str(db_path))
            await backend.initialize()

            try:
                from app.migrations import apply_chunking_migration

                # Should not raise - gracefully skip when embedding_metadata doesn't exist
                await apply_chunking_migration(backend=backend)

                # Verify embedding_chunks was NOT created (since semantic search tables don't exist)
                def _check_no_table(conn: sqlite3.Connection) -> bool:
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_chunks'",
                    )
                    return cursor.fetchone() is None

                no_table = await backend.execute_read(_check_no_table)
                assert no_table, 'embedding_chunks table should NOT exist when embedding_metadata missing'

            finally:
                await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_file_not_found_raises(self, tmp_path: Path) -> None:
        """Verify RuntimeError when migration file missing."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'STORAGE_BACKEND': 'sqlite',
        }

        import os

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'

            # Patch Path.exists to return False for chunking migration file
            original_exists = Path.exists

            def mock_exists(self: Path) -> bool:
                if 'add_chunking' in str(self):
                    return False
                return original_exists(self)

            with patch.object(Path, 'exists', mock_exists):
                from app.migrations import apply_chunking_migration

                with pytest.raises(RuntimeError, match='migration file not found'):
                    await apply_chunking_migration(backend=mock_backend)
