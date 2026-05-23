"""Tests for chunking migration functions.

This module tests the apply_chunking_migration() function that enables
1:N embedding relationships (one context entry can have multiple chunk embeddings).

Migration creates:
- SQLite: embedding_chunks mapping table
- PostgreSQL: id BIGSERIAL column in vec_context_embeddings
- Both: chunk_count column in embedding_metadata
"""

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.ids import generate_id


class TestApplyChunkingMigration:
    """Tests for apply_chunking_migration()."""

    @pytest.mark.asyncio
    async def test_migration_skipped_when_semantic_search_disabled(self) -> None:
        """Verify no-op when ENABLE_SEMANTIC_SEARCH=false."""
        from app.migrations import apply_chunking_migration

        # Create mock backend
        mock_backend = MagicMock()
        mock_backend.backend_type = 'sqlite'

        # Mock settings with semantic search disabled
        mock_settings = MagicMock()
        mock_settings.semantic_search.enabled = False

        with patch('app.migrations.chunking.settings', mock_settings):
            # Call should return early without doing anything
            await apply_chunking_migration(backend=mock_backend)

            # No execute_write calls should have been made
            mock_backend.execute_write.assert_not_called()
            mock_backend.execute_read.assert_not_called()

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

        This test simulates upgrade from pre-f36266c schema where embedding_chunks
        was created WITHOUT start_index and end_index columns.
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
            # Create OLD embedding_chunks WITHOUT boundary columns (pre-f36266c schema)
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
                '''INSERT INTO context_entries (id, thread_id, source, content_type, text_content)
                   VALUES (?, ?, ?, ?, ?)''',
                (ctx_id_a, 'thread-1', 'user', 'text', 'Test content 1'),
            )
            conn.execute(
                '''INSERT INTO context_entries (id, thread_id, source, content_type, text_content)
                   VALUES (?, ?, ?, ?, ?)''',
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


class TestApplyChunkingMigrationPostgreSQL:
    """Tests for apply_chunking_migration() with PostgreSQL backend."""

    @pytest.mark.asyncio
    async def test_migration_postgresql_adds_id_column(self) -> None:
        """Verify id BIGSERIAL column added for PostgreSQL."""
        env = {
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'postgresql',
            'POSTGRESQL_SCHEMA': 'public',
            'ENABLE_SEMANTIC_SEARCH': 'true',
        }

        import os

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'postgresql'
            # First call: check if id column exists (returns False)
            mock_backend.execute_read = AsyncMock(return_value=False)
            mock_backend.execute_write = AsyncMock()

            from app.migrations import apply_chunking_migration

            await apply_chunking_migration(backend=mock_backend)

            # Should have called execute_read (to check id column) and execute_write (to apply)
            mock_backend.execute_read.assert_called_once()
            mock_backend.execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_migration_postgresql_idempotent(self) -> None:
        """Verify PostgreSQL migration skips SQL execution when already applied."""
        env = {
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'postgresql',
            'POSTGRESQL_SCHEMA': 'public',
            'ENABLE_SEMANTIC_SEARCH': 'true',
        }

        import os

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'postgresql'
            # id column already exists (migration already applied)
            mock_backend.execute_read = AsyncMock(return_value=True)
            mock_backend.execute_write = AsyncMock()

            from app.migrations import apply_chunking_migration

            # Should not raise
            await apply_chunking_migration(backend=mock_backend)

            # execute_read called to check if migration already applied
            mock_backend.execute_read.assert_called_once()
            # execute_write should NOT be called (early return when already applied)
            mock_backend.execute_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_chunk_index_column_postgresql_sql(self) -> None:
        """CRITICAL: Verify PostgreSQL SQL has NO chunk_index column (user decision)."""
        migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_chunking_postgresql.sql'

        assert migration_path.exists(), 'PostgreSQL migration file should exist'

        sql_content = migration_path.read_text(encoding='utf-8')

        # Verify NO chunk_index, chunk_start, chunk_end in SQL
        assert 'chunk_index' not in sql_content.lower(), 'chunk_index should NOT be in PostgreSQL migration'
        assert 'chunk_start' not in sql_content.lower(), 'chunk_start should NOT be in PostgreSQL migration'
        assert 'chunk_end' not in sql_content.lower(), 'chunk_end should NOT be in PostgreSQL migration'

        # Verify expected content is present
        assert 'id BIGSERIAL' in sql_content or 'id bigserial' in sql_content.lower(), (
            'id BIGSERIAL column should be in PostgreSQL migration'
        )
        assert 'chunk_count' in sql_content.lower(), 'chunk_count column should be in PostgreSQL migration'


class TestChunkingMigrationSQLFiles:
    """Tests for chunking migration SQL file content."""

    def test_sqlite_sql_file_exists(self) -> None:
        """Verify SQLite migration SQL file exists."""
        migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_chunking_sqlite.sql'
        assert migration_path.exists(), 'SQLite migration file should exist'

    def test_postgresql_sql_file_exists(self) -> None:
        """Verify PostgreSQL migration SQL file exists."""
        migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_chunking_postgresql.sql'
        assert migration_path.exists(), 'PostgreSQL migration file should exist'

    def test_sqlite_sql_creates_embedding_chunks_table(self) -> None:
        """Verify SQLite SQL creates embedding_chunks table."""
        migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_chunking_sqlite.sql'
        sql_content = migration_path.read_text(encoding='utf-8')

        assert 'CREATE TABLE IF NOT EXISTS embedding_chunks' in sql_content
        assert 'context_id TEXT NOT NULL' in sql_content
        assert 'vec_rowid INTEGER NOT NULL' in sql_content

    def test_sqlite_sql_no_forbidden_columns(self) -> None:
        """CRITICAL: Verify SQLite SQL has NO forbidden columns (user decision)."""
        migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_chunking_sqlite.sql'
        sql_content = migration_path.read_text(encoding='utf-8')

        assert 'chunk_index' not in sql_content.lower(), 'chunk_index should NOT be in SQLite migration'
        assert 'chunk_start' not in sql_content.lower(), 'chunk_start should NOT be in SQLite migration'
        assert 'chunk_end' not in sql_content.lower(), 'chunk_end should NOT be in SQLite migration'

    def test_postgresql_sql_uses_bare_tables_and_current_schema(self) -> None:
        """Verify PostgreSQL chunking SQL uses BARE table names and
        ``current_schema()`` for catalog filters.

        BARE table/index DDL relies on the operator's ``search_path``
        configuration (``$POSTGRESQL_SCHEMA, public``). Idempotency-
        check filters against ``information_schema`` and ``pg_indexes``
        use ``current_schema()`` so the check inspects the same schema
        the migration writes to. This matches the BARE-table convention
        established by ``app/schemas/postgresql_schema.sql`` and the
        read path in ``app/repositories/embedding_repository.py``.
        """
        migration_path = (
            Path(__file__).parent.parent.parent
            / 'app' / 'migrations' / 'add_chunking_postgresql.sql'
        )
        sql_content = migration_path.read_text(encoding='utf-8')

        # Strip SQL comments before asserting absence of {SCHEMA}: the
        # header comment legitimately references the placeholder in
        # prose explaining the BARE-DDL convention. Only active DDL
        # lines are part of the contract.
        sql_no_comments = '\n'.join(
            line for line in sql_content.splitlines()
            if not line.strip().startswith('--')
        )
        assert '{SCHEMA}' not in sql_no_comments, (
            'Chunking migration must use BARE table names; '
            '{SCHEMA} substitution was dropped to align with the '
            'project-wide bare-DDL convention.'
        )
        assert 'current_schema()' in sql_content, (
            'Chunking migration idempotency-check filters must use '
            'current_schema() to introspect the resolved schema.'
        )

    def test_both_sql_files_are_idempotent(self) -> None:
        """Verify both SQL files use IF NOT EXISTS / IF EXISTS patterns."""
        sqlite_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_chunking_sqlite.sql'
        postgresql_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_chunking_postgresql.sql'

        sqlite_sql = sqlite_path.read_text(encoding='utf-8')
        postgresql_sql = postgresql_path.read_text(encoding='utf-8')

        # SQLite uses IF NOT EXISTS for table and index creation
        assert 'IF NOT EXISTS' in sqlite_sql, 'SQLite migration should use IF NOT EXISTS'

        # PostgreSQL uses IF NOT id_exists pattern in DO block
        assert 'IF NOT id_exists' in postgresql_sql, 'PostgreSQL migration should check id_exists'
        assert 'IF NOT EXISTS' in postgresql_sql, 'PostgreSQL migration should use IF NOT EXISTS for indexes'


class TestChunkingMigrationIntegration:
    """Integration tests for chunking migration with actual database operations."""

    @pytest.mark.asyncio
    async def test_multiple_chunks_per_context_sqlite(self, tmp_path: Path) -> None:
        """Verify multiple chunks can be stored per context_id after migration."""
        db_path = tmp_path / 'test_multiple_chunks.db'

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
                    chunk_count INTEGER NOT NULL DEFAULT 1,
                    embedded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                )
            ''')

            # Insert test context entry
            conn.execute('''
                INSERT INTO context_entries (id, thread_id, source, content_type, text_content)
                VALUES (1, 'thread-1', 'user', 'text', 'Long content that would be chunked')
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

                # Insert multiple chunks for the same context_id
                def _insert_multiple_chunks(conn: sqlite3.Connection) -> None:
                    # Simulate 3 chunks for context_id='0190abcdef1234567890abcd00000001'
                    conn.execute('''
                        INSERT INTO embedding_chunks (context_id, vec_rowid)
                        VALUES (1, 100), (1, 101), (1, 102)
                    ''')

                await backend.execute_write(_insert_multiple_chunks)

                # Verify multiple chunks exist for same context_id
                def _count_chunks(conn: sqlite3.Connection) -> int:
                    cursor = conn.execute('''
                        SELECT COUNT(*) FROM embedding_chunks WHERE context_id = 1
                    ''')
                    return cursor.fetchone()[0]

                chunk_count = await backend.execute_read(_count_chunks)
                assert chunk_count == 3, 'Should have 3 chunks for context_id=1'

            finally:
                await backend.shutdown()

    @pytest.mark.asyncio
    async def test_cascade_delete_sqlite(self, tmp_path: Path) -> None:
        """Verify embedding_chunks are deleted when context_entry is deleted (CASCADE)."""
        db_path = tmp_path / 'test_cascade.db'

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
                    chunk_count INTEGER NOT NULL DEFAULT 1,
                    embedded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                )
            ''')
            conn.execute('PRAGMA foreign_keys = ON')

            # Insert test context entry
            conn.execute('''
                INSERT INTO context_entries (id, thread_id, source, content_type, text_content)
                VALUES (1, 'thread-1', 'user', 'text', 'Test content')
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

                # Insert chunks for context_id='0190abcdef1234567890abcd00000001'
                def _insert_chunks(conn: sqlite3.Connection) -> None:
                    conn.execute('PRAGMA foreign_keys = ON')
                    conn.execute('''
                        INSERT INTO embedding_chunks (context_id, vec_rowid)
                        VALUES (1, 100), (1, 101)
                    ''')

                await backend.execute_write(_insert_chunks)

                # Delete the context entry
                def _delete_context(conn: sqlite3.Connection) -> None:
                    conn.execute('PRAGMA foreign_keys = ON')
                    conn.execute('DELETE FROM context_entries WHERE id = 1')

                await backend.execute_write(_delete_context)

                # Verify chunks are also deleted
                def _count_chunks(conn: sqlite3.Connection) -> int:
                    cursor = conn.execute('SELECT COUNT(*) FROM embedding_chunks')
                    return cursor.fetchone()[0]

                chunk_count = await backend.execute_read(_count_chunks)
                assert chunk_count == 0, 'Chunks should be deleted when context is deleted (CASCADE)'

            finally:
                await backend.shutdown()
