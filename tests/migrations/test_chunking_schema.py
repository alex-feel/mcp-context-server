"""Tests for the chunking migration SQL files and the SQLite schema they produce."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest


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
        read path in ``app/repositories/embedding_repository/``.
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
            'no {SCHEMA} substitution, per the '
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
                INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
                VALUES (1, 'thread-1', 'user', 'text', 'Long content that would be chunked', 'local')
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
                INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
                VALUES (1, 'thread-1', 'user', 'text', 'Test content', 'local')
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
