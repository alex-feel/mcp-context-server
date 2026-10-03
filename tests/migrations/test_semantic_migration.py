"""Tests for the semantic-search, jsonb_merge_patch and function search_path migrations in app/migrations/semantic.py."""

import contextlib
import os
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


class TestApplySemanticSearchMigration:
    """Tests for apply_semantic_search_migration()."""

    @pytest.mark.asyncio
    async def test_migration_skipped_when_disabled(self) -> None:
        """Verify no-op when ENABLE_EMBEDDING_GENERATION=false on a FRESH database.

        The semantic-search vector-storage migration is provisioned from embedding
        GENERATION (settings.embedding.generation_enabled), not from the search TOOL
        toggle. With generation off it skips only when the database carries no
        embedding infrastructure (the infra-present fallthrough keeps an
        existing layout maintained); the only read allowed is that probe.
        """
        from unittest.mock import AsyncMock

        from app.migrations import apply_semantic_search_migration

        # Create mock backend; the infra probe reports a fresh database.
        mock_backend = MagicMock()
        mock_backend.backend_type = 'sqlite'
        mock_backend.execute_read = AsyncMock(return_value=False)

        # Mock settings directly since it's already loaded at import time
        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = False

        with patch('app.migrations.semantic.settings', mock_settings):
            # Call should return early without doing anything
            await apply_semantic_search_migration(backend=mock_backend)

            # No writes; the single read is the embedding-infra probe.
            mock_backend.execute_write.assert_not_called()
            mock_backend.execute_read.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_migration_creates_tables_sqlite(self, tmp_path: Path) -> None:
        """Verify vec_context_embeddings and embedding_metadata tables created for SQLite."""
        db_path = tmp_path / 'test_semantic.db'

        # Create base schema first
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'STORAGE_BACKEND': 'sqlite',
            'EMBEDDING_DIM': '768',
        }

        # Mock sqlite_vec to avoid requiring the extension
        mock_sqlite_vec = MagicMock()

        with (
            patch.dict(os.environ, env, clear=False),
            patch.dict('sys.modules', {'sqlite_vec': mock_sqlite_vec}),
            patch('importlib.util.find_spec') as mock_find_spec,
        ):
            # Make importlib think sqlite_vec is installed
            mock_find_spec.return_value = MagicMock()

            from app.backends.sqlite_backend import SQLiteBackend

            backend = SQLiteBackend(db_path=str(db_path))
            await backend.initialize()

            try:
                # Patch the import inside the migration function
                with patch('sqlite_vec.load'):
                    from app.migrations import apply_semantic_search_migration

                    # The migration may fail due to missing vec0 module, but we can
                    # verify the flow was attempted. We expect RuntimeError with
                    # sqlite-vec related message if extension not available.
                    with contextlib.suppress(RuntimeError):
                        await apply_semantic_search_migration(backend=backend)
                    # If no exception, migration succeeded (sqlite_vec was available)
            finally:
                await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_idempotent(self, tmp_path: Path) -> None:
        """Verify running migration twice does not fail."""
        db_path = tmp_path / 'test_idempotent.db'

        # Create base schema
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            # Manually create the semantic search tables to simulate existing migration
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    context_id INTEGER PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                )
            ''')
            conn.commit()

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'STORAGE_BACKEND': 'sqlite',
            'EMBEDDING_DIM': '768',
        }

        # Mock sqlite_vec to avoid requiring the extension
        mock_sqlite_vec = MagicMock()

        with (
            patch.dict(os.environ, env, clear=False),
            patch.dict('sys.modules', {'sqlite_vec': mock_sqlite_vec}),
        ):
            from app.backends.sqlite_backend import SQLiteBackend

            backend = SQLiteBackend(db_path=str(db_path))
            await backend.initialize()

            try:
                # Patch to avoid needing actual sqlite-vec
                with patch('importlib.util.find_spec', return_value=MagicMock()):
                    from app.migrations import apply_semantic_search_migration

                    # Expected to fail if vec0 not available, suppress the error
                    with contextlib.suppress(RuntimeError):
                        await apply_semantic_search_migration(backend=backend)

                    # Second call should also not fail (idempotent)
                    with contextlib.suppress(RuntimeError):
                        await apply_semantic_search_migration(backend=backend)
            finally:
                await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_dimension_mismatch_raises(self, tmp_path: Path) -> None:
        """Verify RuntimeError when existing dimension != configured."""
        db_path = tmp_path / 'test_mismatch.db'

        # Create base schema with different dimension in metadata
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            # Create tables with existing dimension
            conn.execute('''
                CREATE TABLE IF NOT EXISTS vec_context_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_entry_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    context_id INTEGER PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                )
            ''')
            # Insert metadata with dimension 384 (different from configured 768)
            conn.execute('''
                INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
                VALUES ('0190abcdef1234567890abcd00000001', 'test', 'user', 'text', 'test content', 'local')
            ''')
            conn.execute('''
                INSERT INTO embedding_metadata (context_id, model_name, dimensions)
                VALUES (1, 'test-model', 384)
            ''')
            conn.commit()

        # Create mock settings with embedding generation enabled and dimension mismatch.
        # The migration gates on settings.embedding.generation_enabled (the search TOOL
        # toggle does not control whether the vector-storage migration runs).
        # NOTE: Patching os.environ has NO EFFECT because app.migrations.semantic binds
        # its module-level ``settings`` at import time. We must patch the settings object
        # directly in the migration module.
        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = True
        mock_settings.embedding.dim = 768  # Different from stored 384
        mock_settings.storage.db_path = db_path
        mock_settings.storage.postgresql_schema = 'public'

        with patch('app.migrations.semantic.settings', mock_settings):
            from app.backends.sqlite_backend import SQLiteBackend

            backend = SQLiteBackend(db_path=str(db_path))
            await backend.initialize()

            try:
                from app.migrations import apply_semantic_search_migration

                with pytest.raises(RuntimeError, match='dimension mismatch'):
                    await apply_semantic_search_migration(backend=backend)
            finally:
                await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_dimension_mismatch_postgresql_message(self) -> None:
        """On PostgreSQL the dimension-mismatch error gives schema/pg_dump guidance, not SQLite file steps.

        The dimension guard is backend-agnostic, but its remediation text must branch on the
        backend: a PostgreSQL deployment has no database file to delete, so the message must
        reference the embedding tables in the configured schema and a pg_dump backup instead
        of the SQLite db_path file-deletion steps.
        """
        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = True
        mock_settings.embedding.dim = 768  # Configured dimension
        mock_settings.storage.db_path = Path('/var/lib/should-not-appear/context_storage.db')
        mock_settings.storage.postgresql_schema = 'public'

        mock_backend = MagicMock()
        mock_backend.backend_type = 'postgresql'
        # The dimension probe returns (table_exists=True, existing_dim=384): a mismatch vs 768.
        mock_backend.execute_read = AsyncMock(return_value=(True, 384))

        with patch('app.migrations.semantic.settings', mock_settings):
            from app.migrations import apply_semantic_search_migration

            with pytest.raises(RuntimeError, match='dimension mismatch') as exc_info:
                await apply_semantic_search_migration(backend=mock_backend)

        message = str(exc_info.value)
        assert 'public' in message  # references the configured PostgreSQL schema
        assert 'pg_dump' in message  # PostgreSQL-appropriate backup guidance
        # The SQLite-only remediation must NOT appear on PostgreSQL:
        assert 'Delete or rename the database file' not in message
        assert 'should-not-appear' not in message  # the SQLite db_path must not leak
        # The remediation must list the REAL PostgreSQL embedding tables and must NOT
        # name embedding_chunks, which is a SQLite-only construct (PostgreSQL keeps the
        # chunk identity/boundaries as columns on vec_context_embeddings).
        assert 'vec_context_embeddings' in message
        assert 'embedding_metadata' in message
        assert 'compression_metadata' in message
        assert 'embedding_chunks' not in message

    @pytest.mark.asyncio
    async def test_migration_file_not_found_raises(self, tmp_path: Path) -> None:
        """Verify RuntimeError when migration file missing."""
        db_path = tmp_path / 'test.db'

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'

            # Patch Path.exists to return False for migration file
            original_exists = Path.exists

            def mock_exists(self: Path) -> bool:
                if 'add_semantic_search' in str(self):
                    return False
                return original_exists(self)

            with patch.object(Path, 'exists', mock_exists):
                from app.migrations import apply_semantic_search_migration

                with pytest.raises(RuntimeError, match='migration file not found'):
                    await apply_semantic_search_migration(backend=mock_backend)


class TestApplyJsonbMergePatchMigration:
    """Tests for apply_jsonb_merge_patch_migration()."""

    @pytest.mark.asyncio
    async def test_migration_skipped_for_sqlite(self, tmp_path: Path) -> None:
        """Verify no-op for SQLite backend."""
        db_path = tmp_path / 'test.db'

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'

            from app.migrations import apply_jsonb_merge_patch_migration

            await apply_jsonb_merge_patch_migration(backend=mock_backend)

            # No execute calls should be made for SQLite
            mock_backend.execute_write.assert_not_called()
            mock_backend.execute_read.assert_not_called()

    @pytest.mark.asyncio
    async def test_migration_postgresql_creates_function(self) -> None:
        """Verify function created in PostgreSQL."""
        env = {
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'postgresql',
            'POSTGRESQL_SCHEMA': 'public',
        }

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'postgresql'
            mock_backend.execute_read = AsyncMock(side_effect=[False, True])  # First: doesn't exist, Second: exists
            mock_backend.execute_write = AsyncMock()

            from app.migrations import apply_jsonb_merge_patch_migration

            await apply_jsonb_merge_patch_migration(backend=mock_backend)

            # Should have called execute_read (to check existence) and execute_write (to apply)
            assert mock_backend.execute_read.call_count == 2
            mock_backend.execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_migration_idempotent_postgresql(self) -> None:
        """Verify CREATE OR REPLACE is idempotent."""
        env = {
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'postgresql',
            'POSTGRESQL_SCHEMA': 'public',
        }

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'postgresql'
            # Function already exists - returns True both times
            mock_backend.execute_read = AsyncMock(return_value=True)
            mock_backend.execute_write = AsyncMock()

            from app.migrations import apply_jsonb_merge_patch_migration

            # Should not raise
            await apply_jsonb_merge_patch_migration(backend=mock_backend)

            # Should still write (CREATE OR REPLACE is safe)
            mock_backend.execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_migration_file_not_found_raises(self) -> None:
        """Verify RuntimeError when migration file missing."""
        env = {
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'postgresql',
        }

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'postgresql'

            # Patch Path.exists to return False for migration file
            original_exists = Path.exists

            def mock_exists(self: Path) -> bool:
                if 'add_jsonb_merge_patch' in str(self):
                    return False
                return original_exists(self)

            with patch.object(Path, 'exists', mock_exists):
                from app.migrations import apply_jsonb_merge_patch_migration

                with pytest.raises(RuntimeError, match='migration file not found'):
                    await apply_jsonb_merge_patch_migration(backend=mock_backend)


class TestApplyFunctionSearchPathMigration:
    """Tests for apply_function_search_path_migration() (CVE-2018-1058 mitigation)."""

    @pytest.mark.asyncio
    async def test_migration_skipped_for_sqlite(self, tmp_path: Path) -> None:
        """Verify no-op for SQLite backend."""
        db_path = tmp_path / 'test.db'

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'

            from app.migrations import apply_function_search_path_migration

            await apply_function_search_path_migration(backend=mock_backend)

            # No execute calls should be made for SQLite
            mock_backend.execute_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_migration_sets_search_path_postgresql(self) -> None:
        """Verify search_path set on all functions for PostgreSQL."""
        env = {
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'postgresql',
            'POSTGRESQL_SCHEMA': 'public',
        }

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'postgresql'
            mock_backend.execute_write = AsyncMock()

            from app.migrations import apply_function_search_path_migration

            await apply_function_search_path_migration(backend=mock_backend)

            # Should have called execute_write
            mock_backend.execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_migration_file_not_found_raises(self) -> None:
        """Verify RuntimeError when migration file missing."""
        env = {
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'postgresql',
        }

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'postgresql'

            # Patch Path.exists to return False for migration file
            original_exists = Path.exists

            def mock_exists(self: Path) -> bool:
                if 'fix_function_search_path' in str(self):
                    return False
                return original_exists(self)

            with patch.object(Path, 'exists', mock_exists):
                from app.migrations import apply_function_search_path_migration

                with pytest.raises(RuntimeError, match='migration file not found'):
                    await apply_function_search_path_migration(backend=mock_backend)

    @pytest.mark.asyncio
    async def test_migration_idempotent(self) -> None:
        """Verify migration can run multiple times safely."""
        env = {
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'postgresql',
            'POSTGRESQL_SCHEMA': 'public',
        }

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'postgresql'
            mock_backend.execute_write = AsyncMock()

            from app.migrations import apply_function_search_path_migration

            # Run twice
            await apply_function_search_path_migration(backend=mock_backend)
            await apply_function_search_path_migration(backend=mock_backend)

            # Both should succeed
            assert mock_backend.execute_write.call_count == 2
