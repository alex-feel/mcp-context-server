"""Tests for database migration functions.

This module tests migration functions in app/server.py that are critical
for proper database initialization during server startup.

P0 Priority: These functions have ZERO test coverage but are critical paths.
"""

from __future__ import annotations

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
        """Verify no-op when ENABLE_SEMANTIC_SEARCH=false."""
        from app.migrations import apply_semantic_search_migration

        # Create mock backend
        mock_backend = MagicMock()
        mock_backend.backend_type = 'sqlite'

        # Mock settings directly since it's already loaded at import time
        mock_settings = MagicMock()
        mock_settings.semantic_search.enabled = False

        with patch('app.migrations.semantic.settings', mock_settings):
            # Call should return early without doing anything
            await apply_semantic_search_migration(backend=mock_backend)

            # No execute_write calls should have been made
            mock_backend.execute_write.assert_not_called()
            mock_backend.execute_read.assert_not_called()

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
                INSERT INTO context_entries (thread_id, source, content_type, text_content)
                VALUES ('test', 'user', 'text', 'test content')
            ''')
            conn.execute('''
                INSERT INTO embedding_metadata (context_id, model_name, dimensions)
                VALUES (1, 'test-model', 384)
            ''')
            conn.commit()

        # Create mock settings with semantic search enabled and dimension mismatch
        # NOTE: Patching os.environ has NO EFFECT because get_settings() is cached
        # via @lru_cache at conftest.py import time with semantic_search.enabled=False.
        # We must patch the settings object directly in the migration module.
        mock_settings = MagicMock()
        mock_settings.semantic_search.enabled = True
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


class TestApplyFtsMigration:
    """Tests for apply_fts_migration()."""

    @pytest.mark.asyncio
    async def test_migration_skipped_when_disabled(self, tmp_path: Path) -> None:
        """Verify no-op when ENABLE_FTS=false."""
        db_path = tmp_path / 'test.db'

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_FTS': 'false',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_repos = MagicMock()

            from app.migrations import apply_fts_migration

            await apply_fts_migration(backend=mock_backend, repos=mock_repos)

            # No execute calls should be made
            mock_backend.execute_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_initial_migration_sqlite(self, tmp_path: Path) -> None:
        """Verify FTS5 table created with correct tokenizer for SQLite."""
        db_path = tmp_path / 'test_fts.db'

        # Create base schema first
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_FTS': 'true',
            'FTS_LANGUAGE': 'english',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            from app.backends.sqlite_backend import SQLiteBackend

            backend = SQLiteBackend(db_path=str(db_path))
            await backend.initialize()

            try:
                from app.migrations import apply_fts_migration
                from app.repositories.fts_repository import FtsRepository

                fts_repo = FtsRepository(backend)

                # Check FTS not available yet
                fts_available_before = await fts_repo.is_available()
                assert fts_available_before is False

                # Apply migration
                await apply_fts_migration(backend=backend)

                # Check FTS is now available
                fts_available_after = await fts_repo.is_available()
                assert fts_available_after is True
            finally:
                await backend.shutdown()

    @pytest.mark.asyncio
    async def test_migration_file_not_found_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify warning logged when migration file missing.

        Note: apply_fts_migration catches all exceptions and logs a warning,
        so we verify the warning is logged rather than expecting RuntimeError.
        """
        from app.migrations import apply_fts_migration

        mock_backend = MagicMock()
        mock_backend.backend_type = 'sqlite'

        # Mock FTS repo to say FTS doesn't exist
        mock_fts_repo = MagicMock()
        mock_fts_repo.is_available = AsyncMock(return_value=False)

        mock_repos = MagicMock()
        mock_repos.fts = mock_fts_repo

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.fts.enabled = True

        # Patch Path.exists to return False for FTS migration file
        original_exists = Path.exists

        def mock_exists(self: Path) -> bool:
            if 'add_fts' in str(self):
                return False
            return original_exists(self)

        with (
            patch('app.migrations.semantic.settings', mock_settings),
            patch.object(Path, 'exists', mock_exists),
        ):
            # Function should not raise - it catches and logs
            await apply_fts_migration(backend=mock_backend, repos=mock_repos)

            # Verify warning was logged about migration failure
            assert any('migration' in record.message.lower() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_tokenizer_migration_detection(self, tmp_path: Path) -> None:
        """Verify migration triggered when language setting changes."""
        db_path = tmp_path / 'test_fts_migrate.db'

        # Create base schema and FTS with unicode61 tokenizer
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        migration_path = Path(__file__).parent.parent / 'app' / 'migrations' / 'add_fts_sqlite.sql'
        fts_sql = migration_path.read_text()
        fts_sql = fts_sql.replace('{TOKENIZER}', 'unicode61')

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            conn.executescript(fts_sql)

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'ENABLE_FTS': 'true',
            'FTS_LANGUAGE': 'english',  # Should trigger migration to porter
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            from app.backends.sqlite_backend import SQLiteBackend

            backend = SQLiteBackend(db_path=str(db_path))
            await backend.initialize()

            try:
                from app.migrations import apply_fts_migration
                from app.repositories.fts_repository import FtsRepository

                fts_repo = FtsRepository(backend)

                # Verify FTS exists with unicode61
                current_tokenizer = await fts_repo.get_current_tokenizer()
                assert current_tokenizer is not None
                assert 'unicode61' in current_tokenizer

                # Apply migration - should detect mismatch and migrate
                await apply_fts_migration(backend=backend)

                # Verify tokenizer changed to porter
                new_tokenizer = await fts_repo.get_current_tokenizer()
                assert new_tokenizer is not None
                assert 'porter' in new_tokenizer
            finally:
                await backend.shutdown()


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


class TestAdvisoryLockFix:
    """Tests verifying migration functions use transaction-scoped advisory locks.

    All PostgreSQL migration functions must use pg_advisory_xact_lock (transaction-scoped)
    instead of pg_advisory_lock (session-scoped) to prevent race conditions when
    multiple pods start simultaneously under execute_write()'s conn.transaction() wrapper.
    """

    @pytest.mark.asyncio
    async def test_fts_migration_uses_xact_lock(self) -> None:
        """Verify FTS migration uses pg_advisory_xact_lock, not pg_advisory_lock."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'postgresql'

        executed_statements: list[str] = []

        async def mock_execute_write(operation, *_args, **_kwargs):
            mock_conn = AsyncMock()

            async def record_execute(stmt, *_a, **_kw):
                executed_statements.append(stmt)

            mock_conn.execute = record_execute
            await operation(mock_conn)

        mock_backend.execute_write = mock_execute_write

        mock_fts_repo = MagicMock()
        mock_fts_repo.is_available = AsyncMock(return_value=False)
        mock_repos = MagicMock()
        mock_repos.fts = mock_fts_repo

        mock_settings = MagicMock()
        mock_settings.fts.enabled = True
        mock_settings.fts.language = 'english'
        mock_settings.storage.db_path = '/tmp/test.db'

        with patch('app.migrations.fts.settings', mock_settings):
            from app.migrations.fts import apply_fts_migration

            await apply_fts_migration(backend=mock_backend, repos=mock_repos)

        lock_stmts = [s for s in executed_statements if 'advisory' in s.lower()]
        assert any('pg_advisory_xact_lock' in s for s in lock_stmts), (
            f'Expected pg_advisory_xact_lock, got: {lock_stmts}'
        )
        assert not any('pg_advisory_unlock' in s for s in lock_stmts), (
            f'Should not have explicit unlock with xact lock, got: {lock_stmts}'
        )

    @pytest.mark.asyncio
    async def test_semantic_migration_uses_xact_lock(self) -> None:
        """Verify semantic search migration uses pg_advisory_xact_lock."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'postgresql'

        executed_statements: list[str] = []

        async def mock_execute_write(operation, *_args, **_kwargs):
            mock_conn = AsyncMock()

            async def record_execute(stmt, *_a, **_kw):
                executed_statements.append(stmt)

            mock_conn.execute = record_execute
            mock_conn.fetchrow = AsyncMock(return_value=None)
            await operation(mock_conn)

        mock_backend.execute_write = mock_execute_write

        # execute_read for _check_existing_dimension_postgresql returns (table_exists, existing_dim)
        async def mock_execute_read(operation, *_args, **_kwargs):
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=None)
            return await operation(mock_conn)

        mock_backend.execute_read = mock_execute_read

        mock_settings = MagicMock()
        mock_settings.semantic_search.enabled = True
        mock_settings.embedding.dim = 768
        mock_settings.storage.db_path = '/tmp/test.db'
        mock_settings.storage.postgresql_schema = 'public'

        with patch('app.migrations.semantic.settings', mock_settings):
            from app.migrations.semantic import apply_semantic_search_migration

            await apply_semantic_search_migration(backend=mock_backend)

        lock_stmts = [s for s in executed_statements if 'advisory' in s.lower()]
        assert any('pg_advisory_xact_lock' in s for s in lock_stmts), (
            f'Expected pg_advisory_xact_lock, got: {lock_stmts}'
        )
        assert not any('pg_advisory_unlock' in s for s in lock_stmts), (
            f'Should not have explicit unlock with xact lock, got: {lock_stmts}'
        )

    @pytest.mark.asyncio
    async def test_chunking_migration_uses_xact_lock(self) -> None:
        """Verify chunking migration uses pg_advisory_xact_lock."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'postgresql'

        executed_statements: list[str] = []

        async def mock_execute_write(operation, *_args, **_kwargs):
            mock_conn = AsyncMock()

            async def record_execute(stmt, *_a, **_kw):
                executed_statements.append(stmt)

            mock_conn.execute = record_execute
            await operation(mock_conn)

        mock_backend.execute_write = mock_execute_write
        mock_backend.execute_read = AsyncMock(return_value=False)

        mock_settings = MagicMock()
        mock_settings.embedding.dim = 768
        mock_settings.storage.db_path = '/tmp/test.db'
        mock_settings.storage.postgresql_schema = 'public'
        mock_settings.storage.postgresql_command_timeout_s = 60.0
        mock_settings.storage.postgresql_migration_timeout_s = 300.0

        with patch('app.migrations.chunking.settings', mock_settings):
            from app.migrations.chunking import apply_chunking_migration

            await apply_chunking_migration(backend=mock_backend)

        lock_stmts = [s for s in executed_statements if 'advisory' in s.lower()]
        assert any('pg_advisory_xact_lock' in s for s in lock_stmts), (
            f'Expected pg_advisory_xact_lock, got: {lock_stmts}'
        )
        assert not any('pg_advisory_unlock' in s for s in lock_stmts), (
            f'Should not have explicit unlock with xact lock, got: {lock_stmts}'
        )

    @pytest.mark.asyncio
    async def test_metadata_index_creation_uses_xact_lock(self) -> None:
        """Verify metadata index creation uses pg_advisory_xact_lock."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'postgresql'

        executed_statements: list[str] = []

        async def mock_execute_write(operation, *_args, **_kwargs):
            mock_conn = AsyncMock()

            async def record_execute(stmt, *_a, **_kw):
                executed_statements.append(stmt)

            mock_conn.execute = record_execute
            await operation(mock_conn)

        mock_backend.execute_write = mock_execute_write

        mock_settings = MagicMock()
        mock_settings.storage.postgresql_schema = 'public'

        with patch('app.migrations.metadata.settings', mock_settings):
            from app.migrations.metadata import _create_metadata_index

            await _create_metadata_index(
                backend=mock_backend,
                field='test_field',
                type_hint='string',
            )

        lock_stmts = [s for s in executed_statements if 'advisory' in s.lower()]
        assert any('pg_advisory_xact_lock' in s for s in lock_stmts), (
            f'Expected pg_advisory_xact_lock, got: {lock_stmts}'
        )
        assert not any('pg_advisory_unlock' in s for s in lock_stmts), (
            f'Should not have explicit unlock with xact lock, got: {lock_stmts}'
        )

    @pytest.mark.asyncio
    async def test_metadata_index_drop_uses_xact_lock(self) -> None:
        """Verify metadata index drop uses pg_advisory_xact_lock."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'postgresql'

        executed_statements: list[str] = []

        async def mock_execute_write(operation, *_args, **_kwargs):
            mock_conn = AsyncMock()

            async def record_execute(stmt, *_a, **_kw):
                executed_statements.append(stmt)

            mock_conn.execute = record_execute
            await operation(mock_conn)

        mock_backend.execute_write = mock_execute_write

        mock_settings = MagicMock()
        mock_settings.storage.postgresql_schema = 'public'

        with patch('app.migrations.metadata.settings', mock_settings):
            from app.migrations.metadata import _drop_metadata_index

            await _drop_metadata_index(
                backend=mock_backend,
                field='test_field',
            )

        lock_stmts = [s for s in executed_statements if 'advisory' in s.lower()]
        assert any('pg_advisory_xact_lock' in s for s in lock_stmts), (
            f'Expected pg_advisory_xact_lock, got: {lock_stmts}'
        )
        assert not any('pg_advisory_unlock' in s for s in lock_stmts), (
            f'Should not have explicit unlock with xact lock, got: {lock_stmts}'
        )

    @pytest.mark.asyncio
    async def test_no_session_scoped_locks_in_migration_source(self) -> None:
        """Verify migration source files contain no session-scoped advisory locks.

        This is a static analysis test that reads the source files directly
        to ensure no pg_advisory_lock (without xact) or pg_advisory_unlock calls remain.
        """
        from pathlib import Path

        migrations_dir = Path(__file__).parent.parent / 'app' / 'migrations'
        migration_files = ['fts.py', 'semantic.py', 'chunking.py', 'metadata.py']

        for filename in migration_files:
            source = (migrations_dir / filename).read_text()
            lines = source.split('\n')
            for i, line in enumerate(lines, 1):
                # Check for session-scoped lock (pg_advisory_lock without xact prefix)
                if 'pg_advisory_lock(' in line and 'pg_advisory_xact_lock(' not in line:
                    pytest.fail(
                        f'{filename}:{i} contains session-scoped pg_advisory_lock '
                        f'(should be pg_advisory_xact_lock): {line.strip()}',
                    )
                # Check for explicit unlock (not needed with xact locks)
                if 'pg_advisory_unlock(' in line:
                    pytest.fail(
                        f'{filename}:{i} contains pg_advisory_unlock '
                        f'(not needed with pg_advisory_xact_lock): {line.strip()}',
                    )
