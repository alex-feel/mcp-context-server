"""Tests for apply_fts_migration() in app/migrations/fts.py."""

import os
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


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

        # apply_fts_migration reads the module-level settings binding of app.migrations.fts,
        # which the environment patch does not reach.
        mock_settings = MagicMock()
        mock_settings.fts.enabled = False

        with (
            patch.dict(os.environ, env, clear=False),
            patch('app.migrations.fts.settings', mock_settings),
        ):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_repos = MagicMock()

            from app.migrations import apply_fts_migration

            await apply_fts_migration(backend=mock_backend, repos=mock_repos)

            # No execute calls should be made
            mock_backend.execute_write.assert_not_called()
            # The disabled check returns before the FTS availability probe
            mock_repos.fts.is_available.assert_not_called()

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
    async def test_force_provisions_fts_when_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """force=True provisions FTS even when ENABLE_FTS=false.

        This is the migration-CLI parity path: the PostgreSQL target-init must be
        able to provision FTS on the target whenever the source had it, decoupled
        from the CLI process's ENABLE_FTS toggle.
        """
        from app.settings import get_settings

        db_path = tmp_path / 'test_fts_forced.db'

        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        monkeypatch.setenv('ENABLE_FTS', 'false')
        monkeypatch.setenv('STORAGE_BACKEND', 'sqlite')
        monkeypatch.setenv('DB_PATH', str(db_path))
        get_settings.cache_clear()
        import app.migrations.fts as fts_module

        monkeypatch.setattr(fts_module, 'settings', get_settings())

        from app.backends.sqlite_backend import SQLiteBackend
        from app.repositories.fts_repository import FtsRepository

        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()
        try:
            fts_repo = FtsRepository(backend)
            assert await fts_repo.is_available() is False

            # ENABLE_FTS=false: a normal migration is a no-op.
            await fts_module.apply_fts_migration(backend=backend)
            assert await fts_repo.is_available() is False

            # force=True provisions FTS regardless of the disabled toggle.
            await fts_module.apply_fts_migration(backend=backend, force=True)
            assert await fts_repo.is_available() is True
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

        # Patch Path.exists to return False for FTS migration file
        original_exists = Path.exists

        def mock_exists(self: Path) -> bool:
            if 'add_fts' in str(self):
                return False
            return original_exists(self)

        with patch.object(Path, 'exists', mock_exists):
            # Function should not raise - it catches and logs
            await apply_fts_migration(backend=mock_backend, repos=mock_repos)

            # Verify warning was logged about migration failure
            assert any('migration' in record.message.lower() for record in caplog.records)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
    async def test_rebuild_estimate_counts_every_entry(self, backend_type: str) -> None:
        """A tokenizer or language rebuild sizes its estimate from every entry, through the system scope."""
        import app.migrations.fts as fts_module
        from app.access_scope import SYSTEM_SCOPE

        settings = MagicMock()
        settings.fts.enabled = True
        settings.fts.language = 'english'

        fts_repo = MagicMock()
        fts_repo.is_available = AsyncMock(return_value=True)
        fts_repo.get_current_tokenizer = AsyncMock(return_value='unicode61')
        fts_repo.get_desired_tokenizer = AsyncMock(return_value='porter unicode61')
        fts_repo.get_current_language = AsyncMock(return_value='german')
        fts_repo.get_statistics = AsyncMock(return_value={'total_entries': 42})
        records_counted: list[int | None] = []

        async def _migrate(_target: str) -> dict[str, object]:
            records_counted.append(fts_module.get_fts_migration_status().records_count)
            return {
                'entries_migrated': 42, 'old_tokenizer': 'unicode61', 'new_tokenizer': 'porter unicode61',
                'old_language': 'german', 'new_language': 'english',
            }

        fts_repo.migrate_tokenizer = AsyncMock(side_effect=_migrate)
        fts_repo.migrate_language = AsyncMock(side_effect=_migrate)
        repos = MagicMock()
        repos.fts = fts_repo
        backend = MagicMock()
        backend.backend_type = backend_type

        with patch.object(fts_module, 'settings', settings):
            await fts_module.apply_fts_migration(backend=backend, repos=repos)

        fts_repo.get_statistics.assert_awaited_once_with(scope=SYSTEM_SCOPE)
        assert records_counted == [42]

    @pytest.mark.asyncio
    async def test_tokenizer_migration_detection(self, tmp_path: Path) -> None:
        """Verify migration triggered when language setting changes."""
        db_path = tmp_path / 'test_fts_migrate.db'

        # Create base schema and FTS with unicode61 tokenizer
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_fts_sqlite.sql'
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
