"""Tests that the PostgreSQL migrations take transaction-scoped advisory locks and quote the identifiers of
the metadata-index DROP.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


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
        mock_settings.embedding.generation_enabled = True
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
        mock_settings.embedding.generation_enabled = True
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
    async def test_metadata_index_drop_quotes_mixed_case_schema(self) -> None:
        """A mixed-case POSTGRESQL_SCHEMA is quoted in the qualified DROP so it is not folded.

        PostgreSQL case-folds an unquoted identifier to lowercase, so an unquoted ``MyApp.``
        qualifier would resolve to schema ``myapp`` and make ``DROP INDEX IF EXISTS`` silently
        skip the orphan index that actually lives in ``MyApp``. The DROP must route the schema
        through quote_pg_identifier, matching the search_path, CREATE SCHEMA, and target-probe sites.
        """
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
        mock_settings.storage.postgresql_schema = 'MyApp'

        with patch('app.migrations.metadata.settings', mock_settings):
            from app.migrations.metadata import _drop_metadata_index

            await _drop_metadata_index(
                backend=mock_backend,
                field='test_field',
            )

        drop_sql = next(s for s in executed_statements if 'DROP INDEX' in s)
        # Both the schema qualifier AND the generated index identifier are quoted so the
        # qualified DROP targets the exact case-preserved object the quoted CREATE produced.
        assert '"MyApp"."idx_metadata_test_field"' in drop_sql
        # The unquoted schema form would be case-folded by PostgreSQL and silently miss the real index.
        assert 'MyApp.idx_metadata_test_field' not in drop_sql

    @pytest.mark.asyncio
    async def test_metadata_index_drop_quotes_mixed_case_index_identifier(self) -> None:
        """A mixed-case field yields a quoted index identifier in the qualified DROP.

        The CREATE path quotes idx_metadata_{field} so PostgreSQL stores it case-preserved. The
        DROP must quote the SAME identifier, otherwise PostgreSQL folds the unquoted name to
        lowercase and DROP INDEX IF EXISTS silently skips the case-preserved index the
        reconciliation diff asked to remove, leaving auto-mode churning it every boot.
        """
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
                field='camelField',
            )

        drop_sql = next(s for s in executed_statements if 'DROP INDEX' in s)
        # The generated index identifier is quoted, matching the quoted CREATE.
        assert '"idx_metadata_camelField"' in drop_sql
        # The bare (fold-susceptible) index name must NOT appear as the DROP target.
        assert 'IF EXISTS "public".idx_metadata_camelField' not in drop_sql

    @pytest.mark.asyncio
    async def test_no_session_scoped_locks_in_migration_source(self) -> None:
        """Verify migration source files contain no session-scoped advisory locks.

        This is a static analysis test that reads the source files directly
        to ensure they contain no pg_advisory_lock (without xact) or pg_advisory_unlock calls.
        """
        from pathlib import Path

        migrations_dir = Path(__file__).parent.parent.parent / 'app' / 'migrations'
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
