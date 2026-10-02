"""Tests that PostgreSQL migration DDL runs under POSTGRESQL_MIGRATION_TIMEOUT_S through the app/migrations/_pg_ddl.py
helpers.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


class TestMigrationDdlTimeout:
    """PostgreSQL migration DDL must run under POSTGRESQL_MIGRATION_TIMEOUT_S.

    asyncpg applies the pool's ``command_timeout`` (default 60s) as the per-call
    CLIENT-side deadline for every ``conn.execute`` that passes no explicit
    ``timeout``. Migration DDL raises the SERVER-side ``statement_timeout`` to
    POSTGRESQL_MIGRATION_TIMEOUT_S (default 300s) via ``SET LOCAL`` -- which is
    transaction-scoped, so PostgreSQL auto-reverts it on COMMIT or ROLLBACK and no
    ``finally`` restore is needed. A plain ``SET`` + a ``finally`` restore must NOT be
    used: a failed DDL aborts the transaction and the restore ``SET`` would then raise
    InFailedSQLTransactionError (25P02), masking the real error and defeating the
    QueryCanceledError retry. The DDL and the advisory-lock acquire carry the migration
    budget PLUS a small client-side margin so an overrun is canceled server-side (a
    retryable QueryCanceledError) rather than client-side (a non-retryable
    asyncio.TimeoutError). These tests pin all of that; the advisory-lock
    tests in test_advisory_locks.py record only the SQL string and cannot catch it.
    """

    @pytest.mark.asyncio
    async def test_execute_migration_ddl_passes_timeout(self) -> None:
        """The shared helper forwards ``timeout_s`` plus the client margin as the per-call timeout."""
        from app.migrations._pg_ddl import _CLIENT_TIMEOUT_MARGIN_S
        from app.migrations._pg_ddl import execute_migration_ddl

        mock_conn = AsyncMock()
        await execute_migration_ddl(mock_conn, 'CREATE TABLE x (id int)', 300.0)
        mock_conn.execute.assert_awaited_once_with(
            'CREATE TABLE x (id int)',
            timeout=300.0 + _CLIENT_TIMEOUT_MARGIN_S,
        )

    @pytest.mark.asyncio
    async def test_begin_migration_uses_set_local_then_locks_under_budget(self) -> None:
        """begin_migration issues SET LOCAL (transaction-scoped, no finally) then the lock under budget."""
        from app.migrations._pg_ddl import _CLIENT_TIMEOUT_MARGIN_S
        from app.migrations._pg_ddl import begin_migration

        recorded: list[tuple[str, float | None]] = []

        async def record_execute(stmt, *_a, **kw):
            recorded.append((stmt, kw.get('timeout')))

        mock_conn = AsyncMock()
        mock_conn.execute = record_execute
        await begin_migration(mock_conn, 300.0)

        # First statement raises the budget transaction-scoped (no per-call timeout needed).
        assert recorded[0] == ('SET LOCAL statement_timeout = 300000', None)
        # Then the advisory-lock acquire carries the budget + client margin.
        assert any(
            'advisory' in stmt.lower() and t == 300.0 + _CLIENT_TIMEOUT_MARGIN_S
            for stmt, t in recorded
        )

    @pytest.mark.asyncio
    async def test_begin_migration_floors_sub_millisecond_budget_at_one(self) -> None:
        """A sub-millisecond budget must not truncate to statement_timeout = 0.

        PostgreSQL treats ``SET LOCAL statement_timeout = 0`` as UNLIMITED, so
        truncating a sub-millisecond POSTGRESQL_MIGRATION_TIMEOUT_S (the
        settings bound is only gt=0) would silently REMOVE the server-side
        backstop and invert the server-cancels-first design.
        """
        from app.migrations._pg_ddl import begin_migration

        recorded: list[str] = []

        async def record_execute(stmt, *_a, **kw):
            del kw
            recorded.append(stmt)

        mock_conn = AsyncMock()
        mock_conn.execute = record_execute
        await begin_migration(mock_conn, 0.0005)

        assert recorded[0] == 'SET LOCAL statement_timeout = 1'

    def test_migration_statement_timeout_ms_floor_and_default(self) -> None:
        """The shared millisecond conversion floors at 1 and is exact at the default."""
        from app.migrations._pg_ddl import migration_statement_timeout_ms

        assert migration_statement_timeout_ms(0.0005) == 1
        assert migration_statement_timeout_ms(300.0) == 300000

    @staticmethod
    def _recording_pg_backend(executed: list[tuple[str, float | None]]) -> MagicMock:
        """A PostgreSQL backend whose execute_write records (statement, timeout) pairs."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'postgresql'

        async def mock_execute_write(operation, *_args, **_kwargs):
            mock_conn = AsyncMock()

            async def record_execute(stmt, *_a, **kw):
                executed.append((stmt, kw.get('timeout')))

            async def record_fetchval(stmt, *_a, **kw):
                executed.append((stmt, kw.get('timeout')))
                return MagicMock()

            mock_conn.execute = record_execute
            mock_conn.fetchval = record_fetchval
            mock_conn.fetchrow = AsyncMock(return_value=None)
            await operation(mock_conn)

        mock_backend.execute_write = mock_execute_write
        return mock_backend

    @staticmethod
    def _assert_ddl_carries_timeout(
        executed: list[tuple[str, float | None]],
        *,
        migration_timeout: float,
    ) -> None:
        """Assert advisory-lock / CREATE / ALTER / DROP / DO statements carry the budget + client margin."""
        from app.migrations._pg_ddl import _CLIENT_TIMEOUT_MARGIN_S

        expected = migration_timeout + _CLIENT_TIMEOUT_MARGIN_S
        ddl = [
            (stmt, t)
            for stmt, t in executed
            if 'advisory' in stmt.lower()
            or stmt.strip().upper().startswith(('CREATE', 'ALTER', 'DROP', 'DO', 'REINDEX'))
        ]
        assert ddl, f'no migration DDL was recorded: {executed}'
        offenders = [(stmt, t) for stmt, t in ddl if t != expected]
        assert not offenders, (
            f'migration DDL must carry the {expected}s per-call asyncpg timeout '
            f'(migration budget {migration_timeout}s + client margin); offenders: {offenders}'
        )

    @staticmethod
    def _assert_set_local_no_finally_restore(executed: list[tuple[str, float | None]]) -> None:
        """The budget is raised via SET LOCAL, and no finally-restore SET statement_timeout is issued.

        A plain ``SET statement_timeout = <default>`` in a ``finally`` would raise
        InFailedSQLTransactionError (25P02) when a DDL failure has aborted the
        transaction -- masking the real error and defeating the QueryCanceledError
        retry. ``SET LOCAL`` is transaction-scoped and auto-reverts, so no restore runs.
        """
        set_locals = [stmt for stmt, _ in executed if 'set local statement_timeout' in stmt.lower()]
        assert set_locals, f'migration must raise the budget via SET LOCAL; executed: {executed}'
        bare_restores = [
            stmt
            for stmt, _ in executed
            if stmt.strip().lower().startswith('set ')
            and 'statement_timeout' in stmt.lower()
            and 'set local' not in stmt.lower()
        ]
        assert not bare_restores, (
            f'migration must not issue a plain SET statement_timeout (a finally-restore that '
            f'raises 25P02 in an aborted transaction and masks the real error); found: {bare_restores}'
        )

    @staticmethod
    def _assert_count_probe_carries_timeout(
        executed: list[tuple[str, float | None]],
        *,
        migration_timeout: float,
    ) -> None:
        """Assert a COUNT(*) probe issued inside the migration transaction carries the budget.

        A bare ``conn.fetchval`` would inherit the pool's shorter command_timeout and be
        canceled client-side on a large table (a non-retryable asyncio.TimeoutError)
        before the budgeted DDL runs, so the probe must share the migration deadline too.
        """
        from app.migrations._pg_ddl import _CLIENT_TIMEOUT_MARGIN_S

        expected = migration_timeout + _CLIENT_TIMEOUT_MARGIN_S
        probes = [(stmt, t) for stmt, t in executed if 'count(*)' in stmt.lower()]
        assert probes, f'no COUNT(*) probe was recorded: {executed}'
        offenders = [(stmt, t) for stmt, t in probes if t != expected]
        assert not offenders, (
            f'COUNT(*) probe inside the migration transaction must carry the {expected}s '
            f'per-call asyncpg timeout; offenders: {offenders}'
        )

    @pytest.mark.asyncio
    async def test_fts_migrate_language_count_probe_uses_migration_timeout(self) -> None:
        """The COUNT(*) probe inside the FTS language migration carries the migration budget.

        It runs inside the same transaction as the heavy DDL (after begin_migration raises
        the server-side budget), so a bare fetchval inheriting the pool's command_timeout
        would be canceled client-side on a large table before the rewrite ever runs.
        """
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)

        mock_settings = MagicMock()
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0

        from app.repositories.fts_repository import FtsRepository

        repo = FtsRepository(mock_backend)
        with (
            patch('app.settings.get_settings', return_value=mock_settings),
            patch.object(FtsRepository, 'get_current_language', AsyncMock(return_value='english')),
        ):
            await repo.migrate_language('german')

        self._assert_count_probe_carries_timeout(executed, migration_timeout=300.0)

    @pytest.mark.asyncio
    async def test_index_tree_migration_ddl_uses_migration_timeout(self) -> None:
        """index_tree CREATE TABLE/INDEX + advisory lock carry the migration timeout."""
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)

        mock_settings = MagicMock()
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0

        with patch('app.migrations.index_tree.settings', mock_settings):
            from app.migrations.index_tree import apply_index_tree_migration

            await apply_index_tree_migration(backend=mock_backend, force=True)

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_chunking_migration_ddl_uses_migration_timeout(self) -> None:
        """chunking migration DDL loop carries the migration timeout."""
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)
        mock_backend.execute_read = AsyncMock(return_value=False)  # not already applied

        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = True
        mock_settings.compression.enabled = False
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0
        mock_settings.storage.postgresql_schema = 'public'

        with patch('app.migrations.chunking.settings', mock_settings):
            from app.migrations.chunking import apply_chunking_migration

            await apply_chunking_migration(backend=mock_backend)

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_semantic_migration_ddl_uses_migration_timeout(self) -> None:
        """semantic migration (heaviest DDL: fp32 vec table + HNSW) carries the timeout."""
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)

        async def mock_execute_read(operation, *_args, **_kwargs):
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=None)
            return await operation(mock_conn)

        mock_backend.execute_read = mock_execute_read

        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = True
        mock_settings.embedding.dim = 768
        mock_settings.compression.enabled = False
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0
        mock_settings.storage.postgresql_schema = 'public'

        with patch('app.migrations.semantic.settings', mock_settings):
            from app.migrations.semantic import apply_semantic_search_migration

            await apply_semantic_search_migration(backend=mock_backend)

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_fts_migrate_language_ddl_uses_migration_timeout(self) -> None:
        """The PostgreSQL FTS language-change rewrite carries the migration timeout.

        This heaviest-DDL path (DROP + ADD GENERATED ALWAYS AS (...) STORED tsvector
        full-table rewrite + GIN index build) lives in the repository layer
        (FtsRepository.migrate_language), not app/migrations, and by definition runs on
        an already-populated table -- so it must raise the budget via begin_migration the
        same way the initial FTS migration does, or a slow rewrite is canceled at the
        pool's shorter command_timeout and the language change silently never applies.
        """
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)

        mock_settings = MagicMock()
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0

        from app.repositories.fts_repository import FtsRepository

        repo = FtsRepository(mock_backend)
        with (
            patch('app.settings.get_settings', return_value=mock_settings),
            patch.object(FtsRepository, 'get_current_language', AsyncMock(return_value='english')),
        ):
            await repo.migrate_language('german')

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_compression_migration_ddl_uses_migration_timeout(self) -> None:
        """compression migration DDL loop carries the migration timeout."""
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)
        # Reads in order: the enable-guard's provenance read (None = no sealed
        # row) and its fp32 data probe (False = no populated fp32 table, guard
        # passes), then the already-applied check (False = first-time).
        mock_backend.execute_read = AsyncMock(side_effect=[None, False, False])

        mock_settings = MagicMock()
        mock_settings.compression.enabled = True
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0
        mock_settings.storage.postgresql_schema = 'public'

        with patch('app.migrations.compression.settings', mock_settings):
            from app.migrations.compression import apply_compression_migration

            await apply_compression_migration(backend=mock_backend)

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_compression_already_applied_fingerprint_ddl_uses_migration_timeout(self) -> None:
        """The already-applied branch's fingerprint ALTER carries the migration timeout.

        This branch runs on EVERY steady-state startup of a compressed
        PostgreSQL deployment; its ALTER must go through the same advisory
        lock + SET LOCAL discipline as the first-time path, which routes the
        identical statement through execute_migration_ddl.
        """
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)
        # Reads in order: the enable-guard's provenance read (a sealed row
        # exists -- returned as the raw 6-column tuple read_compression_metadata
        # unpacks -- so the guard's fp32 probe short-circuits away) and the
        # already-applied check (True = both tables present).
        mock_backend.execute_read = AsyncMock(
            side_effect=[('turboquant', 4, 'ip', 0, 1024, None), True],
        )

        mock_settings = MagicMock()
        mock_settings.compression.enabled = True
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0
        mock_settings.storage.postgresql_schema = 'public'

        with patch('app.migrations.compression.settings', mock_settings):
            from app.migrations.compression import apply_compression_migration

            await apply_compression_migration(backend=mock_backend)

        assert any('codebook_fingerprint' in stmt for stmt, _t in executed)
        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_metadata_index_create_ddl_uses_migration_timeout(self) -> None:
        """metadata index CREATE + advisory lock carry the migration timeout."""
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)

        mock_settings = MagicMock()
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0
        mock_settings.storage.postgresql_schema = 'public'

        with patch('app.migrations.metadata.settings', mock_settings):
            from app.migrations.metadata import _create_metadata_index

            await _create_metadata_index(backend=mock_backend, field='test_field', type_hint='string')

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_metadata_index_drop_ddl_uses_migration_timeout(self) -> None:
        """metadata index DROP + advisory lock carry the migration timeout."""
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)

        mock_settings = MagicMock()
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0
        mock_settings.storage.postgresql_schema = 'public'

        with patch('app.migrations.metadata.settings', mock_settings):
            from app.migrations.metadata import _drop_metadata_index

            await _drop_metadata_index(backend=mock_backend, field='test_field')

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_fts_initial_migration_ddl_uses_migration_timeout(self) -> None:
        """FTS initial PG migration (heaviest DDL: STORED tsvector rewrite + GIN) carries the timeout."""
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)

        mock_settings = MagicMock()
        mock_settings.fts.language = 'english'
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0

        with patch('app.migrations.fts.settings', mock_settings):
            from app.migrations.fts import _apply_initial_fts_migration

            await _apply_initial_fts_migration(mock_backend, 'postgresql')

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_version_migration_ddl_uses_migration_timeout(self) -> None:
        """version-column PG migration takes the shared lock + ADD COLUMN under the migration timeout."""
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)

        mock_settings = MagicMock()
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0

        with patch('app.migrations.version.settings', mock_settings):
            from app.migrations.version import apply_version_migration

            await apply_version_migration(backend=mock_backend)

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_content_hash_migration_ddl_uses_migration_timeout(self) -> None:
        """content_hash PG migration takes the shared lock + ADD COLUMN/INDEX under the migration timeout."""
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)

        mock_settings = MagicMock()
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0

        with patch('app.migrations.content_hash.settings', mock_settings):
            from app.migrations.content_hash import apply_content_hash_migration

            await apply_content_hash_migration(backend=mock_backend)

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_summary_migration_ddl_uses_migration_timeout(self) -> None:
        """summary-column PG migration takes the shared lock + ADD COLUMN under the migration timeout."""
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)

        mock_settings = MagicMock()
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0

        with patch('app.migrations.summary.settings', mock_settings):
            from app.migrations.summary import apply_summary_migration

            await apply_summary_migration(backend=mock_backend)

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_jsonb_merge_patch_migration_ddl_uses_migration_timeout(self) -> None:
        """jsonb_merge_patch PG function migration takes the shared lock + DDL under the timeout."""
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)
        # function_exists check + post-migration verification both read True
        mock_backend.execute_read = AsyncMock(return_value=True)

        mock_settings = MagicMock()
        mock_settings.storage.postgresql_schema = 'public'
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0

        with patch('app.migrations.semantic.settings', mock_settings):
            from app.migrations.semantic import apply_jsonb_merge_patch_migration

            await apply_jsonb_merge_patch_migration(backend=mock_backend)

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)

    @pytest.mark.asyncio
    async def test_function_search_path_migration_ddl_uses_migration_timeout(self) -> None:
        """function search_path PG migration takes the shared lock + DDL under the migration timeout."""
        executed: list[tuple[str, float | None]] = []
        mock_backend = self._recording_pg_backend(executed)

        mock_settings = MagicMock()
        mock_settings.storage.postgresql_schema = 'public'
        mock_settings.storage.postgresql_migration_timeout_s = 300.0
        mock_settings.storage.postgresql_command_timeout_s = 60.0

        with patch('app.migrations.semantic.settings', mock_settings):
            from app.migrations.semantic import apply_function_search_path_migration

            await apply_function_search_path_migration(backend=mock_backend)

        self._assert_ddl_carries_timeout(executed, migration_timeout=300.0)
        self._assert_set_local_no_finally_restore(executed)
