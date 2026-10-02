"""Tests for the PostgreSQL target initialization and its full-text search backstop."""

from unittest import mock

import pytest

from app.cli.migrate_uuid.records import MigrationStats


class TestEnsureTargetPgFts:
    """FTS backstop for a PRE-EXISTING PostgreSQL target.

    ``initialize_target_postgresql`` provisions FTS only when the target had
    no ``context_entries`` table at all; a pre-existing target (bootstrapped
    by a server run with ``ENABLE_FTS=false``, or by any means other than
    this CLI) would silently lose the source's full-text search -- the same
    loss the source-presence gate prevents for freshly initialized
    targets. Unlike embeddings (not derivable -> abort), FTS is fully
    derivable from the copied rows, so the backstop provisions it.
    """

    @staticmethod
    def _column_probe_conn(has_fts_column: bool) -> object:
        """Build a fake target connection answering the column probe.

        Args:
            has_fts_column: Whether the probe reports text_search_vector.

        Returns:
            An object exposing the async ``fetchval`` surface the probe uses.
        """

        class _Conn:
            async def fetchval(self, query: str, *args: object) -> bool:
                del query, args
                return has_fts_column

        return _Conn()

    @pytest.mark.asyncio
    async def test_noop_when_source_has_no_fts(self) -> None:
        """No probe, no provisioning, no note when the source lacks FTS."""
        from typing import Any
        from typing import cast

        from app.cli.migrate_uuid.pg_target import ensure_target_pg_fts

        stats = MigrationStats()
        await ensure_target_pg_fts(
            'postgresql://ignored', cast(Any, object()),
            target_schema='public', source_has_fts=False,
            dry_run=False, stats=stats,
        )
        assert stats.warnings == []
        assert stats.errors == []

    @pytest.mark.asyncio
    async def test_noop_when_target_already_has_fts(self) -> None:
        """A target already carrying text_search_vector is left untouched."""
        from typing import Any
        from typing import cast

        from app.cli.migrate_uuid.pg_target import ensure_target_pg_fts

        stats = MigrationStats()
        await ensure_target_pg_fts(
            'postgresql://ignored', cast(Any, self._column_probe_conn(True)),
            target_schema='public', source_has_fts=True,
            dry_run=False, stats=stats,
        )
        assert stats.warnings == []

    @pytest.mark.asyncio
    async def test_dry_run_records_the_provisioning_plan(self) -> None:
        """A dry run records the plan instead of touching the target."""
        from typing import Any
        from typing import cast

        from app.cli.migrate_uuid.pg_target import ensure_target_pg_fts

        stats = MigrationStats()
        await ensure_target_pg_fts(
            'postgresql://ignored', cast(Any, self._column_probe_conn(False)),
            target_schema='public', source_has_fts=True,
            dry_run=True, stats=stats,
        )
        assert any('would be provisioned' in w for w in stats.warnings)
        assert stats.errors == []

    @pytest.mark.asyncio
    async def test_real_run_applies_the_fts_migration_with_force(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A real run applies apply_fts_migration(force=True) against the target."""
        from typing import Any
        from typing import cast
        from unittest.mock import AsyncMock

        import app.backends as backends_module
        import app.migrations.fts as fts_module
        from app.cli.migrate_uuid.pg_target import ensure_target_pg_fts

        fake_backend = AsyncMock()
        monkeypatch.setattr(
            backends_module, 'create_backend', lambda **_kw: fake_backend,
        )
        apply_spy = AsyncMock()
        monkeypatch.setattr(fts_module, 'apply_fts_migration', apply_spy)

        stats = MigrationStats()
        await ensure_target_pg_fts(
            'postgresql://ignored', cast(Any, self._column_probe_conn(False)),
            target_schema='public', source_has_fts=True,
            dry_run=False, stats=stats,
        )

        apply_spy.assert_awaited_once_with(fake_backend, force=True)
        fake_backend.initialize.assert_awaited_once()
        fake_backend.shutdown.assert_awaited_once()
        assert any('provisioned full-text search' in w for w in stats.warnings)


class TestInitializeTargetPostgresqlVectorGating:
    """initialize_target_postgresql touches pgvector only for a vector-carrying target.

    A cross-backend migration drops embeddings (with_semantic=False), so its target
    init must complete on a pgvector-less PostgreSQL host: no CREATE EXTENSION, and
    the backend built for the migrations must not let the CLI process's env-driven
    gate force pgvector provisioning. The schema, by contrast, is always created
    (schema-qualified function DDL needs it), and it must not be skipped when a
    later extension statement fails.
    """

    @staticmethod
    def _install_pipeline_mocks(
        monkeypatch: pytest.MonkeyPatch,
        executed: list[str],
        extension_error: Exception | None = None,
        schema_error: Exception | None = None,
    ) -> dict[str, mock.AsyncMock | mock.MagicMock]:
        """Fake the target connection, backend factory, and migration pipeline.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
            executed: Mutated with each SQL statement the fake connection runs.
            extension_error: When set, raised by the fake CREATE EXTENSION.
            schema_error: When set, raised by the fake CREATE SCHEMA.

        Returns:
            Mapping of mock names to the installed mock objects.
        """
        import asyncpg as asyncpg_mod

        import app.backends as backends_mod
        import app.migrations.chunking as chunking_mod
        import app.migrations.fts as fts_mod
        import app.migrations.index_tree as index_tree_mod
        import app.migrations.semantic as semantic_mod
        import app.startup as startup_mod

        class _FakeExtConn:
            async def execute(self, sql: str) -> None:
                if schema_error is not None and 'CREATE SCHEMA' in sql:
                    raise schema_error
                if extension_error is not None and 'CREATE EXTENSION' in sql:
                    raise extension_error
                executed.append(sql)

            async def close(self) -> None:
                return None

        async def _fake_connect(_url: str, **_kwargs: object) -> _FakeExtConn:
            return _FakeExtConn()

        monkeypatch.setattr(asyncpg_mod, 'connect', _fake_connect)

        backend = mock.MagicMock()
        backend.initialize = mock.AsyncMock()
        backend.shutdown = mock.AsyncMock()
        mocks: dict[str, mock.AsyncMock | mock.MagicMock] = {
            'create_backend': mock.MagicMock(return_value=backend),
            'init_database': mock.AsyncMock(),
            'semantic': mock.AsyncMock(),
            'jsonb': mock.AsyncMock(),
            'search_path': mock.AsyncMock(),
            'fts': mock.AsyncMock(),
            'chunking': mock.AsyncMock(),
            'index_tree': mock.AsyncMock(),
        }
        monkeypatch.setattr(backends_mod, 'create_backend', mocks['create_backend'])
        monkeypatch.setattr(startup_mod, 'init_database', mocks['init_database'])
        monkeypatch.setattr(semantic_mod, 'apply_semantic_search_migration', mocks['semantic'])
        monkeypatch.setattr(semantic_mod, 'apply_jsonb_merge_patch_migration', mocks['jsonb'])
        monkeypatch.setattr(semantic_mod, 'apply_function_search_path_migration', mocks['search_path'])
        monkeypatch.setattr(fts_mod, 'apply_fts_migration', mocks['fts'])
        monkeypatch.setattr(chunking_mod, 'apply_chunking_migration', mocks['chunking'])
        monkeypatch.setattr(index_tree_mod, 'apply_index_tree_migration', mocks['index_tree'])
        return mocks

    @pytest.mark.asyncio
    async def test_without_semantic_never_touches_pgvector(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A with_semantic=False init issues no CREATE EXTENSION and disables provisioning."""
        from app.cli.migrate_uuid.pg_target import initialize_target_postgresql

        executed: list[str] = []
        mocks = self._install_pipeline_mocks(monkeypatch, executed)
        stats = MigrationStats()

        await initialize_target_postgresql(
            'postgresql://u:p@localhost:5432/tgt',
            embedding_dim=None,
            with_semantic=False,
            source_has_fts=False,
            stats=stats,
        )

        assert any('CREATE SCHEMA' in sql for sql in executed)
        assert not any('CREATE EXTENSION' in sql for sql in executed)
        assert mocks['create_backend'].call_args.kwargs['provision_vector'] is False
        mocks['semantic'].assert_not_called()
        mocks['chunking'].assert_not_called()

    @pytest.mark.asyncio
    async def test_with_semantic_creates_schema_then_extension(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A with_semantic=True init creates the schema first, then the extension."""
        from app.cli.migrate_uuid.pg_target import initialize_target_postgresql

        executed: list[str] = []
        mocks = self._install_pipeline_mocks(monkeypatch, executed)
        stats = MigrationStats()

        await initialize_target_postgresql(
            'postgresql://u:p@localhost:5432/tgt',
            embedding_dim=1024,
            with_semantic=True,
            source_has_fts=True,
            stats=stats,
        )

        schema_idx = next(i for i, sql in enumerate(executed) if 'CREATE SCHEMA' in sql)
        ext_idx = next(i for i, sql in enumerate(executed) if 'CREATE EXTENSION' in sql)
        assert schema_idx < ext_idx
        assert mocks['create_backend'].call_args.kwargs['provision_vector'] is True
        mocks['semantic'].assert_called_once()
        mocks['chunking'].assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_pgvector_extension_raises_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """pgvector absent from the host (58P01) surfaces actionable guidance, not a traceback."""
        import asyncpg as asyncpg_mod

        from app.cli.migrate_uuid.pg_target import initialize_target_postgresql

        executed: list[str] = []
        self._install_pipeline_mocks(
            monkeypatch,
            executed,
            extension_error=asyncpg_mod.UndefinedFileError('could not open extension control file'),
        )
        stats = MigrationStats()

        with pytest.raises(RuntimeError, match='pgvector extension is not installed'):
            await initialize_target_postgresql(
                'postgresql://u:p@localhost:5432/tgt',
                embedding_dim=1024,
                with_semantic=True,
                source_has_fts=False,
                stats=stats,
            )
        # The schema was still created before the extension failure.
        assert any('CREATE SCHEMA' in sql for sql in executed)

    @pytest.mark.asyncio
    async def test_schema_privilege_failure_raises_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unprivileged CREATE SCHEMA surfaces guidance naming the schema statement."""
        import asyncpg as asyncpg_mod

        from app.cli.migrate_uuid.pg_target import initialize_target_postgresql

        executed: list[str] = []
        self._install_pipeline_mocks(
            monkeypatch,
            executed,
            schema_error=asyncpg_mod.InsufficientPrivilegeError('permission denied'),
        )
        stats = MigrationStats()

        with pytest.raises(RuntimeError, match='CREATE SCHEMA'):
            await initialize_target_postgresql(
                'postgresql://u:p@localhost:5432/tgt',
                embedding_dim=None,
                with_semantic=False,
                source_has_fts=False,
                stats=stats,
            )

    @pytest.mark.asyncio
    async def test_semantic_init_refuses_dim_above_pgvector_index_cap(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An embedding_dim pgvector cannot index fails fast BEFORE any target DDL.

        pgvector caps HNSW index dimensionality at 2000, and the semantic
        migration templates the source-detected dimension into vector(dim) and
        then builds that index; without the pre-flight the pipeline would die
        at CREATE INDEX and leave the target schema partially initialized.
        """
        from app.cli.migrate_uuid.pg_target import initialize_target_postgresql
        from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT

        executed: list[str] = []
        mocks = self._install_pipeline_mocks(monkeypatch, executed)
        stats = MigrationStats()

        with pytest.raises(RuntimeError, match='pgvector index limit'):
            await initialize_target_postgresql(
                'postgresql://u:p@localhost:5432/tgt',
                embedding_dim=PGVECTOR_INDEX_DIM_LIMIT + 1,
                with_semantic=True,
                source_has_fts=False,
                stats=stats,
            )

        # No DDL of any kind ran: no schema/extension statements, no backend,
        # no base-schema init, no migrations.
        assert executed == []
        mocks['create_backend'].assert_not_called()
        mocks['init_database'].assert_not_awaited()
        mocks['semantic'].assert_not_awaited()
        mocks['chunking'].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_semantic_init_allows_dim_at_pgvector_index_cap(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An embedding_dim exactly at the pgvector index cap initializes normally."""
        from app.cli.migrate_uuid.pg_target import initialize_target_postgresql
        from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT

        executed: list[str] = []
        mocks = self._install_pipeline_mocks(monkeypatch, executed)
        stats = MigrationStats()

        await initialize_target_postgresql(
            'postgresql://u:p@localhost:5432/tgt',
            embedding_dim=PGVECTOR_INDEX_DIM_LIMIT,
            with_semantic=True,
            source_has_fts=False,
            stats=stats,
        )

        assert any('CREATE EXTENSION' in sql for sql in executed)
        mocks['semantic'].assert_awaited_once()

    @pytest.mark.asyncio
    async def test_semantic_init_refuses_over_limit_settings_fallback_when_dim_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """embedding_dim=None (empty source embedding_metadata) validates the settings fallback.

        When the source embedding_metadata table exists but is empty, the source
        dimension detects as None and the semantic migration falls back to
        settings.embedding.dim for the vector(dim) DDL. Validating only the
        source-detected dim would leave that fallback -- the value the DDL
        actually templates -- unchecked and crash mid-index. The pre-flight
        resolves the same fallback and refuses it before any target DDL.
        """
        from app.cli.migrate_uuid.pg_target import initialize_target_postgresql
        from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT
        from app.settings import get_settings

        executed: list[str] = []
        mocks = self._install_pipeline_mocks(monkeypatch, executed)
        stats = MigrationStats()

        # ENABLE_EMBEDDING_GENERATION=false is the typical migrate-CLI case and
        # keeps the settings-level pgvector guard from rejecting the over-limit
        # EMBEDDING_DIM at construction, so the CLI pre-flight is what must catch it.
        monkeypatch.setenv('EMBEDDING_DIM', str(PGVECTOR_INDEX_DIM_LIMIT + 1))
        monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
        get_settings.cache_clear()
        try:
            with pytest.raises(RuntimeError, match='pgvector index limit') as exc_info:
                await initialize_target_postgresql(
                    'postgresql://u:p@localhost:5432/tgt',
                    embedding_dim=None,
                    with_semantic=True,
                    source_has_fts=False,
                    stats=stats,
                )
        finally:
            monkeypatch.delenv('EMBEDDING_DIM', raising=False)
            monkeypatch.delenv('ENABLE_EMBEDDING_GENERATION', raising=False)
            get_settings.cache_clear()

        # The message names EMBEDDING_DIM as the fallback source, not a source dim.
        assert 'EMBEDDING_DIM' in str(exc_info.value)

        # No DDL of any kind ran: the refusal precedes every connection and migration.
        assert executed == []
        mocks['create_backend'].assert_not_called()
        mocks['init_database'].assert_not_awaited()
        mocks['semantic'].assert_not_awaited()
        mocks['chunking'].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_semantic_init_allows_in_limit_settings_fallback_when_dim_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """embedding_dim=None with an in-limit settings fallback initializes normally.

        The reported dimension and the dimension handed to the semantic migration must
        both be the RESOLVED fallback: the vector column is built at that width, so a
        report naming the raw ``None`` would leave the operator's only record of an
        irreversible schema decision stating the dimension is unknown.
        """
        from app.cli.migrate_uuid.pg_target import initialize_target_postgresql
        from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT
        from app.settings import get_settings

        executed: list[str] = []
        mocks = self._install_pipeline_mocks(monkeypatch, executed)
        stats = MigrationStats()

        monkeypatch.setenv('EMBEDDING_DIM', str(PGVECTOR_INDEX_DIM_LIMIT))
        monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
        get_settings.cache_clear()
        try:
            await initialize_target_postgresql(
                'postgresql://u:p@localhost:5432/tgt',
                embedding_dim=None,
                with_semantic=True,
                source_has_fts=False,
                stats=stats,
            )
        finally:
            monkeypatch.delenv('EMBEDDING_DIM', raising=False)
            monkeypatch.delenv('ENABLE_EMBEDDING_GENERATION', raising=False)
            get_settings.cache_clear()

        assert any('CREATE EXTENSION' in sql for sql in executed)
        mocks['semantic'].assert_awaited_once()
        assert mocks['semantic'].await_args is not None
        assert mocks['semantic'].await_args.kwargs['embedding_dim'] == PGVECTOR_INDEX_DIM_LIMIT

        init_warnings = [w for w in stats.warnings if 'auto-initialized target PostgreSQL' in w]
        assert len(init_warnings) == 1
        assert f'embedding_dim={PGVECTOR_INDEX_DIM_LIMIT}' in init_warnings[0]
        assert 'embedding_dim=None' not in init_warnings[0]

    @pytest.mark.asyncio
    async def test_semantic_init_reports_the_source_dimension_when_known(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A known source dimension is reported verbatim and passed to the migration."""
        from app.cli.migrate_uuid.pg_target import initialize_target_postgresql

        executed: list[str] = []
        mocks = self._install_pipeline_mocks(monkeypatch, executed)
        stats = MigrationStats()

        await initialize_target_postgresql(
            'postgresql://u:p@localhost:5432/tgt',
            embedding_dim=384,
            with_semantic=True,
            source_has_fts=False,
            stats=stats,
        )

        assert mocks['semantic'].await_args is not None
        assert mocks['semantic'].await_args.kwargs['embedding_dim'] == 384
        assert any('embedding_dim=384' in w for w in stats.warnings)

    @pytest.mark.asyncio
    async def test_semantic_free_init_reports_no_dimension(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A with_semantic=False run builds no vector column and reports n/a."""
        from app.cli.migrate_uuid.pg_target import initialize_target_postgresql

        executed: list[str] = []
        mocks = self._install_pipeline_mocks(monkeypatch, executed)
        stats = MigrationStats()

        await initialize_target_postgresql(
            'postgresql://u:p@localhost:5432/tgt',
            embedding_dim=None,
            with_semantic=False,
            source_has_fts=False,
            stats=stats,
        )

        mocks['semantic'].assert_not_awaited()
        assert any('embedding_dim=n/a' in w for w in stats.warnings)
