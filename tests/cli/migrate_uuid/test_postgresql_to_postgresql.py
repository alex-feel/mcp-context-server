"""Tests for the PostgreSQL-to-PostgreSQL migration runner: connection cleanup and the pgvector dimension pre-flight."""

from unittest import mock

import pytest

from app.cli.migrate_uuid.records import MigrationOptions


@pytest.mark.asyncio
async def test_pg_pg_migration_closes_source_when_target_connect_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed target connect closes the source and propagates the original error.

    Mutation guard for run_migration_postgresql's None-guarded target connection:
    target_conn is opened INSIDE the try with a None-guarded finally, so a target
    connect failure does NOT leak the already-open source connection and does NOT
    raise UnboundLocalError / AttributeError from a None.close() in the finally.
    """
    import asyncpg as asyncpg_mod

    from app.cli.migrate_uuid.postgresql_to_postgresql import run_migration_postgresql

    closed = {'source': False}

    class _FakeConn:
        async def execute(self, _sql: str, *_args: object) -> str:
            """Accept the session-parameter statements applied right after the dial."""
            return 'SET'

        async def close(self) -> None:
            closed['source'] = True

    calls = {'n': 0}

    async def _fake_connect(_url: str, **_kwargs: object) -> _FakeConn:
        calls['n'] += 1
        if calls['n'] == 1:
            return _FakeConn()  # source connection succeeds
        raise OSError('target unreachable')  # target connect fails

    monkeypatch.setattr(asyncpg_mod, 'connect', _fake_connect)

    options = MigrationOptions(
        source_url='postgresql://u:p@localhost/src',
        target_url='postgresql://u:p@localhost/tgt',
    )
    with pytest.raises(OSError, match='target unreachable'):
        await run_migration_postgresql(options)
    # The source was closed by the finally; no UnboundLocalError / None.close().
    assert closed['source'] is True


class _FakeMigrationSourceConn:
    """Minimal stand-in for the read-only PG->PG source connection."""

    async def execute(self, _sql: str) -> None:
        return None

    async def fetchval(self, sql: str) -> object:
        if 'COUNT(*)' in sql:
            return 0
        return 'integer'  # integer-keyed id column, so the migration proceeds

    async def fetch(self, _sql: str) -> list[object]:
        return []

    async def close(self) -> None:
        return None


class _FakeMigrationTargetConn:
    """Minimal stand-in for the PG->PG target connection."""

    async def execute(self, _sql: str) -> None:
        return None

    async def close(self) -> None:
        return None


def _install_pg_runner_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_dim: int | None,
) -> mock.AsyncMock:
    """Fake the PG->PG runner's connections and probes up to the auto-init decision.

    Shapes the probes as: empty target (no context_entries), embedding-carrying
    source (embedding_metadata + vec_context_embeddings present, no FTS), with
    the given source-detected embedding dimension. ``source_dim=None`` models an
    embedding_metadata table that exists but is empty (the settings-fallback path).

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        source_dim: Dimension returned by the faked source-dim detection, or None
            for an empty embedding_metadata table.

    Returns:
        The AsyncMock installed in place of ``initialize_target_postgresql``.
    """
    import asyncpg as asyncpg_mod

    import app.cli.migrate_uuid.postgresql_to_postgresql as runner_mod

    conns: list[object] = [_FakeMigrationSourceConn(), _FakeMigrationTargetConn()]

    async def _fake_connect(_url: str, **_kwargs: object) -> object:
        return conns.pop(0)

    async def _fake_has_data(
        _conn: object, *, schema: str | None = None,
    ) -> bool:
        del schema
        return False

    async def _fake_table_exists(
        _conn: object, table_name: str, schema: str | None = None,
    ) -> bool:
        del schema
        if table_name == 'context_entries':
            return False  # the target is uninitialized -> auto-init path
        return table_name in {'embedding_metadata', 'vec_context_embeddings'}

    async def _fake_column_exists(
        _conn: object, _table: str, _column: str, schema: str | None = None,
    ) -> bool:
        del schema
        return False

    async def _fake_detect_dim(_conn: object) -> int | None:
        return source_dim

    init_mock = mock.AsyncMock()
    monkeypatch.setattr(asyncpg_mod, 'connect', _fake_connect)
    monkeypatch.setattr(runner_mod, 'target_pg_has_data', _fake_has_data)
    monkeypatch.setattr(runner_mod, 'pg_table_exists', _fake_table_exists)
    monkeypatch.setattr(runner_mod, 'pg_column_exists', _fake_column_exists)
    monkeypatch.setattr(runner_mod, 'detect_source_embedding_dim_pg', _fake_detect_dim)
    monkeypatch.setattr(runner_mod, 'initialize_target_postgresql', init_mock)
    return init_mock


@pytest.mark.asyncio
async def test_pg_pg_migration_refuses_auto_init_above_pgvector_index_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source dim above the pgvector index cap aborts with a recorded error before any target DDL.

    Without the pre-flight the auto-init would force-build the fp32 layout at
    the source-detected dimension and crash at the HNSW CREATE INDEX, exiting
    with a partially initialized target schema.
    """
    from app.cli.migrate_uuid.postgresql_to_postgresql import run_migration_postgresql
    from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT

    init_mock = _install_pg_runner_fakes(
        monkeypatch, source_dim=PGVECTOR_INDEX_DIM_LIMIT + 1,
    )
    options = MigrationOptions(
        source_url='postgresql://u:p@localhost/src',
        target_url='postgresql://u:p@localhost/tgt',
    )

    stats = await run_migration_postgresql(options)

    assert any('pgvector index limit' in e for e in stats.errors)
    assert any('Aborting before any target DDL' in e for e in stats.errors)
    init_mock.assert_not_awaited()
    assert stats.rows_migrated == 0


@pytest.mark.asyncio
async def test_pg_pg_dry_run_reports_pgvector_index_cap_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dry run surfaces the dimension-cap refusal as a plan warning, not an error."""
    from app.cli.migrate_uuid.postgresql_to_postgresql import run_migration_postgresql
    from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT

    init_mock = _install_pg_runner_fakes(
        monkeypatch, source_dim=PGVECTOR_INDEX_DIM_LIMIT + 1,
    )
    options = MigrationOptions(
        source_url='postgresql://u:p@localhost/src',
        target_url='postgresql://u:p@localhost/tgt',
        dry_run=True,
    )

    stats = await run_migration_postgresql(options)

    assert not stats.errors
    assert any('a real run would abort' in w for w in stats.warnings)
    assert any('pgvector index limit' in w for w in stats.warnings)
    init_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_pg_pg_migration_refuses_over_limit_settings_fallback_when_source_dim_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty source embedding_metadata table with an over-limit settings fallback aborts cleanly.

    The source carries embeddings (so with_semantic is true) but the
    embedding_metadata table is empty, so source_dim detects as None and the
    auto-init's semantic migration would fall back to settings.embedding.dim for
    the vector(dim) DDL. A pre-flight checking source_dim alone would let the run
    reach CREATE INDEX with an over-limit fallback and crash mid-pipeline,
    leaving the target partially initialized. The pre-flight resolves and
    validates the fallback, recording a clean error before any target DDL.
    """
    from app.cli.migrate_uuid.postgresql_to_postgresql import run_migration_postgresql
    from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT
    from app.settings import get_settings

    init_mock = _install_pg_runner_fakes(monkeypatch, source_dim=None)
    options = MigrationOptions(
        source_url='postgresql://u:p@localhost/src',
        target_url='postgresql://u:p@localhost/tgt',
    )

    # ENABLE_EMBEDDING_GENERATION=false is the typical migrate-CLI case and keeps
    # the settings-level pgvector guard from rejecting the over-limit EMBEDDING_DIM
    # at construction, so the CLI pre-flight is what must catch it.
    monkeypatch.setenv('EMBEDDING_DIM', str(PGVECTOR_INDEX_DIM_LIMIT + 1))
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    try:
        stats = await run_migration_postgresql(options)
    finally:
        monkeypatch.delenv('EMBEDDING_DIM', raising=False)
        monkeypatch.delenv('ENABLE_EMBEDDING_GENERATION', raising=False)
        get_settings.cache_clear()

    assert any('pgvector index limit' in e for e in stats.errors)
    assert any('Aborting before any target DDL' in e for e in stats.errors)
    # The message identifies EMBEDDING_DIM as the fallback the DDL would template.
    assert any('EMBEDDING_DIM' in e for e in stats.errors)
    init_mock.assert_not_awaited()
    assert stats.rows_migrated == 0


@pytest.mark.asyncio
async def test_pg_pg_dry_run_reports_over_limit_settings_fallback_when_source_dim_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dry run surfaces the settings-fallback dimension-cap refusal for an empty source table."""
    from app.cli.migrate_uuid.postgresql_to_postgresql import run_migration_postgresql
    from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT
    from app.settings import get_settings

    init_mock = _install_pg_runner_fakes(monkeypatch, source_dim=None)
    options = MigrationOptions(
        source_url='postgresql://u:p@localhost/src',
        target_url='postgresql://u:p@localhost/tgt',
        dry_run=True,
    )

    monkeypatch.setenv('EMBEDDING_DIM', str(PGVECTOR_INDEX_DIM_LIMIT + 1))
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    try:
        stats = await run_migration_postgresql(options)
    finally:
        monkeypatch.delenv('EMBEDDING_DIM', raising=False)
        monkeypatch.delenv('ENABLE_EMBEDDING_GENERATION', raising=False)
        get_settings.cache_clear()

    assert not stats.errors
    assert any('a real run would abort' in w for w in stats.warnings)
    assert any('pgvector index limit' in w for w in stats.warnings)
    assert any('EMBEDDING_DIM' in w for w in stats.warnings)
    init_mock.assert_not_awaited()
