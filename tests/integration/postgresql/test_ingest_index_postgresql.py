"""The ingest index migration on a live PostgreSQL server.

The server's startup preparation creates the partial index keyed on the thread, the
``C``-collated source id and the id, and the extended statistics on the thread and the
block's scope, which ``ANALYZE`` fills with the most common pairs, their dependency and
the scope expression's own distribution. A later application finds both through its
probes and takes neither the migration lock nor any DDL; once either is dropped, the
next application builds it again under the lock, and new statistics are sampled at once.
"""

from collections.abc import AsyncGenerator
from typing import cast

import asyncpg
import pytest
import pytest_asyncio

import app.migrations.ingest as ingest_module
from app.ingest_sql import INGEST_INDEX_NAME
from app.ingest_sql import ingest_index_ddl
from app.migrations import apply_ingest_index_migration
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations.ingest import INGEST_STATISTICS_NAME
from app.migrations.ingest import ingest_statistics_ddl
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import postgresql_scoped_db

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

ANALYZE = 'ANALYZE context_entries'
MANAGED_ROWS = 400

# Managed entries of one corpus thread, so the sample holds pairs of a thread and a scope.
_SEED_MANAGED_SQL = (
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility, metadata) '
    "SELECT gen_random_uuid(), 'kb:stats', 'agent', 'text', 'Document ' || g, 'alice', 'private', "
    "jsonb_build_object('ingest', jsonb_build_object('scope', 'kb:stats', 'source_id', 'doc-' || g, "
    "'meta_digest', repeat('0', 64), 'synced_at', '2026-10-18T12:00:00Z')) "
    f'FROM generate_series(1, {MANAGED_ROWS}) AS g'
)


@pytest_asyncio.fixture
async def prepared_db(pg_test_url: str, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[ScopedDb]:
    """An isolated database prepared by the server's startup preparation."""
    async with postgresql_scoped_db(pg_test_url, 'fp32', monkeypatch) as db:
        yield db


def _spy_on_migration_steps(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every migration lock and DDL statement of the migration, still running each one."""
    steps: list[str] = []

    async def _begin(conn: asyncpg.Connection, timeout_s: float) -> None:
        steps.append('begin_migration')
        await begin_migration(conn, timeout_s)

    async def _ddl(conn: asyncpg.Connection, statement: str, timeout_s: float) -> None:
        steps.append(statement)
        await execute_migration_ddl(conn, statement, timeout_s)

    monkeypatch.setattr(ingest_module, 'begin_migration', _begin)
    monkeypatch.setattr(ingest_module, 'execute_migration_ddl', _ddl)
    return steps


async def _index_definitions(db: ScopedDb) -> list[str]:
    """Return the definition of every index named like the ingest index on ``context_entries``."""
    rows = await db.fetch_all(
        "SELECT indexdef FROM pg_indexes WHERE tablename = 'context_entries' AND indexname = ?", (INGEST_INDEX_NAME,),
    )
    return [str(row[0]) for row in rows]


async def _statistics_kinds(db: ScopedDb) -> list[list[str]]:
    """Return the kinds of every statistics object named like the ingest statistics on the resolved ``context_entries``."""
    rows = await db.fetch_all(
        "SELECT stxkind::text[] FROM pg_statistic_ext WHERE stxrelid = to_regclass('context_entries') AND stxname = ?",
        (INGEST_STATISTICS_NAME,),
    )
    return [sorted(cast('list[str]', row[0])) for row in rows]


@pytest.mark.asyncio
async def test_startup_creates_the_partial_index_on_the_collated_source_id(prepared_db: ScopedDb) -> None:
    definitions = await _index_definitions(prepared_db)

    assert len(definitions) == 1
    definition = definitions[0]
    assert "USING btree (thread_id, ((metadata #>> '{ingest,source_id}'::text[])) COLLATE \"C\", id)" in definition
    assert definition.endswith("WHERE ((metadata #>> '{ingest,source_id}'::text[]) IS NOT NULL)")


@pytest.mark.asyncio
async def test_startup_creates_the_statistics_on_the_thread_and_the_scope(prepared_db: ScopedDb) -> None:
    """The object keeps most common values and dependencies; the scope expression adds statistics of its own."""
    assert await _statistics_kinds(prepared_db) == [['e', 'f', 'm']]


@pytest.mark.asyncio
async def test_analyze_fills_the_statistics_once_managed_rows_exist(prepared_db: ScopedDb) -> None:
    await prepared_db.execute(_SEED_MANAGED_SQL)
    await prepared_db.execute(ANALYZE)

    pairs = await prepared_db.fetch_all(
        'SELECT most_common_vals::text, dependencies::text FROM pg_stats_ext '
        "WHERE tablename = 'context_entries' AND statistics_name = ?",
        (INGEST_STATISTICS_NAME,),
    )
    scopes = await prepared_db.fetch_all(
        'SELECT expr, null_frac, most_common_vals::text FROM pg_stats_ext_exprs '
        "WHERE tablename = 'context_entries' AND statistics_name = ?",
        (INGEST_STATISTICS_NAME,),
    )

    assert len(pairs) == 1
    most_common_pairs, dependencies = pairs[0]
    assert 'kb:stats' in str(most_common_pairs)
    assert dependencies is not None
    assert len(scopes) == 1
    expression, null_fraction, most_common_scopes = scopes[0]
    assert "'{ingest,scope}'" in str(expression)
    assert 0.0 < cast(float, null_fraction) < 1.0
    assert 'kb:stats' in str(most_common_scopes)


@pytest.mark.asyncio
async def test_a_second_application_takes_no_lock_and_runs_no_ddl(
    prepared_db: ScopedDb, monkeypatch: pytest.MonkeyPatch,
) -> None:
    steps = _spy_on_migration_steps(monkeypatch)

    await apply_ingest_index_migration(prepared_db.seed_backend)

    assert steps == []
    assert len(await _index_definitions(prepared_db)) == 1
    assert await _statistics_kinds(prepared_db) == [['e', 'f', 'm']]


@pytest.mark.asyncio
async def test_a_dropped_index_is_built_again_once(prepared_db: ScopedDb, monkeypatch: pytest.MonkeyPatch) -> None:
    await prepared_db.execute(f'DROP INDEX {INGEST_INDEX_NAME}')
    steps = _spy_on_migration_steps(monkeypatch)

    await apply_ingest_index_migration(prepared_db.seed_backend)
    await apply_ingest_index_migration(prepared_db.seed_backend)

    assert steps == ['begin_migration', ingest_index_ddl('postgresql')]
    assert len(await _index_definitions(prepared_db)) == 1


@pytest.mark.asyncio
async def test_dropped_statistics_are_built_and_sampled_again_once(
    prepared_db: ScopedDb, monkeypatch: pytest.MonkeyPatch,
) -> None:
    await prepared_db.execute(_SEED_MANAGED_SQL)
    await prepared_db.execute(f'DROP STATISTICS {INGEST_STATISTICS_NAME}')
    steps = _spy_on_migration_steps(monkeypatch)

    await apply_ingest_index_migration(prepared_db.seed_backend)
    await apply_ingest_index_migration(prepared_db.seed_backend)

    assert steps == ['begin_migration', ingest_statistics_ddl(), ANALYZE]
    assert await _statistics_kinds(prepared_db) == [['e', 'f', 'm']]
    sampled = await prepared_db.fetch_all(
        "SELECT count(*) FROM pg_stats_ext WHERE tablename = 'context_entries' AND statistics_name = ? "
        'AND most_common_vals IS NOT NULL',
        (INGEST_STATISTICS_NAME,),
    )
    assert sampled == [(1,)]
