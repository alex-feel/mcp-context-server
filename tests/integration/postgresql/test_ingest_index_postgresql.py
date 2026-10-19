"""The ingest index migration on a live PostgreSQL server.

The server's startup preparation creates the partial index keyed on the thread, the
``C``-collated source id and the id. A later application finds the index through its
probe and takes neither the migration lock nor any DDL; once the index is dropped, the
next application builds it again under the lock.
"""

from collections.abc import AsyncGenerator

import asyncpg
import pytest
import pytest_asyncio

import app.migrations.ingest as ingest_module
from app.ingest_sql import INGEST_INDEX_NAME
from app.ingest_sql import ingest_index_ddl
from app.migrations import apply_ingest_index_migration
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import postgresql_scoped_db

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]


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


@pytest.mark.asyncio
async def test_startup_creates_the_partial_index_on_the_collated_source_id(prepared_db: ScopedDb) -> None:
    definitions = await _index_definitions(prepared_db)

    assert len(definitions) == 1
    definition = definitions[0]
    assert "USING btree (thread_id, ((metadata #>> '{ingest,source_id}'::text[])) COLLATE \"C\", id)" in definition
    assert definition.endswith("WHERE ((metadata #>> '{ingest,source_id}'::text[]) IS NOT NULL)")


@pytest.mark.asyncio
async def test_a_second_application_takes_no_lock_and_runs_no_ddl(
    prepared_db: ScopedDb, monkeypatch: pytest.MonkeyPatch,
) -> None:
    steps = _spy_on_migration_steps(monkeypatch)

    await apply_ingest_index_migration(prepared_db.seed_backend)

    assert steps == []
    assert len(await _index_definitions(prepared_db)) == 1


@pytest.mark.asyncio
async def test_a_dropped_index_is_built_again_once(prepared_db: ScopedDb, monkeypatch: pytest.MonkeyPatch) -> None:
    await prepared_db.execute(f'DROP INDEX {INGEST_INDEX_NAME}')
    steps = _spy_on_migration_steps(monkeypatch)

    await apply_ingest_index_migration(prepared_db.seed_backend)
    await apply_ingest_index_migration(prepared_db.seed_backend)

    assert steps == ['begin_migration', ingest_index_ddl('postgresql')]
    assert len(await _index_definitions(prepared_db)) == 1
