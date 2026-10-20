"""Tests for the ingest index migration.

The partial index serving the ingest reads is created by the migration alone, never by a
base schema: its SQLite key names ``content_hash``, ``owner_id`` and ``visibility``, which
the column migrations add on an in-place upgrade, and the base script also runs over
databases that predate them. On PostgreSQL the migration also keeps extended statistics
on the thread and the block's scope, rendered from the scope expression the reads
compare. The SQLite tests run the server's own startup preparation on a fresh database
and on one that predates the access-control columns; the PostgreSQL tests pin the
statistics statement and drive the migration's branch through a recording backend, and
the integration tests under ``tests/integration/postgresql`` run it on a live server.
"""

import sqlite3
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import app.migrations.ingest as ingest_module
import app.migrations.metadata as metadata_module
from app.backends import StorageBackend
from app.backends import create_backend
from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.ingest_sql import INGEST_INDEX_NAME
from app.ingest_sql import ingest_index_ddl
from app.ingest_sql import scope_sql
from app.migrations import apply_ingest_index_migration
from app.migrations import handle_metadata_indexes
from app.migrations.ingest import INGEST_STATISTICS_NAME
from app.migrations.ingest import ingest_statistics_ddl
from app.schemas import load_schema
from app.settings import get_settings
from app.startup import init_database
from tests.repositories._access_scope_layouts import sqlite_scoped_db


async def _ingest_index_sql(backend: StorageBackend) -> list[str]:
    """Return the stored definition of every index named like the ingest index."""

    def _read(conn: sqlite3.Connection) -> list[str]:
        cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?", (INGEST_INDEX_NAME,))
        return [str(row[0]) for row in cursor.fetchall()]

    return await backend.execute_read(_read, access_scope=MAINTENANCE)


@pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
def test_no_base_schema_declares_the_ingest_index(backend_type: str) -> None:
    assert INGEST_INDEX_NAME not in load_schema(backend_type)


@pytest.mark.asyncio
async def test_init_database_alone_creates_no_ingest_index(tmp_path: Path) -> None:
    backend = create_backend(backend_type='sqlite', db_path=str(tmp_path / 'base_schema_only.db'))
    await backend.initialize()
    try:
        await init_database(backend=backend)

        assert await _ingest_index_sql(backend) == []
    finally:
        await backend.shutdown()


@pytest.mark.asyncio
async def test_prepare_database_creates_the_ingest_index(
    async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with sqlite_scoped_db(async_db_initialized, 'fp32', monkeypatch) as db:
        definitions = await _ingest_index_sql(db.backend)

    assert len(definitions) == 1
    assert "WHERE json_extract(metadata, '$.ingest.source_id') IS NOT NULL" in definitions[0]


@pytest.mark.asyncio
async def test_a_second_application_changes_nothing(
    async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with sqlite_scoped_db(async_db_initialized, 'fp32', monkeypatch) as db:
        before = await _ingest_index_sql(db.backend)

        await apply_ingest_index_migration(db.backend)

        assert await _ingest_index_sql(db.backend) == before


@pytest.mark.asyncio
async def test_a_database_predating_the_access_columns_boots_with_the_ingest_index(
    backend_pre_migration: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The startup preparation upgrades the old table first, so the index naming its new columns builds."""
    async with sqlite_scoped_db(backend_pre_migration, 'fp32', monkeypatch) as db:
        assert len(await _ingest_index_sql(db.backend)) == 1


@pytest.mark.parametrize('sync_mode', ['strict', 'auto'])
@pytest.mark.asyncio
async def test_the_metadata_index_reconciler_leaves_the_ingest_index_alone(
    sync_mode: str, async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reconciler manages ``idx_metadata_*`` names only: strict mode accepts the database and auto mode drops nothing."""
    async with sqlite_scoped_db(async_db_initialized, 'fp32', monkeypatch) as db:
        monkeypatch.setenv('METADATA_INDEX_SYNC_MODE', sync_mode)
        get_settings.cache_clear()
        monkeypatch.setattr(metadata_module, 'settings', get_settings())

        await handle_metadata_indexes(db.backend)

        assert len(await _ingest_index_sql(db.backend)) == 1


_STATISTICS_DDL = (
    "CREATE STATISTICS IF NOT EXISTS stat_context_ingest_scope (mcv, dependencies) "
    "ON thread_id, ((metadata #>> '{ingest,scope}')) FROM context_entries"
)
_ANALYZE = 'ANALYZE context_entries'


def test_statistics_ddl() -> None:
    """The statistics carry the most common pairs and the dependency of the thread and the scope, created idempotently."""
    assert INGEST_STATISTICS_NAME == 'stat_context_ingest_scope'
    assert ingest_statistics_ddl() == _STATISTICS_DDL


def test_statistics_describe_the_scope_expression_the_reads_compare() -> None:
    """The statistics expression is the scope fragment itself, so the planner matches it to the reads' comparison."""
    assert f'ON thread_id, ({scope_sql("postgresql")}) FROM context_entries' in ingest_statistics_ddl()


class _RecordingPostgreSQLBackend:
    """A PostgreSQL backend whose probes answer fixed values and whose writes run on a placeholder connection."""

    backend_type = 'postgresql'

    def __init__(self, *, index_exists: bool, statistics_exist: bool) -> None:
        self._index_exists = index_exists
        self._statistics_exist = statistics_exist
        self.probe_args: list[tuple[object, ...]] = []
        self.writes = 0

    async def execute_read(self, operation: Callable[[object], Awaitable[object]], *, access_scope: BackendAccess) -> object:
        assert access_scope is MAINTENANCE
        return await operation(SimpleNamespace(fetchval=self._fetchval))

    async def execute_write(self, operation: Callable[[object], Awaitable[None]], *, access_scope: BackendAccess) -> None:
        assert access_scope is MAINTENANCE
        self.writes += 1
        await operation(object())

    async def _fetchval(self, sql: str, *args: object) -> bool:
        self.probe_args.append(args)
        if 'pg_statistic_ext' in sql:
            return self._statistics_exist
        return self._index_exists


def _record_migration_steps(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Replace the lock and DDL helpers of the migration with recorders, under a 42-second budget."""
    steps: list[tuple[str, ...]] = []

    async def _begin(_conn: object, timeout_s: float) -> None:
        steps.append(('begin_migration', str(timeout_s)))

    async def _ddl(_conn: object, statement: str, timeout_s: float) -> None:
        steps.append(('execute_migration_ddl', statement, str(timeout_s)))

    monkeypatch.setattr(ingest_module, 'begin_migration', _begin)
    monkeypatch.setattr(ingest_module, 'execute_migration_ddl', _ddl)
    monkeypatch.setattr(
        ingest_module, 'settings', SimpleNamespace(storage=SimpleNamespace(postgresql_migration_timeout_s=42.0)),
    )
    return steps


async def _migrate(
    monkeypatch: pytest.MonkeyPatch, *, index_exists: bool, statistics_exist: bool,
) -> tuple[_RecordingPostgreSQLBackend, list[tuple[str, ...]]]:
    """Apply the migration to a recording backend in the given state and return the backend and the recorded steps."""
    steps = _record_migration_steps(monkeypatch)
    backend = _RecordingPostgreSQLBackend(index_exists=index_exists, statistics_exist=statistics_exist)

    await apply_ingest_index_migration(cast(StorageBackend, backend))

    assert backend.probe_args == [(INGEST_INDEX_NAME,), (INGEST_STATISTICS_NAME,)]
    return backend, steps


@pytest.mark.asyncio
async def test_postgresql_creates_the_index_and_the_statistics_under_one_lock_when_both_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, steps = await _migrate(monkeypatch, index_exists=False, statistics_exist=False)

    assert backend.writes == 1
    assert steps == [
        ('begin_migration', '42.0'),
        ('execute_migration_ddl', ingest_index_ddl('postgresql'), '42.0'),
        ('execute_migration_ddl', _STATISTICS_DDL, '42.0'),
        ('execute_migration_ddl', _ANALYZE, '42.0'),
    ]


@pytest.mark.asyncio
async def test_postgresql_creates_and_populates_the_statistics_when_only_they_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, steps = await _migrate(monkeypatch, index_exists=True, statistics_exist=False)

    assert backend.writes == 1
    assert steps == [
        ('begin_migration', '42.0'),
        ('execute_migration_ddl', _STATISTICS_DDL, '42.0'),
        ('execute_migration_ddl', _ANALYZE, '42.0'),
    ]


@pytest.mark.asyncio
async def test_postgresql_creates_only_the_index_when_the_statistics_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    """The statistics already describe the table, so rebuilding the index needs no new sample."""
    backend, steps = await _migrate(monkeypatch, index_exists=False, statistics_exist=True)

    assert backend.writes == 1
    assert steps == [
        ('begin_migration', '42.0'),
        ('execute_migration_ddl', ingest_index_ddl('postgresql'), '42.0'),
    ]


@pytest.mark.asyncio
async def test_postgresql_takes_no_lock_and_writes_nothing_when_both_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, steps = await _migrate(monkeypatch, index_exists=True, statistics_exist=True)

    assert backend.writes == 0
    assert steps == []
