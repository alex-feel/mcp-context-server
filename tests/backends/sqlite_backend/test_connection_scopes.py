"""Tests for app/backends/sqlite_backend/connection_scopes.py: the read entry point and readable relations."""

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessScope
from app.backends.base import MAINTENANCE
from app.backends.base import ROUTABLE_TABLES
from app.backends.base import BackendAccess
from app.backends.sqlite_backend import SQLiteBackend

ALICE = AccessScope('alice', frozenset({'team-x'}), may_publish=True)


@pytest_asyncio.fixture
async def backend(tmp_path: Path) -> AsyncIterator[SQLiteBackend]:
    """Yield an initialized backend over a database holding the base schema."""
    from app.schemas import load_schema

    db_path = tmp_path / 'connection_scopes.db'
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(load_schema('sqlite'))
    sqlite_backend = SQLiteBackend(db_path=str(db_path))
    await sqlite_backend.initialize()
    try:
        yield sqlite_backend
    finally:
        await sqlite_backend.shutdown()


class TestExecuteReadAccessScope:
    """SQLite enforces access in the application layer, so the read path accepts and drops the identity."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('access', [ALICE, SYSTEM_SCOPE, MAINTENANCE], ids=['request', 'system', 'maintenance'])
    async def test_access_scope_never_reaches_the_operation(self, backend: SQLiteBackend, access: BackendAccess) -> None:
        def _operation(_conn: sqlite3.Connection, value: str, **kwargs: object) -> tuple[str, dict[str, object]]:
            return value, dict(kwargs)

        result = await backend.execute_read(_operation, 'read', access_scope=access, marker='kept')

        assert result == ('read', {'marker': 'kept'})

    @pytest.mark.asyncio
    async def test_read_runs_its_statement(self, backend: SQLiteBackend) -> None:
        def _count(conn: sqlite3.Connection) -> int:
            return int(conn.execute('SELECT COUNT(*) FROM context_entries').fetchone()[0])

        assert await backend.execute_read(_count, access_scope=ALICE) == 0


class TestReadableRelation:
    """SQLite reads every routable table by its own name."""

    @pytest.mark.parametrize('table', sorted(ROUTABLE_TABLES))
    def test_alias_equal_to_the_table_renders_the_bare_name(self, backend: SQLiteBackend, table: str) -> None:
        assert backend.readable_relation(table, alias=table) == table

    @pytest.mark.parametrize('table', sorted(ROUTABLE_TABLES))
    def test_distinct_alias_follows_the_table(self, backend: SQLiteBackend, table: str) -> None:
        assert backend.readable_relation(table, alias='x_1') == f'{table} x_1'

    @pytest.mark.parametrize(
        'table', ['context_entry_grants', 'context_entries_fts', 'compression_metadata', 'CONTEXT_ENTRIES'],
    )
    def test_table_outside_the_routable_set_is_refused(self, backend: SQLiteBackend, table: str) -> None:
        with pytest.raises(ValueError, match='routable'):
            backend.readable_relation(table, alias='ce')

    @pytest.mark.parametrize('alias', ['', 'CE', '1ce', 'ce x', 'ce;', 'ce\n'])
    def test_malformed_alias_is_refused(self, backend: SQLiteBackend, alias: str) -> None:
        with pytest.raises(ValueError, match='alias'):
            backend.readable_relation('context_entries', alias=alias)
