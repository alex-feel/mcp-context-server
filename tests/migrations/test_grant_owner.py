"""Tests for the grant owner migration.

On PostgreSQL the migration gives every access grant the owner of its entry, in
``context_entry_grants.entry_owner_id``, and keeps it exact with a composite foreign key
onto a unique index on ``context_entries (id, owner_id)``. No base schema declares these
objects, because the base script also runs over databases that predate ``owner_id``.
SQLite keeps no copy. The SQLite tests run the migration on a database built from the
base schema; the PostgreSQL tests pin the statements and drive every branch through a
recording backend, and ``tests/integration/postgresql/test_grant_owner_migration_postgresql.py``
runs the migration on a live server.
"""

import logging
import sqlite3
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast

import pytest

import app.migrations.grant_owner as grant_owner_module
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.migrations.access_control import _CREATE_GRANTS_TABLE_POSTGRESQL
from app.migrations.grant_owner import ENTRY_OWNER_INDEX
from app.migrations.grant_owner import GRANT_OWNER_COLUMN
from app.migrations.grant_owner import GRANT_OWNER_FOREIGN_KEY
from app.migrations.grant_owner import apply_grant_owner_migration
from app.schemas import load_schema

_STATE_ARGS = (GRANT_OWNER_COLUMN, ENTRY_OWNER_INDEX, GRANT_OWNER_FOREIGN_KEY)
_COMPLETE = {'column_not_null': True, 'index_present': True, 'foreign_key_present': True}
_ABSENT = {'column_not_null': None, 'index_present': False, 'foreign_key_present': False}
_SINGLE_COLUMN_FOREIGN_KEY = 'context_entry_grants_context_entry_id_fkey'
_BACKFILLED = 3


@pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
def test_no_base_schema_declares_the_grant_owner_objects(backend_type: str) -> None:
    schema = load_schema(backend_type)

    assert GRANT_OWNER_COLUMN not in schema
    assert ENTRY_OWNER_INDEX not in schema
    assert GRANT_OWNER_FOREIGN_KEY not in schema


def test_the_access_control_migration_creates_the_grants_table_without_the_owner_column() -> None:
    assert GRANT_OWNER_COLUMN not in _CREATE_GRANTS_TABLE_POSTGRESQL


def test_the_foreign_key_cascades_ownership_changes_and_deletes() -> None:
    statement = grant_owner_module._ADD_FOREIGN_KEY_SQL

    assert statement == (
        f'ALTER TABLE context_entry_grants ADD CONSTRAINT {GRANT_OWNER_FOREIGN_KEY} '
        f'FOREIGN KEY (context_entry_id, {GRANT_OWNER_COLUMN}) REFERENCES context_entries (id, owner_id) '
        'ON UPDATE CASCADE ON DELETE CASCADE'
    )


def test_the_index_is_unique_on_the_entry_and_its_owner() -> None:
    statement = grant_owner_module._CREATE_INDEX_SQL

    assert statement == (
        f'CREATE UNIQUE INDEX IF NOT EXISTS {ENTRY_OWNER_INDEX} ON context_entries (id, owner_id)'
    )


def test_the_backfill_copies_each_entry_owner_into_its_grants_without_one() -> None:
    backfill = ' '.join(grant_owner_module._BACKFILL_SQL.split())

    assert f'UPDATE context_entry_grants g SET {GRANT_OWNER_COLUMN} = ce.owner_id FROM context_entries ce' in backfill
    assert f'WHERE ce.id = g.context_entry_id AND g.{GRANT_OWNER_COLUMN} IS NULL' in backfill


def test_the_single_column_foreign_key_is_found_by_its_column() -> None:
    lookup = ' '.join(grant_owner_module._SINGLE_COLUMN_FOREIGN_KEYS_SQL.split())

    assert "contype = 'f'" in lookup
    assert "confrelid = to_regclass('context_entries')" in lookup
    assert 'conkey = ARRAY[entry_column.attnum]' in lookup
    assert 'conname =' not in lookup


@pytest.mark.asyncio
async def test_sqlite_keeps_no_copy_of_the_entry_owner(backend: StorageBackend) -> None:
    def _foreign_keys(conn: sqlite3.Connection) -> list[tuple[str, ...]]:
        rows = conn.execute('PRAGMA foreign_key_list(context_entry_grants)').fetchall()
        return [(str(row[2]), str(row[3]), str(row[4]), str(row[6])) for row in rows]

    def _columns(conn: sqlite3.Connection) -> list[str]:
        return [str(row[1]) for row in conn.execute('PRAGMA table_info(context_entry_grants)').fetchall()]

    keys_before = await backend.execute_read(_foreign_keys, access_scope=MAINTENANCE)

    await apply_grant_owner_migration(backend)

    assert GRANT_OWNER_COLUMN not in await backend.execute_read(_columns, access_scope=MAINTENANCE)
    assert await backend.execute_read(_foreign_keys, access_scope=MAINTENANCE) == keys_before
    assert keys_before == [('context_entries', 'context_entry_id', 'id', 'CASCADE')]


class _RecordingPostgreSQLBackend:
    """A PostgreSQL backend whose lock-free probe answers a fixed state and whose writes run on a placeholder connection."""

    backend_type = 'postgresql'

    def __init__(self, state: Mapping[str, object]) -> None:
        self._state = state
        self.probe_args: list[tuple[object, ...]] = []
        self.writes = 0

    async def execute_read(self, operation: Callable[[object], Awaitable[object]], *, access_scope: BackendAccess) -> object:
        assert access_scope is MAINTENANCE
        return await operation(SimpleNamespace(fetch=self._fetch))

    async def execute_write(self, operation: Callable[[object], Awaitable[object]], *, access_scope: BackendAccess) -> object:
        assert access_scope is MAINTENANCE
        self.writes += 1
        return await operation(object())

    async def _fetch(self, sql: str, *args: object) -> list[Mapping[str, object]]:
        assert sql == grant_owner_module._STATE_SQL
        self.probe_args.append(args)
        return [self._state]


def _record_migration_steps(monkeypatch: pytest.MonkeyPatch, *, under_lock: Mapping[str, object]) -> list[tuple[str, ...]]:
    """Replace the lock, DDL and read helpers of the migration with recorders, under a 42-second budget.

    Args:
        monkeypatch: Restores the helpers after the test.
        under_lock: The state the probe repeated under the lock finds.

    Returns:
        The recorded steps, filled in as the migration runs.
    """
    steps: list[tuple[str, ...]] = []

    async def _begin(_conn: object, timeout_s: float) -> None:
        steps.append(('begin_migration', str(timeout_s)))

    async def _ddl(_conn: object, statement: str, timeout_s: float) -> None:
        steps.append(('ddl', statement, str(timeout_s)))

    async def _fetchval(_conn: object, statement: str, timeout_s: float) -> int:
        assert statement == grant_owner_module._BACKFILL_SQL
        steps.append(('backfill', str(timeout_s)))
        return _BACKFILLED

    async def _fetch(_conn: object, statement: str, timeout_s: float, *args: object) -> list[Mapping[str, object]]:
        if statement == grant_owner_module._STATE_SQL:
            assert args == _STATE_ARGS
            steps.append(('probe under the lock', str(timeout_s)))
            return [under_lock]
        assert statement == grant_owner_module._SINGLE_COLUMN_FOREIGN_KEYS_SQL
        steps.append(('find the single-column foreign key', str(timeout_s)))
        return [{'conname': _SINGLE_COLUMN_FOREIGN_KEY}]

    monkeypatch.setattr(grant_owner_module, 'begin_migration', _begin)
    monkeypatch.setattr(grant_owner_module, 'execute_migration_ddl', _ddl)
    monkeypatch.setattr(grant_owner_module, 'fetchval_migration', _fetchval)
    monkeypatch.setattr(grant_owner_module, 'fetch_migration', _fetch)
    monkeypatch.setattr(
        grant_owner_module, 'settings', SimpleNamespace(storage=SimpleNamespace(postgresql_migration_timeout_s=42.0)),
    )
    return steps


async def _migrate(
    monkeypatch: pytest.MonkeyPatch, *, before: Mapping[str, object], under_lock: Mapping[str, object] | None = None,
) -> tuple[_RecordingPostgreSQLBackend, list[tuple[str, ...]]]:
    """Apply the migration to a recording backend in the given state and return the backend and the recorded steps."""
    steps = _record_migration_steps(monkeypatch, under_lock=before if under_lock is None else under_lock)
    backend = _RecordingPostgreSQLBackend(before)

    await apply_grant_owner_migration(cast(StorageBackend, backend))

    assert backend.probe_args == [_STATE_ARGS]
    return backend, steps


_DROP_SINGLE_COLUMN_FOREIGN_KEY = f'ALTER TABLE context_entry_grants DROP CONSTRAINT "{_SINGLE_COLUMN_FOREIGN_KEY}"'


@pytest.mark.asyncio
async def test_postgresql_builds_every_object_under_one_lock_on_a_database_without_them(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=grant_owner_module.__name__):
        backend, steps = await _migrate(monkeypatch, before=_ABSENT)

    assert backend.writes == 1
    assert steps == [
        ('begin_migration', '42.0'),
        ('probe under the lock', '42.0'),
        ('ddl', grant_owner_module._CREATE_INDEX_SQL, '42.0'),
        ('ddl', grant_owner_module._ADD_COLUMN_SQL, '42.0'),
        ('backfill', '42.0'),
        ('ddl', grant_owner_module._SET_NOT_NULL_SQL, '42.0'),
        ('find the single-column foreign key', '42.0'),
        ('ddl', _DROP_SINGLE_COLUMN_FOREIGN_KEY, '42.0'),
        ('ddl', grant_owner_module._ADD_FOREIGN_KEY_SQL, '42.0'),
    ]
    assert f'Grant owner migration applied: {_BACKFILLED} grants backfilled' in caplog.messages


@pytest.mark.asyncio
async def test_postgresql_backfills_a_nullable_column_without_adding_it_again(monkeypatch: pytest.MonkeyPatch) -> None:
    before = {'column_not_null': False, 'index_present': True, 'foreign_key_present': False}

    _backend, steps = await _migrate(monkeypatch, before=before)

    assert [step[:2] for step in steps] == [
        ('begin_migration', '42.0'),
        ('probe under the lock', '42.0'),
        ('backfill', '42.0'),
        ('ddl', grant_owner_module._SET_NOT_NULL_SQL),
        ('find the single-column foreign key', '42.0'),
        ('ddl', _DROP_SINGLE_COLUMN_FOREIGN_KEY),
        ('ddl', grant_owner_module._ADD_FOREIGN_KEY_SQL),
    ]


@pytest.mark.asyncio
async def test_postgresql_replaces_only_a_missing_foreign_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete column needs no backfill, so only the foreign key is rebuilt."""
    before = {'column_not_null': True, 'index_present': True, 'foreign_key_present': False}

    _backend, steps = await _migrate(monkeypatch, before=before)

    assert [step[:2] for step in steps] == [
        ('begin_migration', '42.0'),
        ('probe under the lock', '42.0'),
        ('find the single-column foreign key', '42.0'),
        ('ddl', _DROP_SINGLE_COLUMN_FOREIGN_KEY),
        ('ddl', grant_owner_module._ADD_FOREIGN_KEY_SQL),
    ]


@pytest.mark.asyncio
async def test_postgresql_rebuilds_only_a_missing_index(monkeypatch: pytest.MonkeyPatch) -> None:
    before = {'column_not_null': True, 'index_present': False, 'foreign_key_present': True}

    _backend, steps = await _migrate(monkeypatch, before=before)

    assert [step[:2] for step in steps] == [
        ('begin_migration', '42.0'),
        ('probe under the lock', '42.0'),
        ('ddl', grant_owner_module._CREATE_INDEX_SQL),
    ]


@pytest.mark.asyncio
async def test_postgresql_takes_no_lock_and_writes_nothing_when_every_object_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, steps = await _migrate(monkeypatch, before=_COMPLETE)

    assert backend.writes == 0
    assert steps == []


@pytest.mark.asyncio
async def test_postgresql_changes_nothing_a_concurrent_start_built_while_it_waited(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=grant_owner_module.__name__):
        backend, steps = await _migrate(monkeypatch, before=_ABSENT, under_lock=_COMPLETE)

    assert backend.writes == 1
    assert steps == [('begin_migration', '42.0'), ('probe under the lock', '42.0')]
    assert not [message for message in caplog.messages if message.startswith('Grant owner migration applied')]
