"""Tests for the grant repository.

Covers app.repositories.grant_repository against a real temp SQLite database
built from the full base schema: group read grants and single grants stored on
the caller's transaction, recorded as granted by the caller and only on an entry
the caller owns, idempotent; a named grant removed only by the entry's owner;
the grants of an entry listed only to its owner, in principal type, principal id
and permission order; and every grant removed with its entry. The rendered
statements are pinned for both backends, none of them returning rows.
"""

import dataclasses
import re
import sqlite3
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.base import TransactionContext
from app.repositories.context_repository import ContextRepository
from app.repositories.grant_repository import GrantPermission
from app.repositories.grant_repository import GrantPrincipalType
from app.repositories.grant_repository import GrantRecord
from app.repositories.grant_repository import GrantRepository
from tests.helpers import insert_grant
from tests.helpers import read_grants

ALICE = AccessScope('alice', frozenset())
BOB = AccessScope('bob', frozenset({'team-a'}))

# The canonical UTC form every timestamp a tool returns takes.
CANONICAL_TIMESTAMP = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z')

# An id no entry carries.
ABSENT_ID = 'f' * 32


async def _store_entry(backend: StorageBackend, thread_id: str = 'grant-thread') -> str:
    repo = ContextRepository(backend)
    context_id, _ = await repo.store_with_deduplication(
        thread_id=thread_id,
        source='agent',
        content_type='text',
        text_content=f'grant test entry for {thread_id}',
        scope=ALICE,
        visibility='private',
    )
    return context_id


async def _store_group_read_grants(
    backend: StorageBackend, context_id: str, groups: set[str] | list[str], *, scope: AccessScope,
) -> None:
    """Store the group read grants on a transaction of ``scope``, as the store tool does."""
    async with backend.begin_transaction(access_scope=scope) as txn:
        await GrantRepository(backend).store_group_read_grants(context_id, groups, scope=scope, txn=txn)


async def _store_grant(
    backend: StorageBackend,
    context_id: str,
    principal_type: GrantPrincipalType,
    principal_id: str,
    permission: GrantPermission,
    *,
    scope: AccessScope,
) -> bool:
    """Store one grant on its own transaction of ``scope`` and report whether it was created."""
    async with backend.begin_transaction(access_scope=scope) as txn:
        return await GrantRepository(backend).store_grant(
            context_id, principal_type, principal_id, permission, scope=scope, txn=txn,
        )


async def _delete_grant(
    backend: StorageBackend,
    context_id: str,
    principal_type: GrantPrincipalType,
    principal_id: str,
    permission: GrantPermission,
    *,
    scope: AccessScope,
) -> bool:
    """Remove one grant on its own transaction of ``scope`` and report whether a row was removed."""
    async with backend.begin_transaction(access_scope=scope) as txn:
        return await GrantRepository(backend).delete_grant(
            context_id, principal_type, principal_id, permission, scope=scope, txn=txn,
        )


def _keys(records: Sequence[GrantRecord]) -> list[tuple[str, str, str, str]]:
    """Return each record's principal type, principal id, permission and grantor."""
    return [(record.principal_type, record.principal_id, record.permission, record.granted_by) for record in records]


class TestStoreGroupReadGrants:
    """Tests for store_group_read_grants."""

    @pytest.mark.asyncio
    async def test_grants_stored_sorted_with_the_caller_as_grantor(self, async_db_initialized: StorageBackend) -> None:
        """Each group receives one read grant recording the calling owner as the grantor."""
        context_id = await _store_entry(async_db_initialized)

        await _store_group_read_grants(async_db_initialized, context_id, {'team-b', 'team-a'}, scope=ALICE)

        assert await read_grants(async_db_initialized, context_id) == [
            ('group', 'team-a', 'read', 'alice'),
            ('group', 'team-b', 'read', 'alice'),
        ]

    @pytest.mark.asyncio
    async def test_a_caller_who_does_not_own_the_entry_stores_nothing(self, async_db_initialized: StorageBackend) -> None:
        """The owner predicate inside the INSERT leaves another principal's entry untouched, for its own groups too."""
        context_id = await _store_entry(async_db_initialized)

        await _store_group_read_grants(async_db_initialized, context_id, {'team-a'}, scope=BOB)

        assert await read_grants(async_db_initialized, context_id) == []

    @pytest.mark.asyncio
    async def test_repeated_grants_are_idempotent(self, async_db_initialized: StorageBackend) -> None:
        """Re-granting the same groups inserts no duplicate rows."""
        context_id = await _store_entry(async_db_initialized)

        await _store_group_read_grants(async_db_initialized, context_id, ['team-a'], scope=ALICE)
        await _store_group_read_grants(async_db_initialized, context_id, ['team-a'], scope=ALICE)

        assert await read_grants(async_db_initialized, context_id) == [
            ('group', 'team-a', 'read', 'alice'),
        ]

    @pytest.mark.asyncio
    async def test_empty_groups_is_a_noop(self, async_db_initialized: StorageBackend) -> None:
        """An empty group collection writes nothing."""
        context_id = await _store_entry(async_db_initialized)

        await _store_group_read_grants(async_db_initialized, context_id, [], scope=ALICE)

        assert await read_grants(async_db_initialized, context_id) == []

    @pytest.mark.asyncio
    async def test_grants_cascade_with_entry_delete(self, async_db_initialized: StorageBackend) -> None:
        """Deleting the parent entry removes its grant rows (FK ON DELETE CASCADE)."""
        context_id = await _store_entry(async_db_initialized)
        await _store_group_read_grants(async_db_initialized, context_id, ['team-a'], scope=ALICE)

        await ContextRepository(async_db_initialized).delete_by_ids([context_id], scope=ALICE)

        def _count(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entry_grants WHERE context_entry_id = ?',
                (context_id,),
            )
            return int(cursor.fetchone()[0])

        assert await async_db_initialized.execute_read(_count, access_scope=MAINTENANCE) == 0


class TestStoreGrant:
    """Tests for store_grant."""

    @pytest.mark.asyncio
    async def test_the_owner_stores_a_grant_once(self, async_db_initialized: StorageBackend) -> None:
        """The first store creates the row with the owner as grantor; a repeat creates nothing."""
        context_id = await _store_entry(async_db_initialized)

        first = await _store_grant(async_db_initialized, context_id, 'user', 'dave', 'read', scope=ALICE)
        second = await _store_grant(async_db_initialized, context_id, 'user', 'dave', 'read', scope=ALICE)

        assert (first, second) == (True, False)
        assert await read_grants(async_db_initialized, context_id) == [('user', 'dave', 'read', 'alice')]

    @pytest.mark.asyncio
    async def test_a_read_and_a_write_grant_are_separate_rows(self, async_db_initialized: StorageBackend) -> None:
        """A write grant beside a read grant of the same principal is a row of its own."""
        context_id = await _store_entry(async_db_initialized)

        read = await _store_grant(async_db_initialized, context_id, 'user', 'dave', 'read', scope=ALICE)
        write = await _store_grant(async_db_initialized, context_id, 'user', 'dave', 'write', scope=ALICE)

        assert (read, write) == (True, True)
        assert await read_grants(async_db_initialized, context_id) == [
            ('user', 'dave', 'read', 'alice'),
            ('user', 'dave', 'write', 'alice'),
        ]

    @pytest.mark.asyncio
    async def test_a_write_grantee_stores_nothing(self, async_db_initialized: StorageBackend) -> None:
        """A caller who may edit the entry but does not own it extends nobody's access."""
        context_id = await _store_entry(async_db_initialized)
        await insert_grant(async_db_initialized, context_id, 'user', 'bob', 'write', 'alice')

        created = await _store_grant(async_db_initialized, context_id, 'user', 'carol', 'read', scope=BOB)

        assert created is False
        assert await read_grants(async_db_initialized, context_id) == [('user', 'bob', 'write', 'alice')]

    @pytest.mark.asyncio
    async def test_an_absent_entry_stores_nothing(self, async_db_initialized: StorageBackend) -> None:
        """A grant on an id no entry carries creates no row."""
        created = await _store_grant(async_db_initialized, ABSENT_ID, 'user', 'dave', 'read', scope=ALICE)

        assert created is False
        assert await read_grants(async_db_initialized, ABSENT_ID) == []


class TestDeleteGrant:
    """Tests for delete_grant."""

    @pytest.mark.asyncio
    async def test_the_owner_removes_exactly_the_named_grant(self, async_db_initialized: StorageBackend) -> None:
        """The first removal deletes the one named row; a repeat removes nothing; other rows stay."""
        context_id = await _store_entry(async_db_initialized)
        await _store_grant(async_db_initialized, context_id, 'user', 'bob', 'read', scope=ALICE)
        await _store_grant(async_db_initialized, context_id, 'user', 'bob', 'write', scope=ALICE)

        first = await _delete_grant(async_db_initialized, context_id, 'user', 'bob', 'read', scope=ALICE)
        second = await _delete_grant(async_db_initialized, context_id, 'user', 'bob', 'read', scope=ALICE)

        assert (first, second) == (True, False)
        assert await read_grants(async_db_initialized, context_id) == [('user', 'bob', 'write', 'alice')]

    @pytest.mark.asyncio
    async def test_a_group_grant_is_not_a_user_grant_of_the_same_id(self, async_db_initialized: StorageBackend) -> None:
        """Removing a user grant leaves the group grant whose id is the same text."""
        context_id = await _store_entry(async_db_initialized)
        await _store_grant(async_db_initialized, context_id, 'group', 'team-a', 'read', scope=ALICE)
        await _store_grant(async_db_initialized, context_id, 'user', 'team-a', 'read', scope=ALICE)

        removed = await _delete_grant(async_db_initialized, context_id, 'user', 'team-a', 'read', scope=ALICE)

        assert removed is True
        assert await read_grants(async_db_initialized, context_id) == [('group', 'team-a', 'read', 'alice')]

    @pytest.mark.asyncio
    async def test_the_grantee_cannot_remove_its_own_grant(self, async_db_initialized: StorageBackend) -> None:
        """Only the entry's owner removes a grant, the grantee of that very grant included."""
        context_id = await _store_entry(async_db_initialized)
        await _store_grant(async_db_initialized, context_id, 'user', 'bob', 'write', scope=ALICE)

        removed = await _delete_grant(async_db_initialized, context_id, 'user', 'bob', 'write', scope=BOB)

        assert removed is False
        assert await read_grants(async_db_initialized, context_id) == [('user', 'bob', 'write', 'alice')]


class TestListGrants:
    """Tests for list_grants."""

    @pytest.mark.asyncio
    async def test_the_owner_lists_every_grant_in_order(self, async_db_initialized: StorageBackend) -> None:
        """Every grant of the entry, ordered by principal type, principal id and permission, with its fields."""
        context_id = await _store_entry(async_db_initialized)
        for principal_type, principal_id, permission in (
            ('user', 'dave', 'write'), ('group', 'team-b', 'read'), ('user', 'bob', 'write'),
            ('group', 'team-a', 'read'), ('user', 'bob', 'read'),
        ):
            await insert_grant(async_db_initialized, context_id, principal_type, principal_id, permission, 'alice')

        records = await GrantRepository(async_db_initialized).list_grants(context_id, scope=ALICE)

        assert _keys(records) == [
            ('group', 'team-a', 'read', 'alice'),
            ('group', 'team-b', 'read', 'alice'),
            ('user', 'bob', 'read', 'alice'),
            ('user', 'bob', 'write', 'alice'),
            ('user', 'dave', 'write', 'alice'),
        ]
        assert all(CANONICAL_TIMESTAMP.fullmatch(record.created_at) for record in records)

    @pytest.mark.asyncio
    async def test_a_grantee_lists_nothing(self, async_db_initialized: StorageBackend) -> None:
        """A caller who may read and edit the entry but does not own it sees none of its grants."""
        context_id = await _store_entry(async_db_initialized)
        await insert_grant(async_db_initialized, context_id, 'user', 'bob', 'read', 'alice')
        await insert_grant(async_db_initialized, context_id, 'user', 'bob', 'write', 'alice')

        assert await GrantRepository(async_db_initialized).list_grants(context_id, scope=BOB) == []

    @pytest.mark.asyncio
    async def test_the_listing_inside_a_transaction_matches_the_one_outside(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """On the caller's transaction the listing reads the same rows."""
        context_id = await _store_entry(async_db_initialized)
        await insert_grant(async_db_initialized, context_id, 'user', 'bob', 'read', 'alice')
        repository = GrantRepository(async_db_initialized)

        outside = await repository.list_grants(context_id, scope=ALICE)
        async with async_db_initialized.begin_transaction(access_scope=ALICE) as txn:
            inside = await repository.list_grants(context_id, scope=ALICE, txn=txn)

        assert inside == outside
        assert _keys(inside) == [('user', 'bob', 'read', 'alice')]

    @pytest.mark.asyncio
    async def test_the_grants_of_a_deleted_entry_are_gone(self, async_db_initialized: StorageBackend) -> None:
        """Deleting the entry removes every grant stored on it."""
        context_id = await _store_entry(async_db_initialized)
        await _store_grant(async_db_initialized, context_id, 'user', 'dave', 'read', scope=ALICE)

        await ContextRepository(async_db_initialized).delete_by_ids([context_id], scope=ALICE)

        assert await GrantRepository(async_db_initialized).list_grants(context_id, scope=ALICE) == []
        assert await read_grants(async_db_initialized, context_id) == []


class TestGrantRecord:
    """The record list_grants returns."""

    def test_a_record_carries_the_listed_fields_and_is_immutable(self) -> None:
        record = GrantRecord('user', 'bob', 'read', 'alice', '2026-01-02T03:04:05Z')
        field_name = 'permission'

        assert [field.name for field in dataclasses.fields(record)] == [
            'principal_type', 'principal_id', 'permission', 'granted_by', 'created_at',
        ]
        assert not hasattr(record, '__dict__')
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, field_name, 'write')


def _backend(backend_type: str) -> MagicMock:
    """Return a backend that reports ``backend_type`` and names each relation by its base name and alias."""
    backend = MagicMock(spec=StorageBackend)
    backend.backend_type = backend_type
    backend.readable_relation.side_effect = lambda table, *, alias: f'{table} {alias}'
    return backend


def _repository(backend_type: str) -> GrantRepository:
    """Return a grant repository over :func:`_backend`."""
    return GrantRepository(_backend(backend_type))


class TestGrantInsertStatement:
    """The one INSERT every grant goes through: the caller as grantor, the owner predicate in the statement."""

    def test_sqlite_selects_the_owned_entry_and_binds_the_select_list_first(self) -> None:
        sql, params = _repository('sqlite')._grant_insert_sql('sqlite', 'entry-id', 'user', 'dave', 'read', scope=ALICE)

        assert sql == (
            'INSERT INTO context_entry_grants (context_entry_id, principal_type, principal_id, permission, granted_by) '
            'SELECT ce.id, ?, ?, ?, ? FROM context_entries ce WHERE ce.id = ? AND ce.owner_id = ? '
            'ON CONFLICT (context_entry_id, principal_type, principal_id, permission) DO NOTHING'
        )
        assert params == ['user', 'dave', 'read', 'alice', 'entry-id', 'alice']

    def test_postgresql_also_records_the_entry_owner(self) -> None:
        sql, params = _repository('postgresql')._grant_insert_sql(
            'postgresql', 'entry-id', 'group', 'team-x', 'write', scope=ALICE,
        )

        assert sql == (
            'INSERT INTO context_entry_grants '
            '(context_entry_id, principal_type, principal_id, permission, granted_by, entry_owner_id) '
            'SELECT ce.id, $2, $3, $4, $5, ce.owner_id FROM context_entries ce WHERE ce.id = $1 AND ce.owner_id = $6 '
            'ON CONFLICT (context_entry_id, principal_type, principal_id, permission) DO NOTHING'
        )
        assert params == ['entry-id', 'group', 'team-x', 'write', 'alice', 'alice']

    def test_the_entry_is_read_through_the_read_seam(self) -> None:
        backend = _backend('postgresql')

        GrantRepository(backend)._grant_insert_sql('postgresql', 'entry-id', 'user', 'dave', 'read', scope=ALICE)

        backend.readable_relation.assert_called_once_with('context_entries', alias='ce')


type Recorded = list[tuple[str, list[object]]]


class _SqliteCursor:
    """The cursor of a statement that changed one row and selected none."""

    rowcount = 1

    @staticmethod
    def fetchall() -> list[tuple[object, ...]]:
        return []


class _SqliteRecorder:
    """A SQLite connection that records every statement it runs."""

    def __init__(self, statements: Recorded) -> None:
        self.statements = statements

    def execute(self, sql: str, params: Sequence[object] = ()) -> _SqliteCursor:
        self.statements.append((sql, list(params)))
        return _SqliteCursor()


class _PostgresqlRecorder:
    """An asyncpg connection that records every statement it runs."""

    def __init__(self, statements: Recorded) -> None:
        self.statements = statements

    async def execute(self, sql: str, *params: object) -> str:
        self.statements.append((sql, list(params)))
        return 'DELETE 1' if sql.startswith('DELETE') else 'INSERT 0 1'

    async def fetch(self, sql: str, *params: object) -> list[tuple[object, ...]]:
        self.statements.append((sql, list(params)))
        return []


class _StatusConnection:
    """An asyncpg connection that answers every statement with one command status."""

    def __init__(self, status: str) -> None:
        self.status = status

    async def execute(self, sql: str, *params: object) -> str:
        assert sql.startswith(('INSERT', 'DELETE')), sql
        assert params, sql
        return self.status


@dataclasses.dataclass(frozen=True)
class _Transaction:
    """A transaction whose connection is a stand-in."""

    connection: object
    backend_type: str


async def _recorded_statements(backend_type: str) -> Recorded:
    """Run every grant method once on a recording connection and return the statements in order.

    The listing runs twice: on the caller's transaction and on a connection of its own.

    Returns:
        Each statement and its parameters.
    """
    statements: Recorded = []
    connection = _SqliteRecorder(statements) if backend_type == 'sqlite' else _PostgresqlRecorder(statements)

    def _read_sqlite(read: Callable[[object], object], *, access_scope: object) -> object:
        assert access_scope is ALICE
        return read(connection)

    async def _read_postgresql(read: Callable[[object], Awaitable[object]], *, access_scope: object) -> object:
        assert access_scope is ALICE
        return await read(connection)

    backend = _backend(backend_type)
    backend.execute_read = AsyncMock(side_effect=_read_sqlite if backend_type == 'sqlite' else _read_postgresql)
    repository = GrantRepository(backend)
    txn: TransactionContext = _Transaction(connection, backend_type)

    await repository.store_grant('entry-id', 'user', 'dave', 'read', scope=ALICE, txn=txn)
    await repository.store_group_read_grants('entry-id', {'team-x'}, scope=ALICE, txn=txn)
    await repository.delete_grant('entry-id', 'user', 'dave', 'read', scope=ALICE, txn=txn)
    await repository.list_grants('entry-id', scope=ALICE, txn=txn)
    await repository.list_grants('entry-id', scope=ALICE)
    backend.execute_read.assert_awaited_once()
    return statements


_SQLITE_DELETE = (
    'DELETE FROM context_entry_grants WHERE context_entry_id = ? AND principal_type = ? AND principal_id = ? '
    'AND permission = ? AND EXISTS (SELECT 1 FROM context_entries ce '
    'WHERE ce.id = context_entry_grants.context_entry_id AND ce.owner_id = ?)'
)
_SQLITE_LIST = (
    'SELECT g.principal_type, g.principal_id, g.permission, g.granted_by, g.created_at '
    'FROM context_entry_grants g JOIN context_entries ce ON ce.id = g.context_entry_id '
    'WHERE g.context_entry_id = ? AND ce.owner_id = ? ORDER BY g.principal_type, g.principal_id, g.permission'
)
_POSTGRESQL_DELETE = (
    'DELETE FROM context_entry_grants WHERE context_entry_id = $1 AND principal_type = $2 AND principal_id = $3 '
    'AND permission = $4 AND EXISTS (SELECT 1 FROM context_entries ce '
    'WHERE ce.id = context_entry_grants.context_entry_id AND ce.owner_id = $5)'
)
_POSTGRESQL_LIST = (
    'SELECT g.principal_type, g.principal_id, g.permission, g.granted_by, g.created_at '
    'FROM context_entry_grants g JOIN context_entries ce ON ce.id = g.context_entry_id '
    'WHERE g.context_entry_id = $1 AND ce.owner_id = $2 ORDER BY g.principal_type, g.principal_id, g.permission'
)


class TestGrantStatements:
    """Every statement of the grant methods, as each backend receives it."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
    async def test_no_grant_statement_returns_rows(self, backend_type: str) -> None:
        """No statement asks for the rows it changed, which a database policy would also have to admit."""
        statements = await _recorded_statements(backend_type)

        assert len(statements) == 5
        assert [sql for sql, _params in statements if 'RETURNING' in sql.upper()] == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('insert_status', 'delete_status', 'changed'),
        [('INSERT 0 1', 'DELETE 1', True), ('INSERT 0 0', 'DELETE 0', False)],
    )
    async def test_postgresql_reads_the_change_from_the_command_status(
        self, insert_status: str, delete_status: str, changed: bool,
    ) -> None:
        """A store or removal on PostgreSQL reports a change exactly when the command status counts one row."""
        repository = _repository('postgresql')

        created = await repository.store_grant(
            'entry-id', 'user', 'dave', 'read', scope=ALICE,
            txn=_Transaction(_StatusConnection(insert_status), 'postgresql'),
        )
        removed = await repository.delete_grant(
            'entry-id', 'user', 'dave', 'read', scope=ALICE,
            txn=_Transaction(_StatusConnection(delete_status), 'postgresql'),
        )

        assert (created, removed) == (changed, changed)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('backend_type', 'delete_sql', 'list_sql'),
        [('sqlite', _SQLITE_DELETE, _SQLITE_LIST), ('postgresql', _POSTGRESQL_DELETE, _POSTGRESQL_LIST)],
    )
    async def test_the_owner_predicate_sits_in_the_delete_and_the_listing(
        self, backend_type: str, delete_sql: str, list_sql: str,
    ) -> None:
        """The delete names the exact grant and an owned entry; the listing joins the owned entry and orders its grants."""
        insert_sql, insert_params = _repository(backend_type)._grant_insert_sql(
            backend_type, 'entry-id', 'user', 'dave', 'read', scope=ALICE,
        )
        group_sql, group_params = _repository(backend_type)._grant_insert_sql(
            backend_type, 'entry-id', 'group', 'team-x', 'read', scope=ALICE,
        )

        assert await _recorded_statements(backend_type) == [
            (insert_sql, insert_params),
            (group_sql, group_params),
            (delete_sql, ['entry-id', 'user', 'dave', 'read', 'alice']),
            (list_sql, ['entry-id', 'alice']),
            (list_sql, ['entry-id', 'alice']),
        ]
