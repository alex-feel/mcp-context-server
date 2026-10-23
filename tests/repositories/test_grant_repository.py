"""Tests for the grant repository.

Covers app.repositories.grant_repository against a real temp SQLite database
built from the full base schema: group read grants stored on the caller's
transaction, recorded as granted by the caller and only on an entry the caller
owns, idempotent, in a deterministic order, and removed with their entry. The
rendered grant INSERT is pinned for both backends.
"""

import sqlite3
from unittest.mock import MagicMock

import pytest

from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.repositories.context_repository import ContextRepository
from app.repositories.grant_repository import GrantRepository
from tests.helpers import read_grants

ALICE = AccessScope('alice', frozenset())
BOB = AccessScope('bob', frozenset({'team-a'}))


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
