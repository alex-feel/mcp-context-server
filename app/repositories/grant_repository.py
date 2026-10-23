"""
Grant repository for per-entry access grants.

This module handles all database operations on ``context_entry_grants`` -- the
rows that give a specific principal ('user') or group ('group') read or write
access to one context entry. A grant is effective under either visibility:
any grant opens a 'private' entry to its grantee for reading, and a write grant
also authorizes content edits on a 'private' or 'public' entry.

Only an entry's owner grants, revokes and lists access to it: every grant goes
through one INSERT that selects the entry under the owner predicate of the
caller's scope and records the caller as the grantor, and the DELETE and the
listing carry the same predicate, so a caller who does not own the entry stores,
removes and sees nothing. On PostgreSQL the INSERT also records the entry's owner
on the grant, which the grants table's row-level security policies read. No
statement returns the rows it changes: under row-level security a returned row
would also have to pass the grants read policy.
"""

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import starmap
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.access_scope import build_access_predicate
from app.backends.base import StorageBackend
from app.repositories.base import BaseRepository
from app.repositories.base import Statement
from app.repositories.base import canonical_timestamp

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext

type GrantPrincipalType = Literal['user', 'group']
type GrantPermission = Literal['read', 'write']

# A repeated grant is the one conflict that is correct to ignore.
_IGNORE_REPEATED_GRANT = 'ON CONFLICT (context_entry_id, principal_type, principal_id, permission) DO NOTHING'


@dataclass(frozen=True, slots=True)
class GrantRecord:
    """One grant on an entry, as its owner lists it.

    Attributes:
        principal_type: ``'user'`` or ``'group'``.
        principal_id: The grantee principal or group id.
        permission: ``'read'`` or ``'write'``.
        granted_by: The principal recorded as the grantor.
        created_at: When the grant was stored, as a canonical UTC timestamp.
    """

    principal_type: GrantPrincipalType
    principal_id: str
    permission: GrantPermission
    granted_by: str
    created_at: str


class GrantRepository(BaseRepository):
    """Repository for context-entry access grants.

    Stores, removes and lists grant rows. Grant insertion is idempotent: the unique
    index on (context_entry_id, principal_type, principal_id, permission) plus
    ``ON CONFLICT ... DO NOTHING`` makes a repeated grant a no-op rather than a
    duplicate or an error, and the same four columns name the one row a removal
    deletes.
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize grant repository.

        Args:
            backend: Storage backend for executing database operations
        """
        super().__init__(backend)

    def _grant_insert_sql(
        self,
        backend_type: str,
        context_id: str,
        principal_type: str,
        principal_id: str,
        permission: str,
        *,
        scope: AccessScope,
    ) -> Statement:
        """Build the INSERT that stores one grant on an entry the caller owns, tolerating a repeat.

        The statement selects the entry under the owner predicate of ``scope``, so it
        inserts nothing for an entry the caller does not own, and binds the grantor from
        ``scope`` and from nowhere else. On PostgreSQL it also stores the entry's owner,
        which the grants policies compare with the caller and a foreign key keeps exact.
        ``ON CONFLICT ... DO NOTHING`` names the unique index explicitly (the
        tag-repository precedent); the SELECT's WHERE clause keeps SQLite from reading the
        ``ON`` of the conflict clause as a join constraint. The statement returns no rows.

        Args:
            backend_type: ``'sqlite'`` or ``'postgresql'``.
            context_id: ID of the context entry.
            principal_type: ``'user'`` or ``'group'``.
            principal_id: The grantee principal or group id.
            permission: ``'read'`` or ``'write'``.
            scope: The caller's scope: the entry's owner and the grantor.

        Returns:
            The statement and its parameters.
        """
        ce = self._readable('context_entries', alias='ce')
        owner = build_access_predicate(scope, mode=AccessMode.OWNER, backend_type=backend_type, outer='ce', start=6)
        if backend_type == 'sqlite':
            sql = (
                'INSERT INTO context_entry_grants (context_entry_id, principal_type, principal_id, permission, granted_by) '
                f'SELECT ce.id, ?, ?, ?, ? FROM {ce} WHERE ce.id = ?{owner.and_clause()} {_IGNORE_REPEATED_GRANT}'
            )
            return sql, [principal_type, principal_id, permission, scope.principal_id, context_id, *owner.params]
        sql = (
            'INSERT INTO context_entry_grants '
            '(context_entry_id, principal_type, principal_id, permission, granted_by, entry_owner_id) '
            f'SELECT ce.id, $2, $3, $4, $5, ce.owner_id FROM {ce} WHERE ce.id = $1{owner.and_clause()} '
            f'{_IGNORE_REPEATED_GRANT}'
        )
        return sql, [context_id, principal_type, principal_id, permission, scope.principal_id, *owner.params]

    async def store_group_read_grants(
        self,
        context_id: str,
        groups: Iterable[str],
        *,
        scope: AccessScope,
        txn: 'TransactionContext',
    ) -> None:
        """Store read grants for the given groups on one context entry the caller owns.

        Used by the ``ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups`` write
        path: every group of the inserting principal receives a read grant in
        the same transaction as the entry itself, recorded as granted by that
        principal. Insertion order is sorted so the write sequence is deterministic
        regardless of the caller's set iteration order.

        Args:
            context_id: ID of the context entry.
            groups: Group principal ids to grant read access to.
            scope: The caller's scope: the entry's owner and the grantor.
            txn: The caller's transaction, which runs as the identity it bound.
        """
        statements = [
            self._grant_insert_sql(txn.backend_type, context_id, 'group', group, 'read', scope=scope)
            for group in sorted(groups)
        ]
        if not statements:
            return

        if txn.backend_type == 'sqlite':

            def _store_sqlite(conn: sqlite3.Connection) -> None:
                for sql, params in statements:
                    conn.execute(sql, params)

            await self._run_sqlite_txn(_store_sqlite, cast(sqlite3.Connection, txn.connection))
            return

        conn = cast('asyncpg.Connection', txn.connection)
        for sql, params in statements:
            await conn.execute(sql, *params)

    async def store_grant(
        self,
        context_id: str,
        principal_type: GrantPrincipalType,
        principal_id: str,
        permission: GrantPermission,
        *,
        scope: AccessScope,
        txn: 'TransactionContext',
    ) -> bool:
        """Store one grant on an entry the caller owns, recorded as granted by the caller.

        A grant the entry already carries stays as it is, and an entry the caller does
        not own, or that does not exist, receives nothing.

        Args:
            context_id: ID of the context entry.
            principal_type: ``'user'`` or ``'group'``.
            principal_id: The grantee principal or group id.
            permission: ``'read'`` or ``'write'``.
            scope: The caller's scope: the entry's owner and the grantor.
            txn: The caller's transaction, which runs as the identity it bound.

        Returns:
            True when the grant was created; False when the entry already carried it or
            the caller does not own the entry.
        """
        statement = self._grant_insert_sql(
            txn.backend_type, context_id, principal_type, principal_id, permission, scope=scope,
        )
        return await self._changes_one_row(statement, txn)

    async def delete_grant(
        self,
        context_id: str,
        principal_type: GrantPrincipalType,
        principal_id: str,
        permission: GrantPermission,
        *,
        scope: AccessScope,
        txn: 'TransactionContext',
    ) -> bool:
        """Remove exactly the named grant from an entry the caller owns.

        Every other grant of the entry stays, the other permission of the same
        principal included; a grant on an entry the caller does not own stays as well.

        Args:
            context_id: ID of the context entry.
            principal_type: ``'user'`` or ``'group'``.
            principal_id: The grantee principal or group id.
            permission: ``'read'`` or ``'write'``.
            scope: The caller's scope: the entry's owner.
            txn: The caller's transaction, which runs as the identity it bound.

        Returns:
            True when the grant was removed; False when the entry carried no such grant
            or the caller does not own the entry.
        """
        ce = self._readable('context_entries', alias='ce')
        owner = build_access_predicate(scope, mode=AccessMode.OWNER, backend_type=txn.backend_type, outer='ce', start=5)
        sql = (
            f'DELETE FROM context_entry_grants WHERE context_entry_id = {self._placeholder(1)} '
            f'AND principal_type = {self._placeholder(2)} AND principal_id = {self._placeholder(3)} '
            f'AND permission = {self._placeholder(4)} AND EXISTS (SELECT 1 FROM {ce} '
            f'WHERE ce.id = context_entry_grants.context_entry_id{owner.and_clause()})'
        )
        statement: Statement = (sql, [context_id, principal_type, principal_id, permission, *owner.params])
        return await self._changes_one_row(statement, txn)

    async def list_grants(
        self,
        context_id: str,
        *,
        scope: AccessScope,
        txn: 'TransactionContext | None' = None,
    ) -> list[GrantRecord]:
        """List every grant of an entry the caller owns.

        The grants come ordered by principal type, principal id and permission, an order
        the unique index on the entry and those three columns serves without a sort. A
        caller who does not own the entry, whatever it may read or edit, gets no grant.

        Args:
            context_id: ID of the context entry.
            scope: The caller's scope: the entry's owner.
            txn: Optional transaction context. When provided the listing runs on the
                transaction's own connection, so it sees the grants the transaction
                changed.

        Returns:
            The entry's grants; empty when it has none or the caller does not own it.
        """
        backend_type = txn.backend_type if txn is not None else self.backend.backend_type
        ce = self._readable('context_entries', alias='ce')
        owner = build_access_predicate(scope, mode=AccessMode.OWNER, backend_type=backend_type, outer='ce', start=2)
        sql = (
            'SELECT g.principal_type, g.principal_id, g.permission, g.granted_by, g.created_at '
            f'FROM context_entry_grants g JOIN {ce} ON ce.id = g.context_entry_id '
            f'WHERE g.context_entry_id = {self._placeholder(1)}{owner.and_clause()} '
            'ORDER BY g.principal_type, g.principal_id, g.permission'
        )
        params = [context_id, *owner.params]

        if backend_type == 'sqlite':

            def _list_sqlite(conn: sqlite3.Connection) -> list[GrantRecord]:
                return list(starmap(_grant_record, conn.execute(sql, params).fetchall()))

            if txn is not None:
                return await self._run_sqlite_txn(_list_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_read(_list_sqlite, access_scope=scope)

        async def _list_postgresql(conn: 'asyncpg.Connection') -> list[GrantRecord]:
            return list(starmap(_grant_record, await conn.fetch(sql, *params)))

        if txn is not None:
            return await _list_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_read(_list_postgresql, access_scope=scope)

    @classmethod
    async def _changes_one_row(cls, statement: Statement, txn: 'TransactionContext') -> bool:
        """Run one INSERT or DELETE on the transaction's connection and report whether it changed a row.

        Args:
            statement: The statement and its parameters; it changes at most one row.
            txn: The caller's transaction.

        Returns:
            True when the statement changed one row.
        """
        sql, params = statement
        if txn.backend_type == 'sqlite':

            def _execute_sqlite(conn: sqlite3.Connection) -> int:
                return conn.execute(sql, params).rowcount

            return await cls._run_sqlite_txn(_execute_sqlite, cast(sqlite3.Connection, txn.connection)) == 1

        # asyncpg reports the changed row count last in the command status, such as 'INSERT 0 1' or 'DELETE 0'.
        status = await cast('asyncpg.Connection', txn.connection).execute(sql, *params)
        return int(status.rsplit(' ', 1)[1]) == 1


def _grant_record(
    principal_type: object, principal_id: object, permission: object, granted_by: object, created_at: object,
) -> GrantRecord:
    """Build the record of one listed grant row.

    The table's checks admit only the two principal types and the two permissions, and
    its column default stamps every grant with its creation time.

    Returns:
        The record, its creation time in the canonical UTC form.
    """
    return GrantRecord(
        principal_type=cast(GrantPrincipalType, str(principal_type)),
        principal_id=str(principal_id),
        permission=cast(GrantPermission, str(permission)),
        granted_by=str(granted_by),
        created_at=cast(str, canonical_timestamp(created_at)),
    )
