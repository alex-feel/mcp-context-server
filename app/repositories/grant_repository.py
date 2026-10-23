"""
Grant repository for per-entry access grants.

This module handles all database operations on ``context_entry_grants`` -- the
rows that give a specific principal ('user') or group ('group') read or write
access to one context entry. A grant is effective under either visibility:
any grant opens a 'private' entry to its grantee for reading, and a write grant
also authorizes content edits on a 'private' or 'public' entry.

Only an entry's owner grants access to it: every grant goes through one INSERT
that selects the entry under the owner predicate of the caller's scope and records
the caller as the grantor, so a caller who does not own the entry inserts nothing.
On PostgreSQL the INSERT also records the entry's owner on the grant, which the
grants table's row-level security policies read.
"""

import sqlite3
from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing import cast

from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.access_scope import build_access_predicate
from app.backends.base import StorageBackend
from app.repositories.base import BaseRepository
from app.repositories.base import Statement

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext

# A repeated grant is the one conflict that is correct to ignore.
_IGNORE_REPEATED_GRANT = 'ON CONFLICT (context_entry_id, principal_type, principal_id, permission) DO NOTHING'


class GrantRepository(BaseRepository):
    """Repository for context-entry access grants.

    Handles insertion of grant rows. Grant insertion is idempotent: the unique index
    on (context_entry_id, principal_type, principal_id, permission) plus
    ``ON CONFLICT ... DO NOTHING`` makes a repeated grant a no-op rather than a
    duplicate or an error.
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
