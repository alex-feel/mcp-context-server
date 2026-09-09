"""
Grant repository for per-entry access grants.

This module handles all database operations on ``context_entry_grants`` -- the
rows that give a specific principal ('user') or group ('group') read or write
access to one context entry under 'shared' visibility.
"""

import contextlib
import sqlite3
from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import cast

from app.backends.base import StorageBackend
from app.repositories.base import BaseRepository

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext
else:
    with contextlib.suppress(ImportError):
        import asyncpg


class GrantRow(NamedTuple):
    """One access grant on a context entry."""

    principal_type: str
    principal_id: str
    permission: str
    granted_by: str


class GrantRepository(BaseRepository):
    """Repository for context-entry access grants.

    Handles insertion and retrieval of grant rows. Grant insertion is
    idempotent: the unique index on (context_entry_id, principal_type,
    principal_id, permission) plus ``ON CONFLICT ... DO NOTHING`` makes a
    repeated grant a no-op rather than a duplicate or an error.
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize grant repository.

        Args:
            backend: Storage backend for executing database operations
        """
        super().__init__(backend)

    def _insert_grant_sql(self) -> str:
        """Return the INSERT that stores one grant row, tolerating a repeat.

        ``ON CONFLICT ... DO NOTHING`` names the unique index explicitly (the
        tag-repository precedent): a repeated grant is the one conflict that is
        correct to ignore, while a foreign-key violation still raises because it
        means the parent entry is missing.

        Returns:
            The backend-appropriate INSERT statement.
        """
        return (
            f'INSERT INTO context_entry_grants '
            f'(context_entry_id, principal_type, principal_id, permission, granted_by) '
            f'VALUES ({self._placeholders(5)}) '
            f'ON CONFLICT (context_entry_id, principal_type, principal_id, permission) DO NOTHING'
        )

    async def store_group_read_grants(
        self,
        context_id: str,
        groups: Iterable[str],
        granted_by: str,
        txn: 'TransactionContext | None' = None,
    ) -> None:
        """Store read grants for the given groups on one context entry.

        Used by the ``ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups`` write
        path: every group of the inserting principal receives a read grant in
        the same transaction as the entry itself. Insertion order is sorted so
        the write sequence is deterministic regardless of the caller's set
        iteration order.

        Args:
            context_id: ID of the context entry.
            groups: Group principal ids to grant read access to.
            granted_by: Principal id recorded as the grantor.
            txn: Optional transaction context for atomic multi-repository
                operations. When provided, uses the transaction's connection
                directly. When None, uses execute_write() for standalone
                operation.
        """
        sorted_groups = sorted(groups)
        if not sorted_groups:
            return

        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _store_sqlite(conn: sqlite3.Connection) -> None:
                cursor = conn.cursor()
                query = self._insert_grant_sql()
                for group in sorted_groups:
                    cursor.execute(query, (context_id, 'group', group, 'read', granted_by))

            if txn:
                await self._run_sqlite_txn(_store_sqlite, cast(sqlite3.Connection, txn.connection))
            else:
                await self.backend.execute_write(_store_sqlite)
        else:  # postgresql

            async def _store_postgresql(conn: 'asyncpg.Connection') -> None:
                query = self._insert_grant_sql()
                for group in sorted_groups:
                    await conn.execute(query, context_id, 'group', group, 'read', granted_by)

            if txn:
                await _store_postgresql(cast('asyncpg.Connection', txn.connection))
            else:
                await self.backend.execute_write(cast(Any, _store_postgresql))

    async def get_grants_for_context(self, context_id: str) -> list[GrantRow]:
        """Get all grants for a specific context entry.

        Args:
            context_id: ID of the context entry.

        Returns:
            Grant rows ordered by (principal_type, principal_id, permission)
            so the result is deterministic on both backends.
        """
        query = (
            f'SELECT principal_type, principal_id, permission, granted_by '
            f'FROM context_entry_grants WHERE context_entry_id = {self._placeholder(1)} '
            f'ORDER BY principal_type, principal_id, permission'
        )

        if self.backend.backend_type == 'sqlite':

            def _get_sqlite(conn: sqlite3.Connection) -> list[GrantRow]:
                cursor = conn.cursor()
                cursor.execute(query, (context_id,))
                return [
                    GrantRow(
                        principal_type=row['principal_type'],
                        principal_id=row['principal_id'],
                        permission=row['permission'],
                        granted_by=row['granted_by'],
                    )
                    for row in cursor.fetchall()
                ]

            return await self.backend.execute_read(_get_sqlite)

        # postgresql
        async def _get_postgresql(conn: 'asyncpg.Connection') -> list[GrantRow]:
            rows = await conn.fetch(query, context_id)
            return [
                GrantRow(
                    principal_type=row['principal_type'],
                    principal_id=row['principal_id'],
                    permission=row['permission'],
                    granted_by=row['granted_by'],
                )
                for row in rows
            ]

        return await self.backend.execute_read(_get_postgresql)
