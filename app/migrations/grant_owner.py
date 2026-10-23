"""Grant owner migration for mcp-context-server.

On PostgreSQL every access grant also records the owner of the entry it grants access to,
in ``context_entry_grants.entry_owner_id``, so a row-level security policy on the grants
table can tell from the grant row alone that the caller owns the granted entry. A policy
that read ``context_entries`` instead would read the grants table again through the read
policy of ``context_entries``, which PostgreSQL rejects as infinite recursion.

A composite foreign key ``(context_entry_id, entry_owner_id)`` onto a unique index on
``context_entries (id, owner_id)`` keeps the column exact: ``ON UPDATE CASCADE`` carries an
ownership change, such as the one ``mcp-context-server-migrate --reassign-owner`` makes,
into every grant of the entry, and ``ON DELETE CASCADE`` removes the grants with their
entry, as the single-column foreign key it replaces does. PostgreSQL runs no referential
work for an update that leaves both columns unchanged, so an ordinary update of an entry
pays nothing for it.

No base schema declares the column, the index or the foreign key: the base script runs at
every start, also over databases whose ``context_entries`` predates ``owner_id``, which the
access-control migration adds. This migration runs after it and is the only place these
objects come from, so a fresh database and an upgraded one reach the same shape the same
way. SQLite keeps no copy: no policy reads its grants, and the application's access
predicate joins the entry itself.

PostgreSQL first reads, without a lock, whether the column exists and is NOT NULL, whether
the index exists and whether the foreign key exists, by name, on the tables the bare names
resolve to through ``search_path``, the tables the DDL changes. A start that finds all of
them takes no lock and runs no DDL. Otherwise one transaction under the migration budget
and the shared advisory lock reads them again, so a start that waited for a concurrent one
rebuilds nothing that one built, and then builds what is missing: the index, the column,
the owner of every grant without one, NOT NULL, and the composite foreign key in place of
the single-column one, which it finds by its column rather than by a name.
"""

import logging
from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.postgresql_backend.session import quote_pg_identifier
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations._pg_ddl import fetch_migration
from app.migrations._pg_ddl import fetchval_migration
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

GRANT_OWNER_COLUMN = 'entry_owner_id'
ENTRY_OWNER_INDEX = 'idx_context_entries_id_owner'
GRANT_OWNER_FOREIGN_KEY = 'context_entry_grants_entry_owner_fkey'

# The state of the three objects on the tables the bare names resolve to: whether the
# column is NOT NULL (NULL when the column does not exist), and whether the index and the
# foreign key exist.
_STATE_SQL = '''
    SELECT
        (
            SELECT attribute.attnotnull
            FROM pg_attribute AS attribute
            WHERE attribute.attrelid = to_regclass('context_entry_grants')
              AND attribute.attname = $1
              AND NOT attribute.attisdropped
        ) AS column_not_null,
        EXISTS (
            SELECT 1
            FROM pg_index
            JOIN pg_class AS index_relation ON index_relation.oid = pg_index.indexrelid
            WHERE pg_index.indrelid = to_regclass('context_entries')
              AND index_relation.relname = $2
        ) AS index_present,
        EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE pg_constraint.conrelid = to_regclass('context_entry_grants')
              AND pg_constraint.contype = 'f'
              AND pg_constraint.conname = $3
        ) AS foreign_key_present
'''
_STATE_ARGS = (GRANT_OWNER_COLUMN, ENTRY_OWNER_INDEX, GRANT_OWNER_FOREIGN_KEY)

_CREATE_INDEX_SQL = f'CREATE UNIQUE INDEX IF NOT EXISTS {ENTRY_OWNER_INDEX} ON context_entries (id, owner_id)'
_ADD_COLUMN_SQL = f'ALTER TABLE context_entry_grants ADD COLUMN IF NOT EXISTS {GRANT_OWNER_COLUMN} text'
# Copies each entry's owner into its grants that carry none, and counts them.
_BACKFILL_SQL = f'''
    WITH backfilled AS (
        UPDATE context_entry_grants g SET {GRANT_OWNER_COLUMN} = ce.owner_id
        FROM context_entries ce
        WHERE ce.id = g.context_entry_id AND g.{GRANT_OWNER_COLUMN} IS NULL
        RETURNING 1
    )
    SELECT count(*) FROM backfilled
'''
_SET_NOT_NULL_SQL = f'ALTER TABLE context_entry_grants ALTER COLUMN {GRANT_OWNER_COLUMN} SET NOT NULL'
# The foreign keys from the grants table's entry column alone onto context_entries, whatever
# name they were given.
_SINGLE_COLUMN_FOREIGN_KEYS_SQL = '''
    SELECT pg_constraint.conname
    FROM pg_constraint
    JOIN pg_attribute AS entry_column
      ON entry_column.attrelid = pg_constraint.conrelid AND entry_column.attname = 'context_entry_id'
    WHERE pg_constraint.contype = 'f'
      AND pg_constraint.conrelid = to_regclass('context_entry_grants')
      AND pg_constraint.confrelid = to_regclass('context_entries')
      AND pg_constraint.conkey = ARRAY[entry_column.attnum]
'''
_ADD_FOREIGN_KEY_SQL = (
    f'ALTER TABLE context_entry_grants ADD CONSTRAINT {GRANT_OWNER_FOREIGN_KEY} '
    f'FOREIGN KEY (context_entry_id, {GRANT_OWNER_COLUMN}) REFERENCES context_entries (id, owner_id) '
    'ON UPDATE CASCADE ON DELETE CASCADE'
)


class _StateRow(Protocol):
    """A row read by column name, such as the one :data:`_STATE_SQL` returns."""

    def __getitem__(self, key: str, /) -> object: ...


@dataclass(frozen=True, slots=True)
class _GrantOwnerState:
    """What exists of the grant owner objects.

    Attributes:
        column_not_null: Whether the owner column is NOT NULL; None when it does not exist.
        index_present: Whether the unique index on the entry and its owner exists.
        foreign_key_present: Whether the composite foreign key exists.
    """

    column_not_null: bool | None
    index_present: bool
    foreign_key_present: bool

    @classmethod
    def from_row(cls, row: _StateRow) -> '_GrantOwnerState':
        """Read the state from the one row :data:`_STATE_SQL` returns."""
        column_not_null = row['column_not_null']
        return cls(
            column_not_null=None if column_not_null is None else bool(column_not_null),
            index_present=bool(row['index_present']),
            foreign_key_present=bool(row['foreign_key_present']),
        )

    @property
    def complete(self) -> bool:
        """Whether every object exists and the column is NOT NULL."""
        return self.column_not_null is True and self.index_present and self.foreign_key_present


async def _probe(backend: StorageBackend) -> _GrantOwnerState:
    """Read the state of the grant owner objects without a lock.

    Args:
        backend: The PostgreSQL maintenance backend.

    Returns:
        The state.
    """

    async def _read(conn: asyncpg.Connection) -> _GrantOwnerState:
        rows = await conn.fetch(_STATE_SQL, *_STATE_ARGS)
        return _GrantOwnerState.from_row(rows[0])

    return await backend.execute_read(cast(Any, _read), access_scope=MAINTENANCE)


async def _build_missing(conn: asyncpg.Connection, timeout_s: float) -> int | None:
    """Build every grant owner object a probe under the shared lock finds missing.

    Args:
        conn: The connection of the migration transaction.
        timeout_s: The migration budget in seconds.

    Returns:
        The number of grants given their entry's owner, or None when a concurrent start
        built everything while this one waited for the lock.
    """
    await begin_migration(conn, timeout_s)
    rows = await fetch_migration(conn, _STATE_SQL, timeout_s, *_STATE_ARGS)
    state = _GrantOwnerState.from_row(rows[0])
    if state.complete:
        return None

    backfilled = 0
    if not state.index_present:
        await execute_migration_ddl(conn, _CREATE_INDEX_SQL, timeout_s)
    if state.column_not_null is None:
        await execute_migration_ddl(conn, _ADD_COLUMN_SQL, timeout_s)
    if not state.column_not_null:
        backfilled = await fetchval_migration(conn, _BACKFILL_SQL, timeout_s) or 0
        await execute_migration_ddl(conn, _SET_NOT_NULL_SQL, timeout_s)
    if not state.foreign_key_present:
        for row in await fetch_migration(conn, _SINGLE_COLUMN_FOREIGN_KEYS_SQL, timeout_s):
            drop = f'ALTER TABLE context_entry_grants DROP CONSTRAINT {quote_pg_identifier(str(row["conname"]))}'
            await execute_migration_ddl(conn, drop, timeout_s)
        await execute_migration_ddl(conn, _ADD_FOREIGN_KEY_SQL, timeout_s)
    return backfilled


async def apply_grant_owner_migration(backend: StorageBackend) -> None:
    """Record the owner of each grant's entry on PostgreSQL and keep it exact with a foreign key.

    Idempotent: a start that finds every object takes no lock and runs no DDL. SQLite keeps
    no copy of the owner, so the migration does nothing there.

    Args:
        backend: The backend to migrate: on PostgreSQL the maintenance backend, logged in
            as the table owner.
    """
    if backend.backend_type == 'sqlite':
        return

    if (await _probe(backend)).complete:
        logger.debug(
            'Grant owner column, index %s and foreign key %s present, skipping', ENTRY_OWNER_INDEX, GRANT_OWNER_FOREIGN_KEY,
        )
        return

    migration_timeout_s = settings.storage.postgresql_migration_timeout_s

    async def _migrate(conn: asyncpg.Connection) -> int | None:
        return await _build_missing(conn, migration_timeout_s)

    backfilled = await backend.execute_write(cast(Any, _migrate), access_scope=MAINTENANCE)
    if backfilled is None:
        logger.debug('Grant owner objects built by a concurrent start, nothing to do')
        return
    logger.info('Grant owner migration applied: %d grants backfilled', backfilled)
