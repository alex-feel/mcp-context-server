"""Version column migration for mcp-context-server.

Adds the optimistic-concurrency ``version`` column to context_entries so a
database created BEFORE the column was added to the base schema gains it on an
in-place upgrade. Mirrors the content_hash / summary column migrations:
auto-applied and idempotent. Fresh databases (and the v2->v3 migrate CLI target,
which builds tables from the base schema) already have the column, so this
migration is a no-op there.
"""

import logging
import sqlite3
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def apply_version_migration(backend: StorageBackend) -> None:
    """Add the optimistic-concurrency ``version`` column to context_entries.

    Idempotent:
    - SQLite: checks ``PRAGMA table_info`` and runs ``ADD COLUMN`` when absent.
    - PostgreSQL: ``ADD COLUMN IF NOT EXISTS`` under the schema-init advisory lock.

    The column backs the compare-and-set guard used by ``update_context`` /
    ``update_context_batch`` and is bumped by the dedup-store UPDATE, so it must
    exist before those paths run on an upgraded database. ``NOT NULL DEFAULT 0``
    backfills existing rows.

    Args:
        backend: Storage backend instance.
    """
    if backend.backend_type == 'sqlite':

        def _migrate_sqlite(conn: sqlite3.Connection) -> None:
            cursor = conn.execute('PRAGMA table_info(context_entries)')
            columns = [row[1] for row in cursor.fetchall()]
            if 'version' not in columns:
                conn.execute('ALTER TABLE context_entries ADD COLUMN version INTEGER NOT NULL DEFAULT 0')
                logger.info('Added version column to context_entries (SQLite)')
            else:
                logger.debug('version column already exists in context_entries (SQLite)')

        await backend.execute_write(_migrate_sqlite)

    else:  # postgresql
        migration_timeout_s = settings.storage.postgresql_migration_timeout_s

        async def _migrate_postgresql(conn: asyncpg.Connection) -> None:
            # Raise the transaction-scoped statement_timeout to the migration budget and take
            # the shared advisory lock under it, so a wait on a multi-pod peer holding the lock
            # during a slow build is bounded by the migration budget, not the pool's shorter
            # command_timeout (which would cancel the wait with a non-retryable timeout and crash
            # startup). SET LOCAL auto-reverts on COMMIT/ROLLBACK; no finally-restore that would
            # raise 25P02 in an aborted transaction.
            await begin_migration(conn, migration_timeout_s)
            await execute_migration_ddl(
                conn,
                'ALTER TABLE context_entries ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 0',
                migration_timeout_s,
            )
            logger.info('Added version column to context_entries (PostgreSQL)')

        await backend.execute_write(cast(Any, _migrate_postgresql))
