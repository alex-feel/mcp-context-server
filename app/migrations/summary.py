"""Summary column migration for mcp-context-server.

Adds a 'summary' TEXT column to the context_entries table.
Auto-applied, idempotent migration.
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


async def apply_summary_migration(backend: StorageBackend) -> None:
    """Add summary column to context_entries if not present.

    This is an idempotent migration that:
    - SQLite: Checks PRAGMA table_info, adds column if missing
    - PostgreSQL: Uses ADD COLUMN IF NOT EXISTS (native idempotent DDL)

    The summary column stores LLM-generated summaries for search result display.
    It is always present regardless of ENABLE_SUMMARY_GENERATION setting,
    so search tools can return it even when summary generation is disabled.

    Args:
        backend: Storage backend instance.
    """
    if backend.backend_type == 'sqlite':

        def _migrate_sqlite(conn: sqlite3.Connection) -> None:
            cursor = conn.execute('PRAGMA table_info(context_entries)')
            columns = [row[1] for row in cursor.fetchall()]
            if 'summary' not in columns:
                conn.execute('ALTER TABLE context_entries ADD COLUMN summary TEXT')
                logger.info('Added summary column to context_entries (SQLite)')
            else:
                logger.debug('Summary column already exists in context_entries (SQLite)')

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
                'ALTER TABLE context_entries ADD COLUMN IF NOT EXISTS summary TEXT',
                migration_timeout_s,
            )
            logger.info('Added summary column to context_entries (PostgreSQL)')

        await backend.execute_write(cast(Any, _migrate_postgresql))
