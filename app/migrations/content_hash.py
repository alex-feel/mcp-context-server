"""Content hash column migration for mcp-context-server.

Adds a 'content_hash' TEXT column and a composite deduplication index
to the context_entries table. The hash enables efficient deduplication
checks without transferring full text_content over the network.
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


async def apply_content_hash_migration(backend: StorageBackend) -> None:
    """Add content_hash column and deduplication index to context_entries.

    This is an idempotent migration that:
    - SQLite: Checks PRAGMA table_info, adds column and index if missing
    - PostgreSQL: Uses ADD COLUMN IF NOT EXISTS and CREATE INDEX IF NOT EXISTS

    The content_hash column stores a SHA-256 hex digest of text_content,
    enabling hash-based deduplication without fetching full text over the network.
    Nullable for backward compatibility with existing rows.

    Args:
        backend: Storage backend instance.
    """
    if backend.backend_type == 'sqlite':

        def _migrate_sqlite(conn: sqlite3.Connection) -> None:
            cursor = conn.execute('PRAGMA table_info(context_entries)')
            columns = [row[1] for row in cursor.fetchall()]
            if 'content_hash' not in columns:
                conn.execute('ALTER TABLE context_entries ADD COLUMN content_hash TEXT')
                conn.execute(
                    'CREATE INDEX IF NOT EXISTS idx_context_entries_dedup_hash '
                    'ON context_entries(thread_id, source, content_hash)',
                )
                logger.info('Added content_hash column and dedup index to context_entries (SQLite)')
            else:
                # Column exists, ensure index exists too
                conn.execute(
                    'CREATE INDEX IF NOT EXISTS idx_context_entries_dedup_hash '
                    'ON context_entries(thread_id, source, content_hash)',
                )
                logger.debug('content_hash column already exists in context_entries (SQLite)')

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
                'ALTER TABLE context_entries ADD COLUMN IF NOT EXISTS content_hash TEXT',
                migration_timeout_s,
            )
            await execute_migration_ddl(
                conn,
                'CREATE INDEX IF NOT EXISTS idx_context_entries_dedup_hash '
                'ON context_entries(thread_id, source, content_hash)',
                migration_timeout_s,
            )
            logger.info('Added content_hash column and dedup index to context_entries (PostgreSQL)')

        await backend.execute_write(cast(Any, _migrate_postgresql))
