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

logger = logging.getLogger(__name__)


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
        async def _migrate_postgresql(conn: asyncpg.Connection) -> None:
            # Use advisory lock for multi-pod DDL safety (matches fts.py pattern)
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'))")
            await conn.execute(
                'ALTER TABLE context_entries ADD COLUMN IF NOT EXISTS content_hash TEXT',
            )
            await conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_context_entries_dedup_hash '
                'ON context_entries(thread_id, source, content_hash)',
            )
            logger.info('Added content_hash column and dedup index to context_entries (PostgreSQL)')

        await backend.execute_write(cast(Any, _migrate_postgresql))
