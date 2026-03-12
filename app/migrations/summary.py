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

logger = logging.getLogger(__name__)


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
        async def _migrate_postgresql(conn: asyncpg.Connection) -> None:
            # Use advisory lock for multi-pod DDL safety (matches fts.py pattern)
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'))")
            await conn.execute(
                'ALTER TABLE context_entries ADD COLUMN IF NOT EXISTS summary TEXT',
            )
            logger.info('Added summary column to context_entries (PostgreSQL)')

        await backend.execute_write(cast(Any, _migrate_postgresql))
