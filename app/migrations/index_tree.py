"""Index-tree node table migration for mcp-context-server.

Provisions ``context_index_nodes`` -- the table that stores per-node LLM
summaries for the navigate_context index_tree. The table exists ONLY when
per-node summaries are enabled (``ENABLE_INDEX_TREE_NODE_SUMMARIES=true``); the
code-derived outline itself never needs it. DDL is idempotent (CREATE ... IF NOT
EXISTS) and parity-matched across backends: TEXT FK on SQLite, UUID FK on
PostgreSQL, both ON DELETE CASCADE so node rows vanish with their entry.
"""

import logging
import sqlite3
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.errors import format_exception_message
from app.migrations._pg_ddl import execute_migration_ddl
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# SQLite: TEXT FK to context_entries(id); INTEGER surrogate PK.
_CREATE_TABLE_SQLITE = '''
CREATE TABLE IF NOT EXISTS context_index_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id TEXT NOT NULL REFERENCES context_entries(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    level INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    title TEXT NOT NULL,
    node_summary TEXT,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    UNIQUE(context_id, node_id)
)
'''
_CREATE_INDEX_SQLITE = (
    'CREATE INDEX IF NOT EXISTS idx_index_nodes_context ON context_index_nodes(context_id)'
)

# PostgreSQL: native UUID FK; BIGSERIAL surrogate PK. BARE table/index names so a
# non-default POSTGRESQL_SCHEMA resolves via search_path (peer migration convention).
_CREATE_TABLE_POSTGRESQL = '''
CREATE TABLE IF NOT EXISTS context_index_nodes (
    id BIGSERIAL PRIMARY KEY,
    context_id UUID NOT NULL REFERENCES context_entries(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    level INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    title TEXT NOT NULL,
    node_summary TEXT,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    UNIQUE(context_id, node_id)
)
'''
_CREATE_INDEX_POSTGRESQL = (
    'CREATE INDEX IF NOT EXISTS idx_index_nodes_context ON context_index_nodes(context_id)'
)


async def apply_index_tree_migration(backend: StorageBackend, *, force: bool = False) -> None:
    """Provision the ``context_index_nodes`` table when node summaries are enabled.

    Gated on ``settings.index_tree.node_summaries_enabled`` (parity with how the
    FTS migration gates on ``settings.fts.enabled``): when node summaries are off
    the table is never created and navigation stays purely code-derived. The DDL
    is idempotent on both backends.

    Args:
        backend: Storage backend instance.
        force: Apply regardless of the toggle (migration CLI parity hook).

    Raises:
        RuntimeError: If migration execution fails.
    """
    if not force and not settings.index_tree.node_summaries_enabled:
        return

    try:
        if backend.backend_type == 'postgresql':
            await _apply_postgresql(backend)
        else:
            await _apply_sqlite(backend)
    except Exception as e:
        logger.error(f'Failed to apply index_tree migration: {e}')
        raise RuntimeError(f'Index tree migration failed: {format_exception_message(e)}') from e


async def _apply_sqlite(backend: StorageBackend) -> None:
    """Create the SQLite context_index_nodes table and index (idempotent)."""

    def _apply(conn: sqlite3.Connection) -> None:
        conn.execute(_CREATE_TABLE_SQLITE)
        conn.execute(_CREATE_INDEX_SQLITE)

    await backend.execute_write(_apply)
    logger.info('Applied index_tree migration for SQLite: context_index_nodes ready')


async def _apply_postgresql(backend: StorageBackend) -> None:
    """Create the PostgreSQL context_index_nodes table and index (idempotent)."""
    migration_timeout_s = settings.storage.postgresql_migration_timeout_s
    migration_timeout_ms = int(migration_timeout_s * 1000)
    default_timeout_ms = int(settings.storage.postgresql_command_timeout_s * 1000 * 0.9)

    async def _apply(conn: asyncpg.Connection) -> None:
        # Transaction-scoped advisory lock for multi-pod DDL safety (releases on
        # COMMIT/ROLLBACK), mirroring the chunking/semantic/fts migrations. The lock
        # acquire and the DDL run under the migration timeout so the pool's shorter
        # command_timeout cannot cancel a slow build (or a lock wait on a peer pod's
        # migration) before the server-side statement_timeout applies.
        await execute_migration_ddl(
            conn,
            "SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'))",
            migration_timeout_s,
        )
        await conn.execute(f'SET statement_timeout = {migration_timeout_ms}')
        try:
            await execute_migration_ddl(conn, _CREATE_TABLE_POSTGRESQL, migration_timeout_s)
            await execute_migration_ddl(conn, _CREATE_INDEX_POSTGRESQL, migration_timeout_s)
        finally:
            await conn.execute(f'SET statement_timeout = {default_timeout_ms}')

    await backend.execute_write(cast(Any, _apply))
    logger.info('Applied index_tree migration for PostgreSQL: context_index_nodes ready')


def sqlite_index_tree_ddl() -> tuple[str, str]:
    """Return the (CREATE TABLE, CREATE INDEX) DDL for context_index_nodes on SQLite.

    Exposed so the migration CLI's raw-connection SQLite target init can provision
    the table identically to the server's startup migration (single DDL source, no
    drift). PostgreSQL targets call :func:`apply_index_tree_migration` directly via
    a backend.

    Returns:
        The idempotent CREATE TABLE and CREATE INDEX statements.
    """
    return _CREATE_TABLE_SQLITE, _CREATE_INDEX_SQLITE
