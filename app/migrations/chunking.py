"""
Chunking migration functions for mcp-context-server.

This module handles:
- Schema migration for 1:N embedding relationship
- Backward compatibility for existing embeddings
- embedding_chunks table creation for SQLite
- chunk_count column addition to embedding_metadata
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.errors import format_exception_message
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations._probes import embedding_metadata_table_exists
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def _check_chunking_migration_applied_postgresql(conn: asyncpg.Connection) -> bool:
    """Check if chunking migration already applied for PostgreSQL.

    Args:
        conn: PostgreSQL connection

    Returns:
        True if migration already applied (id column exists), False otherwise
    """
    schema = settings.storage.postgresql_schema
    result = await conn.fetchval('''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = $1
              AND table_name = 'vec_context_embeddings'
              AND column_name = 'id'
        )
    ''', schema)
    return bool(result)


def _check_chunking_migration_applied_sqlite(conn: sqlite3.Connection) -> bool:
    """Check if chunking migration already applied for SQLite.

    Args:
        conn: SQLite connection

    Returns:
        True if migration already applied (embedding_chunks table exists), False otherwise
    """
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_chunks'",
    )
    return cursor.fetchone() is not None


def _check_chunk_count_column_sqlite(conn: sqlite3.Connection) -> bool:
    """Check if chunk_count column exists in embedding_metadata for SQLite.

    Args:
        conn: SQLite connection

    Returns:
        True if chunk_count column exists, False otherwise
    """
    cursor = conn.execute('PRAGMA table_info(embedding_metadata)')
    columns = [row[1] for row in cursor.fetchall()]
    return 'chunk_count' in columns


def _check_chunk_boundary_columns_sqlite(conn: sqlite3.Connection) -> bool:
    """Check if chunk boundary columns exist in embedding_chunks for SQLite.

    Args:
        conn: SQLite connection

    Returns:
        True if both start_index and end_index columns exist, False otherwise
    """
    cursor = conn.execute('PRAGMA table_info(embedding_chunks)')
    columns = [row[1] for row in cursor.fetchall()]
    return 'start_index' in columns and 'end_index' in columns


def _check_embedding_metadata_exists_sqlite(conn: sqlite3.Connection) -> bool:
    """Check if embedding_metadata table exists for SQLite.

    Args:
        conn: SQLite connection

    Returns:
        True if table exists, False otherwise
    """
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_metadata'",
    )
    return cursor.fetchone() is not None


async def apply_chunking_migration(backend: StorageBackend, *, force: bool = False) -> None:
    """Apply chunking migration for 1:N embedding relationship.

    This migration:
    1. Converts vec_context_embeddings from 1:1 to 1:N relationship
    2. Preserves existing embeddings (they become single-chunk entries)
    3. Adds chunk_count to embedding_metadata

    Args:
        backend: Storage backend instance.
        force: When True, apply the migration regardless of
            ``settings.embedding.generation_enabled``. Used by the migration CLI to
            build the chunk-boundary columns on a target database; the server
            keeps its default gated behavior.

    Raises:
        RuntimeError: If migration execution fails.

    Note:
        - Applies when embedding generation is enabled, and ALSO -- with
          generation off -- when the database already carries embedding
          infrastructure (the ``embedding_metadata`` table); only a FRESH
          generation-off database skips.
        - Idempotent: Uses IF NOT EXISTS / IF EXISTS patterns
        - Must be called after apply_semantic_search_migration()
    """
    # Infra-present fallthrough shared with the semantic and compression
    # migrations (probe in app/migrations/_probes.py): an infra-carrying
    # database keeps its active-format embedding layout maintained even
    # while generation is toggled off; a fresh generation-off database
    # skips.
    if (
        not force
        and not settings.embedding.generation_enabled
        and not await embedding_metadata_table_exists(backend)
    ):
        return

    backend_type = backend.backend_type

    # Select migration file based on backend type
    migration_filename = ('add_chunking_postgresql.sql' if backend_type == 'postgresql'
                          else 'add_chunking_sqlite.sql')

    migration_path = Path(__file__).parent / migration_filename

    if not migration_path.exists():
        error_msg = f'Chunking migration file not found: {migration_path}'
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        migration_sql_template = migration_path.read_text(encoding='utf-8')
        # When compression is enabled the compressed table replaces the fp32
        # vec_context_embeddings table; on PostgreSQL the chunking migration must NOT
        # restructure the now-absent fp32 table (ALTER on a missing table errors). Skip
        # those statements while still adding chunk_count to embedding_metadata. The CLI
        # (force=True) builds the full fp32 layout and never runs the compression
        # migration, so it keeps the restructure. (SQLite chunking only creates
        # embedding_chunks, which compression preserves, so it needs no skip.)
        skip_fp32_vec = settings.compression.enabled and not force
        await _apply_chunking_migration_with_backend(
            backend, migration_sql_template, skip_fp32_vec=skip_fp32_vec,
        )
    except Exception as e:
        logger.error(f'Failed to apply chunking migration: {e}')
        raise RuntimeError(f'Chunking migration failed: {format_exception_message(e)}') from e


async def _apply_chunking_migration_with_backend(
    manager: StorageBackend,
    migration_sql_template: str,
    skip_fp32_vec: bool = False,
) -> None:
    """Apply chunking migration with a given backend.

    Args:
        manager: The backend to use for migration
        migration_sql_template: The migration SQL template. Chunking
            migration uses BARE table names and current_schema()
            filters; the loader does not substitute {SCHEMA} for this
            migration. Operators with a non-default POSTGRESQL_SCHEMA
            configure search_path on their connection pool.
        skip_fp32_vec: When True (server path with compression enabled), omit the
            PostgreSQL statements that restructure the fp32 ``vec_context_embeddings``
            table -- it does not exist under compression -- while still adding the
            ``chunk_count`` column to ``embedding_metadata``. Has no effect on SQLite,
            whose chunking migration only creates ``embedding_chunks`` (preserved
            under compression) and never references the fp32 vec table in executable DDL.
    """
    if manager.backend_type == 'postgresql':
        # Check if already applied
        already_applied = await manager.execute_read(
            cast(Any, _check_chunking_migration_applied_postgresql),
        )

        # Early return if migration already applied (avoid unnecessary SQL execution)
        if already_applied:
            logger.info('Chunking migration: already applied for PostgreSQL, skipping SQL execution')
            return

        # Migration SQL uses BARE table names; idempotency-check
        # filters use current_schema() to introspect the resolved
        # schema. No template substitution required.
        migration_sql = migration_sql_template

        # Get migration timeout from settings
        migration_timeout_s = settings.storage.postgresql_migration_timeout_s

        async def _apply_postgresql(conn: asyncpg.Connection) -> None:
            # Raise the transaction-scoped statement_timeout to the migration budget and take
            # the shared advisory lock under it. SET LOCAL auto-reverts on COMMIT/ROLLBACK, so
            # there is no finally-restore SET -- which, in an aborted transaction, would raise
            # 25P02 and mask the real DDL error. The lock acquire and the DDL run under the
            # migration timeout so the pool's shorter command_timeout cannot cancel them.
            await begin_migration(conn, migration_timeout_s)

            # Parse SQL statements, handling dollar-quoted DO blocks
            statements: list[str] = []
            current_stmt: list[str] = []
            in_dollar_quote = False

            for line in migration_sql.split('\n'):
                stripped = line.strip()
                # Skip comment-only lines outside dollar quotes
                if stripped.startswith('--') and not in_dollar_quote:
                    continue
                # Track dollar-quoted strings (DO blocks)
                if '$$' in stripped:
                    in_dollar_quote = not in_dollar_quote
                if stripped:
                    current_stmt.append(line)
                # End of statement: semicolon when not in dollar quotes
                if stripped.endswith(';') and not in_dollar_quote:
                    statements.append('\n'.join(current_stmt))
                    current_stmt = []

            # Add any remaining statement
            if current_stmt:
                statements.append('\n'.join(current_stmt))

            # Execute each statement
            for stmt in statements:
                stmt = stmt.strip()
                if not stmt or stmt.startswith('--'):
                    continue
                # Under compression the fp32 vec_context_embeddings table is absent
                # (replaced by the compressed table), so skip the DO blocks that
                # ALTER / index it; the chunk_count ALTER on embedding_metadata,
                # which carries no such reference, still runs.
                if skip_fp32_vec and 'vec_context_embeddings' in stmt:
                    continue
                await execute_migration_ddl(conn, stmt, migration_timeout_s)

        await manager.execute_write(cast(Any, _apply_postgresql))
        logger.info('Applied chunking migration for PostgreSQL: 1:N embedding relationship enabled')

    else:  # sqlite
        # Check if already applied
        already_applied = await manager.execute_read(_check_chunking_migration_applied_sqlite)

        def _apply_sqlite(conn: sqlite3.Connection) -> None:
            # Check if embedding_metadata table exists before migration
            # If it doesn't exist, skip this migration (semantic search not initialized)
            if not _check_embedding_metadata_exists_sqlite(conn):
                logger.debug('Chunking migration: skipping - embedding_metadata table does not exist')
                return

            # Check if embedding_chunks table exists but lacks boundary columns
            # This handles upgrade from pre-f36266c schema where table was created
            # without start_index and end_index columns
            if _check_chunking_migration_applied_sqlite(conn) and not _check_chunk_boundary_columns_sqlite(conn):
                logger.info('Chunking migration: adding missing boundary columns to embedding_chunks')
                conn.execute(
                    'ALTER TABLE embedding_chunks ADD COLUMN start_index INTEGER NOT NULL DEFAULT 0',
                )
                conn.execute(
                    'ALTER TABLE embedding_chunks ADD COLUMN end_index INTEGER NOT NULL DEFAULT 0',
                )

            # Execute the migration SQL
            conn.executescript(migration_sql_template)

            # Handle chunk_count column (SQLite doesn't support ADD COLUMN IF NOT EXISTS)
            if not _check_chunk_count_column_sqlite(conn):
                conn.execute(
                    'ALTER TABLE embedding_metadata ADD COLUMN chunk_count INTEGER NOT NULL DEFAULT 1',
                )

        await manager.execute_write(_apply_sqlite)

        if not already_applied:
            logger.info('Applied chunking migration for SQLite: 1:N embedding relationship enabled')
        else:
            logger.info('Chunking migration: verified for SQLite')
