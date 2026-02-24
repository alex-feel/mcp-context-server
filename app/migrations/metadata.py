"""
Metadata field index management for mcp-context-server.

This module handles creation, synchronization, and cleanup of expression
indexes on metadata JSON/JSONB fields based on configuration settings.
"""

import logging
import sqlite3
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _generate_create_index_sqlite(field: str) -> str:
    """Generate SQLite CREATE INDEX statement for metadata field.

    Args:
        field: Metadata field name.

    Returns:
        SQL CREATE INDEX statement using json_extract for expression index.
    """
    return f'''
CREATE INDEX IF NOT EXISTS idx_metadata_{field}
ON context_entries(json_extract(metadata, '$.{field}'))
WHERE json_extract(metadata, '$.{field}') IS NOT NULL;
'''


def _generate_create_index_postgresql(field: str, type_hint: str) -> str:
    """Generate PostgreSQL CREATE INDEX statement for metadata field.

    Args:
        field: Metadata field name.
        type_hint: Type hint for casting (string, integer, boolean, float).

    Returns:
        SQL CREATE INDEX statement using JSONB operators.
    """
    # Type cast mapping for typed comparisons
    type_cast_map = {
        'integer': '::INTEGER',
        'boolean': '::BOOLEAN',
        'float': '::NUMERIC',
        'string': '',
    }
    type_cast = type_cast_map.get(type_hint, '')

    if type_cast:
        return f'''
CREATE INDEX IF NOT EXISTS idx_metadata_{field}
ON context_entries(((metadata->>'{field}'){type_cast}))
WHERE metadata->>'{field}' IS NOT NULL;
'''
    return f'''
CREATE INDEX IF NOT EXISTS idx_metadata_{field}
ON context_entries((metadata->>'{field}'))
WHERE metadata->>'{field}' IS NOT NULL;
'''


async def _get_existing_metadata_indexes(backend: StorageBackend) -> tuple[set[str], set[str]]:
    """Query database for existing metadata field indexes.

    Args:
        backend: The storage backend to query.

    Returns:
        A tuple of (simple_indexes, orphan_compound_indexes):
        - simple_indexes: Field names from idx_metadata_{field} pattern
        - orphan_compound_indexes: Field names from idx_thread_metadata_{field} pattern
          (ALL compound indexes are considered orphans since they are not dynamically managed)

        Excludes GIN index (idx_metadata_gin).
    """
    if backend.backend_type == 'sqlite':

        def _query_sqlite_indexes(conn: sqlite3.Connection) -> tuple[set[str], set[str]]:
            # Query both idx_metadata_* and idx_thread_metadata_* patterns
            cursor = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND tbl_name='context_entries' "
                "AND (name LIKE 'idx_metadata_%' OR name LIKE 'idx_thread_metadata_%')",
            )
            simple_indexes: set[str] = set()
            orphan_compound_indexes: set[str] = set()
            for row in cursor:
                name = row[0]
                if name.startswith('idx_thread_metadata_'):
                    # ALL compound indexes are orphans (not dynamically managed)
                    # Extract field: idx_thread_metadata_priority -> priority
                    field = name[20:]  # len('idx_thread_metadata_') = 20
                    orphan_compound_indexes.add(field)
                elif name.startswith('idx_metadata_'):
                    # Simple index - extract field name
                    field = name[13:]  # len('idx_metadata_') = 13
                    # Skip GIN index marker (PostgreSQL only, shouldn't appear in SQLite)
                    if field != 'gin':
                        simple_indexes.add(field)
            return simple_indexes, orphan_compound_indexes

        return await backend.execute_read(_query_sqlite_indexes)

    # postgresql

    async def _query_postgresql_indexes(conn: asyncpg.Connection) -> tuple[set[str], set[str]]:
        # Query both patterns with schema qualification for proper isolation
        # Use configured schema (default: 'public') instead of current_schema()
        # which may return wrong schema in Supabase environments
        schema = settings.storage.postgresql_schema
        rows = await conn.fetch(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = 'context_entries' "
            "AND schemaname = $1 "
            "AND (indexname LIKE 'idx_metadata_%' OR indexname LIKE 'idx_thread_metadata_%')",
            schema,
        )
        simple_indexes: set[str] = set()
        orphan_compound_indexes: set[str] = set()
        for row in rows:
            name = row['indexname']
            if name.startswith('idx_thread_metadata_'):
                # ALL compound indexes are orphans (not dynamically managed)
                # Extract field: idx_thread_metadata_priority -> priority
                field = name[20:]  # len('idx_thread_metadata_') = 20
                orphan_compound_indexes.add(field)
            elif name.startswith('idx_metadata_'):
                # Simple index - extract field name
                field = name[13:]  # len('idx_metadata_') = 13
                # Skip GIN index (idx_metadata_gin)
                if field != 'gin':
                    simple_indexes.add(field)
        return simple_indexes, orphan_compound_indexes

    return await backend.execute_read(cast(Any, _query_postgresql_indexes))


async def _create_metadata_index(backend: StorageBackend, field: str, type_hint: str) -> None:
    """Create expression index for a metadata field.

    Args:
        backend: Storage backend.
        field: Metadata field name.
        type_hint: Type hint for the field (string, integer, boolean, float, array, object).

    Note:
        Array and object fields are skipped for SQLite as they require GIN indexes
        which SQLite does not support. PostgreSQL uses the existing GIN index
        (idx_metadata_gin) for array/object containment queries.
    """
    backend_type = backend.backend_type

    # Skip array and object fields for SQLite - they require GIN indexes
    if backend_type == 'sqlite' and type_hint in ('array', 'object'):
        logger.info(
            f'Skipping index for {type_hint} field "{field}" on SQLite. '
            f'Array/object fields cannot be efficiently indexed in SQLite. '
            f'For high-performance queries on these fields, use PostgreSQL with GIN index.',
        )
        return

    # For PostgreSQL, array/object types use existing GIN index - no additional index needed
    if backend_type == 'postgresql' and type_hint in ('array', 'object'):
        logger.info(
            f'Field "{field}" is {type_hint} type - using existing GIN index (idx_metadata_gin) '
            f'for containment queries. No additional expression index needed.',
        )
        return

    logger.info(f'Creating metadata index: idx_metadata_{field} (type={type_hint})')

    if backend_type == 'sqlite':
        sql = _generate_create_index_sqlite(field)

        def _create_sqlite_index(conn: sqlite3.Connection) -> None:
            conn.execute(sql)

        await backend.execute_write(_create_sqlite_index)

    else:  # postgresql
        sql = _generate_create_index_postgresql(field, type_hint)

        async def _create_postgresql_index(conn: asyncpg.Connection) -> None:
            # Acquire transaction-scoped advisory lock for multi-pod DDL safety.
            # pg_advisory_xact_lock releases automatically on COMMIT or ROLLBACK,
            # aligning with execute_write()'s conn.transaction() wrapper.
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'))")
            await conn.execute(sql)

        await backend.execute_write(cast(Any, _create_postgresql_index))


async def _drop_metadata_index(backend: StorageBackend, field: str, *, is_compound: bool = False) -> None:
    """Drop expression index for a metadata field.

    Args:
        backend: Storage backend.
        field: Metadata field name.
        is_compound: If True, drops compound index (idx_thread_metadata_{field}),
                     otherwise drops simple index (idx_metadata_{field}).
    """
    index_name = f'idx_thread_metadata_{field}' if is_compound else f'idx_metadata_{field}'

    logger.info(f'Dropping metadata index: {index_name}')

    if backend.backend_type == 'sqlite':
        sql = f'DROP INDEX IF EXISTS {index_name};'

        def _drop_sqlite_index(conn: sqlite3.Connection) -> None:
            conn.execute(sql)

        await backend.execute_write(_drop_sqlite_index)

    else:  # postgresql
        # Use schema-qualified DROP to ensure correct index is dropped
        # This handles multi-schema environments like Supabase
        async def _drop_postgresql_index(conn: asyncpg.Connection) -> None:
            # Acquire transaction-scoped advisory lock for multi-pod DDL safety.
            # pg_advisory_xact_lock releases automatically on COMMIT or ROLLBACK,
            # aligning with execute_write()'s conn.transaction() wrapper.
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'))")
            # Use configured schema for qualified drop.
            # This ensures correct schema is used in Supabase environments.
            schema = settings.storage.postgresql_schema
            sql = f'DROP INDEX IF EXISTS {schema}.{index_name};'
            await conn.execute(sql)

        await backend.execute_write(cast(Any, _drop_postgresql_index))


async def handle_metadata_indexes(backend: StorageBackend) -> None:
    """Handle metadata field indexing based on configuration and sync mode.

    This function manages expression indexes on metadata JSON fields according to
    METADATA_INDEXED_FIELDS and METADATA_INDEX_SYNC_MODE environment variables.

    Sync Modes:
        - strict: Fail startup if indexes don't match configuration exactly
        - auto: Automatically add missing and drop extra indexes (including orphan compound indexes)
        - warn: Log warnings about mismatches but continue startup
        - additive: Only add missing indexes, never drop (default)

    Args:
        backend: The storage backend to use for database operations.

    Raises:
        RuntimeError: In strict mode, if index configuration doesn't match database.
    """
    configured_fields = settings.storage.metadata_indexed_fields
    sync_mode = settings.storage.metadata_index_sync_mode
    backend_type = backend.backend_type

    logger.debug(f'Handling metadata indexes: mode={sync_mode}, fields={list(configured_fields.keys())}')

    # Get existing indexes from database (returns tuple of simple indexes and orphan compound indexes)
    existing_simple_indexes, orphan_compound_indexes = await _get_existing_metadata_indexes(backend)

    # Calculate differences for simple indexes
    # For SQLite, exclude array/object fields from configured set (they can't be indexed)
    if backend_type == 'sqlite':
        configured_set = {
            field for field, type_hint in configured_fields.items() if type_hint not in ('array', 'object')
        }
    else:
        # For PostgreSQL, array/object fields use GIN index, so also exclude from expression index comparison
        configured_set = {
            field for field, type_hint in configured_fields.items() if type_hint not in ('array', 'object')
        }

    missing = configured_set - existing_simple_indexes
    extra = existing_simple_indexes - configured_set

    # Log current state
    if missing:
        logger.info(f'Missing metadata indexes: {missing}')
    if extra:
        logger.info(f'Extra metadata indexes not in config: {extra}')
    if orphan_compound_indexes:
        logger.info(f'Orphan compound indexes from old schema: {orphan_compound_indexes}')

    # Handle based on sync mode
    if sync_mode == 'strict':
        if missing or extra or orphan_compound_indexes:
            raise RuntimeError(
                f'Metadata index mismatch (METADATA_INDEX_SYNC_MODE=strict). '
                f'Missing: {missing or "none"}, Extra: {extra or "none"}, '
                f'Orphan compound: {orphan_compound_indexes or "none"}. '
                f'Update METADATA_INDEXED_FIELDS or run with different sync mode.',
            )

    elif sync_mode == 'auto':
        # Drop extra simple indexes
        for field in extra:
            await _drop_metadata_index(backend, field)

        # Drop orphan compound indexes (from old schema versions)
        for field in orphan_compound_indexes:
            await _drop_metadata_index(backend, field, is_compound=True)

        # Create missing indexes
        for field in missing:
            type_hint = configured_fields.get(field, 'string')
            await _create_metadata_index(backend, field, type_hint)

    elif sync_mode == 'warn':
        if missing:
            logger.warning(
                f'Missing metadata indexes: {missing}. '
                f'Queries filtering on these fields may be slow.',
            )
        if extra:
            logger.warning(
                f'Extra metadata indexes not in config: {extra}. '
                f'Consider cleanup or updating METADATA_INDEXED_FIELDS.',
            )
        if orphan_compound_indexes:
            logger.warning(
                f'Orphan compound indexes from old schema: {orphan_compound_indexes}. '
                f'Use METADATA_INDEX_SYNC_MODE=auto to remove them.',
            )

    elif sync_mode == 'additive':
        # Only create missing indexes, never drop
        for field in missing:
            type_hint = configured_fields.get(field, 'string')
            await _create_metadata_index(backend, field, type_hint)

        if extra:
            logger.info(
                f'Extra metadata indexes detected: {extra}. '
                f'Use METADATA_INDEX_SYNC_MODE=auto to remove them.',
            )
        if orphan_compound_indexes:
            logger.info(
                f'Orphan compound indexes detected: {orphan_compound_indexes}. '
                f'Use METADATA_INDEX_SYNC_MODE=auto to remove them.',
            )

    logger.debug('Metadata index handling completed')
