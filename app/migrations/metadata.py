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
from app.backends.base import MAINTENANCE
from app.errors import ConfigurationError
from app.metadata_sql import pg_number_guard
from app.metadata_sql import pg_numeric_accessor
from app.metadata_types import PG_INDEX_TEXT_CASTS
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _escape_sql_string_literal(value: str) -> str:
    """Double single quotes so a value is safe inside a single-quoted SQL literal.

    Defense in depth for the generated JSON path/key literals below: the field names
    are already constrained to plain SQL identifiers at the configuration boundary
    (``StorageSettings._validate_metadata_indexed_field_names``), so no quote can
    reach here in practice, but escaping the literal keeps the DDL generator safe on
    its own rather than relying solely on the upstream validator.

    Args:
        value: The value to embed inside a single-quoted SQL string literal.

    Returns:
        The value with every single quote doubled.
    """
    return value.replace("'", "''")


def _generate_create_index_sqlite(field: str) -> str:
    """Generate SQLite CREATE INDEX statement for metadata field.

    Args:
        field: Metadata field name (a validated SQL identifier).

    Returns:
        SQL CREATE INDEX statement using json_extract for expression index.
    """
    path_literal = _escape_sql_string_literal(f'$.{field}')
    return f'''
CREATE INDEX IF NOT EXISTS idx_metadata_{field}
ON context_entries(json_extract(metadata, '{path_literal}'))
WHERE json_extract(metadata, '{path_literal}') IS NOT NULL;
'''


def _postgresql_index_definition(field: str, type_hint: str) -> str:
    """Return the indexed expression and predicate of a field's PostgreSQL index.

    A ``float`` field indexes the total numeric accessor the numeric filters compare,
    over JSON numbers only: the expression and the predicate are the filters' own
    accessor and number guard, so the index serves those filters, and the cast runs
    only for a JSON number, so no stored value can make an INSERT fail on it. An
    ``integer`` or ``boolean`` field indexes a cast of the text form; any other field
    indexes the text form.

    Args:
        field: Metadata field name (a validated SQL identifier).
        type_hint: The field's type hint (string, integer, boolean, float).

    Returns:
        ``(<expression>) WHERE <predicate>``, the part of CREATE INDEX after the table name.
    """
    key_literal = _escape_sql_string_literal(field)
    if type_hint == 'float':
        return f'(({pg_numeric_accessor(key_literal)})) WHERE {pg_number_guard(key_literal)}'
    text = f"metadata->>'{key_literal}'"
    type_cast = PG_INDEX_TEXT_CASTS.get(type_hint, '')
    if type_cast:
        return f'((({text}){type_cast})) WHERE {text} IS NOT NULL'
    return f'(({text})) WHERE {text} IS NOT NULL'


def _generate_create_index_postgresql(field: str, type_hint: str) -> str:
    """Generate PostgreSQL CREATE INDEX statement for metadata field.

    Args:
        field: Metadata field name (a validated SQL identifier).
        type_hint: Type hint for casting (string, integer, boolean, float).

    Returns:
        SQL CREATE INDEX statement using JSONB operators.
    """
    from app.backends.postgresql_backend.session import quote_pg_identifier

    # Quote the generated index identifier so a mixed-case field name is stored
    # case-preserved in the catalog. PostgreSQL folds an unquoted identifier to
    # lowercase, which would make pg_indexes report a lowercased name that does
    # not round-trip against the case-preserved configured field, breaking the
    # reconciliation diff (perpetual drop/recreate in auto mode, a permanent
    # false positive in strict mode). The JSON key literal stays case-sensitive.
    index_name = quote_pg_identifier(f'idx_metadata_{field}')
    return f'''
CREATE INDEX IF NOT EXISTS {index_name}
ON context_entries{_postgresql_index_definition(field, type_hint)};
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

    Raises:
        ConfigurationError: When an integer or boolean field's index cannot be built
            because rows already stored hold a value its cast rejects. That is an
            operator decision, not a transient fault, so it must not be crash-looped.

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

        await backend.execute_write(_create_sqlite_index, access_scope=MAINTENANCE)

    else:  # postgresql
        sql = _generate_create_index_postgresql(field, type_hint)
        migration_timeout_s = settings.storage.postgresql_migration_timeout_s

        async def _create_postgresql_index(conn: asyncpg.Connection) -> None:
            # Raise the transaction-scoped statement_timeout to the migration budget and take
            # the shared advisory lock under it (SET LOCAL auto-reverts on COMMIT/ROLLBACK, so
            # no finally-restore that would raise 25P02 in an aborted transaction and mask the
            # real error). CREATE INDEX on a large context_entries table can exceed the regular
            # per-query budget, so the lock acquire and the DDL both run under the migration
            # timeout, mirroring the schema migrations.
            await begin_migration(conn, migration_timeout_s)
            await execute_migration_ddl(conn, sql, migration_timeout_s)

        try:
            await backend.execute_write(cast(Any, _create_postgresql_index), access_scope=MAINTENANCE)
        except asyncpg.exceptions.DataError as exc:
            raise ConfigurationError(_uncastable_rows_message(field, type_hint, exc)) from exc


def _uncastable_rows_message(field: str, type_hint: str, exc: asyncpg.exceptions.DataError) -> str:
    """Name the field and configuration behind an index build that stored rows make fail.

    An ``integer`` or ``boolean`` field puts a hard SQL cast inside the index
    expression, and PostgreSQL evaluates it for EVERY existing row while building the
    index. A row stored before the field carried this type hint -- or migrated in from
    SQLite, whose uncast json_extract index accepts anything -- makes that cast fail,
    and the raw driver error reads like a database fault the supervisor should retry.
    It is neither: the same rows fail the same way on every restart. Naming the field
    and the configuration that selected the cast turns a crash loop into a decision the
    operator can act on. PostgreSQL's own message quotes the offending value, so it is
    carried through verbatim.

    Args:
        field: The metadata field whose index could not be built.
        type_hint: The field's configured type hint.
        exc: The driver error the index build raised.

    Returns:
        The message of the configuration error raised in place of the driver error.
    """
    return (
        f'Cannot index metadata field "{field}" as {type_hint}: rows already stored '
        f'hold a value that cast rejects ({exc}). The write boundary refuses such a '
        f'value now, so this comes from data stored earlier or migrated in from '
        f'SQLite, whose index applies no cast. Correct those rows, or change this '
        f'field in METADATA_INDEXED_FIELDS (drop the type hint to index it as text, '
        f'or remove the field).'
    )


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

        await backend.execute_write(_drop_sqlite_index, access_scope=MAINTENANCE)

    else:  # postgresql
        # Use schema-qualified DROP to ensure correct index is dropped
        # This handles multi-schema environments like Supabase
        migration_timeout_s = settings.storage.postgresql_migration_timeout_s

        async def _drop_postgresql_index(conn: asyncpg.Connection) -> None:
            # Raise the transaction-scoped statement_timeout to the migration budget and take
            # the shared advisory lock under it (SET LOCAL auto-reverts on COMMIT/ROLLBACK, so
            # no finally-restore that would raise 25P02 in an aborted transaction and mask the
            # real error). The lock wait, or a DROP blocked on a concurrent ACCESS EXCLUSIVE
            # lock, can exceed the regular per-query budget, so both run under the migration
            # timeout, mirroring the schema migrations and the create path.
            await begin_migration(conn, migration_timeout_s)
            # Quote both the configured schema AND the generated index identifier via the shared
            # quote_pg_identifier helper so the qualified DROP targets the exact object that the
            # CREATE path created. PostgreSQL folds an unquoted mixed-case or reserved identifier
            # to lowercase; an unquoted schema would resolve to the wrong (or a nonexistent) schema
            # and an unquoted index name would target the lowercased name rather than the
            # case-preserved one the quoted CREATE produced, either of which lets IF EXISTS
            # silently skip the real index the reconciliation diff asked to drop.
            from app.backends.postgresql_backend.session import quote_pg_identifier

            schema = quote_pg_identifier(settings.storage.postgresql_schema)
            quoted_index_name = quote_pg_identifier(index_name)
            sql = f'DROP INDEX IF EXISTS {schema}.{quoted_index_name};'
            await execute_migration_ddl(conn, sql, migration_timeout_s)

        await backend.execute_write(cast(Any, _drop_postgresql_index), access_scope=MAINTENANCE)


# Empty temporary table the drift check builds probe indexes on: an index built from
# the current definition on it costs no scan, and the table is dropped at commit.
_PROBE_TABLE = 'mcp_metadata_index_probe'

# Name a rebuild gives the replacement index until the drifted one is dropped. A
# configured field name may already fill the 63-byte identifier budget behind
# 'idx_metadata_', so the staging name is fixed rather than derived from the field.
_REBUILD_INDEX_NAME = 'mcp_metadata_index_rebuild'


async def _index_drifted(conn: asyncpg.Connection, field: str, type_hint: str, *, probe: str) -> bool:
    """Return True when a field's existing PostgreSQL index differs from the current definition.

    PostgreSQL stores an index definition in normalized form, so the existing index is
    compared with a probe index built from the current definition on the empty
    temporary table, both read back through ``pg_get_indexdef`` from their access
    method on, which leaves the index and table names out. The caller creates the
    temporary table in the same transaction.

    Args:
        conn: A connection inside the transaction that created the probe table.
        field: The metadata field.
        type_hint: The field's configured type hint.
        probe: Name for the probe index, unique within the transaction.

    Returns:
        Whether the existing index differs; False when the field has no index.
    """
    existing = await conn.fetchval(
        'SELECT pg_get_indexdef(c.oid) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace '
        "WHERE n.nspname = $1 AND c.relname = $2 AND c.relkind = 'i'",
        settings.storage.postgresql_schema,
        f'idx_metadata_{field}',
    )
    if existing is None:
        return False
    await conn.execute(f'CREATE INDEX {probe} ON pg_temp.{_PROBE_TABLE} {_postgresql_index_definition(field, type_hint)}')
    expected = await conn.fetchval('SELECT pg_get_indexdef(to_regclass($1))', f'pg_temp.{probe}')
    return bool(existing.partition(' USING ')[2] != str(expected).partition(' USING ')[2])


async def _create_probe_table(conn: asyncpg.Connection) -> None:
    """Create the empty temporary table the drift check builds its probe indexes on."""
    await conn.execute(f'CREATE TEMPORARY TABLE IF NOT EXISTS {_PROBE_TABLE} (metadata JSONB) ON COMMIT DROP')


async def _drifted_metadata_indexes(backend: StorageBackend, fields: dict[str, str]) -> set[str]:
    """Return the configured fields whose existing PostgreSQL index differs from the current definition.

    Args:
        backend: The PostgreSQL storage backend.
        fields: Configured fields that have an index, mapped to their type hints.

    Returns:
        The drifted field names.
    """
    if not fields:
        return set()

    async def _compare(conn: asyncpg.Connection) -> set[str]:
        await _create_probe_table(conn)
        return {
            field
            for position, (field, type_hint) in enumerate(sorted(fields.items()))
            if await _index_drifted(conn, field, type_hint, probe=f'{_PROBE_TABLE}_{position}')
        }

    return cast(set[str], await backend.execute_write(cast(Any, _compare), access_scope=MAINTENANCE))


async def _rebuild_metadata_index(backend: StorageBackend, field: str, type_hint: str) -> None:
    """Replace a field's drifted PostgreSQL index with the current definition.

    One transaction under the shared migration lock re-checks the drift, so a server
    that finds it already rebuilt by another one does nothing, then builds the new index
    under a staging name, drops the drifted one and gives the new one its name. A
    definition that cannot be built (stored rows hold a value a typed cast rejects)
    rolls the whole transaction back and leaves the existing index in place.

    Args:
        backend: The PostgreSQL storage backend.
        field: The metadata field.
        type_hint: The field's configured type hint.

    Raises:
        ConfigurationError: When stored rows hold a value the new definition's cast rejects.
    """
    from app.backends.postgresql_backend.session import quote_pg_identifier

    logger.info(f'Rebuilding metadata index: idx_metadata_{field} (type={type_hint})')
    schema = quote_pg_identifier(settings.storage.postgresql_schema)
    index_name = quote_pg_identifier(f'idx_metadata_{field}')
    staging_name = quote_pg_identifier(_REBUILD_INDEX_NAME)
    migration_timeout_s = settings.storage.postgresql_migration_timeout_s
    statements = (
        f'CREATE INDEX {staging_name} ON {schema}.context_entries {_postgresql_index_definition(field, type_hint)}',
        f'DROP INDEX {schema}.{index_name}',
        f'ALTER INDEX {schema}.{staging_name} RENAME TO {index_name}',
    )

    async def _rebuild_postgresql_index(conn: asyncpg.Connection) -> None:
        await begin_migration(conn, migration_timeout_s)
        await _create_probe_table(conn)
        if not await _index_drifted(conn, field, type_hint, probe=f'{_PROBE_TABLE}_rebuild'):
            return
        for statement in statements:
            await execute_migration_ddl(conn, statement, migration_timeout_s)

    try:
        await backend.execute_write(cast(Any, _rebuild_postgresql_index), access_scope=MAINTENANCE)
    except asyncpg.exceptions.DataError as exc:
        raise ConfigurationError(_uncastable_rows_message(field, type_hint, exc)) from exc


async def handle_metadata_indexes(backend: StorageBackend) -> None:
    """Handle metadata field indexing based on configuration and sync mode.

    This function manages expression indexes on metadata JSON fields according to
    METADATA_INDEXED_FIELDS and METADATA_INDEX_SYNC_MODE environment variables.

    On PostgreSQL an existing index whose definition differs from the one this server
    creates for the field's configured type hint is DRIFTED (compared through
    ``pg_get_indexdef``). A drifted index is handled like a missing one: the modes that
    create missing indexes rebuild it, warn mode reports it, and strict mode refuses it.

    Sync Modes:
        - strict: Fail startup if indexes don't match configuration exactly
        - auto: Add missing indexes, rebuild drifted ones, and drop extra indexes
          (including orphan compound indexes)
        - warn: Log warnings about mismatches but continue startup
        - additive: Add missing indexes and rebuild drifted ones, never drop an index
          the configuration does not name (default)

    Args:
        backend: The storage backend to use for database operations.

    Raises:
        ConfigurationError: In strict mode, if index configuration doesn't match the
            database. A mismatch is a permanent, human-intervention-required
            misconfiguration (the message names the config to change), so it is
            classified as a configuration error (exit 78) that supervisors must not
            auto-retry rather than a generic failure they crash-loop on. Strict mode
            verifies indexes without provisioning or rebuilding them, so it requires the
            indexes to be pre-provisioned -- bootstrap a fresh database once in additive
            or auto mode before flipping to strict.
    """
    configured_fields = settings.storage.metadata_indexed_fields
    sync_mode = settings.storage.metadata_index_sync_mode

    logger.debug(f'Handling metadata indexes: mode={sync_mode}, fields={list(configured_fields.keys())}')

    # Get existing indexes from database (returns tuple of simple indexes and orphan compound indexes)
    existing_simple_indexes, orphan_compound_indexes = await _get_existing_metadata_indexes(backend)

    # Array/object fields get no expression index on either backend: SQLite cannot index
    # them, and PostgreSQL serves them from the GIN index.
    configured_set = {
        field for field, type_hint in configured_fields.items() if type_hint not in ('array', 'object')
    }

    missing = configured_set - existing_simple_indexes
    extra = existing_simple_indexes - configured_set
    drifted: set[str] = set()
    if backend.backend_type == 'postgresql':
        drifted = await _drifted_metadata_indexes(
            backend,
            {field: configured_fields[field] for field in configured_set & existing_simple_indexes},
        )

    # Log current state
    if missing:
        logger.info(f'Missing metadata indexes: {missing}')
    if drifted:
        logger.info(f'Metadata indexes whose definition differs from the configured one: {drifted}')
    if extra:
        logger.info(f'Extra metadata indexes not in config: {extra}')
    if orphan_compound_indexes:
        logger.info(f'Orphan compound indexes from old schema: {orphan_compound_indexes}')

    # Handle based on sync mode
    if sync_mode == 'strict':
        if missing or drifted or extra or orphan_compound_indexes:
            raise ConfigurationError(
                f'Metadata index mismatch (METADATA_INDEX_SYNC_MODE=strict). '
                f'Missing: {missing or "none"}, Drifted: {drifted or "none"}, Extra: {extra or "none"}, '
                f'Orphan compound: {orphan_compound_indexes or "none"}. '
                f'Update METADATA_INDEXED_FIELDS or run with different sync mode. '
                f'Strict mode does not provision or rebuild indexes; bootstrap a fresh database, '
                f'or rebuild drifted indexes, once with additive or auto before flipping to strict.',
            )

    elif sync_mode == 'auto':
        # Drop extra simple indexes
        for field in extra:
            await _drop_metadata_index(backend, field)

        # Drop orphan compound indexes (from old schema versions)
        for field in orphan_compound_indexes:
            await _drop_metadata_index(backend, field, is_compound=True)

        await _provision_metadata_indexes(backend, configured_fields, missing=missing, drifted=drifted)

    elif sync_mode == 'warn':
        if missing:
            logger.warning(
                f'Missing metadata indexes: {missing}. '
                f'Queries filtering on these fields may be slow.',
            )
        if drifted:
            logger.warning(
                f'Metadata indexes whose definition differs from the configured one: {drifted}. '
                f'Queries filtering on these fields may be slow. '
                f'Use METADATA_INDEX_SYNC_MODE=additive or auto to rebuild them.',
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
        # Create missing and rebuild drifted indexes, never drop one the configuration omits
        await _provision_metadata_indexes(backend, configured_fields, missing=missing, drifted=drifted)

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


async def _provision_metadata_indexes(
    backend: StorageBackend,
    configured_fields: dict[str, str],
    *,
    missing: set[str],
    drifted: set[str],
) -> None:
    """Create the missing metadata indexes and rebuild the drifted ones.

    Args:
        backend: The storage backend.
        configured_fields: Configured fields mapped to their type hints.
        missing: Configured fields that have no index.
        drifted: Configured fields whose index differs from the current definition.
    """
    for field in drifted:
        await _rebuild_metadata_index(backend, field, configured_fields[field])
    for field in missing:
        await _create_metadata_index(backend, field, configured_fields.get(field, 'string'))
