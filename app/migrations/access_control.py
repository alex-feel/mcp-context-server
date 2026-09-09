"""Access-control schema migration for mcp-context-server.

Adds the server-stamped ``owner_id`` and ``visibility`` columns to
``context_entries`` and provisions the ``context_entry_grants`` table plus the
access-control lookup indexes, so a database created BEFORE these existed in
the base schema gains them on an in-place upgrade. Mirrors the version-column
migration: auto-applied, unconditional, and idempotent. Fresh databases (and
migration-CLI targets, which build tables from the base schema) already carry
everything, so this migration is a no-op there.

Backfill semantics (fail-closed): existing rows receive
``visibility='private'`` and ``owner_id=ACCESS_CONTROL_DEFAULT_PRINCIPAL`` via
the ``ADD COLUMN ... NOT NULL DEFAULT`` literal -- a single statement per
column, no separate UPDATE pass. The configured principal is re-validated
against :data:`app.settings.SAFE_PRINCIPAL_ID_PATTERN` immediately before DDL
interpolation, so it cannot break out of the quoted literal. The lingering
column default on upgraded databases is inert: the application always stamps
``owner_id`` explicitly on INSERT (fresh base schemas declare no default).
"""

import logging
import sqlite3
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.errors import ConfigurationError
from app.errors import format_exception_message
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.settings import SAFE_PRINCIPAL_ID_PATTERN
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Grants table + index DDL, duplicated from the base schema files the way the
# version and content_hash migrations duplicate theirs: the base schema serves
# fresh initializations (server startup AND migration-CLI targets), this module
# serves in-place upgrades of databases that predate the table.
_CREATE_GRANTS_TABLE_SQLITE = '''
CREATE TABLE IF NOT EXISTS context_entry_grants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id TEXT NOT NULL,
    principal_type TEXT NOT NULL CHECK(principal_type IN ('user', 'group')),
    principal_id TEXT NOT NULL,
    permission TEXT NOT NULL CHECK(permission IN ('read', 'write')),
    granted_by TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
)
'''

_CREATE_GRANTS_TABLE_POSTGRESQL = '''
CREATE TABLE IF NOT EXISTS context_entry_grants (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id UUID NOT NULL,
    principal_type TEXT NOT NULL CHECK(principal_type IN ('user', 'group')),
    principal_id TEXT NOT NULL,
    permission TEXT NOT NULL CHECK(permission IN ('read', 'write')),
    granted_by TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
)
'''

# Index DDL is backend-portable (bare names resolve via search_path on
# PostgreSQL, matching the peer migration convention).
_INDEX_STATEMENTS = (
    (
        'CREATE UNIQUE INDEX IF NOT EXISTS idx_grants_entry_principal '
        'ON context_entry_grants(context_entry_id, principal_type, principal_id, permission)'
    ),
    (
        'CREATE INDEX IF NOT EXISTS idx_grants_principal '
        'ON context_entry_grants(principal_type, principal_id, context_entry_id)'
    ),
    'CREATE INDEX IF NOT EXISTS idx_context_owner ON context_entries(owner_id)',
    'CREATE INDEX IF NOT EXISTS idx_context_owner_thread ON context_entries(owner_id, thread_id)',
    "CREATE INDEX IF NOT EXISTS idx_context_public ON context_entries(visibility) WHERE visibility = 'public'",
)


def _validated_default_principal() -> str:
    """Return the configured default principal, re-checked for DDL safety.

    The settings validator already enforces the same pattern at construction
    time; this re-check runs immediately before the value is interpolated into
    an ``ADD COLUMN ... DEFAULT '<value>'`` literal, so DDL safety never
    depends on the settings layer alone.

    Returns:
        The validated principal id.

    Raises:
        ConfigurationError: If the value falls outside the safe character set.
    """
    principal = settings.access_control.default_principal
    if not SAFE_PRINCIPAL_ID_PATTERN.fullmatch(principal):
        raise ConfigurationError(
            f'ACCESS_CONTROL_DEFAULT_PRINCIPAL {principal!r} is not safe for SQL '
            'interpolation (allowed: 1-128 characters from A-Z a-z 0-9 . _ @ : + -)',
        )
    return principal


async def apply_access_control_migration(backend: StorageBackend) -> None:
    """Provision the access-control columns, grants table, and indexes.

    Idempotent:
    - SQLite: ``PRAGMA table_info`` probes gate the ``ADD COLUMN`` statements;
      table and index creation use ``IF NOT EXISTS``.
    - PostgreSQL: ``ADD COLUMN IF NOT EXISTS`` and ``IF NOT EXISTS`` DDL under
      the schema-init advisory lock.

    Must run before any write path stamps ``owner_id``/``visibility``, so it is
    applied during startup alongside the other column migrations.

    Args:
        backend: Storage backend instance.

    Raises:
        ConfigurationError: If the configured default principal is not safe for
            DDL interpolation.
        RuntimeError: If migration execution fails.
    """
    default_principal = _validated_default_principal()

    try:
        if backend.backend_type == 'sqlite':
            await _apply_sqlite(backend, default_principal)
        else:
            await _apply_postgresql(backend, default_principal)
    except ConfigurationError:
        raise
    except Exception as e:
        logger.error(f'Failed to apply access-control migration: {e}')
        raise RuntimeError(f'Access-control migration failed: {format_exception_message(e)}') from e


async def _apply_sqlite(backend: StorageBackend, default_principal: str) -> None:
    """Apply the SQLite access-control DDL (idempotent)."""

    def _migrate(conn: sqlite3.Connection) -> None:
        cursor = conn.execute('PRAGMA table_info(context_entries)')
        columns = [row[1] for row in cursor.fetchall()]
        if 'owner_id' not in columns:
            conn.execute(
                'ALTER TABLE context_entries ADD COLUMN owner_id TEXT NOT NULL '
                f"DEFAULT '{default_principal}'",
            )
            logger.info('Added owner_id column to context_entries (SQLite)')
        if 'visibility' not in columns:
            conn.execute(
                'ALTER TABLE context_entries ADD COLUMN visibility TEXT NOT NULL '
                "DEFAULT 'private' CHECK(visibility IN ('private', 'shared', 'public'))",
            )
            logger.info('Added visibility column to context_entries (SQLite)')
        conn.execute(_CREATE_GRANTS_TABLE_SQLITE)
        for statement in _INDEX_STATEMENTS:
            conn.execute(statement)

    await backend.execute_write(_migrate)
    logger.info('Applied access-control migration for SQLite')


async def _apply_postgresql(backend: StorageBackend, default_principal: str) -> None:
    """Apply the PostgreSQL access-control DDL (idempotent, advisory-locked)."""
    migration_timeout_s = settings.storage.postgresql_migration_timeout_s

    async def _migrate(conn: asyncpg.Connection) -> None:
        # Raise the transaction-scoped statement_timeout to the migration budget and
        # take the shared advisory lock under it, mirroring the version migration:
        # a wait on a multi-pod peer holding the lock is bounded by the migration
        # budget, not the pool's shorter command_timeout. SET LOCAL auto-reverts on
        # COMMIT/ROLLBACK.
        await begin_migration(conn, migration_timeout_s)
        await execute_migration_ddl(
            conn,
            'ALTER TABLE context_entries ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL '
            f"DEFAULT '{default_principal}'",
            migration_timeout_s,
        )
        await execute_migration_ddl(
            conn,
            'ALTER TABLE context_entries ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL '
            "DEFAULT 'private' CHECK(visibility IN ('private', 'shared', 'public'))",
            migration_timeout_s,
        )
        await execute_migration_ddl(conn, _CREATE_GRANTS_TABLE_POSTGRESQL, migration_timeout_s)
        for statement in _INDEX_STATEMENTS:
            await execute_migration_ddl(conn, statement, migration_timeout_s)

    await backend.execute_write(cast(Any, _migrate))
    logger.info('Applied access-control migration for PostgreSQL')
