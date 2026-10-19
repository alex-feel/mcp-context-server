"""Ingest index migration for mcp-context-server.

Creates the partial index that serves the reads of ingest-managed entries, rendered by
:func:`app.ingest_sql.ingest_index_ddl` from the same fragments the reads use, so a query
expression always matches the index expression. It runs at every start, right after the
access-control migration: the SQLite key names ``content_hash``, ``owner_id`` and
``visibility``, which the column migrations add on an in-place upgrade, so no base schema
declares the index. The index is partial, holding only entries whose ``ingest`` block
carries a source id, so it stays empty until such an entry is stored.

SQLite creates the index with ``IF NOT EXISTS`` on the write queue. PostgreSQL first looks
for an index of that name on the ``context_entries`` table the bare name resolves to
through ``search_path``, the table and therefore the schema the DDL builds it in: a start
that finds it takes no lock and runs no DDL, and one that does not builds it under the
migration budget and the shared advisory lock, where a concurrent start's ``IF NOT EXISTS``
is a no-op. A bare ``to_regclass`` of the index name would instead also find an index of
that name in a later ``search_path`` schema, such as one a deployment left in ``public``
before it configured another schema, and would skip the build.
"""

import logging
import sqlite3
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.ingest_sql import INGEST_INDEX_NAME
from app.ingest_sql import ingest_index_ddl
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# True when an index of the given name belongs to the context_entries table the bare
# name resolves to, which is the table, and so the schema, the CREATE INDEX statement
# builds the index on.
_INDEX_ON_RESOLVED_TABLE_SQL = '''
    SELECT EXISTS (
        SELECT 1
        FROM pg_index
        JOIN pg_class AS index_relation ON index_relation.oid = pg_index.indexrelid
        WHERE pg_index.indrelid = to_regclass('context_entries')
          AND index_relation.relname = $1
    )
'''


async def _postgresql_index_exists(backend: StorageBackend) -> bool:
    """Return True when the ingest index exists on the ``context_entries`` table ``search_path`` resolves.

    Args:
        backend: The PostgreSQL backend the migration runs on.

    Returns:
        True when the index exists on that table.
    """

    async def _probe(conn: asyncpg.Connection) -> bool:
        return bool(await conn.fetchval(_INDEX_ON_RESOLVED_TABLE_SQL, INGEST_INDEX_NAME))

    return await backend.execute_read(cast(Any, _probe), access_scope=MAINTENANCE)


async def apply_ingest_index_migration(backend: StorageBackend) -> None:
    """Create the partial index serving the reads of ingest-managed entries, when it is missing.

    Idempotent on both backends: SQLite's statement is ``IF NOT EXISTS``, and PostgreSQL
    skips the lock and the DDL outright when its probe finds the index.

    Args:
        backend: The backend to migrate: on PostgreSQL the maintenance backend, logged in
            as the table owner.
    """
    if backend.backend_type == 'sqlite':
        statement = ingest_index_ddl('sqlite')

        def _migrate_sqlite(conn: sqlite3.Connection) -> None:
            conn.execute(statement)

        await backend.execute_write(_migrate_sqlite, access_scope=MAINTENANCE)
        return

    if await _postgresql_index_exists(backend):
        logger.debug('Ingest index %s present, skipping', INGEST_INDEX_NAME)
        return

    migration_timeout_s = settings.storage.postgresql_migration_timeout_s

    async def _migrate_postgresql(conn: asyncpg.Connection) -> None:
        # The index build scans the whole table, so the shared advisory lock is taken and
        # the build runs under the migration budget rather than the pool's shorter
        # command timeout.
        await begin_migration(conn, migration_timeout_s)
        await execute_migration_ddl(conn, ingest_index_ddl('postgresql'), migration_timeout_s)

    await backend.execute_write(cast(Any, _migrate_postgresql), access_scope=MAINTENANCE)
    logger.info('Applied the ingest index migration for PostgreSQL (%s)', INGEST_INDEX_NAME)
