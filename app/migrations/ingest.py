"""Ingest index migration for mcp-context-server.

Creates the partial index that serves the reads of ingest-managed entries, rendered by
:func:`app.ingest_sql.ingest_index_ddl` from the same fragments the reads use, so a query
expression always matches the index expression. It runs at every start, right after the
access-control migration: the SQLite key names ``content_hash``, ``owner_id`` and
``visibility``, which the column migrations add on an in-place upgrade, so no base schema
declares the index. The index is partial, holding only entries whose ``ingest`` block
carries a source id, so it stays empty until such an entry is stored.

On PostgreSQL the migration also keeps extended statistics on the thread and the block's
scope. The reads require a member's scope to equal its own thread while they bind that
thread, so the planner rewrites the comparison to one of the scope expression with the
bound thread. Without statistics on that expression it assumes a fraction of a percent of
the thread's rows qualify when nearly all of a corpus thread does, and plans an owner page
as a fetch and sort of the members instead of an ordered walk of the index. The statistics
give it the expression's distribution and its pairing with the thread. Creating them
samples the table at once with ``ANALYZE``, so they describe the table before the first
request; autovacuum's analyze keeps them current afterwards.

SQLite creates the index with ``IF NOT EXISTS`` on the write queue. PostgreSQL first looks
for the index and the statistics, by name, on the ``context_entries`` table the bare name
resolves to through ``search_path``, the table and therefore the schema the DDL builds
them in: a start that finds both takes no lock and runs no DDL, and one that does not
builds what is missing under the migration budget and the shared advisory lock, where a
concurrent start's ``IF NOT EXISTS`` is a no-op. A bare ``to_regclass`` of the index name
would instead also find an index of that name in a later ``search_path`` schema, such as
one a deployment left in ``public`` before it configured another schema, and would skip
the build; matching the table the same way keeps a statistics object of that name on
another schema's table from standing in for this one.
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
from app.ingest_sql import scope_sql
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

INGEST_STATISTICS_NAME = 'stat_context_ingest_scope'

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

# True when a statistics object of the given name describes the context_entries table
# the bare name resolves to, the table the CREATE STATISTICS statement names.
_STATISTICS_ON_RESOLVED_TABLE_SQL = '''
    SELECT EXISTS (
        SELECT 1
        FROM pg_statistic_ext
        WHERE stxrelid = to_regclass('context_entries')
          AND stxname = $1
    )
'''

# Samples the table, which fills a statistics object created in the same transaction.
_ANALYZE_SQL = 'ANALYZE context_entries'


def ingest_statistics_ddl() -> str:
    """Render the PostgreSQL extended statistics on the thread and the scope of the ``ingest`` block.

    The statistics expression is :func:`app.ingest_sql.scope_sql` itself, the expression the
    reads compare with the thread, so the planner matches the two. The object keeps the most
    common ``(thread, scope)`` pairs and the dependency between them, and, because one of its
    items is an expression, the expression's own distribution. The statement is idempotent
    (``IF NOT EXISTS``).

    Returns:
        The ``CREATE STATISTICS`` statement.
    """
    return (
        f'CREATE STATISTICS IF NOT EXISTS {INGEST_STATISTICS_NAME} (mcv, dependencies) '
        f"ON thread_id, ({scope_sql('postgresql')}) FROM context_entries"
    )


async def _postgresql_objects_present(backend: StorageBackend) -> tuple[bool, bool]:
    """Report whether the ingest index and the ingest statistics exist on the table ``search_path`` resolves.

    Args:
        backend: The PostgreSQL backend the migration runs on.

    Returns:
        Whether the index exists on that table, and whether the statistics object does.
    """

    async def _probe(conn: asyncpg.Connection) -> tuple[bool, bool]:
        index_present = bool(await conn.fetchval(_INDEX_ON_RESOLVED_TABLE_SQL, INGEST_INDEX_NAME))
        statistics_present = bool(await conn.fetchval(_STATISTICS_ON_RESOLVED_TABLE_SQL, INGEST_STATISTICS_NAME))
        return index_present, statistics_present

    return await backend.execute_read(cast(Any, _probe), access_scope=MAINTENANCE)


async def apply_ingest_index_migration(backend: StorageBackend) -> None:
    """Create the partial index serving the reads of ingest-managed entries, and on PostgreSQL its statistics, when missing.

    Idempotent on both backends: SQLite's statement is ``IF NOT EXISTS``, and PostgreSQL
    skips the lock and every statement outright when its probes find both the index and
    the statistics.

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

    index_present, statistics_present = await _postgresql_objects_present(backend)
    if index_present and statistics_present:
        logger.debug('Ingest index %s and statistics %s present, skipping', INGEST_INDEX_NAME, INGEST_STATISTICS_NAME)
        return

    created: list[str] = []
    statements: list[str] = []
    if not index_present:
        created.append(f'index {INGEST_INDEX_NAME}')
        statements.append(ingest_index_ddl('postgresql'))
    if not statistics_present:
        created.append(f'statistics {INGEST_STATISTICS_NAME}')
        statements.extend([ingest_statistics_ddl(), _ANALYZE_SQL])

    migration_timeout_s = settings.storage.postgresql_migration_timeout_s

    async def _migrate_postgresql(conn: asyncpg.Connection) -> None:
        # The index build scans the whole table and the sample reads blocks across it, so
        # the shared advisory lock is taken and both run under the migration budget rather
        # than the pool's shorter command timeout.
        await begin_migration(conn, migration_timeout_s)
        for migration_statement in statements:
            await execute_migration_ddl(conn, migration_statement, migration_timeout_s)

    await backend.execute_write(cast(Any, _migrate_postgresql), access_scope=MAINTENANCE)
    logger.info('Applied the ingest index migration for PostgreSQL: created %s', ', '.join(created))
