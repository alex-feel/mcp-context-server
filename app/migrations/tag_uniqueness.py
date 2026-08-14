"""Tag uniqueness migration for mcp-context-server.

Repairs duplicate ``tags`` rows and installs the unique index that prevents new ones,
so a database created before the write path deduplicated gains both on an in-place
upgrade. Fresh databases already carry the index from the base schema and hold no
duplicates, so the migration is a no-op there.

The two halves are inseparable and ordered: the index cannot be created while a
duplicate pair exists, and the delete is the ONLY thing that repairs entries already
stored with repeated labels -- a write-path fix reaches new writes only, while every
reader keeps returning the old duplicate rows verbatim and the tag statistics keep
counting them.

The index itself is the completion marker: once it exists, the database engine
enforces uniqueness and no duplicate can have survived its creation, so the migration
probes for it first and skips both statements when present. Without that probe the
repair delete would rescan the whole tags table on EVERY startup, and the delete is
deliberately a correlated EXISTS self-join (not ``NOT IN`` over a grouped subquery,
which PostgreSQL degrades to a per-row rescan of the materialized subplan once the
group list outgrows ``work_mem`` -- quadratic in the table size) so the one genuine
upgrade run stays an indexed semi-join on both backends.
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

# Keep the lowest surviving row per (context_entry_id, tag). Which row survives is
# immaterial -- the duplicates are byte-identical labels for the same entry -- but
# choosing deterministically keeps the outcome identical on both backends and across
# reruns. The correlated EXISTS probes through idx_tags_entry (present since the
# original schema), so both engines execute an indexed semi-join instead of the
# quadratic subplan rescan a NOT IN over the grouped id list degrades to on
# PostgreSQL when the list outgrows work_mem.
_DELETE_DUPLICATE_TAGS_SQL = '''
    DELETE FROM tags
    WHERE EXISTS (
        SELECT 1 FROM tags AS keeper
        WHERE keeper.context_entry_id = tags.context_entry_id
          AND keeper.tag = tags.tag
          AND keeper.id < tags.id
    )
'''

_CREATE_UNIQUE_TAG_INDEX_SQL = (
    'CREATE UNIQUE INDEX IF NOT EXISTS idx_tags_entry_tag ON tags(context_entry_id, tag)'
)


async def _unique_tag_index_exists(backend: StorageBackend) -> bool:
    """Return True when ``idx_tags_entry_tag`` already exists.

    The unique index is the migration's completion marker: it cannot coexist with a
    duplicate pair, so its presence proves the repair already ran (or the database was
    born with it in the base schema) and the whole migration can be skipped.

    Resolution mirrors :mod:`app.migrations._probes`: ``sqlite_master`` on SQLite,
    ``to_regclass`` (connection ``search_path``) on PostgreSQL, matching how the
    unqualified ``CREATE UNIQUE INDEX`` statement itself resolves.

    Args:
        backend: Storage backend instance.

    Returns:
        True when the index is reachable by an unqualified reference.
    """
    if backend.backend_type == 'sqlite':

        def _probe_sqlite(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_tags_entry_tag'",
            )
            return cursor.fetchone() is not None

        return await backend.execute_read(_probe_sqlite)

    async def _probe_pg(conn: asyncpg.Connection) -> bool:
        exists = await conn.fetchval("SELECT to_regclass('idx_tags_entry_tag') IS NOT NULL")
        return bool(exists)

    return await backend.execute_read(cast(Any, _probe_pg))


async def apply_tag_uniqueness_migration(backend: StorageBackend) -> None:
    """Remove duplicate tag rows, then enforce one row per (entry, tag).

    Idempotent on both backends: once the unique index exists the migration skips
    outright (the index proves no duplicate survived), so reruns never rescan the
    tags table. Concurrent first runs stay safe without the probe being atomic: the
    advisory lock (PostgreSQL) or the write queue (SQLite) serializes the write path,
    the repair delete matches nothing on a clean table, and the index creation is
    ``IF NOT EXISTS``.

    Args:
        backend: Storage backend instance.
    """
    if await _unique_tag_index_exists(backend):
        logger.debug('Tag uniqueness already enforced (idx_tags_entry_tag present), skipping')
        return

    if backend.backend_type == 'sqlite':

        def _migrate_sqlite(conn: sqlite3.Connection) -> None:
            removed = conn.execute(_DELETE_DUPLICATE_TAGS_SQL).rowcount
            conn.execute(_CREATE_UNIQUE_TAG_INDEX_SQL)
            if removed > 0:
                logger.info('Removed %d duplicate tag rows and enforced tag uniqueness (SQLite)', removed)
            else:
                logger.debug('Tag uniqueness already enforced, no duplicates found (SQLite)')

        await backend.execute_write(_migrate_sqlite)

    else:  # postgresql
        migration_timeout_s = settings.storage.postgresql_migration_timeout_s

        async def _migrate_postgresql(conn: asyncpg.Connection) -> None:
            # Raise the transaction-scoped statement_timeout to the migration budget and take
            # the shared advisory lock under it, so a wait on a multi-pod peer holding the lock
            # during a slow build is bounded by the migration budget rather than the pool's
            # shorter command_timeout (which would cancel the wait and crash startup).
            await begin_migration(conn, migration_timeout_s)
            await execute_migration_ddl(conn, _DELETE_DUPLICATE_TAGS_SQL, migration_timeout_s)
            await execute_migration_ddl(conn, _CREATE_UNIQUE_TAG_INDEX_SQL, migration_timeout_s)
            logger.info('Enforced tag uniqueness on tags(context_entry_id, tag) (PostgreSQL)')

        await backend.execute_write(cast(Any, _migrate_postgresql))
