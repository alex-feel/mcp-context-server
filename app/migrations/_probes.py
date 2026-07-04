"""Shared table-presence probes for migration provisioning gates.

``embedding_metadata`` presence is the durable signal that embedding storage
was ever provisioned (the semantic-search migration creates it). The
generation-off provisioning gates in the semantic, chunking, and compression
migrations all consume this single probe so they cannot drift apart: the
runtime delete/update cleanup paths gate on the same table and then touch the
ACTIVE-format payload table (fp32 when compression is off, compressed when
on), so whichever migration owns the active format must keep provisioning it
for an infra-carrying database even while embedding generation is toggled
off.

Resolution matches the runtime cleanup gate exactly: ``sqlite_master`` on
SQLite, ``to_regclass`` (connection ``search_path``) on PostgreSQL -- the
reads and writes these gates protect use BARE table names resolved the same
way, so a probe pinned to the configured schema would disagree with them for
a table living in ``public``.
"""

import sqlite3
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend


async def embedding_metadata_table_exists(backend: StorageBackend) -> bool:
    """Return True when the ``embedding_metadata`` table exists.

    Args:
        backend: Storage backend instance.

    Returns:
        True when the table is reachable by an unqualified reference on the
        backend's connections.
    """
    if backend.backend_type == 'sqlite':

        def _probe_sqlite(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_metadata'",
            )
            return cursor.fetchone() is not None

        return await backend.execute_read(_probe_sqlite)

    async def _probe_pg(conn: asyncpg.Connection) -> bool:
        exists = await conn.fetchval(
            "SELECT to_regclass('embedding_metadata') IS NOT NULL",
        )
        return bool(exists)

    return await backend.execute_read(cast(Any, _probe_pg))
