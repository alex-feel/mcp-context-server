"""Table probes, fp32 BLOB codecs and the PostgreSQL statement budget for the compression CLI."""

import sqlite3
import struct
from typing import Any
from typing import cast

import asyncpg
import numpy as np
from numpy.typing import NDArray

from app.backends import StorageBackend
from app.migrations._pg_ddl import fetchval_migration
from app.migrations._pg_ddl import migration_statement_timeout_ms
from app.settings import get_settings

# Batch size for the streaming compression migration. Bounds peak fp32
# memory at O(batch * dim * 4) bytes (~40 MB at d=1024). A module-level
# constant rather than an env var: env vars configure the running server,
# while the CLI is a one-shot operator tool. The execution modules read it
# as ``storage.MIGRATION_BATCH_SIZE`` at call time, so rebinding this
# attribute changes the batch size of every streamed migration.
MIGRATION_BATCH_SIZE: int = 10_000


def fp32_blob_to_list_sqlite(blob: bytes, dim: int) -> list[float]:
    """Deserialize a SQLite vec0 BLOB payload into a Python float list."""
    if len(blob) != dim * 4:
        raise ValueError(
            f'unexpected fp32 BLOB size {len(blob)}; expected {dim * 4}',
        )
    return list(struct.unpack(f'<{dim}f', blob))


def list_to_fp32_blob_sqlite(values: list[float]) -> bytes:
    """Serialize a Python float list to a sqlite-vec compatible fp32 BLOB."""
    return struct.pack(f'<{len(values)}f', *values)


def make_2d_array(values: list[float]) -> NDArray[np.float32]:
    """Reshape a 1-D float list into a (1, d) NumPy float32 array."""
    return np.asarray([values], dtype=np.float32)


# Allow-list of table names used in unparameterized SQL constructions. Each
# entry corresponds to a table defined by the schema/migrations; the value
# is the literal SQL fragment safe to substitute. This keeps callers off
# the f-string-with-user-input path that linters flag.
_ALLOWED_TABLES: dict[str, str] = {
    'vec_context_embeddings': 'vec_context_embeddings',
    'vec_context_embeddings_compressed': 'vec_context_embeddings_compressed',
}


def _safe_table(name: str) -> str:
    """Return ``name`` if it is in the schema allow-list, else raise."""
    try:
        return _ALLOWED_TABLES[name]
    except KeyError as exc:
        raise ValueError(f'table name {name!r} not in allow-list') from exc


async def table_exists(
    backend: StorageBackend, table_name: str,
) -> bool:
    """Return True if ``table_name`` exists in the database.

    The PostgreSQL probe resolves the name with ``to_regclass`` -- through the
    connection ``search_path`` -- because every consumer of this probe then
    reads, counts, or drops the table with a BARE name resolved the same way.
    A probe pinned to the configured schema would miss a table living in
    ``public`` (the natural state of a deployment that predates a non-default
    ``POSTGRESQL_SCHEMA``) while the bare-name SQL keeps reaching it.

    Args:
        backend: Storage backend to query.
        table_name: Unqualified table name to probe.

    Returns:
        True when the table is reachable by an unqualified reference.
    """
    if backend.backend_type == 'sqlite':

        def _check(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cursor.fetchone() is not None

        return await backend.execute_read(_check)

    async def _check_pg(conn: asyncpg.Connection) -> bool:
        result = await conn.fetchval(
            'SELECT to_regclass($1) IS NOT NULL',
            table_name,
        )
        return bool(result)

    return await backend.execute_read(cast(Any, _check_pg))


async def count_table(backend: StorageBackend, table_name: str) -> int:
    """Return the row count for ``table_name``.

    On PostgreSQL this pre-lock estimate is a full-table ``COUNT(*)`` on the SAME
    (potentially large) vec table the under-lock recount reads, so it runs under the
    migration budget: a bare ``fetchval`` on the borrowed server pool would inherit the
    ~60s ``command_timeout`` / ~54s session ``statement_timeout`` and be cancelled here --
    before the budgeted transaction even begins -- on exactly the large corpus where
    raising ``POSTGRESQL_MIGRATION_TIMEOUT_S`` is meant to help. A short read transaction
    raises both the server-side (``SET LOCAL``) and client-side deadlines for the scan.

    Args:
        backend: The storage backend to count on.
        table_name: The table whose rows to count (validated via ``_safe_table``).

    Returns:
        The row count for ``table_name``.
    """
    safe = _safe_table(table_name)
    if backend.backend_type == 'sqlite':

        def _count(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(f'SELECT COUNT(*) FROM {safe}')
            return int(cursor.fetchone()[0])

        return await backend.execute_read(_count)

    migration_timeout_s = get_settings().storage.postgresql_migration_timeout_s

    async def _count_pg(conn: asyncpg.Connection) -> int:
        async with conn.transaction():
            await raise_pg_migration_budget(conn, migration_timeout_s)
            result = await fetchval_migration(
                conn, f'SELECT COUNT(*) FROM {safe}', migration_timeout_s,
            )
        return int(result or 0)

    return await backend.execute_read(cast(Any, _count_pg))


async def raise_pg_migration_budget(
    conn: 'asyncpg.Connection',
    migration_timeout_s: float,
) -> None:
    """Raise this transaction's PostgreSQL statement budget to the migration timeout.

    The compression CLI borrows the server ``PostgreSQLBackend`` pool (via
    ``create_backend`` in :func:`app.cli._backend.make_backend`), whose ``command_timeout``
    (``POSTGRESQL_COMMAND_TIMEOUT_S``, ~60s) asyncpg applies as the default client-side
    deadline and whose session ``statement_timeout`` is ~54s. The heavy vec-table DDL
    these paths run -- a full HNSW index build, a whole-table DROP -- can exceed that on
    a large corpus and be cancelled client-side as a non-retryable ``asyncio.TimeoutError``
    before the longer migration budget ever applies. Raise the server-side
    ``statement_timeout`` to ``POSTGRESQL_MIGRATION_TIMEOUT_S`` for THIS transaction only
    via ``SET LOCAL`` (PostgreSQL auto-reverts it on COMMIT/ROLLBACK, so no finally-restore
    that would raise 25P02 in an aborted transaction); the individual heavy statements
    additionally carry the matching client-side deadline via :func:`execute_migration_ddl`
    and :func:`fetch_migration`. Unlike :func:`begin_migration` this does NOT take the
    schema-init advisory lock: the CLI creates its tables inline and runs as a deliberate
    one-shot operator action, not concurrent multi-pod schema init. The millisecond
    conversion is floored at 1 via :func:`migration_statement_timeout_ms` -- a
    sub-millisecond budget would truncate to 0, which PostgreSQL treats as UNLIMITED.
    """
    await conn.execute(
        f'SET LOCAL statement_timeout = {migration_statement_timeout_ms(migration_timeout_s)}',
    )
