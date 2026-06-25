"""Read-only embedding-metadata introspection for the migration CLI.

Shared by the ``--embed-missing`` and ``--re-embed`` flows to probe the
existing embedding state of a database -- whether the ``embedding_metadata``
table exists, and the distinct model names and dimensions already recorded --
so each flow can validate its configuration before generating embeddings.

Backend-specific SQL is used because PostgreSQL ``UUID`` and SQLite ``TEXT``
bind differently and the table-presence probe differs per backend. The
helpers are intentionally public (no leading underscore) so both CLI modules
import them without crossing a module's private boundary.
"""

import sqlite3
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend


async def embedding_metadata_table_exists(backend: StorageBackend) -> bool:
    """Return True when the ``embedding_metadata`` table exists.

    Args:
        backend: Storage backend to query.

    Returns:
        True when the ``embedding_metadata`` table is present in the database.
    """
    if backend.backend_type == 'sqlite':

        def _check(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='embedding_metadata'",
            )
            return cursor.fetchone() is not None

        return await backend.execute_read(_check)

    async def _check_pg(conn: asyncpg.Connection) -> bool:
        result = await conn.fetchval(
            '''
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = current_schema()
                  AND table_name = 'embedding_metadata'
            )
            ''',
        )
        return bool(result)

    return await backend.execute_read(cast(Any, _check_pg))


async def dimension_conflict_error(
    backend: StorageBackend,
    *,
    configured_dim: int,
) -> str | None:
    """Return an actionable message if ``configured_dim`` conflicts with stored data.

    Single source of truth for the dimension-change guard shared by
    ``--embed-missing`` and ``--re-embed`` (keeping the two from diverging).
    Checks BOTH the per-row ``embedding_metadata`` dimensions AND, when
    compression is enabled, the seed-locked ``compression_metadata`` dimension
    -- the latter matters even when no embeddings exist yet (a freshly sealed
    compressed database has a ``compression_metadata`` row but an empty
    ``embedding_metadata`` table). A dimension change rewrites the
    vector-storage geometry (the fp32 vector column width is fixed at creation,
    and under compression the dimension is part of the seed-locked codebook),
    so it cannot be done in place.

    Args:
        backend: Storage backend to query.
        configured_dim: The process's configured ``EMBEDDING_DIM``.

    Returns:
        An operator-actionable message, or ``None`` when there is no conflict.
    """
    dims = await distinct_embedding_dimensions(backend)
    conflicting = sorted(d for d in dims if d != configured_dim)
    if conflicting:
        return (
            f'existing embeddings have dimension {conflicting} but '
            f'EMBEDDING_DIM={configured_dim}. Changing the embedding dimension '
            f'rewrites the vector-storage geometry and cannot be done in place; '
            f'set EMBEDDING_DIM={conflicting[0]}, or rebuild the database at the '
            f'new dimension (see '
            f'docs/migration-v2-to-v3.md#changing-the-embedding-model-or-dimensions).'
        )

    from app.settings import get_settings

    if get_settings().compression.enabled:
        from app.compression.provenance import read_compression_metadata

        meta = await read_compression_metadata(backend)
        if meta is not None and meta.dim != configured_dim:
            return (
                f'the compressed database was sealed at dimension {meta.dim} '
                f'(seed-locked compression_metadata) but EMBEDDING_DIM='
                f'{configured_dim}. Embedding at a different dimension would '
                f'corrupt the compressed codebook; set EMBEDDING_DIM={meta.dim}, '
                f'or decompress and rebuild to change it.'
            )
    return None


async def distinct_embedding_models(backend: StorageBackend) -> set[str]:
    """Return the set of distinct ``model_name`` values in embedding_metadata.

    Args:
        backend: Storage backend to query.

    Returns:
        The distinct ``model_name`` values; empty when no embeddings exist.
    """
    if backend.backend_type == 'sqlite':

        def _read_sqlite(conn: sqlite3.Connection) -> set[str]:
            cursor = conn.execute('SELECT DISTINCT model_name FROM embedding_metadata')
            return {str(r[0]) for r in cursor.fetchall()}

        return await backend.execute_read(_read_sqlite)

    async def _read_pg(conn: asyncpg.Connection) -> set[str]:
        rows = await conn.fetch('SELECT DISTINCT model_name FROM embedding_metadata')
        return {str(r['model_name']) for r in rows}

    return await backend.execute_read(cast(Any, _read_pg))


async def distinct_embedding_dimensions(backend: StorageBackend) -> set[int]:
    """Return the set of distinct ``dimensions`` values in embedding_metadata.

    Args:
        backend: Storage backend to query.

    Returns:
        The distinct ``dimensions`` values; empty when no embeddings exist.
    """
    if backend.backend_type == 'sqlite':

        def _read_sqlite(conn: sqlite3.Connection) -> set[int]:
            cursor = conn.execute('SELECT DISTINCT dimensions FROM embedding_metadata')
            return {int(r[0]) for r in cursor.fetchall()}

        return await backend.execute_read(_read_sqlite)

    async def _read_pg(conn: asyncpg.Connection) -> set[int]:
        rows = await conn.fetch('SELECT DISTINCT dimensions FROM embedding_metadata')
        return {int(r['dimensions']) for r in rows}

    return await backend.execute_read(cast(Any, _read_pg))


__all__ = [
    'dimension_conflict_error',
    'distinct_embedding_dimensions',
    'distinct_embedding_models',
    'embedding_metadata_table_exists',
]
