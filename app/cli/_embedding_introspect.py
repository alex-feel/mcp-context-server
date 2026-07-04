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

# Re-exported from the shared migration-probe module so the CLI flows and the
# migration provisioning gates consult the SAME probe (sqlite_master on
# SQLite; to_regclass -- connection search_path -- on PostgreSQL, matching
# the bare-name reads it gates) and cannot drift apart.
from app.migrations._probes import embedding_metadata_table_exists


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

    comp = get_settings().compression
    if comp.enabled:
        from app.compression.provenance import read_compression_metadata

        meta = await read_compression_metadata(backend)
        if meta is not None:
            # The encode path (--embed-missing / --re-embed) builds the codebook
            # purely from the process COMPRESSION_* settings, never the sealed
            # row. A dim/seed/bits/variant/provider that disagrees with the
            # seed-locked compression_metadata would encode payloads against an
            # INCOMPATIBLE codebook (the read path concat then raises, or a later
            # server boot refuses to start). Guard the full sealed tuple, not
            # just dim -- the read path and the startup validator both compare
            # all of it.
            mismatches: list[str] = []
            if meta.dim != configured_dim:
                mismatches.append(f'EMBEDDING_DIM={configured_dim} but sealed dim={meta.dim}')
            if meta.seed != comp.seed:
                mismatches.append(f'COMPRESSION_SEED={comp.seed} but sealed seed={meta.seed}')
            if meta.bits != comp.bits:
                mismatches.append(f'COMPRESSION_BITS={comp.bits} but sealed bits={meta.bits}')
            if str(meta.variant) != str(comp.variant):
                mismatches.append(f'COMPRESSION_VARIANT={comp.variant} but sealed variant={meta.variant}')
            if str(meta.provider) != str(comp.provider):
                mismatches.append(f'COMPRESSION_PROVIDER={comp.provider} but sealed provider={meta.provider}')
            if mismatches:
                return (
                    'the compressed database was sealed with a different compression '
                    'configuration (seed-locked compression_metadata) than this process '
                    'is configured for. Embedding now would encode payloads against an '
                    'incompatible codebook; restore the COMPRESSION_* / EMBEDDING_DIM '
                    'env vars to the sealed values, or decompress and rebuild to change '
                    'them. Mismatches: ' + '; '.join(mismatches) + '.'
                )

            # The scalar tuple matching is NOT sufficient: the same (dim, seed)
            # can materialize a DIFFERENT numpy.linalg.qr rotation on a host with
            # a different BLAS/LAPACK build or CPU dispatch, so encoding here
            # would write payloads in the wrong rotation basis -- unreadable by
            # the original server host, whose own startup fingerprint check
            # still passes (its realized rotation matches the sealed digest).
            # Compare the REALIZED fingerprint like the server startup validator
            # and the --decompress guard do; a row that predates fingerprinting
            # (None) cannot be checked and proceeds (the decompress guard's
            # warn-only stance for the same state).
            if meta.codebook_fingerprint is not None:
                import asyncio

                from app.compression.factory import create_compression_provider

                provider = create_compression_provider()
                realized = await asyncio.to_thread(provider.codebook_fingerprint)
                if realized != meta.codebook_fingerprint:
                    return (
                        'the realized compression codebook fingerprint on THIS host does '
                        'not match the one recorded when the database was first compressed '
                        '(typically a different BLAS/LAPACK build or CPU materializing a '
                        'different rotation for the same dim and seed). Embedding now would '
                        'write payloads in the wrong rotation basis, silently corrupting '
                        'semantic search for the new entries. Run this command on a host '
                        'whose numerical libraries reproduce the original codebook (for '
                        'example the server host itself). Expected '
                        f'fingerprint={meta.codebook_fingerprint}, realized={realized}.'
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
