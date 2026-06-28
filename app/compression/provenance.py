"""Read and write the singleton ``compression_metadata`` provenance row.

The provenance row anchors the compression subsystem: it stores the
provider, bit width, variant, rotation seed, and embedding dimension that
the server used at first startup. The startup validator, the read path,
and the migration CLI all consult this row to derive their behavior.

Centralizing the SQL here avoids duplicating the read/write logic across
``app.startup.compression_validator``, ``app.repositories.embedding_repository``,
and ``app.cli.migrate_compression``.
"""

import sqlite3
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.compression.types import CompressionMetadata


async def read_compression_metadata(
    backend: StorageBackend,
) -> CompressionMetadata | None:
    """Read the singleton provenance row.

    Returns ``None`` when the ``compression_metadata`` table itself does
    not exist (pre-bootstrap state) AND when the table exists but is
    empty. Callers that need to distinguish the two states must check
    table presence separately.

    Args:
        backend: Storage backend instance.

    Returns:
        :class:`CompressionMetadata` when a row exists, ``None`` when the
        table is absent or empty.
    """
    if backend.backend_type == 'sqlite':

        def _read_sqlite(
            conn: sqlite3.Connection,
        ) -> tuple[str, int, str, int, int, str | None] | None:
            try:
                cursor = conn.execute(
                    'SELECT provider, bits, variant, seed, dim, codebook_fingerprint '
                    'FROM compression_metadata WHERE id = 1',
                )
            except sqlite3.OperationalError as exc:
                # Pre-bootstrap state: the compression_metadata table has
                # not been created yet. Treat as "no row". Errors that
                # reference a different missing table indicate database
                # corruption and MUST propagate so the caller can fail
                # loudly instead of silently masking the defect. Mirrors
                # the type-safe asyncpg.UndefinedTableError precision in
                # the PostgreSQL branch below.
                if 'no such table: compression_metadata' in str(exc).lower():
                    return None
                raise
            row = cursor.fetchone()
            if row is None:
                return None
            return (row[0], int(row[1]), row[2], int(row[3]), int(row[4]), row[5])

        row = await backend.execute_read(_read_sqlite)
    else:

        async def _read_pg(
            conn: asyncpg.Connection,
        ) -> tuple[str, int, str, int, int, str | None] | None:
            try:
                record = await conn.fetchrow(
                    'SELECT provider, bits, variant, seed, dim, codebook_fingerprint '
                    'FROM compression_metadata WHERE id = 1',
                )
            except asyncpg.UndefinedTableError:
                return None
            if record is None:
                return None
            return (
                record['provider'],
                int(record['bits']),
                record['variant'],
                int(record['seed']),
                int(record['dim']),
                record['codebook_fingerprint'],
            )

        row = await backend.execute_read(cast(Any, _read_pg))

    if row is None:
        return None
    # CompressionMetadata's Pydantic v2 validators enforce ranges
    # (bits in [2, 4], dim > 0, seed >= 0) and Literal membership for
    # provider and variant. A row that violates them indicates manual SQL
    # tampering or corruption -- model_validate raises loudly.
    provider, bits, variant, seed, dim, codebook_fingerprint = row
    return CompressionMetadata.model_validate({
        'provider': provider,
        'bits': bits,
        'variant': variant,
        'seed': seed,
        'dim': dim,
        'codebook_fingerprint': codebook_fingerprint,
    })


async def insert_compression_metadata(
    backend: StorageBackend,
    meta: CompressionMetadata,
) -> None:
    """Insert the singleton provenance row at bootstrap.

    Args:
        backend: Storage backend instance.
        meta: Provenance to persist.
    """
    if backend.backend_type == 'sqlite':

        def _ins_sqlite(conn: sqlite3.Connection) -> None:
            conn.execute(
                'INSERT INTO compression_metadata '
                '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
                'VALUES (1, ?, ?, ?, ?, ?, ?)',
                (meta.provider, meta.bits, meta.variant, meta.seed, meta.dim, meta.codebook_fingerprint),
            )

        await backend.execute_write(_ins_sqlite)
    else:

        async def _ins_pg(conn: asyncpg.Connection) -> None:
            await conn.execute(
                'INSERT INTO compression_metadata '
                '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
                'VALUES (1, $1, $2, $3, $4, $5, $6)',
                meta.provider,
                meta.bits,
                meta.variant,
                meta.seed,
                meta.dim,
                meta.codebook_fingerprint,
            )

        await backend.execute_write(cast(Any, _ins_pg))


async def delete_compression_metadata(backend: StorageBackend) -> None:
    """Remove the singleton provenance row.

    Used by the migration CLI when reversing a compressed deployment back
    to fp32 storage.

    Args:
        backend: Storage backend instance.
    """
    if backend.backend_type == 'sqlite':

        def _del_sqlite(conn: sqlite3.Connection) -> None:
            conn.execute('DELETE FROM compression_metadata WHERE id = 1')

        await backend.execute_write(_del_sqlite)
    else:

        async def _del_pg(conn: asyncpg.Connection) -> None:
            await conn.execute('DELETE FROM compression_metadata WHERE id = 1')

        await backend.execute_write(cast(Any, _del_pg))


__all__ = [
    'delete_compression_metadata',
    'insert_compression_metadata',
    'read_compression_metadata',
]
