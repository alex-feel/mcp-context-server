"""``--compress``: plan the fp32-to-compressed migration and run it."""

import asyncio
import logging
import sqlite3
import sys
import time
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.cli._backend import make_backend
from app.cli._backend import shutdown_backend
from app.cli._database_url import mask_credentials
from app.cli.migrate_compression.compress_execution import execute_compress
from app.cli.migrate_compression.console import PROBE_BATCH_SIZE
from app.cli.migrate_compression.console import load_cli_settings
from app.cli.migrate_compression.console import print_plan
from app.cli.migrate_compression.console import print_warning
from app.cli.migrate_compression.storage import count_table
from app.cli.migrate_compression.storage import fp32_blob_to_list_sqlite
from app.cli.migrate_compression.storage import make_2d_array
from app.cli.migrate_compression.storage import table_exists
from app.compression.provenance import read_compression_metadata
from app.compression.types import CompressionMetadata
from app.errors import ConfigurationError
from app.settings import get_settings

logger = logging.getLogger(__name__)


def run_compress(source_url: str, *, dry_run: bool) -> int:
    """Compress fp32 embeddings into bit-packed payloads.

    Args:
        source_url: Database URL passed to ``--source-url``.
        dry_run: When True, run all checks and the probe batch but roll
            back instead of committing the encode pass.

    Returns:
        Process exit code (0 success or success-no-op, 1 invalid input,
        2 unrecoverable failure, 78 EX_CONFIG on an invalid environment
        configuration).
    """
    masked = mask_credentials(source_url)
    settings = load_cli_settings()
    if settings is None:
        return ConfigurationError.EXIT_CODE
    comp = settings.compression

    print_warning(
        source_url=masked,
        mode='--compress',
        mode_description='encode fp32 -> compressed',
        dry_run=dry_run,
    )

    if not comp.enabled:
        print(
            '[ERROR] ENABLE_EMBEDDING_COMPRESSION=true must be set for '
            '--compress. Export the env var (along with COMPRESSION_SEED, '
            'COMPRESSION_BITS, COMPRESSION_VARIANT) and re-run.',
            file=sys.stderr,
        )
        return 1

    # Guard the DESTRUCTIVE fp32->compressed conversion the same way the server
    # guards startup: a misaligned (dim, bits, variant) would let the per-row
    # encode + DROP TABLE succeed, then make every compressed search raise and
    # the server refuse to start. Fail BEFORE any DROP so fp32 data is never lost.
    from app.startup.compression_validator import compression_byte_alignment_error
    align_error = compression_byte_alignment_error(
        settings.embedding.dim, comp.bits, comp.variant,
    )
    if align_error is not None:
        print(f'[ERROR] {align_error}', file=sys.stderr)
        return 1

    try:
        return asyncio.run(_compress_async(source_url, dry_run=dry_run))
    except Exception as exc:
        logger.exception('compression failed: %s', exc)
        return 2


async def _compress_async(source_url: str, *, dry_run: bool) -> int:
    """Async body for :func:`run_compress`."""
    masked = mask_credentials(source_url)
    settings = get_settings()
    comp = settings.compression

    backend = make_backend(source_url)
    await backend.initialize()
    try:
        existing = await read_compression_metadata(backend)
        if existing is not None:
            print(
                f'[INFO] compression_metadata row already present '
                f'(bits={existing.bits} variant={existing.variant} '
                f'dim={existing.dim} seed={existing.seed}). '
                'Nothing to do.',
                file=sys.stderr,
            )
            return 0

        if not await table_exists(backend, 'vec_context_embeddings'):
            print(
                '[INFO] vec_context_embeddings not present; assuming '
                'compressed-only deployment. Aborting.',
                file=sys.stderr,
            )
            return 1

        # Capture the source row count and probe data BEFORE doing any
        # schema work: the compress execution path drops
        # ``vec_context_embeddings`` as part of its atomic transaction,
        # so this is the only safe window to read it.
        row_count = await count_table(backend, 'vec_context_embeddings')
        probe_rows = await _read_fp32_probe(backend, PROBE_BATCH_SIZE)

        from app.compression import create_compression_provider

        provider = create_compression_provider()
        if not probe_rows:
            estimated_rate = 0.0
            estimated_seconds = 0.0
        else:
            start = time.perf_counter()
            for _ctx_id, _chunk_idx, _start, _end, vec in probe_rows:
                provider.encode_sync(make_2d_array(vec))
            elapsed = max(time.perf_counter() - start, 1e-6)
            estimated_rate = len(probe_rows) / elapsed
            estimated_seconds = (
                row_count / estimated_rate if estimated_rate > 0 else 0.0
            )

        provenance = CompressionMetadata(
            provider=comp.provider,
            bits=comp.bits,
            variant=comp.variant,
            seed=comp.seed,
            dim=settings.embedding.dim,
            # Record the REALIZED rotation-matrix digest so the server's startup
            # validator can detect a cross-host BLAS/LAPACK/CPU QR divergence (the
            # same (dim, seed) materializing a different rotation) and fail loudly
            # rather than silently corrupting every decode/search of this DB later.
            codebook_fingerprint=provider.codebook_fingerprint(),
        )

        print_plan(
            source_url=masked,
            from_table='vec_context_embeddings',
            from_rows=row_count,
            to_table='vec_context_embeddings_compressed',
            provenance=provenance,
            estimated_seconds=estimated_seconds,
            estimated_rate=estimated_rate,
        )

        if dry_run:
            print(
                '[DRY-RUN] Plan above would be applied. Re-run without '
                '--dry-run to execute.',
                file=sys.stderr,
            )
            return 0

        await execute_compress(
            backend=backend,
            provider=provider,
            provenance=provenance,
        )

        print(
            'Compression complete. Restart the server with '
            'ENABLE_EMBEDDING_COMPRESSION=true to use the compressed '
            'read path.',
            file=sys.stderr,
        )
        return 0
    finally:
        await shutdown_backend(backend)


async def _read_fp32_probe(
    backend: StorageBackend, n: int,
) -> list[tuple[str, int, int, int, list[float]]]:
    """Read up to ``n`` fp32 rows for the probe-batch latency estimate."""
    limit = int(n)
    if backend.backend_type == 'sqlite':
        dim = get_settings().embedding.dim

        def _read(
            conn: sqlite3.Connection,
        ) -> list[tuple[str, int, int, int, list[float]]]:
            cur = conn.execute(
                'SELECT ec.context_id, ec.id, ec.start_index, ec.end_index, '
                'v.embedding FROM embedding_chunks ec '
                'JOIN vec_context_embeddings v ON v.rowid = ec.vec_rowid '
                'LIMIT ?',
                (limit,),
            )
            rows: list[tuple[str, int, int, int, list[float]]] = []
            for ctx_id, chunk_id, start_index, end_index, blob in cur.fetchall():
                rows.append((
                    str(ctx_id),
                    int(chunk_id),
                    int(start_index),
                    int(end_index),
                    fp32_blob_to_list_sqlite(bytes(blob), dim),
                ))
            return rows

        return await backend.execute_read(_read)

    async def _read_pg(
        conn: asyncpg.Connection,
    ) -> list[tuple[str, int, int, int, list[float]]]:
        rows = await conn.fetch(
            'SELECT context_id, id, start_index, end_index, embedding '
            'FROM vec_context_embeddings LIMIT $1',
            limit,
        )
        out: list[tuple[str, int, int, int, list[float]]] = []
        for r in rows:
            embedding = r['embedding']
            out.append((
                str(r['context_id']),
                int(r['id']),
                int(r['start_index']),
                int(r['end_index']),
                [float(x) for x in embedding],
            ))
        return out

    return await backend.execute_read(cast(Any, _read_pg))
