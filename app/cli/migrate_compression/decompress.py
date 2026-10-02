"""``--decompress``: plan the compressed-to-fp32 reverse migration and run it."""

import asyncio
import logging
import sqlite3
import sys
import time
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends.postgresql_backend.session import apply_session_gucs
from app.backends.postgresql_backend.session import build_asyncpg_connect_kwargs
from app.cli._backend import make_backend
from app.cli._backend import shutdown_backend
from app.cli._database_url import mask_credentials
from app.cli._database_url import parse_backend_url
from app.cli.migrate_compression.console import PROBE_BATCH_SIZE
from app.cli.migrate_compression.console import load_cli_settings
from app.cli.migrate_compression.console import print_plan
from app.cli.migrate_compression.console import print_warning
from app.cli.migrate_compression.decompress_execution import execute_decompress
from app.cli.migrate_compression.storage import count_table
from app.cli.migrate_compression.storage import table_exists
from app.compression.base import CompressionProvider
from app.compression.provenance import read_compression_metadata
from app.compression.types import CompressionMetadata
from app.errors import ConfigurationError
from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT
from app.pgvector_limits import exceeds_pgvector_index_dim_limit
from app.settings import get_settings

logger = logging.getLogger(__name__)


def run_decompress(source_url: str, *, dry_run: bool) -> int:
    """Reverse the compression: decode compressed payloads back to fp32.

    Args:
        source_url: Database URL passed to ``--source-url``.
        dry_run: When True, run all checks and the probe batch but roll
            back instead of committing the decode pass.

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
        mode='--decompress',
        mode_description='decode compressed -> fp32 (lossy reconstruction)',
        dry_run=dry_run,
    )

    if comp.enabled:
        print(
            '[ERROR] ENABLE_EMBEDDING_COMPRESSION must be unset (or false) '
            'for --decompress. The compressed read path expects the active '
            'compression toggle to remain on while compressed data is in '
            'the database; for a clean decompress run, unset the env var '
            'before invoking the CLI.',
            file=sys.stderr,
        )
        return 1

    try:
        return asyncio.run(_decompress_async(source_url, dry_run=dry_run))
    except Exception as exc:
        logger.exception('decompression failed: %s', exc)
        return 2


async def _decompress_async(source_url: str, *, dry_run: bool) -> int:
    """Async body for :func:`run_decompress`."""
    masked = mask_credentials(source_url)
    backend_kind, address = parse_backend_url(source_url)
    # Decide pgvector provisioning BEFORE constructing/initializing the backend.
    # On PostgreSQL the backend's own settings-and-probe resolution returns True
    # for the decompress env shape (compression off, embedding generation on),
    # so initialize() would run CREATE EXTENSION vector and crash on a
    # pgvector-less host BEFORE the compressed-row count is taken -- wedging the
    # zero-data reverse path (decompress_empty.execute_decompress_empty), which
    # needs no vector type. Probe the compressed-row count up front on a throwaway plain
    # connection and provision pgvector only when the fp32 rebuild will actually
    # run. See _decompress_needs_vector.
    provision_vector: bool | None = None
    if backend_kind == 'postgresql':
        provision_vector = await _decompress_needs_vector(address)
    backend = make_backend(source_url, provision_vector=provision_vector)
    await backend.initialize()
    try:
        existing = await read_compression_metadata(backend)
        compressed_present = await table_exists(
            backend, 'vec_context_embeddings_compressed',
        )
        fp32_present = await table_exists(backend, 'vec_context_embeddings')

        if existing is None and fp32_present and (
            not compressed_present
            or await count_table(backend, 'vec_context_embeddings_compressed') == 0
        ):
            print(
                '[INFO] vec_context_embeddings_compressed empty and '
                'vec_context_embeddings already present. Nothing to do.',
                file=sys.stderr,
            )
            return 0

        if existing is None or not compressed_present:
            print(
                '[ERROR] compression_metadata row missing or compressed '
                'table absent; cannot decompress. The CLI requires both '
                'to be present to reconstruct fp32 vectors.',
                file=sys.stderr,
            )
            return 1

        row_count = await count_table(
            backend, 'vec_context_embeddings_compressed',
        )

        # fp32 rebuild capability pre-flight, BEFORE any DDL or data streaming:
        # pgvector cannot build an HNSW index over vector columns wider than
        # PGVECTOR_INDEX_DIM_LIMIT dimensions, and the reverse migration's
        # CREATE INDEX runs LAST -- after every compressed row has been
        # streamed and decoded -- so without this check the run wastes the
        # whole decode pass and dies at commit time with a raw pgvector error
        # (the transaction rolls back; data is safe but unexplained). A
        # compressed database legitimately holds such dimensions (BYTEA
        # payloads carry no cap), which is exactly why fp32 cannot host them.
        # Zero-row databases skip the gate: the zero-data reverse path drops
        # the empty compressed table and clears the provenance row WITHOUT
        # provisioning any fp32 infrastructure, so it stays available as the
        # documented escape hatch. SQLite is exempt (sqlite-vec has no
        # per-dimension index cap).
        if (
            backend.backend_type == 'postgresql'
            and row_count > 0
            and exceeds_pgvector_index_dim_limit(existing.dim)
        ):
            print(
                f'[ERROR] compression_metadata records dim={existing.dim}, above the '
                f'pgvector index limit of {PGVECTOR_INDEX_DIM_LIMIT} dimensions for '
                'fp32 vectors on PostgreSQL: --decompress would rebuild '
                'vec_context_embeddings and fail at the trailing HNSW CREATE INDEX '
                'after decoding every row. This database must stay compressed; to '
                'obtain an fp32 layout, re-embed at a dimension of at most '
                f'{PGVECTOR_INDEX_DIM_LIMIT} (--re-embed with a suitable '
                'EMBEDDING_MODEL/EMBEDDING_DIM) before decompressing.',
                file=sys.stderr,
            )
            return 1

        # Build a provider whose configuration mirrors the stored row so
        # decoding produces the original codebook geometry.
        provider = _provider_for(existing)

        # Guard against a cross-host codebook divergence BEFORE decoding any
        # payload or dropping the compressed source. The same (dim, seed) can
        # materialize a DIFFERENT numpy.linalg.qr rotation on a host with a
        # different BLAS/LAPACK build or CPU dispatch, so decoding here would
        # silently corrupt every reconstructed fp32 vector -- and decompress
        # then DROPs the compressed table (the only correctly-decodable copy) in
        # the same transaction. The server startup validator catches exactly this
        # divergence with exit 78; the standalone CLI must not bypass it. A row
        # that predates fingerprinting (None) can only be warned about. The gate
        # applies only when rows exist: with ZERO compressed rows nothing is
        # decoded and no corruption is possible, and the zero-data reverse path
        # exists precisely to unwedge deployments -- including a host whose
        # numerical libraries changed -- so blocking it on a fingerprint
        # mismatch would leave the operator with no working escape (the server
        # refuses to start on the same divergence).
        if row_count > 0 and existing.codebook_fingerprint is not None:
            realized_fingerprint = await asyncio.to_thread(provider.codebook_fingerprint)
            if realized_fingerprint != existing.codebook_fingerprint:
                print(
                    '[ERROR] compression codebook fingerprint mismatch: the realized '
                    'numpy.linalg.qr rotation for this (dim, seed) does NOT match the '
                    'one recorded when the data was compressed (typically a different '
                    'BLAS/LAPACK build or CPU). Decompressing here would silently '
                    'corrupt every reconstructed vector and then drop the only '
                    'correctly-decodable copy. Run --decompress on a host whose '
                    'numerical libraries reproduce the original codebook, or restore '
                    f'from backup. Expected fingerprint={existing.codebook_fingerprint}, '
                    f'realized={realized_fingerprint}.',
                    file=sys.stderr,
                )
                return 1
        elif row_count > 0 and existing.codebook_fingerprint is None:
            print(
                '[WARN] compression_metadata row predates codebook fingerprinting; a '
                'cross-host rotation-matrix divergence cannot be detected for this '
                'database. Verify --decompress runs on a host that reproduces the '
                'original codebook, or restore from backup if the decoded vectors '
                'look wrong.',
                file=sys.stderr,
            )

        probe_rows = await _read_compressed_probe(backend, PROBE_BATCH_SIZE)
        if not probe_rows:
            estimated_rate = 0.0
            estimated_seconds = 0.0
        else:
            start = time.perf_counter()
            for _ctx_id, _chunk_idx, _start, _end, payload in probe_rows:
                provider.decode_sync(payload)
            elapsed = max(time.perf_counter() - start, 1e-6)
            estimated_rate = len(probe_rows) / elapsed
            estimated_seconds = (
                row_count / estimated_rate if estimated_rate > 0 else 0.0
            )

        print_plan(
            source_url=masked,
            from_table='vec_context_embeddings_compressed',
            from_rows=row_count,
            to_table='vec_context_embeddings',
            provenance=existing,
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

        await execute_decompress(
            backend=backend,
            provider=provider,
            provenance=existing,
            row_count=row_count,
        )

        print(
            'Decompression complete. Reconstruction is LOSSY -- the '
            'decoded vectors approximate the original MSE component. '
            'Restart the server with ENABLE_EMBEDDING_COMPRESSION unset '
            '(or false) to use the fp32 read path.',
            file=sys.stderr,
        )
        return 0
    finally:
        await shutdown_backend(backend)


async def _decompress_needs_vector(address: str) -> bool:
    """Probe whether ``--decompress`` on a PostgreSQL database needs pgvector.

    Runs BEFORE ``backend.initialize()`` on a throwaway PLAIN asyncpg
    connection -- no pool, no vector-codec registration, and crucially no
    ``CREATE EXTENSION vector`` -- so it works on a host that lacks pgvector.
    The extension and the vector codec are needed only when the full reverse
    path rebuilds the fp32 ``vec_context_embeddings`` table and its HNSW index,
    which happens exactly when at least one compressed row will be decoded.
    With no compressed rows the zero-data reverse path
    (:func:`app.cli.migrate_compression.decompress_empty.execute_decompress_empty`)
    runs instead: it only DROPs the empty compressed table and DELETEs the
    provenance row, needing neither the vector type nor the extension.
    Provisioning pgvector in that case would force
    ``CREATE EXTENSION vector`` and crash ``initialize()`` -- before the
    compressed-row count is ever taken -- wedging the disable direction on a
    pgvector-less deployment (the compression validator seeds a provenance row
    with zero compressed rows, so a fresh compressed database that never wrote
    an embedding lands exactly here).

    The connection reuses :func:`build_asyncpg_connect_kwargs` and applies the
    session parameters through :func:`apply_session_gucs`, so the ``search_path``
    matches the bare-name DML the reverse path runs. ``EXISTS`` is used instead of
    ``COUNT(*)`` so the probe stops at the first row rather than scanning a large
    compressed table.

    Args:
        address: PostgreSQL connection string (the parsed ``--source-url``).

    Returns:
        True when a compressed row exists (the full fp32 rebuild needs
        pgvector); False when the compressed table is absent or empty (the
        zero-data reverse path needs no vector type).
    """
    connect_kwargs = build_asyncpg_connect_kwargs()
    conn = await asyncpg.connect(
        address,
        timeout=get_settings().storage.postgresql_connect_timeout_s,
        **connect_kwargs,
    )
    try:
        await apply_session_gucs(conn)
        reachable = bool(
            await conn.fetchval(
                "SELECT to_regclass('vec_context_embeddings_compressed') IS NOT NULL",
            ),
        )
        if not reachable:
            return False
        return bool(
            await conn.fetchval(
                'SELECT EXISTS(SELECT 1 FROM vec_context_embeddings_compressed)',
            ),
        )
    finally:
        await conn.close()


def _provider_for(meta: CompressionMetadata) -> CompressionProvider:
    """Build a TurboQuant provider whose configuration mirrors ``meta``.

    The CLI's ``--decompress`` flow must reconstruct the provider that
    ENCODED the data on its original write. The stored provenance row
    carries the exact ``(bits, variant, seed, dim)`` tuple that produced
    the codebook geometry; passing it as explicit constructor kwargs
    bypasses the settings singleton and avoids any process-env mutation.

    Args:
        meta: Singleton provenance row read from ``compression_metadata``.

    Returns:
        Initialized :class:`CompressionProvider` whose codebook geometry
        matches the encoded payloads.

    Raises:
        ValueError: If ``meta.provider`` is unsupported.
    """
    # Direct construction sidesteps the create_compression_provider()
    # factory because the factory reads from settings, and the CLI must
    # be able to decode payloads whose encoded configuration differs
    # from the current settings. The provenance row IS the source of
    # truth for decode.
    if meta.provider != 'turboquant':
        raise ValueError(
            f"Unsupported compression provider in provenance row: '{meta.provider}'. "
            f"Only 'turboquant' is supported.",
        )

    from app.compression.providers.turboquant import TurboQuantProvider

    return TurboQuantProvider(
        bits=meta.bits,
        variant=meta.variant,
        seed=meta.seed,
        dim=meta.dim,
    )


async def _read_compressed_probe(
    backend: StorageBackend, n: int,
) -> list[tuple[str, int, int, int, bytes]]:
    """Read up to ``n`` compressed rows for the probe-batch estimate."""
    limit = int(n)
    if backend.backend_type == 'sqlite':

        def _read(
            conn: sqlite3.Connection,
        ) -> list[tuple[str, int, int, int, bytes]]:
            cur = conn.execute(
                'SELECT context_id, chunk_index, start_index, end_index, '
                'payload FROM vec_context_embeddings_compressed LIMIT ?',
                (limit,),
            )
            return [
                (
                    str(r[0]),
                    int(r[1]),
                    int(r[2]),
                    int(r[3]),
                    bytes(r[4]),
                )
                for r in cur.fetchall()
            ]

        return await backend.execute_read(_read)

    async def _read_pg(
        conn: asyncpg.Connection,
    ) -> list[tuple[str, int, int, int, bytes]]:
        rows = await conn.fetch(
            'SELECT context_id, chunk_index, start_index, end_index, '
            'payload FROM vec_context_embeddings_compressed LIMIT $1',
            limit,
        )
        return [
            (
                str(r['context_id']),
                int(r['chunk_index']),
                int(r['start_index']),
                int(r['end_index']),
                bytes(r['payload']),
            )
            for r in rows
        ]

    return await backend.execute_read(cast(Any, _read_pg))
