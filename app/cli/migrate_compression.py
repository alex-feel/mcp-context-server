"""Compression CLI handler for ``mcp-context-server-migrate``.

Implements the ``--compress`` and ``--decompress`` flags. The handler is
kept in a separate module from ``app.cli.migrate`` to bound the size of
the v2->v3 migration script.

Operations:
    ``run_compress``:
        Read fp32 rows from ``vec_context_embeddings``, encode each with
        the active TurboQuant provider, INSERT into
        ``vec_context_embeddings_compressed``, INSERT the singleton
        provenance row, then DROP TABLE ``vec_context_embeddings``. The
        PostgreSQL HNSW index is dropped first when present.

    ``run_decompress``:
        Reverse path. Decode each compressed payload with the active
        provider, INSERT into a freshly-recreated fp32 vec table, DROP
        the compressed table, DELETE the singleton provenance row.
        Reconstruction is LOSSY (variant='mse' loses precision; variant=
        'ip' cannot recover the QJL residual).

Both operations:
    * Print a multi-line ASCII ``BACKUP REQUIRED`` warning to stderr.
    * Detect already-compressed (or already-decompressed) state and
      no-op idempotently.
    * Wrap the encode/decode + DDL in a single transaction; on failure
      the source table is left intact.
    * Honor ``--dry-run`` by rolling back the transaction after the
      probe-batch step.

The CLI is single-backend: source and destination are the same database.
For cross-backend migration users run the v2->v3 ``migrate.py`` first,
then ``--compress`` on the target.
"""

import asyncio
import contextlib
import logging
import sqlite3
import struct
import sys
import time
from typing import Any
from typing import cast

import asyncpg
import numpy as np
from numpy.typing import NDArray

from app.backends import StorageBackend
from app.backends import create_backend
from app.cli.migrate import mask_credentials
from app.cli.migrate import parse_backend_url
from app.compression.base import CompressionProvider
from app.compression.provenance import read_compression_metadata
from app.compression.types import CompressionMetadata
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations._pg_ddl import fetch_migration
from app.migrations._pg_ddl import fetchval_migration
from app.migrations._pg_ddl import migration_statement_timeout_ms
from app.settings import get_settings

logger = logging.getLogger(__name__)

# Default probe-batch size used to estimate total runtime in the dry-run
# preview. Small enough to keep dry-run cheap while large enough to
# smooth out single-call jitter.
PROBE_BATCH_SIZE = 128

# Batch size for the streaming compression migration. Bounds peak fp32
# memory at O(batch * dim * 4) bytes (~40 MB at d=1024). Held as a
# module-level constant rather than an env var because there is no
# tested operator scenario requiring a different value at this time.
# A future iteration that needs operator-tunable batch size should add a
# ``--batch-size N`` CLI flag rather than a global env var (env vars
# affect the running server; the CLI is a one-shot operator tool).
_MIGRATION_BATCH_SIZE: int = 10_000

# Warning text printed before any destructive operation. Plain ASCII so
# Windows cmd, PowerShell, and POSIX shells all render the same.
WARNING_BORDER = '=' * 60
WARNING_RULE = '-' * 60


def _print_warning(
    *,
    source_url: str,
    mode: str,
    mode_description: str,
    dry_run: bool,
) -> None:
    """Print the BACKUP REQUIRED warning block to stderr.

    Args:
        source_url: URL of the source database (already credential-masked).
        mode: Short mode flag (``--compress`` or ``--decompress``).
        mode_description: Human-readable description of the operation.
        dry_run: Append a DRY-RUN line when True.
    """
    lines = [
        WARNING_BORDER,
        'BACKUP REQUIRED -- THIS OPERATION IS NOT REVERSIBLE WITHOUT',
        'YOUR OWN BACKUP. The source vector table will be permanently',
        'modified by this run.',
        WARNING_RULE,
        f'Source: {source_url}',
        f'Mode:   {mode} ({mode_description})',
    ]
    if dry_run:
        lines.append('DRY-RUN: no writes will be issued')
    lines.append(WARNING_BORDER)
    print('\n'.join(lines), file=sys.stderr)


def _print_plan(
    *,
    source_url: str,
    from_table: str,
    from_rows: int,
    to_table: str,
    provenance: CompressionMetadata,
    estimated_seconds: float,
    estimated_rate: float,
) -> None:
    """Print the structured migration plan to stderr."""
    lines = [
        '',
        'Plan',
        f'  source:        {source_url}',
        f'  from_table:    {from_table} ({from_rows} rows)',
        f'  to_table:      {to_table}',
        (
            f'  provenance:    compression_metadata (singleton; '
            f'bits={provenance.bits} variant={provenance.variant} '
            f'seed={provenance.seed} dim={provenance.dim})'
        ),
        (
            f'  estimated_time: {estimated_seconds:.2f} seconds at '
            f'{estimated_rate:.0f} rows/s (extrapolated from a '
            f'{PROBE_BATCH_SIZE}-row probe)'
        ),
    ]
    print('\n'.join(lines), file=sys.stderr)


def _make_backend(source_url: str) -> StorageBackend:
    """Build a backend pointed at ``source_url``.

    Args:
        source_url: URL passed to ``--source-url``.

    Returns:
        Initialized :class:`StorageBackend` matching the URL scheme. The
        URL classification is performed by :func:`parse_backend_url`
        which raises ``ValueError`` for unrecognized schemes; the
        exception propagates to the caller.
    """
    backend_kind, address = parse_backend_url(source_url)
    if backend_kind == 'sqlite':
        return create_backend(backend_type='sqlite', db_path=address)
    return create_backend(
        backend_type='postgresql', connection_string=address,
    )


async def _shutdown(backend: StorageBackend) -> None:
    """Suppress timeouts during backend shutdown.

    A long-running CLI invocation may keep PG sessions open until the
    process exits; bound the wait so a stuck connection does not block
    teardown.
    """
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=10.0)


# ---------------------------------------------------------------------------
# fp32 row I/O helpers (SQLite + PostgreSQL).
# ---------------------------------------------------------------------------


def _fp32_blob_to_list_sqlite(blob: bytes, dim: int) -> list[float]:
    """Deserialize a SQLite vec0 BLOB payload into a Python float list."""
    if len(blob) != dim * 4:
        raise ValueError(
            f'unexpected fp32 BLOB size {len(blob)}; expected {dim * 4}',
        )
    return list(struct.unpack(f'<{dim}f', blob))


def _list_to_fp32_blob_sqlite(values: list[float]) -> bytes:
    """Serialize a Python float list to a sqlite-vec compatible fp32 BLOB."""
    return struct.pack(f'<{len(values)}f', *values)


# ---------------------------------------------------------------------------
# State queries (compression already applied / fp32 still present).
# ---------------------------------------------------------------------------

# Allow-list of table names used in unparameterized SQL constructions. Each
# entry corresponds to a table defined by the schema/migrations; the value
# is the literal SQL fragment safe to substitute. This keeps callers off
# the f-string-with-user-input path that linters flag.
_ALLOWED_TABLES: dict[str, str] = {
    'vec_context_embeddings': 'vec_context_embeddings',
    'vec_context_embeddings_compressed': 'vec_context_embeddings_compressed',
    'embedding_chunks': 'embedding_chunks',
    'compression_metadata': 'compression_metadata',
}


def _safe_table(name: str) -> str:
    """Return ``name`` if it is in the schema allow-list, else raise."""
    try:
        return _ALLOWED_TABLES[name]
    except KeyError as exc:
        raise ValueError(f'table name {name!r} not in allow-list') from exc


async def _table_exists(
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


async def _count_table(backend: StorageBackend, table_name: str) -> int:
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
            await _raise_pg_migration_budget(conn, migration_timeout_s)
            result = await fetchval_migration(
                conn, f'SELECT COUNT(*) FROM {safe}', migration_timeout_s,
            )
        return int(result or 0)

    return await backend.execute_read(cast(Any, _count_pg))


# ---------------------------------------------------------------------------
# Public CLI entry points.
# ---------------------------------------------------------------------------


def run_compress(source_url: str, *, dry_run: bool) -> int:
    """Compress fp32 embeddings into bit-packed payloads.

    Args:
        source_url: Database URL passed to ``--source-url``.
        dry_run: When True, run all checks and the probe batch but roll
            back instead of committing the encode pass.

    Returns:
        Process exit code (0 success or success-no-op, 1 invalid input,
        2 unrecoverable failure).
    """
    masked = mask_credentials(source_url)
    settings = get_settings()
    comp = settings.compression

    _print_warning(
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


def run_decompress(source_url: str, *, dry_run: bool) -> int:
    """Reverse the compression: decode compressed payloads back to fp32.

    Args:
        source_url: Database URL passed to ``--source-url``.
        dry_run: When True, run all checks and the probe batch but roll
            back instead of committing the decode pass.

    Returns:
        Process exit code (0 success or success-no-op, 1 invalid input,
        2 unrecoverable failure).
    """
    masked = mask_credentials(source_url)
    settings = get_settings()
    comp = settings.compression

    _print_warning(
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


# ---------------------------------------------------------------------------
# Async pipelines.
# ---------------------------------------------------------------------------


async def _compress_async(source_url: str, *, dry_run: bool) -> int:
    """Async body for :func:`run_compress`."""
    masked = mask_credentials(source_url)
    settings = get_settings()
    comp = settings.compression

    backend = _make_backend(source_url)
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

        if not await _table_exists(backend, 'vec_context_embeddings'):
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
        row_count = await _count_table(backend, 'vec_context_embeddings')
        probe_rows = await _read_fp32_probe(backend, PROBE_BATCH_SIZE)

        from app.compression import create_compression_provider

        provider = create_compression_provider()
        if not probe_rows:
            estimated_rate = 0.0
            estimated_seconds = 0.0
        else:
            start = time.perf_counter()
            for _ctx_id, _chunk_idx, _start, _end, vec in probe_rows:
                provider.encode_sync(_make_2d_array(vec))
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

        _print_plan(
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

        await _execute_compress(
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
        await _shutdown(backend)


async def _decompress_async(source_url: str, *, dry_run: bool) -> int:
    """Async body for :func:`run_decompress`."""
    masked = mask_credentials(source_url)
    backend = _make_backend(source_url)
    await backend.initialize()
    try:
        existing = await read_compression_metadata(backend)
        compressed_present = await _table_exists(
            backend, 'vec_context_embeddings_compressed',
        )
        fp32_present = await _table_exists(backend, 'vec_context_embeddings')

        if existing is None and fp32_present and (
            not compressed_present
            or await _count_table(backend, 'vec_context_embeddings_compressed') == 0
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

        # Build a provider whose configuration mirrors the stored row so
        # decoding produces the original codebook geometry.
        provider = _provider_for(existing)

        row_count = await _count_table(
            backend, 'vec_context_embeddings_compressed',
        )

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

        _print_plan(
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

        await _execute_decompress(
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
        await _shutdown(backend)


# ---------------------------------------------------------------------------
# Provider construction.
# ---------------------------------------------------------------------------


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


def _make_2d_array(values: list[float]) -> NDArray[np.float32]:
    """Reshape a 1-D float list into a (1, d) NumPy float32 array."""
    return np.asarray([values], dtype=np.float32)


# ---------------------------------------------------------------------------
# Row readers.
# ---------------------------------------------------------------------------


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
                    _fp32_blob_to_list_sqlite(bytes(blob), dim),
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


# ---------------------------------------------------------------------------
# Transactional execution.
# ---------------------------------------------------------------------------


async def _execute_compress(
    *,
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
) -> None:
    """Encode every fp32 row and write the compressed payload table.

    The whole operation runs inside a single
    :meth:`StorageBackend.begin_transaction` transaction on both backends,
    committing on success and rolling back on exception. Each branch locks
    the source table (``BEGIN IMMEDIATE`` / ``LOCK TABLE ... ACCESS
    EXCLUSIVE``) BEFORE streaming and recounts it INSIDE the transaction, so
    no planning-time row count is consumed here.
    """
    if backend.backend_type == 'sqlite':
        await _execute_compress_sqlite(
            backend, provider, provenance,
        )
        return
    await _execute_compress_postgresql(
        backend, provider, provenance,
    )


async def _execute_compress_sqlite(
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
) -> None:
    """SQLite branch of :func:`_execute_compress`.

    Streams fp32 rows in batches of :data:`_MIGRATION_BATCH_SIZE` from
    ``vec_context_embeddings`` (joined to ``embedding_chunks`` for the
    chunk metadata), encodes each batch in-process, and INSERTs each
    batch into ``vec_context_embeddings_compressed``. The whole streamed
    migration -- every batch INSERT plus the provenance row and the
    source table DROP -- runs inside ONE
    :meth:`StorageBackend.begin_transaction` so any failure rolls back
    all work and the source table remains intact.

    Bounded peak memory: ``O(_MIGRATION_BATCH_SIZE * dim * 4)`` bytes
    (~40 MB at ``dim == 1024``).

    Raises:
        ValueError: If the streamed read did not cover every fp32 row in
            ``vec_context_embeddings`` (an orphan lacking an embedding_chunks
            bridge), so the transaction aborts rather than dropping unread data.
    """
    settings = get_settings()
    dim = settings.embedding.dim
    batch_size = _MIGRATION_BATCH_SIZE

    async with backend.begin_transaction() as txn:
        conn = cast(sqlite3.Connection, txn.connection)

        # Create the compressed tables inline (we cannot use the
        # migration loader because it drops vec_context_embeddings up
        # front, which would invalidate the source rows being streamed
        # in this transaction).
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS vec_context_embeddings_compressed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_index INTEGER NOT NULL DEFAULT 0,
                end_index INTEGER NOT NULL DEFAULT 0,
                payload BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_vec_compressed_context
                ON vec_context_embeddings_compressed(context_id);
            CREATE TABLE IF NOT EXISTS compression_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                provider TEXT NOT NULL,
                bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),
                variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),
                seed INTEGER NOT NULL CHECK (seed >= 0),
                dim INTEGER NOT NULL CHECK (dim > 0),
                codebook_fingerprint TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            ''',
        )

        # Take the write lock BEFORE the first streamed read: sqlite3's
        # legacy transaction control would otherwise open the implicit
        # transaction only at the first INSERT, so a concurrent-process
        # writer could commit between the planning count / early batches and
        # that first INSERT (an equal-count replacement would slip past the
        # count guard below and the DROP would destroy never-encoded rows).
        # BEGIN IMMEDIATE freezes the database for the whole stream + swap;
        # executescript above ran first because it COMMITS any open
        # transaction (its IF NOT EXISTS DDL is idempotent autocommit).
        conn.execute('BEGIN IMMEDIATE')

        # Idempotency check: a populated provenance singleton means the
        # migration has already been applied. Re-running is a no-op.
        existing_provenance = conn.execute(
            'SELECT COUNT(*) FROM compression_metadata WHERE id = 1',
        ).fetchone()
        if existing_provenance and int(existing_provenance[0]) == 1:
            logger.info(
                'Compression migration already applied (provenance row '
                'present); --compress is a no-op.',
            )
            return

        # Stream fp32 rows in batches; encode + INSERT each batch.
        # ORDER BY (context_id, id) is stable so the batch composition is
        # deterministic across runs and per-batch memory is bounded.
        rows_processed = 0
        offset = 0
        # chunk_index is the per-context sequential position (0, 1, 2, ...),
        # matching the live compressed write path so the on-disk contract is
        # identical regardless of how a compressed row was produced. The
        # streamed read is ordered by (context_id, id), so a context's chunks
        # arrive contiguously and in order across batch boundaries.
        current_ctx: str | None = None
        chunk_seq = 0
        while True:
            cur = conn.execute(
                'SELECT ec.context_id, ec.id, ec.start_index, ec.end_index, '
                'v.embedding FROM embedding_chunks ec '
                'JOIN vec_context_embeddings v ON v.rowid = ec.vec_rowid '
                'ORDER BY ec.context_id, ec.id '
                'LIMIT ? OFFSET ?',
                (batch_size, offset),
            )
            batch = cur.fetchall()
            if not batch:
                break
            for ctx_id, _chunk_id, start_index, end_index, blob in batch:
                ctx_str = str(ctx_id)
                if ctx_str != current_ctx:
                    current_ctx = ctx_str
                    chunk_seq = 0
                else:
                    chunk_seq += 1
                vec_list = _fp32_blob_to_list_sqlite(bytes(blob), dim)
                payload = provider.encode_sync(_make_2d_array(vec_list))
                conn.execute(
                    'INSERT INTO vec_context_embeddings_compressed '
                    '(context_id, chunk_index, start_index, end_index, payload) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (
                        ctx_str, chunk_seq, int(start_index),
                        int(end_index), payload,
                    ),
                )
            rows_processed += len(batch)
            offset += len(batch)

        # Integrity guard: the streamed read joins embedding_chunks to
        # vec_context_embeddings, so a vec row with no embedding_chunks bridge
        # (an orphan from a corrupted source) would be silently skipped and then
        # destroyed by the DROP below. Recount INSIDE the write transaction
        # (the planning-time count ran outside it and may predate a
        # concurrent commit that BEGIN IMMEDIATE has since locked out) and
        # abort rather than drop unread fp32 data, mirroring the decompress
        # rebuild's mismatch guard.
        live_count = int(
            conn.execute(
                'SELECT COUNT(*) FROM vec_context_embeddings',
            ).fetchone()[0],
        )
        if rows_processed != live_count:
            raise ValueError(
                f'compress read {rows_processed} fp32 row(s) but '
                f'vec_context_embeddings holds {live_count} '
                f'(mismatch of {abs(live_count - rows_processed)} row(s); fewer reads '
                'mean orphaned vec rows lack an embedding_chunks bridge, more mean '
                'duplicate bridges). Aborting so no fp32 vector is dropped.',
            )

        # Provenance INSERT + source DROP last inside the same
        # transaction. The CHECK (id = 1) constraint guarantees this
        # INSERT is the only one ever applied.
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            'VALUES (1, ?, ?, ?, ?, ?, ?)',
            (
                provenance.provider,
                provenance.bits,
                provenance.variant,
                provenance.seed,
                provenance.dim,
                provenance.codebook_fingerprint,
            ),
        )
        conn.execute('DROP TABLE IF EXISTS vec_context_embeddings')

    logger.info(
        'Compressed %d fp32 row(s) into vec_context_embeddings_compressed; '
        'dropped legacy table.',
        rows_processed,
    )


async def _raise_pg_migration_budget(
    conn: 'asyncpg.Connection',
    migration_timeout_s: float,
) -> None:
    """Raise this transaction's PostgreSQL statement budget to the migration timeout.

    The compression CLI borrows the server ``PostgreSQLBackend`` pool (via
    ``create_backend`` in :func:`_make_backend`), whose ``command_timeout``
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


async def _execute_compress_postgresql(
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
) -> None:
    """PostgreSQL branch of :func:`_execute_compress`.

    Streams fp32 rows in batches of :data:`_MIGRATION_BATCH_SIZE` from
    ``vec_context_embeddings`` using LIMIT/OFFSET pagination ordered by
    ``(context_id, id)`` and encodes each batch in-process. Server-side
    cursors would keep a portal open against the source table that
    PostgreSQL would reject when the trailing ``DROP TABLE`` runs in the
    same transaction. The source table is locked ``ACCESS EXCLUSIVE``
    BEFORE the first batch: the stream runs under READ COMMITTED, so a
    concurrent writer replacing rows mid-stream with an equal-cardinality
    set would otherwise slip past a count-only guard and the trailing DROP
    would destroy never-encoded rows. Every batch INSERT plus the
    provenance row, the HNSW index DROP, and the source table DROP run
    inside ONE :meth:`StorageBackend.begin_transaction` so any failure
    rolls back all work and the source table remains intact.

    Bounded peak memory: ``O(_MIGRATION_BATCH_SIZE * dim * 4)`` bytes
    (~40 MB at ``dim == 1024``).

    Raises:
        ValueError: If the streamed read did not cover every fp32 row the
            in-transaction recount sees under the lock, so the transaction
            aborts rather than dropping unread data.
    """
    batch_size = _MIGRATION_BATCH_SIZE
    migration_timeout_s = get_settings().storage.postgresql_migration_timeout_s

    async with backend.begin_transaction() as txn:
        conn = cast('asyncpg.Connection', txn.connection)
        # Raise this transaction's statement budget so the heavy vec-table DDL below
        # (DROP TABLE / DROP INDEX) and the streamed batch reads run under the migration
        # budget, not the pool's ~60s command_timeout. See _raise_pg_migration_budget.
        await _raise_pg_migration_budget(conn, migration_timeout_s)

        # Bare table names rely on PostgreSQL's search_path resolution;
        # this matches the project-wide pattern in
        # ``embedding_repository.py`` and ``postgresql_schema.sql``.
        # Operators using a non-default schema configure ``search_path``
        # accordingly. Create the compressed tables inline (the
        # migration loader would drop the source vec table at this
        # point, invalidating the streamed read).
        await conn.execute(
            'CREATE TABLE IF NOT EXISTS '
            'vec_context_embeddings_compressed ('
            '  id BIGSERIAL PRIMARY KEY,'
            '  context_id UUID NOT NULL,'
            '  chunk_index INTEGER NOT NULL,'
            '  start_index INTEGER NOT NULL DEFAULT 0,'
            '  end_index INTEGER NOT NULL DEFAULT 0,'
            '  payload BYTEA NOT NULL,'
            '  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,'
            '  FOREIGN KEY (context_id) REFERENCES context_entries(id) '
            '  ON DELETE CASCADE'
            ')',
        )
        await conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_vec_compressed_context '
            'ON vec_context_embeddings_compressed(context_id)',
        )
        await conn.execute(
            'CREATE TABLE IF NOT EXISTS compression_metadata ('
            '  id INTEGER PRIMARY KEY CHECK (id = 1),'
            '  provider TEXT NOT NULL,'
            '  bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),'
            "  variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),"
            '  seed BIGINT NOT NULL CHECK (seed >= 0),'
            '  dim INTEGER NOT NULL CHECK (dim > 0),'
            '  codebook_fingerprint TEXT,'
            '  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP'
            ')',
        )

        # Idempotency check: skip work if provenance already populated.
        existing_count = await conn.fetchval(
            'SELECT COUNT(*) FROM compression_metadata WHERE id = 1',
        )
        if int(existing_count or 0) == 1:
            logger.info(
                'Compression migration already applied (provenance row '
                'present); --compress is a no-op.',
            )
            return

        # Freeze the source table BEFORE streaming: the batches read under
        # READ COMMITTED, so a concurrent writer replacing rows mid-stream
        # with an equal-cardinality set would defeat a count-only guard --
        # the stream would have encoded the OLD rows and the DROP below
        # would destroy the never-encoded replacements. ACCESS EXCLUSIVE
        # (waiting out in-flight writers, bounded by the migration budget)
        # makes every batch, the recount, and the DROP see one frozen state.
        await execute_migration_ddl(
            conn,
            'LOCK TABLE vec_context_embeddings IN ACCESS EXCLUSIVE MODE',
            migration_timeout_s,
        )
        # Route the recount through the migration budget (not a bare fetchval,
        # which inherits the pool's ~60s command_timeout): this COUNT(*) is the
        # first full-table scan under the lock, and on a large corpus it must
        # use POSTGRESQL_MIGRATION_TIMEOUT_S like the streamed reads that follow.
        live_count = int(
            await fetchval_migration(
                conn,
                'SELECT COUNT(*) FROM vec_context_embeddings',
                migration_timeout_s,
            )
            or 0,
        )

        # Stream fp32 rows in batches via LIMIT/OFFSET pagination.
        # Server-side cursors would keep a portal open against the
        # source table that PostgreSQL would reject when the trailing
        # DROP TABLE runs in the same transaction. The
        # ``ORDER BY (context_id, id)`` clause is backed by the
        # composite (context_id, id) ordering on the source so the
        # per-batch read cost stays bounded.
        rows_processed = 0
        offset = 0
        # chunk_index is the per-context sequential position (0, 1, 2, ...),
        # matching the live compressed write path. The streamed read is ordered
        # by (context_id, id), so a context's chunks arrive contiguously and in
        # order across batch boundaries.
        current_ctx: str | None = None
        chunk_seq = 0
        while True:
            batch = await fetch_migration(
                conn,
                'SELECT context_id, id, start_index, end_index, embedding '
                'FROM vec_context_embeddings '
                'ORDER BY context_id, id '
                'LIMIT $1 OFFSET $2',
                migration_timeout_s,
                batch_size, offset,
            )
            if not batch:
                break
            for r in batch:
                ctx_str = str(r['context_id'])
                if ctx_str != current_ctx:
                    current_ctx = ctx_str
                    chunk_seq = 0
                else:
                    chunk_seq += 1
                vec_list = [float(x) for x in r['embedding']]
                # Validate the stored fp32 dimension against the configured compression
                # dim BEFORE encoding. SQLite fails safe (_fp32_blob_to_list_sqlite raises
                # on a size mismatch); PostgreSQL would otherwise feed a wrong-width vector
                # to the encoder, silently corrupting the payload and then DROPping the
                # fp32 source in this same transaction. Abort instead so EMBEDDING_DIM can
                # be set to the stored dimension and the run retried -- no fp32 vector is
                # corrupted or dropped.
                if len(vec_list) != provenance.dim:
                    raise ValueError(
                        f'fp32 vector for context {ctx_str} has dimension {len(vec_list)} but '
                        f'compression is configured for dim {provenance.dim} (EMBEDDING_DIM); '
                        f'aborting so no fp32 vector is corrupted or dropped.',
                    )
                payload = provider.encode_sync(_make_2d_array(vec_list))
                await conn.execute(
                    'INSERT INTO vec_context_embeddings_compressed '
                    '(context_id, chunk_index, start_index, end_index, payload) '
                    'VALUES ($1, $2, $3, $4, $5)',
                    ctx_str, chunk_seq,
                    int(r['start_index']), int(r['end_index']), payload,
                )
            rows_processed += len(batch)
            offset += len(batch)

        # Integrity guard: abort rather than drop vec_context_embeddings if
        # the streamed read did not cover every fp32 row the in-transaction
        # recount saw under the ACCESS EXCLUSIVE lock. With the table frozen
        # since before the first batch, a mismatch signals OFFSET-pagination
        # drift or an anomaly; abort so no fp32 vector is lost.
        if rows_processed != live_count:
            raise ValueError(
                f'compress read {rows_processed} fp32 row(s) but '
                f'vec_context_embeddings holds {live_count}; aborting so no '
                'fp32 vector is dropped.',
            )

        # Provenance INSERT + HNSW index DROP + source DROP last inside
        # the same transaction.
        await conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            'VALUES (1, $1, $2, $3, $4, $5, $6)',
            provenance.provider,
            provenance.bits,
            provenance.variant,
            provenance.seed,
            provenance.dim,
            provenance.codebook_fingerprint,
        )
        await execute_migration_ddl(
            conn,
            'DROP INDEX IF EXISTS idx_vec_context_embeddings_hnsw',
            migration_timeout_s,
        )
        await execute_migration_ddl(
            conn,
            'DROP TABLE IF EXISTS vec_context_embeddings',
            migration_timeout_s,
        )

    logger.info(
        'Compressed %d fp32 row(s) into vec_context_embeddings_compressed; '
        'dropped legacy table and HNSW index.',
        rows_processed,
    )


async def _execute_decompress(
    *,
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
    row_count: int,
) -> None:
    """Decode every compressed row and write the fp32 vec table.

    The whole operation runs inside a single
    :meth:`StorageBackend.begin_transaction` transaction on both backends,
    committing on success and rolling back on exception. With ZERO compressed rows the
    zero-data reverse path runs instead: it drops the empty compressed table
    and clears the provenance row WITHOUT provisioning any fp32
    infrastructure, so the disable direction also works on deployments whose
    embedding storage (embedding_chunks, sqlite-vec, the pgvector extension)
    was never provisioned.
    """
    if row_count == 0:
        await _execute_decompress_empty(backend)
        return
    if backend.backend_type == 'sqlite':
        await _execute_decompress_sqlite(
            backend, provider, provenance, row_count,
        )
        return
    await _execute_decompress_postgresql(
        backend, provider, provenance, row_count,
    )


async def _execute_decompress_empty(backend: StorageBackend) -> None:
    """Zero-data reverse migration: drop the empty compressed table, clear the row.

    With no compressed rows there is nothing to decode, so the fp32
    infrastructure the full reverse path provisions (the vec0 virtual table /
    the pgvector-typed table plus the embedding_chunks rebuild) is not needed
    -- and may be genuinely absent: a deployment running with
    ``ENABLE_EMBEDDING_GENERATION=false`` never provisioned embedding_chunks,
    sqlite-vec, or the pgvector extension, yet could still carry the
    compression schema and a provenance row, wedging the disable direction
    behind a ``--decompress`` that crashed on the missing infrastructure.
    Dropping the empty table and deleting the provenance row is the complete
    reverse migration for that state; the next startup provisions fp32
    storage per ``ENABLE_EMBEDDING_GENERATION`` as usual. The caller has
    already verified both the provenance row and the compressed table exist.

    The planning-time row count ran OUTSIDE this transaction, so emptiness is
    re-checked INSIDE it before the drop: a compression-on server running
    concurrently could commit compressed rows in the gap, and dropping them
    here would destroy embeddings that were never decoded. On PostgreSQL the
    table is locked ACCESS EXCLUSIVE first (waiting out in-flight writers) so
    the recount is authoritative; on SQLite an explicit ``BEGIN IMMEDIATE``
    takes the write lock up front -- sqlite3's legacy transaction control
    opens the implicit transaction only before DML, so a bare SELECT pins no
    snapshot and a DROP with no transaction open would run in AUTOCOMMIT,
    committing past the guard -- making the recount authoritative and the
    drop + provenance delete one atomic commit; a concurrent write-lock
    holder surfaces as SQLITE_BUSY, bounded by the busy timeout.

    Raises:
        ValueError: When the recount inside the transaction finds compressed
            rows -- a concurrent writer stored embeddings after the zero-data
            path was planned. The transaction rolls back and nothing is
            dropped.
    """
    if backend.backend_type == 'sqlite':
        async with backend.begin_transaction() as txn:
            conn = cast(sqlite3.Connection, txn.connection)
            conn.execute('BEGIN IMMEDIATE')
            live_row = conn.execute(
                'SELECT COUNT(*) FROM vec_context_embeddings_compressed',
            ).fetchone()
            live_count = int(live_row[0])
            if live_count != 0:
                raise ValueError(
                    f'vec_context_embeddings_compressed holds {live_count} row(s) at '
                    'drop time but was empty when the zero-data reverse path was '
                    'planned -- a concurrent writer stored compressed embeddings '
                    'while --decompress was running. Aborting so no compressed '
                    'payload is dropped; stop the server writing to this database '
                    'and re-run --decompress.',
                )
            conn.execute('DROP TABLE IF EXISTS vec_context_embeddings_compressed')
            conn.execute('DELETE FROM compression_metadata WHERE id = 1')
    else:
        migration_timeout_s = get_settings().storage.postgresql_migration_timeout_s
        async with backend.begin_transaction() as txn:
            pg_conn = cast('asyncpg.Connection', txn.connection)
            await _raise_pg_migration_budget(pg_conn, migration_timeout_s)
            await execute_migration_ddl(
                pg_conn,
                'LOCK TABLE vec_context_embeddings_compressed '
                'IN ACCESS EXCLUSIVE MODE',
                migration_timeout_s,
            )
            # Route the under-lock recount through the migration budget (not a bare
            # fetchval, which inherits the pool's ~60s command_timeout), matching the
            # compress/decompress recounts: this COUNT(*) runs under the lock and must
            # use POSTGRESQL_MIGRATION_TIMEOUT_S on a large table.
            live_count = int(
                await fetchval_migration(
                    pg_conn,
                    'SELECT COUNT(*) FROM vec_context_embeddings_compressed',
                    migration_timeout_s,
                )
                or 0,
            )
            if live_count != 0:
                raise ValueError(
                    f'vec_context_embeddings_compressed holds {live_count} row(s) at '
                    'drop time but was empty when the zero-data reverse path was '
                    'planned -- a concurrent writer stored compressed embeddings '
                    'while --decompress was running. Aborting so no compressed '
                    'payload is dropped; stop the server writing to this database '
                    'and re-run --decompress.',
                )
            await execute_migration_ddl(
                pg_conn,
                'DROP TABLE IF EXISTS vec_context_embeddings_compressed',
                migration_timeout_s,
            )
            await pg_conn.execute('DELETE FROM compression_metadata WHERE id = 1')
    logger.info(
        'No compressed rows to decode; dropped the empty compressed table '
        'and cleared the provenance row.',
    )


async def _execute_decompress_sqlite(
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
    row_count: int,
) -> None:
    """SQLite branch of :func:`_execute_decompress`.

    Symmetric streamed reverse migration. Reads compressed rows in
    batches of :data:`_MIGRATION_BATCH_SIZE` from
    ``vec_context_embeddings_compressed`` (ordered by
    ``(context_id, chunk_index)``), decodes each batch in-process, and
    rebuilds BOTH the recreated ``vec_context_embeddings`` fp32 table and
    the ``embedding_chunks`` bridge directly from each compressed row's own
    ``context_id``/``start_index``/``end_index``. The rebuild does NOT depend
    on any pre-existing ``embedding_chunks`` row: a server-compressed-from-
    start database never wrote ``embedding_chunks`` (the live compressed
    write path stores only ``vec_context_embeddings_compressed`` with a
    per-context sequential ``chunk_index``), so the earlier "look up
    ``embedding_chunks`` by ``chunk_index`` and skip on miss" approach
    skipped every row and silently dropped all embeddings on a default
    deployment. ``BEGIN IMMEDIATE`` takes the write lock BEFORE the first
    streamed read, so a concurrent-process writer cannot commit (or replace
    rows with an equal-cardinality set) mid-stream. Provenance DELETE +
    compressed source DROP run last inside the same transaction.

    Raises:
        ValueError: When the recount inside the transaction disagrees with
            the number of rows streamed -- with the write lock held since
            before the first batch this signals pagination drift or an
            anomaly, and the drop below could destroy payloads that were
            never decoded. Mirrors the compress path's integrity guard; the
            transaction rolls back.
    """
    batch_size = _MIGRATION_BATCH_SIZE

    async with backend.begin_transaction() as txn:
        conn = cast(sqlite3.Connection, txn.connection)

        # Recreate the fp32 virtual table (idempotent).
        conn.executescript(
            f'CREATE VIRTUAL TABLE IF NOT EXISTS vec_context_embeddings '
            f'USING vec0(embedding float[{provenance.dim}])',
        )

        # Take the write lock BEFORE the first streamed read (see the
        # compress branch): without BEGIN IMMEDIATE the implicit transaction
        # opens only at the first DML, leaving early batch reads exposed to a
        # concurrent-process writer whose equal-count replacement would slip
        # past the recount guard below. executescript above ran first because
        # it COMMITS any open transaction (its IF NOT EXISTS DDL is
        # idempotent autocommit).
        conn.execute('BEGIN IMMEDIATE')

        # Idempotency check: if the compressed source table no longer
        # exists, the reverse migration has already run.
        check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='vec_context_embeddings_compressed'",
        ).fetchone()
        if not check:
            logger.info(
                'Reverse migration already applied (compressed source '
                'table absent); --decompress is a no-op.',
            )
            return

        # Clear the (possibly stale or empty) chunk->vec bridge and rebuild
        # it alongside vec_context_embeddings from the compressed rows
        # themselves, mirroring the context_id-based PostgreSQL reverse path.
        # A fresh running rowid links each decoded fp32 vector to its new
        # embedding_chunks row so the SQLite fp32 search join
        # (embedding_chunks.vec_rowid = vec_context_embeddings.rowid) resolves
        # after decompress regardless of how the compressed rows were produced.
        conn.execute('DELETE FROM embedding_chunks')
        next_rowid = int(
            conn.execute(
                'SELECT COALESCE(MAX(rowid), 0) + 1 FROM vec_context_embeddings',
            ).fetchone()[0],
        )

        # Stream compressed rows; decode and INSERT each batch.
        rows_processed = 0
        offset = 0
        while True:
            cur = conn.execute(
                'SELECT context_id, chunk_index, start_index, end_index, '
                'payload FROM vec_context_embeddings_compressed '
                'ORDER BY context_id, chunk_index '
                'LIMIT ? OFFSET ?',
                (batch_size, offset),
            )
            batch = cur.fetchall()
            if not batch:
                break
            for ctx_id, _chunk_idx, start_index, end_index, payload in batch:
                vec = provider.decode_sync(bytes(payload))[0]
                blob = _list_to_fp32_blob_sqlite(
                    [float(x) for x in vec.tolist()],
                )
                conn.execute(
                    'INSERT INTO vec_context_embeddings (rowid, embedding) '
                    'VALUES (?, ?)',
                    (next_rowid, blob),
                )
                conn.execute(
                    'INSERT INTO embedding_chunks '
                    '(context_id, vec_rowid, start_index, end_index) '
                    'VALUES (?, ?, ?, ?)',
                    (str(ctx_id), next_rowid, int(start_index), int(end_index)),
                )
                next_rowid += 1
            rows_processed += len(batch)
            offset += len(batch)

        # Integrity guard mirroring the compress path: with BEGIN IMMEDIATE
        # holding the write lock since before the first batch, a mismatch
        # against the in-transaction recount signals pagination drift or an
        # anomaly; abort rather than drop undecoded payloads.
        live_count = int(
            conn.execute(
                'SELECT COUNT(*) FROM vec_context_embeddings_compressed',
            ).fetchone()[0],
        )
        if live_count != rows_processed:
            raise ValueError(
                f'decompress streamed {rows_processed} compressed row(s) but '
                f'vec_context_embeddings_compressed holds {live_count} at drop '
                'time -- a concurrent writer modified the table while '
                '--decompress was running. Aborting so no undecoded payload is '
                'dropped; stop the server writing to this database and re-run '
                '--decompress.',
            )

        # Provenance DELETE + compressed source DROP last inside the
        # same transaction.
        conn.execute('DROP TABLE IF EXISTS vec_context_embeddings_compressed')
        conn.execute('DELETE FROM compression_metadata WHERE id = 1')

    logger.info(
        'Decompressed %d compressed row(s) into vec_context_embeddings; '
        'dropped compressed table and provenance row.',
        rows_processed or row_count,
    )


async def _execute_decompress_postgresql(
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
    row_count: int,
) -> None:
    """PostgreSQL branch of :func:`_execute_decompress`.

    Symmetric streamed reverse migration using LIMIT/OFFSET pagination
    ordered by ``(context_id, chunk_index)``. Server-side cursors would
    keep a portal open against the source table that PostgreSQL would
    reject when the trailing ``DROP TABLE`` runs in the same
    transaction. The source table is locked ``ACCESS EXCLUSIVE`` BEFORE
    the first batch: the stream runs under READ COMMITTED, so a concurrent
    same-chunk-count update committing mid-stream would otherwise replace
    rows the stream had already decoded without changing the cardinality a
    trailing recount could check. Recreates ``vec_context_embeddings`` +
    HNSW index inside the same transaction that streams decompressed rows
    in batches.

    Raises:
        ValueError: When the number of rows streamed disagrees with the
            recount taken under the lock before streaming -- with the table
            frozen this signals OFFSET-pagination drift or an anomaly, and
            the drop below could destroy payloads that were never decoded.
            Mirrors the compress path's integrity guard; the transaction
            rolls back.
    """
    batch_size = _MIGRATION_BATCH_SIZE
    migration_timeout_s = get_settings().storage.postgresql_migration_timeout_s

    async with backend.begin_transaction() as txn:
        conn = cast('asyncpg.Connection', txn.connection)
        # Raise this transaction's statement budget so the heavy vec-table DDL below
        # (the HNSW CREATE INDEX / DROP TABLE) and the streamed batch reads run under the
        # migration budget, not the pool's ~60s command_timeout. See
        # _raise_pg_migration_budget.
        await _raise_pg_migration_budget(conn, migration_timeout_s)

        # Bare table names rely on PostgreSQL's search_path resolution;
        # this matches the project-wide pattern in
        # ``embedding_repository.py`` and ``postgresql_schema.sql``.
        # Operators using a non-default schema configure ``search_path``
        # accordingly. The existence probe below uses ``to_regclass`` so
        # it resolves through the SAME search_path as the bare-name DML
        # and the trailing DROP TABLE.
        await conn.execute(
            f'CREATE TABLE IF NOT EXISTS vec_context_embeddings ('
            f'  id BIGSERIAL PRIMARY KEY,'
            f'  context_id UUID NOT NULL,'
            f'  embedding vector({provenance.dim}),'
            f'  start_index INTEGER NOT NULL DEFAULT 0,'
            f'  end_index INTEGER NOT NULL DEFAULT 0,'
            f'  FOREIGN KEY (context_id) REFERENCES context_entries(id) '
            f'  ON DELETE CASCADE'
            f')',
        )
        await conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_vec_embeddings_context_id '
            'ON vec_context_embeddings(context_id)',
        )

        # Idempotency check: the compressed source table must exist for
        # reverse migration to have work to do.
        source_reachable = await conn.fetchval(
            "SELECT to_regclass('vec_context_embeddings_compressed') IS NOT NULL",
        )
        if not source_reachable:
            logger.info(
                'Reverse migration already applied (compressed source '
                'table absent); --decompress is a no-op.',
            )
            return

        # Freeze the source table BEFORE streaming: the batches read under
        # READ COMMITTED, so a concurrent writer replacing an entry's N
        # compressed rows with N new ones mid-stream (a same-chunk-count
        # update, the most common write shape) would defeat a count-only
        # recount taken after the fact -- the stream would have decoded the
        # OLD rows, the cardinality would still match, and the DROP below
        # would destroy the never-decoded new payloads while fp32 kept the
        # stale ones. ACCESS EXCLUSIVE (waiting out in-flight writers,
        # bounded by the migration budget) makes every batch, the recount,
        # and the DROP see one frozen state.
        await execute_migration_ddl(
            conn,
            'LOCK TABLE vec_context_embeddings_compressed '
            'IN ACCESS EXCLUSIVE MODE',
            migration_timeout_s,
        )
        # Route the recount through the migration budget (see the compress
        # path): a bare fetchval would cap this first under-lock full-table
        # scan at the pool command_timeout instead of the migration budget.
        live_count = int(
            await fetchval_migration(
                conn,
                'SELECT COUNT(*) FROM vec_context_embeddings_compressed',
                migration_timeout_s,
            )
            or 0,
        )

        # Stream compressed rows in batches via LIMIT/OFFSET pagination.
        # Server-side cursors would keep a portal open against the
        # source table that PostgreSQL would reject when the trailing
        # DROP TABLE runs in the same transaction.
        rows_processed = 0
        offset = 0
        while True:
            batch = await fetch_migration(
                conn,
                'SELECT context_id, chunk_index, start_index, end_index, '
                'payload FROM vec_context_embeddings_compressed '
                'ORDER BY context_id, chunk_index '
                'LIMIT $1 OFFSET $2',
                migration_timeout_s,
                batch_size, offset,
            )
            if not batch:
                break
            for r in batch:
                vec = provider.decode_sync(bytes(r['payload']))[0]
                await conn.execute(
                    'INSERT INTO vec_context_embeddings '
                    '(context_id, embedding, start_index, end_index) '
                    'VALUES ($1, $2, $3, $4)',
                    str(r['context_id']),
                    [float(x) for x in vec.tolist()],
                    int(r['start_index']),
                    int(r['end_index']),
                )
            rows_processed += len(batch)
            offset += len(batch)

        # Integrity guard mirroring the compress path: with the ACCESS
        # EXCLUSIVE lock held since before the first batch, the table cannot
        # have changed mid-stream; a mismatch against the under-lock recount
        # signals OFFSET-pagination drift or an anomaly, and aborting is
        # still safer than dropping an undecoded payload.
        if live_count != rows_processed:
            raise ValueError(
                f'decompress streamed {rows_processed} compressed row(s) but '
                f'vec_context_embeddings_compressed holds {live_count} at drop '
                'time -- a concurrent writer modified the table while '
                '--decompress was running. Aborting so no undecoded payload is '
                'dropped; stop the server writing to this database and re-run '
                '--decompress.',
            )

        # HNSW index CREATE + compressed source DROP + provenance DELETE
        # last inside the same transaction.
        await execute_migration_ddl(
            conn,
            'CREATE INDEX IF NOT EXISTS idx_vec_context_embeddings_hnsw '
            'ON vec_context_embeddings '
            'USING hnsw (embedding vector_l2_ops) '
            'WITH (m = 16, ef_construction = 64)',
            migration_timeout_s,
        )
        await execute_migration_ddl(
            conn,
            'DROP TABLE IF EXISTS vec_context_embeddings_compressed',
            migration_timeout_s,
        )
        await conn.execute(
            'DELETE FROM compression_metadata WHERE id = 1',
        )

    logger.info(
        'Decompressed %d compressed row(s) into vec_context_embeddings; '
        'dropped compressed table and recreated HNSW index.',
        rows_processed or row_count,
    )


__all__ = ['run_compress', 'run_decompress']
