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
    """Return True if ``table_name`` exists in the database."""
    if backend.backend_type == 'sqlite':

        def _check(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cursor.fetchone() is not None

        return await backend.execute_read(_check)

    schema = get_settings().storage.postgresql_schema

    async def _check_pg(conn: asyncpg.Connection) -> bool:
        result = await conn.fetchval(
            '''
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = $1
                  AND table_name = $2
            )
            ''',
            schema, table_name,
        )
        return bool(result)

    return await backend.execute_read(cast(Any, _check_pg))


async def _count_table(backend: StorageBackend, table_name: str) -> int:
    """Return the row count for ``table_name``."""
    safe = _safe_table(table_name)
    if backend.backend_type == 'sqlite':

        def _count(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(f'SELECT COUNT(*) FROM {safe}')
            return int(cursor.fetchone()[0])

        return await backend.execute_read(_count)

    async def _count_pg(conn: asyncpg.Connection) -> int:
        result = await conn.fetchval(f'SELECT COUNT(*) FROM {safe}')
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
            row_count=row_count,
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
    row_count: int,
) -> None:
    """Encode every fp32 row and write the compressed payload table.

    The whole operation runs inside a single transaction (PG: native;
    SQLite: implicit via ``execute_write`` which commits on success and
    rolls back on exception).
    """
    if backend.backend_type == 'sqlite':
        await _execute_compress_sqlite(
            backend, provider, provenance, row_count,
        )
        return
    await _execute_compress_postgresql(
        backend, provider, provenance, row_count,
    )


async def _execute_compress_sqlite(
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
    row_count: int,
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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            ''',
        )

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

        # Provenance INSERT + source DROP last inside the same
        # transaction. The CHECK (id = 1) constraint guarantees this
        # INSERT is the only one ever applied.
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim) '
            'VALUES (1, ?, ?, ?, ?, ?)',
            (
                provenance.provider,
                provenance.bits,
                provenance.variant,
                provenance.seed,
                provenance.dim,
            ),
        )
        conn.execute('DROP TABLE IF EXISTS vec_context_embeddings')

    logger.info(
        'Compressed %d fp32 row(s) into vec_context_embeddings_compressed; '
        'dropped legacy table.',
        rows_processed or row_count,
    )


async def _execute_compress_postgresql(
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
    row_count: int,
) -> None:
    """PostgreSQL branch of :func:`_execute_compress`.

    Streams fp32 rows in batches of :data:`_MIGRATION_BATCH_SIZE` from
    ``vec_context_embeddings`` using LIMIT/OFFSET pagination ordered by
    ``(context_id, id)`` and encodes each batch in-process. Server-side
    cursors would keep a portal open against the source table that
    PostgreSQL would reject when the trailing ``DROP TABLE`` runs in the
    same transaction. Every batch INSERT plus the provenance row, the
    HNSW index DROP, and the source table DROP run inside ONE
    :meth:`StorageBackend.begin_transaction` so any failure rolls back
    all work and the source table remains intact.

    Bounded peak memory: ``O(_MIGRATION_BATCH_SIZE * dim * 4)`` bytes
    (~40 MB at ``dim == 1024``).
    """
    batch_size = _MIGRATION_BATCH_SIZE

    async with backend.begin_transaction() as txn:
        conn = cast('asyncpg.Connection', txn.connection)

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
            batch = await conn.fetch(
                'SELECT context_id, id, start_index, end_index, embedding '
                'FROM vec_context_embeddings '
                'ORDER BY context_id, id '
                'LIMIT $1 OFFSET $2',
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

        # Provenance INSERT + HNSW index DROP + source DROP last inside
        # the same transaction.
        await conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim) '
            'VALUES (1, $1, $2, $3, $4, $5)',
            provenance.provider,
            provenance.bits,
            provenance.variant,
            provenance.seed,
            provenance.dim,
        )
        await conn.execute(
            'DROP INDEX IF EXISTS idx_vec_context_embeddings_hnsw',
        )
        await conn.execute('DROP TABLE IF EXISTS vec_context_embeddings')

    logger.info(
        'Compressed %d fp32 row(s) into vec_context_embeddings_compressed; '
        'dropped legacy table and HNSW index.',
        rows_processed or row_count,
    )


async def _execute_decompress(
    *,
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
    row_count: int,
) -> None:
    """Decode every compressed row and write the fp32 vec table.

    The whole operation runs inside a single transaction (PG: native;
    SQLite: implicit via ``execute_write``).
    """
    if backend.backend_type == 'sqlite':
        await _execute_decompress_sqlite(
            backend, provider, provenance, row_count,
        )
        return
    await _execute_decompress_postgresql(
        backend, provider, provenance, row_count,
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
    deployment. Provenance DELETE + compressed source DROP run last inside
    the same transaction.
    """
    batch_size = _MIGRATION_BATCH_SIZE

    async with backend.begin_transaction() as txn:
        conn = cast(sqlite3.Connection, txn.connection)

        # Recreate the fp32 virtual table (idempotent).
        conn.executescript(
            f'CREATE VIRTUAL TABLE IF NOT EXISTS vec_context_embeddings '
            f'USING vec0(embedding float[{provenance.dim}])',
        )

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
    transaction. Recreates ``vec_context_embeddings`` + HNSW index
    inside the same transaction that streams decompressed rows in
    batches.
    """
    batch_size = _MIGRATION_BATCH_SIZE

    async with backend.begin_transaction() as txn:
        conn = cast('asyncpg.Connection', txn.connection)

        # Bare table names rely on PostgreSQL's search_path resolution;
        # this matches the project-wide pattern in
        # ``embedding_repository.py`` and ``postgresql_schema.sql``.
        # Operators using a non-default schema configure ``search_path``
        # accordingly. The ``information_schema.tables`` lookup below
        # passes the schema name as a parameterized bind value rather
        # than composing it into the SQL string, matching the binding
        # pattern used by ``_check_pg`` at lines 233-244.
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
        existing_count = await conn.fetchval(
            'SELECT COUNT(*) FROM information_schema.tables '
            'WHERE table_schema = $1 '
            "AND table_name = 'vec_context_embeddings_compressed'",
            get_settings().storage.postgresql_schema,
        )
        if int(existing_count or 0) == 0:
            logger.info(
                'Reverse migration already applied (compressed source '
                'table absent); --decompress is a no-op.',
            )
            return

        # Stream compressed rows in batches via LIMIT/OFFSET pagination.
        # Server-side cursors would keep a portal open against the
        # source table that PostgreSQL would reject when the trailing
        # DROP TABLE runs in the same transaction.
        rows_processed = 0
        offset = 0
        while True:
            batch = await conn.fetch(
                'SELECT context_id, chunk_index, start_index, end_index, '
                'payload FROM vec_context_embeddings_compressed '
                'ORDER BY context_id, chunk_index '
                'LIMIT $1 OFFSET $2',
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

        # HNSW index CREATE + compressed source DROP + provenance DELETE
        # last inside the same transaction.
        await conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_vec_context_embeddings_hnsw '
            'ON vec_context_embeddings '
            'USING hnsw (embedding vector_l2_ops) '
            'WITH (m = 16, ef_construction = 64)',
        )
        await conn.execute(
            'DROP TABLE IF EXISTS vec_context_embeddings_compressed',
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
