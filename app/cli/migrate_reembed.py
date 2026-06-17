"""Re-embed an existing corpus under the currently configured model.

Implements the ``--re-embed`` flag of ``mcp-context-server-migrate``. Unlike
``--embed-missing`` (which only fills entries that LACK an
``embedding_metadata`` row), ``--re-embed`` regenerates embeddings for EVERY
``context_entries`` row that has ``text_content``, deleting any existing
embeddings first. This is the one-command path for switching the embedding
MODEL on an existing database -- the operation the old manual "drop the
embedding tables, set the new model, regenerate" procedure described.

Scope -- model change at the SAME dimension:
    ``--re-embed`` re-embeds with the configured ``EMBEDDING_MODEL`` at the
    EXISTING vector dimension. It deliberately REFUSES a dimension change
    (a configured ``EMBEDDING_DIM`` different from the stored dimension)
    because a dimension change rewrites the vector-storage geometry: the
    fp32 ``vec_context_embeddings`` column width is fixed at creation, and
    under compression the dimension is part of the seed-locked
    ``compression_metadata`` codebook (immutable by design). A dimension
    change is the documented destructive rebuild (back up, recreate the
    database at the new dimension, re-store the data); see
    ``docs/migration-v2-to-v3.md``.

Layouts:
    Works for both fp32 and compressed databases. The delete and store paths
    branch on ``ENABLE_EMBEDDING_COMPRESSION`` internally
    (``EmbeddingRepository.delete_all_chunks`` /
    ``EmbeddingRepository.store_chunked``), and
    ``generate_compression_with_timeout`` populates the payload bytes when
    compression is enabled.

The CLI is single-backend: source is the database under operation;
``--target-url`` is ignored (mirrors ``--compress`` / ``--decompress`` /
``--embed-missing``).
"""

import asyncio
import contextlib
import logging
import sqlite3
import sys
import time
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends import create_backend
from app.cli._embedding_introspect import dimension_conflict_error
from app.cli._embedding_introspect import distinct_embedding_models
from app.cli._embedding_introspect import embedding_metadata_table_exists
from app.cli._embedding_runtime import EmbeddingPipelineUnavailableError
from app.cli._embedding_runtime import initialize_cli_embedding_pipeline
from app.cli._embedding_runtime import shutdown_cli_embedding_pipeline
from app.cli.migrate import mask_credentials
from app.cli.migrate import parse_backend_url
from app.embeddings.base import EmbeddingProvider
from app.settings import get_settings

logger = logging.getLogger(__name__)


def _print_warning(*, source_url: str, dry_run: bool) -> None:
    """Print the operator warning describing the cost surface."""
    lines = [
        '=' * 60,
        'RE-EMBED ALL -- this operation deletes every existing embedding',
        'and regenerates it with the configured EMBEDDING_MODEL, issuing',
        'live embedding-provider requests for every entry. Cost (API',
        'requests, latency) scales with the TOTAL number of entries.',
        'Back up the database before running without --dry-run.',
        '-' * 60,
        f'Source: {source_url}',
    ]
    if dry_run:
        lines.append('DRY-RUN: provider is not called; only the plan is reported')
    lines.append('=' * 60)
    print('\n'.join(lines), file=sys.stderr)


def _make_backend(source_url: str) -> StorageBackend:
    """Build a backend pointed at ``source_url``.

    Mirrors :func:`app.cli.migrate_embeddings._make_backend`.

    Args:
        source_url: URL passed to ``--source-url``.

    Returns:
        Uninitialized :class:`StorageBackend` matching the URL scheme.
    """
    backend_kind, address = parse_backend_url(source_url)
    if backend_kind == 'sqlite':
        return create_backend(backend_type='sqlite', db_path=address)
    return create_backend(backend_type='postgresql', connection_string=address)


def run_reembed(source_url: str, *, dry_run: bool) -> int:
    """Public entry point for the ``--re-embed`` flag.

    Args:
        source_url: Database URL passed to ``--source-url``.
        dry_run: When True, report the plan only; no provider calls.

    Returns:
        Process exit code: 0 success or success-no-op, 1 invalid input or
        provider unavailable, 2 unrecoverable failure.
    """
    masked = mask_credentials(source_url)
    _print_warning(source_url=masked, dry_run=dry_run)
    settings = get_settings()
    if not settings.embedding.generation_enabled:
        print(
            '[ERROR] ENABLE_EMBEDDING_GENERATION=true must be set for '
            '--re-embed. The pipeline calls the embedding provider '
            'configured via EMBEDDING_PROVIDER/EMBEDDING_MODEL.',
            file=sys.stderr,
        )
        return 1

    try:
        return asyncio.run(_reembed_async(source_url, dry_run=dry_run))
    except Exception as exc:
        logger.exception('re-embed failed: %s', exc)
        return 2


async def _reembed_async(source_url: str, *, dry_run: bool) -> int:
    """Async body: validate, then delete + regenerate per entry atomically.

    Defers imports of ``app.tools._shared`` and ``app.repositories`` until
    runtime so the module can be imported by the CLI dispatcher without
    pulling in the embedding/compression dependency chain unconditionally.

    Args:
        source_url: URL passed to ``--source-url``.
        dry_run: When True, report the plan only; no provider calls.

    Returns:
        Process exit code: 0 on success or no-op, 1 on a structural or
        configuration error.
    """
    masked = mask_credentials(source_url)
    backend = _make_backend(source_url)
    await backend.initialize()
    try:
        from app.repositories import RepositoryContainer
        from app.tools._shared import generate_compression_with_timeout
        from app.tools._shared import generate_embeddings_with_timeout

        repos = RepositoryContainer(backend)

        if not await embedding_metadata_table_exists(backend):
            print(
                '[ERROR] embedding_metadata table not present in source. '
                'Initialize the database against the live server first '
                '(or run the semantic-search migration) so the embedding '
                'tables exist before re-embedding.',
                file=sys.stderr,
            )
            return 1

        settings = get_settings()
        configured_model = settings.embedding.model
        configured_dim = settings.embedding.dim

        # Refuse a dimension change: --re-embed swaps the MODEL at the
        # existing dimension. Changing the dimension rewrites the vector
        # storage geometry (and the seed-locked compressed codebook) and is
        # the documented destructive rebuild. Shared with --embed-missing so
        # the two guards cannot diverge.
        dim_error = await dimension_conflict_error(
            backend, configured_dim=configured_dim,
        )
        if dim_error is not None:
            print(f'[ERROR] {dim_error}', file=sys.stderr)
            return 1

        existing_models = await distinct_embedding_models(backend)
        entries = await _find_all_entries(backend)

        print(
            f'Re-embedding {len(entries)} entries with model '
            f'{configured_model!r} on {masked}.',
            file=sys.stderr,
        )
        changing_models = sorted(m for m in existing_models if m != configured_model)
        if changing_models:
            print(
                f'Existing embeddings used model(s) {changing_models}; '
                f'replacing all with {configured_model!r}.',
                file=sys.stderr,
            )
        if not entries:
            return 0
        if dry_run:
            print(
                '[DRY-RUN] Provider not called; rerun without --dry-run '
                'to re-embed.',
                file=sys.stderr,
            )
            return 0

        provider: EmbeddingProvider | None = None
        try:
            provider = await initialize_cli_embedding_pipeline()
        except EmbeddingPipelineUnavailableError as exc:
            print(f'[ERROR] {exc}', file=sys.stderr)
            return 1

        try:
            start = time.perf_counter()
            succeeded = 0
            for i, (entry_id, text) in enumerate(entries, start=1):
                chunk_embeddings = await generate_embeddings_with_timeout(text)
                if chunk_embeddings is None:
                    print(
                        f'[SKIP] entry {entry_id}: embedding provider returned '
                        f'no result',
                        file=sys.stderr,
                    )
                    continue

                # No-op when compression is disabled; populates payload bytes
                # when enabled. Mirrors store_context / update_context.
                chunk_embeddings = await generate_compression_with_timeout(chunk_embeddings)
                if chunk_embeddings is None:
                    continue

                # Delete-then-store inside ONE transaction so an entry is
                # never left without embeddings: delete_all_chunks removes the
                # old vectors + embedding_metadata, store_chunked writes the
                # new ones (and the new model_name).
                async with backend.begin_transaction() as txn:
                    await repos.embeddings.delete_all_chunks(entry_id, txn=txn)
                    await repos.embeddings.store_chunked(
                        context_id=entry_id,
                        chunk_embeddings=chunk_embeddings,
                        model=configured_model,
                        txn=txn,
                        upsert=False,
                    )
                succeeded += 1

                if i % 50 == 0:
                    elapsed = time.perf_counter() - start
                    rate = i / max(elapsed, 1e-6)
                    eta = (len(entries) - i) / rate if rate > 0 else 0.0
                    print(
                        f'[PROGRESS] {i}/{len(entries)} '
                        f'({rate:.1f} entries/s, ETA {eta:.0f}s)',
                        file=sys.stderr,
                    )

            elapsed = time.perf_counter() - start
            print(
                f'Re-embed complete: {succeeded}/{len(entries)} entries '
                f'embedded with {configured_model!r} in {elapsed:.1f}s.',
                file=sys.stderr,
            )
            return 0
        finally:
            await shutdown_cli_embedding_pipeline(provider)
    finally:
        await _shutdown(backend)


async def _shutdown(backend: StorageBackend) -> None:
    """Bounded backend shutdown.

    Mirrors :func:`app.cli.migrate_embeddings._shutdown`.
    """
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=10.0)


async def _find_all_entries(backend: StorageBackend) -> list[tuple[str, str]]:
    """Find every context_entries row with non-NULL ``text_content``.

    ``ORDER BY ce.id`` produces a deterministic order for stable progress
    reporting and reproducible test runs.

    Args:
        backend: Storage backend to query.

    Returns:
        List of ``(id, text_content)`` tuples for every row that has a
        non-NULL ``text_content`` (the full corpus, regardless of whether the
        row currently has embeddings).
    """
    if backend.backend_type == 'sqlite':

        def _read_sqlite(conn: sqlite3.Connection) -> list[tuple[str, str]]:
            cursor = conn.execute(
                'SELECT id, text_content FROM context_entries '
                'WHERE text_content IS NOT NULL ORDER BY id ASC',
            )
            return [(str(r[0]), str(r[1])) for r in cursor.fetchall()]

        return await backend.execute_read(_read_sqlite)

    async def _read_pg(conn: asyncpg.Connection) -> list[tuple[str, str]]:
        rows = await conn.fetch(
            'SELECT id, text_content FROM context_entries '
            'WHERE text_content IS NOT NULL ORDER BY id ASC',
        )
        return [(str(r['id']), str(r['text_content'])) for r in rows]

    return await backend.execute_read(cast(Any, _read_pg))


__all__ = ['run_reembed']
