"""Backfill missing embeddings via the active embedding provider.

Implements the ``--embed-missing`` CLI flag of
``mcp-context-server-migrate``. Identifies context_entries rows that
lack an embedding_metadata row and runs the live embedding pipeline
against their ``text_content`` to fill the gap.

Shape gamma (HYBRID):
    Standalone:
        Backfills against the existing storage layout. When
        ``ENABLE_EMBEDDING_COMPRESSION=true`` is set in the invocation
        env, the freshly generated embeddings land in
        ``vec_context_embeddings_compressed``; otherwise they land in
        the fp32 ``vec_context_embeddings`` table.

    Composed with ``--compress``:
        The orchestrator in :mod:`app.cli.migrate` runs ``--compress``
        first (fp32 -> compressed), then dispatches to this module to
        backfill any missing entries directly into the compressed
        layout.

The CLI is single-backend: source is the database under operation;
``--target-url`` is ignored (mirrors ``--compress`` / ``--decompress``).
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
from app.cli.migrate import mask_credentials
from app.cli.migrate import parse_backend_url
from app.settings import get_settings

logger = logging.getLogger(__name__)


def _print_warning(*, source_url: str, dry_run: bool) -> None:
    """Print the operator warning describing the cost surface."""
    lines = [
        '=' * 60,
        'EMBEDDING BACKFILL -- this operation issues live embedding-',
        'provider requests against text_content for every entry that',
        'lacks an embedding. Cost (API requests, latency) scales with',
        'the number of missing entries.',
        '-' * 60,
        f'Source: {source_url}',
    ]
    if dry_run:
        lines.append('DRY-RUN: provider is not called; only the count is reported')
    lines.append('=' * 60)
    print('\n'.join(lines), file=sys.stderr)


def _make_backend(source_url: str) -> StorageBackend:
    """Build a backend pointed at ``source_url``.

    Mirrors :func:`app.cli.migrate_compression._make_backend`.

    Args:
        source_url: URL passed to ``--source-url``.

    Returns:
        Initialized :class:`StorageBackend` matching the URL scheme.
    """
    backend_kind, address = parse_backend_url(source_url)
    if backend_kind == 'sqlite':
        return create_backend(backend_type='sqlite', db_path=address)
    return create_backend(
        backend_type='postgresql', connection_string=address,
    )


async def _shutdown(backend: StorageBackend) -> None:
    """Bounded shutdown.

    Mirrors :func:`app.cli.migrate_compression._shutdown`.
    """
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=10.0)


def run_embed_missing(source_url: str, *, dry_run: bool) -> int:
    """Public entry point for the ``--embed-missing`` flag.

    Args:
        source_url: Database URL passed to ``--source-url``.
        dry_run: When True, only count and report; no provider calls.

    Returns:
        Process exit code: 0 success or success-no-op, 1 invalid input,
        2 unrecoverable failure.
    """
    masked = mask_credentials(source_url)
    _print_warning(source_url=masked, dry_run=dry_run)
    settings = get_settings()
    if not settings.embedding.generation_enabled:
        print(
            '[ERROR] ENABLE_EMBEDDING_GENERATION=true must be set for '
            '--embed-missing. The pipeline calls the embedding provider '
            'configured via EMBEDDING_PROVIDER/EMBEDDING_MODEL.',
            file=sys.stderr,
        )
        return 1

    try:
        return asyncio.run(_embed_missing_async(source_url, dry_run=dry_run))
    except Exception as exc:
        logger.exception('embed-missing failed: %s', exc)
        return 2


async def _embed_missing_async(source_url: str, *, dry_run: bool) -> int:
    """Async body: identify missing entries, generate, store atomically.

    Defers imports of ``app.tools._shared`` and ``app.repositories`` until
    runtime so the module can be imported by the CLI dispatcher without
    pulling in the embedding/compression dependency chain unconditionally.

    Args:
        source_url: URL passed to ``--source-url``.
        dry_run: When True, only count and report; no provider calls.

    Returns:
        Process exit code: 0 on success or no-op, 1 when the
        ``embedding_metadata`` table is absent from the source database.
    """
    masked = mask_credentials(source_url)
    backend = _make_backend(source_url)
    await backend.initialize()
    try:
        # Deferred imports: keep numpy / heavy provider machinery out of
        # the dispatcher's import graph.
        from app.repositories import RepositoryContainer
        from app.tools._shared import generate_compression_with_timeout
        from app.tools._shared import generate_embeddings_with_timeout

        repos = RepositoryContainer(backend)

        if not await _embedding_metadata_table_exists(backend):
            # Schema predates the semantic-search migration; every row
            # would qualify as missing but the storage path would fail
            # because the target tables do not exist. Surface a clearer
            # error here first.
            print(
                '[ERROR] embedding_metadata table not present in source. '
                'Initialize the database against the live server first '
                '(or run the semantic-search migration) so the target '
                'tables exist.',
                file=sys.stderr,
            )
            return 1

        missing = await _find_missing_entries(backend)
        print(
            f'Found {len(missing)} entries with missing embeddings on {masked}.',
            file=sys.stderr,
        )
        if not missing:
            return 0
        if dry_run:
            print(
                '[DRY-RUN] Provider not called; rerun without --dry-run '
                'to backfill.',
                file=sys.stderr,
            )
            return 0

        settings = get_settings()
        model = settings.embedding.model

        start = time.perf_counter()
        succeeded = 0
        for i, (entry_id, text) in enumerate(missing, start=1):
            chunk_embeddings = await generate_embeddings_with_timeout(text)
            if chunk_embeddings is None:
                print(
                    f'[SKIP] entry {entry_id}: embedding provider not '
                    f'configured (ENABLE_EMBEDDING_GENERATION may be true '
                    f'but no provider returned a result)',
                    file=sys.stderr,
                )
                continue

            # generate_compression_with_timeout is a no-op when
            # compression.enabled is False; populates the payload field
            # when True. Calling it unconditionally mirrors the live
            # server's store_context / update_context code path.
            chunk_embeddings = await generate_compression_with_timeout(chunk_embeddings)
            if chunk_embeddings is None:
                continue

            async with backend.begin_transaction() as txn:
                await repos.embeddings.store_chunked(
                    context_id=entry_id,
                    chunk_embeddings=chunk_embeddings,
                    model=model,
                    txn=txn,
                    upsert=False,
                )
            succeeded += 1

            if i % 50 == 0:
                elapsed = time.perf_counter() - start
                rate = i / max(elapsed, 1e-6)
                eta = (len(missing) - i) / rate if rate > 0 else 0.0
                print(
                    f'[PROGRESS] {i}/{len(missing)} '
                    f'({rate:.1f} entries/s, ETA {eta:.0f}s)',
                    file=sys.stderr,
                )

        elapsed = time.perf_counter() - start
        print(
            f'Backfill complete: {succeeded}/{len(missing)} entries '
            f'embedded in {elapsed:.1f}s.',
            file=sys.stderr,
        )
        return 0
    finally:
        await _shutdown(backend)


async def _embedding_metadata_table_exists(backend: StorageBackend) -> bool:
    """Probe the ``embedding_metadata`` table presence.

    Args:
        backend: Storage backend to query.

    Returns:
        True when the ``embedding_metadata`` table exists in the database.
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
                WHERE table_name = 'embedding_metadata'
            )
            ''',
        )
        return bool(result)

    return await backend.execute_read(cast(Any, _check_pg))


async def _find_missing_entries(backend: StorageBackend) -> list[tuple[str, str]]:
    """Find context_entries rows lacking an embedding_metadata row.

    Backend-specific SQL because PostgreSQL UUID and SQLite TEXT have
    different binding shapes. ORDER BY ``ce.id`` produces a deterministic
    order for stable progress reporting and reproducible test runs.

    Args:
        backend: Storage backend to query.

    Returns:
        List of ``(id, text_content)`` tuples for rows that have no
        corresponding ``embedding_metadata`` entry and a non-NULL
        ``text_content``.
    """
    if backend.backend_type == 'sqlite':

        def _read_sqlite(conn: sqlite3.Connection) -> list[tuple[str, str]]:
            cursor = conn.execute(
                'SELECT ce.id, ce.text_content FROM context_entries ce '
                'LEFT JOIN embedding_metadata em ON em.context_id = ce.id '
                'WHERE em.context_id IS NULL AND ce.text_content IS NOT NULL '
                'ORDER BY ce.id ASC',
            )
            return [(str(r[0]), str(r[1])) for r in cursor.fetchall()]

        return await backend.execute_read(_read_sqlite)

    async def _read_pg(conn: asyncpg.Connection) -> list[tuple[str, str]]:
        rows = await conn.fetch(
            'SELECT ce.id, ce.text_content FROM context_entries ce '
            'LEFT JOIN embedding_metadata em ON em.context_id = ce.id '
            'WHERE em.context_id IS NULL AND ce.text_content IS NOT NULL '
            'ORDER BY ce.id ASC',
        )
        return [(str(r['id']), str(r['text_content'])) for r in rows]

    return await backend.execute_read(cast(Any, _read_pg))


__all__ = ['run_embed_missing']
