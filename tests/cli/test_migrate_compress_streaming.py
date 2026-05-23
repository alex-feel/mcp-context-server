"""Tests for the streaming compression migration's atomicity guarantees.

Exercises the SQLite branch end-to-end with multi-batch corpora to
verify:

* Multi-batch processing INSERTs every row when the corpus spans
  multiple ``_MIGRATION_BATCH_SIZE`` chunks.
* Re-running ``run_compress`` on an already-compressed database is a
  no-op (the singleton ``compression_metadata`` row guards the second
  run).
* When an injected failure raises mid-stream, the surrounding
  ``begin_transaction()`` rolls back: the source ``vec_context_embeddings``
  table stays intact and the compressed destination is empty/absent
  with no provenance row.
* Peak Python-allocated memory during ``run_compress`` is bounded by
  ``O(batch * dim * 4)`` plus a generous driver-overhead slack.
* The symmetric ``run_decompress`` round-trips a multi-batch corpus.

These tests live in ``tests/cli/`` so they run inside the default unit
gate and exercise the SQLite branch directly. PostgreSQL coverage is
gated on Docker pgvector availability and lives in
``tests/integration/postgresql/``.
"""

import asyncio
import contextlib
import sqlite3
import struct
import tracemalloc
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from app.backends import create_backend
from app.cli import migrate_compression
from app.cli.migrate_compression import run_compress
from app.cli.migrate_compression import run_decompress
from app.compression.base import CompressionProvider
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository import _reset_compression_cache
from app.settings import get_settings

DIM = 128


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset the settings + compression caches around every test."""
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()


def _seed_fp32_database(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_docs: int,
) -> None:
    """Write ``n_docs`` fp32 vectors to an isolated SQLite database."""
    monkeypatch.setenv('DB_PATH', str(db_path))
    monkeypatch.setenv('STORAGE_BACKEND', 'sqlite')
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    monkeypatch.delenv('ENABLE_SEMANTIC_SEARCH', raising=False)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.delenv('COMPRESSION_SEED', raising=False)
    get_settings.cache_clear()

    async def _setup() -> None:
        from app.schemas import load_schema

        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(load_schema('sqlite'))
            conn.executescript(
                '''
                CREATE TABLE IF NOT EXISTS vec_context_embeddings (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embedding_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id TEXT NOT NULL,
                    vec_rowid INTEGER NOT NULL,
                    start_index INTEGER NOT NULL DEFAULT 0,
                    end_index INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    context_id TEXT NOT NULL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
                );
                ''',
            )
        finally:
            conn.close()

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            repos = RepositoryContainer(backend)
            rng = np.random.default_rng(7)
            for i in range(n_docs):
                vec = rng.standard_normal(DIM).astype(np.float32)
                vec /= np.linalg.norm(vec)
                cid, _ = await repos.context.store_with_deduplication(
                    thread_id='stream-e2e',
                    source='user',
                    content_type='text',
                    text_content=f'doc-{i}',
                    metadata=None,
                )
                blob = struct.pack(f'<{DIM}f', *vec.tolist())

                def _write_chunk(
                    conn: sqlite3.Connection,
                    *,
                    context_id: str = cid,
                    payload: bytes = blob,
                ) -> None:
                    cur = conn.execute(
                        'INSERT INTO vec_context_embeddings (embedding) VALUES (?)',
                        (payload,),
                    )
                    vec_rowid = cur.lastrowid
                    conn.execute(
                        'INSERT INTO embedding_chunks '
                        '(context_id, vec_rowid, start_index, end_index) '
                        'VALUES (?, ?, ?, ?)',
                        (context_id, vec_rowid, 0, DIM),
                    )
                    conn.execute(
                        'INSERT INTO embedding_metadata '
                        '(context_id, model_name, dimensions, chunk_count) '
                        'VALUES (?, ?, ?, ?)',
                        (context_id, 'test-model', DIM, 1),
                    )

                await backend.execute_write(_write_chunk)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_setup())


def _enable_compression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable IP-4 compression with a fixed seed."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    get_settings.cache_clear()


def _table_exists(db_path: Path, name: str) -> bool:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def _count_compressed(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return int(
            conn.execute(
                'SELECT COUNT(*) FROM vec_context_embeddings_compressed',
            ).fetchone()[0],
        )
    finally:
        conn.close()


def _count_fp32(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return int(
            conn.execute(
                'SELECT COUNT(*) FROM vec_context_embeddings',
            ).fetchone()[0],
        )
    finally:
        conn.close()


def _count_provenance(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='compression_metadata'",
        )
        if cur.fetchone() is None:
            return 0
        return int(
            conn.execute(
                'SELECT COUNT(*) FROM compression_metadata WHERE id = 1',
            ).fetchone()[0],
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Multi-batch execution
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_streaming_compress_multi_batch_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming compress writes every row across multiple batches."""
    db = tmp_path / 'multi_batch.db'
    n_docs = 25
    # Shrink the batch size so the loop runs multiple iterations without
    # seeding 25k rows in test setup.
    monkeypatch.setattr(migrate_compression, '_MIGRATION_BATCH_SIZE', 8)
    _seed_fp32_database(db, monkeypatch, n_docs=n_docs)
    _enable_compression(monkeypatch)

    rc = run_compress(f'sqlite:///{db}', dry_run=False)
    assert rc == 0
    assert _count_compressed(db) == n_docs
    assert not _table_exists(db, 'vec_context_embeddings')
    assert _count_provenance(db) == 1


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_streaming_compress_idempotent_no_op_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second --compress invocation is a no-op via the singleton check."""
    db = tmp_path / 'idempotent.db'
    n_docs = 6
    monkeypatch.setattr(migrate_compression, '_MIGRATION_BATCH_SIZE', 4)
    _seed_fp32_database(db, monkeypatch, n_docs=n_docs)
    _enable_compression(monkeypatch)

    assert run_compress(f'sqlite:///{db}', dry_run=False) == 0
    after_first = _count_compressed(db)
    assert after_first == n_docs

    rc2 = run_compress(f'sqlite:///{db}', dry_run=False)
    assert rc2 == 0
    assert _count_compressed(db) == after_first
    assert _count_provenance(db) == 1


# ---------------------------------------------------------------------------
# Atomic rollback under injected mid-stream failure
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_streaming_compress_atomic_rollback_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure mid-encode rolls back the entire transaction.

    Seeds a corpus, then wraps the cached compression provider's
    ``encode_sync`` so it raises after a few successful calls. The
    surrounding ``begin_transaction()`` must roll back, leaving the
    source fp32 table intact and the compressed table absent or empty.
    """
    db = tmp_path / 'rollback.db'
    n_docs = 200
    monkeypatch.setattr(migrate_compression, '_MIGRATION_BATCH_SIZE', 16)
    _seed_fp32_database(db, monkeypatch, n_docs=n_docs)
    _enable_compression(monkeypatch)

    from app.compression import factory as compression_factory

    real_provider = compression_factory.create_compression_provider()
    original_encode = real_provider.encode_sync
    call_count = {'n': 0}
    # Probe phase encodes min(PROBE_BATCH_SIZE, n_docs) rows. Allow the
    # probe to complete plus 20 rows of the actual streaming loop before
    # injecting the failure.
    probe_rows = min(migrate_compression.PROBE_BATCH_SIZE, n_docs)
    fail_after = probe_rows + 20

    def _failing_encode(vectors: NDArray[np.float32]) -> bytes:
        call_count['n'] += 1
        if call_count['n'] > fail_after:
            raise RuntimeError('injected encode failure for rollback test')
        return original_encode(vectors)

    def _patched_create_provider() -> CompressionProvider:
        # Each invocation rewires encode_sync against the same wrapped
        # function so the probe and the streaming loop share a counter.
        monkeypatch.setattr(real_provider, 'encode_sync', _failing_encode)
        return real_provider

    monkeypatch.setattr(
        compression_factory,
        'create_compression_provider',
        _patched_create_provider,
    )
    # The CLI imports create_compression_provider lazily from
    # app.compression (package re-export); rebind that name too.
    import app.compression as compression_pkg
    monkeypatch.setattr(
        compression_pkg,
        'create_compression_provider',
        _patched_create_provider,
    )

    rc = run_compress(f'sqlite:///{db}', dry_run=False)
    # run_compress catches the exception and returns a non-zero exit code.
    assert rc != 0

    # Source intact, no compressed rows committed, no provenance row.
    assert _table_exists(db, 'vec_context_embeddings')
    assert _count_fp32(db) == n_docs
    if _table_exists(db, 'vec_context_embeddings_compressed'):
        assert _count_compressed(db) == 0
    assert _count_provenance(db) == 0


# ---------------------------------------------------------------------------
# Bounded peak memory
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_streaming_compress_memory_bounded_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Peak Python-allocated bytes do NOT scale with corpus size.

    Runs ``run_compress`` against two corpora of different sizes (small
    and large) with the SAME batch size. Asserts that the peak memory
    of the large run does NOT grow proportionally with the corpus size
    -- i.e., per-row peak memory is bounded. Concretely, the large-run
    peak must stay below a generous multiple of the small-run peak
    (the multiplier is small relative to the 5x corpus-size delta the
    bulk path would force).

    The test's purpose is to catch O(N) regressions where peak memory
    scales with corpus size, NOT to assert tight memory bounds. Encoder
    setup cost, libpython runtime overhead, NumPy ndarray headers, and
    sqlite3 driver buffers all factor into the absolute peak.
    """
    batch_size = 16
    monkeypatch.setattr(migrate_compression, '_MIGRATION_BATCH_SIZE', batch_size)

    def _measure_peak(n_docs: int, db_name: str) -> int:
        db = tmp_path / db_name
        _seed_fp32_database(db, monkeypatch, n_docs=n_docs)
        _enable_compression(monkeypatch)
        tracemalloc.start()
        try:
            rc = run_compress(f'sqlite:///{db}', dry_run=False)
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert rc == 0
        return peak_bytes

    n_small = 50
    n_large = 250
    small_peak = _measure_peak(n_small, 'memory_small.db')
    large_peak = _measure_peak(n_large, 'memory_large.db')

    # If streaming works, large_peak ~ small_peak + small constant. The
    # bulk-load path would produce large_peak ~ (n_large / n_small) *
    # working_set. Allow the large peak to be up to 2x the small peak;
    # bulk-load would be ~5x.
    assert large_peak <= 2 * small_peak, (
        f'peak memory scaled from {small_peak} to {large_peak} bytes when '
        f'corpus grew from {n_small} to {n_large} rows; streaming may have '
        f'regressed to bulk-load behavior (ratio {large_peak / max(small_peak, 1):.2f}).'
    )


# ---------------------------------------------------------------------------
# Streaming decompress round-trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_streaming_decompress_roundtrip_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-batch compress then decompress restores the original row count."""
    db = tmp_path / 'roundtrip.db'
    n_docs = 12
    monkeypatch.setattr(migrate_compression, '_MIGRATION_BATCH_SIZE', 5)
    _seed_fp32_database(db, monkeypatch, n_docs=n_docs)
    _enable_compression(monkeypatch)

    assert run_compress(f'sqlite:///{db}', dry_run=False) == 0
    assert _count_compressed(db) == n_docs

    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', 'true')
    get_settings.cache_clear()
    _reset_compression_cache()
    import app.backends.sqlite_backend as sqlite_backend_module
    monkeypatch.setattr(
        sqlite_backend_module, 'settings', get_settings(),
    )

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)
    assert rc == 0
    assert _table_exists(db, 'vec_context_embeddings')
    assert not _table_exists(db, 'vec_context_embeddings_compressed')
    assert _count_provenance(db) == 0
    conn = sqlite3.connect(str(db))
    try:
        chunk_count = int(
            conn.execute(
                'SELECT COUNT(*) FROM embedding_chunks',
            ).fetchone()[0],
        )
    finally:
        conn.close()
    assert chunk_count == n_docs
