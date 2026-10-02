"""Tests for the transactional ``--compress`` data movement on SQLite and PostgreSQL.

The SQLite tests stream multi-batch corpora (``storage.MIGRATION_BATCH_SIZE`` shrunk per test) and check that every
row is encoded, that a mid-stream failure rolls the whole transaction back, that an orphaned fp32 row aborts before
the source DROP, that peak memory does not grow with the corpus, and that a provenance table predating
``codebook_fingerprint`` gains the column. The PostgreSQL tests drive the branch with a recording connection.
"""

import asyncio
import sqlite3
import struct
import tracemalloc
from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray

from app.cli.migrate_compression import console
from app.cli.migrate_compression import storage
from app.cli.migrate_compression.compress import run_compress
from app.cli.migrate_compression.compress_execution import _execute_compress_postgresql
from app.compression.base import CompressionProvider
from app.settings import get_settings
from tests.cli.migrate_compression._pg_fakes import FakeBackend
from tests.cli.migrate_compression._pg_fakes import RecordingConn
from tests.cli.migrate_compression._pg_fakes import budget_set_local
from tests.cli.migrate_compression._pg_fakes import expected_client_timeout
from tests.cli.migrate_compression._pg_fakes import sample_provenance
from tests.cli.migrate_compression._pg_fakes import timeouts_for
from tests.cli.migrate_compression._sqlite_db import DIM
from tests.cli.migrate_compression._sqlite_db import bootstrap_schema
from tests.cli.migrate_compression._sqlite_db import count_compressed
from tests.cli.migrate_compression._sqlite_db import count_fp32
from tests.cli.migrate_compression._sqlite_db import count_provenance
from tests.cli.migrate_compression._sqlite_db import create_fp32_vec_table
from tests.cli.migrate_compression._sqlite_db import enable_compression
from tests.cli.migrate_compression._sqlite_db import seed_fp32_database
from tests.cli.migrate_compression._sqlite_db import table_exists
from tests.conftest import requires_numpy

_PRE_FINGERPRINT_PROVENANCE_DDL = '''
CREATE TABLE IF NOT EXISTS compression_metadata (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    provider TEXT NOT NULL,
    bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),
    variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),
    seed INTEGER NOT NULL CHECK (seed >= 0),
    dim INTEGER NOT NULL CHECK (dim > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
'''


def _provenance_columns(path: Path) -> list[str]:
    """Return the column names of the ``compression_metadata`` table at ``path``.

    Returns:
        Column names in declaration order.
    """
    conn = sqlite3.connect(str(path))
    try:
        return [str(row[1]) for row in conn.execute('PRAGMA table_info(compression_metadata)')]
    finally:
        conn.close()


@requires_numpy
def test_compress_upgrades_provenance_table_missing_fingerprint_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--compress`` adds ``codebook_fingerprint`` to a table created before the column.

    A database whose ``compression_metadata`` table was created by a build predating
    the column, and whose provenance ROW was later cleared by ``--decompress`` (which
    deletes the row but keeps the table), reaches ``--compress`` with a six-column
    table. ``CREATE TABLE IF NOT EXISTS`` never adds a column, so without an idempotent
    add the seven-column provenance INSERT would fail with 'no such column' AFTER the
    whole encode pass -- a dead end, because re-running fails identically while a
    compression-enabled boot refuses to start before the server migration can add the
    column. The CLI mirrors that migration's idempotent add.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '64')
    get_settings.cache_clear()

    db = tmp_path / 'pre_fingerprint.db'
    bootstrap_schema(db)
    create_fp32_vec_table(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(_PRE_FINGERPRINT_PROVENANCE_DDL)
        conn.commit()
    finally:
        conn.close()
    assert 'codebook_fingerprint' not in _provenance_columns(db)

    rc = run_compress(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    assert 'codebook_fingerprint' in _provenance_columns(db)

    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            'SELECT bits, variant, seed, dim, codebook_fingerprint '
            'FROM compression_metadata WHERE id = 1',
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert (row[0], row[1], row[2], row[3]) == (4, 'ip', 42, 64)
    # The realized rotation digest is recorded, not left NULL.
    assert isinstance(row[4], str)
    assert len(row[4]) == 64


@requires_numpy
def test_compress_leaves_current_shape_provenance_table_intact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The idempotent add is a no-op when the table already carries the column."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '64')
    get_settings.cache_clear()

    db = tmp_path / 'current_shape.db'
    bootstrap_schema(db)
    create_fp32_vec_table(db)

    rc = run_compress(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    columns = _provenance_columns(db)
    assert columns.count('codebook_fingerprint') == 1


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
    monkeypatch.setattr(storage, 'MIGRATION_BATCH_SIZE', 8)
    seed_fp32_database(db, monkeypatch, n_docs=n_docs)
    enable_compression(monkeypatch)

    rc = run_compress(f'sqlite:///{db}', dry_run=False)
    assert rc == 0
    assert count_compressed(db) == n_docs
    assert not table_exists(db, 'vec_context_embeddings')
    assert count_provenance(db) == 1


@pytest.mark.integration
def test_streaming_compress_idempotent_no_op_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second --compress invocation is a no-op via the singleton check."""
    db = tmp_path / 'idempotent.db'
    n_docs = 6
    monkeypatch.setattr(storage, 'MIGRATION_BATCH_SIZE', 4)
    seed_fp32_database(db, monkeypatch, n_docs=n_docs)
    enable_compression(monkeypatch)

    assert run_compress(f'sqlite:///{db}', dry_run=False) == 0
    after_first = count_compressed(db)
    assert after_first == n_docs

    rc2 = run_compress(f'sqlite:///{db}', dry_run=False)
    assert rc2 == 0
    assert count_compressed(db) == after_first
    assert count_provenance(db) == 1


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
    monkeypatch.setattr(storage, 'MIGRATION_BATCH_SIZE', 16)
    seed_fp32_database(db, monkeypatch, n_docs=n_docs)
    enable_compression(monkeypatch)

    from app.compression import factory as compression_factory

    real_provider = compression_factory.create_compression_provider()
    original_encode = real_provider.encode_sync
    call_count = {'n': 0}
    # Probe phase encodes min(PROBE_BATCH_SIZE, n_docs) rows. Allow the
    # probe to complete plus 20 rows of the actual streaming loop before
    # injecting the failure.
    probe_rows = min(console.PROBE_BATCH_SIZE, n_docs)
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
    assert table_exists(db, 'vec_context_embeddings')
    assert count_fp32(db) == n_docs
    if table_exists(db, 'vec_context_embeddings_compressed'):
        assert count_compressed(db) == 0
    assert count_provenance(db) == 0


@pytest.mark.integration
def test_streaming_compress_aborts_on_orphan_fp32_row_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An fp32 row with no embedding_chunks bridge aborts compress (no data dropped).

    The streamed read joins embedding_chunks to vec_context_embeddings, so an
    orphan vec row would be skipped and then destroyed by the source DROP. The
    integrity guard must abort the transaction instead, leaving the source intact.
    """
    db = tmp_path / 'orphan.db'
    seed_fp32_database(db, monkeypatch, n_docs=3)

    # Inject an ORPHAN vec row: present in vec_context_embeddings but with NO
    # embedding_chunks bridge pointing at it (the streamed inner join skips it).
    orphan_blob = struct.pack(f'<{DIM}f', *([0.1] * DIM))
    conn = sqlite3.connect(str(db))
    try:
        conn.execute('INSERT INTO vec_context_embeddings (embedding) VALUES (?)', (orphan_blob,))
        conn.commit()
    finally:
        conn.close()

    enable_compression(monkeypatch)

    rc = run_compress(f'sqlite:///{db}', dry_run=False)
    # The guard aborts (non-zero) rather than silently dropping the orphan.
    assert rc != 0
    # Source preserved intact: all four fp32 rows still present, no provenance.
    assert table_exists(db, 'vec_context_embeddings')
    assert count_fp32(db) == 4
    assert count_provenance(db) == 0


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
    monkeypatch.setattr(storage, 'MIGRATION_BATCH_SIZE', batch_size)

    def _measure_peak(n_docs: int, db_name: str) -> int:
        db = tmp_path / db_name
        seed_fp32_database(db, monkeypatch, n_docs=n_docs)
        enable_compression(monkeypatch)
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


def test_compress_postgresql_runs_heavy_ddl_under_migration_budget() -> None:
    conn = RecordingConn()
    backend = FakeBackend(conn)
    asyncio.run(
        _execute_compress_postgresql(
            cast(Any, backend),
            Mock(),
            sample_provenance(),
        ),
    )

    # The transaction raises its server-side statement budget.
    set_local = [stmt for stmt, _ in conn.execute_calls if stmt.startswith('SET LOCAL statement_timeout')]
    assert set_local == [budget_set_local()]

    expected = expected_client_timeout()
    # The destructive DDL carries the explicit client-side deadline, not the bare pool timeout.
    drop_table = timeouts_for(conn, 'DROP TABLE IF EXISTS vec_context_embeddings')
    assert drop_table
    assert all(t == expected for t in drop_table)
    drop_index = timeouts_for(conn, 'DROP INDEX IF EXISTS idx_vec_context_embeddings_hnsw')
    assert drop_index
    assert all(t == expected for t in drop_index)
    # The pre-stream source freeze runs under the migration budget too.
    lock_stmts = timeouts_for(conn, 'LOCK TABLE vec_context_embeddings IN ACCESS EXCLUSIVE MODE')
    assert lock_stmts
    assert all(t == expected for t in lock_stmts)
    # The streamed batch read is budgeted too.
    assert conn.fetch_calls
    assert all(t == expected for _, t in conn.fetch_calls)
    # The under-lock COUNT(*) recount is budgeted too (not the bare pool timeout):
    # it is the first full-table scan under the lock on a large corpus.
    recount = [t for stmt, t in conn.fetchval_calls if 'COUNT(*) FROM vec_context_embeddings' in stmt]
    assert recount
    assert all(t == expected for t in recount)
    # Sanity: the empty-table CREATE statements stay bare (timeout None), so the
    # recording distinguishes budgeted statements from un-budgeted ones.
    assert any(t is None for stmt, t in conn.execute_calls if stmt.startswith('CREATE TABLE'))


def test_compress_postgresql_adds_missing_fingerprint_column_before_insert() -> None:
    """The compress transaction upgrades a pre-fingerprint provenance table in place.

    Without the idempotent ADD COLUMN, a ``compression_metadata`` table created by a
    build predating ``codebook_fingerprint`` -- whose row a prior ``--decompress``
    cleared, so the CLI does not early-return -- survives the ``CREATE TABLE IF NOT
    EXISTS`` unchanged and the six-parameter provenance INSERT fails with
    ``UndefinedColumnError`` after the whole encode pass has already run.
    """
    conn = RecordingConn()
    backend = FakeBackend(conn)
    asyncio.run(
        _execute_compress_postgresql(
            cast(Any, backend),
            Mock(),
            sample_provenance(),
        ),
    )

    statements = [stmt for stmt, _ in conn.execute_calls]
    create_idx = next(
        i for i, stmt in enumerate(statements)
        if stmt.startswith('CREATE TABLE IF NOT EXISTS compression_metadata')
    )
    alter_idx = next(
        i for i, stmt in enumerate(statements)
        if stmt.startswith('ALTER TABLE compression_metadata')
    )
    insert_idx = next(
        i for i, stmt in enumerate(statements)
        if stmt.startswith('INSERT INTO compression_metadata')
    )
    assert statements[alter_idx] == (
        'ALTER TABLE compression_metadata ADD COLUMN IF NOT EXISTS codebook_fingerprint TEXT'
    )
    assert create_idx < alter_idx < insert_idx
