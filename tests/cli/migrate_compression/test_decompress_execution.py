"""Tests for the transactional ``--decompress`` data movement on SQLite and PostgreSQL.

The SQLite tests round-trip a multi-batch corpus and recover a database the server compressed, which carries no
``embedding_chunks`` rows. The PostgreSQL tests check that heavy statements run under the migration budget and that
the source-presence probe resolves the bare table name through the connection ``search_path`` (``to_regclass``),
exactly like the bare-name DML and the trailing ``DROP TABLE`` it gates: a probe pinned to the configured
``POSTGRESQL_SCHEMA`` would report a table living in ``public`` as absent while the bare-name SQL keeps reaching it.
The operator-controlled schema value must also never be composed into the SQL string.
"""

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

from app.backends import StorageBackend
from app.backends import create_backend
from app.backends.base import TransactionContext
from app.cli.migrate_compression import storage
from app.cli.migrate_compression.compress import run_compress
from app.cli.migrate_compression.decompress import run_decompress
from app.cli.migrate_compression.decompress_execution import _execute_decompress_postgresql
from app.compression.base import CompressionProvider
from app.compression.types import CompressionMetadata
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository.compression_cache import _reset_compression_cache
from app.settings import get_settings
from tests.cli.migrate_compression._pg_fakes import FakeBackend
from tests.cli.migrate_compression._pg_fakes import RecordingConn
from tests.cli.migrate_compression._pg_fakes import budget_set_local
from tests.cli.migrate_compression._pg_fakes import expected_client_timeout
from tests.cli.migrate_compression._pg_fakes import sample_provenance
from tests.cli.migrate_compression._pg_fakes import timeouts_for
from tests.cli.migrate_compression._sqlite_db import DIM
from tests.cli.migrate_compression._sqlite_db import count_compressed
from tests.cli.migrate_compression._sqlite_db import count_provenance
from tests.cli.migrate_compression._sqlite_db import enable_compression
from tests.cli.migrate_compression._sqlite_db import seed_fp32_database
from tests.cli.migrate_compression._sqlite_db import table_exists
from tests.helpers import LOCAL_SCOPE


@pytest.mark.integration
def test_streaming_decompress_roundtrip_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-batch compress then decompress restores the original row count."""
    db = tmp_path / 'roundtrip.db'
    n_docs = 12
    monkeypatch.setattr(storage, 'MIGRATION_BATCH_SIZE', 5)
    seed_fp32_database(db, monkeypatch, n_docs=n_docs)
    enable_compression(monkeypatch)

    assert run_compress(f'sqlite:///{db}', dry_run=False) == 0
    assert count_compressed(db) == n_docs

    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    _reset_compression_cache()

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)
    assert rc == 0
    assert table_exists(db, 'vec_context_embeddings')
    assert not table_exists(db, 'vec_context_embeddings_compressed')
    assert count_provenance(db) == 0
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


def _seed_compressed_database(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_docs: int,
) -> None:
    """Write ``n_docs`` server-compressed rows to an isolated DB.

    Mirrors the LIVE compressed write path (``_store_chunked_compressed``): one
    ``vec_context_embeddings_compressed`` row per chunk with a per-context
    sequential ``chunk_index``, NO ``embedding_chunks`` rows, and NO fp32
    ``vec_context_embeddings`` table -- the default v3 deployment shape. The
    singleton ``compression_metadata`` provenance row matches the IP-4 seed-42
    encode so ``run_decompress`` reconstructs the same provider.
    """
    monkeypatch.setenv('DB_PATH', str(db_path))
    monkeypatch.setenv('STORAGE_BACKEND', 'sqlite')
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    monkeypatch.delenv('ENABLE_SEMANTIC_SEARCH', raising=False)
    enable_compression(monkeypatch)

    async def _setup() -> None:
        from app.compression import factory as compression_factory
        from app.schemas import load_schema

        provider = compression_factory.create_compression_provider()
        settings = get_settings()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(load_schema('sqlite'))
            conn.executescript(
                '''
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
                CREATE TABLE IF NOT EXISTS compression_metadata (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    provider TEXT NOT NULL,
                    bits INTEGER NOT NULL,
                    variant TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    dim INTEGER NOT NULL,
                    codebook_fingerprint TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                ''',
            )
        finally:
            conn.close()

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            repos = RepositoryContainer(backend)
            rng = np.random.default_rng(11)
            for i in range(n_docs):
                vec = rng.standard_normal(DIM).astype(np.float32)
                vec /= np.linalg.norm(vec)
                cid, _ = await repos.context.store_with_deduplication(
                    scope=LOCAL_SCOPE,
                    visibility='private',
                    thread_id='server-compressed',
                    source='user',
                    content_type='text',
                    text_content=f'doc-{i}',
                    metadata=None,
                )
                payload = provider.encode_sync(vec.reshape(1, DIM))

                def _write(
                    conn: sqlite3.Connection,
                    *,
                    context_id: str = cid,
                    blob: bytes = payload,
                ) -> None:
                    # chunk_index = 0 (single chunk per context); the live path
                    # never writes embedding_chunks.
                    conn.execute(
                        'INSERT INTO vec_context_embeddings_compressed '
                        '(context_id, chunk_index, start_index, end_index, payload) '
                        'VALUES (?, ?, ?, ?, ?)',
                        (context_id, 0, 0, DIM, blob),
                    )
                    conn.execute(
                        'INSERT INTO embedding_metadata '
                        '(context_id, model_name, dimensions, chunk_count) '
                        'VALUES (?, ?, ?, ?)',
                        (context_id, 'test-model', DIM, 1),
                    )

                await backend.execute_write(_write)

            def _write_provenance(conn: sqlite3.Connection) -> None:
                conn.execute(
                    'INSERT INTO compression_metadata '
                    '(id, provider, bits, variant, seed, dim) '
                    'VALUES (1, ?, ?, ?, ?, ?)',
                    (
                        settings.compression.provider,
                        settings.compression.bits,
                        settings.compression.variant,
                        settings.compression.seed,
                        DIM,
                    ),
                )

            await backend.execute_write(_write_provenance)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_setup())


def _count_fp32_vec0(db_path: Path) -> int:
    """Count rows in the vec0 virtual ``vec_context_embeddings`` table."""
    import sqlite_vec

    conn = sqlite3.connect(str(db_path))
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return int(
            conn.execute('SELECT COUNT(*) FROM vec_context_embeddings').fetchone()[0],
        )
    finally:
        conn.close()


@pytest.mark.integration
def test_streaming_decompress_recovers_server_compressed_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--decompress recovers a server-compressed database.

    A server-compressed-from-start database -- the v3 default -- has NO
    ``embedding_chunks`` rows and ``chunk_index=0`` per context. A reverse loop
    that looked up a pre-existing ``embedding_chunks`` row by ``(context_id,
    chunk_index)`` and ``continue``d on a miss would skip EVERY row, then DROP the
    compressed source and DELETE provenance: zero embeddings recovered with a
    success exit code. Decompress therefore rebuilds both the fp32 table and the
    ``embedding_chunks`` bridge directly from the compressed rows, so every
    embedding is recovered.
    """
    db = tmp_path / 'server_compressed.db'
    n_docs = 12
    monkeypatch.setattr(storage, 'MIGRATION_BATCH_SIZE', 5)
    _seed_compressed_database(db, monkeypatch, n_docs=n_docs)
    assert count_compressed(db) == n_docs

    # Reverse compression (abandon-compression flow): run_decompress refuses to
    # start while ENABLE_EMBEDDING_COMPRESSION is enabled; the provider is
    # reconstructed from the provenance row, not the env.
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    _reset_compression_cache()

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)
    assert rc == 0
    assert table_exists(db, 'vec_context_embeddings')
    assert not table_exists(db, 'vec_context_embeddings_compressed')
    assert count_provenance(db) == 0

    # The core assertions: every embedding recovered into BOTH the fp32 vec
    # table and the rebuilt embedding_chunks bridge.
    assert _count_fp32_vec0(db) == n_docs
    import sqlite_vec

    conn = sqlite3.connect(str(db))
    try:
        # The bridge join touches the vec0 virtual table, so load the extension.
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        chunk_count = int(
            conn.execute('SELECT COUNT(*) FROM embedding_chunks').fetchone()[0],
        )
        # Bridge consistency: every embedding_chunks row points at a real fp32
        # rowid (no orphaned vec_rowid).
        orphans = int(
            conn.execute(
                'SELECT COUNT(*) FROM embedding_chunks ec '
                'LEFT JOIN vec_context_embeddings ve ON ve.rowid = ec.vec_rowid '
                'WHERE ve.rowid IS NULL',
            ).fetchone()[0],
        )
    finally:
        conn.close()
    assert chunk_count == n_docs
    assert orphans == 0


class _FetchvalRecorder:
    """Capture the SQL and bind parameters passed to ``fetchval``.

    Mimics the asyncpg ``Connection.fetchval`` surface used by
    ``_execute_decompress_postgresql``: an async callable that accepts a
    SQL string and any number of positional bind parameters, returns a
    probe result, and records the (sql, params) tuple for assertion.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def __call__(self, sql: str, *params: object) -> bool:
        self.calls.append((sql, params))
        # Report the source table as absent so the function takes the
        # early-return branch and avoids subsequent SQL that requires a
        # real connection.
        return False


class _StubConnection:
    """Stand-in for ``asyncpg.Connection`` covering only the methods
    invoked before ``_execute_decompress_postgresql`` early-returns.
    """

    def __init__(self, fetchval: _FetchvalRecorder) -> None:
        self.fetchval = fetchval

    async def execute(self, sql: str, *params: object) -> None:
        # The two ``conn.execute`` calls (CREATE TABLE / CREATE INDEX)
        # run before the source-probe early-return. They produce no
        # observable side effects in this test.
        del sql, params


class _StubTxn:
    """Minimal :class:`TransactionContext` carrying the stub connection."""

    def __init__(self, conn: _StubConnection) -> None:
        self._conn = conn

    @property
    def connection(self) -> object:
        return self._conn

    @property
    def backend_type(self) -> str:
        return 'postgresql'


class _StubBackend:
    """Minimal backend exposing ``backend_type`` and ``begin_transaction``.

    Only the surface that ``_execute_decompress_postgresql`` touches is
    implemented; everything else stays unimplemented to keep the test
    scoped to the source-probe resolution invariant.
    """

    def __init__(self, txn: _StubTxn) -> None:
        self._txn = txn

    @property
    def backend_type(self) -> str:
        return 'postgresql'

    @asynccontextmanager
    async def begin_transaction(self) -> AsyncGenerator[TransactionContext, None]:
        yield cast(TransactionContext, self._txn)


@pytest.mark.asyncio
async def test_decompress_source_probe_resolves_via_search_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source-presence probe MUST use ``to_regclass``, not a schema pin.

    ``to_regclass`` resolves the bare name through the connection
    ``search_path`` (configured schema first, then ``public``) -- the same
    rule the streamed reads and the trailing ``DROP TABLE`` follow. The
    configured schema value must not appear anywhere in the SQL: an
    injection-shaped ``POSTGRESQL_SCHEMA`` proves the probe neither pins
    nor composes it.
    """
    fetchval = _FetchvalRecorder()
    conn = _StubConnection(fetchval=fetchval)
    txn = _StubTxn(conn=conn)
    backend = _StubBackend(txn=txn)

    # An injection-shaped schema: under f-string composition this would
    # corrupt the SQL; under a schema-pinned probe it would appear as a
    # bind parameter. The probe must show neither.
    monkeypatch.setenv('POSTGRESQL_SCHEMA', "evil'--")
    get_settings.cache_clear()
    try:
        provenance = CompressionMetadata(
            provider='turboquant', bits=4, variant='ip', seed=42, dim=1024,
        )

        # Provider is unused because the early-return branch fires before
        # any decode_sync call. Cast keeps the type checker honest at the
        # call site without an inline ignore.
        await _execute_decompress_postgresql(
            backend=cast(StorageBackend, backend),
            provider=cast(CompressionProvider, None),
            provenance=provenance,
            row_count=0,
        )
    finally:
        # Restore the default schema so subsequent tests get a clean cache.
        monkeypatch.delenv('POSTGRESQL_SCHEMA', raising=False)
        get_settings.cache_clear()

    probe_calls = [
        (sql, params) for sql, params in fetchval.calls
        if 'to_regclass' in sql
    ]
    assert len(probe_calls) == 1, (
        f'Expected exactly one to_regclass source probe; '
        f'captured={fetchval.calls}'
    )
    sql, params = probe_calls[0]
    assert "to_regclass('vec_context_embeddings_compressed')" in sql, (
        f'Probe must resolve the compressed source table by bare name. sql={sql!r}'
    )
    assert params == (), (
        f'The bare-name probe takes no bind parameters. params={params!r}'
    )
    # No schema-pinned lookup is issued, and the operator-controlled
    # schema value must not leak into ANY SQL string.
    assert all('information_schema' not in s for s, _ in fetchval.calls)
    assert all('evil' not in s for s, _ in fetchval.calls)


def test_decompress_postgresql_runs_heavy_ddl_under_migration_budget() -> None:
    conn = RecordingConn()
    backend = FakeBackend(conn)
    asyncio.run(
        _execute_decompress_postgresql(
            cast(Any, backend),
            Mock(),
            sample_provenance(),
            row_count=0,
        ),
    )

    set_local = [stmt for stmt, _ in conn.execute_calls if stmt.startswith('SET LOCAL statement_timeout')]
    assert set_local == [budget_set_local()]

    expected = expected_client_timeout()
    # The HNSW index build is the heaviest op and must run under the migration budget.
    create_hnsw = timeouts_for(conn, 'USING hnsw (embedding vector_l2_ops)')
    assert create_hnsw
    assert all(t == expected for t in create_hnsw)
    drop_compressed = timeouts_for(conn, 'DROP TABLE IF EXISTS vec_context_embeddings_compressed')
    assert drop_compressed
    assert all(t == expected for t in drop_compressed)
    assert conn.fetch_calls
    assert all(t == expected for _, t in conn.fetch_calls)
    # The under-lock COUNT(*) recount is budgeted too.
    recount = [t for stmt, t in conn.fetchval_calls if 'COUNT(*) FROM vec_context_embeddings_compressed' in stmt]
    assert recount
    assert all(t == expected for t in recount)
