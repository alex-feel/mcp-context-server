"""End-to-end migration CLI tests on SQLite.

Exercises ``mcp-context-server-migrate --compress`` and ``--decompress``
against a freshly-created SQLite database that contains fp32 (or
compressed) vectors written through the production storage backend.

The tests bypass the spawned-server harness used elsewhere in this
directory because the CLI runs inline (no MCP transport) and the
required setup (planted vectors) is faster to do directly.
"""

import asyncio
import contextlib
import sqlite3
import struct
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from app.backends import create_backend
from app.cli.migrate_compression import run_compress
from app.cli.migrate_compression import run_decompress
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository import _reset_compression_cache
from app.settings import get_settings

DIM = 1024

_ALLOWED_COUNT_TABLES = {
    'vec_context_embeddings_compressed',
    'compression_metadata',
}


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
    n_docs: int = 3,
) -> list[np.ndarray]:
    """Write ``n_docs`` fp32 vectors to an isolated SQLite database.

    Creates the schema directly (without relying on the sqlite-vec
    virtual table) because the CLI's encode/decode pass talks to the
    ``embedding_chunks`` -> ``vec_context_embeddings`` join, and we need
    a plain physical table compatible with sqlite_vec's
    ``serialize_float32`` BLOB layout.

    Args:
        db_path: Path the SQLite database will be created at.
        monkeypatch: Test-scope env manipulator.
        n_docs: Number of documents to seed.

    Returns:
        The list of float32 unit vectors that were stored.
    """
    monkeypatch.setenv('DB_PATH', str(db_path))
    monkeypatch.setenv('STORAGE_BACKEND', 'sqlite')
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    monkeypatch.delenv('ENABLE_SEMANTIC_SEARCH', raising=False)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.delenv('COMPRESSION_SEED', raising=False)
    get_settings.cache_clear()

    async def _setup() -> list[np.ndarray]:
        from app.schemas import load_schema

        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(load_schema('sqlite'))
            # Create the embedding tables manually. The CLI's read path
            # only needs (id, embedding) on vec_context_embeddings and a
            # standard embedding_chunks mapping; we skip the vec0 virtual
            # table so sqlite-vec extension loading is unnecessary.
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
            rng = np.random.default_rng(42)
            planted: list[np.ndarray] = []
            for i in range(n_docs):
                vec = rng.standard_normal(DIM).astype(np.float32)
                vec /= np.linalg.norm(vec)
                planted.append(vec)
                cid, _ = await repos.context.store_with_deduplication(
                    thread_id='compress-e2e',
                    source='user',
                    content_type='text',
                    text_content=f'doc-{i}',
                    metadata=None,
                )

                # Serialize the float32 vector with sqlite_vec's wire
                # format so the CLI's read path can deserialize it.
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
            return planted
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    return asyncio.run(_setup())


def _table_exists_sqlite(db_path: Path, table_name: str) -> bool:
    """Return True when ``table_name`` exists in the SQLite database."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def _read_provenance(db_path: Path) -> tuple[str, int, str, int, int] | None:
    """Return the singleton compression_metadata row, or None."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            'SELECT provider, bits, variant, seed, dim '
            'FROM compression_metadata WHERE id = 1',
        )
        return cur.fetchone()
    finally:
        conn.close()


def _count_table(db_path: Path, table_name: str) -> int:
    """Return COUNT(*) for ``table_name``; assumes the table exists.

    The table name is restricted to the test allow-list so the SQL
    construction is safe from injection.

    Args:
        db_path: SQLite database path.
        table_name: Name of the table to count.

    Returns:
        Number of rows in the table.

    Raises:
        ValueError: If ``table_name`` is not in the test allow-list.
    """
    if table_name not in _ALLOWED_COUNT_TABLES:
        raise ValueError(f'table {table_name!r} not in count allow-list')
    conn = sqlite3.connect(str(db_path))
    try:
        return int(
            conn.execute(
                f'SELECT COUNT(*) FROM {table_name}',
            ).fetchone()[0],
        )
    finally:
        conn.close()


@pytest.mark.integration
def test_compress_e2e_sqlite_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--compress --dry-run`` previews without mutating the source db."""
    db = tmp_path / 'compress_dry.db'
    _seed_fp32_database(db, monkeypatch, n_docs=3)
    # Sanity check: the seed populated the vec table.
    sanity_conn = sqlite3.connect(str(db))
    try:
        n_seeded = sanity_conn.execute(
            'SELECT COUNT(*) FROM vec_context_embeddings',
        ).fetchone()[0]
    finally:
        sanity_conn.close()
    assert n_seeded == 3, f'seed only wrote {n_seeded} rows'

    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    get_settings.cache_clear()

    rc = run_compress(f'sqlite:///{db}', dry_run=True)

    assert rc == 0
    err = capsys.readouterr().err
    assert 'BACKUP REQUIRED' in err
    assert '[DRY-RUN]' in err

    # Source fp32 table still present.
    assert _table_exists_sqlite(db, 'vec_context_embeddings')
    # Singleton row NOT written by the dry-run.
    if _table_exists_sqlite(db, 'compression_metadata'):
        assert _count_table(db, 'compression_metadata') == 0


@pytest.mark.integration
def test_compress_e2e_sqlite_execute_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--compress`` swaps the schema and is safe to re-run."""
    db = tmp_path / 'compress_exec.db'
    planted = _seed_fp32_database(db, monkeypatch, n_docs=3)

    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    get_settings.cache_clear()

    rc = run_compress(f'sqlite:///{db}', dry_run=False)
    assert rc == 0

    # Schema swap: compressed table populated, legacy table gone.
    assert _table_exists_sqlite(db, 'vec_context_embeddings_compressed')
    assert not _table_exists_sqlite(db, 'vec_context_embeddings')
    assert _count_table(db, 'vec_context_embeddings_compressed') == len(planted)

    # Provenance row matches the env-derived configuration.
    row = _read_provenance(db)
    assert row is not None
    provider, bits, variant, seed, dim = row
    assert provider == 'turboquant'
    assert bits == 4
    assert variant == 'ip'
    assert seed == 42
    assert dim == DIM

    # Drain the first run's captured output.
    capsys.readouterr()

    # Idempotency: second --compress no-ops.
    rc2 = run_compress(f'sqlite:///{db}', dry_run=False)
    assert rc2 == 0
    err2 = capsys.readouterr().err
    assert 'already present' in err2


@pytest.mark.integration
def test_decompress_e2e_sqlite_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--decompress`` restores the fp32 schema after a compress pass."""
    db = tmp_path / 'decompress_exec.db'
    planted = _seed_fp32_database(db, monkeypatch, n_docs=3)

    # First compress.
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    get_settings.cache_clear()
    assert run_compress(f'sqlite:///{db}', dry_run=False) == 0
    assert _table_exists_sqlite(db, 'vec_context_embeddings_compressed')
    assert not _table_exists_sqlite(db, 'vec_context_embeddings')

    # Then decompress. The fp32 vec table is a sqlite-vec virtual table,
    # so we need ENABLE_SEMANTIC_SEARCH=true to ensure the backend loads
    # the extension when re-opening the database.
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', 'true')
    get_settings.cache_clear()
    _reset_compression_cache()
    # Refresh the cached settings binding the SQLite backend reads at
    # connection time so semantic_search.enabled becomes true for the
    # new connection.
    import app.backends.sqlite_backend as sqlite_backend_module
    monkeypatch.setattr(
        sqlite_backend_module, 'settings', get_settings(),
    )

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)
    assert rc == 0

    # Schema restored.
    assert _table_exists_sqlite(db, 'vec_context_embeddings')
    assert not _table_exists_sqlite(db, 'vec_context_embeddings_compressed')
    # Singleton row removed.
    assert _read_provenance(db) is None

    # Confirm the row count round-tripped: the chunking mapping in
    # ``embedding_chunks`` still has one entry per planted document. The
    # actual embedding bytes live in the sqlite-vec virtual table whose
    # vec0 module is only loaded when ENABLE_SEMANTIC_SEARCH is true in
    # the same process that opens the connection -- a plain sqlite3
    # connection here cannot decode the embedding column, so we limit
    # the verification to the structural check.
    conn = sqlite3.connect(str(db))
    try:
        count = conn.execute(
            'SELECT COUNT(*) FROM embedding_chunks',
        ).fetchone()[0]
    finally:
        conn.close()
    assert int(count) == len(planted)
