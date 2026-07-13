"""Unit tests for the compression CLI handler.

These tests cover the CLI surface only: argparse flag handling,
``--dry-run`` output shape, idempotency logic, and env-var validation.
End-to-end coverage with real SQLite/PostgreSQL databases lives in
``tests/integration/sqlite/test_migrate_compress_e2e.py`` and
``tests/integration/postgresql/test_migrate_compress_e2e_postgresql.py``.
"""

import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest import mock

import pytest

from app.cli.migrate import build_parser
from app.cli.migrate import main as cli_main
from app.cli.migrate_compression import WARNING_BORDER
from app.cli.migrate_compression import run_compress
from app.cli.migrate_compression import run_decompress
from app.compression.types import CompressionMetadata
from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT
from app.repositories.embedding_repository import _reset_compression_cache
from app.settings import get_settings
from tests.conftest import requires_numpy


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset settings + compression caches around every test."""
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()


def _bootstrap_schema(path: Path) -> None:
    """Apply the SQLite schema to ``path``."""
    from app.schemas import load_schema

    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(load_schema('sqlite'))
        # The semantic-search migration owns embedding_metadata in
        # production; recreate the table directly so the CLI's structural
        # checks pass without dragging in sqlite-vec.
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                context_id TEXT NOT NULL PRIMARY KEY,
                model_name TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS embedding_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                vec_rowid INTEGER NOT NULL,
                start_index INTEGER NOT NULL DEFAULT 0,
                end_index INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
            );
            ''',
        )
    finally:
        conn.close()


def _create_fp32_vec_table(path: Path) -> None:
    """Create a minimal stand-in for the fp32 ``vec_context_embeddings`` table.

    The real table is a sqlite-vec virtual table; the CLI's preflight check
    just verifies the name exists, so a plain table with the same column
    name is sufficient for flag-handling tests that do NOT execute the
    encode pass.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS vec_context_embeddings (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB
            );
            ''',
        )
    finally:
        conn.close()


def test_build_parser_accepts_compress_flag() -> None:
    """``--compress`` is parsed as a boolean toggle."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--compress',
    ])
    assert args.compress is True
    assert args.decompress is False


def test_build_parser_accepts_decompress_flag() -> None:
    """``--decompress`` is parsed as a boolean toggle."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--decompress',
    ])
    assert args.decompress is True
    assert args.compress is False


def test_build_parser_rejects_both_flags() -> None:
    """``--compress`` and ``--decompress`` are mutually exclusive."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            '--source-url', 'sqlite:///fake.db',
            '--compress', '--decompress',
        ])


def test_build_parser_target_url_optional_when_compress() -> None:
    """``--target-url`` is no longer required when --compress is set."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--compress',
    ])
    assert args.target_url is None


def test_main_requires_compression_enabled_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_compress`` exits 1 when ENABLE_EMBEDDING_COMPRESSION is unset."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--compress'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'ENABLE_EMBEDDING_COMPRESSION=true' in err
    assert 'BACKUP REQUIRED' in err  # warning was printed before validation


def test_main_rejects_decompress_when_compression_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_decompress`` exits 1 when ENABLE_EMBEDDING_COMPRESSION is true."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)

    assert rc == 1
    err = capsys.readouterr().err
    assert 'must be unset' in err


def test_main_compress_aborts_when_fp32_table_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_compress`` exits 1 when the fp32 source table is absent."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)  # schema only; no vec_context_embeddings table

    rc = run_compress(f'sqlite:///{db}', dry_run=True)

    assert rc == 1
    err = capsys.readouterr().err
    assert 'vec_context_embeddings not present' in err


def test_main_compress_dry_run_prints_plan_for_empty_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dry-run prints the plan and exits 0 even when the fp32 table is empty."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)
    _create_fp32_vec_table(db)

    rc = run_compress(f'sqlite:///{db}', dry_run=True)

    assert rc == 0
    err = capsys.readouterr().err
    assert 'BACKUP REQUIRED' in err
    assert WARNING_BORDER in err
    assert '[DRY-RUN]' in err
    assert 'from_table:    vec_context_embeddings' in err
    assert 'to_table:      vec_context_embeddings_compressed' in err

    # Verify dry-run made no destructive changes.
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            )
        }
    finally:
        conn.close()
    assert 'vec_context_embeddings' in tables
    # compression_metadata is created by the migration that the dry-run
    # also applies (idempotent CREATE IF NOT EXISTS); a dry-run does not
    # roll the migration back -- the migration is non-destructive.
    # The singleton row, however, is NOT inserted by the dry run.
    cursor = conn = sqlite3.connect(str(db))
    try:
        if 'compression_metadata' in tables:
            count = cursor.execute(
                'SELECT COUNT(*) FROM compression_metadata',
            ).fetchone()[0]
            assert count == 0
    finally:
        cursor.close()


def test_main_compress_idempotent_when_already_compressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Running ``--compress`` twice no-ops with an informational message."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    # Pre-seed the database with the compressed schema + singleton row so
    # the second --compress call recognizes "already compressed" state.
    conn = sqlite3.connect(str(db))
    try:
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
            INSERT INTO compression_metadata
            (id, provider, bits, variant, seed, dim)
            VALUES (1, 'turboquant', 4, 'ip', 42, 1024);
            ''',
        )
    finally:
        conn.close()

    rc = run_compress(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    err = capsys.readouterr().err
    assert 'already present' in err
    assert 'bits=4 variant=ip dim=1024 seed=42' in err


def test_main_decompress_noop_when_nothing_to_do(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--decompress`` no-ops when the source is already fp32-only."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)
    _create_fp32_vec_table(db)

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Nothing to do' in err


def test_main_decompress_errors_when_provenance_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--decompress`` exits 1 when the provenance row is absent but the
    compressed table exists (corrupt state)."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    conn = sqlite3.connect(str(db))
    try:
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

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)

    assert rc == 1
    err = capsys.readouterr().err
    assert 'compression_metadata row missing' in err


@requires_numpy
def test_decompress_zero_rows_drops_table_without_fp32_infrastructure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--decompress`` with ZERO compressed rows drops the table and clears the row.

    A deployment running with ``ENABLE_EMBEDDING_GENERATION=false`` could
    carry the compression schema and a provenance row while embedding_chunks,
    sqlite-vec, and any fp32 vec table were never provisioned. The full
    reverse path crashes there (the vec0 ``CREATE VIRTUAL TABLE``, the
    ``DELETE FROM embedding_chunks``), wedging the disable direction behind
    its own prescribed remedy. The zero-data path must instead drop the empty
    compressed table and delete the provenance row WITHOUT touching fp32
    infrastructure, so the next startup proceeds cleanly.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    # Base schema ONLY -- deliberately NOT _bootstrap_schema: a generation-off
    # deployment never provisioned embedding_chunks or embedding_metadata, and
    # the zero-data reverse path must not require them.
    from app.schemas import load_schema

    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(load_schema('sqlite'))
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
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            "VALUES (1, 'turboquant', 4, 'ip', 0, 1024, NULL)",
        )
        conn.commit()
    finally:
        conn.close()

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Decompression complete' in err
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            )
        }
        assert 'vec_context_embeddings_compressed' not in tables
        # No fp32 infrastructure may be provisioned by the zero-data path.
        assert 'vec_context_embeddings' not in tables
        assert 'embedding_chunks' not in tables
        count = conn.execute('SELECT COUNT(*) FROM compression_metadata').fetchone()[0]
        assert count == 0
    finally:
        conn.close()


@requires_numpy
def test_decompress_aborts_on_codebook_fingerprint_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--decompress`` aborts (rc 1) BEFORE any decode/DROP on a fingerprint mismatch.

    Regression: the CLI rebuilt the provider from the stored (dim, seed, bits,
    variant) but never re-derived and compared the realized codebook fingerprint.
    A cross-host numpy.linalg.qr divergence would silently corrupt every
    reconstructed vector and then DROP the only correctly-decodable copy. The CLI
    now re-derives the fingerprint and aborts on mismatch, mirroring the startup
    validator. Here the stored fingerprint is a deliberately wrong value, so the
    realized digest cannot match and decompress must refuse without dropping. A
    compressed row is present because the gate applies only when rows exist --
    the zero-data reverse path decodes nothing and bypasses it.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    wrong_fingerprint = 'deadbeef' * 8  # 64 hex chars, never the realized QR digest
    conn = sqlite3.connect(str(db))
    try:
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
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            "VALUES (1, 'turboquant', 4, 'ip', 0, 512, ?)",
            (wrong_fingerprint,),
        )
        conn.execute(
            'INSERT INTO vec_context_embeddings_compressed '
            '(context_id, chunk_index, start_index, end_index, payload) '
            'VALUES (?, 0, 0, 10, ?)',
            ('0' * 32, b'\x00'),
        )
        conn.commit()
    finally:
        conn.close()

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)

    assert rc == 1
    err = capsys.readouterr().err
    assert 'fingerprint mismatch' in err
    # The compressed source MUST still exist -- nothing decoded or dropped.
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()
    assert 'vec_context_embeddings_compressed' in tables


@requires_numpy
def test_zero_data_decompress_bypasses_fingerprint_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Zero compressed rows unwedge even on a codebook fingerprint mismatch.

    The fingerprint gate exists solely to prevent decoding corruption, and the
    zero-data reverse path decodes nothing. Blocking it on a cross-host QR
    divergence left the operator with no working escape: the server refuses to
    start on the identical divergence, and the mismatch error's own remedy
    (run --decompress on a reproducing host) is impossible advice when the
    goal is clearing an empty table. With zero rows the gate is bypassed and
    the reverse migration completes.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    from app.schemas import load_schema

    wrong_fingerprint = 'deadbeef' * 8  # 64 hex chars, never the realized QR digest
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(load_schema('sqlite'))
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
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            "VALUES (1, 'turboquant', 4, 'ip', 0, 512, ?)",
            (wrong_fingerprint,),
        )
        conn.commit()
    finally:
        conn.close()

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    err = capsys.readouterr().err
    assert 'fingerprint mismatch' not in err
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        count = conn.execute('SELECT COUNT(*) FROM compression_metadata').fetchone()[0]
    finally:
        conn.close()
    assert 'vec_context_embeddings_compressed' not in tables
    assert count == 0


@requires_numpy
@pytest.mark.asyncio
async def test_zero_data_decompress_aborts_when_rows_appear_before_drop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero-data drop re-checks emptiness inside its own transaction.

    The planning-time row count runs outside the drop transaction, so a
    compression-on server running concurrently can commit compressed rows in
    the gap; dropping the table then would destroy embeddings that were never
    decoded. Calling the drop helper against a table that already holds a row
    models exactly that interleaving: the in-transaction recount must abort
    and leave both the table and the provenance row intact.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    from app.schemas import load_schema

    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(load_schema('sqlite'))
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
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            "VALUES (1, 'turboquant', 4, 'ip', 0, 512, NULL)",
        )
        conn.execute(
            'INSERT INTO vec_context_embeddings_compressed '
            '(context_id, chunk_index, start_index, end_index, payload) '
            'VALUES (?, 0, 0, 10, ?)',
            ('0' * 32, b'\x00'),
        )
        conn.commit()
    finally:
        conn.close()

    from app.backends import create_backend
    from app.cli.migrate_compression import _execute_decompress_empty

    backend = create_backend(backend_type='sqlite', db_path=str(db))
    await backend.initialize()
    try:
        with pytest.raises(ValueError, match='concurrent writer'):
            await _execute_decompress_empty(backend)
    finally:
        await backend.shutdown()

    conn = sqlite3.connect(str(db))
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        payload_count = conn.execute(
            'SELECT COUNT(*) FROM vec_context_embeddings_compressed',
        ).fetchone()[0]
        provenance_count = conn.execute(
            'SELECT COUNT(*) FROM compression_metadata',
        ).fetchone()[0]
    finally:
        conn.close()
    assert 'vec_context_embeddings_compressed' in tables
    assert payload_count == 1
    assert provenance_count == 1


@pytest.mark.asyncio
async def test_zero_data_drop_is_transactional_on_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero-data DROP runs inside an explicit write transaction on SQLite.

    sqlite3's legacy transaction control opens the implicit transaction only
    before DML: without an explicit BEGIN IMMEDIATE the recount pinned no
    snapshot and the DROP ran in AUTOCOMMIT -- durable immediately -- so a
    failure between the DROP and the provenance DELETE left the database in
    {table dropped, provenance row present}, the exact wedge state the
    zero-data path exists to remove (the compression-off startup guard exits
    78 there and re-running --decompress refuses on the absent table). With
    the explicit transaction, the DROP executes with a transaction open and
    an injected failure after it rolls EVERYTHING back.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    from app.schemas import load_schema

    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(load_schema('sqlite'))
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
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            "VALUES (1, 'turboquant', 4, 'ip', 0, 512, NULL)",
        )
        conn.commit()
    finally:
        conn.close()

    from collections.abc import AsyncGenerator
    from contextlib import asynccontextmanager
    from typing import cast

    from app.backends import create_backend
    from app.backends.base import TransactionContext
    from app.cli.migrate_compression import _execute_decompress_empty

    backend = create_backend(backend_type='sqlite', db_path=str(db))
    await backend.initialize()
    drop_in_transaction: list[bool] = []
    try:
        real_begin = backend.begin_transaction

        class _FailAfterDropConn:
            """Delegate to the real connection, failing on the provenance DELETE."""

            def __init__(self, real: sqlite3.Connection) -> None:
                self._real = real

            def execute(self, sql: str) -> sqlite3.Cursor:
                """Record the DROP's transaction state; inject a failure on DELETE.

                The zero-data path issues only parameterless statements, so the
                proxy deliberately accepts a bare SQL string.

                Returns:
                    The real cursor for every pass-through statement.

                Raises:
                    sqlite3.OperationalError: On the provenance DELETE, to model
                        a crash between the DROP and the DELETE.
                """
                if sql.startswith('DROP TABLE'):
                    drop_in_transaction.append(self._real.in_transaction)
                if sql.startswith('DELETE FROM compression_metadata'):
                    raise sqlite3.OperationalError('injected failure between DROP and DELETE')
                return self._real.execute(sql)

        class _ProxyTxn:
            """Transaction context whose connection is the failing proxy."""

            def __init__(self, real: TransactionContext) -> None:
                self._real = real

            @property
            def connection(self) -> object:
                return _FailAfterDropConn(cast(sqlite3.Connection, self._real.connection))

            @property
            def backend_type(self) -> str:
                return self._real.backend_type

        @asynccontextmanager
        async def _proxied_begin() -> AsyncGenerator[TransactionContext, None]:
            """Wrap the real transaction so the body sees the failing proxy.

            Yields:
                The proxy transaction context delegating to the real one.
            """
            async with real_begin() as txn:
                yield cast(TransactionContext, _ProxyTxn(txn))

        monkeypatch.setattr(backend, 'begin_transaction', _proxied_begin)
        with pytest.raises(sqlite3.OperationalError, match='injected failure'):
            await _execute_decompress_empty(backend)
    finally:
        await backend.shutdown()

    # The DROP executed with the explicit transaction open...
    assert drop_in_transaction == [True]
    # ...so the injected failure rolled the whole operation back: the table
    # survives and the provenance row is intact (no wedge state).
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        provenance_count = conn.execute(
            'SELECT COUNT(*) FROM compression_metadata',
        ).fetchone()[0]
    finally:
        conn.close()
    assert 'vec_context_embeddings_compressed' in tables
    assert provenance_count == 1


# ---------------------------------------------------------------------------
# fp32 rebuild pre-flight for the pgvector index dimension cap
# ---------------------------------------------------------------------------


def _install_decompress_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend_type: str,
    dim: int,
    row_count: int,
) -> dict[str, mock.AsyncMock | mock.MagicMock]:
    """Fake the decompress pipeline around the fp32 dimension pre-flight.

    Installs a stub backend plus provenance/probe fakes in the
    ``app.cli.migrate_compression`` namespace so ``run_decompress`` reaches the
    dimension gate without a real database, pgvector host, or provider work.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        backend_type: Backend the stub reports ('sqlite' or 'postgresql').
        dim: Provenance dimension recorded in the faked metadata row.
        row_count: Compressed-row count the faked counter reports.

    Returns:
        Mapping with the 'execute_decompress' AsyncMock and the 'provider_for'
        MagicMock for call assertions.
    """
    import app.cli.migrate_compression as migrate_compression_mod

    backend = mock.MagicMock()
    backend.backend_type = backend_type
    backend.initialize = mock.AsyncMock()
    backend.shutdown = mock.AsyncMock()

    meta = CompressionMetadata(
        provider='turboquant', bits=4, variant='ip', seed=0, dim=dim,
    )

    async def _fake_needs_vector(_address: str) -> bool:
        return row_count > 0

    def _fake_make_backend(
        _source_url: str, *, provision_vector: bool | None = None,
    ) -> mock.MagicMock:
        del provision_vector
        return backend

    async def _fake_read_metadata(_backend: object) -> CompressionMetadata:
        return meta

    async def _fake_table_exists(_backend: object, table_name: str) -> bool:
        return table_name == 'vec_context_embeddings_compressed'

    async def _fake_count_table(_backend: object, _table_name: str) -> int:
        return row_count

    async def _fake_read_probe(_backend: object, _n: int) -> list[object]:
        return []

    execute_decompress = mock.AsyncMock()
    provider_for = mock.MagicMock()

    monkeypatch.setattr(migrate_compression_mod, '_decompress_needs_vector', _fake_needs_vector)
    monkeypatch.setattr(migrate_compression_mod, '_make_backend', _fake_make_backend)
    monkeypatch.setattr(migrate_compression_mod, 'read_compression_metadata', _fake_read_metadata)
    monkeypatch.setattr(migrate_compression_mod, '_table_exists', _fake_table_exists)
    monkeypatch.setattr(migrate_compression_mod, '_count_table', _fake_count_table)
    monkeypatch.setattr(migrate_compression_mod, '_read_compressed_probe', _fake_read_probe)
    monkeypatch.setattr(migrate_compression_mod, '_provider_for', provider_for)
    monkeypatch.setattr(migrate_compression_mod, '_execute_decompress', execute_decompress)
    return {'execute_decompress': execute_decompress, 'provider_for': provider_for}


def test_decompress_pg_refuses_dim_above_pgvector_index_cap_before_any_ddl(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """PostgreSQL --decompress refuses a provenance dim pgvector cannot index, before any DDL.

    Without the pre-flight the reverse migration streamed and decoded every
    compressed row and only then died at the trailing HNSW CREATE INDEX
    (pgvector caps indexable fp32 dimensionality at 2000): a rolled-back
    exit 2 after the full decode pass, with only a raw pgvector error to
    explain it.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    mocks = _install_decompress_fakes(
        monkeypatch,
        backend_type='postgresql',
        dim=PGVECTOR_INDEX_DIM_LIMIT + 1,
        row_count=7,
    )

    rc = run_decompress('postgresql://u:p@localhost:5432/ctx', dry_run=False)

    assert rc == 1
    err = capsys.readouterr().err
    assert 'pgvector index limit' in err
    assert 'must stay compressed' in err
    # Refused before any DDL or data streaming, and before provider construction.
    mocks['execute_decompress'].assert_not_awaited()
    mocks['provider_for'].assert_not_called()


def test_decompress_sqlite_dim_above_pg_cap_is_not_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dimension gate is PostgreSQL-scoped: sqlite-vec has no per-dimension index cap."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    mocks = _install_decompress_fakes(
        monkeypatch,
        backend_type='sqlite',
        dim=PGVECTOR_INDEX_DIM_LIMIT + 1,
        row_count=7,
    )

    rc = run_decompress('sqlite:///ignored.db', dry_run=False)

    assert rc == 0
    mocks['execute_decompress'].assert_awaited_once()


def test_decompress_invalid_env_surfaces_clean_cli_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A settings ValidationError surfaces as a one-line CLI error and EX_CONFIG, not a traceback.

    The documented --decompress prerequisite (unset ENABLE_EMBEDDING_COMPRESSION)
    recreates exactly the env shape the fp32 pgvector dimension validator rejects
    on a PostgreSQL deployment whose EMBEDDING_DIM exceeds the index cap, so the
    settings resolution inside run_decompress must not escape as a raw pydantic
    traceback. Exit 78 mirrors main()'s classification of the same
    ValidationError when it surfaces at module-import time.
    """
    from app.errors import ConfigurationError

    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('EMBEDDING_DIM', str(PGVECTOR_INDEX_DIM_LIMIT + 1))
    get_settings.cache_clear()

    rc = run_decompress('postgresql://u:p@localhost:5432/ctx', dry_run=False)

    assert rc == ConfigurationError.EXIT_CODE
    err = capsys.readouterr().err
    assert 'Configuration invalid' in err
    assert 'pgvector index limit' in err


def test_compress_invalid_env_surfaces_clean_cli_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """run_compress shares the clean env-validation error path with run_decompress."""
    from app.errors import ConfigurationError

    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('EMBEDDING_DIM', str(PGVECTOR_INDEX_DIM_LIMIT + 1))
    get_settings.cache_clear()

    rc = run_compress('postgresql://u:p@localhost:5432/ctx', dry_run=False)

    assert rc == ConfigurationError.EXIT_CODE
    err = capsys.readouterr().err
    assert 'Configuration invalid' in err
    assert 'pgvector index limit' in err
