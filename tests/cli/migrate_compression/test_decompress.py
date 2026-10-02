"""Tests for ``--decompress`` planning: environment validation, the provenance and codebook-fingerprint gates,
and the pgvector dimension pre-flight.
"""

import sqlite3
from pathlib import Path
from unittest import mock

import pytest

from app.cli.migrate_compression.decompress import run_decompress
from app.compression.types import CompressionMetadata
from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT
from app.settings import get_settings
from tests.cli.migrate_compression._sqlite_db import bootstrap_schema
from tests.cli.migrate_compression._sqlite_db import create_fp32_vec_table
from tests.conftest import requires_numpy


def test_main_rejects_decompress_when_compression_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_decompress`` exits 1 when ENABLE_EMBEDDING_COMPRESSION is true."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    db = tmp_path / 'test.db'
    bootstrap_schema(db)

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)

    assert rc == 1
    err = capsys.readouterr().err
    assert 'must be unset' in err


def test_main_decompress_noop_when_nothing_to_do(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--decompress`` no-ops when the source is already fp32-only."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)
    create_fp32_vec_table(db)

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
    bootstrap_schema(db)

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
def test_decompress_aborts_on_codebook_fingerprint_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--decompress`` aborts (rc 1) BEFORE any decode/DROP on a fingerprint mismatch.

    The CLI rebuilds the provider from the stored (dim, seed, bits, variant), then
    re-derives the realized codebook fingerprint and compares it, mirroring the
    startup validator: a cross-host numpy.linalg.qr divergence would otherwise
    silently corrupt every reconstructed vector and then DROP the only
    correctly-decodable copy. Here the stored fingerprint is a deliberately wrong value, so the
    realized digest cannot match and decompress must refuse without dropping. A
    compressed row is present because the gate applies only when rows exist --
    the zero-data reverse path decodes nothing and bypasses it.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)

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
    divergence would leave the operator with no working escape: the server refuses to
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


def _install_decompress_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend_type: str,
    dim: int,
    row_count: int,
) -> dict[str, mock.AsyncMock | mock.MagicMock]:
    """Fake the decompress pipeline around the fp32 dimension pre-flight.

    Installs a stub backend plus provenance/probe fakes in the
    ``app.cli.migrate_compression.decompress`` namespace so ``run_decompress`` reaches the
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
    import app.cli.migrate_compression.decompress as decompress_mod

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

    monkeypatch.setattr(decompress_mod, '_decompress_needs_vector', _fake_needs_vector)
    monkeypatch.setattr(decompress_mod, 'make_backend', _fake_make_backend)
    monkeypatch.setattr(decompress_mod, 'read_compression_metadata', _fake_read_metadata)
    monkeypatch.setattr(decompress_mod, 'table_exists', _fake_table_exists)
    monkeypatch.setattr(decompress_mod, 'count_table', _fake_count_table)
    monkeypatch.setattr(decompress_mod, '_read_compressed_probe', _fake_read_probe)
    monkeypatch.setattr(decompress_mod, '_provider_for', provider_for)
    monkeypatch.setattr(decompress_mod, 'execute_decompress', execute_decompress)
    return {'execute_decompress': execute_decompress, 'provider_for': provider_for}


def test_decompress_pg_refuses_dim_above_pgvector_index_cap_before_any_ddl(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """PostgreSQL --decompress refuses a provenance dim pgvector cannot index, before any DDL.

    Without the pre-flight the reverse migration would stream and decode every
    compressed row and only then die at the trailing HNSW CREATE INDEX
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
