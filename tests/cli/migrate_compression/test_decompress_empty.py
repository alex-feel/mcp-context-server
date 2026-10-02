"""Tests for the zero-data ``--decompress`` path: drop the empty compressed table and clear the provenance row.

The path needs no fp32 infrastructure, re-checks emptiness inside its own transaction, rolls back as a unit on
SQLite, and runs its PostgreSQL recount under the migration budget.
"""

import asyncio
import sqlite3
from pathlib import Path
from typing import Any
from typing import cast

import pytest

from app.cli.migrate_compression.decompress import run_decompress
from app.settings import get_settings
from tests.cli.migrate_compression._pg_fakes import FakeBackend
from tests.cli.migrate_compression._pg_fakes import RecordingConn
from tests.cli.migrate_compression._pg_fakes import budget_set_local
from tests.cli.migrate_compression._pg_fakes import expected_client_timeout
from tests.conftest import requires_numpy


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
    # Base schema ONLY -- deliberately NOT bootstrap_schema: a generation-off
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
    from app.cli.migrate_compression.decompress_empty import execute_decompress_empty

    backend = create_backend(backend_type='sqlite', db_path=str(db))
    await backend.initialize()
    try:
        with pytest.raises(ValueError, match='concurrent writer'):
            await execute_decompress_empty(backend)
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
    before DML: without an explicit BEGIN IMMEDIATE the recount would pin no
    snapshot and the DROP would run in AUTOCOMMIT -- durable immediately -- so a
    failure between the DROP and the provenance DELETE would leave the database in
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
    from app.cli.migrate_compression.decompress_empty import execute_decompress_empty

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
            await execute_decompress_empty(backend)
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


def test_decompress_empty_recount_runs_under_migration_budget() -> None:
    """The zero-data reverse path's under-lock recount also carries the migration budget.

    The third under-lock recount must use the budget like its compress/decompress
    siblings, not the pool command_timeout.
    """
    from app.cli.migrate_compression.decompress_empty import execute_decompress_empty

    conn = RecordingConn()
    backend = FakeBackend(conn)
    asyncio.run(execute_decompress_empty(cast(Any, backend)))

    set_local = [stmt for stmt, _ in conn.execute_calls if stmt.startswith('SET LOCAL statement_timeout')]
    assert set_local == [budget_set_local()]
    recount = [t for stmt, t in conn.fetchval_calls if 'COUNT(*) FROM vec_context_embeddings_compressed' in stmt]
    assert recount
    assert all(t == expected_client_timeout() for t in recount)
