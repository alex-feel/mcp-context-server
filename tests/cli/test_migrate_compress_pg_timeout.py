"""The PostgreSQL compression CLI runs its heavy DDL under the migration budget.

The ``--compress`` / ``--decompress`` PostgreSQL paths borrow the server
``PostgreSQLBackend`` pool, whose ``command_timeout`` (~60s) asyncpg applies as the
default client-side deadline. Their heaviest operations -- a full HNSW index build,
a whole-table DROP, the streamed batch reads -- can exceed that on a large corpus and
be cancelled client-side as a non-retryable ``asyncio.TimeoutError`` before the longer
migration budget applies. These tests assert that each transaction raises its
statement budget via ``SET LOCAL statement_timeout`` and that every heavy statement
carries an explicit per-call deadline matching the migration budget (so it is NOT
capped at the pool ``command_timeout``).

The PostgreSQL branches are driven here with a recording mock connection so the
assertions run in the fast unit gate without Docker; live coverage runs against the
pgvector container in the integration suite.
"""

import asyncio
from collections.abc import Generator
from typing import Any
from typing import cast
from unittest.mock import Mock

import pytest

from app.cli.migrate_compression import _execute_compress_postgresql
from app.cli.migrate_compression import _execute_decompress_postgresql
from app.compression.types import CompressionMetadata
from app.settings import get_settings

# The client-side margin _pg_ddl adds on top of the server-side budget so an overrun
# is cancelled server-side (retryable) before the client gives up (non-retryable).
_CLIENT_MARGIN_S = 5.0


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _RecordingConn:
    """An asyncpg-shaped connection that records statements and their timeouts."""

    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, float | None]] = []
        self.fetch_calls: list[tuple[str, float | None]] = []

    async def execute(self, statement: str, *_args: Any, **kwargs: Any) -> str:
        self.execute_calls.append((statement, kwargs.get('timeout')))
        return 'OK'

    async def fetch(self, statement: str, *_args: Any, **kwargs: Any) -> list[Any]:
        self.fetch_calls.append((statement, kwargs.get('timeout')))
        # An empty batch makes the streaming loop exit after the first read.
        return []

    async def fetchval(self, statement: str, *_args: Any, **_kwargs: Any) -> int:
        # Idempotency probes: compress wants 0 (provenance absent -> proceed);
        # decompress wants 1 (compressed source table present -> proceed).
        return 1 if 'information_schema' in statement else 0


class _FakeTxn:
    def __init__(self, conn: _RecordingConn) -> None:
        self.connection = conn
        self.backend_type = 'postgresql'


class _TxnContext:
    def __init__(self, conn: _RecordingConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeTxn:
        return _FakeTxn(self._conn)

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _FakeBackend:
    backend_type = 'postgresql'

    def __init__(self, conn: _RecordingConn) -> None:
        self._conn = conn

    def begin_transaction(self) -> _TxnContext:
        return _TxnContext(self._conn)


def _provenance() -> CompressionMetadata:
    return CompressionMetadata(
        provider='turboquant',
        bits=4,
        variant='ip',
        seed=42,
        dim=128,
        codebook_fingerprint='ab' * 32,
    )


def _budget_set_local() -> str:
    budget = get_settings().storage.postgresql_migration_timeout_s
    return f'SET LOCAL statement_timeout = {int(budget * 1000)}'


def _expected_client_timeout() -> float:
    return get_settings().storage.postgresql_migration_timeout_s + _CLIENT_MARGIN_S


def _timeouts_for(conn: _RecordingConn, needle: str) -> list[float | None]:
    return [t for stmt, t in conn.execute_calls if needle in stmt]


def test_compress_postgresql_runs_heavy_ddl_under_migration_budget() -> None:
    conn = _RecordingConn()
    backend = _FakeBackend(conn)
    asyncio.run(
        _execute_compress_postgresql(
            cast(Any, backend),
            Mock(),
            _provenance(),
            row_count=0,
        ),
    )

    # The transaction raises its server-side statement budget.
    set_local = [stmt for stmt, _ in conn.execute_calls if stmt.startswith('SET LOCAL statement_timeout')]
    assert set_local == [_budget_set_local()]

    expected = _expected_client_timeout()
    # The destructive DDL carries the explicit client-side deadline, not the bare pool timeout.
    drop_table = _timeouts_for(conn, 'DROP TABLE IF EXISTS vec_context_embeddings')
    assert drop_table
    assert all(t == expected for t in drop_table)
    drop_index = _timeouts_for(conn, 'DROP INDEX IF EXISTS idx_vec_context_embeddings_hnsw')
    assert drop_index
    assert all(t == expected for t in drop_index)
    # The streamed batch read is budgeted too.
    assert conn.fetch_calls
    assert all(t == expected for _, t in conn.fetch_calls)
    # Sanity: the empty-table CREATE statements stay bare (timeout None), so the
    # recording distinguishes budgeted statements from un-budgeted ones.
    assert any(t is None for stmt, t in conn.execute_calls if stmt.startswith('CREATE TABLE'))


def test_decompress_postgresql_runs_heavy_ddl_under_migration_budget() -> None:
    conn = _RecordingConn()
    backend = _FakeBackend(conn)
    asyncio.run(
        _execute_decompress_postgresql(
            cast(Any, backend),
            Mock(),
            _provenance(),
            row_count=0,
        ),
    )

    set_local = [stmt for stmt, _ in conn.execute_calls if stmt.startswith('SET LOCAL statement_timeout')]
    assert set_local == [_budget_set_local()]

    expected = _expected_client_timeout()
    # The HNSW index build is the heaviest op and must run under the migration budget.
    create_hnsw = _timeouts_for(conn, 'USING hnsw (embedding vector_l2_ops)')
    assert create_hnsw
    assert all(t == expected for t in create_hnsw)
    drop_compressed = _timeouts_for(conn, 'DROP TABLE IF EXISTS vec_context_embeddings_compressed')
    assert drop_compressed
    assert all(t == expected for t in drop_compressed)
    assert conn.fetch_calls
    assert all(t == expected for _, t in conn.fetch_calls)
