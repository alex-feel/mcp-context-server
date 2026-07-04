"""Verify the decompress branch's source-presence probe resolves via search_path.

``_execute_decompress_postgresql`` probes whether the compressed source
table is present before streaming. The probe must use ``to_regclass`` so it
resolves the BARE table name through the connection ``search_path`` --
exactly like the bare-name DML and the trailing ``DROP TABLE`` it gates. A
probe pinned to the configured ``POSTGRESQL_SCHEMA`` (the pre-fix
``information_schema.tables`` lookup) reported a table living in ``public``
as absent while the bare-name SQL kept reaching it. The operator-controlled
schema value must also never be composed into the SQL string.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

import pytest

from app.backends import StorageBackend
from app.backends.base import TransactionContext
from app.cli.migrate_compression import _execute_decompress_postgresql
from app.compression.base import CompressionProvider
from app.compression.types import CompressionMetadata
from app.settings import get_settings


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
    # The schema-pinned lookup must be gone, and the operator-controlled
    # schema value must not leak into ANY SQL string.
    assert all('information_schema' not in s for s, _ in fetchval.calls)
    assert all('evil' not in s for s, _ in fetchval.calls)
