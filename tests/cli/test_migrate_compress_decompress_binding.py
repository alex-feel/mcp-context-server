"""Verify the decompress branch's catalog query uses parameterized binding.

The ``_execute_decompress_postgresql`` function probes
``information_schema.tables`` to detect whether the compressed source
table is present. The SQL must bind the schema name as a parameter
($1) rather than composing it via f-string interpolation; this avoids
exposing operator-controlled ``POSTGRESQL_SCHEMA`` to SQL composition
even when the value contains characters that would change query
semantics if interpolated.
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
    SQL string and any number of positional bind parameters, returns an
    integer count, and records the (sql, params) tuple for assertion.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def __call__(self, sql: str, *params: object) -> int:
        self.calls.append((sql, params))
        # Return 0 so the function takes the early-return branch and
        # avoids subsequent SQL that requires a real connection.
        return 0


class _StubConnection:
    """Stand-in for ``asyncpg.Connection`` covering only the methods
    invoked before ``_execute_decompress_postgresql`` early-returns.
    """

    def __init__(self, fetchval: _FetchvalRecorder) -> None:
        self.fetchval = fetchval

    async def execute(self, sql: str, *params: object) -> None:
        # The two ``conn.execute`` calls (CREATE TABLE / CREATE INDEX)
        # run before the catalog-query early-return. They produce no
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
    scoped to the catalog-query binding invariant.
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
async def test_decompress_catalog_query_binds_schema_as_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The catalog query MUST use ``$1`` binding for the schema name.

    A schema string containing a single quote ``'`` would change query
    semantics under f-string composition; under parameterized binding
    asyncpg either rejects the value as an invalid identifier or
    passes it through as a plain string literal. Either outcome is
    safe; the property under test is that the schema does NOT appear
    inside the SQL string sent to ``fetchval``.
    """
    fetchval = _FetchvalRecorder()
    conn = _StubConnection(fetchval=fetchval)
    txn = _StubTxn(conn=conn)
    backend = _StubBackend(txn=txn)

    # Patch the settings to return a schema containing an injection-shaped
    # character; under f-string composition this would corrupt the SQL.
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

    # Find the catalog-query call. The first or only fetchval invocation
    # is the information_schema lookup.
    catalog_calls = [
        (sql, params) for sql, params in fetchval.calls
        if 'information_schema.tables' in sql
    ]
    assert len(catalog_calls) == 1, (
        f'Expected exactly one information_schema.tables fetchval; '
        f'captured={catalog_calls}'
    )
    sql, params = catalog_calls[0]
    assert "'evil'--'" not in sql, (
        'Schema name appears inside the SQL string; binding is broken. '
        f'sql={sql!r}'
    )
    assert '$1' in sql, (
        'SQL must use $1 placeholder for the schema name. '
        f'sql={sql!r}'
    )
    assert params == ("evil'--",), (
        'Schema name must be passed as the first fetchval parameter. '
        f'params={params!r}'
    )
