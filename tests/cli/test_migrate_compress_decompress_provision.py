"""The decompress CLI decides pgvector provisioning BEFORE it initializes.

A fresh v3.0.0-defaults deployment on a PostgreSQL host WITHOUT pgvector has a
provenance row seeded with ZERO compressed rows. Turning compression off then
requires ``--decompress`` (a bare ``ENABLE_EMBEDDING_COMPRESSION=false`` is
refused at startup). The decompress env shape is compression-off with embedding
generation defaulting on, for which ``PostgreSQLBackend._resolve_provision_vector``
returns True unconditionally -- so ``backend.initialize()`` would run
``CREATE EXTENSION vector`` and crash on the pgvector-less host BEFORE the
compressed-row count, never reaching the zero-data reverse path.

``_decompress_async`` closes this by probing the compressed-row count up front on
a throwaway plain asyncpg connection (no vector codec, no extension) and passing
``provision_vector=False`` when the zero-data reverse path will run, and
``provision_vector=True`` when compressed rows exist (the fp32 rebuild genuinely
needs pgvector). These tests drive that decision with recording stubs so they run
in the fast unit gate without a live PostgreSQL server.
"""

from collections.abc import Generator

import pytest

from app.cli import migrate_compression
from app.cli.migrate_compression import _decompress_needs_vector
from app.cli.migrate_compression import _make_backend
from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Isolate each test from a settings singleton left by a sibling test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeProbeConn:
    """Minimal asyncpg-connection stand-in for the pre-init decompress probe.

    Records the SQL passed to ``fetchval`` and answers the two probe queries
    from constructor flags, so the test drives :func:`_decompress_needs_vector`
    without a live PostgreSQL server.
    """

    def __init__(self, *, reachable: bool, has_rows: bool) -> None:
        self.reachable = reachable
        self.has_rows = has_rows
        self.closed = False
        self.queries: list[str] = []

    async def fetchval(self, sql: str, *args: object) -> object:
        """Answer the source-presence and existence probes; record the SQL.

        Args:
            sql: The probe SQL issued by :func:`_decompress_needs_vector`.
            args: Unused bind parameters (the probes take none).

        Returns:
            The configured ``reachable`` flag for the ``to_regclass`` probe or
            the ``has_rows`` flag for the ``EXISTS`` probe.

        Raises:
            AssertionError: If an unexpected SQL string is issued.
        """
        del args
        self.queries.append(sql)
        if 'to_regclass' in sql:
            return self.reachable
        if 'EXISTS' in sql:
            return self.has_rows
        raise AssertionError(f'unexpected probe SQL: {sql!r}')

    async def close(self) -> None:
        """Mark the throwaway connection closed."""
        self.closed = True


class _ConnectRecorder:
    """Capture the DSN and kwargs asyncpg.connect is called with in the probe."""

    def __init__(self, conn: _FakeProbeConn) -> None:
        self.conn = conn
        self.dsn: str | None = None
        self.kwargs: dict[str, object] = {}

    async def connect(self, dsn: str, **kwargs: object) -> _FakeProbeConn:
        """Record the connect call and return the pre-built fake connection.

        Args:
            dsn: The connection string passed by the probe.
            kwargs: The connect kwargs (``timeout`` plus the shared
                :func:`build_asyncpg_connect_kwargs` mapping).

        Returns:
            The configured :class:`_FakeProbeConn`.
        """
        self.dsn = dsn
        self.kwargs = kwargs
        return self.conn


class _StopInitializeError(Exception):
    """Sentinel raised by the stub backend's ``initialize()`` to halt the flow.

    ``_decompress_async`` calls ``backend.initialize()`` right after construction
    and before any database work, so raising here lets the test capture the
    ``provision_vector`` passed to ``create_backend`` without stubbing the whole
    downstream decompress pipeline.
    """


class _StubBackend:
    """Backend whose ``initialize()`` halts ``_decompress_async`` immediately."""

    @property
    def backend_type(self) -> str:
        return 'postgresql'

    async def initialize(self) -> None:
        """Halt the decompress flow right after construction.

        Raises:
            _StopInitializeError: Always, so the test asserts on the construction
                kwargs without running the downstream pipeline.
        """
        raise _StopInitializeError

    async def shutdown(self) -> None:
        """No-op teardown (unreached because ``initialize()`` raises first)."""


@pytest.mark.asyncio
async def test_decompress_needs_vector_false_when_no_compressed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty compressed table must NOT provision pgvector.

    The zero-data reverse path only DROPs the empty table and clears the
    provenance row, so the vector type is never needed -- and forcing
    ``CREATE EXTENSION vector`` would crash the disable direction on a
    pgvector-less host.
    """
    conn = _FakeProbeConn(reachable=True, has_rows=False)
    recorder = _ConnectRecorder(conn)
    monkeypatch.setattr('asyncpg.connect', recorder.connect)

    result = await _decompress_needs_vector('postgresql://u:p@h:5432/db')

    assert result is False
    # Both probes ran: source presence, then row existence.
    assert any('to_regclass' in q for q in conn.queries)
    assert any('EXISTS' in q for q in conn.queries)
    assert conn.closed is True
    # The throwaway connection reuses the shared search_path kwargs.
    assert 'server_settings' in recorder.kwargs
    assert recorder.dsn == 'postgresql://u:p@h:5432/db'


@pytest.mark.asyncio
async def test_decompress_needs_vector_false_when_table_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent compressed table short-circuits to False without the row probe."""
    conn = _FakeProbeConn(reachable=False, has_rows=True)
    recorder = _ConnectRecorder(conn)
    monkeypatch.setattr('asyncpg.connect', recorder.connect)

    result = await _decompress_needs_vector('postgresql://u:p@h:5432/db')

    assert result is False
    # The existence probe must be skipped once the table is unreachable.
    assert any('to_regclass' in q for q in conn.queries)
    assert not any('EXISTS' in q for q in conn.queries)
    assert conn.closed is True


@pytest.mark.asyncio
async def test_decompress_needs_vector_true_when_rows_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty compressed table MUST provision pgvector for the fp32 rebuild.

    The full reverse path recreates ``vec_context_embeddings`` (a
    ``vector(dim)`` table) and its HNSW index, which genuinely need pgvector; a
    blanket False here would break a real decompress on a host where pgvector is
    installed but the extension is not yet created.
    """
    conn = _FakeProbeConn(reachable=True, has_rows=True)
    recorder = _ConnectRecorder(conn)
    monkeypatch.setattr('asyncpg.connect', recorder.connect)

    result = await _decompress_needs_vector('postgresql://u:p@h:5432/db')

    assert result is True
    assert conn.closed is True


def test_make_backend_forwards_provision_vector_for_postgresql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_make_backend`` threads ``provision_vector`` into the PostgreSQL factory."""
    captured: dict[str, object] = {}

    def _fake_create_backend(**kwargs: object) -> _StubBackend:
        captured.update(kwargs)
        return _StubBackend()

    monkeypatch.setattr(migrate_compression, 'create_backend', _fake_create_backend)

    _make_backend('postgresql://u:p@h:5432/db', provision_vector=False)

    assert captured['backend_type'] == 'postgresql'
    assert captured['connection_string'] == 'postgresql://u:p@h:5432/db'
    assert captured['provision_vector'] is False


def test_make_backend_ignores_provision_vector_for_sqlite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLite construction never receives ``provision_vector`` (its vec load is unconditional)."""
    captured: dict[str, object] = {}

    def _fake_create_backend(**kwargs: object) -> _StubBackend:
        captured.update(kwargs)
        return _StubBackend()

    monkeypatch.setattr(migrate_compression, 'create_backend', _fake_create_backend)

    _make_backend('sqlite:///tmp/context.db', provision_vector=True)

    assert captured['backend_type'] == 'sqlite'
    assert 'provision_vector' not in captured


@pytest.mark.asyncio
@pytest.mark.parametrize(('probe_result', 'expected'), [(False, False), (True, True)])
async def test_decompress_async_provisions_vector_per_probe(
    monkeypatch: pytest.MonkeyPatch, probe_result: bool, expected: bool,
) -> None:
    """``_decompress_async`` constructs the PostgreSQL backend per the up-front probe.

    A zero-compressed-row probe (False) must yield ``provision_vector=False`` so
    ``initialize()`` skips ``CREATE EXTENSION`` and the zero-data reverse path can
    run on a pgvector-less host; a rows-present probe (True) must still provision.
    """
    captured: dict[str, object] = {}

    async def _fake_probe(address: str) -> bool:
        del address
        return probe_result

    def _fake_create_backend(**kwargs: object) -> _StubBackend:
        captured.update(kwargs)
        return _StubBackend()

    monkeypatch.setattr(migrate_compression, '_decompress_needs_vector', _fake_probe)
    monkeypatch.setattr(migrate_compression, 'create_backend', _fake_create_backend)

    with pytest.raises(_StopInitializeError):
        await migrate_compression._decompress_async(
            'postgresql://u:p@h:5432/db', dry_run=True,
        )

    assert captured['backend_type'] == 'postgresql'
    assert captured['provision_vector'] is expected
