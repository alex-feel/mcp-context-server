"""Verify read_compression_metadata only swallows the specific
'no such table: compression_metadata' error and propagates all others.

Defense-in-depth check that mirrors the type-safe
asyncpg.UndefinedTableError handling in the PostgreSQL branch of the
provenance read path: a missing-table error referencing any OTHER table
indicates database corruption and MUST surface to the caller.
"""

import sqlite3
from pathlib import Path
from typing import Any
from typing import cast

import pytest

from app.backends.sqlite_backend import SQLiteBackend
from app.compression.provenance import read_compression_metadata
from app.compression.types import CompressionMetadata


@pytest.mark.asyncio
async def test_returns_none_when_compression_metadata_table_missing(
    tmp_path: Path,
) -> None:
    """The pre-bootstrap state must produce None, not raise."""
    db_path = tmp_path / 'pre_bootstrap.db'
    backend = SQLiteBackend(db_path=db_path)
    await backend.initialize()
    try:
        # No compression migration applied; the table does NOT exist.
        result = await read_compression_metadata(backend)
        assert result is None
    finally:
        await backend.shutdown()


@pytest.mark.asyncio
async def test_propagates_other_missing_table_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``read_compression_metadata`` must propagate non-target missing-table
    errors instead of returning ``None``.

    The narrow substring check in the SQLite branch matches only
    ``no such table: compression_metadata`` and MUST let any other
    ``OperationalError`` surface so the caller can fail loudly. This
    test wraps ``execute_read`` so the production ``_read_sqlite``
    callback runs against a fake SQLite connection whose ``.execute()``
    raises the unrelated missing-table error. A sentinel flag asserts
    the callback was actually invoked, pinning the invariant that the
    substring check at ``app/compression/provenance.py:58`` IS executed
    (not merely bypassed by an upstream raise).
    """
    db_path = tmp_path / 'corrupt.db'
    backend = SQLiteBackend(db_path=db_path)
    await backend.initialize()
    try:
        original_execute_read = backend.execute_read
        callback_invoked = False

        class _RaisingConnection:
            """Stand-in for ``sqlite3.Connection`` whose ``execute`` raises.

            Used to exercise ``_read_sqlite``'s try/except so the
            narrow substring check at ``provenance.py:58`` is reached.
            """

            def execute(self, *args: object, **kwargs: object) -> object:
                del args, kwargs
                raise sqlite3.OperationalError(
                    'no such table: embedding_chunks',
                )

        async def _spy_execute_read(
            callback: object,
            *args: object,
            **kwargs: object,
        ) -> object:
            nonlocal callback_invoked

            def _wrapped(_real_conn: sqlite3.Connection) -> object:
                nonlocal callback_invoked
                callback_invoked = True
                fake_conn = _RaisingConnection()
                return cast(Any, callback)(fake_conn)

            return await original_execute_read(_wrapped, *args, **kwargs)

        monkeypatch.setattr(backend, 'execute_read', _spy_execute_read)

        with pytest.raises(sqlite3.OperationalError, match='embedding_chunks'):
            await read_compression_metadata(backend)

        assert callback_invoked, (
            '_read_sqlite was not invoked; the narrow substring check at '
            'app/compression/provenance.py:58 is structurally unpinned. '
            'The spy wrapper must forward the callback so the production '
            'code path executes against the fake connection.'
        )
    finally:
        await backend.shutdown()


@pytest.mark.asyncio
async def test_reads_none_fingerprint_when_column_missing(
    tmp_path: Path,
) -> None:
    """A pre-fingerprint table (codebook_fingerprint column absent) reads as None.

    The codebook_fingerprint column was added after the original compression
    feature, and the compression migration adds it idempotently on the next
    server start. The standalone migration CLI, however, reads the provenance
    row BEFORE any migration runs, so a table created before the column existed
    must be tolerated rather than crashing the CLI: a missing column is treated
    identically to a NULL fingerprint value (no rotation digest recorded yet).
    Every other schema deviation still propagates loudly.
    """
    db_path = tmp_path / 'pre_fingerprint.db'
    backend = SQLiteBackend(db_path=db_path)
    await backend.initialize()
    try:

        def _create_pre_fingerprint(conn: sqlite3.Connection) -> None:
            # The compression_metadata schema as it existed before the
            # codebook_fingerprint column was introduced (six columns, no digest).
            conn.execute(
                'CREATE TABLE compression_metadata ('
                '  id INTEGER PRIMARY KEY CHECK (id = 1),'
                '  provider TEXT NOT NULL,'
                '  bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),'
                "  variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),"
                '  seed INTEGER NOT NULL CHECK (seed >= 0),'
                '  dim INTEGER NOT NULL CHECK (dim > 0),'
                '  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP'
                ')',
            )
            conn.execute(
                'INSERT INTO compression_metadata (id, provider, bits, variant, seed, dim) '
                "VALUES (1, 'turboquant', 4, 'ip', 0, 1024)",
            )

        await backend.execute_write(_create_pre_fingerprint)

        result = await read_compression_metadata(backend)
        assert isinstance(result, CompressionMetadata)
        assert result.codebook_fingerprint is None
        assert result.provider == 'turboquant'
        assert result.bits == 4
        assert result.variant == 'ip'
        assert result.seed == 0
        assert result.dim == 1024
    finally:
        await backend.shutdown()
