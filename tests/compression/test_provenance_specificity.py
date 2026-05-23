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
