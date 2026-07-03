"""Regression test: the sqlite-vec (vec0) extension loads independently of the
embedding-generation toggle.

The fp32 ``vec_context_embeddings`` vec0 virtual table can physically persist
from an earlier session that ran with embedding generation enabled. The
delete/update stale-embedding cleanup paths gate on the durable
``embedding_tables_exist()`` table-presence signal, NOT on the runtime
``ENABLE_EMBEDDING_GENERATION`` toggle, so they attempt vec0 access whenever the
table exists. If the vec0 module is not loaded on the connection, that access
raises ``no such module: vec0`` -- which the delete path swallows (permanently
orphaning the FK-less vec0 rows once the embedding_chunks bridge cascades) and
the update path propagates (rolling back the whole text update). The extension
load must therefore be decoupled from the generation toggle.
"""

import sqlite3
from pathlib import Path

import pytest

from app.backends import sqlite_backend as sqlite_backend_module
from app.backends.sqlite_backend import ManagedConnection
from app.backends.sqlite_backend import SQLiteBackend
from app.settings import get_settings
from tests.conftest import requires_sqlite_vec


@requires_sqlite_vec
def test_vec_extension_loads_when_generation_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With generation disabled, a connection still loads vec0 so vec-table access works.

    Pre-fix the method early-returned when generation was disabled, leaving the
    vec0 module absent: ``CREATE VIRTUAL TABLE ... USING vec0`` and the
    ``DELETE FROM vec_context_embeddings`` that ``delete_all_chunks`` runs both
    raised ``no such module: vec0``.
    """
    # Force the module-level settings the backend reads to generation-disabled,
    # following the documented per-test settings-refresh pattern.
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    monkeypatch.setattr(sqlite_backend_module, 'settings', get_settings())
    assert sqlite_backend_module.settings.embedding.generation_enabled is False

    backend = SQLiteBackend(db_path=str(tmp_path / 'vecgate.db'))
    # Use the production connection subclass (ManagedConnection): a bare
    # sqlite3.Connection has no __dict__ and cannot carry the _vec_loaded flag.
    conn = sqlite3.connect(':memory:', factory=ManagedConnection)
    try:
        backend._load_sqlite_vec_extension(conn)
        assert getattr(conn, '_vec_loaded', False) is True
        # The vec0 module must be usable even though generation is disabled.
        conn.execute('CREATE VIRTUAL TABLE v USING vec0(embedding float[4])')
        # The exact statement delete_all_chunks runs on the SQLite cleanup path.
        conn.execute('DELETE FROM v WHERE rowid = 1')
    finally:
        conn.close()
        get_settings.cache_clear()
