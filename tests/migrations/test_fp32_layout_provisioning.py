"""Tests for the fp32 vector-table layout across the semantic, chunking and compression migrations.

Covers the absence of the fp32 vec0 table under compression across a restart,
its presence without compression, the stripped semantic migration on an
install without sqlite-vec, and the fp32 layout re-provisioned after a
compression-off flip with generation off. The semantic migration loads the
sqlite-vec extension whenever its script still carries the fp32 vec0 DDL, so
the tests that run that script skip where the package is absent.
"""

import sqlite3

import pytest

from app.backends import StorageBackend
from app.migrations.compression import apply_compression_migration
from app.settings import get_settings
from tests.conftest import requires_sqlite_vec
from tests.helpers import disable_compression
from tests.helpers import enable_compression

pytestmark = pytest.mark.usefixtures('clear_settings_cache')


@pytest.mark.asyncio
@requires_sqlite_vec
async def test_compression_skips_fp32_vec_provisioning_across_restart(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With compression enabled, the server migration sequence
    (semantic -> chunking -> compression) must NOT create the fp32
    vec_context_embeddings table -- only embedding_metadata (required by the
    compressed write path), embedding_chunks (the SQLite 1:N bridge, preserved
    under compression), and the compressed/provenance tables. The invariant must
    hold across a simulated restart, with NO create-then-drop churn.
    """
    enable_compression(monkeypatch)
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    get_settings.cache_clear()
    import app.migrations.chunking as chunking_module
    import app.migrations.compression as compression_module
    import app.migrations.semantic as semantic_module
    monkeypatch.setattr(semantic_module, 'settings', get_settings())
    monkeypatch.setattr(chunking_module, 'settings', get_settings())
    monkeypatch.setattr(compression_module, 'settings', get_settings())

    from app.migrations.chunking import apply_chunking_migration
    from app.migrations.semantic import apply_semantic_search_migration

    async def _run_startup_sequence() -> None:
        # Mirrors the server lifespan migration order.
        await apply_semantic_search_migration(backend=backend)
        await apply_chunking_migration(backend=backend)
        await apply_compression_migration(backend=backend)

    def _tables(conn: sqlite3.Connection) -> set[str]:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {row[0] for row in cur.fetchall()}

    await _run_startup_sequence()
    tables = await backend.execute_read(_tables)
    assert 'vec_context_embeddings' not in tables  # fp32 vec0 table NOT created
    assert 'embedding_metadata' in tables          # required by the compressed write path
    assert 'embedding_chunks' in tables            # SQLite 1:N bridge, preserved
    assert 'vec_context_embeddings_compressed' in tables
    assert 'compression_metadata' in tables

    # Simulated restart: the fp32 table must STILL be absent (no reappearance).
    await _run_startup_sequence()
    tables_after = await backend.execute_read(_tables)
    assert 'vec_context_embeddings' not in tables_after


@pytest.mark.asyncio
@requires_sqlite_vec
async def test_no_compression_creates_fp32_vec_table(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: with compression DISABLED the semantic migration still creates the
    fp32 vec_context_embeddings table (the skip is compression-gated, not default)."""
    disable_compression(monkeypatch)
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    get_settings.cache_clear()
    import app.migrations.semantic as semantic_module
    monkeypatch.setattr(semantic_module, 'settings', get_settings())

    from app.migrations.semantic import apply_semantic_search_migration

    await apply_semantic_search_migration(backend=backend)

    def _exists(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='vec_context_embeddings'",
        )
        return cur.fetchone() is not None

    assert await backend.execute_read(_exists) is True


@pytest.mark.asyncio
async def test_semantic_migration_under_compression_does_not_require_sqlite_vec(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stripped semantic migration must not require sqlite-vec.

    With compression on, skip_fp32_vec strips every vec0 statement, so the
    executed script is only embedding_metadata + its index -- the vec0 module
    is not needed. A slim install without an embeddings-* extra (where
    sqlite-vec ships) must still boot a compressed, generation-off database
    that reached the semantic migration via the infra-present fallthrough;
    requiring the package for DDL already stripped from the script fails boot
    on a functionally unneeded dependency.
    """
    import sys

    import app.migrations.semantic as semantic_module
    from app.migrations.semantic import apply_semantic_search_migration

    # Provision the real embedding_metadata schema with generation ON first
    # (sqlite-vec is available in the dev env, so this full run succeeds).
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.delenv('ENABLE_EMBEDDING_COMPRESSION', raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(semantic_module, 'settings', get_settings())
    await apply_semantic_search_migration(backend=backend)

    def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        return cur.fetchone() is not None

    assert await backend.execute_read(lambda c: _table_exists(c, 'embedding_metadata')) is True

    # Now flip to compression ON + generation OFF (the infra-present fallthrough
    # fires because embedding_metadata exists), and simulate sqlite-vec NOT
    # installed: `import sqlite_vec` raises ImportError. The backend's own load
    # is ImportError-guarded (skips gracefully); only the migration's stripped
    # path (skip_fp32_vec strips every vec0 statement) must avoid demanding it.
    enable_compression(monkeypatch)
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    monkeypatch.setattr(semantic_module, 'settings', get_settings())
    monkeypatch.setitem(sys.modules, 'sqlite_vec', None)

    # Must NOT raise despite sqlite-vec being unavailable (no vec0 DDL to run).
    await apply_semantic_search_migration(backend=backend)
    assert await backend.execute_read(lambda c: _table_exists(c, 'embedding_metadata')) is True


@pytest.mark.asyncio
@requires_sqlite_vec
async def test_sqlite_fp32_reprovisioned_after_compression_off_flip_generation_off(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compression-off flip on an infra-carrying generation-off database self-heals.

    Chain: a generation-on past provisions the fp32 embedding layout; a
    generation-off boot with compression on swaps the (empty) fp32 table for
    the compressed schema, and the validator seeds no provenance row there
    (nothing can write compressed data), so a later bare
    ENABLE_EMBEDDING_COMPRESSION=false flip boots without the --decompress
    ceremony. The cleanup paths gate on embedding_metadata presence and then
    touch the fp32 table, so the semantic migration's infra-present
    fallthrough must re-provision it -- without the fallthrough, every
    text-carrying update would fail inside its transaction on PostgreSQL and
    silently no-op on SQLite.
    """
    import app.migrations.chunking as chunking_module
    import app.migrations.compression as compression_module
    import app.migrations.semantic as semantic_module
    from app.migrations.chunking import apply_chunking_migration
    from app.migrations.semantic import apply_semantic_search_migration

    def _rebind_settings() -> None:
        get_settings.cache_clear()
        fresh = get_settings()
        monkeypatch.setattr(semantic_module, 'settings', fresh)
        monkeypatch.setattr(chunking_module, 'settings', fresh)
        monkeypatch.setattr(compression_module, 'settings', fresh)

    def _fp32_exists(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='vec_context_embeddings'",
        )
        return cur.fetchone() is not None

    # Phase 1: a generation-on past provisions the real fp32 embedding infra.
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    _rebind_settings()
    await apply_semantic_search_migration(backend=backend)
    await apply_chunking_migration(backend=backend)
    assert await backend.execute_read(_fp32_exists) is True

    # Phase 2: generation off + compression on -> the table swap removes the
    # empty fp32 table and provisions the compressed schema.
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    _rebind_settings()
    await apply_compression_migration(backend=backend)
    assert await backend.execute_read(_fp32_exists) is False

    # Phase 3: bare compression-off flip (generation still off; no provenance
    # row, so no disable-direction guard) -> the semantic and chunking
    # fallthrough re-provisions the fp32 layout the cleanup paths depend on.
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.delenv('COMPRESSION_SEED', raising=False)
    _rebind_settings()
    await apply_semantic_search_migration(backend=backend)
    await apply_chunking_migration(backend=backend)
    await apply_compression_migration(backend=backend)
    assert await backend.execute_read(_fp32_exists) is True
