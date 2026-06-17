"""End-to-end migration CLI tests on SQLite for --re-embed.

Exercises ``mcp-context-server-migrate --re-embed`` against a freshly-created
SQLite database that already contains embeddings, verifying the headline
behavior: switching the embedding MODEL re-embeds the entire corpus (deleting
the old vectors first), backfilling any missing entries along the way, on both
the fp32 and compressed layouts.

Mirrors the structural pattern of ``test_migrate_embed_missing_e2e.py``.
"""

import asyncio
import contextlib
import sqlite3
import struct
from collections.abc import Generator
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from app.backends import create_backend
from app.cli.migrate_embeddings import run_embed_missing
from app.cli.migrate_reembed import run_reembed
from app.embeddings.base import EmbeddingProvider
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository import _reset_compression_cache
from app.settings import get_settings

DIM = 1024

_ALLOWED_COUNT_TABLES = {
    'vec_context_embeddings',
    'vec_context_embeddings_compressed',
    'embedding_chunks',
    'embedding_metadata',
    'compression_metadata',
    'context_entries',
}


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset settings + compression caches around every test."""
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()


class _FakeEmbeddingProvider:
    """Deterministic stub embedding provider returning unit float32 vectors."""

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim

    async def initialize(self) -> None:  # pragma: no cover - trivial
        return None

    async def shutdown(self) -> None:  # pragma: no cover - trivial
        return None

    async def embed_query(self, text: str) -> list[float]:
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed=seed)
        vec = rng.standard_normal(self._dim).astype(np.float32)
        vec /= np.linalg.norm(vec) or 1.0
        return vec.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_query(t) for t in texts]

    async def is_available(self) -> bool:  # pragma: no cover - trivial
        return True

    def get_dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return 'fake'


def _base_env(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the shared base environment for a SQLite re-embed test."""
    monkeypatch.setenv('DB_PATH', str(db_path))
    monkeypatch.setenv('STORAGE_BACKEND', 'sqlite')
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    monkeypatch.delenv('ENABLE_SEMANTIC_SEARCH', raising=False)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.delenv('COMPRESSION_SEED', raising=False)
    get_settings.cache_clear()


def _create_schema(conn: sqlite3.Connection) -> None:
    """Create the base schema plus plain fp32 embedding tables."""
    from app.schemas import load_schema

    conn.executescript(load_schema('sqlite'))
    conn.executescript(
        '''
        CREATE TABLE IF NOT EXISTS vec_context_embeddings (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS embedding_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context_id TEXT NOT NULL,
            vec_rowid INTEGER NOT NULL,
            start_index INTEGER NOT NULL DEFAULT 0,
            end_index INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS embedding_metadata (
            context_id TEXT NOT NULL PRIMARY KEY,
            model_name TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
        );
        ''',
    )


def _seed_fp32(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_embedded: int,
    model: str,
    n_missing: int = 0,
) -> tuple[list[str], list[str]]:
    """Seed a SQLite database with embedded and (optionally) un-embedded rows.

    Returns:
        Tuple of (embedded_ids, missing_ids).
    """
    _base_env(db_path, monkeypatch)

    async def _setup() -> tuple[list[str], list[str]]:
        conn = sqlite3.connect(str(db_path))
        try:
            _create_schema(conn)
        finally:
            conn.close()

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            repos = RepositoryContainer(backend)
            rng = np.random.default_rng(7)
            embedded: list[str] = []
            for i in range(n_embedded):
                vec = rng.standard_normal(DIM).astype(np.float32)
                vec /= np.linalg.norm(vec)
                cid, _ = await repos.context.store_with_deduplication(
                    thread_id='reembed-e2e',
                    source='user',
                    content_type='text',
                    text_content=f'embedded-doc-{i}',
                    metadata=None,
                )
                embedded.append(cid)
                blob = struct.pack(f'<{DIM}f', *vec.tolist())

                def _write(
                    cn: sqlite3.Connection,
                    *,
                    context_id: str = cid,
                    payload: bytes = blob,
                ) -> None:
                    cur = cn.execute(
                        'INSERT INTO vec_context_embeddings (embedding) VALUES (?)',
                        (payload,),
                    )
                    vec_rowid = cur.lastrowid
                    cn.execute(
                        'INSERT INTO embedding_chunks '
                        '(context_id, vec_rowid, start_index, end_index) '
                        'VALUES (?, ?, ?, ?)',
                        (context_id, vec_rowid, 0, DIM),
                    )
                    cn.execute(
                        'INSERT INTO embedding_metadata '
                        '(context_id, model_name, dimensions, chunk_count) '
                        'VALUES (?, ?, ?, ?)',
                        (context_id, model, DIM, 1),
                    )

                await backend.execute_write(_write)

            missing: list[str] = []
            for i in range(n_missing):
                cid, _ = await repos.context.store_with_deduplication(
                    thread_id='reembed-e2e',
                    source='user',
                    content_type='text',
                    text_content=f'missing-doc-{i}',
                    metadata=None,
                )
                missing.append(cid)
            return embedded, missing
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    return asyncio.run(_setup())


def _table_exists(db_path: Path, table_name: str) -> bool:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def _count(db_path: Path, table_name: str) -> int:
    if table_name not in _ALLOWED_COUNT_TABLES:
        raise ValueError(f'table {table_name!r} not in count allow-list')
    conn = sqlite3.connect(str(db_path))
    try:
        return int(conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0])
    finally:
        conn.close()


def _distinct_models(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            str(r[0])
            for r in conn.execute('SELECT DISTINCT model_name FROM embedding_metadata').fetchall()
        }
    finally:
        conn.close()


@pytest.mark.integration
def test_reembed_fp32_model_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--re-embed`` regenerates fp32 embeddings under the new model."""
    db = tmp_path / 'reembed_fp32.db'
    embedded, _ = _seed_fp32(db, monkeypatch, n_embedded=2, model='old-model')
    assert _count(db, 'embedding_metadata') == 2
    assert _distinct_models(db) == {'old-model'}

    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('EMBEDDING_MODEL', 'new-model')
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    monkeypatch.setenv('ENABLE_CHUNKING', 'false')
    get_settings.cache_clear()

    provider = cast(EmbeddingProvider, _FakeEmbeddingProvider())
    with patch('app.cli._embedding_runtime.create_embedding_provider', return_value=provider):
        rc = run_reembed(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    # Same number of entries, all re-recorded under the new model.
    assert _count(db, 'embedding_metadata') == len(embedded)
    assert _count(db, 'vec_context_embeddings') == len(embedded)
    assert _distinct_models(db) == {'new-model'}


@pytest.mark.integration
def test_reembed_also_embeds_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--re-embed`` is a superset of --embed-missing: it embeds gaps too."""
    db = tmp_path / 'reembed_superset.db'
    embedded, missing = _seed_fp32(
        db, monkeypatch, n_embedded=2, model='old-model', n_missing=2,
    )
    assert _count(db, 'embedding_metadata') == 2  # only the embedded ones

    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('EMBEDDING_MODEL', 'new-model')
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    monkeypatch.setenv('ENABLE_CHUNKING', 'false')
    get_settings.cache_clear()

    provider = cast(EmbeddingProvider, _FakeEmbeddingProvider())
    with patch('app.cli._embedding_runtime.create_embedding_provider', return_value=provider):
        rc = run_reembed(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    total = len(embedded) + len(missing)
    assert _count(db, 'embedding_metadata') == total
    assert _count(db, 'vec_context_embeddings') == total
    assert _distinct_models(db) == {'new-model'}


@pytest.mark.integration
def test_reembed_compressed_model_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--re-embed`` regenerates compressed payloads under the new model."""
    db = tmp_path / 'reembed_compressed.db'
    # Seed entries with NO fp32 embeddings, then populate the compressed layout
    # via --embed-missing under 'old-model'.
    _, missing = _seed_fp32(db, monkeypatch, n_embedded=0, model='unused', n_missing=2)
    assert _count(db, 'embedding_metadata') == 0
    # Drop the unused fp32 table; the compressed path uses its own.
    conn = sqlite3.connect(str(db))
    try:
        conn.execute('DROP TABLE IF EXISTS vec_context_embeddings')
        conn.commit()
    finally:
        conn.close()

    # Apply the compression schema migration so the compressed table +
    # provenance row exist.
    def _enable_compression(model: str) -> None:
        monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
        monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
        monkeypatch.setenv('COMPRESSION_SEED', '0')
        monkeypatch.setenv('COMPRESSION_BITS', '4')
        monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
        monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
        monkeypatch.setenv('EMBEDDING_MODEL', model)
        monkeypatch.setenv('ENABLE_CHUNKING', 'false')
        get_settings.cache_clear()
        _reset_compression_cache()
        # Rebind the module-level ``settings`` captured at import time in both
        # the compression-migration module (so apply_compression_migration sees
        # compression=on) and app.tools._shared (so
        # generate_compression_with_timeout populates payloads).
        import app.migrations.compression as _compression_migration_module
        import app.tools._shared as _shared_module
        monkeypatch.setattr(_shared_module, 'settings', get_settings())
        monkeypatch.setattr(
            _compression_migration_module, 'settings', get_settings(),
        )

    _enable_compression('old-model')

    async def _apply_compression_schema() -> None:
        from app.migrations.compression import apply_compression_migration
        backend = create_backend(backend_type='sqlite', db_path=str(db))
        await backend.initialize()
        try:
            await apply_compression_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_apply_compression_schema())

    provider = cast(EmbeddingProvider, _FakeEmbeddingProvider())
    with patch('app.cli._embedding_runtime.create_embedding_provider', return_value=provider):
        rc_seed = run_embed_missing(f'sqlite:///{db}', dry_run=False)
    assert rc_seed == 0
    assert _count(db, 'vec_context_embeddings_compressed') == len(missing)
    assert _distinct_models(db) == {'old-model'}

    # Now re-embed the compressed corpus under a NEW model.
    _enable_compression('new-model')
    with patch('app.cli._embedding_runtime.create_embedding_provider', return_value=provider):
        rc = run_reembed(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    assert _table_exists(db, 'vec_context_embeddings_compressed')
    assert _count(db, 'vec_context_embeddings_compressed') == len(missing)
    assert _count(db, 'embedding_metadata') == len(missing)
    assert _distinct_models(db) == {'new-model'}
