"""SQLite databases shared by the compression migration CLI tests.

The helpers lay down the tables the CLI checks for (the base schema plus plain-table stand-ins for the embedding
tables, so no sqlite-vec is needed), seed fp32 corpora through the repository layer, switch the compression
environment on, and count rows in the fp32 source, the compressed destination and the provenance table.
"""

import asyncio
import contextlib
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from app.backends import create_backend
from app.repositories import RepositoryContainer
from app.settings import get_settings
from tests.helpers import LOCAL_SCOPE


def bootstrap_schema(path: Path) -> None:
    """Apply the SQLite schema to ``path``."""
    from app.schemas import load_schema

    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(load_schema('sqlite'))
        # The semantic-search migration owns embedding_metadata in
        # production; recreate the table directly so the CLI's structural
        # checks pass without dragging in sqlite-vec.
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                context_id TEXT NOT NULL PRIMARY KEY,
                model_name TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS embedding_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                vec_rowid INTEGER NOT NULL,
                start_index INTEGER NOT NULL DEFAULT 0,
                end_index INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
            );
            ''',
        )
    finally:
        conn.close()


def create_fp32_vec_table(path: Path) -> None:
    """Create a minimal stand-in for the fp32 ``vec_context_embeddings`` table.

    The real table is a sqlite-vec virtual table; the CLI's preflight check
    just verifies the name exists, so a plain table with the same column
    name is sufficient for flag-handling tests that do NOT execute the
    encode pass.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS vec_context_embeddings (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB
            );
            ''',
        )
    finally:
        conn.close()


DIM = 128


def seed_fp32_database(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_docs: int,
) -> None:
    """Write ``n_docs`` fp32 vectors to an isolated SQLite database."""
    monkeypatch.setenv('DB_PATH', str(db_path))
    monkeypatch.setenv('STORAGE_BACKEND', 'sqlite')
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    monkeypatch.delenv('ENABLE_SEMANTIC_SEARCH', raising=False)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.delenv('COMPRESSION_SEED', raising=False)
    get_settings.cache_clear()

    async def _setup() -> None:
        from app.schemas import load_schema

        conn = sqlite3.connect(str(db_path))
        try:
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
        finally:
            conn.close()

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            repos = RepositoryContainer(backend)
            rng = np.random.default_rng(7)
            for i in range(n_docs):
                vec = rng.standard_normal(DIM).astype(np.float32)
                vec /= np.linalg.norm(vec)
                cid, _ = await repos.context.store_with_deduplication(
                    scope=LOCAL_SCOPE,
                    visibility='private',
                    thread_id='stream-e2e',
                    source='user',
                    content_type='text',
                    text_content=f'doc-{i}',
                    metadata=None,
                )
                blob = struct.pack(f'<{DIM}f', *vec.tolist())

                def _write_chunk(
                    conn: sqlite3.Connection,
                    *,
                    context_id: str = cid,
                    payload: bytes = blob,
                ) -> None:
                    cur = conn.execute(
                        'INSERT INTO vec_context_embeddings (embedding) VALUES (?)',
                        (payload,),
                    )
                    vec_rowid = cur.lastrowid
                    conn.execute(
                        'INSERT INTO embedding_chunks '
                        '(context_id, vec_rowid, start_index, end_index) '
                        'VALUES (?, ?, ?, ?)',
                        (context_id, vec_rowid, 0, DIM),
                    )
                    conn.execute(
                        'INSERT INTO embedding_metadata '
                        '(context_id, model_name, dimensions, chunk_count) '
                        'VALUES (?, ?, ?, ?)',
                        (context_id, 'test-model', DIM, 1),
                    )

                await backend.execute_write(_write_chunk)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_setup())


def enable_compression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable IP-4 compression with a fixed seed."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    get_settings.cache_clear()


def table_exists(db_path: Path, name: str) -> bool:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def count_compressed(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return int(
            conn.execute(
                'SELECT COUNT(*) FROM vec_context_embeddings_compressed',
            ).fetchone()[0],
        )
    finally:
        conn.close()


def count_fp32(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return int(
            conn.execute(
                'SELECT COUNT(*) FROM vec_context_embeddings',
            ).fetchone()[0],
        )
    finally:
        conn.close()


def count_provenance(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='compression_metadata'",
        )
        if cur.fetchone() is None:
            return 0
        return int(
            conn.execute(
                'SELECT COUNT(*) FROM compression_metadata WHERE id = 1',
            ).fetchone()[0],
        )
    finally:
        conn.close()
