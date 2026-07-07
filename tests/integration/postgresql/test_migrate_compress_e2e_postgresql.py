"""End-to-end migration CLI tests on PostgreSQL.

Exercises ``mcp-context-server-migrate --compress`` and ``--decompress``
against the pgvector docker-compose container provisioned by the
session-scoped ``pg_test_url`` fixture. The tests use an isolated
PostgreSQL database name per scenario so the singleton compression
provenance invariant does not leak across cases.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from collections.abc import Generator
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import numpy as np
import pytest
import pytest_asyncio

from app.backends import create_backend
from app.cli.migrate_compression import run_compress
from app.cli.migrate_compression import run_decompress
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository import ChunkEmbedding
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.embedding_repository import _reset_compression_cache
from app.settings import get_settings

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

DIM = 1024


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset settings + compression caches around every test."""
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()


def _replace_db_name(pg_url: str, new_db: str) -> str:
    """Return ``pg_url`` with the database name replaced by ``new_db``."""
    parts = urlsplit(pg_url)
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        f'/{new_db}',
        parts.query,
        parts.fragment,
    ))


@pytest_asyncio.fixture
async def isolated_pg_db(
    pg_test_url: str,
) -> AsyncIterator[str]:
    """Create an isolated database, yield its connection string, drop after."""
    db_name = 'mcp_migrate_compress_e2e'
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {db_name}')
        await admin.execute(f'CREATE DATABASE {db_name}')
    finally:
        await admin.close()

    target_url = _replace_db_name(pg_test_url, db_name)

    # Install pgvector inside the isolated DB.
    setup = await asyncpg.connect(target_url)
    try:
        await setup.execute('CREATE EXTENSION IF NOT EXISTS vector')
    finally:
        await setup.close()

    yield target_url

    admin = await asyncpg.connect(pg_test_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {db_name}')
    finally:
        await admin.close()


async def _seed_fp32_pg(
    pg_url: str, monkeypatch: pytest.MonkeyPatch, n_docs: int = 3,
) -> int:
    """Provision the schema + fp32 embeddings on PostgreSQL.

    Args:
        pg_url: PostgreSQL connection string.
        monkeypatch: Test-scope env manipulator.
        n_docs: Number of documents to seed.

    Returns:
        The number of rows written so the caller can verify that later
        compress/decompress runs preserved counts.
    """
    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('POSTGRESQL_CONNECTION_STRING', pg_url)
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.delenv('COMPRESSION_SEED', raising=False)
    get_settings.cache_clear()

    # The backend's module-level ``settings`` (postgresql_backend.settings) is captured at
    # import and would otherwise stay at the ambient compression-on default, so
    # _resolve_provision_vector would skip the pgvector extension + codec for this
    # compression-off fp32 seed (the codec is what encodes the vector column). Refresh the
    # binding to the compression-off settings before initialize(), mirroring the semantic and
    # chunking module refreshes below, so the backend provisions the fp32 vector layout.
    import app.backends.postgresql_backend as _pg_backend_module
    monkeypatch.setattr(_pg_backend_module, 'settings', get_settings())

    backend = create_backend(
        backend_type='postgresql', connection_string=pg_url,
    )
    await backend.initialize()
    try:
        # Apply the production base schema (context_entries etc.) and
        # then the semantic-search migration so vec_context_embeddings
        # is created. Both migrations are idempotent.
        import app.migrations.semantic as semantic_module
        from app.migrations.semantic import apply_semantic_search_migration
        from app.startup import init_database
        monkeypatch.setattr(semantic_module, 'settings', get_settings())
        await init_database(backend=backend)
        await apply_semantic_search_migration(backend=backend)
        import app.migrations.chunking as chunking_module
        from app.migrations.chunking import apply_chunking_migration
        monkeypatch.setattr(chunking_module, 'settings', get_settings())
        await apply_chunking_migration(backend=backend)

        repos = RepositoryContainer(backend)
        repo = EmbeddingRepository(backend)
        rng = np.random.default_rng(42)
        for i in range(n_docs):
            vec = rng.standard_normal(DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            cid, _ = await repos.context.store_with_deduplication(
                thread_id='pg-compress-e2e',
                source='user',
                content_type='text',
                text_content=f'doc-{i}',
                metadata=None,
            )
            await repo.store_chunked(
                cid,
                [
                    ChunkEmbedding(
                        embedding=vec.tolist(),
                        start_index=0,
                        end_index=len(f'doc-{i}'),
                    ),
                ],
                model='test-model',
            )
        return n_docs
    finally:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(backend.shutdown(), timeout=10.0)


async def _table_exists_pg(pg_url: str, table: str) -> bool:
    """Return True when ``table`` is present in the database at ``pg_url``."""
    conn = await asyncpg.connect(pg_url)
    try:
        result = await conn.fetchval(
            '''
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = $1
            )
            ''',
            table,
        )
        return bool(result)
    finally:
        await conn.close()


async def _read_provenance_pg(
    pg_url: str,
) -> tuple[str, int, str, int, int] | None:
    """Return the singleton compression_metadata row or None."""
    conn = await asyncpg.connect(pg_url)
    try:
        row = await conn.fetchrow(
            'SELECT provider, bits, variant, seed, dim '
            'FROM compression_metadata WHERE id = 1',
        )
    finally:
        await conn.close()
    if row is None:
        return None
    return (
        row['provider'],
        int(row['bits']),
        row['variant'],
        int(row['seed']),
        int(row['dim']),
    )


async def _index_exists_pg(pg_url: str, indexname: str) -> bool:
    """Return True when ``indexname`` is present in ``pg_indexes``.

    All asyncpg I/O happens inside a single coroutine so the underlying
    connection stays bound to a single event loop. Each ``asyncio.run``
    call creates a fresh loop, and asyncpg connections cannot be reused
    across loops (raises ``RuntimeError: ... attached to a different loop``).

    Args:
        pg_url: PostgreSQL connection string.
        indexname: Index name to look up in ``pg_indexes``.

    Returns:
        True when the index exists, False otherwise.
    """
    conn = await asyncpg.connect(pg_url)
    try:
        result = await conn.fetchval(
            'SELECT EXISTS ('
            '    SELECT 1 FROM pg_indexes WHERE indexname = $1'
            ')',
            indexname,
        )
    finally:
        await conn.close()
    return bool(result)


def test_compress_e2e_postgresql_dry_run(
    isolated_pg_db: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--compress --dry-run`` previews without mutating the source db."""
    asyncio.run(_seed_fp32_pg(isolated_pg_db, monkeypatch, n_docs=3))

    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    get_settings.cache_clear()

    rc = run_compress(isolated_pg_db, dry_run=True)

    assert rc == 0
    err = capsys.readouterr().err
    assert 'BACKUP REQUIRED' in err
    assert '[DRY-RUN]' in err
    assert asyncio.run(_table_exists_pg(isolated_pg_db, 'vec_context_embeddings'))


def test_compress_e2e_postgresql_execute_and_idempotent(
    isolated_pg_db: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--compress`` swaps the schema, drops HNSW, and is idempotent."""
    asyncio.run(_seed_fp32_pg(isolated_pg_db, monkeypatch, n_docs=3))

    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    get_settings.cache_clear()

    rc = run_compress(isolated_pg_db, dry_run=False)
    assert rc == 0

    assert asyncio.run(
        _table_exists_pg(isolated_pg_db, 'vec_context_embeddings_compressed'),
    )
    assert not asyncio.run(
        _table_exists_pg(isolated_pg_db, 'vec_context_embeddings'),
    )

    row = asyncio.run(_read_provenance_pg(isolated_pg_db))
    assert row is not None
    provider, bits, variant, seed, dim = row
    assert provider == 'turboquant'
    assert bits == 4
    assert variant == 'ip'
    assert seed == 42
    assert dim == DIM

    capsys.readouterr()
    rc2 = run_compress(isolated_pg_db, dry_run=False)
    assert rc2 == 0
    err2 = capsys.readouterr().err
    assert 'already present' in err2


def test_decompress_e2e_postgresql_execute(
    isolated_pg_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--decompress`` restores fp32 schema + HNSW index after a compress."""
    asyncio.run(_seed_fp32_pg(isolated_pg_db, monkeypatch, n_docs=3))

    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    get_settings.cache_clear()
    assert run_compress(isolated_pg_db, dry_run=False) == 0

    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    _reset_compression_cache()

    rc = run_decompress(isolated_pg_db, dry_run=False)
    assert rc == 0

    assert asyncio.run(_table_exists_pg(isolated_pg_db, 'vec_context_embeddings'))
    assert not asyncio.run(
        _table_exists_pg(isolated_pg_db, 'vec_context_embeddings_compressed'),
    )
    assert asyncio.run(_read_provenance_pg(isolated_pg_db)) is None

    # HNSW index re-created. Drive connect + fetchval + close from a
    # single ``asyncio.run`` invocation so the underlying asyncpg
    # connection lives entirely inside one event loop.
    assert asyncio.run(
        _index_exists_pg(isolated_pg_db, 'idx_vec_context_embeddings_hnsw'),
    )
