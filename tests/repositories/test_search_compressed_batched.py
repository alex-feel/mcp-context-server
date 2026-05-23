"""Tests for the batched-provider-call structure of ``search_compressed``.

Verifies that the compressed read path invokes the provider's scoring
method exactly ONCE per query (regardless of the number of candidate
rows), that the batched implementation returns IDENTICAL results to a
per-row reference loop, and that mismatched stored payloads vs metadata
variant surface as a runtime corruption error.
"""

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import pytest_asyncio
from numpy.typing import NDArray

from app.backends import StorageBackend
from app.backends import create_backend
from app.migrations.compression import apply_compression_migration
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository import ChunkEmbedding
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.embedding_repository import _reset_compression_cache
from app.settings import get_settings

DIM = 256
SEED = 42

VariantT = Literal['mse', 'ip']
BackendFactory = Callable[
    [VariantT, int], Awaitable[tuple[StorageBackend, RepositoryContainer]],
]


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset the settings singleton + compression caches around every test."""
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()


def _enable_compression(
    monkeypatch: pytest.MonkeyPatch,
    *,
    variant: VariantT = 'ip',
    bits: int = 4,
) -> None:
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', str(SEED))
    monkeypatch.setenv('COMPRESSION_BITS', str(bits))
    monkeypatch.setenv('COMPRESSION_VARIANT', variant)
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())


@pytest_asyncio.fixture
async def compressed_backend_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[BackendFactory, None]:
    """Build a fresh SQLite backend per (variant, bits) cell."""
    created: list[StorageBackend] = []

    async def _build(
        variant: VariantT, bits: int,
    ) -> tuple[StorageBackend, RepositoryContainer]:
        _enable_compression(monkeypatch, variant=variant, bits=bits)
        db_path = tmp_path / f'batched_search_{variant}_{bits}.db'
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        from app.schemas import load_schema

        conn.executescript(load_schema('sqlite'))
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
            )
            ''',
        )
        conn.close()

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        await apply_compression_migration(backend=backend)
        from app.compression.provenance import insert_compression_metadata
        from app.compression.types import CompressionMetadata

        await insert_compression_metadata(
            backend,
            CompressionMetadata(
                provider='turboquant',
                bits=bits,
                variant=variant,
                seed=SEED,
                dim=DIM,
            ),
        )
        repos = RepositoryContainer(backend)
        created.append(backend)
        return backend, repos

    yield _build

    for backend in created:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(backend.shutdown(), timeout=5.0)


def _encode_vector(
    vec: NDArray[np.float32], *, variant: VariantT, bits: int,
) -> bytes:
    from app.compression.providers.turboquant.encoder import encode

    return encode(
        vec[None, :].astype(np.float32),
        bits=bits,
        variant=variant,
        seed=SEED,
    ).to_bytes()


async def _seed_random_corpus(
    repos: RepositoryContainer,
    repo: EmbeddingRepository,
    *,
    n: int,
    variant: VariantT,
    bits: int,
) -> tuple[list[str], NDArray[np.float32]]:
    """Seed ``n`` random unit-norm vectors plus a fixed query."""
    rng = np.random.default_rng(SEED)
    query = rng.standard_normal(DIM).astype(np.float32)
    query /= np.linalg.norm(query)

    cids: list[str] = []
    for i in range(n):
        v = rng.standard_normal(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        cid, _ = await repos.context.store_with_deduplication(
            thread_id='t-batched',
            source='user',
            content_type='text',
            text_content=f'doc-{i}',
            metadata=None,
        )
        cids.append(cid)
        payload = _encode_vector(v, variant=variant, bits=bits)
        await repo.store_chunked(
            cid,
            [
                ChunkEmbedding(
                    embedding=v.tolist(),
                    start_index=0,
                    end_index=len(f'doc-{i}'),
                    payload=payload,
                ),
            ],
            model='test-model',
        )
    return cids, query


# ---------------------------------------------------------------------------
# Provider invocation counters: prove the per-row loop is collapsed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_compressed_invokes_provider_once_for_ip(
    compressed_backend_factory: BackendFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For variant='ip', ``estimate_inner_product_sync`` is called exactly ONCE."""
    backend, repos = await compressed_backend_factory('ip', 4)
    repo = EmbeddingRepository(backend)
    _cids, query = await _seed_random_corpus(
        repos, repo, n=50, variant='ip', bits=4,
    )

    from app.compression import factory as compression_factory

    real_provider = await compression_factory.get_cached_compression_provider()
    ip_call_count = 0
    real_estimate = real_provider.estimate_inner_product_sync

    def _counting_estimate(*args, **kwargs):
        nonlocal ip_call_count
        ip_call_count += 1
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(
        real_provider, 'estimate_inner_product_sync', _counting_estimate,
    )

    results, stats = await repo.search_compressed(
        query_embedding=query.tolist(), limit=10,
    )
    assert ip_call_count == 1, (
        f'estimate_inner_product_sync was invoked {ip_call_count} times; '
        f'the batched implementation must call it exactly ONCE per query.'
    )
    assert stats['rows_returned'] == 10
    assert len(results) == 10


@pytest.mark.asyncio
async def test_search_compressed_invokes_provider_once_for_mse(
    compressed_backend_factory: BackendFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For variant='mse', ``decode_sync`` is called exactly ONCE."""
    backend, repos = await compressed_backend_factory('mse', 4)
    repo = EmbeddingRepository(backend)
    _cids, query = await _seed_random_corpus(
        repos, repo, n=50, variant='mse', bits=4,
    )

    from app.compression import factory as compression_factory

    real_provider = await compression_factory.get_cached_compression_provider()
    decode_call_count = 0
    real_decode = real_provider.decode_sync

    def _counting_decode(*args, **kwargs):
        nonlocal decode_call_count
        decode_call_count += 1
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(real_provider, 'decode_sync', _counting_decode)

    results, stats = await repo.search_compressed(
        query_embedding=query.tolist(), limit=10,
    )
    assert decode_call_count == 1, (
        f'decode_sync was invoked {decode_call_count} times; the batched '
        f'implementation must call it exactly ONCE per query.'
    )
    assert stats['rows_returned'] == 10
    assert len(results) == 10


# ---------------------------------------------------------------------------
# Result equivalence vs a per-row reference loop.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_compressed_results_unchanged_after_batching(
    compressed_backend_factory: BackendFactory,
) -> None:
    """Batched implementation returns identical results to the per-row loop."""
    backend, repos = await compressed_backend_factory('ip', 4)
    repo = EmbeddingRepository(backend)
    cids, query = await _seed_random_corpus(
        repos, repo, n=100, variant='ip', bits=4,
    )

    results, stats = await repo.search_compressed(
        query_embedding=query.tolist(), limit=100,
    )
    assert stats['rows_returned'] == len(cids)

    # Build the per-row reference by replaying the scoring loop directly
    # against the stored payloads.
    from app.compression import factory as compression_factory

    provider = await compression_factory.get_cached_compression_provider()
    query_matrix = np.asarray([query], dtype=np.float32)

    def _read_payloads(conn: sqlite3.Connection) -> list[tuple[str, bytes]]:
        cur = conn.execute(
            'SELECT context_id, payload FROM vec_context_embeddings_compressed '
            'ORDER BY context_id',
        )
        return [(str(r[0]), bytes(r[1])) for r in cur.fetchall()]

    rows = await backend.execute_read(_read_payloads)
    per_row_distances_by_ctx: dict[str, float] = {}
    for ctx_id, payload_bytes in rows:
        d = -float(
            provider.estimate_inner_product_sync(payload_bytes, query_matrix)[0, 0],
        )
        # Each context_id has exactly one chunk in this corpus, so direct
        # assignment is the per-row distance.
        per_row_distances_by_ctx[ctx_id] = d

    batched_distances_by_ctx = {r['id']: r['distance'] for r in results}
    assert set(per_row_distances_by_ctx) == set(batched_distances_by_ctx)
    for ctx_id in per_row_distances_by_ctx:
        np.testing.assert_allclose(
            batched_distances_by_ctx[ctx_id],
            per_row_distances_by_ctx[ctx_id],
            rtol=0,
            atol=1e-4,
        )
    # Order parity: the batched ranked list matches the per-row ranked list.
    import operator

    per_row_ranked = sorted(
        per_row_distances_by_ctx.items(), key=operator.itemgetter(1),
    )
    batched_ranked = [(r['id'], r['distance']) for r in results]
    assert [cid for cid, _ in per_row_ranked] == [cid for cid, _ in batched_ranked]


# ---------------------------------------------------------------------------
# Storage corruption detection.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_compressed_storage_corruption_detection(
    compressed_backend_factory: BackendFactory,
) -> None:
    """Stored MSE payload under variant='ip' provenance raises a RuntimeError."""
    backend, repos = await compressed_backend_factory('ip', 4)
    repo = EmbeddingRepository(backend)
    _cids, query = await _seed_random_corpus(
        repos, repo, n=5, variant='ip', bits=4,
    )

    # Overwrite one stored payload with an MSE-encoded blob.
    rng = np.random.default_rng(SEED + 100)
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    mse_payload = _encode_vector(v, variant='mse', bits=4)

    def _overwrite_one(conn: sqlite3.Connection) -> None:
        conn.execute(
            'UPDATE vec_context_embeddings_compressed SET payload = ? '
            'WHERE id = (SELECT MIN(id) FROM vec_context_embeddings_compressed)',
            (mse_payload,),
        )

    await backend.execute_write(_overwrite_one)

    with pytest.raises(RuntimeError, match='storage corruption'):
        await repo.search_compressed(
            query_embedding=query.tolist(), limit=10,
        )
