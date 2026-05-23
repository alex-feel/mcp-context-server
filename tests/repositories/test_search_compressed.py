"""Tests for ``EmbeddingRepository.search_compressed`` on SQLite.

Exercises the compressed read path end-to-end against a SQLite backend
prepared with real compressed payloads written via the production write
path. The provider produces real TurboQuant encodings so the test
verifies the SQL filter pipeline, the distance polarity, the
per-context aggregation, and the hydration step.

PostgreSQL coverage is in
``tests/integration/postgresql/test_search_compressed_postgresql.py``
which runs against a Docker-compose container.
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

DIM = 1024
SEED = 42

VariantT = Literal['mse', 'ip']
BackendFactory = Callable[
    [VariantT, int], Awaitable[tuple[StorageBackend, RepositoryContainer]],
]


def _to_variant(name: str) -> VariantT:
    """Resolve a string to the encoder's Literal type."""
    if name == 'ip':
        return 'ip'
    if name == 'mse':
        return 'mse'
    raise ValueError(f'unknown variant: {name!r}')


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
    """Flip the toggle, configure the provider, and refresh module caches."""
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
    """Return a factory that builds a fresh backend per (variant, bits) cell.

    Yields:
        A callable ``(variant, bits) -> (backend, repos)`` that the test
        body invokes to provision and seed a database. The factory keeps
        track of created backends and shuts them down at teardown.
    """
    created: list[StorageBackend] = []

    async def _build(
        variant: VariantT, bits: int,
    ) -> tuple[StorageBackend, RepositoryContainer]:
        _enable_compression(monkeypatch, variant=variant, bits=bits)
        db_path = tmp_path / f'compressed_search_{variant}_{bits}.db'
        # Bootstrap the schema directly (avoid sqlite-vec dependency).
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
        # Seed the singleton provenance row so the read path's
        # _get_cached_compression_metadata can find it.
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
    """Encode a unit-norm vector with the given compression settings."""
    from app.compression.providers.turboquant.encoder import encode

    return encode(
        vec[None, :].astype(np.float32),
        bits=bits,
        variant=variant,
        seed=SEED,
    ).to_bytes()


async def _seed_three_entries(
    repos: RepositoryContainer,
    repo: EmbeddingRepository,
    *,
    variant: VariantT,
    bits: int,
) -> tuple[str, str, str, NDArray[np.float32]]:
    """Insert three context entries with planted embeddings.

    The first two entries are planted near a chosen query vector; the
    third is sampled from a random direction so it should rank last.

    Returns:
        ``(cid_a, cid_b, cid_c, query_vec)`` where the query is the
        ground-truth top vector and cid_a is its closest neighbor.
    """
    rng = np.random.default_rng(SEED)
    query = rng.standard_normal(DIM).astype(np.float32)
    query /= np.linalg.norm(query)

    def _close(noise_scale: float) -> NDArray[np.float32]:
        noise = rng.standard_normal(DIM).astype(np.float32)
        v = query + noise_scale * noise
        v /= np.linalg.norm(v)
        return v.astype(np.float32)

    def _far() -> NDArray[np.float32]:
        v = rng.standard_normal(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        return v.astype(np.float32)

    docs = [_close(0.05), _close(0.10), _far()]

    cids: list[str] = []
    for i, vec in enumerate(docs):
        cid, _ = await repos.context.store_with_deduplication(
            thread_id='t-search',
            source='user',
            content_type='text',
            text_content=f'doc-{i}',
            metadata=None,
        )
        cids.append(cid)
        payload = _encode_vector(vec, variant=variant, bits=bits)
        await repo.store_chunked(
            cid,
            [
                ChunkEmbedding(
                    embedding=vec.tolist(),
                    start_index=0,
                    end_index=len(f'doc-{i}'),
                    payload=payload,
                ),
            ],
            model='test-model',
        )
    return cids[0], cids[1], cids[2], query


@pytest.mark.asyncio
@pytest.mark.parametrize('variant_name', ['ip', 'mse'])
async def test_search_compressed_orders_planted_first(
    variant_name: str,
    compressed_backend_factory: BackendFactory,
) -> None:
    """Planted neighbors out-rank the random background entry for both variants."""
    variant = _to_variant(variant_name)
    backend, repos = await compressed_backend_factory(variant, 4)
    repo = EmbeddingRepository(backend)
    cid_a, cid_b, cid_c, query = await _seed_three_entries(
        repos, repo, variant=variant, bits=4,
    )

    results, stats = await repo.search_compressed(
        query_embedding=query.tolist(),
        limit=3,
    )

    assert stats['backend'] == 'sqlite'
    assert stats['rows_returned'] == 3
    assert len(results) == 3
    returned_ids = [r['id'] for r in results]
    # Planted neighbors come first; the random doc lands last.
    assert cid_a in returned_ids[:2]
    assert cid_b in returned_ids[:2]
    assert returned_ids[-1] == cid_c
    # distances are monotone non-decreasing (smaller is closer).
    distances = [r['distance'] for r in results]
    assert distances == sorted(distances)


@pytest.mark.asyncio
async def test_search_compressed_dispatch_via_search(
    compressed_backend_factory: BackendFactory,
) -> None:
    """``search()`` dispatches to ``search_compressed`` when toggle is on."""
    backend, repos = await compressed_backend_factory('ip', 4)
    repo = EmbeddingRepository(backend)
    cid_a, cid_b, _cid_c, query = await _seed_three_entries(
        repos, repo, variant='ip', bits=4,
    )

    results, stats = await repo.search(
        query_embedding=query.tolist(),
        limit=2,
    )

    assert stats['backend'] == 'sqlite'
    assert stats['rows_returned'] == 2
    ids = {r['id'] for r in results}
    assert ids == {cid_a, cid_b}


@pytest.mark.asyncio
async def test_search_compressed_thread_filter(
    compressed_backend_factory: BackendFactory,
) -> None:
    """``thread_id`` filter narrows the candidate set before scoring."""
    backend, repos = await compressed_backend_factory('ip', 4)
    repo = EmbeddingRepository(backend)
    cid_a, _cid_b, _cid_c, query = await _seed_three_entries(
        repos, repo, variant='ip', bits=4,
    )

    # Seed one more entry in a DIFFERENT thread with a planted-close vector;
    # the filter should EXCLUDE it from the results.
    rng = np.random.default_rng(SEED + 1)
    other_vec = query + 0.05 * rng.standard_normal(DIM).astype(np.float32)
    other_vec /= np.linalg.norm(other_vec)
    other_cid, _ = await repos.context.store_with_deduplication(
        thread_id='other-thread',
        source='user',
        content_type='text',
        text_content='other thread entry',
        metadata=None,
    )
    await repo.store_chunked(
        other_cid,
        [
            ChunkEmbedding(
                embedding=other_vec.tolist(),
                start_index=0,
                end_index=10,
                payload=_encode_vector(other_vec, variant='ip', bits=4),
            ),
        ],
        model='test-model',
    )

    results, stats = await repo.search_compressed(
        query_embedding=query.tolist(),
        thread_id='t-search',
        limit=10,
    )

    returned_ids = {r['id'] for r in results}
    assert other_cid not in returned_ids
    assert cid_a in returned_ids
    assert stats['filters_applied'] >= 1


@pytest.mark.asyncio
async def test_search_compressed_no_candidates_returns_empty(
    compressed_backend_factory: BackendFactory,
) -> None:
    """Empty candidate set short-circuits to empty results without scoring."""
    backend, _repos = await compressed_backend_factory('ip', 4)
    repo = EmbeddingRepository(backend)
    # Use a thread_id that has no entries -- filter yields zero candidates.
    query = np.ones(DIM, dtype=np.float32)
    query /= np.linalg.norm(query)
    results, stats = await repo.search_compressed(
        query_embedding=query.tolist(),
        thread_id='nonexistent-thread',
        limit=10,
    )
    assert results == []
    assert stats['rows_returned'] == 0


@pytest.mark.asyncio
async def test_search_compressed_dimension_mismatch_raises(
    compressed_backend_factory: BackendFactory,
) -> None:
    """A query embedding with the wrong dim raises a clear RuntimeError."""
    backend, _repos = await compressed_backend_factory('ip', 4)
    repo = EmbeddingRepository(backend)
    with pytest.raises(RuntimeError, match='does not match'):
        await repo.search_compressed(
            query_embedding=[0.0] * (DIM - 1),
            limit=5,
        )


@pytest.mark.asyncio
async def test_search_compressed_offset_pagination(
    compressed_backend_factory: BackendFactory,
) -> None:
    """``offset`` and ``limit`` slice the ranked list as documented."""
    backend, repos = await compressed_backend_factory('ip', 4)
    repo = EmbeddingRepository(backend)
    _cid_a, cid_b, cid_c, query = await _seed_three_entries(
        repos, repo, variant='ip', bits=4,
    )

    results, stats = await repo.search_compressed(
        query_embedding=query.tolist(),
        limit=2,
        offset=1,
    )

    assert len(results) == 2
    returned = [r['id'] for r in results]
    # First element of the global ranking is excluded by offset=1, so the
    # window contains the second planted doc and the random doc.
    assert cid_b in returned
    assert cid_c in returned
    assert stats['rows_returned'] == 2
