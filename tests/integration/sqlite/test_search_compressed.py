"""End-to-end search_compressed integration test on SQLite.

Validates that the compressed semantic-search read path returns the
planted top-K with at least 0.85 overlap against the fp32 ground truth
on a small synthetic corpus. This complements the unit-scale tests in
``tests/repositories/test_search_compressed.py`` (which use 3 docs and
focus on routing/filters) by exercising the full pipeline end-to-end
with statistically meaningful recall measurements.
"""

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Generator
from pathlib import Path

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
N_QUERIES = 10
N_BACKGROUND = 300
TOP_K = 5
PLANT_NOISE = 0.10
SEED = 42
OVERLAP_FLOOR = 0.85


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset settings + compression caches around every test."""
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()


@pytest_asyncio.fixture
async def compressed_corpus_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[
    tuple[
        StorageBackend,
        RepositoryContainer,
        NDArray[np.float32],
        NDArray[np.float32],
        dict[int, set[str]],
    ],
    None,
]:
    """Provision a SQLite backend with N_QUERIES * TOP_K + N_BACKGROUND
    compressed embeddings + N_QUERIES query vectors + ground-truth.

    Yields:
        Tuple of ``(backend, repos, queries, docs, ground_truth)``.
        ``ground_truth`` maps query index to the set of context_ids
        that were planted near that query.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', str(SEED))
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', str(DIM))
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())

    db_path = tmp_path / 'search_compressed_e2e.db'
    from app.schemas import load_schema

    conn = sqlite3.connect(str(db_path))
    try:
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
            );
            ''',
        )
    finally:
        conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    await apply_compression_migration(backend=backend)
    from app.compression.provenance import insert_compression_metadata
    from app.compression.types import CompressionMetadata
    await insert_compression_metadata(
        backend,
        CompressionMetadata(
            provider='turboquant', bits=4, variant='ip',
            seed=SEED, dim=DIM,
        ),
    )

    # Build the corpus: planted near-neighbors per query + random background.
    rng = np.random.default_rng(SEED)
    queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    repos = RepositoryContainer(backend)
    repo = EmbeddingRepository(backend)
    from app.compression.providers.turboquant.encoder import encode

    ground_truth: dict[int, set[str]] = {q: set() for q in range(N_QUERIES)}
    docs: list[NDArray[np.float32]] = []

    # Planted neighbors per query. Each text_content must be unique
    # because ``store_with_deduplication`` would otherwise collapse
    # identical (thread, source, text) tuples into a single entry.
    doc_counter = 0
    for q_idx in range(N_QUERIES):
        q = queries[q_idx]
        for _ in range(TOP_K):
            noise = rng.standard_normal(DIM).astype(np.float32)
            doc = q + PLANT_NOISE * noise
            doc /= np.linalg.norm(doc)
            doc = doc.astype(np.float32)
            docs.append(doc)
            cid, _ = await repos.context.store_with_deduplication(
                thread_id='search-e2e',
                source='user',
                content_type='text',
                text_content=f'planted-q{q_idx}-{doc_counter}',
                metadata=None,
            )
            doc_counter += 1
            ground_truth[q_idx].add(cid)
            payload = encode(
                doc[None, :], bits=4, variant='ip', seed=SEED,
            ).to_bytes()
            await repo.store_chunked(
                cid,
                [
                    ChunkEmbedding(
                        embedding=doc.tolist(),
                        start_index=0,
                        end_index=10,
                        payload=payload,
                    ),
                ],
                model='test-model',
            )

    # Background documents (unit-random; should not appear in top-K).
    for bg_idx in range(N_BACKGROUND):
        bg = rng.standard_normal(DIM).astype(np.float32)
        bg /= np.linalg.norm(bg)
        bg = bg.astype(np.float32)
        docs.append(bg)
        cid, _ = await repos.context.store_with_deduplication(
            thread_id='search-e2e',
            source='user',
            content_type='text',
            text_content=f'background-{bg_idx}',
            metadata=None,
        )
        payload = encode(
            bg[None, :], bits=4, variant='ip', seed=SEED,
        ).to_bytes()
        await repo.store_chunked(
            cid,
            [
                ChunkEmbedding(
                    embedding=bg.tolist(),
                    start_index=0,
                    end_index=10,
                    payload=payload,
                ),
            ],
            model='test-model',
        )

    docs_arr = np.stack(docs).astype(np.float32)
    yield backend, repos, queries, docs_arr, ground_truth

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=5.0)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_compressed_recall_on_real_storage(
    compressed_corpus_backend: tuple[
        StorageBackend,
        RepositoryContainer,
        NDArray[np.float32],
        NDArray[np.float32],
        dict[int, set[str]],
    ],
) -> None:
    """search_compressed returns planted top-K with >= OVERLAP_FLOOR overlap."""
    backend, _repos, queries, _docs, ground_truth = compressed_corpus_backend
    repo = EmbeddingRepository(backend)

    overlaps: list[float] = []
    for q_idx in range(N_QUERIES):
        q = queries[q_idx]
        results, stats = await repo.search_compressed(
            query_embedding=q.tolist(),
            limit=TOP_K,
            thread_id='search-e2e',
        )
        assert stats['backend'] == 'sqlite'
        assert len(results) == TOP_K
        returned_ids = {r['id'] for r in results}
        truth = ground_truth[q_idx]
        overlap = len(returned_ids & truth) / TOP_K
        overlaps.append(overlap)

    min_overlap = min(overlaps)
    mean_overlap = float(np.mean(overlaps))
    print(
        f'\n[search_compressed e2e] min_overlap={min_overlap:.3f} '
        f'mean_overlap={mean_overlap:.3f}',
    )
    assert min_overlap >= OVERLAP_FLOOR, (
        f'recall regression: min overlap {min_overlap:.3f} < {OVERLAP_FLOOR}; '
        f'per-query overlaps={overlaps}'
    )
