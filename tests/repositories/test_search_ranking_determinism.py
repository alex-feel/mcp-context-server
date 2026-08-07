"""Regression tests for deterministic ordering of ranked search results.

Every ranked search (FTS, semantic fp32, semantic compressed, RRF fusion) applies the
caller's ``limit``/``offset`` to a score-ordered list. Scores tie routinely -- identical or
near-identical documents produce identical bm25/ts_rank_cd scores and identical embedding
distances -- and a score-only ordering leaves tied rows in whatever order the scan produced.
That order is not stable: on PostgreSQL an UPDATE writes a new physical tuple (MVCC), so an
unrelated metadata-only write reshuffles equal-score rows, and on SQLite nothing in the SQL
promises an order at all. A client that reads page 1, then page 2 after such a write, then
silently never receives one of the tied rows -- or receives one twice.

The fix is an explicit UNIQUE secondary sort key at every ranked ordering, mirroring the
browse path's ``ORDER BY created_at DESC, id DESC``. These tests pin that contract on the
locally executable SQLite paths and pin the PostgreSQL SQL text, which needs a live server
to execute (the cross-backend behavior is covered by the real-server integration harness).
"""

import importlib.util
import inspect
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.backends import create_backend
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.fts_repository import FtsRepository

requires_sqlite_vec = pytest.mark.skipif(
    importlib.util.find_spec('sqlite_vec') is None,
    reason='sqlite-vec package not installed',
)

# Byte-identical text in every seeded entry, so the FTS engine scores them exactly equal.
_TIED_TEXT = 'zulu alpha beta ranking parity document'

# Ids chosen so the INSERT order differs from both the ascending and the descending id
# order: any ordering that merely reflects insertion order fails the assertions below.
_ID_LOW = '11111111111111111111111111111111'
_ID_MID = '22222222222222222222222222222222'
_ID_HIGH = '33333333333333333333333333333333'
_INSERT_ORDER = (_ID_MID, _ID_LOW, _ID_HIGH)


@pytest_asyncio.fixture
async def fts_tied_repos(tmp_path: Path) -> AsyncGenerator[RepositoryContainer, None]:
    """SQLite backend whose FTS index holds three entries with byte-identical text.

    Yields:
        RepositoryContainer whose FtsRepository searches the tied documents.
    """
    from app.schemas import load_schema

    db_path = tmp_path / 'fts_ranking_determinism.db'
    migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_fts_sqlite.sql'
    fts_sql = migration_path.read_text(encoding='utf-8').replace('{TOKENIZER}', 'porter unicode61')

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(load_schema('sqlite'))
        conn.executescript(fts_sql)
        for entry_id in _INSERT_ORDER:
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                'VALUES (?, ?, ?, ?, ?)',
                (entry_id, 'tied-thread', 'agent', 'text', _TIED_TEXT),
            )
        conn.commit()
    finally:
        conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    repos = RepositoryContainer(backend)
    try:
        yield repos
    finally:
        await backend.shutdown()


class TestFtsTiedScoreOrdering:
    """SQLite FTS results with equal bm25 scores order on the unique id."""

    @pytest.mark.asyncio
    async def test_tied_scores_order_by_id_descending(
        self, fts_tied_repos: RepositoryContainer,
    ) -> None:
        """All three documents score identically, so the id decides -- newest id first."""
        results, _stats = await fts_tied_repos.fts.search(query='zulu', mode='match', limit=10)

        scores = [row['score'] for row in results]
        assert len(set(scores)) == 1, 'seeded documents must tie for this test to mean anything'
        assert [row['id'] for row in results] == [_ID_HIGH, _ID_MID, _ID_LOW]

    @pytest.mark.asyncio
    async def test_tied_scores_paginate_without_gaps_or_duplicates(
        self, fts_tied_repos: RepositoryContainer,
    ) -> None:
        """Paging one tied document at a time yields the full set exactly once."""
        unpaginated, _ = await fts_tied_repos.fts.search(query='zulu', mode='match', limit=10)
        expected = [row['id'] for row in unpaginated]

        paged: list[str] = []
        for offset in range(len(expected)):
            page, _ = await fts_tied_repos.fts.search(query='zulu', mode='match', limit=1, offset=offset)
            assert len(page) == 1
            paged.append(page[0]['id'])

        assert paged == expected
        assert len(set(paged)) == len(expected)


@pytest.mark.asyncio
class TestSemanticTiedDistanceOrdering:
    """SQLite fp32 semantic results with equal distances order on the unique context id."""

    async def _seed_identical_embeddings(
        self, backend: StorageBackend, embedding_dim: int,
    ) -> EmbeddingRepository:
        """Insert three entries carrying the SAME embedding, so distances tie exactly.

        Args:
            backend: Initialized SQLite backend with the embedding tables.
            embedding_dim: Configured embedding dimension.

        Returns:
            The repository bound to the seeded backend.
        """
        def _insert(conn: sqlite3.Connection) -> None:
            for entry_id in _INSERT_ORDER:
                conn.execute(
                    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (entry_id, 'tied-thread', 'agent', 'text', _TIED_TEXT),
                )

        await backend.execute_write(_insert)

        embedding_repo = EmbeddingRepository(backend)
        for entry_id in _INSERT_ORDER:
            await embedding_repo.store(
                context_id=entry_id,
                embedding=[0.25] * embedding_dim,
                model='test-model',
            )
        return embedding_repo

    @requires_sqlite_vec
    async def test_tied_distances_order_by_context_id_ascending(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Equal distances resolve on the id, not on the order the scan produced rows."""
        embedding_repo = await self._seed_identical_embeddings(async_db_with_embeddings, embedding_dim)

        results, _stats = await embedding_repo.search(
            query_embedding=[0.25] * embedding_dim,
            limit=10,
        )

        distances = [row['distance'] for row in results]
        assert len(set(distances)) == 1, 'seeded embeddings must tie for this test to mean anything'
        assert [row['id'] for row in results] == [_ID_LOW, _ID_MID, _ID_HIGH]

    @requires_sqlite_vec
    async def test_tied_distances_paginate_without_gaps_or_duplicates(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Paging one tied result at a time yields the full set exactly once."""
        embedding_repo = await self._seed_identical_embeddings(async_db_with_embeddings, embedding_dim)

        unpaginated, _ = await embedding_repo.search(query_embedding=[0.25] * embedding_dim, limit=10)
        expected = [row['id'] for row in unpaginated]

        paged: list[str] = []
        for offset in range(len(expected)):
            page, _ = await embedding_repo.search(
                query_embedding=[0.25] * embedding_dim,
                limit=1,
                offset=offset,
            )
            assert len(page) == 1
            paged.append(page[0]['id'])

        assert paged == expected
        assert len(set(paged)) == len(expected)


class TestRankedStatementsCarryTheTiebreak:
    """Pin the secondary sort key in the ranked SQL of BOTH backends.

    PostgreSQL is where the defect bites hardest -- MVCC rewrites a tuple on every UPDATE, so
    a metadata-only write reorders equal-score rows even though no score changed -- and those
    statements need a live server to execute, so only the real-server integration suite can
    run them. The SQLite statements are executed by the behavior tests above, but a query plan
    that happens to emit tied rows in id order today would hide a dropped tiebreak tomorrow,
    so the statements are pinned for both backends here.
    """

    def test_fts_postgresql_orders_by_score_then_id(self) -> None:
        """Both the ranked subquery and the outer re-sort carry the id tiebreak."""
        src = inspect.getsource(FtsRepository._search_postgresql)

        assert 'ORDER BY score DESC, ce.id DESC' in src
        assert 'ORDER BY sub.score DESC, sub.id DESC' in src

    def test_fts_sqlite_orders_by_score_then_id(self) -> None:
        """The FTS5 statement carries the same tiebreak as its PostgreSQL counterpart."""
        src = inspect.getsource(FtsRepository._search_sqlite)

        assert 'ORDER BY score DESC, ce.id DESC' in src

    def test_both_semantic_branches_order_by_distance_then_context_id(self) -> None:
        """SQLite and PostgreSQL ranked output both resolve tied distances on the id."""
        src = inspect.getsource(EmbeddingRepository.search)

        assert src.count('ORDER BY bc.best_distance ASC, bc.context_id ASC') == 2

    def test_best_chunk_pick_is_deterministic_on_both_backends(self) -> None:
        """A tie between two equidistant chunks of one context resolves on start_index.

        The chunk chosen here is reported as matched_chunk_start/matched_chunk_end and feeds
        rerank passage extraction, so leaving it to scan order makes the response text for an
        unchanged entry vary between identical queries.
        """
        src = inspect.getsource(EmbeddingRepository.search)

        # SQLite picks the best chunk with a ROW_NUMBER window; PostgreSQL with DISTINCT ON.
        assert 'ORDER BY cd.distance, cd.start_index' in src
        assert 'ORDER BY context_id, distance, start_index' in src
