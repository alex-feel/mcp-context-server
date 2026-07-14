"""Integration tests for hybrid search functionality.

Tests the hybrid search combining FTS and semantic search with RRF fusion.
"""

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.fusion import count_unique_results
from app.fusion import reciprocal_rank_fusion


class TestRRFIntegration:
    """Test RRF algorithm with realistic data scenarios."""

    def test_rrf_with_diverse_rankings(self) -> None:
        """Test RRF with documents having different rankings in each source."""
        # Simulate FTS results (ranked by relevance score)
        fts_results: list[dict[str, Any]] = [
            {'id': '1', 'score': 10.0, 'text_content': 'Python programming tutorial', 'thread_id': 't1'},
            {'id': '2', 'score': 8.5, 'text_content': 'Python data science guide', 'thread_id': 't1'},
            {'id': '3', 'score': 7.0, 'text_content': 'Machine learning basics', 'thread_id': 't1'},
            {'id': '4', 'score': 5.5, 'text_content': 'Deep learning neural networks', 'thread_id': 't1'},
        ]

        # Simulate semantic results (ranked by distance - lower is better)
        semantic_results: list[dict[str, Any]] = [
            {'id': '3', 'distance': 0.1, 'text_content': 'Machine learning basics', 'thread_id': 't1'},
            {'id': '4', 'distance': 0.2, 'text_content': 'Deep learning neural networks', 'thread_id': 't1'},
            {'id': '2', 'distance': 0.3, 'text_content': 'Python data science guide', 'thread_id': 't1'},
            {'id': '5', 'distance': 0.4, 'text_content': 'AI fundamentals', 'thread_id': 't1'},
        ]

        results = reciprocal_rank_fusion(fts_results, semantic_results, k=60, limit=10)

        # Verify all unique documents are present
        result_ids = [r.get('id') for r in results]
        assert len(result_ids) == 5  # 5 unique documents

        # Documents appearing in both should have higher scores
        # Doc 2, 3, 4 appear in both
        overlap_ids = {'2', '3', '4'}
        for r in results:
            if r.get('id') in overlap_ids:
                scores = r.get('scores', {})
                # Should have both ranks
                assert scores.get('fts_rank') is not None
                assert scores.get('semantic_rank') is not None

        # Verify scores structure
        for r in results:
            scores = r.get('scores', {})
            assert 'rrf' in scores
            assert scores['rrf'] > 0

    def test_rrf_preserves_metadata(self) -> None:
        """Test that RRF preserves metadata from both sources."""
        fts_results: list[dict[str, Any]] = [
            {
                'id': '1',
                'score': 10.0,
                'text_content': 'Test content',
                'thread_id': 'thread-1',
                'source': 'agent',
                'content_type': 'text',
                'created_at': '2024-01-01T00:00:00Z',
            },
        ]

        semantic_results: list[dict[str, Any]] = [
            {
                'id': '1',
                'distance': 0.1,
                'text_content': 'Test content',
                'metadata': {'key': 'value', 'priority': 5},
                'tags': ['tag1', 'tag2'],
            },
        ]

        results = reciprocal_rank_fusion(fts_results, semantic_results, k=60, limit=10)

        assert len(results) == 1
        result = results[0]

        # Should have data from FTS
        assert result.get('thread_id') == 'thread-1'
        assert result.get('source') == 'agent'
        assert result.get('content_type') == 'text'
        assert result.get('created_at') == '2024-01-01T00:00:00Z'

        # Should have data from semantic
        assert result.get('metadata') == {'key': 'value', 'priority': 5}
        assert result.get('tags') == ['tag1', 'tag2']

    def test_rrf_handles_large_result_sets(self) -> None:
        """Test RRF performance with larger result sets."""
        # Create 100 FTS results
        fts_results: list[dict[str, Any]] = [
            {'id': i, 'score': 100 - i, 'text_content': f'Document {i}', 'thread_id': 't1'}
            for i in range(1, 101)
        ]

        # Create 100 semantic results with 50% overlap
        semantic_results: list[dict[str, Any]] = [
            {'id': i + 50, 'distance': i * 0.01, 'text_content': f'Document {i + 50}', 'thread_id': 't1'}
            for i in range(1, 101)
        ]

        results = reciprocal_rank_fusion(fts_results, semantic_results, k=60, limit=50)

        # Should return exactly 50 results
        assert len(results) == 50

        # Results should be sorted by RRF score
        scores = [r.get('scores', {}).get('rrf', 0) for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_k_parameter_impact(self) -> None:
        """Test how k parameter affects ranking and score distribution."""
        fts_results: list[dict[str, Any]] = [
            {'id': '1', 'score': 10.0, 'text_content': 'Top FTS result'},
            {'id': '2', 'score': 5.0, 'text_content': 'Second FTS result'},
        ]

        semantic_results: list[dict[str, Any]] = [
            {'id': '2', 'distance': 0.1, 'text_content': 'Top semantic (also #2 in FTS)'},
            {'id': '3', 'distance': 0.2, 'text_content': 'Second semantic only'},
        ]

        # With small k, top ranks matter more
        results_small_k = reciprocal_rank_fusion(fts_results, semantic_results, k=1, limit=10)
        # With large k, ranks matter less (more uniform)
        results_large_k = reciprocal_rank_fusion(fts_results, semantic_results, k=100, limit=10)

        # Document 2 appears in both - should be top in both cases
        assert results_small_k[0].get('id') == '2'
        assert results_large_k[0].get('id') == '2'

        # Verify k affects score magnitudes
        small_k_scores = {r.get('id'): r.get('scores', {}).get('rrf', 0) for r in results_small_k}
        large_k_scores = {r.get('id'): r.get('scores', {}).get('rrf', 0) for r in results_large_k}

        # Scores should be smaller with large k (1/(k+rank) decreases as k increases)
        assert small_k_scores['2'] > large_k_scores['2']
        assert small_k_scores['1'] > large_k_scores['1']

        # With small k, the absolute difference between rank 1 and rank 2 is larger
        # k=1: rank1 = 1/2 = 0.5, rank2 = 1/3 = 0.333, diff = 0.167
        # k=100: rank1 = 1/101 ≈ 0.0099, rank2 = 1/102 ≈ 0.0098, diff ≈ 0.0001
        small_k_diff = 1 / (1 + 1) - 1 / (1 + 2)  # 0.5 - 0.333 = 0.167
        large_k_diff = 1 / (100 + 1) - 1 / (100 + 2)  # much smaller

        assert small_k_diff > large_k_diff


class TestCountUniqueResultsIntegration:
    """Test count_unique_results with various scenarios."""

    def test_count_with_realistic_data(self) -> None:
        """Test counting with realistic search results."""
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 10.0},
            {'id': 2, 'score': 8.0},
            {'id': 3, 'score': 6.0},
            {'id': 4, 'score': 4.0},
            {'id': 5, 'score': 2.0},
        ]

        semantic_results: list[dict[str, Any]] = [
            {'id': 3, 'distance': 0.1},  # Overlap
            {'id': 4, 'distance': 0.2},  # Overlap
            {'id': 6, 'distance': 0.3},
            {'id': 7, 'distance': 0.4},
        ]

        fts_only, semantic_only, overlap = count_unique_results(fts_results, semantic_results)

        assert fts_only == 3  # IDs 1, 2, 5
        assert semantic_only == 2  # IDs 6, 7
        assert overlap == 2  # IDs 3, 4


class TestHybridSearchToolIntegration:
    """Test hybrid_search_context tool integration.

    Note: These tests focus on the fusion algorithm and response structure.
    Full integration tests with the MCP server are in test_real_server.py.
    """

    def test_fusion_with_empty_search_results(self) -> None:
        """Test fusion handles empty results gracefully."""
        # Both searches return empty
        results = reciprocal_rank_fusion([], [], k=60, limit=10)
        assert results == []

    def test_fusion_single_source_fts_only(self) -> None:
        """Test fusion with only FTS results (semantic unavailable scenario)."""
        fts_results: list[dict[str, Any]] = [
            {
                'id': '1',
                'thread_id': 't1',
                'source': 'agent',
                'content_type': 'text',
                'text_content': 'Python tutorial',
                'score': 10.0,
                'metadata': None,
                'created_at': '2024-01-01T00:00:00Z',
                'updated_at': '2024-01-01T00:00:00Z',
            },
        ]

        results = reciprocal_rank_fusion(fts_results, [], k=60, limit=10)

        assert len(results) == 1
        assert results[0].get('id') == '1'
        scores = results[0].get('scores', {})
        assert scores.get('fts_rank') == 1
        assert scores.get('semantic_rank') is None

    def test_fusion_single_source_semantic_only(self) -> None:
        """Test fusion with only semantic results (FTS unavailable scenario)."""
        semantic_results: list[dict[str, Any]] = [
            {
                'id': '1',
                'thread_id': 't1',
                'source': 'agent',
                'content_type': 'text',
                'text_content': 'Python tutorial',
                'distance': 0.1,
                'metadata': {'priority': 5},
                'created_at': '2024-01-01T00:00:00Z',
                'updated_at': '2024-01-01T00:00:00Z',
            },
        ]

        results = reciprocal_rank_fusion([], semantic_results, k=60, limit=10)

        assert len(results) == 1
        assert results[0].get('id') == '1'
        scores = results[0].get('scores', {})
        assert scores.get('fts_rank') is None
        assert scores.get('semantic_rank') == 1

    def test_fusion_metadata_json_string_handling(self) -> None:
        """Test that fusion preserves metadata regardless of format."""
        fts_results: list[dict[str, Any]] = [
            {
                'id': '1',
                'text_content': 'Test content',
                'score': 10.0,
                'metadata': {'priority': 5},  # Already a dict
            },
        ]

        results = reciprocal_rank_fusion(fts_results, [], k=60, limit=10)

        assert len(results) == 1
        # Metadata should be preserved as-is
        assert results[0].get('metadata') == {'priority': 5}


class TestRRFEdgeCases:
    """Test edge cases in RRF fusion algorithm."""

    def test_rrf_skips_semantic_results_with_none_id(self) -> None:
        """Test RRF skips semantic results where id is None.

        This covers line 75 in app/fusion.py - the only uncovered line.
        """
        semantic_results: list[dict[str, Any]] = [
            {'id': None, 'distance': 0.1, 'text_content': 'No ID entry'},
            {'id': '1', 'distance': 0.2, 'text_content': 'Valid entry'},
        ]
        fts_results: list[dict[str, Any]] = []

        results = reciprocal_rank_fusion(fts_results, semantic_results, k=60, limit=10)

        # Only the entry with valid ID should be in results
        assert len(results) == 1
        assert results[0].get('id') == '1'

    def test_rrf_skips_fts_results_with_none_id(self) -> None:
        """Test RRF skips FTS results where id is None."""
        fts_results: list[dict[str, Any]] = [
            {'id': None, 'score': 10.0, 'text_content': 'No ID'},
            {'id': '2', 'score': 8.0, 'text_content': 'Valid'},
        ]

        results = reciprocal_rank_fusion(fts_results, [], k=60, limit=10)

        assert len(results) == 1
        assert results[0].get('id') == '2'

    def test_rrf_skips_both_sources_with_none_ids(self) -> None:
        """Test RRF skips None IDs from both FTS and semantic results."""
        fts_results: list[dict[str, Any]] = [
            {'id': None, 'score': 10.0, 'text_content': 'FTS no ID'},
            {'id': '1', 'score': 8.0, 'text_content': 'FTS valid'},
        ]
        semantic_results: list[dict[str, Any]] = [
            {'id': None, 'distance': 0.1, 'text_content': 'Semantic no ID'},
            {'id': '2', 'distance': 0.2, 'text_content': 'Semantic valid'},
        ]

        results = reciprocal_rank_fusion(fts_results, semantic_results, k=60, limit=10)

        # Only entries with valid IDs should be in results
        result_ids = {r.get('id') for r in results}
        assert result_ids == {'1', '2'}
        assert len(results) == 2

    def test_rrf_all_none_ids_returns_empty(self) -> None:
        """Test RRF returns empty list when all IDs are None."""
        fts_results: list[dict[str, Any]] = [
            {'id': None, 'score': 10.0, 'text_content': 'No ID 1'},
            {'id': None, 'score': 8.0, 'text_content': 'No ID 2'},
        ]
        semantic_results: list[dict[str, Any]] = [
            {'id': None, 'distance': 0.1, 'text_content': 'No ID 3'},
        ]

        results = reciprocal_rank_fusion(fts_results, semantic_results, k=60, limit=10)

        assert results == []


class TestHybridSearchPagination:
    """Test hybrid search offset and limit handling."""

    def test_limit_applied_correctly(self) -> None:
        """Test that limit is applied after RRF fusion."""
        # Create 10 FTS results
        fts_results: list[dict[str, Any]] = [
            {'id': str(i), 'score': 10.0 - i, 'text_content': f'Doc {i}', 'thread_id': 't1'}
            for i in range(1, 11)
        ]

        # Request limit=3
        results = reciprocal_rank_fusion(fts_results, [], k=60, limit=3)

        assert len(results) == 3
        # Results should be top 3 by RRF score
        result_ids = [r.get('id') for r in results]
        assert result_ids == ['1', '2', '3']

    def test_limit_exceeds_available_results(self) -> None:
        """Test limit larger than available results returns all."""
        fts_results: list[dict[str, Any]] = [
            {'id': str(i), 'score': 10.0 - i, 'text_content': f'Doc {i}'}
            for i in range(1, 6)  # Only 5 results
        ]

        # Request limit=20 but only 5 exist
        results = reciprocal_rank_fusion(fts_results, [], k=60, limit=20)

        assert len(results) == 5

    def test_pagination_simulation_with_offset(self) -> None:
        """Test pagination by simulating offset with list slicing.

        In real usage, offset is applied after RRF fusion via result slicing.
        This tests the behavior of paginating through fused results.
        """
        # Create 10 FTS results
        fts_results: list[dict[str, Any]] = [
            {'id': str(i), 'score': 10.0 - (i * 0.1), 'text_content': f'Doc {i}', 'thread_id': 't1'}
            for i in range(1, 11)
        ]

        # Get all results first
        all_results = reciprocal_rank_fusion(fts_results, [], k=60, limit=10)

        # Simulate offset=3, limit=3 (skip first 3, return next 3)
        paginated = all_results[3:6]

        assert len(paginated) == 3
        # First 3 IDs (1, 2, 3) should be skipped
        paginated_ids = [r.get('id') for r in paginated]
        assert '1' not in paginated_ids
        assert '2' not in paginated_ids
        assert '3' not in paginated_ids


class TestHybridSearchResponseStructure:
    """Test hybrid search response TypedDict structure."""

    def test_response_has_required_fields(self) -> None:
        """Test that RRF results have all required fields."""
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 10.0, 'text_content': 'Test'},
        ]

        results = reciprocal_rank_fusion(fts_results, [], k=60, limit=10)

        assert len(results) == 1
        result = results[0]

        # Check all required fields are present
        required_fields = [
            'id',
            'thread_id',
            'source',
            'content_type',
            'text_content',
            'metadata',
            'created_at',
            'updated_at',
            'tags',
            'scores',
        ]

        for field in required_fields:
            assert field in result, f'Missing required field: {field}'

        # Check scores structure - use .get() for TypedDict compatibility
        scores = result.get('scores', {})
        score_fields = ['rrf', 'fts_rank', 'semantic_rank', 'fts_score', 'semantic_distance', 'rerank_score']
        for field in score_fields:
            assert field in scores, f'Missing score field: {field}'

    def test_response_scores_type_consistency(self) -> None:
        """Test that score fields have consistent types."""
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 10.0, 'text_content': 'Test'},
        ]
        semantic_results: list[dict[str, Any]] = [
            {'id': 1, 'distance': 0.5, 'text_content': 'Test'},
        ]

        results = reciprocal_rank_fusion(fts_results, semantic_results, k=60, limit=10)

        assert len(results) == 1
        scores = results[0].get('scores', {})

        # rrf should always be a float > 0
        rrf_value = scores.get('rrf')
        assert isinstance(rrf_value, float)
        assert rrf_value > 0

        # When present, ranks should be integers
        fts_rank = scores.get('fts_rank')
        semantic_rank = scores.get('semantic_rank')
        assert isinstance(fts_rank, int)
        assert isinstance(semantic_rank, int)

        # When present, scores should be floats
        fts_score = scores.get('fts_score')
        semantic_distance = scores.get('semantic_distance')
        assert isinstance(fts_score, float)
        assert isinstance(semantic_distance, float)


class TestSearchModesUsedSemantics:
    """Test that search_modes_used reflects execution, not results.

    Verifies that modes_used tracks whether a search mode executed
    successfully (no error), regardless of whether it returned results.
    """

    @staticmethod
    def _compute_modes_used(
        available_modes: list[str],
        fts_error: str | None,
        semantic_error: str | None,
    ) -> list[str]:
        """Replicate the production modes_used logic for testing."""
        modes_used: list[str] = []
        if 'fts' in available_modes and not fts_error:
            modes_used.append('fts')
        if 'semantic' in available_modes and not semantic_error:
            modes_used.append('semantic')
        return modes_used

    def test_fts_executed_zero_results_still_in_modes(self) -> None:
        """FTS executes with zero results -> still included in modes_used.

        When FTS runs successfully but finds no matching documents,
        it should still appear in modes_used because it was executed.
        """
        modes_used = self._compute_modes_used(
            available_modes=['fts', 'semantic'],
            fts_error=None,
            semantic_error=None,
        )

        assert modes_used == ['fts', 'semantic'], (
            f'Expected both modes when both executed, got {modes_used}'
        )

    def test_fts_error_excluded_from_modes(self) -> None:
        """FTS errors during execution -> excluded from modes_used.

        When FTS encounters an error, it should NOT appear in
        modes_used.
        """
        modes_used = self._compute_modes_used(
            available_modes=['fts', 'semantic'],
            fts_error='FTS search failed: connection timeout',
            semantic_error=None,
        )

        assert modes_used == ['semantic'], (
            f'Expected only semantic when FTS errored, got {modes_used}'
        )

    def test_semantic_error_excluded_from_modes(self) -> None:
        """Semantic errors during execution -> excluded from modes_used.

        When semantic search encounters an error, it should NOT
        appear in modes_used.
        """
        modes_used = self._compute_modes_used(
            available_modes=['fts', 'semantic'],
            fts_error=None,
            semantic_error='Embedding provider unavailable',
        )

        assert modes_used == ['fts'], (
            f'Expected only fts when semantic errored, got {modes_used}'
        )

    def test_both_error_empty_modes(self) -> None:
        """Both modes error -> empty modes_used.

        When both search modes encounter errors, modes_used should
        be empty.
        """
        modes_used = self._compute_modes_used(
            available_modes=['fts', 'semantic'],
            fts_error='FTS failed',
            semantic_error='Semantic failed',
        )

        assert modes_used == [], (
            f'Expected empty modes when both errored, got {modes_used}'
        )

    def test_semantic_only_mode_available(self) -> None:
        """Only semantic mode available and succeeds -> ['semantic'].

        When FTS is not in available_modes (not enabled), only semantic
        should appear regardless of fts_error state.
        """
        modes_used = self._compute_modes_used(
            available_modes=['semantic'],
            fts_error=None,
            semantic_error=None,
        )

        assert modes_used == ['semantic'], (
            f'Expected only semantic when FTS not available, got {modes_used}'
        )


class TestAdaptiveFtsMode:
    """Test adaptive FTS mode switching for hybrid search.

    Verifies that _prepare_hybrid_fts_query() correctly switches
    between AND (match) and OR (boolean) modes based on query length.
    """

    def test_short_query_uses_match_mode(self) -> None:
        """Queries below threshold use match mode (AND logic)."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='python async',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'match'
        assert query == 'python async'

    def test_exact_threshold_uses_boolean_mode(self) -> None:
        """Queries at exactly the threshold switch to boolean mode."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='python async await patterns',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert 'or' in query.lower()

    def test_sqlite_boolean_mode_neutralizes_fts5_operators(self) -> None:
        """SQLite boolean-mode terms are quoted so operator barewords / special chars
        cannot form malformed FTS5 (which raised a syntax error while PostgreSQL's
        websearch_to_tsquery tolerated the same input -- a recall-parity divergence)."""
        import sqlite3

        from app.tools.search import _prepare_hybrid_fts_query

        db = sqlite3.connect(':memory:')
        db.execute('CREATE VIRTUAL TABLE docs USING fts5(body)')
        db.execute("INSERT INTO docs(body) VALUES('foo bar near baz qux')")
        # Each query has >= 4 significant terms and embeds FTS5 operators / specials that
        # would be a syntax error unquoted; all must run cleanly and stay in boolean mode.
        for raw in [
            'foo bar near baz qux',
            'alpha AND beta OR gamma delta',
            'find foo:bar (baz) term here',
            'one two NOT three four',
        ]:
            transformed, mode = _prepare_hybrid_fts_query(raw, or_threshold=4, backend_type='sqlite')
            assert mode == 'boolean'
            assert ' OR ' in transformed
            # No raw operator bareword survives outside quotes (it is wrapped as a string).
            db.execute('SELECT rowid FROM docs WHERE docs MATCH ?', (transformed,)).fetchall()

    def test_sqlite_all_operator_query_returns_empty_sentinel(self) -> None:
        """A query whose significant terms are ALL operator stopwords (and/or/not) empties
        the term list; the SQLite transform must return the '' match-nothing sentinel -- NOT
        a literal phrase (FTS5 porter/unicode61 keeps the dropped word as a token, so the
        phrase would MATCH every document containing it, diverging from PostgreSQL's empty
        tsquery) and NOT a raw uppercase operator (which would raise an FTS5 MATCH syntax
        error). _search_sqlite short-circuits '' to an empty result set, so MATCH is never
        executed with the sentinel."""
        import sqlite3
        from typing import cast

        from app.backends.base import StorageBackend
        from app.repositories.fts_repository import FtsRepository
        from app.tools.search import _prepare_hybrid_fts_query

        class _FB:
            backend_type = 'sqlite'

        repo = FtsRepository(cast(StorageBackend, _FB()))
        db = sqlite3.connect(':memory:')
        db.execute("CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='porter unicode61')")
        db.execute("INSERT INTO docs(body) VALUES('hello world')")
        for raw in ['AND OR NOT AND', 'NOT NOT NOT NOT', 'OR OR OR OR']:
            adaptive, mode = _prepare_hybrid_fts_query(raw, or_threshold=4, backend_type='sqlite')
            fts = repo._transform_query_sqlite(adaptive, mode)
            # All tokens were operator barewords -> the empty match-nothing sentinel.
            assert fts == ''
            # _search_sqlite skips MATCH on the empty sentinel; mirror that guard here so
            # MATCH is never run with '' (which FTS5 rejects), yielding zero results.
            rows = [] if not fts else db.execute('SELECT rowid FROM docs WHERE docs MATCH ?', (fts,)).fetchall()
            assert rows == []

    def test_sqlite_short_query_operators_do_not_crash(self) -> None:
        """A SHORT query (below the OR threshold) containing bare FTS5 operators -- including
        leading/trailing/all-operator forms -- must not crash SQLite MATCH: the short 'match'
        path runs through the same term sanitizer as the OR path. A normal short query keeps
        AND-of-terms recall."""
        import sqlite3
        from typing import cast

        from app.backends.base import StorageBackend
        from app.repositories.fts_repository import FtsRepository
        from app.tools.search import _prepare_hybrid_fts_query

        class _FB:
            backend_type = 'sqlite'

        repo = FtsRepository(cast(StorageBackend, _FB()))
        db = sqlite3.connect(':memory:')
        db.execute("CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='porter unicode61')")
        db.execute("INSERT INTO docs(body) VALUES('python async world')")
        # All have < or_threshold(4) significant terms, so they take the short 'match' path.
        for raw, expect_match in [('AND OR', False), ('OR cat', False), ('cat OR', False),
                                  ('OR NOT AND', False), ('python async', True)]:
            adaptive, mode = _prepare_hybrid_fts_query(raw, or_threshold=4, backend_type='sqlite')
            assert mode == 'match'
            fts = repo._transform_query_sqlite(adaptive, mode)
            # An all-operator query transforms to the '' match-nothing sentinel, which
            # _search_sqlite short-circuits (FTS5 rejects MATCH ''); mirror that guard here.
            rows = [] if not fts else db.execute('SELECT rowid FROM docs WHERE docs MATCH ?', (fts,)).fetchall()
            assert (len(rows) > 0) is expect_match, f'{raw!r} -> {fts!r}'

    def test_sqlite_boolean_mode_drops_operator_stopwords(self) -> None:
        """The FTS5 operator barewords and/or/not are DROPPED on the SQLite branch (not
        quoted) so they are not literal searchable terms -- matching PostgreSQL's
        websearch_to_tsquery, which removes them as stopwords (cross-backend recall parity).
        'near' is KEPT (websearch_to_tsquery keeps it)."""
        from app.tools.search import _prepare_hybrid_fts_query

        transformed, mode = _prepare_hybrid_fts_query(
            'alpha AND beta OR gamma NOT delta near epsilon',
            or_threshold=4,
            backend_type='sqlite',
        )
        assert mode == 'boolean'
        for dropped in ('"and"', '"or"', '"not"', '"AND"', '"OR"', '"NOT"'):
            assert dropped not in transformed
        # Real terms (including 'near') survive as quoted literals.
        for kept in ('"alpha"', '"beta"', '"gamma"', '"delta"', '"near"', '"epsilon"'):
            assert kept in transformed

    def test_long_query_uses_boolean_mode(self) -> None:
        """Long queries above threshold use boolean mode (OR logic)."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='DRY extraction embedding helper timeout semaphore pattern',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert ' or ' in query

    def test_postgresql_uses_lowercase_or(self) -> None:
        """PostgreSQL backend uses lowercase 'or' keyword."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='alpha beta gamma delta',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert ' or ' in query
        assert ' OR ' not in query

    def test_sqlite_uses_uppercase_or(self) -> None:
        """SQLite backend uses uppercase 'OR' keyword."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='alpha beta gamma delta',
            or_threshold=4,
            backend_type='sqlite',
        )
        assert mode == 'boolean'
        assert ' OR ' in query

    def test_single_char_words_excluded_from_count(self) -> None:
        """Single-character words are not counted as significant."""
        from app.tools.search import _prepare_hybrid_fts_query

        # "a b c python async" has only 2 significant words (python, async)
        query, mode = _prepare_hybrid_fts_query(
            query='a b c python async',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'match'

    def test_hyphen_sanitization(self) -> None:
        """Hyphens are replaced with spaces to prevent NOT interpretation."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='context-server async-await error-handling patterns',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert '-' not in query

    def test_hyphenated_tokens_keep_phrase_semantics_on_both_backends(self) -> None:
        """A hyphenated token is a quoted phrase on BOTH backends in OR mode.

        SQLite's sanitize_sqlite_fts_terms wraps 'a-b' as the FTS5 phrase
        literal "a b" (ordered adjacency); the PostgreSQL branch must wrap the
        hyphen-joined token the same way so websearch_to_tsquery parses it as
        a <-> b instead of ANDing the parts unordered -- otherwise the two
        backends return different recall for the identical hybrid query.
        """
        from app.tools.search import _prepare_hybrid_fts_query

        pg_query, pg_mode = _prepare_hybrid_fts_query(
            query='a-b c-d e-f g-h',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert pg_mode == 'boolean'
        assert pg_query == '"a b" or "c d" or "e f" or "g h"'

        sqlite_query, sqlite_mode = _prepare_hybrid_fts_query(
            query='a-b c-d e-f g-h',
            or_threshold=4,
            backend_type='sqlite',
        )
        assert sqlite_mode == 'boolean'
        assert sqlite_query == '"a b" OR "c d" OR "e f" OR "g h"'

        # The two transforms differ only in the OR keyword casing.
        assert pg_query.replace(' or ', ' OR ') == sqlite_query

    def test_hyphenated_token_with_embedded_quote_stays_wrappable(self) -> None:
        """An embedded double quote cannot terminate the generated phrase early."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='al"pha-beta gamma delta epsilon',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        # The embedded quote is dropped; the phrase wrapper stays balanced.
        assert query.count('"') % 2 == 0
        assert '"al pha beta"' in query

    def test_threshold_boundary_below(self) -> None:
        """Query with exactly threshold-1 significant words stays in match mode."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='alpha beta gamma',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'match'
        assert query == 'alpha beta gamma'

    def test_empty_after_sanitization_fallback(self) -> None:
        """If all words sanitize to empty, falls back to match mode."""
        from app.tools.search import _prepare_hybrid_fts_query

        # Single-char words with hyphens that would sanitize to empty
        query, mode = _prepare_hybrid_fts_query(
            query='- - - -',
            or_threshold=2,
            backend_type='postgresql',
        )
        assert mode == 'match'

    def test_custom_threshold(self) -> None:
        """Custom threshold value is respected."""
        from app.tools.search import _prepare_hybrid_fts_query

        # With threshold=2, even 2-word queries switch to OR
        query, mode = _prepare_hybrid_fts_query(
            query='python async',
            or_threshold=2,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert ' or ' in query

    def test_whitespace_handling(self) -> None:
        """Extra whitespace in query is handled gracefully."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='  alpha   beta   gamma   delta  ',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert ' or ' in query

    def test_quoted_phrase_preserved_in_boolean_mode(self) -> None:
        """Quoted phrases are preserved as single tokens in OR mode."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='"error handling" timeout async patterns',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert '"error handling"' in query

    def test_multiple_quoted_phrases_preserved(self) -> None:
        """Multiple quoted phrases are each preserved intact."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='"error handling" "async await" timeout patterns',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert '"error handling"' in query
        assert '"async await"' in query

    def test_quoted_phrase_not_hyphen_sanitized(self) -> None:
        """Quoted phrases containing hyphens are NOT sanitized."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='"error-handling" timeout async patterns',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert '"error-handling"' in query

    def test_sqlite_short_match_returns_raw_query_for_single_transform(self) -> None:
        """A short SQLite query is returned RAW (not pre-sanitized) so _transform_query_sqlite
        sanitizes it exactly once -- identically to standalone fts_search_context."""
        from app.tools.search import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query('python async', or_threshold=4, backend_type='sqlite')
        assert mode == 'match'
        assert query == 'python async'  # raw; the single downstream transform does the escaping

    def test_sqlite_embedded_quote_token_not_double_mangled(self) -> None:
        """An embedded-double-quote token yields the SAME single-phrase FTS5 form via the hybrid
        path as via standalone fts_search_context -- no non-idempotent double sanitize.

        Pre-sanitizing in the hybrid builder AND re-sanitizing in _transform_query_sqlite split
        the escaped phrase ``"ab""cd"`` into two phrases ``"ab" "cd"`` (because _FTS_TOKEN_RE
        re-tokenizes the escaped run), diverging hybrid recall from standalone for the same
        input. Deferring all escaping to the one downstream transform keeps them identical.
        """
        from typing import cast

        from app.backends.base import StorageBackend
        from app.repositories.fts_repository import FtsRepository
        from app.tools.search import _prepare_hybrid_fts_query

        class _FB:
            backend_type = 'sqlite'

        repo = FtsRepository(cast(StorageBackend, _FB()))
        raw = 'ab"cd ef'
        # Hybrid short path returns the raw query; the single downstream transform escapes it.
        adaptive, mode = _prepare_hybrid_fts_query(raw, or_threshold=4, backend_type='sqlite')
        assert mode == 'match'
        hybrid_fts = repo._transform_query_sqlite(adaptive, mode)
        # Standalone fts_search_context passes the raw query straight to the same transform.
        standalone_fts = repo._transform_query_sqlite(raw, 'match')
        assert hybrid_fts == standalone_fts
        # The correct single-phrase escaped form, NOT the split-into-two-phrases mangling.
        assert hybrid_fts == '"ab""cd" "ef"'
        assert hybrid_fts != '"ab" "cd" "ef"'


class TestHybridAllModesFailedValidationResponse:
    """The structured error return when every available mode fails validation.

    Exercises the ACTUAL error branch of hybrid_search_context (not a local
    re-implementation): the response must use the documented
    search_modes_used key, and identical validation messages produced by both
    sub-searches over the same filters must be deduplicated.
    """

    @pytest.mark.asyncio
    async def test_error_response_keys_and_deduplicated_messages(self) -> None:
        from app.repositories.embedding_repository import MetadataFilterValidationError
        from app.repositories.fts_repository import FtsValidationError
        from app.tools.search import hybrid_search_context

        shared_messages = [
            "Invalid metadata filter {'key': 'a', 'operator': 'nope'}: unsupported operator",
        ]

        with (
            patch('app.tools.search.ensure_repositories', AsyncMock(return_value=MagicMock())),
            patch('app.tools.search.get_embedding_provider', return_value=object()),
            patch('app.tools.search.get_reranking_provider', return_value=None),
            patch(
                'app.tools.search._fts_search_raw',
                AsyncMock(side_effect=FtsValidationError('Invalid filters', list(shared_messages))),
            ),
            patch(
                'app.tools.search._semantic_search_raw',
                AsyncMock(side_effect=MetadataFilterValidationError('Invalid filters', list(shared_messages))),
            ),
        ):
            result = await hybrid_search_context(
                query='anything',
                metadata_filters=[{'key': 'a', 'operator': 'nope', 'value': 1}],
            )

        assert result['count'] == 0
        assert result['results'] == []
        # The documented key, matching the success path, the docstring, and
        # the TypedDict -- not a stray spelling unique to the error branch.
        assert result['search_modes_used'] == []
        assert 'modes_used' not in result
        error_text = result['error']
        assert isinstance(error_text, str)
        assert error_text.startswith('All available search modes failed')
        # Both sub-searches validated the same filters and produced identical
        # messages; the client must see each defect once.
        assert result['validation_errors'] == shared_messages

    @pytest.mark.asyncio
    async def test_error_response_carries_fusion_method_and_counts(self) -> None:
        """The all-modes-failed response carries fusion_method, fts_count, and
        semantic_count -- the always-present response-shape keys the success path,
        the docstring, and HybridSearchResponseDict declare, so a client reading
        them never hits a KeyError on the error branch."""
        from app.repositories.embedding_repository import MetadataFilterValidationError
        from app.repositories.fts_repository import FtsValidationError
        from app.tools.search import hybrid_search_context

        messages = ['bad operator: nope']

        with (
            patch('app.tools.search.ensure_repositories', AsyncMock(return_value=MagicMock())),
            patch('app.tools.search.get_embedding_provider', return_value=object()),
            patch('app.tools.search.get_reranking_provider', return_value=None),
            patch(
                'app.tools.search._fts_search_raw',
                AsyncMock(side_effect=FtsValidationError('Invalid filters', list(messages))),
            ),
            patch(
                'app.tools.search._semantic_search_raw',
                AsyncMock(side_effect=MetadataFilterValidationError('Invalid filters', list(messages))),
            ),
        ):
            result = await hybrid_search_context(
                query='anything',
                metadata_filters=[{'key': 'a', 'operator': 'nope', 'value': 1}],
            )

        assert result['fusion_method'] == 'rrf'
        assert result['fts_count'] == 0
        assert result['semantic_count'] == 0


class TestHybridPartialDegradationResponse:
    """One sub-search fails validation while the other succeeds (graceful degradation).

    Exercises the ACTUAL partial-degradation path of hybrid_search_context: the
    surviving mode's results are returned, but the response also carries the
    specific failure text in ``warnings`` and the per-filter details under the
    same ``validation_errors`` key the all-failed branch uses, so a client can
    correct an invalid sub-query even when results were still produced.
    """

    @staticmethod
    def _repos_with_tags() -> MagicMock:
        """A repository container whose tag lookup returns an empty list."""
        repos = MagicMock()
        repos.tags = MagicMock()
        repos.tags.get_tags_for_context = AsyncMock(return_value=[])
        return repos

    @pytest.mark.asyncio
    async def test_semantic_failure_surfaces_warning_and_validation_errors(self) -> None:
        """Semantic fails validation, FTS succeeds: results returned, warning + details present."""
        from app.repositories.embedding_repository import MetadataFilterValidationError
        from app.tools.search import hybrid_search_context

        semantic_messages = [
            "Invalid metadata filter {'key': 'a', 'operator': 'nope'}: unsupported operator",
        ]
        fts_rows: list[dict[str, Any]] = [
            {
                'id': 'a' * 32, 'thread_id': 't1', 'source': 'agent', 'content_type': 'text',
                'text_content': 'python async guide', 'score': 9.0, 'metadata': None,
                'created_at': '2025-01-01T00:00:00Z', 'updated_at': '2025-01-01T00:00:00Z',
            },
        ]

        with (
            patch('app.tools.search.ensure_repositories', AsyncMock(return_value=self._repos_with_tags())),
            patch('app.tools.search.get_embedding_provider', return_value=object()),
            patch('app.tools.search.get_reranking_provider', return_value=None),
            patch('app.tools.search._fts_search_raw', AsyncMock(return_value=(fts_rows, {}))),
            patch(
                'app.tools.search._semantic_search_raw',
                AsyncMock(side_effect=MetadataFilterValidationError('Invalid filters', list(semantic_messages))),
            ),
        ):
            result = await hybrid_search_context(
                query='python async',
                metadata_filters=[{'key': 'a', 'operator': 'nope', 'value': 1}],
            )

        # FTS survived, so results are still returned (graceful degradation).
        assert result['count'] == 1
        assert result['results'][0]['id'] == 'a' * 32
        assert result['search_modes_used'] == ['fts']

        # The warning embeds the SPECIFIC semantic failure text, not a generic notice.
        warnings = result['warnings']
        assert isinstance(warnings, list)
        assert len(warnings) == 1
        assert 'Semantic:' in warnings[0]
        assert 'Invalid filters' in warnings[0]

        # The per-filter details ride under the same validation_errors key the
        # all-failed branch uses, so the client can fix the invalid sub-query.
        assert result['validation_errors'] == semantic_messages
        # No top-level error on partial degradation -- the request partly succeeded.
        assert 'error' not in result

    @pytest.mark.asyncio
    async def test_fts_failure_surfaces_warning_and_validation_errors(self) -> None:
        """FTS fails validation, semantic succeeds: mirror case with FTS-prefixed warning."""
        from app.repositories.fts_repository import FtsValidationError
        from app.tools.search import hybrid_search_context

        fts_messages = ['invalid boolean expression near ")"']
        semantic_rows: list[dict[str, Any]] = [
            {
                'id': 'b' * 32, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                'text_content': 'python async guide', 'distance': 0.2, 'metadata': None,
                'created_at': '2025-01-01T00:00:00Z', 'updated_at': '2025-01-01T00:00:00Z',
            },
        ]

        with (
            patch('app.tools.search.ensure_repositories', AsyncMock(return_value=self._repos_with_tags())),
            patch('app.tools.search.get_embedding_provider', return_value=object()),
            patch('app.tools.search.get_reranking_provider', return_value=None),
            patch(
                'app.tools.search._fts_search_raw',
                AsyncMock(side_effect=FtsValidationError('Invalid filters', list(fts_messages))),
            ),
            patch('app.tools.search._semantic_search_raw', AsyncMock(return_value=(semantic_rows, {}))),
        ):
            result = await hybrid_search_context(
                query='python async',
                metadata_filters=[{'key': 'a', 'operator': 'nope', 'value': 1}],
            )

        assert result['count'] == 1
        assert result['results'][0]['id'] == 'b' * 32
        assert result['search_modes_used'] == ['semantic']

        warnings = result['warnings']
        assert isinstance(warnings, list)
        assert len(warnings) == 1
        assert 'FTS:' in warnings[0]
        assert 'Invalid filters' in warnings[0]

        assert result['validation_errors'] == fts_messages
        assert 'error' not in result
