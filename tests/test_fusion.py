"""Unit tests for the fusion module (RRF algorithm)."""

from __future__ import annotations

from typing import Any

import pytest

from app.fusion import count_unique_results
from app.fusion import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion function."""

    def test_empty_inputs(self) -> None:
        """Test RRF with empty result lists."""
        results = reciprocal_rank_fusion([], [], k=60, limit=10)
        assert results == []

    def test_fts_only(self) -> None:
        """Test RRF with only FTS results."""
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 2.5, 'text_content': 'doc1'},
            {'id': 2, 'score': 1.8, 'text_content': 'doc2'},
        ]

        results = reciprocal_rank_fusion(fts_results, [], k=60, limit=10)

        assert len(results) == 2
        # Doc 1 should be first (higher FTS rank)
        assert results[0].get('id') == 1
        assert results[1].get('id') == 2

        # Check scores structure
        scores = results[0].get('scores', {})
        assert scores.get('fts_rank') == 1
        assert scores.get('semantic_rank') is None
        assert scores.get('fts_score') == 2.5
        assert scores.get('semantic_distance') is None
        assert scores.get('rrf') == pytest.approx(1 / (60 + 1))

    def test_semantic_only(self) -> None:
        """Test RRF with only semantic results."""
        semantic_results: list[dict[str, Any]] = [
            {'id': 3, 'distance': 0.3, 'text_content': 'doc3'},
            {'id': 4, 'distance': 0.5, 'text_content': 'doc4'},
        ]

        results = reciprocal_rank_fusion([], semantic_results, k=60, limit=10)

        assert len(results) == 2
        # Doc 3 should be first (lower distance = rank 1)
        assert results[0].get('id') == 3
        assert results[1].get('id') == 4

        # Check scores structure
        scores = results[0].get('scores', {})
        assert scores.get('fts_rank') is None
        assert scores.get('semantic_rank') == 1
        assert scores.get('fts_score') is None
        assert scores.get('semantic_distance') == 0.3

    def test_overlapping_results_score_higher(self) -> None:
        """Test that documents appearing in both result sets score higher."""
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 2.5, 'text_content': 'doc1'},
            {'id': 2, 'score': 1.8, 'text_content': 'doc2'},
        ]
        semantic_results: list[dict[str, Any]] = [
            {'id': 2, 'distance': 0.3, 'text_content': 'doc2'},  # Overlap!
            {'id': 3, 'distance': 0.5, 'text_content': 'doc3'},
        ]

        results = reciprocal_rank_fusion(fts_results, semantic_results, k=60, limit=10)

        assert len(results) == 3

        # Doc 2 appears in both, should score highest
        doc2 = next(r for r in results if r.get('id') == 2)
        doc1 = next(r for r in results if r.get('id') == 1)
        doc3 = next(r for r in results if r.get('id') == 3)

        assert doc2.get('scores', {}).get('rrf', 0) > doc1.get('scores', {}).get('rrf', 0)
        assert doc2.get('scores', {}).get('rrf', 0) > doc3.get('scores', {}).get('rrf', 0)

        # Doc 2 should have both ranks
        scores2 = doc2.get('scores', {})
        assert scores2.get('fts_rank') == 2
        assert scores2.get('semantic_rank') == 1
        assert scores2.get('fts_score') == 1.8
        assert scores2.get('semantic_distance') == 0.3

    def test_rrf_formula_correctness(self) -> None:
        """Verify RRF formula: score(d) = sum(1 / (k + rank_i(d)))."""
        k = 60
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 2.5, 'text_content': 'doc1'},
        ]
        semantic_results: list[dict[str, Any]] = [
            {'id': 1, 'distance': 0.3, 'text_content': 'doc1'},  # Same doc
        ]

        results = reciprocal_rank_fusion(fts_results, semantic_results, k=k, limit=10)

        assert len(results) == 1
        # RRF score = 1/(60+1) + 1/(60+1) = 2/(61)
        expected_rrf = 1 / (k + 1) + 1 / (k + 1)
        assert results[0].get('scores', {}).get('rrf') == pytest.approx(expected_rrf)

    def test_limit_parameter(self) -> None:
        """Test that limit parameter correctly restricts results."""
        fts_results: list[dict[str, Any]] = [
            {'id': i, 'score': 10 - i, 'text_content': f'doc{i}'}
            for i in range(1, 6)
        ]

        results = reciprocal_rank_fusion(fts_results, [], k=60, limit=3)

        assert len(results) == 3
        # Should keep top 3 by RRF score
        assert [r.get('id') for r in results] == [1, 2, 3]

    def test_k_parameter_effect(self) -> None:
        """Test that k parameter affects relative ranking."""
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 2.5, 'text_content': 'doc1'},
            {'id': 2, 'score': 1.8, 'text_content': 'doc2'},
        ]

        # With small k, rank difference matters more
        results_small_k = reciprocal_rank_fusion(fts_results, [], k=1, limit=10)
        # With large k, rank difference matters less
        results_large_k = reciprocal_rank_fusion(fts_results, [], k=100, limit=10)

        # Score ratio between rank 1 and rank 2 with k=1: (1/2) / (1/3) = 1.5
        rrf_0_small = results_small_k[0].get('scores', {}).get('rrf', 0)
        rrf_1_small = results_small_k[1].get('scores', {}).get('rrf', 1)
        ratio_small_k = rrf_0_small / rrf_1_small

        # Score ratio between rank 1 and rank 2 with k=100: (1/101) / (1/102) ~ 1.01
        rrf_0_large = results_large_k[0].get('scores', {}).get('rrf', 0)
        rrf_1_large = results_large_k[1].get('scores', {}).get('rrf', 1)
        ratio_large_k = rrf_0_large / rrf_1_large

        # Small k should give larger ratio (more weight to top results)
        assert ratio_small_k > ratio_large_k

    def test_data_merging(self) -> None:
        """Test that data from both sources is merged correctly."""
        fts_results: list[dict[str, Any]] = [
            {
                'id': 1,
                'score': 2.5,
                'text_content': 'doc1',
                'thread_id': 'thread-1',
                'source': 'agent',
            },
        ]
        semantic_results: list[dict[str, Any]] = [
            {
                'id': 1,
                'distance': 0.3,
                'text_content': 'doc1',
                'metadata': {'key': 'value'},
            },
        ]

        results = reciprocal_rank_fusion(fts_results, semantic_results, k=60, limit=10)

        assert len(results) == 1
        # Should have data from FTS
        assert results[0].get('thread_id') == 'thread-1'
        assert results[0].get('source') == 'agent'
        # Should also have data from semantic (metadata was not in FTS)
        assert results[0].get('metadata') == {'key': 'value'}

    def test_results_sorted_by_rrf_score(self) -> None:
        """Test that results are sorted by RRF score descending."""
        # Create a scenario where order differs from input order
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 1.0, 'text_content': 'doc1'},  # FTS rank 1
            {'id': 2, 'score': 0.5, 'text_content': 'doc2'},  # FTS rank 2
        ]
        semantic_results: list[dict[str, Any]] = [
            {'id': 2, 'distance': 0.1, 'text_content': 'doc2'},  # Semantic rank 1
            {'id': 3, 'distance': 0.2, 'text_content': 'doc3'},  # Semantic rank 2
        ]

        results = reciprocal_rank_fusion(fts_results, semantic_results, k=60, limit=10)

        # Doc 2 should be first (appears in both)
        assert results[0].get('id') == 2
        # Verify descending order
        for i in range(len(results) - 1):
            rrf_i = results[i].get('scores', {}).get('rrf', 0)
            rrf_i1 = results[i + 1].get('scores', {}).get('rrf', 0)
            assert rrf_i >= rrf_i1

    def test_none_id_filtered(self) -> None:
        """Test that entries without id are filtered out."""
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 2.5, 'text_content': 'doc1'},
            {'id': None, 'score': 1.0, 'text_content': 'no_id'},  # Should be skipped
            {'score': 0.5, 'text_content': 'missing_id'},  # No id field
        ]

        results = reciprocal_rank_fusion(fts_results, [], k=60, limit=10)

        assert len(results) == 1
        assert results[0].get('id') == 1

    def test_default_values_in_result(self) -> None:
        """Test that missing fields get default values."""
        fts_results: list[dict[str, Any]] = [
            {'id': 1, 'score': 2.5},  # Minimal data
        ]

        results = reciprocal_rank_fusion(fts_results, [], k=60, limit=10)

        assert len(results) == 1
        # Check defaults
        assert results[0].get('thread_id') == ''
        assert results[0].get('source') == ''
        assert results[0].get('content_type') == 'text'
        assert results[0].get('text_content') == ''
        assert results[0].get('created_at') == ''
        assert results[0].get('updated_at') == ''
        assert results[0].get('tags') == []


class TestCountUniqueResults:
    """Tests for count_unique_results function."""

    def test_empty_inputs(self) -> None:
        """Test with empty lists."""
        fts_only, semantic_only, overlap = count_unique_results([], [])
        assert fts_only == 0
        assert semantic_only == 0
        assert overlap == 0

    def test_no_overlap(self) -> None:
        """Test with completely disjoint result sets."""
        fts_results: list[dict[str, Any]] = [{'id': 1}, {'id': 2}]
        semantic_results: list[dict[str, Any]] = [{'id': 3}, {'id': 4}]

        fts_only, semantic_only, overlap = count_unique_results(fts_results, semantic_results)

        assert fts_only == 2
        assert semantic_only == 2
        assert overlap == 0

    def test_full_overlap(self) -> None:
        """Test with identical result sets."""
        fts_results: list[dict[str, Any]] = [{'id': 1}, {'id': 2}]
        semantic_results: list[dict[str, Any]] = [{'id': 1}, {'id': 2}]

        fts_only, semantic_only, overlap = count_unique_results(fts_results, semantic_results)

        assert fts_only == 0
        assert semantic_only == 0
        assert overlap == 2

    def test_partial_overlap(self) -> None:
        """Test with partial overlap."""
        fts_results: list[dict[str, Any]] = [{'id': 1}, {'id': 2}, {'id': 3}]
        semantic_results: list[dict[str, Any]] = [{'id': 2}, {'id': 3}, {'id': 4}]

        fts_only, semantic_only, overlap = count_unique_results(fts_results, semantic_results)

        assert fts_only == 1  # id: 1
        assert semantic_only == 1  # id: 4
        assert overlap == 2  # id: 2, 3

    def test_none_ids_ignored(self) -> None:
        """Test that None ids are ignored in counting."""
        fts_results: list[dict[str, Any]] = [{'id': 1}, {'id': None}, {'other': 'data'}]
        semantic_results: list[dict[str, Any]] = [{'id': 1}, {'id': 2}]

        fts_only, semantic_only, overlap = count_unique_results(fts_results, semantic_results)

        assert fts_only == 0  # Only id: 1, which overlaps
        assert semantic_only == 1  # id: 2
        assert overlap == 1  # id: 1
