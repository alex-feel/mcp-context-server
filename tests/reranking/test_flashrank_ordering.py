"""The reranker's final sort carries an explicit unique key.

``FlashRankProvider.rerank`` is the LAST ranked sort in every search pipeline: with
reranking on (the default), the order it produces is the order the client sees, and
the ``limit`` slice that follows is a top-N window over it. Cross-encoder ties are
ordinary -- near-identical passages score identically -- so an implicit ordering
here would leave the client-visible result set decided by nothing the data
determines, exactly as the score-only ORDER BY clauses upstream did before they
gained a secondary key.

The tiebreak is the INCOMING position: that is the retrieval ranking, which is
itself deterministically ordered by the repository and fusion layers, so it is both
meaningful (better-retrieved first) and reproducible.
"""

from typing import Any
from typing import Protocol

import pytest

from app.reranking.providers.flashrank import FlashRankProvider


class _RerankRequestLike(Protocol):
    """The only attribute of ``flashrank.RerankRequest`` the provider uses."""

    passages: list[dict[str, Any]]


class _StubRanker:
    """Ranker stub assigning a caller-supplied score per passage id."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def rerank(self, request: _RerankRequestLike) -> list[dict[str, Any]]:
        """Return the request's passages with their configured scores attached."""
        passages = request.passages
        return [
            {
                'id': passage['id'],
                'text': passage['text'],
                'meta': passage['meta'],
                'score': self._scores[str(passage['id'])],
            }
            for passage in passages
        ]


def _provider(scores: dict[str, float]) -> FlashRankProvider:
    """Build a provider whose model is already loaded with a stub ranker.

    Args:
        scores: Score to report per result id.

    Returns:
        The configured provider.
    """
    provider = FlashRankProvider()
    provider._ranker = _StubRanker(scores)
    return provider


def _results(ids: list[str]) -> list[dict[str, Any]]:
    """Build minimal search results in the given order.

    Args:
        ids: Result ids in incoming (retrieval) order.

    Returns:
        The result dicts the reranker accepts.
    """
    return [{'id': result_id, 'text': f'passage {result_id}'} for result_id in ids]


@pytest.mark.asyncio
async def test_tied_scores_keep_the_incoming_retrieval_order() -> None:
    """Every score identical: the output must be the incoming order, unchanged."""
    ids = ['e', 'c', 'a', 'd', 'b']
    provider = _provider(dict.fromkeys(ids, 0.5))

    ranked = await provider.rerank('query', _results(ids))

    assert [r['id'] for r in ranked] == ids


@pytest.mark.asyncio
async def test_scores_still_dominate_the_tiebreak() -> None:
    """The tiebreak only decides ties; a better score always wins."""
    provider = _provider({'a': 0.1, 'b': 0.9, 'c': 0.5})

    ranked = await provider.rerank('query', _results(['a', 'b', 'c']))

    assert [r['id'] for r in ranked] == ['b', 'c', 'a']


@pytest.mark.asyncio
async def test_limit_window_over_a_tie_is_reproducible() -> None:
    """The top-N slice over a full tie is decided by the data, not by chance."""
    ids = [f'id-{i}' for i in range(10)]
    provider = _provider(dict.fromkeys(ids, 0.42))

    first = await provider.rerank('query', _results(ids), limit=3)
    second = await provider.rerank('query', _results(ids), limit=3)

    assert [r['id'] for r in first] == ids[:3]
    assert [r['id'] for r in second] == [r['id'] for r in first]
