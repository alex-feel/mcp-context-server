"""Structurally invalid filters are rejected before the query embedding is generated.

A bad operator, an unsafe metadata key, or an all-blank tags list can never match
anything, yet the repositories only discover that inside their read callables --
after the semantic path has already paid a full round trip to the embedding
provider (seconds to tens of seconds, plus provider quota), and after the
rejection stats have been built claiming zero embedding time for it.
"""

from typing import Any

import pytest

import app.tools.search as search_mod
from app.repositories.embedding_repository import MetadataFilterValidationError

BAD_FILTER: list[dict[str, Any]] = [{'key': 'priority', 'operator': 'bogus_op', 'value': 5}]
BLANK_TAGS = ['   ', '']


class _CountingEmbeddingProvider:
    """Embedding provider that records every query it is asked to embed."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed_query(self, query: str) -> list[float]:
        """Record the call and return a fixed vector.

        Args:
            query: The query text.

        Returns:
            A fixed embedding vector.
        """
        self.calls.append(query)
        return [0.1, 0.2, 0.3]


class _FakeEmbeddingsRepo:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return no rows, recording that the repository was reached.

        Args:
            kwargs: Unused search arguments.

        Returns:
            An empty result set with empty stats.
        """
        del kwargs
        self.calls += 1
        return [], {}


class _FakeRepos:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsRepo()


@pytest.fixture
def embedding_provider(monkeypatch: pytest.MonkeyPatch) -> _CountingEmbeddingProvider:
    """Patch the semantic stack with a counting embedding provider.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        The provider whose calls the tests assert on.
    """
    provider = _CountingEmbeddingProvider()
    repos = _FakeRepos()

    async def fake_ensure_repositories() -> _FakeRepos:
        return repos

    monkeypatch.setattr(search_mod, 'get_embedding_provider', lambda: provider)
    monkeypatch.setattr(search_mod, 'ensure_repositories', fake_ensure_repositories)
    monkeypatch.setattr(search_mod, 'get_reranking_provider', lambda: None)
    return provider


class TestRawSemanticSearch:
    """The pre-check lives on the shared raw path, so both callers inherit it."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('kwargs', 'fragment'),
        [
            ({'metadata_filters': BAD_FILTER}, 'Invalid metadata filter'),
            ({'metadata': {'bad key!': 'x'}}, 'Invalid metadata key'),
            ({'tags': BLANK_TAGS}, 'non-blank tag'),
        ],
    )
    async def test_invalid_filter_rejected_without_embedding(
        self,
        embedding_provider: _CountingEmbeddingProvider,
        kwargs: dict[str, Any],
        fragment: str,
    ) -> None:
        with pytest.raises(MetadataFilterValidationError) as excinfo:
            await search_mod._semantic_search_raw(query='anything', limit=5, **kwargs)

        assert excinfo.value.message == 'Metadata filter validation failed'
        assert any(fragment in message for message in excinfo.value.validation_errors)
        assert embedding_provider.calls == [], 'the embedding round trip ran before validation'

    @pytest.mark.asyncio
    async def test_valid_filters_still_reach_the_repository(
        self,
        embedding_provider: _CountingEmbeddingProvider,
    ) -> None:
        """A legal filter must not be rejected by the boundary check."""
        results, _stats = await search_mod._semantic_search_raw(
            query='anything',
            limit=5,
            tags=['Real'],
            metadata={'status': 'done'},
            metadata_filters=[{'key': 'priority', 'operator': 'gt', 'value': 5}],
        )

        assert results == []
        assert embedding_provider.calls == ['anything']


class TestSemanticSearchTool:
    """The tool response is unchanged: same error, same details, honest stats."""

    @pytest.mark.asyncio
    async def test_structured_error_without_embedding(
        self,
        embedding_provider: _CountingEmbeddingProvider,
    ) -> None:
        response = await search_mod.semantic_search_context(
            query='anything',
            metadata_filters=BAD_FILTER,
            explain_query=True,
        )

        assert response['error'] == 'Metadata filter validation failed'
        assert response['validation_errors']
        assert response['results'] == []
        assert embedding_provider.calls == []
        # No embedding ran, so the reported embedding time is now the truth rather
        # than a zero standing in for a call that really happened.
        assert response['stats']['embedding_generation_ms'] == 0.0


class TestHybridSearchTool:
    """Hybrid inherits the same protection through its semantic leg."""

    @pytest.mark.asyncio
    async def test_no_embedding_for_an_invalid_filter(
        self,
        embedding_provider: _CountingEmbeddingProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.repositories.fts_repository import FtsValidationError

        async def failing_fts(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            del kwargs
            raise FtsValidationError('Invalid filters', ['Invalid metadata filter'])

        monkeypatch.setattr(search_mod, '_fts_search_raw', failing_fts)

        response = await search_mod.hybrid_search_context(
            query='anything',
            metadata_filters=BAD_FILTER,
        )

        assert response['results'] == []
        assert response['validation_errors']
        assert embedding_provider.calls == []
