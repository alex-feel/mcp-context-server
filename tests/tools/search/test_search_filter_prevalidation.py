"""Structurally invalid filters are rejected before the query embedding is generated.

A bad operator, an unsafe metadata key, or an all-blank tags list can never match
anything, yet the repositories only discover that inside their read callables --
after the semantic path has already paid a full round trip to the embedding
provider (seconds to tens of seconds, plus provider quota), and after the
rejection stats have been built claiming zero embedding time for it.
"""

from typing import Any
from typing import cast

import pytest

import app.tools.search.hybrid as search_hybrid
import app.tools.search.semantic as search_semantic
from app.embeddings.base import EmbeddingProvider
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository.records import MetadataFilterValidationError
from app.tools.search.legs import semantic_search_raw
from tests.helpers import LOCAL_SCOPE

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
def fake_repos() -> _FakeRepos:
    """Provide the repository container the semantic stack searches.

    Returns:
        A container whose embeddings repository returns no rows.
    """
    return _FakeRepos()


@pytest.fixture
def embedding_provider(monkeypatch: pytest.MonkeyPatch, fake_repos: _FakeRepos) -> _CountingEmbeddingProvider:
    """Patch the semantic and hybrid tools with a counting embedding provider.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        fake_repos: The repository container the tools resolve.

    Returns:
        The provider whose calls the tests assert on.
    """
    provider = _CountingEmbeddingProvider()

    async def fake_ensure_repositories() -> _FakeRepos:
        return fake_repos

    for module in (search_semantic, search_hybrid):
        monkeypatch.setattr(module, 'get_embedding_provider', lambda: provider)
        monkeypatch.setattr(module, 'ensure_repositories', fake_ensure_repositories)
        monkeypatch.setattr(module, 'get_reranking_provider', lambda: None)
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
        fake_repos: _FakeRepos,
        kwargs: dict[str, Any],
        fragment: str,
    ) -> None:
        with pytest.raises(MetadataFilterValidationError) as excinfo:
            await semantic_search_raw(
                query='anything',
                limit=5,
                **kwargs,
                repos=cast(RepositoryContainer, fake_repos),
                embedding_provider=cast(EmbeddingProvider, embedding_provider),
                scope=LOCAL_SCOPE,
            )

        assert excinfo.value.message == 'Metadata filter validation failed'
        assert any(fragment in message for message in excinfo.value.validation_errors)
        assert embedding_provider.calls == [], 'the embedding round trip ran before validation'

    @pytest.mark.asyncio
    async def test_valid_filters_still_reach_the_repository(
        self,
        embedding_provider: _CountingEmbeddingProvider,
        fake_repos: _FakeRepos,
    ) -> None:
        """A legal filter must not be rejected by the boundary check."""
        results, _stats = await semantic_search_raw(
            query='anything',
            limit=5,
            tags=['Real'],
            metadata={'status': 'done'},
            metadata_filters=[{'key': 'priority', 'operator': 'gt', 'value': 5}],
            repos=cast(RepositoryContainer, fake_repos),
            embedding_provider=cast(EmbeddingProvider, embedding_provider),
            scope=LOCAL_SCOPE,
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
        response = await search_semantic.semantic_search_context(
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
        from app.repositories.fts_repository.faults import FtsValidationError

        async def failing_fts(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            del kwargs
            raise FtsValidationError('Invalid filters', ['Invalid metadata filter'])

        monkeypatch.setattr(search_hybrid, 'fts_search_raw', failing_fts)

        response = await search_hybrid.hybrid_search_context(
            query='anything',
            metadata_filters=BAD_FILTER,
        )

        assert response['results'] == []
        assert response['validation_errors']
        assert embedding_provider.calls == []
