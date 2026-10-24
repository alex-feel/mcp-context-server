"""Fixtures shared by the search tool tests that run the ranked searches against a real database."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

import app.startup
from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.migrations import apply_fts_migration
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository import EmbeddingRepository
from tests.helpers import store_single_chunk_embedding

RANKED_THREAD = 'ranked-read-back'
RANKED_KEYWORD = 'lattice'


@pytest_asyncio.fixture
async def ranked_entries(
    async_db_with_embeddings: StorageBackend,
    embedding_dim: int,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, str]:
    """Store alice's private and public entries and bob's private one, ready for every ranked search.

    Each entry mentions :data:`RANKED_KEYWORD` and carries one embedding: alice's private
    entry lies on the query vector the fake embedding provider returns, her public entry
    next and bob's farthest, so the full-text, semantic and hybrid searches all reach every
    entry the caller may read. No reranking provider is installed.

    Args:
        async_db_with_embeddings: The SQLite database with the vector tables; the FTS index is added here.
        embedding_dim: The embedding dimension of the vector tables.
        monkeypatch: Installs the fake embedding provider and removes any reranking provider.

    Returns:
        The id of every entry keyed by its label.
    """
    backend = async_db_with_embeddings
    repos = RepositoryContainer(backend)
    await apply_fts_migration(backend, force=True)
    embeddings = EmbeddingRepository(backend)
    ids: dict[str, str] = {}
    for label, owner, visibility, weight in (
        ('alice_private', 'alice', 'private', 0.1),
        ('alice_public', 'alice', 'public', 0.2),
        ('bob_private', 'bob', 'private', 0.3),
    ):
        context_id, _ = await repos.context.store_with_deduplication(
            scope=AccessScope(owner, frozenset(), may_publish=True),
            visibility=visibility,
            thread_id=RANKED_THREAD,
            source='agent',
            content_type='text',
            text_content=f'{label} mentions the {RANKED_KEYWORD} keyword',
            metadata=None,
        )
        await store_single_chunk_embedding(embeddings, context_id, [weight] * embedding_dim)
        ids[label] = context_id

    provider = MagicMock()
    provider.embed_query = AsyncMock(return_value=[0.1] * embedding_dim)
    monkeypatch.setattr(app.startup, '_embedding_provider', provider)
    monkeypatch.setattr(app.startup, '_reranking_provider', None)
    return ids
