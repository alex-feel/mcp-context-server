"""The raw semantic and full-text legs hand the caller's scope to their repository search.

Both legs run as the scope their caller resolved: the vector search and the full-text search
receive the very scope object, alongside the client filters, and the repository stats come
back unchanged apart from the semantic leg's embedding timing.
"""

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from app.access_scope import AccessScope
from app.tools.search.legs import fts_search_raw
from app.tools.search.legs import semantic_search_raw

_SCOPE = AccessScope('bob', frozenset({'team-x'}))
_STATS: dict[str, Any] = {'filters_applied': 2, 'rows_returned': 0}


@pytest.mark.asyncio
async def test_semantic_leg_forwards_the_scope() -> None:
    """The vector search runs as the given scope with the client filters."""
    repos = MagicMock()
    repos.embeddings.search = AsyncMock(return_value=([], dict(_STATS)))
    embedding_provider = MagicMock()
    embedding_provider.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

    _results, stats = await semantic_search_raw(
        query='q', limit=5, thread_id='t', source='agent', repos=repos,
        embedding_provider=embedding_provider, scope=_SCOPE,
    )

    call = repos.embeddings.search.await_args
    assert call is not None
    assert call.kwargs['scope'] is _SCOPE
    assert (call.kwargs['thread_id'], call.kwargs['source']) == ('t', 'agent')
    assert stats['filters_applied'] == _STATS['filters_applied']


@pytest.mark.asyncio
async def test_fts_leg_forwards_the_scope() -> None:
    """The full-text search runs as the given scope with the client filters."""
    repos = MagicMock()
    repos.fts.is_available = AsyncMock(return_value=True)
    repos.fts.search = AsyncMock(return_value=([], dict(_STATS)))

    _results, stats = await fts_search_raw(
        query='q', limit=5, thread_id='t', source='agent', repos=repos, scope=_SCOPE,
    )

    call = repos.fts.search.await_args
    assert call is not None
    assert call.kwargs['scope'] is _SCOPE
    assert (call.kwargs['thread_id'], call.kwargs['source']) == ('t', 'agent')
    assert stats == _STATS
