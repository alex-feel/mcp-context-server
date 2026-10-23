"""Each fp32 vector-search row reports its entry's visibility and whether the caller owns it.

The final select of both fp32 statements selects ``visibility`` and an ownership flag, 1 or
0, computed from the caller's OWNER predicate. On SQLite its bind follows the query vector
and precedes LIMIT and OFFSET in statement text order; on PostgreSQL it is numbered after
the READ predicate's binds, and LIMIT and OFFSET after it.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.backends.base import BackendAccess
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository import EmbeddingRepository
from tests.conftest import requires_semantic_search
from tests.helpers import LOCAL_SCOPE
from tests.helpers import store_single_chunk_embedding

_THREAD = 'fp32-read-back'
_BOB = AccessScope('bob', frozenset(), may_publish=True)


async def _store(
    repos: RepositoryContainer,
    embeddings: EmbeddingRepository,
    *,
    scope: AccessScope,
    visibility: str,
    text: str,
    embedding: list[float],
) -> str:
    """Store one entry owned by the scope with one embedding chunk and return its id."""
    context_id, _ = await repos.context.store_with_deduplication(
        scope=scope,
        visibility=visibility,
        thread_id=_THREAD,
        source='user',
        content_type='text',
        text_content=text,
        metadata=None,
    )
    await store_single_chunk_embedding(embeddings, context_id, embedding)
    return context_id


@pytest.mark.asyncio
@requires_semantic_search
async def test_sqlite_rows_report_visibility_and_ownership(
    async_db_with_embeddings: StorageBackend,
    embedding_dim: int,
) -> None:
    """Each caller reads back the visibility of every row it may read and owns only its own, on every page."""
    backend = async_db_with_embeddings
    repos = RepositoryContainer(backend)
    embeddings = EmbeddingRepository(backend)
    own_private = await _store(
        repos, embeddings, scope=LOCAL_SCOPE, visibility='private', text='own private', embedding=[0.1] * embedding_dim,
    )
    bob_public = await _store(
        repos, embeddings, scope=_BOB, visibility='public', text='bob public', embedding=[0.2] * embedding_dim,
    )
    bob_private = await _store(
        repos, embeddings, scope=_BOB, visibility='private', text='bob private', embedding=[0.3] * embedding_dim,
    )
    query = [0.1] * embedding_dim

    async def read_back(scope: AccessScope, *, limit: int, offset: int) -> list[tuple[str, object, object]]:
        rows, _stats = await embeddings.search_fp32(
            query_embedding=query, limit=limit, offset=offset, thread_id=_THREAD, source='user', scope=scope,
        )
        return [(row['id'], row['visibility'], row['is_owner']) for row in rows]

    assert await read_back(LOCAL_SCOPE, limit=10, offset=0) == [(own_private, 'private', 1), (bob_public, 'public', 0)]
    assert await read_back(LOCAL_SCOPE, limit=1, offset=1) == [(bob_public, 'public', 0)]
    assert await read_back(_BOB, limit=10, offset=0) == [(bob_public, 'public', 1), (bob_private, 'private', 1)]
    assert await read_back(_BOB, limit=1, offset=1) == [(bob_private, 'private', 1)]


@pytest.mark.asyncio
async def test_postgresql_owner_bind_follows_the_read_predicate() -> None:
    """The ownership flag binds the caller after the READ predicate; LIMIT and OFFSET follow it."""
    captured: list[tuple[str, tuple[object, ...]]] = []

    async def capture_fetch(sql: str, *args: object) -> list[object]:
        captured.append((sql, args))
        return []

    async def execute_read(operation: Callable[[AsyncMock], Awaitable[object]], *, access_scope: BackendAccess) -> object:
        _ = access_scope
        connection = AsyncMock()
        connection.fetch = capture_fetch
        return await operation(connection)

    backend = MagicMock()
    backend.backend_type = 'postgresql'
    backend.readable_relation = lambda table, *, alias: f'{table} {alias}'
    backend.execute_read = execute_read
    repository = EmbeddingRepository(backend)

    await repository.search_fp32(query_embedding=[0.5, 0.5], limit=7, offset=3, thread_id='t', scope=LOCAL_SCOPE)

    [(sql, args)] = captured
    # $1 the query vector, $2 the thread, $3-$5 the READ predicate, $6 the owner, $7 LIMIT, $8 OFFSET.
    assert 'ce.visibility' in sql
    assert 'CASE WHEN ce.owner_id = $6 THEN 1 ELSE 0 END AS is_owner' in sql
    assert 'LIMIT $7 OFFSET $8' in sql
    assert args[1:2] == ('t',)
    assert args[5:] == ('local', 7, 3)
