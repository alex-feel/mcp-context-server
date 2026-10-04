"""EmbeddingRepository.search hands the caller's scope to the fp32 or the compressed search.

The compression toggle picks the branch; on both, the scope reaches the search as the very
object the caller passed, alongside every other argument.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessScope
from app.access_scope import Scope
from app.repositories.embedding_repository import EmbeddingRepository

_RESULT: tuple[list[dict[str, Any]], dict[str, Any]] = ([{'id': 'a'}], {'filters_applied': 1})
_SCOPES = pytest.mark.parametrize(
    'scope', [AccessScope('bob', frozenset({'team-x'})), SYSTEM_SCOPE], ids=['bob', 'system'],
)


def _repository(monkeypatch: pytest.MonkeyPatch, *, compressed: bool) -> tuple[EmbeddingRepository, AsyncMock, AsyncMock]:
    """Build a repository whose two searches are spies, with the compression toggle set."""
    monkeypatch.setattr(
        'app.settings.get_settings', lambda: SimpleNamespace(compression=SimpleNamespace(enabled=compressed)),
    )
    backend = MagicMock()
    backend.backend_type = 'sqlite'
    repository = EmbeddingRepository(backend)
    fp32 = AsyncMock(return_value=_RESULT)
    compressed_search = AsyncMock(return_value=_RESULT)
    monkeypatch.setattr(repository, 'search_fp32', fp32)
    monkeypatch.setattr(repository, 'search_compressed', compressed_search)
    return repository, fp32, compressed_search


async def _search(repository: EmbeddingRepository, scope: Scope) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search with every argument set, as the scope."""
    return await repository.search(
        query_embedding=[0.1, 0.2],
        limit=7,
        offset=3,
        thread_id='t',
        source='agent',
        content_type='text',
        tags=['x'],
        start_date='2026-01-01',
        end_date='2026-02-01',
        metadata={'k': 'v'},
        metadata_filters=[{'key': 'k', 'operator': 'eq', 'value': 'v'}],
        explain_query=True,
        scope=scope,
    )


def _assert_forwarded(spy: AsyncMock, scope: Scope) -> None:
    """Assert the spy got every argument of :func:`_search` and the very scope object."""
    spy.assert_awaited_once_with(
        query_embedding=[0.1, 0.2],
        limit=7,
        offset=3,
        thread_id='t',
        source='agent',
        content_type='text',
        tags=['x'],
        start_date='2026-01-01',
        end_date='2026-02-01',
        metadata={'k': 'v'},
        metadata_filters=[{'key': 'k', 'operator': 'eq', 'value': 'v'}],
        explain_query=True,
        scope=scope,
    )
    assert spy.await_args is not None
    assert spy.await_args.kwargs['scope'] is scope


@pytest.mark.asyncio
@_SCOPES
async def test_compressed_branch_receives_the_scope(monkeypatch: pytest.MonkeyPatch, scope: Scope) -> None:
    """With compression on, the compressed search gets the caller's scope and every argument."""
    repository, fp32, compressed_search = _repository(monkeypatch, compressed=True)

    assert await _search(repository, scope) == _RESULT

    fp32.assert_not_awaited()
    _assert_forwarded(compressed_search, scope)


@pytest.mark.asyncio
@_SCOPES
async def test_fp32_branch_receives_the_scope(monkeypatch: pytest.MonkeyPatch, scope: Scope) -> None:
    """With compression off, the fp32 search gets the caller's scope and every argument."""
    repository, fp32, compressed_search = _repository(monkeypatch, compressed=False)

    assert await _search(repository, scope) == _RESULT

    compressed_search.assert_not_awaited()
    _assert_forwarded(fp32, scope)
