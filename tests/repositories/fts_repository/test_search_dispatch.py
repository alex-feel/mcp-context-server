"""FtsRepository.search hands the caller's scope to the SQLite or the PostgreSQL executor.

The backend type picks the executor; on both, the scope reaches it as the very object the
caller passed, alongside every other argument.
"""

from typing import Any
from typing import Literal
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessScope
from app.access_scope import Scope
from app.repositories.fts_repository import FtsRepository

_RESULT: tuple[list[dict[str, Any]], dict[str, Any]] = ([{'id': 'a'}], {'filters_applied': 1})


def _repository(
    monkeypatch: pytest.MonkeyPatch, backend_type: Literal['sqlite', 'postgresql'],
) -> tuple[FtsRepository, AsyncMock, AsyncMock]:
    """Build a repository on a backend of ``backend_type`` whose two executors are spies."""
    backend = MagicMock()
    backend.backend_type = backend_type
    repository = FtsRepository(backend)
    sqlite_executor = AsyncMock(return_value=_RESULT)
    postgresql_executor = AsyncMock(return_value=_RESULT)
    monkeypatch.setattr(repository, '_search_sqlite', sqlite_executor)
    monkeypatch.setattr(repository, '_search_postgresql', postgresql_executor)
    return repository, sqlite_executor, postgresql_executor


async def _search(repository: FtsRepository, scope: Scope) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search with every argument set, as the scope."""
    return await repository.search(
        query='lattice',
        mode='prefix',
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
        highlight=True,
        language='english',
        explain_query=True,
        scope=scope,
    )


def _assert_forwarded(spy: AsyncMock, scope: Scope) -> None:
    """Assert the spy got every argument of :func:`_search` and the very scope object."""
    spy.assert_awaited_once_with(
        query='lattice',
        mode='prefix',
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
        highlight=True,
        language='english',
        explain_query=True,
        scope=scope,
    )
    assert spy.await_args is not None
    assert spy.await_args.kwargs['scope'] is scope


@pytest.mark.asyncio
@pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
@pytest.mark.parametrize('scope', [AccessScope('bob', frozenset({'team-x'})), SYSTEM_SCOPE], ids=['bob', 'system'])
async def test_executor_receives_the_scope(
    monkeypatch: pytest.MonkeyPatch, backend_type: Literal['sqlite', 'postgresql'], scope: Scope,
) -> None:
    """The backend's executor gets the caller's scope and every argument; the other is never called."""
    repository, sqlite_executor, postgresql_executor = _repository(monkeypatch, backend_type)

    assert await _search(repository, scope) == _RESULT

    called, idle = (
        (sqlite_executor, postgresql_executor) if backend_type == 'sqlite' else (postgresql_executor, sqlite_executor)
    )
    idle.assert_not_awaited()
    _assert_forwarded(called, scope)
