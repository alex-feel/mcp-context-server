"""Tests for the cleanup-gate predicate on delete_context / delete_context_batch.

``delete_context`` and ``delete_context_batch`` run the explicit
``repos.embeddings.delete(...)`` cleanup when the embedding tables were ever
PROVISIONED (``repos.embeddings.embedding_tables_exist()`` is true), NOT when
the runtime ``ENABLE_EMBEDDING_GENERATION`` / ``ENABLE_EMBEDDING_COMPRESSION``
toggles are on.

The durable table-presence signal is the correct gate because a prior session
may have written embeddings that a now-disabled toggle would skip cleaning: on
SQLite the vec0 virtual table has no FK CASCADE and is reachable only through
the ``embedding_chunks`` bridge, so once that bridge cascades away with the
context row the orphaned vectors can never be found or deleted. Gating on the
runtime toggles instead left those rows behind permanently when generation and
compression were both turned off at delete time. The update path already gates
the structurally identical stale-embedding cleanup on the same
``embedding_tables_exist()`` signal (see ``app/tools/_shared.py``); the delete
paths now mirror it.
"""

from collections.abc import Callable
from collections.abc import Generator
from unittest.mock import AsyncMock

import pytest

import app.tools.batch as batch_module
import app.tools.context as context_module
from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset the settings singleton between tests so env flips do not leak.

    Yields:
        Control to the test body.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


VALID_ID = '0190abcdef1234567890abcdef123456'


class _FakeRepos:
    """Minimal repositories stub exposing only the methods exercised by the tests.

    Mirrors the RepositoryContainer shape closely enough for delete_context and
    delete_context_batch to run without a real backend. ``embeddings.delete`` and
    ``embeddings.embedding_tables_exist`` are AsyncMocks so the test can drive the
    gate and assert the cleanup call presence/absence.
    """

    def __init__(self, *, tables_exist: bool) -> None:
        self.embeddings = type(
            '_Embeddings', (),
            {
                'delete': AsyncMock(return_value=None),
                'embedding_tables_exist': AsyncMock(return_value=tables_exist),
            },
        )()
        self.context = type(
            '_Context', (),
            {
                'delete_by_ids': AsyncMock(return_value=1),
                'delete_by_thread': AsyncMock(return_value=0),
                'backend': type('_Backend', (), {'backend_type': 'sqlite'})(),
                'search_contexts': AsyncMock(return_value=([], None)),
                # The criteria query returns the ids the combined delete will remove;
                # for these single-id tests that is exactly VALID_ID. The cleanup
                # (SQLite-only) deletes embeddings for this returned subset.
                'get_ids_matching_batch_criteria': AsyncMock(return_value=[VALID_ID]),
                'delete_contexts_batch': AsyncMock(return_value=(1, ['context_ids'])),
            },
        )()


@pytest.fixture
def make_fake_repos(monkeypatch: pytest.MonkeyPatch) -> Callable[..., _FakeRepos]:
    """Return a factory that injects a fake repos container with a chosen gate state.

    Returns:
        A callable ``(*, tables_exist: bool) -> _FakeRepos`` that patches
        ``ensure_repositories`` in both tool modules to return the fake and
        hands the fake back so the test can assert on its mock interactions.
    """

    def _factory(*, tables_exist: bool) -> _FakeRepos:
        fake = _FakeRepos(tables_exist=tables_exist)

        async def _ensure_repositories() -> _FakeRepos:
            return fake

        monkeypatch.setattr(context_module, 'ensure_repositories', _ensure_repositories)
        monkeypatch.setattr(batch_module, 'ensure_repositories', _ensure_repositories)
        return fake

    return _factory


@pytest.mark.asyncio
async def test_delete_context_cleanup_runs_when_embedding_tables_exist(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """When the embedding tables are provisioned, the explicit cleanup MUST run."""
    fake = make_fake_repos(tables_exist=True)

    await context_module.delete_context(context_ids=[VALID_ID])

    fake.embeddings.delete.assert_awaited_once_with(VALID_ID)


@pytest.mark.asyncio
async def test_delete_context_batch_cleanup_runs_when_embedding_tables_exist(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """Same as above but for delete_context_batch (SQLite-only cleanup branch)."""
    fake = make_fake_repos(tables_exist=True)

    await batch_module.delete_context_batch(context_ids=[VALID_ID])

    fake.embeddings.delete.assert_awaited_once_with(VALID_ID)


@pytest.mark.asyncio
async def test_delete_context_cleanup_skipped_when_embedding_tables_absent(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """When embeddings were never provisioned, the cleanup is a no-op (skipped).

    A database that never created the embedding tables has nothing to clean, so
    the explicit ``embeddings.delete`` is not called.
    """
    fake = make_fake_repos(tables_exist=False)

    await context_module.delete_context(context_ids=[VALID_ID])

    fake.embeddings.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_context_cleanup_runs_after_generation_disabled(
    monkeypatch: pytest.MonkeyPatch, make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """Regression: a prior session's embeddings are cleaned even with both toggles OFF.

    The orphan bug: when ``ENABLE_EMBEDDING_GENERATION`` and
    ``ENABLE_EMBEDDING_COMPRESSION`` are BOTH false at delete time but the
    embedding tables still exist (a prior session wrote fp32 vec0 rows), the old
    toggle-based gate skipped the cleanup, permanently orphaning the FK-less vec0
    vectors once the ``embedding_chunks`` bridge cascaded away. The gate now keys
    on table presence, so the cleanup still runs regardless of the toggles.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    monkeypatch.setattr(context_module, 'settings', get_settings())
    monkeypatch.setattr(batch_module, 'settings', get_settings())

    # Tables exist (prior session provisioned + wrote embeddings).
    fake = make_fake_repos(tables_exist=True)

    await context_module.delete_context(context_ids=[VALID_ID])

    fake.embeddings.delete.assert_awaited_once_with(VALID_ID)


@pytest.mark.asyncio
async def test_delete_context_batch_cleanup_respects_combined_criteria(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """Combined context_ids + source deletes embeddings only for the matching subset.

    Regression: deleting embeddings for EVERY context_id while the row-delete is
    AND-combined with source/older_than_days orphaned a surviving entry. The cleanup
    now pre-queries the exact to-be-deleted ids and deletes embeddings only for those.
    """
    fake = make_fake_repos(tables_exist=True)
    other_id = '0190abcdef1234567890abcdef654321'
    # Only VALID_ID matches the combined criteria; other_id is excluded by source.
    fake.context.get_ids_matching_batch_criteria = AsyncMock(return_value=[VALID_ID])

    await batch_module.delete_context_batch(context_ids=[VALID_ID, other_id], source='user')

    # Embeddings deleted ONLY for the matching id, never the excluded one.
    fake.embeddings.delete.assert_awaited_once_with(VALID_ID)
    # The criteria query received BOTH context_ids and the source filter.
    fake.context.get_ids_matching_batch_criteria.assert_awaited_once()
    await_args = fake.context.get_ids_matching_batch_criteria.await_args
    assert await_args is not None
    call_kwargs = await_args.kwargs
    assert call_kwargs.get('context_ids') == [VALID_ID, other_id]
    assert call_kwargs.get('source') == 'user'
