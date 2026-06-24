"""Tests for the cleanup-gate predicate on delete_context / delete_context_batch.

``delete_context`` and ``delete_context_batch`` run the explicit
``repos.embeddings.delete(...)`` cleanup when
``settings.embedding.generation_enabled or settings.compression.enabled``
is true. Either flag being true is sufficient evidence that embedding rows
may exist on disk and the explicit cleanup should run.

``settings.semantic_search.enabled`` is unrelated to this gate: it
controls registration of the ``semantic_search_context`` MCP tool ONLY
(per its docstring at ``app/settings.py``) and does not indicate whether
embedding rows exist on disk. The default user configuration
(``ENABLE_EMBEDDING_GENERATION=true, ENABLE_SEMANTIC_SEARCH=false,
ENABLE_EMBEDDING_COMPRESSION=true``) produces a DB full of embeddings;
the explicit cleanup must still run as defense-in-depth alongside FK
CASCADE.
"""

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


def _flip_env_and_refresh_bindings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embedding_generation_enabled: bool,
    semantic_search_enabled: bool,
    compression_enabled: bool,
) -> None:
    """Set the three relevant env vars + refresh module-level settings bindings."""
    monkeypatch.setenv(
        'ENABLE_EMBEDDING_GENERATION',
        'true' if embedding_generation_enabled else 'false',
    )
    monkeypatch.setenv(
        'ENABLE_SEMANTIC_SEARCH',
        'true' if semantic_search_enabled else 'false',
    )
    monkeypatch.setenv(
        'ENABLE_EMBEDDING_COMPRESSION',
        'true' if compression_enabled else 'false',
    )
    get_settings.cache_clear()
    monkeypatch.setattr(context_module, 'settings', get_settings())
    monkeypatch.setattr(batch_module, 'settings', get_settings())


class _FakeRepos:
    """Minimal repositories stub exposing only the methods exercised by the tests.

    Mirrors the RepositoryContainer shape closely enough for delete_context and
    delete_context_batch to run without a real backend. ``embeddings.delete`` is
    an AsyncMock so the test can assert the call presence/absence.
    """

    def __init__(self) -> None:
        self.embeddings = type(
            '_Embeddings', (), {'delete': AsyncMock(return_value=None)},
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
def fake_repos_factory(monkeypatch: pytest.MonkeyPatch) -> _FakeRepos:
    """Inject a fake repositories container by patching ``ensure_repositories``.

    Returns:
        The same fake repos instance both modules will see, so the test
        can assert on its mock interactions.
    """
    fake = _FakeRepos()

    async def _ensure_repositories() -> _FakeRepos:
        return fake

    monkeypatch.setattr(context_module, 'ensure_repositories', _ensure_repositories)
    monkeypatch.setattr(batch_module, 'ensure_repositories', _ensure_repositories)
    return fake


VALID_ID = '0190abcdef1234567890abcdef123456'


@pytest.mark.asyncio
async def test_delete_context_cleanup_runs_under_default_user_config(
    monkeypatch: pytest.MonkeyPatch, fake_repos_factory: _FakeRepos,
) -> None:
    """Default config (embedding gen ON, semantic search OFF, compression ON)
    MUST trigger the explicit embedding cleanup.

    The gate predicate is ``generation_enabled OR compression.enabled``;
    both terms evaluate True in this configuration, so the cleanup runs.
    """
    _flip_env_and_refresh_bindings(
        monkeypatch,
        embedding_generation_enabled=True,
        semantic_search_enabled=False,
        compression_enabled=True,
    )

    await context_module.delete_context(context_ids=[VALID_ID])

    fake_repos_factory.embeddings.delete.assert_awaited_once_with(VALID_ID)


@pytest.mark.asyncio
async def test_delete_context_batch_cleanup_runs_under_default_user_config(
    monkeypatch: pytest.MonkeyPatch, fake_repos_factory: _FakeRepos,
) -> None:
    """Same as above but for delete_context_batch."""
    _flip_env_and_refresh_bindings(
        monkeypatch,
        embedding_generation_enabled=True,
        semantic_search_enabled=False,
        compression_enabled=True,
    )

    await batch_module.delete_context_batch(context_ids=[VALID_ID])

    fake_repos_factory.embeddings.delete.assert_awaited_once_with(VALID_ID)


@pytest.mark.asyncio
async def test_delete_context_cleanup_skipped_only_when_both_disabled(
    monkeypatch: pytest.MonkeyPatch, fake_repos_factory: _FakeRepos,
) -> None:
    """When BOTH generation and compression are disabled, cleanup is SKIPPED.

    This is the only case where there are guaranteed no embedding rows on
    disk, so the explicit cleanup is unnecessary.
    """
    _flip_env_and_refresh_bindings(
        monkeypatch,
        embedding_generation_enabled=False,
        semantic_search_enabled=False,
        compression_enabled=False,
    )

    await context_module.delete_context(context_ids=[VALID_ID])

    fake_repos_factory.embeddings.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_context_cleanup_runs_when_only_compression_enabled(
    monkeypatch: pytest.MonkeyPatch, fake_repos_factory: _FakeRepos,
) -> None:
    """Even if embedding generation is OFF, if compression is ON there may be
    compressed payloads on disk -- cleanup MUST run."""
    _flip_env_and_refresh_bindings(
        monkeypatch,
        embedding_generation_enabled=False,
        semantic_search_enabled=False,
        compression_enabled=True,
    )

    await context_module.delete_context(context_ids=[VALID_ID])

    fake_repos_factory.embeddings.delete.assert_awaited_once_with(VALID_ID)


@pytest.mark.asyncio
async def test_delete_context_batch_cleanup_respects_combined_criteria(
    monkeypatch: pytest.MonkeyPatch, fake_repos_factory: _FakeRepos,
) -> None:
    """Combined context_ids + source deletes embeddings only for the matching subset.

    Regression: deleting embeddings for EVERY context_id while the row-delete is
    AND-combined with source/older_than_days orphaned a surviving entry. The cleanup
    now pre-queries the exact to-be-deleted ids and deletes embeddings only for those.
    """
    _flip_env_and_refresh_bindings(
        monkeypatch,
        embedding_generation_enabled=True,
        semantic_search_enabled=False,
        compression_enabled=True,
    )
    other_id = '0190abcdef1234567890abcdef654321'
    # Only VALID_ID matches the combined criteria; other_id is excluded by source.
    fake_repos_factory.context.get_ids_matching_batch_criteria = AsyncMock(return_value=[VALID_ID])

    await batch_module.delete_context_batch(context_ids=[VALID_ID, other_id], source='user')

    # Embeddings deleted ONLY for the matching id, never the excluded one.
    fake_repos_factory.embeddings.delete.assert_awaited_once_with(VALID_ID)
    # The criteria query received BOTH context_ids and the source filter.
    fake_repos_factory.context.get_ids_matching_batch_criteria.assert_awaited_once()
    call_kwargs = fake_repos_factory.context.get_ids_matching_batch_criteria.await_args.kwargs
    assert call_kwargs.get('context_ids') == [VALID_ID, other_id]
    assert call_kwargs.get('source') == 'user'
