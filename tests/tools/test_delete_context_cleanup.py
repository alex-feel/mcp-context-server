"""Tests for the embedding cleanup on delete_context / delete_context_batch.

Three properties are pinned here:

1. The cleanup gate keys on whether the embedding tables were ever PROVISIONED
   (``repos.embeddings.embedding_tables_exist()``), NOT on the runtime
   ``ENABLE_EMBEDDING_GENERATION`` / ``ENABLE_EMBEDDING_COMPRESSION`` toggles. A
   prior session may have written embeddings that a now-disabled toggle would
   skip cleaning: on SQLite the fp32 vec0 virtual table has no FK CASCADE and is
   reachable only through the ``embedding_chunks`` bridge, so once that bridge
   cascades away with the context row the orphaned vectors can never be found or
   deleted.
2. The explicit per-entry cleanup is SKIPPED whenever CASCADE already covers the
   embedding rows -- with compression enabled the fp32 vec0 table does not exist
   and the compressed payload table carries ``ON DELETE CASCADE``, so the loop
   would only add one write round trip per deleted entry. On a thread-wide or
   criteria-wide delete that loop is unbounded and holds the single SQLite
   writer while every other client stalls.
3. Cleanup and the row delete share ONE transaction, so a failure between them
   can never leave entries stripped of their vectors while their rows survive.
"""

import sqlite3
from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Generator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError

import app.tools._shared as shared_module
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


@pytest.fixture(autouse=True)
def fp32_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run these tests in fp32 mode, where the explicit cleanup is required.

    With compression enabled (the default) the compressed payload table cascades
    with the context row, so the explicit loop is correctly skipped; the tests
    that assert the loop RUNS therefore need the fp32 layout. The dedicated
    cascade tests below re-enable compression explicitly.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    monkeypatch.setattr(shared_module, 'settings', get_settings())


VALID_ID = '0190abcdef1234567890abcdef123456'


class _FakeTransaction:
    """Stand-in for TransactionContext exposing only what the tools read."""

    def __init__(self, backend_type: str) -> None:
        self.connection = object()
        self.backend_type = backend_type


class _FakeBackend:
    """Backend stub whose begin_transaction yields a recording fake transaction."""

    def __init__(self, backend_type: str) -> None:
        self.backend_type = backend_type
        self.transactions: list[_FakeTransaction] = []

    @asynccontextmanager
    async def begin_transaction(self) -> AsyncIterator[_FakeTransaction]:
        """Yield a fake transaction and record it for assertions.

        Yields:
            The fake transaction context handed to the tool body.
        """
        txn = _FakeTransaction(self.backend_type)
        self.transactions.append(txn)
        yield txn


class _FakeRepos:
    """Minimal repositories stub exposing only the methods exercised by the tests.

    Mirrors the RepositoryContainer shape closely enough for delete_context and
    delete_context_batch to run without a real backend.
    """

    def __init__(self, *, tables_exist: bool, backend_type: str = 'sqlite') -> None:
        self.embeddings = SimpleNamespace(
            delete_all_chunks_bulk=AsyncMock(return_value=0),
            embedding_tables_exist=AsyncMock(return_value=tables_exist),
        )
        self.context = SimpleNamespace(
            delete_by_ids=AsyncMock(return_value=1),
            delete_by_thread=AsyncMock(return_value=0),
            delete_contexts_batch=AsyncMock(return_value=(1, ['context_ids: 1 ids'])),
            backend=_FakeBackend(backend_type),
            search_contexts=AsyncMock(return_value=([], None)),
            # The criteria query returns the SNAPSHOT of ids the combined
            # criteria match; for these single-id tests that is exactly
            # VALID_ID. On SQLite the cleanup deletes embeddings for this
            # subset and the destructive step (delete_by_ids) deletes exactly
            # the same subset.
            get_ids_matching_batch_criteria=AsyncMock(return_value=[VALID_ID]),
        )


@pytest.fixture
def make_fake_repos(monkeypatch: pytest.MonkeyPatch) -> Callable[..., _FakeRepos]:
    """Return a factory that injects a fake repos container with a chosen gate state.

    Returns:
        A callable ``(*, tables_exist: bool, backend_type: str) -> _FakeRepos``
        that patches ``ensure_repositories`` in both tool modules to return the
        fake and hands the fake back so the test can assert on its mocks.
    """

    def _factory(*, tables_exist: bool, backend_type: str = 'sqlite') -> _FakeRepos:
        fake = _FakeRepos(tables_exist=tables_exist, backend_type=backend_type)

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

    fake.embeddings.delete_all_chunks_bulk.assert_awaited_once()
    assert fake.embeddings.delete_all_chunks_bulk.await_args is not None
    assert fake.embeddings.delete_all_chunks_bulk.await_args.args[0] == [VALID_ID]


@pytest.mark.asyncio
async def test_delete_context_batch_cleanup_runs_when_embedding_tables_exist(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """Same as above but for delete_context_batch (SQLite-only cleanup branch)."""
    fake = make_fake_repos(tables_exist=True)

    await batch_module.delete_context_batch(context_ids=[VALID_ID])

    fake.embeddings.delete_all_chunks_bulk.assert_awaited_once()
    assert fake.embeddings.delete_all_chunks_bulk.await_args is not None
    assert fake.embeddings.delete_all_chunks_bulk.await_args.args[0] == [VALID_ID]


@pytest.mark.asyncio
async def test_delete_context_cleanup_skipped_when_embedding_tables_absent(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """When embeddings were never provisioned, the cleanup is a no-op (skipped).

    A database that never created the embedding tables has nothing to clean.
    """
    fake = make_fake_repos(tables_exist=False)

    await context_module.delete_context(context_ids=[VALID_ID])

    fake.embeddings.delete_all_chunks_bulk.assert_not_awaited()


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
    monkeypatch.setattr(shared_module, 'settings', get_settings())

    # Tables exist (prior session provisioned + wrote embeddings).
    fake = make_fake_repos(tables_exist=True)

    await context_module.delete_context(context_ids=[VALID_ID])

    fake.embeddings.delete_all_chunks_bulk.assert_awaited_once()


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
    fake.embeddings.delete_all_chunks_bulk.assert_awaited_once()
    assert fake.embeddings.delete_all_chunks_bulk.await_args is not None
    assert fake.embeddings.delete_all_chunks_bulk.await_args.args[0] == [VALID_ID]
    # The criteria query received BOTH context_ids and the source filter.
    fake.context.get_ids_matching_batch_criteria.assert_awaited_once()
    await_args = fake.context.get_ids_matching_batch_criteria.await_args
    assert await_args is not None
    call_kwargs = await_args.kwargs
    assert call_kwargs.get('context_ids') == [VALID_ID, other_id]
    assert call_kwargs.get('source') == 'user'


@pytest.mark.asyncio
async def test_delete_context_cleanup_and_row_delete_share_one_transaction(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """Cleanup and the row delete must run inside the SAME transaction.

    Without the shared transaction, a failure or cancellation after the cleanup
    loop leaves entries whose vectors were removed while their rows survive:
    permanently missing from semantic and hybrid search, with nothing that
    regenerates embeddings for an unchanged entry.
    """
    fake = make_fake_repos(tables_exist=True)

    await context_module.delete_context(context_ids=[VALID_ID])

    assert len(fake.context.backend.transactions) == 1
    txn = fake.context.backend.transactions[0]
    assert fake.embeddings.delete_all_chunks_bulk.await_args is not None
    assert fake.embeddings.delete_all_chunks_bulk.await_args.kwargs['txn'] is txn
    assert fake.context.delete_by_ids.await_args is not None
    assert fake.context.delete_by_ids.await_args.kwargs['txn'] is txn


@pytest.mark.asyncio
async def test_delete_context_batch_cleanup_and_row_delete_share_one_transaction(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """Same single-transaction guarantee for the batch criteria delete."""
    fake = make_fake_repos(tables_exist=True)

    await batch_module.delete_context_batch(thread_ids=['thread-abc'])

    assert len(fake.context.backend.transactions) == 1
    txn = fake.context.backend.transactions[0]
    assert fake.embeddings.delete_all_chunks_bulk.await_args is not None
    assert fake.embeddings.delete_all_chunks_bulk.await_args.kwargs['txn'] is txn
    assert fake.context.delete_by_ids.await_args is not None
    assert fake.context.delete_by_ids.await_args.kwargs['txn'] is txn


@pytest.mark.asyncio
async def test_delete_context_skips_per_entry_cleanup_under_compression(
    monkeypatch: pytest.MonkeyPatch, make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """With compression on, CASCADE covers the embedding rows: no per-entry loop.

    The compressed payload table is an ordinary table with ON DELETE CASCADE on
    context_id (the fp32 vec0 virtual table, the only FK-less one, is dropped by
    the compression migration), so a single DELETE removes every embedding row.
    Issuing one extra write round trip per entry would be pure overhead -- and on
    a thread-wide or criteria-wide delete an unbounded one that monopolizes the
    single SQLite writer.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('SQLITE_FOREIGN_KEYS', 'true')
    get_settings.cache_clear()
    monkeypatch.setattr(shared_module, 'settings', get_settings())

    fake = make_fake_repos(tables_exist=True)

    await context_module.delete_context(context_ids=[VALID_ID])

    fake.embeddings.delete_all_chunks_bulk.assert_not_awaited()
    fake.context.delete_by_ids.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_batch_skips_per_entry_cleanup_under_compression(
    monkeypatch: pytest.MonkeyPatch, make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """The criteria-wide SQLite delete skips the loop under compression too."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('SQLITE_FOREIGN_KEYS', 'true')
    get_settings.cache_clear()
    monkeypatch.setattr(shared_module, 'settings', get_settings())

    fake = make_fake_repos(tables_exist=True)

    # older_than_days alone is refused (it would reach the whole database), so the
    # age criterion is combined with a source filter here.
    await batch_module.delete_context_batch(older_than_days=30, source='agent')

    fake.embeddings.delete_all_chunks_bulk.assert_not_awaited()
    fake.context.delete_by_ids.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_context_keeps_cleanup_when_foreign_keys_disabled(
    monkeypatch: pytest.MonkeyPatch, make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """Compression alone is not enough: CASCADE needs PRAGMA foreign_keys ON.

    An operator who turned foreign keys off gets no cascade, so the explicit
    cleanup must still run or the compressed payload rows orphan.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('SQLITE_FOREIGN_KEYS', 'false')
    get_settings.cache_clear()
    monkeypatch.setattr(shared_module, 'settings', get_settings())

    fake = make_fake_repos(tables_exist=True)

    await context_module.delete_context(context_ids=[VALID_ID])

    fake.embeddings.delete_all_chunks_bulk.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_context_by_ids_skips_cleanup_on_postgresql(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """On PostgreSQL the context_ids branch relies on ON DELETE CASCADE.

    The explicit pre-delete was redundant there (its two sibling branches were
    already SQLite-gated), so the branch now behaves symmetrically: one atomic
    delete, no per-entry embedding round trips.
    """
    fake = make_fake_repos(tables_exist=True, backend_type='postgresql')

    await context_module.delete_context(context_ids=[VALID_ID])

    fake.embeddings.delete_all_chunks_bulk.assert_not_awaited()
    fake.context.delete_by_ids.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_context_rejects_context_ids_and_thread_id_together(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """Supplying BOTH selectors is refused instead of silently deleting only the ids.

    The two parameters are documented as mutually exclusive and the dispatch is
    if/elif, so accepting the combination executed half the request -- deleting the
    listed ids while leaving the rest of the named thread in place -- and still
    reported success. For an irreversible tool that is the wrong failure mode.
    """
    fake = make_fake_repos(tables_exist=True)

    with pytest.raises(ToolError, match='mutually exclusive'):
        await context_module.delete_context(context_ids=[VALID_ID], thread_id='thread-abc')

    fake.context.delete_by_ids.assert_not_awaited()
    fake.context.delete_by_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_lock_contention_rolls_back_and_retries(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """A locked embedding cleanup aborts the delete, which is then retried whole.

    The cleanup is the FIRST write of a DEFERRED transaction, so an external lock
    holder surfaces SQLITE_BUSY exactly there. Logging it and letting the row delete
    commit anyway removed the context rows while their FK-less vec0 vectors survived
    unreachable -- the very leak the cleanup exists to prevent -- and reported
    success. The retryable family now propagates, rolling the transaction back, and
    the bounded retry re-runs cleanup and delete together once the lock clears.
    """
    fake = make_fake_repos(tables_exist=True)
    fake.embeddings.delete_all_chunks_bulk = AsyncMock(
        side_effect=[sqlite3.OperationalError('database is locked'), 0],
    )

    result = await context_module.delete_context(context_ids=[VALID_ID])

    assert result['success'] is True
    assert result['deleted_count'] == 1
    # Two attempts at the cleanup, but the row delete ran only in the attempt whose
    # cleanup succeeded: the first transaction rolled back before reaching it.
    assert fake.embeddings.delete_all_chunks_bulk.await_count == 2
    fake.context.delete_by_ids.assert_awaited_once()
    assert len(fake.context.backend.transactions) == 2


@pytest.mark.asyncio
async def test_cleanup_unreadable_embeddings_still_fails_open(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """A genuinely unreadable embedding row must not block the delete.

    Fail-open is scoped to the conditions it was written for (a missing vec0
    module, a corrupted row): deleting an entry stays possible, with the row delete
    committing in the same transaction. Only write contention is re-raised.
    """
    fake = make_fake_repos(tables_exist=True)
    fake.embeddings.delete_all_chunks_bulk = AsyncMock(
        side_effect=sqlite3.OperationalError('no such module: vec0'),
    )

    result = await context_module.delete_context(context_ids=[VALID_ID])

    assert result['success'] is True
    fake.context.delete_by_ids.assert_awaited_once()
    assert len(fake.context.backend.transactions) == 1


@pytest.mark.asyncio
async def test_batch_delete_cleanup_lock_contention_rolls_back_and_retries(
    make_fake_repos: Callable[..., _FakeRepos],
) -> None:
    """The criteria-wide SQLite delete shares the same rollback-and-retry behavior."""
    fake = make_fake_repos(tables_exist=True)
    fake.embeddings.delete_all_chunks_bulk = AsyncMock(
        side_effect=[sqlite3.OperationalError('database is locked'), 0],
    )

    result = await batch_module.delete_context_batch(thread_ids=['thread-abc'])

    assert result['deleted_count'] == 1
    assert fake.embeddings.delete_all_chunks_bulk.await_count == 2
    fake.context.delete_by_ids.assert_awaited_once()
    assert len(fake.context.backend.transactions) == 2
