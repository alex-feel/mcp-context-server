"""Tests for ContextRepository deletes by id and the batch-delete id snapshot.

Every delete names its rows by id: a thread or criteria delete first snapshots the
matching ids with ``get_ids_matching_batch_criteria`` and then deletes exactly that
snapshot with ``delete_by_ids``. The two-principal behavior of both methods is proven by
cases X1 and X2 of the access-scope suite; these tests cover chunking, criteria semantics
and the statements each backend receives.
"""

import uuid
from collections.abc import Awaitable
from collections.abc import Callable
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from app.access_scope import AccessMode
from app.access_scope import build_access_predicate
from app.backends.base import StorageBackend
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from tests.helpers import LOCAL_SCOPE


async def _store(repos: RepositoryContainer, thread_id: str, text: str, *, source: str = 'user') -> str:
    """Store one private entry owned by the local principal and return its id."""
    context_id, _ = await repos.context.store_with_deduplication(
        scope=LOCAL_SCOPE,
        visibility='private',
        thread_id=thread_id,
        source=source,
        content_type='text',
        text_content=text,
    )
    return context_id


class TestDeleteByIds:
    """Deleting entries by id."""

    @pytest.mark.asyncio
    async def test_delete_by_ids_spans_multiple_chunks(self, repos: RepositoryContainer) -> None:
        """delete_by_ids chunks an id list exceeding the per-statement bound-parameter limit.

        A very large id list (e.g. every entry in a large thread) would exceed a
        backend's per-statement bound-parameter ceiling if bound in a single IN
        clause, so the delete is issued in bounded chunks. This exercises an id
        list well past the chunk boundary with real ids interleaved on both sides
        of it: every real row must be deleted exactly once, the non-existent ids
        must contribute nothing to the count, and an unrelated entry stays intact.
        """
        keep_id = await _store(repos, 'chunk_keep_thread', 'Keep me')
        real_ids = [await _store(repos, 'chunk_del_thread', f'Chunk delete entry {i}') for i in range(5)]

        # Build an id list well past the 900-id chunk boundary out of non-existent
        # ids, then overwrite a few positions straddling that boundary with the
        # real ids so deleted rows fall in more than one chunk.
        mixed = [generate_id() for _ in range(1300)]
        for pos, real_id in zip((0, 450, 899, 900, 1299), real_ids, strict=True):
            mixed[pos] = real_id

        deleted = await repos.context.delete_by_ids(mixed, scope=LOCAL_SCOPE)

        # Only the real rows are deleted; the non-existent ids match nothing.
        assert deleted == len(real_ids)
        assert await repos.context.get_by_ids(real_ids, scope=LOCAL_SCOPE) == []
        remaining = await repos.context.get_by_ids([keep_id], scope=LOCAL_SCOPE)
        assert [row['id'] for row in remaining] == [keep_id]

    @pytest.mark.asyncio
    async def test_delete_by_ids_of_an_empty_list_is_zero(self, repos: RepositoryContainer) -> None:
        """No ids means no statement and nothing deleted."""
        assert await repos.context.delete_by_ids([], scope=LOCAL_SCOPE) == 0


@pytest.mark.usefixtures('sqlite_999_variables')
class TestDeleteByIdsUnderVariableCap:
    """Each 900-id delete chunk plus its owner bind fits within 999 variables."""

    @pytest.mark.asyncio
    async def test_delete_by_ids_binds_1000_ids_with_scope(self, repos: RepositoryContainer) -> None:
        """A 1,000-id delete under a scope binds without error and deletes every stored row."""
        real_ids = [await _store(repos, 'capped_delete', f'Capped delete entry {i}') for i in range(3)]
        ids = [generate_id() for _ in range(1000)]
        for position, real_id in zip((0, 899, 999), real_ids, strict=True):
            ids[position] = real_id

        assert await repos.context.delete_by_ids(ids, scope=LOCAL_SCOPE) == len(real_ids)
        assert await repos.context.get_by_ids(real_ids, scope=LOCAL_SCOPE) == []


class TestThreadSnapshotDelete:
    """A thread delete snapshots the thread's ids and deletes exactly that snapshot."""

    @pytest.mark.asyncio
    async def test_thread_snapshot_then_delete_keeps_other_threads(self, repos: RepositoryContainer) -> None:
        """The snapshot of one thread deletes its entry and leaves another thread intact."""
        await _store(repos, 'del_thread', 'To delete')
        await _store(repos, 'keep_thread', 'To keep')

        snapshot = await repos.context.get_ids_matching_batch_criteria(
            thread_ids=['del_thread'], scope=LOCAL_SCOPE, mode=AccessMode.OWNER,
        )
        deleted = await repos.context.delete_by_ids(snapshot, scope=LOCAL_SCOPE)

        assert deleted == 1
        rows, _ = await repos.context.search_contexts(thread_id='del_thread', scope=LOCAL_SCOPE)
        assert rows == []
        rows, _ = await repos.context.search_contexts(thread_id='keep_thread', scope=LOCAL_SCOPE)
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_thread_snapshot_covers_every_entry_of_the_thread(self, repos: RepositoryContainer) -> None:
        """Every entry of the thread, of either source, is in the snapshot and deleted."""
        await _store(repos, 'multi_del_thread', 'Message 1')
        await _store(repos, 'multi_del_thread', 'Message 2', source='agent')
        await _store(repos, 'multi_del_thread', 'Message 3')

        snapshot = await repos.context.get_ids_matching_batch_criteria(
            thread_ids=['multi_del_thread'], scope=LOCAL_SCOPE, mode=AccessMode.OWNER,
        )

        assert len(snapshot) == 3
        assert await repos.context.delete_by_ids(snapshot, scope=LOCAL_SCOPE) == 3
        rows, _ = await repos.context.search_contexts(thread_id='multi_del_thread', scope=LOCAL_SCOPE)
        assert rows == []

    @pytest.mark.asyncio
    async def test_snapshot_of_a_nonexistent_thread_is_empty(self, repos: RepositoryContainer) -> None:
        """A thread without entries snapshots to nothing."""
        snapshot = await repos.context.get_ids_matching_batch_criteria(
            thread_ids=['nonexistent'], scope=LOCAL_SCOPE, mode=AccessMode.OWNER,
        )

        assert snapshot == []


class TestCriteriaSnapshotDelete:
    """A criteria delete snapshots the ids its AND-combined criteria match and deletes exactly those."""

    @pytest.mark.asyncio
    async def test_snapshot_of_named_ids_then_delete(self, repos: RepositoryContainer) -> None:
        """Named ids snapshot to themselves and are deleted together."""
        ids = [await _store(repos, 'batch-del-thread', f'Batch delete entry {i}') for i in range(3)]

        snapshot = await repos.context.get_ids_matching_batch_criteria(
            context_ids=ids, scope=LOCAL_SCOPE, mode=AccessMode.READ,
        )

        assert sorted(snapshot) == sorted(ids)
        assert await repos.context.delete_by_ids(snapshot, scope=LOCAL_SCOPE) == 3
        assert await repos.context.get_by_ids(ids, scope=LOCAL_SCOPE) == []

    @pytest.mark.asyncio
    async def test_snapshot_of_named_ids_skips_absent_ids(self, repos: RepositoryContainer) -> None:
        """Named ids no entry carries match nothing, so only the stored entry is deleted."""
        context_id = await _store(repos, 'partial-del-thread', 'Entry to delete partially')

        snapshot = await repos.context.get_ids_matching_batch_criteria(
            context_ids=[context_id, generate_id(), generate_id()], scope=LOCAL_SCOPE, mode=AccessMode.READ,
        )

        assert snapshot == [context_id]
        assert await repos.context.delete_by_ids(snapshot, scope=LOCAL_SCOPE) == 1

    @pytest.mark.asyncio
    async def test_snapshot_of_an_empty_id_list_is_empty(self, repos: RepositoryContainer) -> None:
        """An empty id list is no criterion, and a call without criteria matches nothing."""
        await _store(repos, 'empty-criteria-thread', 'Entry no empty criteria may reach')

        for mode in (AccessMode.READ, AccessMode.OWNER):
            assert await repos.context.get_ids_matching_batch_criteria(
                context_ids=[], scope=LOCAL_SCOPE, mode=mode,
            ) == []

    @pytest.mark.asyncio
    async def test_snapshot_ids_not_duplicated_when_the_read_is_retried(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """The snapshot must not accumulate duplicates if the read closure is retried.

        The matched ids are collected per closure invocation, so a transparent read
        retry (which re-invokes the same closure) must not return any id twice.
        """
        context_id = await _store(repos, 'criteria-retry-thread', 'Entry for criteria retry')

        backend = context_repo.backend
        original_execute_read = backend.execute_read

        async def double_execute_read(fn, *args, **kwargs):
            # Simulate a transparent retry: invoke the same closure twice.
            first = await original_execute_read(fn, *args, **kwargs)
            await original_execute_read(fn, *args, **kwargs)
            return first

        with patch.object(backend, 'execute_read', side_effect=double_execute_read):
            snapshot = await context_repo.get_ids_matching_batch_criteria(
                context_ids=[context_id], scope=LOCAL_SCOPE, mode=AccessMode.READ,
            )

        assert snapshot == [context_id]

    @pytest.mark.asyncio
    async def test_get_ids_matching_batch_criteria_spans_chunks_and_semantics(
        self, repos: RepositoryContainer,
    ) -> None:
        """The criteria snapshot chunks oversized id/thread lists and keeps AND semantics.

        A 33,000-id context_ids list bound as one IN clause exceeds SQLite's default
        32,766 per-statement variable ceiling, so the snapshot SELECT runs one
        statement per (id-chunk, thread-chunk) pair. The criteria stay AND-combined:
        a row whose id is listed but whose thread is not (or whose source differs)
        must not match, exactly as with the single unchunked statement.
        """
        matching_ids = [await _store(repos, 'crit-chunk-a', f'Criteria chunk match {i}') for i in range(2)]
        agent_id = await _store(repos, 'crit-chunk-a', 'Criteria chunk agent entry', source='agent')
        other_thread_id = await _store(repos, 'crit-chunk-b', 'Criteria chunk other-thread entry')

        # Real ids straddle the 900-id chunk boundary inside a 33,000-id list.
        context_ids = [generate_id() for _ in range(33000)]
        for pos, real_id in zip(
            (0, 899, 900, 32999),
            (*matching_ids, agent_id, other_thread_id),
            strict=True,
        ):
            context_ids[pos] = real_id

        # The matching thread sits in the SECOND thread chunk of a 1,000-thread list.
        thread_ids = [f'crit-chunk-absent-{i}' for i in range(1000)]
        thread_ids[950] = 'crit-chunk-a'

        matched = await repos.context.get_ids_matching_batch_criteria(
            context_ids=context_ids,
            thread_ids=thread_ids,
            source='user',
            scope=LOCAL_SCOPE,
            mode=AccessMode.READ,
        )

        assert sorted(matched) == sorted(matching_ids)

    @pytest.mark.asyncio
    async def test_snapshot_then_delete_spans_chunks_and_semantics(self, repos: RepositoryContainer) -> None:
        """A snapshot over an oversized context_ids list deletes exactly the AND-matched rows.

        One SELECT per chunk pair builds the snapshot; the delete then removes exactly
        the rows the criteria matched, summing the per-chunk rowcounts.
        """
        user_ids = [await _store(repos, 'del-chunk-thread', f'Delete chunk match {i}') for i in range(2)]
        agent_id = await _store(repos, 'del-chunk-thread', 'Delete chunk agent survivor', source='agent')
        unlisted_id = await _store(repos, 'del-chunk-thread', 'Delete chunk unlisted survivor')

        context_ids = [generate_id() for _ in range(33000)]
        for pos, real_id in zip((0, 900, 32999), (*user_ids, agent_id), strict=True):
            context_ids[pos] = real_id

        snapshot = await repos.context.get_ids_matching_batch_criteria(
            context_ids=context_ids, source='user', scope=LOCAL_SCOPE, mode=AccessMode.READ,
        )
        deleted_count = await repos.context.delete_by_ids(snapshot, scope=LOCAL_SCOPE)

        assert deleted_count == 2
        # The listed agent entry (source mismatch) and the unlisted user entry survive.
        remaining = await repos.context.get_by_ids([*user_ids, agent_id, unlisted_id], scope=LOCAL_SCOPE)
        assert {row['id'] for row in remaining} == {agent_id, unlisted_id}

    @pytest.mark.asyncio
    async def test_postgresql_snapshot_binds_criteria_then_the_predicate_per_chunk(self) -> None:
        """The PostgreSQL snapshot issues one bounded statement per chunk pair with every value bound.

        Asserted against a recording stand-in connection (no live PostgreSQL): 1,300
        ids plus a source and an age filter produce two SELECT statements, a 900-id and
        a 400-id chunk, each binding its ids, the source and the age as a day count,
        followed by the three READ predicate binds numbered after them. The age never
        appears as literal text, and the returned UUIDs come back as canonical ids.
        """
        executed: list[tuple[str, tuple[object, ...]]] = []
        row_id = uuid.UUID(generate_id())

        class _RecordingConn:
            async def fetch(self, query: str, *params: object) -> list[dict[str, object]]:
                executed.append((query, params))
                return [{'id': row_id}]

        pg_backend = Mock()
        pg_backend.backend_type = 'postgresql'

        async def _execute_read(closure: Callable[[object], Awaitable[list[str]]]) -> list[str]:
            return await closure(_RecordingConn())

        pg_backend.execute_read = _execute_read
        repo_pg = ContextRepository(cast(StorageBackend, pg_backend))

        context_ids = [generate_id() for _ in range(1300)]
        matched = await repo_pg.get_ids_matching_batch_criteria(
            context_ids=context_ids, source='user', older_than_days=7, scope=LOCAL_SCOPE, mode=AccessMode.READ,
        )

        assert [len(params) for _query, params in executed] == [905, 405]
        for (query, params), chunk_size in zip(executed, (900, 400), strict=True):
            predicate = build_access_predicate(
                LOCAL_SCOPE, mode=AccessMode.READ, backend_type='postgresql', outer='context_entries',
                start=chunk_size + 3,
            )
            assert query.endswith(f' AND {predicate.sql}')
            assert f'source = ${chunk_size + 1}' in query
            assert f"${chunk_size + 2}::integer * INTERVAL '1 day'" in query
            assert '7 days' not in query
            assert params[chunk_size:] == ('user', 7, *predicate.params)
        assert matched == [row_id.hex, row_id.hex]
