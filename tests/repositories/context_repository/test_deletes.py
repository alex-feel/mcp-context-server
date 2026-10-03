"""Tests for ContextRepository deletes by id, by thread and by batch criteria."""

from collections.abc import Awaitable
from collections.abc import Callable
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from app.backends.base import StorageBackend
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from tests.helpers import LOCAL_SCOPE


class TestContextRepositoryDelete:
    """Test delete operations in ContextRepository."""

    @pytest.mark.asyncio
    async def test_delete_by_thread_id(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test deleting entries by thread_id."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='del_thread',
            source='user',
            content_type='text',
            text_content='To delete',
        )
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='keep_thread',
            source='user',
            content_type='text',
            text_content='To keep',
        )

        deleted = await repos.context.delete_by_thread(thread_id='del_thread')

        assert deleted == 1

        # Verify deletion
        rows, _ = await repos.context.search_contexts(thread_id='del_thread', scope=LOCAL_SCOPE)
        assert len(rows) == 0

        # Verify other thread kept
        rows, _ = await repos.context.search_contexts(thread_id='keep_thread', scope=LOCAL_SCOPE)
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_delete_multiple_entries(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test deleting multiple entries from same thread."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='multi_del_thread',
            source='user',
            content_type='text',
            text_content='Message 1',
        )
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='multi_del_thread',
            source='agent',
            content_type='text',
            text_content='Message 2',
        )
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='multi_del_thread',
            source='user',
            content_type='text',
            text_content='Message 3',
        )

        deleted = await repos.context.delete_by_thread(thread_id='multi_del_thread')

        assert deleted == 3

        # Verify all deleted
        rows, _ = await repos.context.search_contexts(thread_id='multi_del_thread', scope=LOCAL_SCOPE)
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_thread(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test deleting from nonexistent thread returns 0."""
        deleted = await repos.context.delete_by_thread(thread_id='nonexistent')

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_delete_by_ids_spans_multiple_chunks(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """delete_by_ids chunks an id list exceeding the per-statement bound-parameter limit.

        A very large id list (e.g. every entry in a large thread) would exceed a
        backend's per-statement bound-parameter ceiling if bound in a single IN
        clause, so the delete is issued in bounded chunks. This exercises an id
        list well past the chunk boundary with real ids interleaved on both sides
        of it: every real row must be deleted exactly once, the non-existent ids
        must contribute nothing to the count, and an unrelated entry stays intact.
        """
        keep_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='chunk_keep_thread',
            source='user',
            content_type='text',
            text_content='Keep me',
        )

        real_ids: list[str] = []
        for i in range(5):
            ctx_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='chunk_del_thread',
                source='user',
                content_type='text',
                text_content=f'Chunk delete entry {i}',
            )
            real_ids.append(ctx_id)

        # Build an id list well past the 900-id chunk boundary out of non-existent
        # ids, then overwrite a few positions straddling that boundary with the
        # real ids so deleted rows fall in more than one chunk.
        mixed = [generate_id() for _ in range(1300)]
        for pos, real_id in zip((0, 450, 899, 900, 1299), real_ids, strict=True):
            mixed[pos] = real_id

        deleted = await repos.context.delete_by_ids(mixed)

        # Only the real rows are deleted; the non-existent ids match nothing.
        assert deleted == len(real_ids)

        # All real rows are gone.
        assert await repos.context.get_by_ids(real_ids, scope=LOCAL_SCOPE) == []

        # The unrelated entry is untouched.
        remaining = await repos.context.get_by_ids([keep_id], scope=LOCAL_SCOPE)
        assert len(remaining) == 1
        assert remaining[0]['id'] == keep_id


class TestContextRepositoryBatchDelete:
    """Test delete_contexts_batch method of ContextRepository."""

    @pytest.mark.asyncio
    async def test_delete_contexts_batch(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Batch delete removes multiple entries."""
        ids = []
        for i in range(3):
            ctx_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='batch-del-thread',
                source='user',
                content_type='text',
                text_content=f'Batch delete entry {i}',
            )
            ids.append(ctx_id)

        deleted_count, criteria = await context_repo.delete_contexts_batch(context_ids=ids)
        assert deleted_count == 3

        rows = await context_repo.get_by_ids(ids, scope=LOCAL_SCOPE)
        assert rows == []

    @pytest.mark.asyncio
    async def test_delete_contexts_batch_partial_ids(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Batch delete with mix of existing and nonexistent IDs."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='partial-del-thread',
            source='user',
            content_type='text',
            text_content='Entry to delete partially',
        )
        deleted_count, _ = await context_repo.delete_contexts_batch(
            context_ids=[ctx_id, generate_id(), generate_id()],
        )
        assert deleted_count == 1

    @pytest.mark.asyncio
    async def test_delete_contexts_batch_empty_list(
        self, context_repo: ContextRepository,
    ) -> None:
        """Batch delete with empty list returns 0."""
        deleted_count, _ = await context_repo.delete_contexts_batch(context_ids=[])
        assert deleted_count == 0

    @pytest.mark.asyncio
    async def test_delete_contexts_batch_criteria_not_duplicated_on_retry(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """criteria_used must not accumulate duplicates if the write closure is retried.

        criteria_used is built per closure invocation, so a transparent write
        retry (which re-invokes the same closure) must not append the same
        criteria strings twice into the returned list.
        """
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='criteria-retry-thread',
            source='user',
            content_type='text',
            text_content='Entry for criteria retry',
        )

        backend = context_repo.backend
        original_execute_write = backend.execute_write

        async def double_execute_write(fn, *args, **kwargs):
            # Simulate a transparent retry: invoke the same closure twice.
            first = await original_execute_write(fn, *args, **kwargs)
            await original_execute_write(fn, *args, **kwargs)
            return first

        with patch.object(backend, 'execute_write', side_effect=double_execute_write):
            _, criteria = await context_repo.delete_contexts_batch(context_ids=[ctx_id])

        # Exactly one criteria entry despite the closure running twice.
        assert criteria == ['context_ids: 1 IDs']

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
        matching_ids: list[str] = []
        for i in range(2):
            ctx_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='crit-chunk-a',
                source='user',
                content_type='text',
                text_content=f'Criteria chunk match {i}',
            )
            matching_ids.append(ctx_id)
        agent_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='crit-chunk-a',
            source='agent',
            content_type='text',
            text_content='Criteria chunk agent entry',
        )
        other_thread_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='crit-chunk-b',
            source='user',
            content_type='text',
            text_content='Criteria chunk other-thread entry',
        )

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
        )

        assert sorted(matched) == sorted(matching_ids)

    @pytest.mark.asyncio
    async def test_delete_contexts_batch_spans_chunks_and_semantics(
        self, repos: RepositoryContainer,
    ) -> None:
        """The criteria delete chunks an oversized context_ids list and keeps AND semantics.

        Mirrors the snapshot test on the destructive leg: one DELETE per chunk pair,
        all within the closure's single write, deleting exactly the rows the
        AND-combined criteria match and summing the per-statement rowcounts.
        """
        user_ids: list[str] = []
        for i in range(2):
            ctx_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='del-chunk-thread',
                source='user',
                content_type='text',
                text_content=f'Delete chunk match {i}',
            )
            user_ids.append(ctx_id)
        agent_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='del-chunk-thread',
            source='agent',
            content_type='text',
            text_content='Delete chunk agent survivor',
        )
        unlisted_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='del-chunk-thread',
            source='user',
            content_type='text',
            text_content='Delete chunk unlisted survivor',
        )

        context_ids = [generate_id() for _ in range(33000)]
        for pos, real_id in zip((0, 900, 32999), (*user_ids, agent_id), strict=True):
            context_ids[pos] = real_id

        deleted_count, criteria = await repos.context.delete_contexts_batch(
            context_ids=context_ids,
            source='user',
        )

        assert deleted_count == 2
        assert criteria == ['context_ids: 33000 IDs', 'source: user']
        # The listed agent entry (source mismatch) and the unlisted user entry survive.
        remaining = await repos.context.get_by_ids([*user_ids, agent_id, unlisted_id], scope=LOCAL_SCOPE)
        assert {row['id'] for row in remaining} == {agent_id, unlisted_id}

    @pytest.mark.asyncio
    async def test_delete_contexts_batch_chunks_postgresql_statements(self) -> None:
        """The PostgreSQL criteria delete issues one bounded statement per chunk pair.

        Asserted against a recording stand-in connection (no live PostgreSQL):
        1,300 ids plus a source filter must produce two DELETE statements of 901
        and 401 bind parameters (900-id and 400-id chunks, each AND-combined with
        source), with the reported count summing the per-statement results.
        execute_write wraps the closure in one transaction on the real backend,
        so the per-chunk statements stay atomic.
        """
        executed: list[tuple[str, int]] = []

        class _RecordingConn:
            async def execute(self, query: str, *params: object) -> str:
                executed.append((query, len(params)))
                return 'DELETE 1'

        pg_backend = Mock()
        pg_backend.backend_type = 'postgresql'

        async def _execute_write(
            closure: Callable[[object], Awaitable[tuple[int, list[str]]]],
            *,
            validate_connection: bool = False,
        ) -> tuple[int, list[str]]:
            assert validate_connection is True
            return await closure(_RecordingConn())

        pg_backend.execute_write = _execute_write
        repo_pg = ContextRepository(cast(StorageBackend, pg_backend))

        context_ids = [generate_id() for _ in range(1300)]
        deleted_count, criteria = await repo_pg.delete_contexts_batch(
            context_ids=context_ids,
            source='user',
        )

        assert [param_count for _query, param_count in executed] == [901, 401]
        assert all('id IN (' in query and 'source = ' in query for query, _param_count in executed)
        assert deleted_count == 2
        assert criteria == ['context_ids: 1300 IDs', 'source: user']
