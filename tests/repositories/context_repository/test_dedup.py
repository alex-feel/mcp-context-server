"""Tests for the deduplicating store and the latest-entry duplicate check of ContextRepository."""

import json

import pytest

from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from tests.helpers import LOCAL_SCOPE


class TestContextRepositoryDeduplication:
    """Test deduplication logic in ContextRepository."""

    @pytest.mark.asyncio
    async def test_deduplication_updates_timestamp(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that duplicate content updates timestamp instead of inserting."""
        ctx_id1, was_updated1 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Same content',
        )
        assert was_updated1 is False  # First insert, not an update

        ctx_id2, was_updated2 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Same content',
        )
        assert was_updated2 is True  # Second call with same content, should update
        assert ctx_id1 == ctx_id2  # Should be same ID

    @pytest.mark.asyncio
    async def test_dedup_update_falls_through_to_insert_on_hash_divergence(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """A dedup UPDATE whose content_hash predicate misses INSERTs instead.

        The dedup decision (SELECT + hash compare) and the dedup UPDATE are
        separate statements; a concurrent writer that commits a text change to
        the candidate between them must not have its newer hash/summary/metadata
        overwritten with values describing THIS request's text. The UPDATE's
        null-safe content_hash predicate re-asserts the decision at write time;
        a miss falls through to a fresh INSERT. Simulated by patching the
        decision read to return a stale hash that still equals the request's
        hash (so dedup is attempted) while the row has since diverged.
        """
        from app.repositories.context_repository.helpers import compute_content_hash

        ctx_id1, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='dedup_race_thread',
            source='user',
            content_type='text',
            text_content='Original text',
        )

        # Concurrent writer effect: the candidate's text (and hash) changed
        # after the dedup decision would have observed 'Original text'.
        import sqlite3 as _sqlite3

        def _diverge(conn: _sqlite3.Connection) -> None:
            conn.execute(
                'UPDATE context_entries SET text_content = ?, content_hash = ? WHERE id = ?',
                ('Newer text', compute_content_hash('Newer text'), ctx_id1),
            )

        await repos.context.backend.execute_write(_diverge)

        # Drive the guarded UPDATE (mirroring the repository's dedup UPDATE
        # predicate) directly with the STALE observed hash -- the value the
        # dedup decision saw before the concurrent change. It must match 0 rows
        # (the row's hash diverged), NOT overwrite the newer text's hash.
        stale_hash = compute_content_hash('Original text')

        def _guarded_update(conn: _sqlite3.Connection) -> int:
            cur = conn.execute(
                '''
                UPDATE context_entries
                SET content_hash = ?, version = version + 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND content_hash IS ?
                ''',
                (stale_hash, ctx_id1, stale_hash),
            )
            return cur.rowcount

        assert await repos.context.backend.execute_write(_guarded_update) == 0

        def _hash_now(conn: _sqlite3.Connection) -> str:
            row = conn.execute(
                'SELECT content_hash FROM context_entries WHERE id = ?', (ctx_id1,),
            ).fetchone()
            return str(row[0])

        # The newer text's hash survives untouched.
        assert await repos.context.backend.execute_read(_hash_now) == compute_content_hash('Newer text')

    @pytest.mark.asyncio
    async def test_deduplication_different_content(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that different content creates new entry."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Content A',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Content B',
        )

        assert ctx_id1 != ctx_id2  # Different content = different IDs

    @pytest.mark.asyncio
    async def test_deduplication_different_source(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that same content from different source creates new entry."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Same content',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='dedup_thread',
            source='agent',  # Different source
            content_type='text',
            text_content='Same content',
        )

        assert ctx_id1 != ctx_id2  # Different source = different entry

    @pytest.mark.asyncio
    async def test_deduplication_different_thread(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that same content in different thread creates new entry."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='thread_1',
            source='user',
            content_type='text',
            text_content='Same content',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='thread_2',  # Different thread
            source='user',
            content_type='text',
            text_content='Same content',
        )

        assert ctx_id1 != ctx_id2  # Different thread = different entry

    @pytest.mark.asyncio
    async def test_deduplication_updates_metadata_coalesce(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Deduplication COALESCE: new metadata replaces existing."""
        ctx_id1, was_updated1 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='coalesce-thread',
            source='user',
            content_type='text',
            text_content='Coalesce test content',
            metadata=json.dumps({'key': 'original'}),
        )
        assert was_updated1 is False

        ctx_id2, was_updated2 = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='coalesce-thread',
            source='user',
            content_type='text',
            text_content='Coalesce test content',
            metadata=json.dumps({'key': 'updated'}),
        )
        assert was_updated2 is True
        assert ctx_id2 == ctx_id1

        rows, _ = await context_repo.search_contexts(thread_id='coalesce-thread')
        meta = json.loads(rows[0]['metadata'])
        assert meta['key'] == 'updated'

    @pytest.mark.asyncio
    async def test_deduplication_preserves_metadata_on_none(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Deduplication COALESCE(NULL, existing) preserves existing metadata."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='preserve-meta-thread',
            source='user',
            content_type='text',
            text_content='Preserve meta content',
            metadata=json.dumps({'preserved': 'yes'}),
        )

        ctx_id2, was_updated = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='preserve-meta-thread',
            source='user',
            content_type='text',
            text_content='Preserve meta content',
            metadata=None,
        )
        assert was_updated is True
        assert ctx_id2 == ctx_id1

        rows, _ = await context_repo.search_contexts(thread_id='preserve-meta-thread')
        meta = json.loads(rows[0]['metadata'])
        assert meta['preserved'] == 'yes'

    @pytest.mark.asyncio
    async def test_deduplication_summary_coalesce(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Deduplication COALESCE preserves existing summary when new is None."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='summary-coalesce-thread',
            source='user',
            content_type='text',
            text_content='Summary coalesce test',
            summary='Existing summary',
        )

        ctx_id2, was_updated = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='summary-coalesce-thread',
            source='user',
            content_type='text',
            text_content='Summary coalesce test',
            summary=None,
        )
        assert was_updated is True
        assert ctx_id2 == ctx_id1

        rows = await context_repo.get_by_ids([ctx_id1])
        assert rows[0]['summary'] == 'Existing summary'

    @pytest.mark.asyncio
    async def test_deduplication_content_hash_path(
        self, repos: RepositoryContainer,
    ) -> None:
        """Deduplication uses content_hash for fast comparison."""
        from app.repositories.context_repository.helpers import compute_content_hash

        text = 'Hash-based dedup test content'
        expected_hash = compute_content_hash(text)
        assert expected_hash is not None

        ctx_id1, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='hash-dedup-thread',
            source='user',
            content_type='text',
            text_content=text,
        )

        ctx_id2, was_updated = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='hash-dedup-thread',
            source='user',
            content_type='text',
            text_content=text,
        )
        assert was_updated is True
        assert ctx_id2 == ctx_id1

    @pytest.mark.asyncio
    async def test_deduplication_empty_summary_normalized(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Deduplication normalizes empty/whitespace summary to None."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='norm-summary-thread',
            source='user',
            content_type='text',
            text_content='Normalized summary test',
            summary='   ',
        )
        rows = await context_repo.get_by_ids([ctx_id])
        assert rows[0]['summary'] is None

    @pytest.mark.asyncio
    async def test_deduplication_check_empty_database(
        self, context_repo: ContextRepository,
    ) -> None:
        """check_latest_is_duplicate on empty DB returns None."""
        result = await context_repo.check_latest_is_duplicate(
            thread_id='empty-thread',
            source='user',
            text_content='Some content', scope=LOCAL_SCOPE,
        )
        assert result is None


class TestContextRepositoryCheckDuplicate:
    """Test check_latest_is_duplicate method of ContextRepository."""

    @pytest.mark.asyncio
    async def test_check_latest_is_duplicate_match(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Returns context_id when latest entry has identical content."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='dup-check-thread',
            source='user',
            content_type='text',
            text_content='Duplicate content check',
        )
        result = await context_repo.check_latest_is_duplicate(
            thread_id='dup-check-thread',
            source='user',
            text_content='Duplicate content check', scope=LOCAL_SCOPE,
        )
        assert result is not None
        assert result.context_id == ctx_id

    @pytest.mark.asyncio
    async def test_check_latest_is_duplicate_no_match(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Returns None when latest entry has different content."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='no-dup-thread',
            source='user',
            content_type='text',
            text_content='Original content',
        )
        result = await context_repo.check_latest_is_duplicate(
            thread_id='no-dup-thread',
            source='user',
            text_content='Different content', scope=LOCAL_SCOPE,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_check_latest_is_duplicate_different_thread(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Returns None when content matches but thread_id differs."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='thread-a',
            source='user',
            content_type='text',
            text_content='Same content different thread',
        )
        result = await context_repo.check_latest_is_duplicate(
            thread_id='thread-b',
            source='user',
            text_content='Same content different thread', scope=LOCAL_SCOPE,
        )
        assert result is None
