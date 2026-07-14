"""Extended tests for the context repository.

This module provides additional tests for ContextRepository to improve coverage
of search operations, edge cases, and error handling.
"""

import json
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import pytest_asyncio

from app.backends import create_backend
from app.backends.base import StorageBackend
from app.backends.base import TransactionContext
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from app.schemas import load_schema


@pytest_asyncio.fixture
async def context_test_db(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Create a test database for context repository testing."""
    db_path = tmp_path / 'context_test.db'

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    schema_sql = load_schema('sqlite')

    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(schema_sql)

    await backend.execute_write(_init_schema)

    yield backend

    await backend.shutdown()


@pytest_asyncio.fixture
async def context_repo(context_test_db: StorageBackend) -> ContextRepository:
    """Create a context repository for testing."""
    return ContextRepository(context_test_db)


@pytest_asyncio.fixture
async def repos(context_test_db: StorageBackend) -> RepositoryContainer:
    """Create full repository container."""
    return RepositoryContainer(context_test_db)


class TestContextRepositorySearch:
    """Test search functionality of ContextRepository."""

    @pytest.mark.asyncio
    async def test_search_empty_database(self, context_repo: ContextRepository) -> None:
        """Test searching empty database returns empty results."""
        rows, stats = await context_repo.search_contexts()

        assert rows == []
        assert 'execution_time_ms' in stats

    @pytest.mark.asyncio
    async def test_search_by_thread_id(
        self,
        context_repo: ContextRepository,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching by thread_id."""
        await repos.context.store_with_deduplication(
            thread_id='thread_a',
            source='user',
            content_type='text',
            text_content='Message A',
        )
        await repos.context.store_with_deduplication(
            thread_id='thread_b',
            source='user',
            content_type='text',
            text_content='Message B',
        )

        rows, stats = await context_repo.search_contexts(thread_id='thread_a')

        assert len(rows) == 1
        assert rows[0]['thread_id'] == 'thread_a'

    @pytest.mark.asyncio
    async def test_search_by_source(
        self,
        context_repo: ContextRepository,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching by source."""
        await repos.context.store_with_deduplication(
            thread_id='source_thread',
            source='user',
            content_type='text',
            text_content='User message',
        )
        await repos.context.store_with_deduplication(
            thread_id='source_thread',
            source='agent',
            content_type='text',
            text_content='Agent message',
        )

        rows, stats = await context_repo.search_contexts(source='agent')

        assert len(rows) == 1
        assert rows[0]['source'] == 'agent'

    @pytest.mark.asyncio
    async def test_search_by_content_type(
        self,
        context_repo: ContextRepository,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching by content_type."""
        await repos.context.store_with_deduplication(
            thread_id='type_thread',
            source='user',
            content_type='text',
            text_content='Text only',
        )
        await repos.context.store_with_deduplication(
            thread_id='type_thread',
            source='user',
            content_type='multimodal',
            text_content='With image',
        )

        rows, stats = await context_repo.search_contexts(content_type='multimodal')

        assert len(rows) == 1
        assert rows[0]['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    async def test_search_by_tags(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching by tags."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            thread_id='tag_thread',
            source='user',
            content_type='text',
            text_content='Tagged 1',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            thread_id='tag_thread',
            source='user',
            content_type='text',
            text_content='Tagged 2',
        )

        await repos.tags.store_tags(ctx_id1, ['important', 'review'])
        await repos.tags.store_tags(ctx_id2, ['other'])

        rows, stats = await repos.context.search_contexts(tags=['important'])

        assert len(rows) == 1
        assert rows[0]['id'] == ctx_id1

    @pytest.mark.asyncio
    async def test_search_with_limit(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with limit parameter."""
        for i in range(10):
            await repos.context.store_with_deduplication(
                thread_id='limit_thread',
                source='user',
                content_type='text',
                text_content=f'Message {i}',
            )

        rows, stats = await repos.context.search_contexts(
            thread_id='limit_thread',
            limit=5,
        )

        assert len(rows) == 5

    @pytest.mark.asyncio
    async def test_search_with_offset(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with offset parameter."""
        for i in range(10):
            await repos.context.store_with_deduplication(
                thread_id='offset_thread',
                source='user',
                content_type='text',
                text_content=f'Message {i}',
            )

        rows, stats = await repos.context.search_contexts(
            thread_id='offset_thread',
            limit=3,
            offset=5,
        )

        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_search_with_metadata_simple(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with simple metadata filter."""
        await repos.context.store_with_deduplication(
            thread_id='meta_thread',
            source='user',
            content_type='text',
            text_content='Priority 1',
            metadata=json.dumps({'priority': 1}),
        )
        await repos.context.store_with_deduplication(
            thread_id='meta_thread',
            source='user',
            content_type='text',
            text_content='Priority 2',
            metadata=json.dumps({'priority': 2}),
        )

        rows, stats = await repos.context.search_contexts(
            thread_id='meta_thread',
            metadata={'priority': 1},
        )

        assert len(rows) == 1
        # Parse metadata JSON and verify
        metadata = json.loads(rows[0]['metadata'])
        assert metadata['priority'] == 1

    @pytest.mark.asyncio
    async def test_search_with_explain_query(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with explain_query=True."""
        await repos.context.store_with_deduplication(
            thread_id='explain_thread',
            source='user',
            content_type='text',
            text_content='Test entry',
        )

        rows, stats = await repos.context.search_contexts(
            thread_id='explain_thread',
            explain_query=True,
        )

        assert len(rows) == 1
        assert 'execution_time_ms' in stats
        assert 'filters_applied' in stats

    @pytest.mark.asyncio
    async def test_search_combined_filters(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with multiple filters combined."""
        await repos.context.store_with_deduplication(
            thread_id='combo_thread',
            source='user',
            content_type='text',
            text_content='User text',
        )
        await repos.context.store_with_deduplication(
            thread_id='combo_thread',
            source='agent',
            content_type='text',
            text_content='Agent text',
        )
        await repos.context.store_with_deduplication(
            thread_id='other_thread',
            source='user',
            content_type='text',
            text_content='Other user',
        )

        rows, stats = await repos.context.search_contexts(
            thread_id='combo_thread',
            source='user',
            content_type='text',
        )

        assert len(rows) == 1
        assert rows[0]['thread_id'] == 'combo_thread'
        assert rows[0]['source'] == 'user'

    @pytest.mark.asyncio
    async def test_search_validation_error_stats_include_backend_sqlite(
        self,
        context_repo: ContextRepository,
    ) -> None:
        """The validation-error stats dict exposes the same 'backend' key as success stats.

        An invalid metadata filter short-circuits search_contexts into a structured
        validation-error stats dict; that dict must mirror the success-path stats
        shape (which always includes 'backend') so clients can rely on a uniform
        key set regardless of outcome.
        """
        rows, stats = await context_repo.search_contexts(
            metadata_filters=[{'key': 'status', 'operator': 'not_a_real_operator', 'value': 'x'}],
        )

        assert rows == []
        assert stats['error'] == 'Metadata filter validation failed'
        assert stats['validation_errors']
        assert stats['execution_time_ms'] == 0.0
        assert stats['filters_applied'] == 0
        assert stats['rows_returned'] == 0
        assert stats['backend'] == 'sqlite'

    @pytest.mark.asyncio
    async def test_search_validation_error_stats_include_backend_postgresql(self) -> None:
        """The PostgreSQL validation-error stats dict also carries 'backend'.

        The PostgreSQL closure short-circuits on validation errors before touching
        the connection, so a recording stand-in connection suffices -- no live
        PostgreSQL needed.
        """
        pg_backend = Mock()
        pg_backend.backend_type = 'postgresql'

        async def _execute_read(
            closure: Callable[[object], Awaitable[tuple[list[object], dict[str, object]]]],
        ) -> tuple[list[object], dict[str, object]]:
            return await closure(object())

        pg_backend.execute_read = _execute_read
        repo_pg = ContextRepository(cast(StorageBackend, pg_backend))

        rows, stats = await repo_pg.search_contexts(
            metadata_filters=[{'key': 'status', 'operator': 'not_a_real_operator', 'value': 'x'}],
        )

        assert rows == []
        assert stats['error'] == 'Metadata filter validation failed'
        assert stats['backend'] == 'postgresql'


class TestContextRepositoryDeduplication:
    """Test deduplication logic in ContextRepository."""

    @pytest.mark.asyncio
    async def test_deduplication_updates_timestamp(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test that duplicate content updates timestamp instead of inserting."""
        ctx_id1, was_updated1 = await repos.context.store_with_deduplication(
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Same content',
        )
        assert was_updated1 is False  # First insert, not an update

        ctx_id2, was_updated2 = await repos.context.store_with_deduplication(
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
        from app.repositories.context_repository import compute_content_hash

        ctx_id1, _ = await repos.context.store_with_deduplication(
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
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Content A',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
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
            thread_id='dedup_thread',
            source='user',
            content_type='text',
            text_content='Same content',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
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
            thread_id='thread_1',
            source='user',
            content_type='text',
            text_content='Same content',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
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
            thread_id='coalesce-thread',
            source='user',
            content_type='text',
            text_content='Coalesce test content',
            metadata=json.dumps({'key': 'original'}),
        )
        assert was_updated1 is False

        ctx_id2, was_updated2 = await repos.context.store_with_deduplication(
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
            thread_id='preserve-meta-thread',
            source='user',
            content_type='text',
            text_content='Preserve meta content',
            metadata=json.dumps({'preserved': 'yes'}),
        )

        ctx_id2, was_updated = await repos.context.store_with_deduplication(
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
            thread_id='summary-coalesce-thread',
            source='user',
            content_type='text',
            text_content='Summary coalesce test',
            summary='Existing summary',
        )

        ctx_id2, was_updated = await repos.context.store_with_deduplication(
            thread_id='summary-coalesce-thread',
            source='user',
            content_type='text',
            text_content='Summary coalesce test',
            summary=None,
        )
        assert was_updated is True
        assert ctx_id2 == ctx_id1

        result = await context_repo.get_summary(ctx_id1)
        assert result == 'Existing summary'

    @pytest.mark.asyncio
    async def test_deduplication_content_hash_path(
        self, repos: RepositoryContainer,
    ) -> None:
        """Deduplication uses content_hash for fast comparison."""
        from app.repositories.context_repository import compute_content_hash

        text = 'Hash-based dedup test content'
        expected_hash = compute_content_hash(text)
        assert expected_hash is not None

        ctx_id1, _ = await repos.context.store_with_deduplication(
            thread_id='hash-dedup-thread',
            source='user',
            content_type='text',
            text_content=text,
        )

        ctx_id2, was_updated = await repos.context.store_with_deduplication(
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
            thread_id='norm-summary-thread',
            source='user',
            content_type='text',
            text_content='Normalized summary test',
            summary='   ',
        )
        result = await context_repo.get_summary(ctx_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_deduplication_check_empty_database(
        self, context_repo: ContextRepository,
    ) -> None:
        """check_latest_is_duplicate on empty DB returns None."""
        result = await context_repo.check_latest_is_duplicate(
            thread_id='empty-thread',
            source='user',
            text_content='Some content',
        )
        assert result is None


class TestContextRepositoryDelete:
    """Test delete operations in ContextRepository."""

    @pytest.mark.asyncio
    async def test_delete_by_thread_id(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test deleting entries by thread_id."""
        await repos.context.store_with_deduplication(
            thread_id='del_thread',
            source='user',
            content_type='text',
            text_content='To delete',
        )
        await repos.context.store_with_deduplication(
            thread_id='keep_thread',
            source='user',
            content_type='text',
            text_content='To keep',
        )

        deleted = await repos.context.delete_by_thread(thread_id='del_thread')

        assert deleted == 1

        # Verify deletion
        rows, _ = await repos.context.search_contexts(thread_id='del_thread')
        assert len(rows) == 0

        # Verify other thread kept
        rows, _ = await repos.context.search_contexts(thread_id='keep_thread')
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_delete_multiple_entries(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test deleting multiple entries from same thread."""
        await repos.context.store_with_deduplication(
            thread_id='multi_del_thread',
            source='user',
            content_type='text',
            text_content='Message 1',
        )
        await repos.context.store_with_deduplication(
            thread_id='multi_del_thread',
            source='agent',
            content_type='text',
            text_content='Message 2',
        )
        await repos.context.store_with_deduplication(
            thread_id='multi_del_thread',
            source='user',
            content_type='text',
            text_content='Message 3',
        )

        deleted = await repos.context.delete_by_thread(thread_id='multi_del_thread')

        assert deleted == 3

        # Verify all deleted
        rows, _ = await repos.context.search_contexts(thread_id='multi_del_thread')
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
            thread_id='chunk_keep_thread',
            source='user',
            content_type='text',
            text_content='Keep me',
        )

        real_ids: list[str] = []
        for i in range(5):
            ctx_id, _ = await repos.context.store_with_deduplication(
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
        assert await repos.context.get_by_ids(real_ids) == []

        # The unrelated entry is untouched.
        remaining = await repos.context.get_by_ids([keep_id])
        assert len(remaining) == 1
        assert remaining[0]['id'] == keep_id


class TestContextRepositoryGetById:
    """Test get_by_ids operations."""

    @pytest.mark.asyncio
    async def test_get_by_ids_single(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting single entry by ID."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='get_thread',
            source='user',
            content_type='text',
            text_content='Test entry',
        )

        rows = await repos.context.get_by_ids([ctx_id])

        assert len(rows) == 1
        assert rows[0]['id'] == ctx_id

    @pytest.mark.asyncio
    async def test_get_by_ids_multiple(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting multiple entries by IDs."""
        ids = []
        for i in range(3):
            ctx_id, _ = await repos.context.store_with_deduplication(
                thread_id='multi_get',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
            )
            ids.append(ctx_id)

        rows = await repos.context.get_by_ids(ids)

        assert len(rows) == 3
        returned_ids = {r['id'] for r in rows}
        assert returned_ids == set(ids)

    @pytest.mark.asyncio
    async def test_get_by_ids_empty_list(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting entries with empty ID list."""
        rows = await repos.context.get_by_ids([])

        assert rows == []

    @pytest.mark.asyncio
    async def test_get_by_ids_nonexistent(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting nonexistent IDs returns empty."""
        rows = await repos.context.get_by_ids([generate_id(), generate_id()])

        assert rows == []

    @pytest.mark.asyncio
    async def test_get_by_ids_partial_match(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting mix of existing and nonexistent IDs."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='partial_get',
            source='user',
            content_type='text',
            text_content='Exists',
        )

        rows = await repos.context.get_by_ids([ctx_id, generate_id()])

        assert len(rows) == 1
        assert rows[0]['id'] == ctx_id

    @pytest.mark.asyncio
    async def test_get_by_ids_spans_multiple_chunks_preserves_order(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """get_by_ids chunks an oversized id list and preserves the ordering contract.

        A 33,000-id list bound as one IN clause exceeds SQLite's default 32,766
        per-statement variable ceiling ("too many SQL variables"), so the fetch is
        issued in bounded chunks like delete_by_ids. Real ids are planted on both
        sides of the 900-id chunk boundary with the OLDEST rows in the FIRST chunk
        and the NEWEST row in the LAST chunk, so plain per-chunk concatenation
        would invert the order -- the assertion therefore also proves the
        accumulated rows are re-sorted into the single-statement
        ORDER BY created_at DESC, id DESC contract.
        """
        real_ids: list[str] = []
        for i in range(5):
            ctx_id, _ = await repos.context.store_with_deduplication(
                thread_id='chunk_get_thread',
                source='user',
                content_type='text',
                text_content=f'Chunk get entry {i}',
            )
            real_ids.append(ctx_id)

        # Expected order = the documented contract, computed from per-row reads
        # (each a single-chunk fetch): created_at DESC, id DESC.
        keyed: list[tuple[str, str]] = []
        for ctx_id in real_ids:
            row = (await repos.context.get_by_ids([ctx_id]))[0]
            keyed.append((row['created_at'], row['id']))
        expected_ids = [entry_id for _created_at, entry_id in sorted(keyed, reverse=True)]

        # Non-existent ids pad the list past the statement variable ceiling; the
        # oldest real rows go into the first chunk and the newest into the last.
        mixed = [generate_id() for _ in range(33000)]
        oldest_first = list(reversed(expected_ids))
        for pos, real_id in zip((0, 450, 899, 900, 32999), oldest_first, strict=True):
            mixed[pos] = real_id

        rows = await repos.context.get_by_ids(mixed)

        assert [row['id'] for row in rows] == expected_ids


class TestContextRepositoryUpdate:
    """Test update operations in ContextRepository."""

    @pytest.mark.asyncio
    async def test_check_entry_exists(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test checking if entry exists."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='exists_thread',
            source='user',
            content_type='text',
            text_content='Exists',
        )

        exists, source, version = await repos.context.check_entry_exists(ctx_id)
        assert exists is True
        assert source == 'user'
        assert isinstance(version, int)
        assert version == 0

        not_exists, no_source, no_version = await repos.context.check_entry_exists(generate_id())
        assert not_exists is False
        assert no_source is None
        assert no_version is None

    @pytest.mark.asyncio
    async def test_entry_exists(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """entry_exists returns True for a stored id and False for an absent one."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='entry_exists_thread',
            source='user',
            content_type='text',
            text_content='Exists',
        )

        assert await repos.context.entry_exists(ctx_id) is True
        assert await repos.context.entry_exists(generate_id()) is False

    @pytest.mark.asyncio
    async def test_entry_exists_locks_parent_row_on_postgresql_transaction(self) -> None:
        """On PostgreSQL the in-transaction presence check locks the parent row.

        The tags-only / images-only update guard runs entry_exists on the open
        transaction connection; it must emit FOR KEY SHARE so a concurrent DELETE
        blocks until commit and cannot leave the child tag/image writes violating
        the foreign key. Outside a transaction the lock would release at statement
        end, so it must NOT be emitted there. Both cases are asserted against a
        recording connection without needing a live PostgreSQL.
        """
        txn_conn = AsyncMock()
        txn_conn.fetchrow = AsyncMock(return_value={'?column?': 1})
        txn_backend = Mock()
        txn_backend.backend_type = 'postgresql'
        txn = Mock()
        txn.backend_type = 'postgresql'
        txn.connection = txn_conn

        repo_txn = ContextRepository(cast(StorageBackend, txn_backend))
        assert await repo_txn.entry_exists('abc123', txn=cast(TransactionContext, txn)) is True
        assert 'FOR KEY SHARE' in txn_conn.fetchrow.call_args.args[0]

        pool_conn = AsyncMock()
        pool_conn.fetchrow = AsyncMock(return_value={'?column?': 1})

        async def _execute_read(closure: Callable[[object], Awaitable[bool]]) -> bool:
            return await closure(pool_conn)

        pool_backend = Mock()
        pool_backend.backend_type = 'postgresql'
        pool_backend.execute_read = _execute_read

        repo_pool = ContextRepository(cast(StorageBackend, pool_backend))
        assert await repo_pool.entry_exists('abc123') is True
        assert 'FOR KEY SHARE' not in pool_conn.fetchrow.call_args.args[0]

    @pytest.mark.asyncio
    async def test_get_content_type(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting content type by ID."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='type_thread',
            source='user',
            content_type='text',
            text_content='Text content',
        )

        content_type = await repos.context.get_content_type(ctx_id)

        assert content_type == 'text'

    @pytest.mark.asyncio
    async def test_get_content_type_nonexistent(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting content type for nonexistent entry."""
        content_type = await repos.context.get_content_type(generate_id())

        assert content_type is None

    @pytest.mark.asyncio
    async def test_update_content_type(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test updating content type."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='update_type',
            source='user',
            content_type='text',
            text_content='Content',
        )

        await repos.context.update_content_type(ctx_id, 'multimodal')

        new_type = await repos.context.get_content_type(ctx_id)
        assert new_type == 'multimodal'


class TestContextRepositoryGetSummary:
    """Test get_summary method of ContextRepository."""

    @pytest.mark.asyncio
    async def test_get_summary_existing_entry(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """get_summary returns summary string for entry with summary."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='summary-thread',
            source='user',
            content_type='text',
            text_content='Entry with summary',
            summary='This is a test summary.',
        )
        result = await context_repo.get_summary(ctx_id)
        assert result == 'This is a test summary.'

    @pytest.mark.asyncio
    async def test_get_summary_no_summary(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """get_summary returns None for entry without summary."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='no-summary-thread',
            source='user',
            content_type='text',
            text_content='Entry without summary',
        )
        result = await context_repo.get_summary(ctx_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_summary_empty_string_normalized_to_none(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """get_summary normalizes empty/whitespace-only summary to None."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='empty-summary-thread',
            source='user',
            content_type='text',
            text_content='Entry with empty summary',
            summary='   ',
        )
        result = await context_repo.get_summary(ctx_id)
        assert result is None


class TestContextRepositoryCheckDuplicate:
    """Test check_latest_is_duplicate method of ContextRepository."""

    @pytest.mark.asyncio
    async def test_check_latest_is_duplicate_match(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Returns context_id when latest entry has identical content."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='dup-check-thread',
            source='user',
            content_type='text',
            text_content='Duplicate content check',
        )
        result = await context_repo.check_latest_is_duplicate(
            thread_id='dup-check-thread',
            source='user',
            text_content='Duplicate content check',
        )
        assert result is not None
        assert result.context_id == ctx_id

    @pytest.mark.asyncio
    async def test_check_latest_is_duplicate_no_match(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Returns None when latest entry has different content."""
        await repos.context.store_with_deduplication(
            thread_id='no-dup-thread',
            source='user',
            content_type='text',
            text_content='Original content',
        )
        result = await context_repo.check_latest_is_duplicate(
            thread_id='no-dup-thread',
            source='user',
            text_content='Different content',
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_check_latest_is_duplicate_different_thread(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Returns None when content matches but thread_id differs."""
        await repos.context.store_with_deduplication(
            thread_id='thread-a',
            source='user',
            content_type='text',
            text_content='Same content different thread',
        )
        result = await context_repo.check_latest_is_duplicate(
            thread_id='thread-b',
            source='user',
            text_content='Same content different thread',
        )
        assert result is None


class TestContextRepositoryPatchMetadata:
    """Test patch_metadata method of ContextRepository (RFC 7396)."""

    @pytest.mark.asyncio
    async def test_patch_metadata_adds_new_key(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Patching adds a new key to existing metadata."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='patch-add-thread',
            source='user',
            content_type='text',
            text_content='Patch test entry',
            metadata=json.dumps({'existing': 'value'}),
        )
        success, fields = await context_repo.patch_metadata(ctx_id, {'new_key': 'new_value'})
        assert success is True
        assert 'metadata' in fields

        rows, _ = await context_repo.search_contexts(thread_id='patch-add-thread')
        assert len(rows) == 1
        meta = json.loads(rows[0]['metadata'])
        assert meta['existing'] == 'value'
        assert meta['new_key'] == 'new_value'

    @pytest.mark.asyncio
    async def test_patch_metadata_updates_existing_key(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Patching updates an existing key's value."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='patch-update-thread',
            source='user',
            content_type='text',
            text_content='Patch update test',
            metadata=json.dumps({'status': 'pending'}),
        )
        success, fields = await context_repo.patch_metadata(ctx_id, {'status': 'done'})
        assert success is True

        rows, _ = await context_repo.search_contexts(thread_id='patch-update-thread')
        meta = json.loads(rows[0]['metadata'])
        assert meta['status'] == 'done'

    @pytest.mark.asyncio
    async def test_patch_metadata_deletes_key_with_null(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Patching with null value deletes the key (RFC 7396)."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='patch-delete-thread',
            source='user',
            content_type='text',
            text_content='Patch delete test',
            metadata=json.dumps({'keep': 'yes', 'remove': 'me'}),
        )
        success, _ = await context_repo.patch_metadata(ctx_id, {'remove': None})
        assert success is True

        rows, _ = await context_repo.search_contexts(thread_id='patch-delete-thread')
        meta = json.loads(rows[0]['metadata'])
        assert 'keep' in meta
        assert 'remove' not in meta

    @pytest.mark.asyncio
    async def test_patch_metadata_nonexistent_entry(
        self, context_repo: ContextRepository,
    ) -> None:
        """Patching nonexistent entry returns (False, [])."""
        success, fields = await context_repo.patch_metadata(generate_id(), {'key': 'value'})
        assert success is False
        assert fields == []

    @pytest.mark.asyncio
    async def test_patch_metadata_empty_patch(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Empty patch is a no-op for data but updates timestamp."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='patch-empty-thread',
            source='user',
            content_type='text',
            text_content='Patch empty test',
            metadata=json.dumps({'unchanged': 'value'}),
        )
        success, fields = await context_repo.patch_metadata(ctx_id, {})
        assert success is True
        assert 'metadata' in fields

        rows, _ = await context_repo.search_contexts(thread_id='patch-empty-thread')
        meta = json.loads(rows[0]['metadata'])
        assert meta['unchanged'] == 'value'


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
                thread_id='batch-del-thread',
                source='user',
                content_type='text',
                text_content=f'Batch delete entry {i}',
            )
            ids.append(ctx_id)

        deleted_count, criteria = await context_repo.delete_contexts_batch(context_ids=ids)
        assert deleted_count == 3

        rows = await context_repo.get_by_ids(ids)
        assert rows == []

    @pytest.mark.asyncio
    async def test_delete_contexts_batch_partial_ids(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Batch delete with mix of existing and nonexistent IDs."""
        ctx_id, _ = await repos.context.store_with_deduplication(
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
                thread_id='crit-chunk-a',
                source='user',
                content_type='text',
                text_content=f'Criteria chunk match {i}',
            )
            matching_ids.append(ctx_id)
        agent_id, _ = await repos.context.store_with_deduplication(
            thread_id='crit-chunk-a',
            source='agent',
            content_type='text',
            text_content='Criteria chunk agent entry',
        )
        other_thread_id, _ = await repos.context.store_with_deduplication(
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
                thread_id='del-chunk-thread',
                source='user',
                content_type='text',
                text_content=f'Delete chunk match {i}',
            )
            user_ids.append(ctx_id)
        agent_id, _ = await repos.context.store_with_deduplication(
            thread_id='del-chunk-thread',
            source='agent',
            content_type='text',
            text_content='Delete chunk agent survivor',
        )
        unlisted_id, _ = await repos.context.store_with_deduplication(
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
        remaining = await repos.context.get_by_ids([*user_ids, agent_id, unlisted_id])
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


class TestContextRepositoryVersionCAS:
    """Regression tests for the optimistic-concurrency version guard on
    ``update_context_entry`` (the new ``version`` column + ``expected_version``
    compare-and-set landed alongside the store/update latency hardening).
    """

    async def _read_row(
        self, context_repo: ContextRepository, context_id: str,
    ) -> tuple[str, int]:
        """Read (text_content, version) for an entry via a direct SELECT.

        check_entry_exists returns version but NOT text_content, so a direct
        SELECT is the most precise way to assert the row was (or was not) mutated.

        Returns:
            Tuple of (text_content, version) for the row.
        """

        def _select(conn: sqlite3.Connection) -> tuple[str, int]:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT text_content, version FROM context_entries WHERE id = ?',
                (context_id,),
            )
            row = cursor.fetchone()
            assert row is not None
            return str(row['text_content']), int(row['version'])

        return await context_repo.backend.execute_read(_select)

    @pytest.mark.asyncio
    async def test_initial_version_is_zero(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """A freshly inserted entry starts at version 0 (schema DEFAULT 0)."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='cas-init-thread',
            source='user',
            content_type='text',
            text_content='Initial version content',
        )

        exists, _source, version = await repos.context.check_entry_exists(ctx_id)
        assert exists is True
        assert version == 0

        _text, row_version = await self._read_row(context_repo, ctx_id)
        assert row_version == 0

    @pytest.mark.asyncio
    async def test_cas_success_bumps_version(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """update with expected_version=0 succeeds and bumps version to 1."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='cas-success-thread',
            source='user',
            content_type='text',
            text_content='v0',
        )

        success, fields = await repos.context.update_context_entry(
            ctx_id, text_content='v1', expected_version=0,
        )
        assert success is True
        assert 'text_content' in fields

        text, version = await self._read_row(context_repo, ctx_id)
        assert text == 'v1'
        assert version == 1

    @pytest.mark.asyncio
    async def test_stale_expected_version_raises_and_leaves_row_unchanged(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """A second CAS with a now-stale expected_version=0 raises
        VersionConflictError and does NOT mutate the row.
        """
        from app.repositories.context_repository import VersionConflictError

        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='cas-stale-thread',
            source='user',
            content_type='text',
            text_content='v0',
        )

        # First update advances version 0 -> 1.
        await repos.context.update_context_entry(ctx_id, text_content='v1', expected_version=0)

        # Second update reuses the now-stale captured version 0.
        with pytest.raises(VersionConflictError) as exc_info:
            await repos.context.update_context_entry(ctx_id, text_content='v2', expected_version=0)
        assert exc_info.value.context_id == ctx_id

        # Row is untouched: text stays 'v1', version stays 1 (no spurious bump).
        text, version = await self._read_row(context_repo, ctx_id)
        assert text == 'v1'
        assert version == 1

    @pytest.mark.asyncio
    async def test_cas_with_current_version_succeeds(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Re-reading the current version (1) and retrying CAS succeeds, bumping to 2."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='cas-current-thread',
            source='user',
            content_type='text',
            text_content='v0',
        )

        await repos.context.update_context_entry(ctx_id, text_content='v1', expected_version=0)

        success, _fields = await repos.context.update_context_entry(
            ctx_id, text_content='v3', expected_version=1,
        )
        assert success is True

        text, version = await self._read_row(context_repo, ctx_id)
        assert text == 'v3'
        assert version == 2

    @pytest.mark.asyncio
    async def test_legacy_path_no_cas_no_version_bump(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """expected_version=None keeps legacy behavior: update succeeds with NO
        CAS predicate and does NOT bump the version column.
        """
        ctx_id, _ = await repos.context.store_with_deduplication(
            thread_id='cas-legacy-thread',
            source='user',
            content_type='text',
            text_content='v0',
        )

        success, fields = await repos.context.update_context_entry(
            ctx_id, text_content='legacy', expected_version=None,
        )
        assert success is True
        assert 'text_content' in fields

        text, version = await self._read_row(context_repo, ctx_id)
        assert text == 'legacy'
        # Legacy path leaves the version untouched (no SET version = version + 1).
        assert version == 0

    @pytest.mark.asyncio
    async def test_cas_against_nonexistent_id_returns_false(
        self, repos: RepositoryContainer,
    ) -> None:
        """A CAS against a non-existent id returns (False, []), NOT a
        VersionConflictError -- the existence pre-check distinguishes
        "no such row" from "row exists but version moved".
        """
        success, fields = await repos.context.update_context_entry(
            generate_id(), text_content='ghost', expected_version=0,
        )
        assert success is False
        assert fields == []


class TestComputeContentHash:
    """Test the compute_content_hash utility function."""

    def test_compute_content_hash_consistency(self) -> None:
        """Same content produces same hash."""
        from app.repositories.context_repository import compute_content_hash

        hash1 = compute_content_hash('test content')
        hash2 = compute_content_hash('test content')
        assert hash1 == hash2

    def test_compute_content_hash_different_content(self) -> None:
        """Different content produces different hashes."""
        from app.repositories.context_repository import compute_content_hash

        hash1 = compute_content_hash('content A')
        hash2 = compute_content_hash('content B')
        assert hash1 != hash2
