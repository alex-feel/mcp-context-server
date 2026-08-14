"""
Tests for tag repository.

Tests the TagRepository class for storing, retrieving, and managing
tags associated with context entries.
"""


from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
from typing import TypeVar
from typing import cast

import pytest

from app.backends import StorageBackend
from app.ids import generate_id
from app.repositories.tag_repository import TagRepository

T = TypeVar('T')


@pytest.mark.asyncio
class TestTagRepository:
    """Test TagRepository functionality."""

    async def test_store_tags(self, async_db_initialized: StorageBackend) -> None:
        """Test storing tags for a context entry."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        # Create a context entry
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test entry for tags',
            metadata=None,
        )

        # Store tags
        await repos.tags.store_tags(context_id, ['python', 'testing', 'pytest'])

        # Retrieve and verify
        tags = await repos.tags.get_tags_for_context(context_id)
        assert len(tags) == 3
        assert 'python' in tags
        assert 'testing' in tags
        assert 'pytest' in tags

    async def test_store_tags_normalizes_case(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test that tags are normalized to lowercase."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='case-thread',
            source='user',
            content_type='text',
            text_content='Case normalization test',
            metadata=None,
        )

        # Store tags with mixed case
        await repos.tags.store_tags(context_id, ['Python', 'TESTING', 'PyTest'])

        tags = await repos.tags.get_tags_for_context(context_id)
        assert 'python' in tags
        assert 'testing' in tags
        assert 'pytest' in tags
        # Original case should not be present
        assert 'Python' not in tags
        assert 'TESTING' not in tags

    async def test_store_tags_strips_whitespace(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test that tags are stripped of whitespace."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='whitespace-thread',
            source='user',
            content_type='text',
            text_content='Whitespace test',
            metadata=None,
        )

        # Store tags with whitespace
        await repos.tags.store_tags(context_id, ['  python  ', '\ttesting\n', ' pytest '])

        tags = await repos.tags.get_tags_for_context(context_id)
        assert 'python' in tags
        assert 'testing' in tags
        assert 'pytest' in tags

    async def test_store_tags_skips_empty(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test that empty tags are skipped."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='empty-tag-thread',
            source='user',
            content_type='text',
            text_content='Empty tag test',
            metadata=None,
        )

        # Store tags including empty ones
        await repos.tags.store_tags(context_id, ['python', '', '   ', 'testing'])

        tags = await repos.tags.get_tags_for_context(context_id)
        assert len(tags) == 2
        assert 'python' in tags
        assert 'testing' in tags

    async def test_get_tags_for_context_empty(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test getting tags for context with no tags."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='no-tags-thread',
            source='user',
            content_type='text',
            text_content='Entry without tags',
            metadata=None,
        )

        tags = await repos.tags.get_tags_for_context(context_id)
        assert tags == []

    async def test_get_tags_for_contexts_batch(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test getting tags for multiple contexts in batch."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_ids = []
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'batch-tag-thread-{i}',
                source='user',
                content_type='text',
                text_content=f'Batch entry {i}',
                metadata=None,
            )
            context_ids.append(context_id)
            # Store different tags for each context
            await repos.tags.store_tags(context_id, [f'tag-{i}', 'common-tag'])

        # Get all tags in batch
        all_tags = await repos.tags.get_tags_for_contexts(context_ids)

        assert len(all_tags) == 3
        for i, ctx_id in enumerate(context_ids):
            assert ctx_id in all_tags
            assert f'tag-{i}' in all_tags[ctx_id]
            assert 'common-tag' in all_tags[ctx_id]

    async def test_get_tags_for_contexts_empty_list(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test getting tags for empty context list."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        result = await repos.tags.get_tags_for_contexts([])
        assert result == {}

    async def test_get_tags_for_contexts_nonexistent(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test getting tags for non-existent contexts."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        missing_id_a = generate_id()
        missing_id_b = generate_id()
        result = await repos.tags.get_tags_for_contexts([missing_id_a, missing_id_b])
        assert missing_id_a in result
        assert missing_id_b in result
        assert result[missing_id_a] == []
        assert result[missing_id_b] == []

    async def test_replace_tags_for_context(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test replacing all tags for a context."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='replace-tag-thread',
            source='user',
            content_type='text',
            text_content='Replace tags test',
            metadata=None,
        )

        # Store initial tags
        await repos.tags.store_tags(context_id, ['old-tag-1', 'old-tag-2', 'old-tag-3'])

        # Verify initial tags
        tags = await repos.tags.get_tags_for_context(context_id)
        assert len(tags) == 3

        # Replace with new tags
        await repos.tags.replace_tags_for_context(context_id, ['new-tag-1', 'new-tag-2'])

        # Verify replacement
        tags = await repos.tags.get_tags_for_context(context_id)
        assert len(tags) == 2
        assert 'new-tag-1' in tags
        assert 'new-tag-2' in tags
        assert 'old-tag-1' not in tags

    async def test_replace_tags_with_empty_list(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test replacing tags with empty list removes all."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='empty-replace-tag-thread',
            source='user',
            content_type='text',
            text_content='Empty replace tags test',
            metadata=None,
        )

        # Store tags
        await repos.tags.store_tags(context_id, ['tag-1', 'tag-2'])

        # Replace with empty list
        await repos.tags.replace_tags_for_context(context_id, [])

        # Verify all deleted
        tags = await repos.tags.get_tags_for_context(context_id)
        assert tags == []

    async def test_tags_returned_in_sorted_order(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test that tags are returned in alphabetical order."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='sorted-tag-thread',
            source='user',
            content_type='text',
            text_content='Sorted tags test',
            metadata=None,
        )

        # Store tags in random order
        await repos.tags.store_tags(context_id, ['zebra', 'apple', 'monkey', 'banana'])

        tags = await repos.tags.get_tags_for_context(context_id)

        # Should be in alphabetical order
        assert tags == ['apple', 'banana', 'monkey', 'zebra']

    async def test_special_characters_in_tags(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test handling of special characters in tags."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='special-char-thread',
            source='user',
            content_type='text',
            text_content='Special characters test',
            metadata=None,
        )

        # Store tags with special characters
        special_tags = ['c++', 'c#', '.net', 'node.js', '@typescript']
        await repos.tags.store_tags(context_id, special_tags)

        tags = await repos.tags.get_tags_for_context(context_id)
        assert len(tags) == 5
        # Verify all special tags are present (lowercase)
        assert 'c++' in tags
        assert 'c#' in tags
        assert '.net' in tags
        assert 'node.js' in tags
        assert '@typescript' in tags

    async def test_unicode_tags(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test handling of Unicode tags."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='unicode-tag-thread',
            source='user',
            content_type='text',
            text_content='Unicode tags test',
            metadata=None,
        )

        # Store Unicode tags
        unicode_tags = ['python', 'pythonic', 'test', 'example']
        await repos.tags.store_tags(context_id, unicode_tags)

        tags = await repos.tags.get_tags_for_context(context_id)
        assert len(tags) == 4
        for tag in unicode_tags:
            assert tag.lower() in tags

    async def test_store_tags_idempotent_duplicate_tags(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Storing the same tags twice leaves one row per label.

        Tags are a set: the second call says exactly what the stored rows already say,
        so it adds nothing. Appending instead would surface the label twice in every
        reader's response and count it twice in the tag statistics.
        """
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='dup-tag-thread',
            source='user',
            content_type='text',
            text_content='Test entry for duplicate tags',
        )

        await repos.tags.store_tags(context_id, ['python', 'testing'])
        await repos.tags.store_tags(context_id, ['python', 'testing'])

        tags = await repos.tags.get_tags_for_context(context_id)
        assert tags == ['python', 'testing']

    async def test_tags_shared_across_multiple_contexts(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Same tag can be associated with multiple context entries."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        ctx_id1, _ = await repos.context.store_with_deduplication(
            thread_id='m2m-thread',
            source='user',
            content_type='text',
            text_content='First entry',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            thread_id='m2m-thread',
            source='agent',
            content_type='text',
            text_content='Second entry',
        )

        await repos.tags.store_tags(ctx_id1, ['shared-tag', 'unique-a'])
        await repos.tags.store_tags(ctx_id2, ['shared-tag', 'unique-b'])

        tags1 = await repos.tags.get_tags_for_context(ctx_id1)
        tags2 = await repos.tags.get_tags_for_context(ctx_id2)

        assert 'shared-tag' in tags1
        assert 'shared-tag' in tags2
        assert 'unique-a' in tags1
        assert 'unique-a' not in tags2
        assert 'unique-b' in tags2
        assert 'unique-b' not in tags1

    async def test_replace_tags_normalizes_to_lowercase(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """replace_tags_for_context normalizes tags to lowercase."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='replace-norm-thread',
            source='user',
            content_type='text',
            text_content='Test entry for replace normalization',
        )

        await repos.tags.store_tags(context_id, ['original'])
        await repos.tags.replace_tags_for_context(context_id, ['UPPER', 'MiXeD', '  spaced  '])

        tags = await repos.tags.get_tags_for_context(context_id)
        assert 'original' not in tags
        assert 'upper' in tags
        assert 'mixed' in tags
        assert 'spaced' in tags


@pytest.mark.asyncio
class TestTagDeduplicationWithinOneWrite:
    """One write stores each distinct label exactly once.

    Tags are a set of labels: an entry either carries a label or it does not.
    The stored rows are returned verbatim by ``get_context_by_ids`` and by every
    search tool, and they are counted by the tag statistics, so a duplicate row
    is visible in every response and inflates ``top_tags``. Normalization makes
    this reachable from ordinary input too, because trimming and lower-casing
    turn distinct wire values (``'Alpha'`` and ``'alpha'``) into one label.

    These assertions compare the RAW list, never a ``set``, since wrapping the
    actual value in ``set()`` passes whether two rows were stored or five.
    """

    async def test_store_tags_stores_each_repeated_tag_once(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """A repeated tag in one call produces a single row."""
        from app.repositories import RepositoryContainer

        repos = RepositoryContainer(async_db_initialized)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='dedup-store-thread',
            source='user',
            content_type='text',
            text_content='Duplicate tags in a single write',
            metadata=None,
        )

        await repos.tags.store_tags(context_id, ['alpha', 'beta', 'alpha', 'beta', 'alpha'])

        tags = await repos.tags.get_tags_for_context(context_id)
        assert tags == ['alpha', 'beta']

    async def test_store_tags_collapses_case_and_whitespace_collisions(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Values that only differ before normalization collapse to one row."""
        from app.repositories import RepositoryContainer

        repos = RepositoryContainer(async_db_initialized)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='dedup-case-thread',
            source='user',
            content_type='text',
            text_content='Case-colliding tags',
            metadata=None,
        )

        await repos.tags.store_tags(context_id, ['Tag', 'tag', 'TAG', '  tag  '])

        tags = await repos.tags.get_tags_for_context(context_id)
        assert tags == ['tag']

    async def test_replace_tags_stores_each_repeated_tag_once(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """The update path deduplicates exactly like the store path."""
        from app.repositories import RepositoryContainer

        repos = RepositoryContainer(async_db_initialized)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='dedup-replace-thread',
            source='user',
            content_type='text',
            text_content='Duplicate tags on replacement',
            metadata=None,
        )

        await repos.tags.store_tags(context_id, ['initial'])
        await repos.tags.replace_tags_for_context(context_id, ['Gamma', 'gamma', 'delta', 'DELTA'])

        tags = await repos.tags.get_tags_for_context(context_id)
        assert tags == ['delta', 'gamma']

    async def test_batch_reader_also_returns_one_row_per_label(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """The multi-context reader sees the same deduplicated rows."""
        from app.repositories import RepositoryContainer

        repos = RepositoryContainer(async_db_initialized)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='dedup-batch-thread',
            source='agent',
            content_type='text',
            text_content='Duplicate tags read in batch',
            metadata=None,
        )
        await repos.tags.store_tags(context_id, ['shared', 'SHARED', 'unique'])

        by_context = await repos.tags.get_tags_for_contexts([context_id])
        assert by_context[context_id] == ['shared', 'unique']

    async def test_normalize_tags_preserves_first_seen_order(self) -> None:
        """Deduplication keeps the caller's ordering of the surviving labels."""
        assert TagRepository.normalize_tags(
            ['Zulu', ' yankee ', 'zulu', 'XRAY', 'yankee'],
        ) == ['zulu', 'yankee', 'xray']

    async def test_normalize_tags_drops_blank_entries(self) -> None:
        """Blank and whitespace-only tags never become rows."""
        assert TagRepository.normalize_tags(['', '   ', '\t\n', 'kept']) == ['kept']


class _OrderRecordingConnection:
    """Async stub recording the SQL text the PostgreSQL tag readers emit."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def fetch(self, query: str, *_args: object) -> list[dict[str, object]]:
        """Record the query and return no rows.

        Args:
            query: SQL text emitted by the repository.
            _args: Bound parameters (unused).

        Returns:
            An empty row list.
        """
        self.queries.append(' '.join(query.split()))
        return []


class _OrderRecordingBackend:
    """Minimal backend stub selecting the PostgreSQL branch of every tag reader."""

    backend_type = 'postgresql'

    def __init__(self) -> None:
        self.connection = _OrderRecordingConnection()

    async def execute_read(self, operation: Callable[[Any], Awaitable[T]]) -> T:
        """Run the repository's async closure against the recording connection.

        Args:
            operation: The async closure the repository passes to the backend.

        Returns:
            Whatever the closure returns.
        """
        return await operation(self.connection)


@pytest.mark.asyncio
class TestTagOrderingIsByteWiseOnBothBackends:
    """The public ``tags`` array serializes in the same order on both backends.

    ``ORDER BY tag`` alone is deterministic per backend but not ACROSS backends:
    SQLite compares TEXT byte-wise while PostgreSQL uses the database locale
    collation, which ranks punctuation differently. The same stored entry then
    serializes its ``tags`` array differently depending on the backend -- visible
    to any client that diffs or hashes a response, and reachable through the
    supported cross-backend migration path. PostgreSQL therefore needs an
    explicit byte-wise collation on the sort term.
    """

    async def test_sqlite_orders_tags_byte_wise(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Punctuation sorts before letters, as raw byte comparison requires."""
        from app.repositories import RepositoryContainer

        repos = RepositoryContainer(async_db_initialized)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='tag-order-thread',
            source='user',
            content_type='text',
            text_content='Collation-sensitive tags',
            metadata=None,
        )
        await repos.tags.store_tags(context_id, ['ra', 'r-z'])

        # '-' (0x2D) precedes 'a' (0x61), so byte ordering puts 'r-z' first.
        # A locale collation that ignores punctuation would return the reverse.
        assert await repos.tags.get_tags_for_context(context_id) == ['r-z', 'ra']

    async def test_postgresql_single_context_reader_forces_byte_collation(self) -> None:
        """The single-entry reader pins the sort term to byte order."""
        backend = _OrderRecordingBackend()
        repo = TagRepository(cast(Any, backend))

        await repo.get_tags_for_context('0190abcdef1234567890abcdef123456')

        assert backend.connection.queries
        assert 'ORDER BY tag COLLATE "C"' in backend.connection.queries[0]

    async def test_postgresql_batch_reader_forces_byte_collation(self) -> None:
        """The multi-entry reader pins the same sort term."""
        backend = _OrderRecordingBackend()
        repo = TagRepository(cast(Any, backend))

        await repo.get_tags_for_contexts(['0190abcdef1234567890abcdef123456'])

        assert backend.connection.queries
        assert 'ORDER BY context_entry_id, tag COLLATE "C"' in backend.connection.queries[0]
