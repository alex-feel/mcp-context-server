"""
Tests for tag repository.

Tests the TagRepository class for storing, retrieving, and managing
tags associated with context entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from app.backends import StorageBackend


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

        result = await repos.tags.get_tags_for_contexts([99999, 99998])
        assert 99999 in result
        assert 99998 in result
        assert result[99999] == []
        assert result[99998] == []

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
