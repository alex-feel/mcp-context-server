"""
Tests for repository transaction support (Phase 2).

This module tests the transaction context parameter (txn) added to repository
write methods, ensuring:
1. Backward compatibility when txn=None (uses execute_write)
2. Direct connection usage when txn is provided
3. Atomic multi-repository operations within transactions
4. Proper rollback on errors
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from app.backends import create_backend
from app.repositories import RepositoryContainer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from app.backends import StorageBackend


@pytest_asyncio.fixture
async def backend_with_repos(temp_db_path: Path) -> AsyncGenerator[tuple[StorageBackend, RepositoryContainer], None]:
    """Create backend and repository container for transaction tests."""
    # Initialize database schema first
    import sqlite3 as stdlib_sqlite3

    from app.schemas import load_schema

    conn = stdlib_sqlite3.connect(str(temp_db_path))
    try:
        schema_sql = load_schema('sqlite')
        conn.executescript(schema_sql)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.commit()
    finally:
        conn.close()

    # Create backend and initialize
    backend = create_backend(backend_type='sqlite', db_path=str(temp_db_path))
    await backend.initialize()

    repos = RepositoryContainer(backend)

    yield backend, repos

    await backend.shutdown()


class TestContextRepositoryTransaction:
    """Tests for ContextRepository transaction support."""

    @pytest.mark.asyncio
    async def test_store_with_deduplication_without_txn(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """Test store_with_deduplication works without transaction (backward compat)."""
        backend, repos = backend_with_repos

        context_id, was_updated = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
            metadata=None,
            txn=None,  # Explicit None for backward compatibility
        )

        assert context_id > 0
        assert was_updated is False

        # Verify data was stored
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1
        assert entries[0]['text_content'] == 'Test content'

    @pytest.mark.asyncio
    async def test_store_with_deduplication_with_txn(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """Test store_with_deduplication works with transaction context."""
        backend, repos = backend_with_repos

        async with backend.begin_transaction() as txn:
            context_id, was_updated = await repos.context.store_with_deduplication(
                thread_id='test-thread',
                source='agent',
                content_type='text',
                text_content='Transaction content',
                metadata=None,
                txn=txn,
            )

            assert context_id > 0
            assert was_updated is False

        # Transaction committed - verify data persisted
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1
        assert entries[0]['text_content'] == 'Transaction content'

    @pytest.mark.asyncio
    async def test_delete_by_ids_with_txn(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """Test delete_by_ids works with transaction context."""
        backend, repos = backend_with_repos

        # First create an entry
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='To be deleted',
        )

        # Delete within transaction
        async with backend.begin_transaction() as txn:
            deleted_count = await repos.context.delete_by_ids([context_id], txn=txn)
            assert deleted_count == 1

        # Verify deletion persisted
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_update_context_entry_with_txn(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """Test update_context_entry works with transaction context."""
        backend, repos = backend_with_repos

        # First create an entry
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Original content',
        )

        # Update within transaction
        async with backend.begin_transaction() as txn:
            success, updated_fields = await repos.context.update_context_entry(
                context_id=context_id,
                text_content='Updated content',
                txn=txn,
            )
            assert success is True
            assert 'text_content' in updated_fields

        # Verify update persisted
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1
        assert entries[0]['text_content'] == 'Updated content'


class TestTagRepositoryTransaction:
    """Tests for TagRepository transaction support."""

    @pytest.mark.asyncio
    async def test_store_tags_without_txn(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """Test store_tags works without transaction (backward compat)."""
        backend, repos = backend_with_repos

        # Create context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
        )

        # Store tags without transaction
        await repos.tags.store_tags(context_id, ['tag1', 'tag2'], txn=None)

        # Verify tags were stored
        tags = await repos.tags.get_tags_for_context(context_id)
        assert set(tags) == {'tag1', 'tag2'}

    @pytest.mark.asyncio
    async def test_store_tags_with_txn(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """Test store_tags works with transaction context."""
        backend, repos = backend_with_repos

        # Create context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
        )

        # Store tags within transaction
        async with backend.begin_transaction() as txn:
            await repos.tags.store_tags(context_id, ['txn-tag1', 'txn-tag2'], txn=txn)

        # Verify tags persisted after commit
        tags = await repos.tags.get_tags_for_context(context_id)
        assert set(tags) == {'txn-tag1', 'txn-tag2'}

    @pytest.mark.asyncio
    async def test_replace_tags_with_txn(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """Test replace_tags_for_context works with transaction context."""
        backend, repos = backend_with_repos

        # Create context with initial tags
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
        )
        await repos.tags.store_tags(context_id, ['old-tag1', 'old-tag2'])

        # Replace tags within transaction
        async with backend.begin_transaction() as txn:
            await repos.tags.replace_tags_for_context(context_id, ['new-tag1', 'new-tag2'], txn=txn)

        # Verify replacement persisted
        tags = await repos.tags.get_tags_for_context(context_id)
        assert set(tags) == {'new-tag1', 'new-tag2'}


class TestImageRepositoryTransaction:
    """Tests for ImageRepository transaction support."""

    @pytest.mark.asyncio
    async def test_store_images_without_txn(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
        sample_image_data: dict[str, str],
    ) -> None:
        """Test store_images works without transaction (backward compat)."""
        backend, repos = backend_with_repos

        # Create context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='multimodal',
            text_content='Image content',
        )

        # Store image without transaction
        await repos.images.store_images(context_id, [sample_image_data], txn=None)

        # Verify image was stored
        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 1
        assert images[0].get('mime_type') == 'image/png'

    @pytest.mark.asyncio
    async def test_store_images_with_txn(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
        sample_image_data: dict[str, str],
    ) -> None:
        """Test store_images works with transaction context."""
        backend, repos = backend_with_repos

        # Create context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='multimodal',
            text_content='Image content',
        )

        # Store image within transaction
        async with backend.begin_transaction() as txn:
            await repos.images.store_images(context_id, [sample_image_data], txn=txn)

        # Verify image persisted after commit
        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 1


class TestMultiRepositoryTransaction:
    """Tests for atomic operations across multiple repositories."""

    @pytest.mark.asyncio
    async def test_atomic_context_with_tags_commit(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """Test atomic commit of context entry with tags."""
        backend, repos = backend_with_repos

        async with backend.begin_transaction() as txn:
            # Store context
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='atomic-test',
                source='agent',
                content_type='text',
                text_content='Atomic content',
                txn=txn,
            )

            # Store tags in same transaction
            await repos.tags.store_tags(context_id, ['atomic', 'test'], txn=txn)

        # Both should be committed
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1

        tags = await repos.tags.get_tags_for_context(context_id)
        assert set(tags) == {'atomic', 'test'}

    @pytest.mark.asyncio
    async def test_atomic_context_with_tags_and_images_commit(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
        sample_image_data: dict[str, str],
    ) -> None:
        """Test atomic commit of context entry with tags and images."""
        backend, repos = backend_with_repos

        async with backend.begin_transaction() as txn:
            # Store context
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='multimodal-atomic',
                source='user',
                content_type='multimodal',
                text_content='Content with image',
                txn=txn,
            )

            # Store tags
            await repos.tags.store_tags(context_id, ['multimodal', 'atomic'], txn=txn)

            # Store image
            await repos.images.store_images(context_id, [sample_image_data], txn=txn)

        # All should be committed
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1

        tags = await repos.tags.get_tags_for_context(context_id)
        assert set(tags) == {'multimodal', 'atomic'}

        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 1

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """Test that transaction rollback prevents partial writes."""
        backend, repos = backend_with_repos

        # Get initial count
        initial_count = 0

        try:
            async with backend.begin_transaction() as txn:
                # Store context - this should succeed
                context_id, _ = await repos.context.store_with_deduplication(
                    thread_id='rollback-test',
                    source='user',
                    content_type='text',
                    text_content='Should be rolled back',
                    txn=txn,
                )

                # Store tags - this should succeed
                await repos.tags.store_tags(context_id, ['rollback'], txn=txn)

                # Force an error to trigger rollback
                raise ValueError('Simulated error for rollback test')

        except ValueError:
            pass  # Expected error

        # Verify nothing was committed - search for the content
        entries, _ = await repos.context.search_contexts(thread_id='rollback-test')
        assert len(entries) == initial_count  # Should be 0 if this was the only test


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility when txn=None."""

    @pytest.mark.asyncio
    async def test_all_methods_work_without_txn(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
        sample_image_data: dict[str, str],
    ) -> None:
        """Comprehensive test that all modified methods work without txn parameter."""
        backend, repos = backend_with_repos

        # ContextRepository methods
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='compat-test',
            source='user',
            content_type='text',
            text_content='Backward compat test',
        )

        success, fields = await repos.context.update_context_entry(
            context_id=context_id,
            text_content='Updated compat test',
        )
        assert success is True

        # TagRepository methods
        await repos.tags.store_tags(context_id, ['compat'])
        await repos.tags.replace_tags_for_context(context_id, ['replaced'])

        tags = await repos.tags.get_tags_for_context(context_id)
        assert tags == ['replaced']

        # ImageRepository methods (create multimodal entry for images)
        mm_id, _ = await repos.context.store_with_deduplication(
            thread_id='compat-test',
            source='user',
            content_type='multimodal',
            text_content='Multimodal compat test',
        )
        await repos.images.store_images(mm_id, [sample_image_data])
        await repos.images.replace_images_for_context(mm_id, [sample_image_data])

        images = await repos.images.get_images_for_context(mm_id)
        assert len(images) == 1

        # Cleanup
        await repos.context.delete_by_ids([context_id, mm_id])
