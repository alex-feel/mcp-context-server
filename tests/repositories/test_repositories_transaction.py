"""
Tests for the repository ``txn`` parameter.

Repository writes, and the reads that run inside store/update transactions, accept an optional transaction context.
These tests check that:
1. txn=None runs through the backend's own execute_write/execute_read path
2. a supplied txn runs on the transaction's connection
3. writes across several repositories commit atomically in one transaction
4. an error inside the transaction rolls every write back
"""


from typing import TYPE_CHECKING

import pytest

from tests.helpers import LOCAL_SCOPE

if TYPE_CHECKING:
    from app.backends import StorageBackend
    from app.repositories import RepositoryContainer


class TestContextRepositoryTransaction:
    """Tests for ContextRepository transaction support."""

    @pytest.mark.asyncio
    async def test_store_with_deduplication_without_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test store_with_deduplication works without a transaction (txn=None)."""
        backend, repos = backend_with_repos

        context_id, was_updated = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test content',
            metadata=None,
            txn=None,  # Explicit None selects the non-transactional path
        )

        assert len(context_id) == 32
        assert was_updated is False

        # Verify data was stored
        entries = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        assert len(entries) == 1
        assert entries[0]['text_content'] == 'Test content'

    @pytest.mark.asyncio
    async def test_store_with_deduplication_with_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test store_with_deduplication works with transaction context."""
        backend, repos = backend_with_repos

        async with backend.begin_transaction() as txn:
            context_id, was_updated = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='test-thread',
                source='agent',
                content_type='text',
                text_content='Transaction content',
                metadata=None,
                txn=txn,
            )

            assert len(context_id) == 32
            assert was_updated is False

        # Transaction committed - verify data persisted
        entries = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        assert len(entries) == 1
        assert entries[0]['text_content'] == 'Transaction content'

    @pytest.mark.asyncio
    async def test_delete_by_ids_with_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test delete_by_ids works with transaction context."""
        backend, repos = backend_with_repos

        # First create an entry
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
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
        entries = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_update_context_entry_with_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test update_context_entry works with transaction context."""
        backend, repos = backend_with_repos

        # First create an entry
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
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
        entries = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        assert len(entries) == 1
        assert entries[0]['text_content'] == 'Updated content'


class TestTagRepositoryTransaction:
    """Tests for TagRepository transaction support."""

    @pytest.mark.asyncio
    async def test_store_tags_without_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test store_tags works without a transaction (txn=None)."""
        backend, repos = backend_with_repos

        # Create context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
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
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test store_tags works with transaction context."""
        backend, repos = backend_with_repos

        # Create context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
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
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test replace_tags_for_context works with transaction context."""
        backend, repos = backend_with_repos

        # Create context with initial tags
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
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
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
        sample_image_data: dict[str, str],
    ) -> None:
        """Test store_images works without a transaction (txn=None)."""
        backend, repos = backend_with_repos

        # Create context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
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
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
        sample_image_data: dict[str, str],
    ) -> None:
        """Test store_images works with transaction context."""
        backend, repos = backend_with_repos

        # Create context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
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
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test atomic commit of context entry with tags."""
        backend, repos = backend_with_repos

        async with backend.begin_transaction() as txn:
            # Store context
            context_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='atomic-test',
                source='agent',
                content_type='text',
                text_content='Atomic content',
                txn=txn,
            )

            # Store tags in same transaction
            await repos.tags.store_tags(context_id, ['atomic', 'test'], txn=txn)

        # Both should be committed
        entries = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        assert len(entries) == 1

        tags = await repos.tags.get_tags_for_context(context_id)
        assert set(tags) == {'atomic', 'test'}

    @pytest.mark.asyncio
    async def test_atomic_context_with_tags_and_images_commit(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
        sample_image_data: dict[str, str],
    ) -> None:
        """Test atomic commit of context entry with tags and images."""
        backend, repos = backend_with_repos

        async with backend.begin_transaction() as txn:
            # Store context
            context_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
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
        entries = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
        assert len(entries) == 1

        tags = await repos.tags.get_tags_for_context(context_id)
        assert set(tags) == {'multimodal', 'atomic'}

        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 1

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        """Test that transaction rollback prevents partial writes."""
        backend, repos = backend_with_repos

        # Get initial count
        initial_count = 0

        try:
            async with backend.begin_transaction() as txn:
                # Store context - this should succeed
                context_id, _ = await repos.context.store_with_deduplication(
                    scope=LOCAL_SCOPE,
                    visibility='private',
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
        entries, _ = await repos.context.search_contexts(thread_id='rollback-test', scope=LOCAL_SCOPE)
        assert len(entries) == initial_count  # Should be 0 if this was the only test


class TestRepositoryMethodsWithoutTransaction:
    """The context, tag and image repository writes work when txn=None."""

    @pytest.mark.asyncio
    async def test_all_methods_work_without_txn(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
        sample_image_data: dict[str, str],
    ) -> None:
        """Call the context, tag and image write methods with the txn argument omitted."""
        backend, repos = backend_with_repos

        # ContextRepository methods
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
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
            scope=LOCAL_SCOPE,
            visibility='private',
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


class TestTxnAwareReadsUseTransactionConnection:
    """Transaction-internal reads run on the txn connection, not a 2nd pool conn.

    get_content_type, count_images_for_context and EmbeddingRepository.exists run
    inside the store/update transaction. Called without ``txn``, each would acquire
    a second pooled connection on PostgreSQL while the transaction connection is
    held, a nested-pool-acquire starvation hazard under saturation; with ``txn``
    they run on the transaction's own connection.
    """

    @pytest.mark.asyncio
    async def test_reads_with_txn_avoid_second_connection(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        import sqlite3 as stdlib_sqlite3
        from unittest.mock import AsyncMock
        from unittest.mock import patch

        backend, repos = backend_with_repos

        # embedding_metadata is created by a migration, not the base schema; create
        # it (empty) so EmbeddingRepository.exists has a table to query.
        def _create_embedding_metadata(conn: stdlib_sqlite3.Connection) -> None:
            conn.execute(
                'CREATE TABLE IF NOT EXISTS embedding_metadata ('
                '  context_id TEXT NOT NULL PRIMARY KEY,'
                '  model_name TEXT NOT NULL,'
                '  dimensions INTEGER NOT NULL,'
                '  chunk_count INTEGER NOT NULL DEFAULT 1,'
                '  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,'
                '  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP'
                ')',
            )

        await backend.execute_write(_create_embedding_metadata)

        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='conc-1', source='user', content_type='text',
            text_content='txn read target', metadata=None,
        )

        async with backend.begin_transaction() as txn:
            # Any pool-acquiring read fails loudly; the txn-aware reads must use the
            # transaction connection instead, so execute_read is never called.
            guard = AsyncMock(side_effect=AssertionError('acquired a second connection'))
            with patch.object(backend, 'execute_read', new=guard):
                content_type = await repos.context.get_content_type(context_id, txn=txn)
                image_count = await repos.images.count_images_for_context(context_id, txn=txn)
                embedding_exists = await repos.embeddings.exists(context_id, txn=txn)

        assert content_type == 'text'
        assert image_count == 0
        assert embedding_exists is False
        guard.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reads_without_txn_use_pool_path(
        self,
        backend_with_repos: 'tuple[StorageBackend, RepositoryContainer]',
    ) -> None:
        # Omitting txn uses the pooled-read path.
        backend, repos = backend_with_repos
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='conc-1b', source='user', content_type='text',
            text_content='pool read target', metadata=None,
        )
        assert await repos.context.get_content_type(context_id) == 'text'
        assert await repos.images.count_images_for_context(context_id) == 0
