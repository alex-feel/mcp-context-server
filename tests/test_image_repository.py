"""
Tests for image repository.

Tests the ImageRepository class for storing, retrieving, and managing
image attachments associated with context entries.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING
from typing import Any

import pytest

if TYPE_CHECKING:
    from app.backends import StorageBackend


@pytest.mark.asyncio
class TestImageRepository:
    """Test ImageRepository functionality."""

    async def test_store_single_image(self, async_db_initialized: StorageBackend) -> None:
        """Test storing a single image attachment."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        # Create a context entry first
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Test entry for image',
            metadata=None,
        )

        # Store single image
        image_data = b'fake image data'
        await repos.images.store_image(
            context_id=context_id,
            image_data=image_data,
            mime_type='image/png',
            metadata={'width': 100, 'height': 100},
            position=0,
        )

        # Retrieve and verify
        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 1
        assert images[0].get('mime_type') == 'image/png'
        img_data = images[0].get('data')
        assert img_data is not None
        assert base64.b64decode(img_data) == image_data

    async def test_store_multiple_images(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test storing multiple images from base64 list."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='multi-img-thread',
            source='user',
            content_type='multimodal',
            text_content='Multiple images',
            metadata=None,
        )

        # Create base64 encoded images
        images_data: list[dict[str, Any]] = [
            {
                'data': base64.b64encode(b'image 1 data').decode('utf-8'),
                'mime_type': 'image/png',
                'metadata': {'index': 0},
            },
            {
                'data': base64.b64encode(b'image 2 data').decode('utf-8'),
                'mime_type': 'image/jpeg',
                'metadata': {'index': 1},
            },
            {
                'data': base64.b64encode(b'image 3 data').decode('utf-8'),
                'mime_type': 'image/gif',
            },
        ]

        await repos.images.store_images(context_id, images_data)

        # Retrieve and verify
        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 3
        assert images[0].get('mime_type') == 'image/png'
        assert images[1].get('mime_type') == 'image/jpeg'
        assert images[2].get('mime_type') == 'image/gif'

    async def test_store_images_validates_data(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test that store_images validates base64 data."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='validation-thread',
            source='user',
            content_type='multimodal',
            text_content='Invalid image test',
            metadata=None,
        )

        # Try to store image with empty data
        with pytest.raises(ValueError, match='has no data'):
            await repos.images.store_images(
                context_id,
                [{'data': '', 'mime_type': 'image/png'}],
            )

    async def test_store_images_invalid_base64(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test that invalid base64 raises error."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='invalid-base64-thread',
            source='user',
            content_type='multimodal',
            text_content='Invalid base64 test',
            metadata=None,
        )

        # Try to store image with invalid base64
        with pytest.raises(ValueError, match='Invalid base64'):
            await repos.images.store_images(
                context_id,
                [{'data': 'not-valid-base64!@#$', 'mime_type': 'image/png'}],
            )

    async def test_get_images_include_data_false(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test getting images without data."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='no-data-thread',
            source='user',
            content_type='multimodal',
            text_content='Image without data',
            metadata=None,
        )

        await repos.images.store_images(
            context_id,
            [
                {
                    'data': base64.b64encode(b'image data').decode('utf-8'),
                    'mime_type': 'image/png',
                },
            ],
        )

        # Get without data
        images = await repos.images.get_images_for_context(
            context_id, include_data=False,
        )
        assert len(images) == 1
        assert images[0].get('mime_type') == 'image/png'
        assert images[0].get('data') is None

    async def test_get_images_for_contexts_batch(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test getting images for multiple contexts in batch."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_ids = []
        for i in range(3):
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id=f'batch-thread-{i}',
                source='user',
                content_type='multimodal',
                text_content=f'Batch entry {i}',
                metadata=None,
            )
            context_ids.append(context_id)

            # Store 2 images per context
            await repos.images.store_images(
                context_id,
                [
                    {
                        'data': base64.b64encode(f'img {i}-0'.encode()).decode('utf-8'),
                        'mime_type': 'image/png',
                    },
                    {
                        'data': base64.b64encode(f'img {i}-1'.encode()).decode('utf-8'),
                        'mime_type': 'image/jpeg',
                    },
                ],
            )

        # Get all images in batch
        all_images = await repos.images.get_images_for_contexts(context_ids)

        assert len(all_images) == 3
        for ctx_id in context_ids:
            assert ctx_id in all_images
            assert len(all_images[ctx_id]) == 2

    async def test_get_images_for_contexts_empty_list(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test getting images for empty context list."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        result = await repos.images.get_images_for_contexts([])
        assert result == {}

    async def test_get_images_for_contexts_nonexistent(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test getting images for non-existent contexts."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        result = await repos.images.get_images_for_contexts([99999, 99998])
        assert 99999 in result
        assert 99998 in result
        assert result[99999] == []
        assert result[99998] == []

    async def test_count_images_for_context(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test counting images for a context."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='count-thread',
            source='user',
            content_type='multimodal',
            text_content='Count test',
            metadata=None,
        )

        # Store 5 images
        images_data: list[dict[str, Any]] = [
            {
                'data': base64.b64encode(f'image {i}'.encode()).decode('utf-8'),
                'mime_type': 'image/png',
            }
            for i in range(5)
        ]
        await repos.images.store_images(context_id, images_data)

        # Count images
        count = await repos.images.count_images_for_context(context_id)
        assert count == 5

    async def test_count_images_for_nonexistent_context(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test counting images for non-existent context."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        count = await repos.images.count_images_for_context(99999)
        assert count == 0

    async def test_replace_images_for_context(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test replacing all images for a context."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='replace-thread',
            source='user',
            content_type='multimodal',
            text_content='Replace test',
            metadata=None,
        )

        # Store initial images
        await repos.images.store_images(
            context_id,
            [
                {
                    'data': base64.b64encode(b'old image 1').decode('utf-8'),
                    'mime_type': 'image/png',
                },
                {
                    'data': base64.b64encode(b'old image 2').decode('utf-8'),
                    'mime_type': 'image/png',
                },
            ],
        )

        # Replace with new images
        await repos.images.replace_images_for_context(
            context_id,
            [
                {
                    'data': base64.b64encode(b'new image').decode('utf-8'),
                    'mime_type': 'image/jpeg',
                },
            ],
        )

        # Verify replacement
        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 1
        assert images[0].get('mime_type') == 'image/jpeg'
        img_data = images[0].get('data')
        assert img_data is not None
        assert base64.b64decode(img_data) == b'new image'

    async def test_replace_images_with_empty_list(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test replacing images with empty list removes all."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='empty-replace-thread',
            source='user',
            content_type='multimodal',
            text_content='Empty replace test',
            metadata=None,
        )

        # Store images
        await repos.images.store_images(
            context_id,
            [
                {
                    'data': base64.b64encode(b'image').decode('utf-8'),
                    'mime_type': 'image/png',
                },
            ],
        )

        # Replace with empty list (delete query runs, no inserts)
        await repos.images.replace_images_for_context(context_id, [])

        # Verify all deleted
        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 0

    async def test_image_metadata_preserved(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test that image metadata is correctly preserved."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='metadata-thread',
            source='user',
            content_type='multimodal',
            text_content='Metadata test',
            metadata=None,
        )

        metadata = {
            'width': 1920,
            'height': 1080,
            'format': 'png',
            'tags': ['screenshot', 'desktop'],
        }

        await repos.images.store_images(
            context_id,
            [
                {
                    'data': base64.b64encode(b'image with metadata').decode('utf-8'),
                    'mime_type': 'image/png',
                    'metadata': metadata,
                },
            ],
        )

        # Retrieve and verify metadata
        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 1
        assert images[0].get('metadata') == metadata

    async def test_image_position_ordering(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Test that images are retrieved in position order."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='position-thread',
            source='user',
            content_type='multimodal',
            text_content='Position test',
            metadata=None,
        )

        # Store images with metadata indicating expected order (using string values)
        images_data: list[dict[str, Any]] = [
            {
                'data': base64.b64encode(f'image {i}'.encode()).decode('utf-8'),
                'mime_type': 'image/png',
                'metadata': {'order': str(i)},
            }
            for i in range(5)
        ]
        await repos.images.store_images(context_id, images_data)

        # Retrieve and verify order
        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 5
        for i, img in enumerate(images):
            img_metadata = img.get('metadata')
            assert img_metadata is not None
            assert img_metadata.get('order') == str(i)
