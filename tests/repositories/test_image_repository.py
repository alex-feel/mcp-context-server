"""
Tests for image repository.

Tests the ImageRepository class for storing, retrieving, and managing
image attachments associated with context entries.
"""

import base64
import sqlite3
from typing import Any

import pytest

from app.backends import StorageBackend
from app.ids import generate_id


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

        missing_id_a = generate_id()
        missing_id_b = generate_id()
        result = await repos.images.get_images_for_contexts([missing_id_a, missing_id_b])
        assert missing_id_a in result
        assert missing_id_b in result
        assert result[missing_id_a] == []
        assert result[missing_id_b] == []

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

        count = await repos.images.count_images_for_context(generate_id())
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
            assert isinstance(img_metadata, dict)
            assert img_metadata.get('order') == str(i)

    async def test_image_cascade_delete_on_context_removal(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Images are cascade-deleted when their parent context entry is deleted."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='cascade-thread',
            source='user',
            content_type='multimodal',
            text_content='Test cascade delete',
        )

        image_data = base64.b64encode(b'cascade test image').decode('utf-8')
        await repos.images.store_images(context_id, [
            {'data': image_data, 'mime_type': 'image/png'},
        ])

        count_before = await repos.images.count_images_for_context(context_id)
        assert count_before == 1

        deleted = await repos.context.delete_by_ids([context_id])
        assert deleted == 1

        count_after = await repos.images.count_images_for_context(context_id)
        assert count_after == 0

    async def test_store_images_large_binary_data(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Store and retrieve a large image (1MB) without data loss."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='large-img-thread',
            source='user',
            content_type='multimodal',
            text_content='Test large image',
        )

        large_data = bytes(range(256)) * 4096
        large_b64 = base64.b64encode(large_data).decode('utf-8')

        await repos.images.store_images(context_id, [
            {'data': large_b64, 'mime_type': 'image/jpeg'},
        ])

        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 1
        img_data = images[0].get('data')
        assert img_data is not None
        retrieved_data = base64.b64decode(img_data)
        assert retrieved_data == large_data

    async def test_store_images_default_mime_type(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Image stored without mime_type gets default 'image/png'."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='default-mime-thread',
            source='user',
            content_type='multimodal',
            text_content='Test default mime type',
        )

        image_data = base64.b64encode(b'no mime type image').decode('utf-8')
        await repos.images.store_images(context_id, [
            {'data': image_data},
        ])

        images = await repos.images.get_images_for_context(context_id)
        assert len(images) == 1
        assert images[0].get('mime_type') == 'image/png'


@pytest.mark.asyncio
class TestMalformedStoredImageMetadata:
    """An unreadable stored payload degrades to "no metadata", never to a failed read.

    SQLite's ``image_metadata`` column has TEXT affinity and validates nothing, so a
    row copied verbatim from a legacy or externally produced database can hold a
    value that is not JSON. The readers here are reached by ``get_context_by_ids``
    (which fetches images by default) and by every search tool called with
    ``include_images=True``, so an unguarded decode would fail the WHOLE call --
    every id in the batch, or the entire search page -- and charge the backend's
    failure accounting on every retry until the circuit breaker opened against
    unrelated operations. Entry-level metadata already degrades the identical
    decode to ``None``; images must behave the same way.
    """

    @staticmethod
    async def _seed_entry_with_raw_metadata(
        backend: StorageBackend, raw_metadata: str,
    ) -> str:
        """Insert one entry whose single image carries a hand-written metadata value.

        Args:
            backend: Initialized SQLite backend.
            raw_metadata: Exact text to place in the image_metadata column.

        Returns:
            The id of the created context entry.
        """
        from app.repositories import RepositoryContainer

        repos = RepositoryContainer(backend)
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='malformed-image-metadata-thread',
            source='agent',
            content_type='multimodal',
            text_content='Entry whose image metadata was written outside the app',
            metadata=None,
        )

        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                'INSERT INTO image_attachments '
                '(context_entry_id, image_data, mime_type, image_metadata, position) '
                'VALUES (?, ?, ?, ?, ?)',
                (context_id, b'binary-image-bytes', 'image/png', raw_metadata, 0),
            )

        await backend.execute_write(_insert)
        return context_id

    async def test_single_context_reader_omits_unreadable_metadata(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """The image is still returned, without a metadata key."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        context_id = await self._seed_entry_with_raw_metadata(backend, 'not json at all')
        repos = RepositoryContainer(backend)

        images = await repos.images.get_images_for_context(context_id)

        assert len(images) == 1
        assert images[0]['mime_type'] == 'image/png'
        assert 'metadata' not in images[0]

    async def test_batch_reader_omits_unreadable_metadata(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """The multi-context reader degrades the same way."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        context_id = await self._seed_entry_with_raw_metadata(backend, '{"unclosed": ')
        repos = RepositoryContainer(backend)

        by_context = await repos.images.get_images_for_contexts([context_id])

        assert len(by_context[context_id]) == 1
        assert 'metadata' not in by_context[context_id][0]

    async def test_unreadable_metadata_does_not_charge_the_failure_counters(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """A static bad row must not accumulate faults toward an open circuit."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        context_id = await self._seed_entry_with_raw_metadata(backend, 'still not json')
        repos = RepositoryContainer(backend)

        before = backend.get_metrics()
        for _ in range(3):
            await repos.images.get_images_for_context(context_id)
            await repos.images.get_images_for_contexts([context_id])
        after = backend.get_metrics()

        assert after['failed_queries'] == before['failed_queries']
        assert after['consecutive_failures'] == before['consecutive_failures']

    async def test_metadata_without_data_payload_also_degrades(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """The include_data=False projection shares the guarded decode."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        context_id = await self._seed_entry_with_raw_metadata(backend, 'nope')
        repos = RepositoryContainer(backend)

        images = await repos.images.get_images_for_context(context_id, include_data=False)

        assert len(images) == 1
        assert 'data' not in images[0]
        assert 'metadata' not in images[0]

    async def test_valid_metadata_is_still_returned(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """The guard only skips what cannot be decoded."""
        from app.repositories import RepositoryContainer

        backend = async_db_initialized
        context_id = await self._seed_entry_with_raw_metadata(backend, '{"caption": "ok"}')
        repos = RepositoryContainer(backend)

        images = await repos.images.get_images_for_context(context_id)

        assert images[0].get('metadata') == {'caption': 'ok'}


@pytest.mark.asyncio
class TestPerImageMetadataValueFidelity:
    """A supplied per-image metadata value round-trips exactly as supplied.

    Per-image metadata crosses the tool boundary as a JSON-encoded string, and the
    empty string is the one valid payload Python treats as falsy. Gating the write
    on truthiness stored SQL NULL for it, and the read then omitted the key
    entirely, so a client that deliberately stored an empty value could not tell it
    apart from never having sent one. Entry-level metadata gates on ``is not None``
    everywhere; per-image metadata must too.
    """

    @staticmethod
    def _image(metadata: object) -> dict[str, Any]:
        """Build one valid image payload carrying the given metadata value.

        Args:
            metadata: Value to place in the image's metadata field.

        Returns:
            An image dictionary accepted by the repository write path.
        """
        return {
            'data': base64.b64encode(b'round-trip image bytes').decode('utf-8'),
            'mime_type': 'image/png',
            'metadata': metadata,
        }

    async def test_empty_string_metadata_round_trips(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """An empty string is stored and returned, not dropped."""
        from app.repositories import RepositoryContainer

        repos = RepositoryContainer(async_db_initialized)
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='image-metadata-fidelity-thread',
            source='user',
            content_type='multimodal',
            text_content='Empty per-image metadata',
            metadata=None,
        )

        await repos.images.store_images(context_id, [self._image('')])

        images = await repos.images.get_images_for_context(context_id)
        assert 'metadata' in images[0]
        assert images[0]['metadata'] == ''

    async def test_empty_string_metadata_round_trips_through_replacement(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """The update path stores the empty value exactly like the store path."""
        from app.repositories import RepositoryContainer

        repos = RepositoryContainer(async_db_initialized)
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='image-metadata-replace-thread',
            source='user',
            content_type='multimodal',
            text_content='Empty per-image metadata on replacement',
            metadata=None,
        )

        await repos.images.store_images(context_id, [self._image('original caption')])
        await repos.images.replace_images_for_context(context_id, [self._image('')])

        images = await repos.images.get_images_for_context(context_id)
        assert images[0].get('metadata') == ''

    async def test_absent_metadata_still_omits_the_key(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Not supplying metadata remains distinguishable from supplying an empty one."""
        from app.repositories import RepositoryContainer

        repos = RepositoryContainer(async_db_initialized)
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='image-metadata-absent-thread',
            source='user',
            content_type='multimodal',
            text_content='No per-image metadata',
            metadata=None,
        )

        await repos.images.store_images(
            context_id,
            [{
                'data': base64.b64encode(b'no metadata bytes').decode('utf-8'),
                'mime_type': 'image/png',
            }],
        )

        images = await repos.images.get_images_for_context(context_id)
        assert 'metadata' not in images[0]

    async def test_empty_mapping_metadata_round_trips_on_the_single_image_writer(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """The single-image writer applies the same gate as the batch writers."""
        from app.repositories import RepositoryContainer

        repos = RepositoryContainer(async_db_initialized)
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='image-metadata-single-writer-thread',
            source='user',
            content_type='multimodal',
            text_content='Empty mapping per-image metadata',
            metadata=None,
        )

        await repos.images.store_image(
            context_id=context_id,
            image_data=b'single writer bytes',
            mime_type='image/png',
            metadata={},
        )

        images = await repos.images.get_images_for_context(context_id)
        assert images[0].get('metadata') == {}

    async def test_batch_reader_returns_the_empty_value_too(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """Reading many contexts preserves the same value fidelity."""
        from app.repositories import RepositoryContainer

        repos = RepositoryContainer(async_db_initialized)
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='image-metadata-batch-thread',
            source='user',
            content_type='multimodal',
            text_content='Empty per-image metadata read in batch',
            metadata=None,
        )
        await repos.images.store_images(context_id, [self._image('')])

        by_context = await repos.images.get_images_for_contexts([context_id])
        assert by_context[context_id][0].get('metadata') == ''
