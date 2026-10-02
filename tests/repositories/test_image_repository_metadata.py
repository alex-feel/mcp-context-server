"""Tests for per-image metadata in ImageRepository.

Unreadable stored values degrade to no metadata, and supplied values round-trip exactly as given.
"""

import base64
import sqlite3
from typing import Any

import pytest

from app.backends import StorageBackend


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
            owner_id='local',
            visibility='private',
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
    on truthiness would store SQL NULL for it and the read would then omit the key
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
            owner_id='local',
            visibility='private',
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
            owner_id='local',
            visibility='private',
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
            owner_id='local',
            visibility='private',
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
            owner_id='local',
            visibility='private',
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
            owner_id='local',
            visibility='private',
            thread_id='image-metadata-batch-thread',
            source='user',
            content_type='multimodal',
            text_content='Empty per-image metadata read in batch',
            metadata=None,
        )
        await repos.images.store_images(context_id, [self._image('')])

        by_context = await repos.images.get_images_for_contexts([context_id])
        assert by_context[context_id][0].get('metadata') == ''
