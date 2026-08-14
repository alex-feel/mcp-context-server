"""
Image repository for managing image attachments.

This module handles all database operations related to image attachments,
including storage and retrieval of base64-encoded images.

Write-path base64 decodes are STRICT (base64.b64decode with validate=True):
image payloads reach these methods already normalized to canonical
standard-alphabet base64 by validate_and_normalize_images (app.tools._shared),
so a strict decode guarantees the stored bytes are exactly the validated bytes.
A decode failure here means a payload bypassed the validation chokepoint and is
raised as ValueError (aborting the enclosing transaction) rather than silently
storing mangled bytes or dropping the image.
"""


import base64
import json
import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.backends.base import StorageBackend
from app.repositories.base import BaseRepository
from app.types import ImageDict
from app.types import JsonValue

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext

logger = logging.getLogger(__name__)


def encode_image_metadata(metadata: object) -> str | None:
    """Render a per-image metadata value for the ``image_metadata`` column.

    The gate is ``is not None``, NOT truthiness. Per-image metadata crosses the
    tool boundary as a JSON-ENCODED STRING, and the empty string is the one
    valid payload Python considers falsy. Coercing it to SQL NULL would make a
    deliberately empty value indistinguishable from metadata never supplied,
    breaking the documented verbatim round trip. Entry-level metadata already
    gates on ``is not None`` everywhere; this keeps per-image metadata on the
    same rule.

    Args:
        metadata: The caller-supplied per-image metadata value, or ``None`` when
            the caller supplied none.

    Returns:
        The JSON encoding of ``metadata``, or ``None`` to store SQL NULL.
    """
    if metadata is None:
        return None
    return json.dumps(metadata)


def decode_image_metadata(raw: object, context_id: object) -> tuple[bool, JsonValue]:
    """Decode a stored ``image_metadata`` payload without ever raising.

    A malformed payload MUST NOT abort the read. SQLite's ``image_metadata``
    column has TEXT affinity and validates nothing, so a row copied verbatim
    from a legacy or externally produced database can hold a value that is not
    JSON at all. An unguarded decode would raise inside the read callable,
    failing the whole tool call (every id in the batch, or the entire search
    page) and charging the SQLite backend's failure accounting on every retry,
    eventually opening the circuit breaker against unrelated operations. The
    entry-level metadata readers already degrade the identical decode to "no
    metadata"; this does the same for images.

    The presence flag is returned separately from the value so a stored JSON
    ``null`` (a legal payload that decodes to ``None``) stays distinguishable
    from an absent or unreadable one.

    Args:
        raw: The raw column value (``None`` when the column is SQL NULL).
        context_id: Identifier of the owning context entry, for the warning log.

    Returns:
        ``(True, value)`` when the payload decoded, ``(False, None)`` when the
        column was NULL or the payload was unreadable.
    """
    if raw is None:
        return False, None
    try:
        # TypeError joins the ValueError family here because SQLite's TEXT
        # affinity also hands back int/float for a legacy numeric bind, which
        # json.loads rejects by type rather than by content.
        return True, cast(JsonValue, json.loads(cast(Any, raw)))
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        logger.warning(
            'Skipping unreadable image_metadata for context %s; the stored value is not valid JSON',
            context_id,
        )
        return False, None


class ImageRepository(BaseRepository):
    """Repository for image attachment operations.

    Handles storage and retrieval of images associated with context entries,
    including metadata and position tracking.
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize image repository.

        Args:
            backend: Storage backend for executing database operations
        """
        super().__init__(backend)

    async def store_image(
        self,
        context_id: str,
        image_data: bytes,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
        position: int = 0,
    ) -> None:
        """Store a single image attachment.

        Args:
            context_id: ID of the context entry
            image_data: Binary image data
            mime_type: MIME type of the image
            metadata: Optional image metadata. Only ``None`` means "no metadata";
                an empty value is stored and read back as supplied
                (see :func:`encode_image_metadata`).
            position: Position/order of the image
        """
        if self.backend.backend_type == 'sqlite':

            def _store_image_sqlite(conn: sqlite3.Connection) -> None:
                cursor = conn.cursor()
                query = f'''
                    INSERT INTO image_attachments
                    (context_entry_id, image_data, mime_type, image_metadata, position)
                    VALUES ({self._placeholder(1)}, {self._placeholder(2)}, {self._placeholder(3)},
                            {self._placeholder(4)}, {self._placeholder(5)})
                '''
                cursor.execute(
                    query,
                    (context_id, image_data, mime_type, encode_image_metadata(metadata), position),
                )

            await self.backend.execute_write(_store_image_sqlite)
        else:  # postgresql

            async def _store_image_postgresql(conn: 'asyncpg.Connection') -> None:
                query = f'''
                    INSERT INTO image_attachments
                    (context_entry_id, image_data, mime_type, image_metadata, position)
                    VALUES ({self._placeholder(1)}, {self._placeholder(2)}, {self._placeholder(3)},
                            {self._placeholder(4)}, {self._placeholder(5)})
                '''
                await conn.execute(
                    query,
                    context_id,
                    image_data,
                    mime_type,
                    encode_image_metadata(metadata),
                    position,
                )

            await self.backend.execute_write(cast(Any, _store_image_postgresql))

    async def store_images(
        self,
        context_id: str,
        images: list[dict[str, Any]],
        txn: 'TransactionContext | None' = None,
    ) -> None:
        """Store multiple image attachments for a context entry.

        Args:
            context_id: ID of the context entry
            images: List of image dictionaries containing data, mime_type, and optional
                metadata (only an absent or ``None`` metadata value stores SQL NULL)
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _store_images_sqlite(conn: sqlite3.Connection) -> None:
                cursor = conn.cursor()
                stored_count = 0
                for idx, img in enumerate(images):
                    img_data_str = img.get('data', '')
                    if not img_data_str:
                        logger.error(f'Image {idx} for context {context_id} has no data - should have been validated')
                        raise ValueError(f'Image {idx} has no data')

                    try:
                        image_binary = base64.b64decode(img_data_str, validate=True)
                    except Exception as e:
                        logger.error(f'Failed to decode base64 for image {idx} in context {context_id}: {e}')
                        raise ValueError(f'Invalid base64 data in image {idx}') from e

                    query = f'''
                        INSERT INTO image_attachments
                        (context_entry_id, image_data, mime_type, image_metadata, position)
                        VALUES ({self._placeholder(1)}, {self._placeholder(2)}, {self._placeholder(3)},
                                {self._placeholder(4)}, {self._placeholder(5)})
                    '''
                    cursor.execute(
                        query,
                        (
                            context_id,
                            image_binary,
                            img.get('mime_type', 'image/png'),
                            encode_image_metadata(img.get('metadata')),
                            idx,
                        ),
                    )
                    stored_count += 1

                logger.debug(f'Stored {stored_count} images for context {context_id} (SQLite)')

            if txn:
                await self._run_sqlite_txn(_store_images_sqlite, cast(sqlite3.Connection, txn.connection))
            else:
                await self.backend.execute_write(_store_images_sqlite)
        else:  # postgresql

            async def _store_images_postgresql(conn: 'asyncpg.Connection') -> None:
                stored_count = 0
                for idx, img in enumerate(images):
                    img_data_str = img.get('data', '')
                    if not img_data_str:
                        logger.error(f'Image {idx} for context {context_id} has no data - should have been validated')
                        raise ValueError(f'Image {idx} has no data')

                    try:
                        image_binary = base64.b64decode(img_data_str, validate=True)
                    except Exception as e:
                        logger.error(f'Failed to decode base64 for image {idx} in context {context_id}: {e}')
                        raise ValueError(f'Invalid base64 data in image {idx}') from e

                    query = f'''
                        INSERT INTO image_attachments
                        (context_entry_id, image_data, mime_type, image_metadata, position)
                        VALUES ({self._placeholder(1)}, {self._placeholder(2)}, {self._placeholder(3)},
                                {self._placeholder(4)}, {self._placeholder(5)})
                    '''
                    await conn.execute(
                        query,
                        context_id,
                        image_binary,
                        img.get('mime_type', 'image/png'),
                        encode_image_metadata(img.get('metadata')),
                        idx,
                    )
                    stored_count += 1

                logger.debug(f'Stored {stored_count} images for context {context_id} (PostgreSQL)')

            if txn:
                await _store_images_postgresql(cast('asyncpg.Connection', txn.connection))
            else:
                await self.backend.execute_write(cast(Any, _store_images_postgresql))

    async def get_images_for_context(
        self,
        context_id: str,
        include_data: bool = True,
    ) -> list[ImageDict]:
        """Get all images for a specific context entry.

        Args:
            context_id: ID of the context entry
            include_data: Whether to include the actual image data

        Returns:
            List of image dictionaries
        """
        if self.backend.backend_type == 'sqlite':

            def _get_images_sqlite(conn: sqlite3.Connection) -> list[ImageDict]:
                cursor = conn.cursor()

                if include_data:
                    query = f'''
                        SELECT image_data, mime_type, image_metadata, position
                        FROM image_attachments
                        WHERE context_entry_id = {self._placeholder(1)}
                        ORDER BY position
                    '''
                else:
                    query = f'''
                        SELECT mime_type, image_metadata, position
                        FROM image_attachments
                        WHERE context_entry_id = {self._placeholder(1)}
                        ORDER BY position
                    '''
                cursor.execute(query, (context_id,))

                images: list[ImageDict] = []
                for img_row in cursor.fetchall():
                    if include_data:
                        img_data: ImageDict = {
                            'data': base64.b64encode(img_row['image_data']).decode('utf-8'),
                            'mime_type': img_row['mime_type'],
                        }
                    else:
                        img_data = ImageDict(
                            mime_type=img_row['mime_type'],
                        )

                    present, metadata_value = decode_image_metadata(img_row['image_metadata'], context_id)
                    if present:
                        img_data['metadata'] = metadata_value
                    images.append(img_data)
                return images

            return await self.backend.execute_read(_get_images_sqlite)

        # postgresql

        async def _get_images_postgresql(conn: 'asyncpg.Connection') -> list[ImageDict]:
            if include_data:
                query = f'''
                        SELECT image_data, mime_type, image_metadata, position
                        FROM image_attachments
                        WHERE context_entry_id = {self._placeholder(1)}
                        ORDER BY position
                    '''
            else:
                query = f'''
                        SELECT mime_type, image_metadata, position
                        FROM image_attachments
                        WHERE context_entry_id = {self._placeholder(1)}
                        ORDER BY position
                    '''
            rows = await conn.fetch(query, context_id)

            images: list[ImageDict] = []
            for img_row in rows:
                if include_data:
                    img_data: ImageDict = {
                        'data': base64.b64encode(img_row['image_data']).decode('utf-8'),
                        'mime_type': img_row['mime_type'],
                    }
                else:
                    img_data = ImageDict(
                        mime_type=img_row['mime_type'],
                    )

                present, metadata_value = decode_image_metadata(img_row['image_metadata'], context_id)
                if present:
                    img_data['metadata'] = metadata_value
                images.append(img_data)
            return images

        return await self.backend.execute_read(_get_images_postgresql)

    async def get_images_for_contexts(
        self,
        context_ids: list[str],
        include_data: bool = True,
    ) -> dict[str, list[ImageDict]]:
        """Get images for multiple context entries in a single query.

        Args:
            context_ids: List of context entry IDs
            include_data: Whether to include the actual image data

        Returns:
            Dictionary mapping context IDs to their images
        """
        if not context_ids:
            return {}

        if self.backend.backend_type == 'sqlite':

            def _get_images_batch_sqlite(conn: sqlite3.Connection) -> dict[str, list[ImageDict]]:
                cursor = conn.cursor()
                placeholders = self._placeholders(len(context_ids))

                if include_data:
                    query = f'''
                        SELECT context_entry_id, image_data, mime_type, image_metadata, position
                        FROM image_attachments
                        WHERE context_entry_id IN ({placeholders})
                        ORDER BY context_entry_id, position
                    '''
                else:
                    query = f'''
                        SELECT context_entry_id, mime_type, image_metadata, position
                        FROM image_attachments
                        WHERE context_entry_id IN ({placeholders})
                        ORDER BY context_entry_id, position
                    '''
                cursor.execute(query, tuple(context_ids))

                result: dict[str, list[ImageDict]] = {}
                for row in cursor.fetchall():
                    ctx_id = row['context_entry_id']
                    if ctx_id not in result:
                        result[ctx_id] = []

                    if include_data:
                        img_data: ImageDict = {
                            'data': base64.b64encode(row['image_data']).decode('utf-8'),
                            'mime_type': row['mime_type'],
                        }
                    else:
                        img_data = ImageDict(
                            mime_type=row['mime_type'],
                        )

                    present, metadata_value = decode_image_metadata(row['image_metadata'], ctx_id)
                    if present:
                        img_data['metadata'] = metadata_value
                    result[ctx_id].append(img_data)

                for ctx_id in context_ids:
                    if ctx_id not in result:
                        result[ctx_id] = []

                return result

            return await self.backend.execute_read(_get_images_batch_sqlite)

        # postgresql

        async def _get_images_batch_postgresql(conn: 'asyncpg.Connection') -> dict[str, list[ImageDict]]:
            placeholders = self._placeholders(len(context_ids))

            if include_data:
                query = f'''
                        SELECT context_entry_id, image_data, mime_type, image_metadata, position
                        FROM image_attachments
                        WHERE context_entry_id IN ({placeholders})
                        ORDER BY context_entry_id, position
                    '''
            else:
                query = f'''
                        SELECT context_entry_id, mime_type, image_metadata, position
                        FROM image_attachments
                        WHERE context_entry_id IN ({placeholders})
                        ORDER BY context_entry_id, position
                    '''
            rows = await conn.fetch(query, *context_ids)

            result: dict[str, list[ImageDict]] = {}
            for row in rows:
                ctx_id = row['context_entry_id']
                if ctx_id not in result:
                    result[ctx_id] = []

                if include_data:
                    img_data: ImageDict = {
                        'data': base64.b64encode(row['image_data']).decode('utf-8'),
                        'mime_type': row['mime_type'],
                    }
                else:
                    img_data = ImageDict(
                        mime_type=row['mime_type'],
                    )

                present, metadata_value = decode_image_metadata(row['image_metadata'], ctx_id)
                if present:
                    img_data['metadata'] = metadata_value
                result[ctx_id].append(img_data)

            for ctx_id in context_ids:
                if ctx_id not in result:
                    result[ctx_id] = []

            return result

        return await self.backend.execute_read(_get_images_batch_postgresql)

    async def count_images_for_context(self, context_id: str, txn: 'TransactionContext | None' = None) -> int:
        """Count the number of images for a context entry.

        Args:
            context_id: ID of the context entry
            txn: Optional transaction context. When provided the count runs on the
                transaction's own connection instead of acquiring a second pooled
                connection, avoiding a nested pool acquire while a transaction
                connection is already held (PostgreSQL pool-starvation hazard).

        Returns:
            Number of images attached to the context
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        if backend_type == 'sqlite':

            def _count_images_sqlite(conn: sqlite3.Connection) -> int:
                cursor = conn.cursor()
                query = f'SELECT COUNT(*) as count FROM image_attachments WHERE context_entry_id = {self._placeholder(1)}'
                cursor.execute(query, (context_id,))
                result = cursor.fetchone()
                return int(result['count']) if result else 0

            if txn is not None:
                return await self._run_sqlite_txn(_count_images_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_read(_count_images_sqlite)

        # postgresql

        async def _count_images_postgresql(conn: 'asyncpg.Connection') -> int:
            query = f'SELECT COUNT(*) as count FROM image_attachments WHERE context_entry_id = {self._placeholder(1)}'
            result = await conn.fetchrow(query, context_id)
            return int(result['count']) if result else 0

        if txn is not None:
            return await _count_images_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_read(_count_images_postgresql)

    async def replace_images_for_context(
        self,
        context_id: str,
        images: list[dict[str, Any]],
        txn: 'TransactionContext | None' = None,
    ) -> None:
        """Replace all images for a context entry.

        This method performs a complete replacement of images:
        1. Deletes all existing images for the context
        2. Inserts new images with proper base64 decoding

        Args:
            context_id: ID of the context entry
            images: List of image dictionaries containing data, mime_type, and optional
                metadata (only an absent or ``None`` metadata value stores SQL NULL)
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _replace_images_sqlite(conn: sqlite3.Connection) -> None:
                cursor = conn.cursor()

                delete_query = f'DELETE FROM image_attachments WHERE context_entry_id = {self._placeholder(1)}'
                cursor.execute(delete_query, (context_id,))

                # Raise (do not skip) on missing data or a decode failure,
                # mirroring store_images: silently dropping an image the caller
                # provided would corrupt the entry without any error surfaced.
                for idx, img in enumerate(images):
                    img_data_str = img.get('data', '')
                    if not img_data_str:
                        logger.error(f'Image {idx} for context {context_id} has no data - should have been validated')
                        raise ValueError(f'Image {idx} has no data')

                    try:
                        image_binary = base64.b64decode(img_data_str, validate=True)
                    except Exception as e:
                        logger.error(f'Failed to decode base64 for image {idx} in context {context_id}: {e}')
                        raise ValueError(f'Invalid base64 data in image {idx}') from e

                    insert_query = f'''
                        INSERT INTO image_attachments
                        (context_entry_id, image_data, mime_type, image_metadata, position)
                        VALUES ({self._placeholder(1)}, {self._placeholder(2)}, {self._placeholder(3)},
                                {self._placeholder(4)}, {self._placeholder(5)})
                    '''
                    cursor.execute(
                        insert_query,
                        (
                            context_id,
                            image_binary,
                            img.get('mime_type', 'image/png'),
                            encode_image_metadata(img.get('metadata')),
                            idx,
                        ),
                    )

            if txn:
                await self._run_sqlite_txn(_replace_images_sqlite, cast(sqlite3.Connection, txn.connection))
            else:
                await self.backend.execute_write(_replace_images_sqlite)
        else:  # postgresql

            async def _replace_images_postgresql(conn: 'asyncpg.Connection') -> None:
                delete_query = f'DELETE FROM image_attachments WHERE context_entry_id = {self._placeholder(1)}'
                await conn.execute(delete_query, context_id)

                # Raise (do not skip) on missing data or a decode failure,
                # mirroring store_images: silently dropping an image the caller
                # provided would corrupt the entry without any error surfaced.
                for idx, img in enumerate(images):
                    img_data_str = img.get('data', '')
                    if not img_data_str:
                        logger.error(f'Image {idx} for context {context_id} has no data - should have been validated')
                        raise ValueError(f'Image {idx} has no data')

                    try:
                        image_binary = base64.b64decode(img_data_str, validate=True)
                    except Exception as e:
                        logger.error(f'Failed to decode base64 for image {idx} in context {context_id}: {e}')
                        raise ValueError(f'Invalid base64 data in image {idx}') from e

                    insert_query = f'''
                        INSERT INTO image_attachments
                        (context_entry_id, image_data, mime_type, image_metadata, position)
                        VALUES ({self._placeholder(1)}, {self._placeholder(2)}, {self._placeholder(3)},
                                {self._placeholder(4)}, {self._placeholder(5)})
                    '''
                    await conn.execute(
                        insert_query,
                        context_id,
                        image_binary,
                        img.get('mime_type', 'image/png'),
                        encode_image_metadata(img.get('metadata')),
                        idx,
                    )

            if txn:
                await _replace_images_postgresql(cast('asyncpg.Connection', txn.connection))
            else:
                await self.backend.execute_write(cast(Any, _replace_images_postgresql))
