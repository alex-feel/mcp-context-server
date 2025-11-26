"""
Image repository for managing image attachments.

This module handles all database operations related to image attachments,
including storage and retrieval of base64-encoded images.
"""

from __future__ import annotations

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

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


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
        context_id: int,
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
            metadata: Optional image metadata
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
                    (context_id, image_data, mime_type, json.dumps(metadata) if metadata else None, position),
                )

            await self.backend.execute_write(_store_image_sqlite)
        else:  # postgresql

            async def _store_image_postgresql(conn: asyncpg.Connection) -> None:
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
                    json.dumps(metadata) if metadata else None,
                    position,
                )

            await self.backend.execute_write(cast(Any, _store_image_postgresql))

    async def store_images(
        self,
        context_id: int,
        images: list[dict[str, Any]],
    ) -> None:
        """Store multiple image attachments for a context entry.

        Args:
            context_id: ID of the context entry
            images: List of image dictionaries containing data, mime_type, and optional metadata
        """
        if self.backend.backend_type == 'sqlite':

            def _store_images_sqlite(conn: sqlite3.Connection) -> None:
                cursor = conn.cursor()
                stored_count = 0
                for idx, img in enumerate(images):
                    img_data_str = img.get('data', '')
                    if not img_data_str:
                        logger.error(f'Image {idx} for context {context_id} has no data - should have been validated')
                        raise ValueError(f'Image {idx} has no data')

                    try:
                        image_binary = base64.b64decode(img_data_str)
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
                            json.dumps(img.get('metadata')) if img.get('metadata') else None,
                            idx,
                        ),
                    )
                    stored_count += 1

                logger.debug(f'Stored {stored_count} images for context {context_id} (SQLite)')

            await self.backend.execute_write(_store_images_sqlite)
        else:  # postgresql

            async def _store_images_postgresql(conn: asyncpg.Connection) -> None:
                stored_count = 0
                for idx, img in enumerate(images):
                    img_data_str = img.get('data', '')
                    if not img_data_str:
                        logger.error(f'Image {idx} for context {context_id} has no data - should have been validated')
                        raise ValueError(f'Image {idx} has no data')

                    try:
                        image_binary = base64.b64decode(img_data_str)
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
                        json.dumps(img.get('metadata')) if img.get('metadata') else None,
                        idx,
                    )
                    stored_count += 1

                logger.debug(f'Stored {stored_count} images for context {context_id} (PostgreSQL)')

            await self.backend.execute_write(cast(Any, _store_images_postgresql))

    async def get_images_for_context(
        self,
        context_id: int,
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
                        img_data = {
                            'mime_type': img_row['mime_type'],
                        }

                    if img_row['image_metadata']:
                        img_data['metadata'] = json.loads(img_row['image_metadata'])
                    images.append(img_data)
                return images

            return await self.backend.execute_read(_get_images_sqlite)

        # postgresql

        async def _get_images_postgresql(conn: asyncpg.Connection) -> list[ImageDict]:
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
                    img_data = {
                        'mime_type': img_row['mime_type'],
                    }

                if img_row['image_metadata']:
                    img_data['metadata'] = json.loads(img_row['image_metadata'])
                images.append(img_data)
            return images

        return await self.backend.execute_read(_get_images_postgresql)

    async def get_images_for_contexts(
        self,
        context_ids: list[int],
        include_data: bool = True,
    ) -> dict[int, list[ImageDict]]:
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

            def _get_images_batch_sqlite(conn: sqlite3.Connection) -> dict[int, list[ImageDict]]:
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

                result: dict[int, list[ImageDict]] = {}
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
                        img_data = {
                            'mime_type': row['mime_type'],
                        }

                    if row['image_metadata']:
                        img_data['metadata'] = json.loads(row['image_metadata'])
                    result[ctx_id].append(img_data)

                for ctx_id in context_ids:
                    if ctx_id not in result:
                        result[ctx_id] = []

                return result

            return await self.backend.execute_read(_get_images_batch_sqlite)

        # postgresql

        async def _get_images_batch_postgresql(conn: asyncpg.Connection) -> dict[int, list[ImageDict]]:
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

            result: dict[int, list[ImageDict]] = {}
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
                    img_data = {
                        'mime_type': row['mime_type'],
                    }

                if row['image_metadata']:
                    img_data['metadata'] = json.loads(row['image_metadata'])
                result[ctx_id].append(img_data)

            for ctx_id in context_ids:
                if ctx_id not in result:
                    result[ctx_id] = []

            return result

        return await self.backend.execute_read(_get_images_batch_postgresql)

    async def count_images_for_context(self, context_id: int) -> int:
        """Count the number of images for a context entry.

        Args:
            context_id: ID of the context entry

        Returns:
            Number of images attached to the context
        """
        if self.backend.backend_type == 'sqlite':

            def _count_images_sqlite(conn: sqlite3.Connection) -> int:
                cursor = conn.cursor()
                query = f'SELECT COUNT(*) as count FROM image_attachments WHERE context_entry_id = {self._placeholder(1)}'
                cursor.execute(query, (context_id,))
                result = cursor.fetchone()
                return int(result['count']) if result else 0

            return await self.backend.execute_read(_count_images_sqlite)

        # postgresql

        async def _count_images_postgresql(conn: asyncpg.Connection) -> int:
            query = f'SELECT COUNT(*) as count FROM image_attachments WHERE context_entry_id = {self._placeholder(1)}'
            result = await conn.fetchrow(query, context_id)
            return int(result['count']) if result else 0

        return await self.backend.execute_read(_count_images_postgresql)

    async def replace_images_for_context(
        self,
        context_id: int,
        images: list[dict[str, Any]],
    ) -> None:
        """Replace all images for a context entry.

        This method performs a complete replacement of images:
        1. Deletes all existing images for the context
        2. Inserts new images with proper base64 decoding

        Args:
            context_id: ID of the context entry
            images: List of image dictionaries containing data, mime_type, and optional metadata
        """
        if self.backend.backend_type == 'sqlite':

            def _replace_images_sqlite(conn: sqlite3.Connection) -> None:
                cursor = conn.cursor()

                delete_query = f'DELETE FROM image_attachments WHERE context_entry_id = {self._placeholder(1)}'
                cursor.execute(delete_query, (context_id,))

                for idx, img in enumerate(images):
                    img_data_str = img.get('data', '')
                    if not img_data_str:
                        continue

                    try:
                        image_binary = base64.b64decode(img_data_str)
                    except Exception as e:
                        logger.error(f'Failed to decode base64 image data: {e}')
                        continue

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
                            json.dumps(img.get('metadata')) if img.get('metadata') else None,
                            idx,
                        ),
                    )

            await self.backend.execute_write(_replace_images_sqlite)
        else:  # postgresql

            async def _replace_images_postgresql(conn: asyncpg.Connection) -> None:
                delete_query = f'DELETE FROM image_attachments WHERE context_entry_id = {self._placeholder(1)}'
                await conn.execute(delete_query, context_id)

                for idx, img in enumerate(images):
                    img_data_str = img.get('data', '')
                    if not img_data_str:
                        continue

                    try:
                        image_binary = base64.b64decode(img_data_str)
                    except Exception as e:
                        logger.error(f'Failed to decode base64 image data: {e}')
                        continue

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
                        json.dumps(img.get('metadata')) if img.get('metadata') else None,
                        idx,
                    )

            await self.backend.execute_write(cast(Any, _replace_images_postgresql))
