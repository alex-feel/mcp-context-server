"""
Image repository for managing image attachments.

This module handles all database operations related to image attachments,
including storage and retrieval of base64-encoded images.
"""

import base64
import json
import sqlite3
from typing import Any

from app.db_manager import DatabaseConnectionManager
from app.repositories.base import BaseRepository
from app.types import ImageDict


class ImageRepository(BaseRepository):
    """Repository for image attachment operations.

    Handles storage and retrieval of images associated with context entries,
    including metadata and position tracking.
    """

    def __init__(self, db_manager: DatabaseConnectionManager) -> None:
        """Initialize image repository.

        Args:
            db_manager: Database connection manager for executing operations
        """
        super().__init__(db_manager)

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
        def _store_image(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO image_attachments
                (context_entry_id, image_data, mime_type, image_metadata, position)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (
                    context_id,
                    image_data,
                    mime_type,
                    json.dumps(metadata) if metadata else None,
                    position,
                ),
            )

        await self.db_manager.execute_write(_store_image)

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
        def _store_images(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            for idx, img in enumerate(images):
                img_data_str = img.get('data', '')
                if not img_data_str:
                    continue

                image_binary = base64.b64decode(img_data_str)
                cursor.execute(
                    '''
                    INSERT INTO image_attachments
                    (context_entry_id, image_data, mime_type, image_metadata, position)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (
                        context_id,
                        image_binary,
                        img.get('mime_type', 'image/png'),
                        json.dumps(img.get('metadata')) if img.get('metadata') else None,
                        idx,
                    ),
                )

        await self.db_manager.execute_write(_store_images)

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
        def _get_images(conn: sqlite3.Connection) -> list[ImageDict]:
            cursor = conn.cursor()

            if include_data:
                cursor.execute(
                    '''
                    SELECT image_data, mime_type, image_metadata, position
                    FROM image_attachments
                    WHERE context_entry_id = ?
                    ORDER BY position
                    ''',
                    (context_id,),
                )
            else:
                cursor.execute(
                    '''
                    SELECT mime_type, image_metadata, position
                    FROM image_attachments
                    WHERE context_entry_id = ?
                    ORDER BY position
                    ''',
                    (context_id,),
                )

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

        return await self.db_manager.execute_read(_get_images)

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

        def _get_images_batch(conn: sqlite3.Connection) -> dict[int, list[ImageDict]]:
            cursor = conn.cursor()
            placeholders = ','.join(['?' for _ in context_ids])

            if include_data:
                cursor.execute(
                    f'''
                    SELECT context_entry_id, image_data, mime_type, image_metadata, position
                    FROM image_attachments
                    WHERE context_entry_id IN ({placeholders})
                    ORDER BY context_entry_id, position
                    ''',
                    tuple(context_ids),
                )
            else:
                cursor.execute(
                    f'''
                    SELECT context_entry_id, mime_type, image_metadata, position
                    FROM image_attachments
                    WHERE context_entry_id IN ({placeholders})
                    ORDER BY context_entry_id, position
                    ''',
                    tuple(context_ids),
                )

            # Group images by context_id
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

            # Ensure all requested IDs have an entry (even if empty)
            for ctx_id in context_ids:
                if ctx_id not in result:
                    result[ctx_id] = []

            return result

        return await self.db_manager.execute_read(_get_images_batch)

    async def count_images_for_context(self, context_id: int) -> int:
        """Count the number of images for a context entry.

        Args:
            context_id: ID of the context entry

        Returns:
            Number of images attached to the context
        """
        def _count_images(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT COUNT(*) as count FROM image_attachments WHERE context_entry_id = ?',
                (context_id,),
            )
            result = cursor.fetchone()
            return int(result['count']) if result else 0

        return await self.db_manager.execute_read(_count_images)
