"""
Repository for vector embeddings using sqlite-vec.

This module provides data access for semantic search embeddings,
handling storage, retrieval, and search operations on vector embeddings.
"""

import logging
import sqlite3
from typing import Any
from typing import Literal

from app.db_manager import DatabaseConnectionManager
from app.logger_config import config_logger
from app.settings import get_settings

# Get settings
settings = get_settings()
# Configure logging
config_logger(settings.log_level)
logger = logging.getLogger(__name__)


class EmbeddingRepository:
    """Repository for vector embeddings using sqlite-vec.

    This repository handles all database operations for semantic search embeddings,
    using the sqlite-vec extension for efficient vector storage and similarity search.
    """

    def __init__(self, db_manager: DatabaseConnectionManager) -> None:
        """Initialize the embedding repository.

        Args:
            db_manager: Database connection manager for all operations
        """
        self.db_manager = db_manager

    async def store(
        self,
        context_id: int,
        embedding: list[float],
        model: str = 'embeddinggemma:latest',
    ) -> None:
        """Store embedding for a context entry.

        Args:
            context_id: ID of the context entry
            embedding: 768-dimensional embedding vector
            model: Model identifier
        """

        def _store(conn: sqlite3.Connection) -> None:
            try:
                import sqlite_vec
            except ImportError as e:
                raise RuntimeError(
                    'sqlite_vec package is required for semantic search. '
                    'Install with: uv sync --extra semantic-search',
                ) from e

            # Serialize embedding
            embedding_blob = sqlite_vec.serialize_float32(embedding)

            # Insert into virtual table (rowid = context_id)
            conn.execute(
                'INSERT INTO vec_context_embeddings(rowid, embedding) VALUES (?, ?)',
                (context_id, embedding_blob),
            )

            # Store metadata
            conn.execute(
                '''
                INSERT INTO embedding_metadata (context_id, model_name, dimensions, created_at, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ''',
                (context_id, model, len(embedding)),
            )

        await self.db_manager.execute_write(_store)
        logger.debug(f'Stored embedding for context {context_id}')

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 20,
        thread_id: str | None = None,
        source: Literal['user', 'agent'] | None = None,
    ) -> list[dict[str, Any]]:
        """KNN search with CTE-based pre-filtering.

        Uses CTE to filter context_entries FIRST, then calculates distances
        using scalar vec_distance_l2() function. This ensures correct number
        of results even with selective filters.

        This approach fixes the bug where sqlite-vec's k parameter in MATCH clause
        limits results at the virtual table level BEFORE JOIN and WHERE filters,
        causing fewer results than requested when filters are applied.

        Performance: O(n * d) where n = filtered entries, d = dimensions.
        Acceptable for n < 100K with d = 768.

        Args:
            query_embedding: Query vector for similarity search
            limit: Maximum number of results to return
            thread_id: Optional filter by thread
            source: Optional filter by source type

        Returns:
            List of search results with context and similarity scores
        """

        def _search(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            try:
                import sqlite_vec
            except ImportError as e:
                raise RuntimeError(
                    'sqlite_vec package is required for semantic search. '
                    'Install with: uv sync --extra semantic-search',
                ) from e

            # Serialize query embedding
            query_blob = sqlite_vec.serialize_float32(query_embedding)

            # Build filter conditions for CTE
            filter_conditions = []
            filter_params: list[Any] = []

            if thread_id:
                filter_conditions.append('thread_id = ?')
                filter_params.append(thread_id)

            if source:
                filter_conditions.append('source = ?')
                filter_params.append(source)

            # Construct WHERE clause (or empty if no filters)
            where_clause = (
                f"WHERE {' AND '.join(filter_conditions)}"
                if filter_conditions
                else ''
            )

            # Query: Filter first (CTE), then calculate distances
            query = f'''
                WITH filtered_contexts AS (
                    SELECT id
                    FROM context_entries
                    {where_clause}
                )
                SELECT
                    ce.id,
                    ce.thread_id,
                    ce.source,
                    ce.content_type,
                    ce.text_content,
                    ce.metadata,
                    ce.created_at,
                    ce.updated_at,
                    vec_distance_l2(?, ve.embedding) as distance
                FROM filtered_contexts fc
                JOIN context_entries ce ON ce.id = fc.id
                JOIN vec_context_embeddings ve ON ve.rowid = fc.id
                ORDER BY distance
                LIMIT ?
            '''

            # Parameters: [filter_params..., query_blob, limit]
            params = filter_params + [query_blob, limit]

            # Execute search
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            # Convert to list of dicts using list comprehension
            return [dict(row) for row in rows]

        return await self.db_manager.execute_read(_search)

    async def update(self, context_id: int, embedding: list[float]) -> None:
        """Update embedding for a context entry.

        Args:
            context_id: ID of the context entry
            embedding: New embedding vector
        """

        def _update(conn: sqlite3.Connection) -> None:
            try:
                import sqlite_vec
            except ImportError as e:
                raise RuntimeError(
                    'sqlite_vec package is required for semantic search. '
                    'Install with: uv sync --extra semantic-search',
                ) from e

            # Serialize embedding
            embedding_blob = sqlite_vec.serialize_float32(embedding)

            # Update virtual table
            conn.execute(
                'UPDATE vec_context_embeddings SET embedding = ? WHERE rowid = ?',
                (embedding_blob, context_id),
            )

            # Update metadata timestamp
            conn.execute(
                'UPDATE embedding_metadata SET updated_at = CURRENT_TIMESTAMP WHERE context_id = ?',
                (context_id,),
            )

        await self.db_manager.execute_write(_update)
        logger.debug(f'Updated embedding for context {context_id}')

    async def delete(self, context_id: int) -> None:
        """Delete embedding for a context entry.

        Args:
            context_id: ID of the context entry
        """

        def _delete(conn: sqlite3.Connection) -> None:
            # Delete from virtual table
            conn.execute('DELETE FROM vec_context_embeddings WHERE rowid = ?', (context_id,))

            # Delete metadata (CASCADE should handle this, but explicit is better)
            conn.execute('DELETE FROM embedding_metadata WHERE context_id = ?', (context_id,))

        await self.db_manager.execute_write(_delete)
        logger.debug(f'Deleted embedding for context {context_id}')

    async def exists(self, context_id: int) -> bool:
        """Check if embedding exists for context entry.

        Args:
            context_id: ID of the context entry

        Returns:
            True if embedding exists, False otherwise
        """

        def _exists(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                'SELECT 1 FROM embedding_metadata WHERE context_id = ? LIMIT 1',
                (context_id,),
            )
            return cursor.fetchone() is not None

        return await self.db_manager.execute_read(_exists)

    async def get_statistics(self, thread_id: str | None = None) -> dict[str, Any]:
        """Get embedding statistics.

        Args:
            thread_id: Optional filter by thread

        Returns:
            Dictionary with statistics (count, coverage, etc.)
        """

        def _get_stats(conn: sqlite3.Connection) -> dict[str, Any]:
            # Get total context entries
            if thread_id:
                cursor = conn.execute(
                    'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                    (thread_id,),
                )
            else:
                cursor = conn.execute('SELECT COUNT(*) FROM context_entries')

            total_entries = cursor.fetchone()[0]

            # Get embedding count
            if thread_id:
                cursor = conn.execute(
                    '''
                    SELECT COUNT(*)
                    FROM embedding_metadata em
                    JOIN context_entries ce ON em.context_id = ce.id
                    WHERE ce.thread_id = ?
                    ''',
                    (thread_id,),
                )
            else:
                cursor = conn.execute('SELECT COUNT(*) FROM embedding_metadata')

            embedding_count = cursor.fetchone()[0]

            # Calculate coverage percentage
            coverage_percentage = (
                (embedding_count / total_entries * 100) if total_entries > 0 else 0.0
            )

            return {
                'total_embeddings': embedding_count,
                'total_entries': total_entries,
                'coverage_percentage': round(coverage_percentage, 2),
            }

        return await self.db_manager.execute_read(_get_stats)

    async def get_table_dimension(self) -> int | None:
        """Get the dimension of the existing vector table.

        This is useful for diagnostics and validation to check if the configured
        EMBEDDING_DIM matches the actual table dimension.

        Returns:
            Dimension of existing embeddings, or None if no embeddings exist
        """

        def _get_dimension(conn: sqlite3.Connection) -> int | None:
            cursor = conn.execute('SELECT dimensions FROM embedding_metadata LIMIT 1')
            row = cursor.fetchone()
            return row[0] if row else None

        return await self.db_manager.execute_read(_get_dimension)
