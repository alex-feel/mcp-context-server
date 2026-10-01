"""Embedding repository: vector embedding storage and KNN search on both backends.

``EmbeddingRepository`` is assembled here from one mixin per group of operations:
chunk writes and deletes, fp32 search, compressed search, and the embedding
inventory. Its ``search`` routes each call to the fp32 or the compressed search
according to the compression toggle. Shared value types live in ``records`` and
the compression provenance cache in ``compression_cache``.
"""

from typing import Any
from typing import Literal

from app.repositories.embedding_repository.chunk_writes import ChunkWriteMixin
from app.repositories.embedding_repository.compressed_search import CompressedSearchMixin
from app.repositories.embedding_repository.fp32_search import Fp32SearchMixin
from app.repositories.embedding_repository.inventory import EmbeddingInventoryMixin


class EmbeddingRepository(
    ChunkWriteMixin,
    Fp32SearchMixin,
    CompressedSearchMixin,
    EmbeddingInventoryMixin,
):
    """Repository for vector embeddings supporting both sqlite-vec and pgvector.

    This repository handles all database operations for semantic search embeddings,
    using either sqlite-vec extension (SQLite) or pgvector extension (PostgreSQL)
    depending on the configured storage backend.

    Supported backends:
    - SQLite: Uses sqlite-vec with BLOB storage and vec_distance_l2()
    - PostgreSQL: Uses pgvector with native vector type and <-> operator
    """

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 20,
        offset: int = 0,
        thread_id: str | None = None,
        source: Literal['user', 'agent'] | None = None,
        content_type: Literal['text', 'multimodal'] | None = None,
        tags: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
        explain_query: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """KNN search with optional filters including date range and metadata.

        SQLite: Uses CTE-based pre-filtering with vec_distance_l2() function
        PostgreSQL: Uses direct JOIN with <-> operator for L2 distance

        Args:
            query_embedding: Query vector for similarity search
            limit: Maximum number of results to return
            offset: Number of results to skip (pagination)
            thread_id: Optional filter by thread
            source: Optional filter by source type
            content_type: Filter by content type (text or multimodal)
            tags: Filter by any of these tags (OR logic)
            start_date: Filter by created_at >= date (ISO 8601 format)
            end_date: Filter by created_at <= date (ISO 8601 format)
            metadata: Simple metadata filters (key=value equality)
            metadata_filters: Advanced metadata filters with operators
            explain_query: If True, include query execution plan in stats

        Returns:
            Tuple of (search results list, statistics dictionary)
        """
        # Dispatch to the compressed read path when the bootstrap-only
        # toggle is on. Settings are read once per process (CLAUDE.md
        # "Settings Singleton Caching") so this branch is stable for the
        # lifetime of the running server.
        from app.settings import get_settings
        if get_settings().compression.enabled:
            return await self.search_compressed(
                query_embedding=query_embedding,
                limit=limit,
                offset=offset,
                thread_id=thread_id,
                source=source,
                content_type=content_type,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
                metadata=metadata,
                metadata_filters=metadata_filters,
                explain_query=explain_query,
            )

        return await self.search_fp32(
            query_embedding=query_embedding,
            limit=limit,
            offset=offset,
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            tags=tags,
            start_date=start_date,
            end_date=end_date,
            metadata=metadata,
            metadata_filters=metadata_filters,
            explain_query=explain_query,
        )
