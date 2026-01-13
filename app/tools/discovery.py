"""
Discovery operations for MCP tools.

This module contains tools for discovering and analyzing stored context:
- list_threads: List all threads with statistics
- get_statistics: Get database metrics and search availability
"""

import logging
from typing import Any

from fastmcp import Context
from fastmcp.exceptions import ToolError

from app.settings import get_settings
from app.startup import DB_PATH
from app.startup import ensure_backend
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.types import ThreadListDict

logger = logging.getLogger(__name__)
settings = get_settings()


async def list_threads(ctx: Context | None = None) -> ThreadListDict:
    """List all threads with entry statistics. Use for thread discovery and overview.

    Fields explained:
    - entry_count: Total context entries in thread
    - source_types: Number of distinct sources (1=user only or agent only, 2=both)
    - multimodal_count: Entries containing images
    - first_entry/last_entry: ISO timestamps of earliest/latest entries
    - last_id: ID of most recent entry (useful for pagination)

    Returns:
        ThreadListDict with threads (list of thread info dicts) and total_threads (int).

    Raises:
        ToolError: If listing threads fails.
    """
    try:
        if ctx:
            await ctx.info('Listing all threads')

        # Get repositories
        repos = await ensure_repositories()

        # Use statistics repository to get thread list
        threads = await repos.statistics.get_thread_list()

        return {
            'threads': threads,
            'total_threads': len(threads),
        }
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error listing threads: {e}')
        raise ToolError(f'Failed to list threads: {str(e)}') from e


async def get_statistics(ctx: Context | None = None) -> dict[str, Any]:
    """Get database statistics for monitoring and debugging.

    Use for: capacity planning, debugging performance issues, verifying search status.

    Returns:
        Dict with total_contexts (int), total_threads (int), total_images (int),
        total_tags (int), database_size_mb (float), connection_metrics (dict),
        semantic_search (dict with enabled, available, model, dimensions, embedding_count,
        coverage_percentage), fts (dict with enabled, available, language, backend,
        engine, indexed_entries, coverage_percentage), chunking (dict with enabled,
        chunk_size, chunk_overlap, aggregation), reranking (dict with enabled,
        available, provider, model).

    Raises:
        ToolError: If retrieving statistics fails.
    """
    try:
        if ctx:
            await ctx.info('Getting database statistics')

        # Get repositories
        repos = await ensure_repositories()

        # Use statistics repository to get database stats
        stats = await repos.statistics.get_database_statistics(DB_PATH)

        # Ensure backend for metrics
        manager = await ensure_backend()

        # Add connection manager metrics for monitoring
        stats['connection_metrics'] = manager.get_metrics()

        # Add semantic search metrics if available
        if settings.enable_semantic_search:
            if get_embedding_provider() is not None:
                embedding_stats = await repos.embeddings.get_statistics()
                logger.debug(f'[STATISTICS] Embedding repository stats: {embedding_stats}')
                stats['semantic_search'] = {
                    'enabled': True,
                    'available': True,
                    'backend': embedding_stats['backend'],
                    'model': settings.embedding.model,
                    'dimensions': settings.embedding.dim,
                    'context_count': embedding_stats['total_embeddings'],
                    'embedding_count': embedding_stats['total_chunks'],
                    'average_chunks_per_entry': embedding_stats['average_chunks_per_entry'],
                    'coverage_percentage': embedding_stats['coverage_percentage'],
                }
            else:
                stats['semantic_search'] = {
                    'enabled': True,
                    'available': False,
                    'message': 'Dependencies not met or initialization failed',
                }
        else:
            stats['semantic_search'] = {
                'enabled': False,
                'available': False,
            }

        # Add FTS metrics if available
        if settings.enable_fts:
            fts_available = await repos.fts.is_available()
            if fts_available:
                fts_stats = await repos.fts.get_statistics()
                stats['fts'] = {
                    'enabled': True,
                    'available': True,
                    'language': settings.fts_language,
                    'backend': fts_stats['backend'],
                    'engine': fts_stats['engine'],
                    'indexed_entries': fts_stats['indexed_entries'],
                    'coverage_percentage': fts_stats['coverage_percentage'],
                }
            else:
                stats['fts'] = {
                    'enabled': True,
                    'available': False,
                    'message': 'FTS migration not applied',
                }
        else:
            stats['fts'] = {
                'enabled': False,
                'available': False,
            }

        # Add chunking configuration with runtime availability check
        from app.startup import get_chunking_service
        chunking_service = get_chunking_service()
        stats['chunking'] = {
            'enabled': settings.chunking.enabled,
            'available': chunking_service is not None and chunking_service.is_enabled,
            'chunk_size': settings.chunking.size,
            'chunk_overlap': settings.chunking.overlap,
            'aggregation': settings.chunking.aggregation,
        }

        # Add reranking configuration
        from app.startup import get_reranking_provider

        reranking_provider = get_reranking_provider()
        if settings.reranking.enabled:
            if reranking_provider is not None:
                stats['reranking'] = {
                    'enabled': True,
                    'available': True,
                    'provider': settings.reranking.provider,
                    'model': settings.reranking.model,
                }
            else:
                stats['reranking'] = {
                    'enabled': True,
                    'available': False,
                    'message': 'Reranking provider not initialized',
                }
        else:
            stats['reranking'] = {
                'enabled': False,
                'available': False,
            }

        return stats
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error getting statistics: {e}')
        raise ToolError(f'Failed to get statistics: {str(e)}') from e
