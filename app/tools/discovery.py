"""
Discovery operations for MCP tools.

This module contains tools for discovering and analyzing stored context:
- list_threads: List all threads with statistics
- get_statistics: Get database metrics and search availability
"""

import logging
from typing import Annotated
from typing import cast

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from app.errors import format_exception_message
from app.settings import get_settings
from app.startup import DB_PATH
from app.startup import ensure_backend
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.types import StatisticsResponseDict
from app.types import ThreadListDict

logger = logging.getLogger(__name__)
settings = get_settings()


async def list_threads(
    limit: Annotated[
        int | None,
        Field(ge=1, le=100, description='Maximum threads to return (1-100); omit for all threads (default)'),
    ] = None,
    offset: Annotated[int, Field(ge=0, description='Pagination offset (default: 0)')] = 0,
    ctx: Context | None = None,
) -> ThreadListDict:
    """List threads with entry statistics. Use for thread discovery and overview.

    Pagination is optional and backward-compatible: with no arguments ALL threads
    are returned. Supply `limit` to bound the result and `offset` to skip rows.
    Threads are ordered by most-recent activity first (last_entry DESC, then by
    latest entry id DESC); pagination applies AFTER this ordering.

    Args:
        limit: Maximum threads to return (1-100). When omitted (None), all
            threads are returned (no limit).
        offset: Number of leading threads to skip for pagination (default 0).
            Ignored when `limit` is omitted.
        ctx: FastMCP context (injected; hidden from clients).

    Fields explained:
    - entry_count: Total context entries in thread
    - source_types: Number of distinct sources (1=user only or agent only, 2=both)
    - multimodal_count: Entries containing images
    - first_entry/last_entry: ISO timestamps of earliest/latest entries
    - last_id: ID of most recent entry (a hint for keyset pagination; not yet
      consumed as a cursor -- limit/offset is the supported pagination today)

    Returns:
        ThreadListDict with threads (list of thread info dicts for the requested
        page) and total_threads (count of threads in THIS response, not the whole
        database).

    Raises:
        ToolError: If listing threads fails.
    """
    try:
        if ctx:
            await ctx.info('Listing threads')

        # Get repositories
        repos = await ensure_repositories()

        # Use statistics repository to get thread list (optionally paginated).
        threads = await repos.statistics.get_thread_list(limit=limit, offset=offset)

        return {
            'threads': threads,
            'total_threads': len(threads),
        }
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error listing threads: {e}')
        raise ToolError(f'Failed to list threads: {format_exception_message(e)}') from e


async def get_statistics(ctx: Context | None = None) -> StatisticsResponseDict:
    """Get server statistics for monitoring and debugging.

    Use for: capacity planning, debugging performance issues, verifying search status.

    Returns:
        StatisticsResponseDict with total_entries (int), total_threads (int),
        total_images (int), unique_tags (int), database_size_mb (float),
        connection_metrics (dict), semantic_search (dict with enabled,
        available, backend, model, dimensions, context_count, embedding_count,
        average_chunks_per_entry, coverage_percentage),
        fts (dict with enabled, available, language, backend, engine,
        indexed_entries, coverage_percentage), chunking (dict with enabled,
        chunk_size, chunk_overlap, aggregation), reranking (dict with
        enabled, available, provider, model), summary (dict with enabled,
        available, provider, model, summary_count, coverage_percentage,
        min_content_length), compression (dict with enabled, available,
        provider, bits, variant, seed, dim, max_concurrent), index_tree
        (dict with enabled, node_count).

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

        # Add embeddings storage size immediately after total database size.
        # Gated on embedding generation OR compression (NOT semantic_search.enabled),
        # so the field still appears in compression-on / semantic-search-off
        # deployments. Degrades to 0.0 if the sub-block computation fails.
        if settings.embedding.generation_enabled or settings.compression.enabled:
            try:
                embeddings_size_mb, embeddings_size_estimated = await repos.embeddings.get_embeddings_size()
                stats['embeddings_size_mb'] = embeddings_size_mb
                stats['embeddings_size_estimated'] = embeddings_size_estimated
            except Exception as e:
                logger.warning(f'Failed to get embeddings size: {e}')
                stats['embeddings_size_mb'] = 0.0
                stats['embeddings_size_estimated'] = False

        # Ensure backend for metrics
        manager = await ensure_backend()

        # Add connection manager metrics for monitoring
        stats['connection_metrics'] = manager.get_metrics()

        # Add semantic search metrics if available
        if settings.semantic_search.enabled:
            if get_embedding_provider() is not None:
                embedding_stats = await repos.embeddings.get_statistics()
                logger.debug(f'Embedding repository stats: {embedding_stats}')
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
        if settings.fts.enabled:
            fts_available = await repos.fts.is_available()
            if fts_available:
                fts_stats = await repos.fts.get_statistics()
                stats['fts'] = {
                    'enabled': True,
                    'available': True,
                    'language': settings.fts.language,
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

        # Add summary generation statistics
        from app.startup import get_summary_provider

        summary_provider = get_summary_provider()
        if settings.summary.generation_enabled:
            if summary_provider is not None:
                summary_stats = await repos.statistics.get_summary_statistics()
                stats['summary'] = {
                    'enabled': True,
                    'available': True,
                    'provider': settings.summary.provider,
                    'model': settings.summary.model,
                    'summary_count': summary_stats['summary_count'],
                    'coverage_percentage': summary_stats['coverage_percentage'],
                    'min_content_length': settings.summary.min_content_length,
                }
            else:
                stats['summary'] = {
                    'enabled': True,
                    'available': False,
                    'message': 'Summary provider not initialized',
                }
        else:
            stats['summary'] = {
                'enabled': False,
                'available': False,
            }

        # Add compression configuration block. Sources provider/bits/variant/
        # seed/dim from the singleton compression_metadata row (DB-truth);
        # max_concurrent from runtime settings. read_compression_metadata
        # returns None gracefully when the table is absent (pre-bootstrap)
        # on both backends, so the call is safe unconditionally.
        try:
            from app.compression.provenance import read_compression_metadata

            if settings.compression.enabled:
                db_meta = await read_compression_metadata(manager)
                if db_meta is not None:
                    stats['compression'] = {
                        'enabled': True,
                        'available': True,
                        'provider': db_meta.provider,
                        'bits': db_meta.bits,
                        'variant': db_meta.variant,
                        'seed': db_meta.seed,
                        'dim': db_meta.dim,
                        'max_concurrent': settings.compression.max_concurrent,
                    }
                else:
                    stats['compression'] = {
                        'enabled': True,
                        'available': False,
                        'message': 'Compression enabled but provenance not bootstrapped',
                    }
            else:
                stats['compression'] = {
                    'enabled': False,
                    'available': False,
                }
        except Exception as e:  # pragma: no cover -- defensive fallback
            logger.warning(f'Failed to read compression metadata for statistics: {e}')
            stats['compression'] = {
                'enabled': settings.compression.enabled,
                'available': False,
                'message': 'Failed to read compression metadata',
            }

        # Add index_tree node-summary block. Gated on the per-node summary
        # toggle; node_count is the total stored per-node summaries (0 when the
        # table is absent, e.g. the feature was never enabled).
        if settings.index_tree.node_summaries_enabled:
            try:
                node_count = await repos.index_nodes.count_all_nodes()
            except Exception as e:  # pragma: no cover -- defensive fallback
                logger.warning(f'Failed to read index_tree node count for statistics: {e}')
                node_count = 0
            stats['index_tree'] = {
                'enabled': True,
                'node_count': node_count,
            }
        else:
            stats['index_tree'] = {
                'enabled': False,
                'node_count': 0,
            }

        # stats is built incrementally as a plain dict so each sub-block
        # assignment stays readable; StatisticsResponseDict declares the
        # contract consumers see. The cast at the boundary applies the
        # typed contract without forcing per-block TypedDict locals.
        return cast(StatisticsResponseDict, stats)
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error getting statistics: {e}')
        raise ToolError(f'Failed to get statistics: {format_exception_message(e)}') from e
