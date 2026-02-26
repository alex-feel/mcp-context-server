"""
Search operations for MCP tools.

This module contains all search-related tools:
- search_context: Keyword-based filtering with metadata support
- semantic_search_context: Vector similarity search using embeddings
- fts_search_context: Full-text search with linguistic analysis
- hybrid_search_context: Combined FTS + semantic search with RRF fusion
"""

import asyncio
import json
import logging
from collections.abc import Coroutine
from datetime import UTC
from datetime import datetime
from typing import Annotated
from typing import Any
from typing import Literal
from typing import cast

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from app.migrations import format_exception_message
from app.migrations import get_fts_migration_status
from app.services.passage_extraction_service import extract_rerank_passage
from app.settings import get_settings
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.startup import get_reranking_provider
from app.startup.validation import truncate_text
from app.startup.validation import validate_date_param
from app.startup.validation import validate_date_range
from app.types import ContextEntryDict

logger = logging.getLogger(__name__)
settings = get_settings()


async def _apply_reranking(
    query: str,
    results: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Apply cross-encoder reranking to search results.

    This is a helper function used by search tools to rerank results
    after initial retrieval (semantic search, FTS, or hybrid).

    Args:
        query: The search query to score results against.
        results: Search results with 'id' and 'text_content' fields.
        limit: Maximum number of results to return after reranking.

    Returns:
        Reranked results with 'rerank_score' injected into the 'scores' object.
        If reranking is disabled or unavailable, returns original results unchanged.
    """
    # Check if reranking is available
    reranking_provider = get_reranking_provider()
    if reranking_provider is None or not settings.reranking.enabled:
        # Return original results (no reranking)
        return results[:limit] if limit else results

    if not results:
        return results

    try:
        # Map rerank_text (if available) or text_content to text for reranking provider
        # FTS results have 'rerank_text' (extracted passage around matches)
        # Semantic results may have 'rerank_text' (matched chunk) in future
        # Fallback to text_content if neither is available
        rerank_input: list[dict[str, Any]] = []
        for result in results:
            # Prefer rerank_text (passage/chunk), fall back to text_content (full document)
            rerank_text = result.get('rerank_text') or result.get('text_content', '')
            rerank_item: dict[str, Any] = {
                'id': result.get('id'),
                'text': rerank_text,
            }
            rerank_input.append(rerank_item)

        # Call reranking provider
        reranked = await reranking_provider.rerank(
            query=query,
            results=rerank_input,
            limit=limit,
        )

        # Build result lookup by ID for fast access
        result_by_id: dict[int, dict[str, Any]] = {
            int(r.get('id', 0)): r for r in results if r.get('id') is not None
        }

        # Merge rerank scores back into original results (inject into scores object)
        final_results: list[dict[str, Any]] = []
        for reranked_item in reranked:
            item_id = reranked_item.get('id')
            if item_id is not None and int(item_id) in result_by_id:
                merged = result_by_id[int(item_id)].copy()
                # Inject rerank_score into scores object
                if 'scores' in merged and isinstance(merged['scores'], dict):
                    merged['scores'] = merged['scores'].copy()
                    merged['scores']['rerank_score'] = reranked_item.get('rerank_score', 0.0)
                else:
                    # Create scores object if not present (shouldn't happen with updated tools)
                    merged['scores'] = {'rerank_score': reranked_item.get('rerank_score', 0.0)}
                final_results.append(merged)

        logger.debug(
            f'Reranked {len(results)} results to {len(final_results)} '
            f'(limit={limit}, provider={reranking_provider.provider_name})',
        )
        return final_results

    except Exception as e:
        logger.warning(f'Reranking failed, returning original results: {e}')
        # Fallback: return original results without reranking
        return results[:limit] if limit else results


async def _semantic_search_raw(
    query: str,
    limit: int,
    offset: int = 0,
    thread_id: str | None = None,
    source: Literal['user', 'agent'] | None = None,
    content_type: Literal['text', 'multimodal'] | None = None,
    tags: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
    metadata_filters: list[dict[str, Any]] | None = None,
    _extract_rerank_text: bool = False,
    explain_query: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Raw semantic search without reranking (Layer 1).

    This is the core semantic search implementation used by both
    semantic_search_context (with reranking) and hybrid_search_context
    (with RRF fusion and single reranking at the end).

    Args:
        query: Natural language search query.
        limit: Maximum results to return.
        offset: Pagination offset.
        thread_id: Optional filter by thread.
        source: Optional filter by source type.
        content_type: Optional filter by content type.
        tags: Optional filter by tags (OR logic).
        start_date: Optional filter by created_at >= date (ISO 8601 string).
        end_date: Optional filter by created_at <= date (ISO 8601 string).
        metadata: Simple metadata filters.
        metadata_filters: Advanced metadata filters.
        _extract_rerank_text: If True, extract matched chunk as rerank_text (internal use).
        explain_query: Include query execution statistics.

    Returns:
        Tuple of (results, stats). Results include text_content and distance.
        If _extract_rerank_text is True, results also include rerank_text.

    Raises:
        ToolError: If semantic search is not available or fails.
        MetadataFilterValidationError: If metadata filters are invalid.
    """
    # Check if semantic search is available
    embedding_provider = get_embedding_provider()
    if embedding_provider is None:
        from app.embeddings.factory import PROVIDER_INSTALL_INSTRUCTIONS

        provider = settings.embedding.provider
        install_cmd = PROVIDER_INSTALL_INSTRUCTIONS.get(provider, 'uv sync --extra embeddings-ollama')

        error_msg = (
            'Semantic search is not available. '
            f'Ensure ENABLE_SEMANTIC_SEARCH=true and {provider} provider is properly configured. '
            f'Install provider: {install_cmd}'
        )
        if provider == 'ollama':
            error_msg += f'. Download model: ollama pull {settings.embedding.model}'
        raise ToolError(error_msg)

    # Get repositories
    repos = await ensure_repositories()

    # Generate embedding for query
    try:
        query_embedding = await embedding_provider.embed_query(query)
    except Exception as e:
        logger.error(f'Failed to generate query embedding: {e}')
        raise ToolError(f'Failed to generate embedding for query: {str(e)}') from e

    # Perform similarity search with optional filtering
    from app.repositories.embedding_repository import MetadataFilterValidationError

    try:
        search_results, search_stats = await repos.embeddings.search(
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
    except MetadataFilterValidationError:
        raise  # Let caller handle validation errors
    except Exception as e:
        logger.error(f'Semantic search failed: {e}')
        raise ToolError(f'Semantic search failed: {format_exception_message(e)}') from e

    # Post-process for reranking: extract chunk text from boundaries
    if _extract_rerank_text:
        for result in search_results:
            text_content = result.get('text_content', '')
            start_idx = result.get('matched_chunk_start')
            end_idx = result.get('matched_chunk_end')

            if start_idx is not None and end_idx is not None and end_idx > start_idx:
                # Extract matched chunk for reranking
                result['rerank_text'] = text_content[start_idx:end_idx]
                logger.debug(
                    f'[SEMANTIC] Extracted rerank_text: {len(result["rerank_text"])} chars '
                    f'from [{start_idx}:{end_idx}]',
                )
            else:
                # Fallback: use beginning of document (legacy data without boundaries)
                max_rerank_len = settings.reranking.max_length * 4  # ~2000 chars
                result['rerank_text'] = text_content[:max_rerank_len]
                logger.debug(
                    f'[SEMANTIC] No chunk boundaries, using document beginning '
                    f'({len(result["rerank_text"])} chars)',
                )

            # Remove internal boundary fields from result
            result.pop('matched_chunk_start', None)
            result.pop('matched_chunk_end', None)

    return search_results, search_stats


async def _fts_search_raw(
    query: str,
    limit: int,
    mode: Literal['match', 'prefix', 'phrase', 'boolean'] = 'match',
    offset: int = 0,
    thread_id: str | None = None,
    source: Literal['user', 'agent'] | None = None,
    content_type: Literal['text', 'multimodal'] | None = None,
    tags: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
    metadata_filters: list[dict[str, Any]] | None = None,
    highlight: bool = False,
    _internal_highlight_for_rerank: bool = False,
    explain_query: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Raw full-text search without reranking (Layer 1).

    This is the core FTS implementation used by both
    fts_search_context (with reranking) and hybrid_search_context
    (with RRF fusion and single reranking at the end).

    Args:
        query: Full-text search query.
        limit: Maximum results to return.
        mode: Search mode (match, prefix, phrase, boolean).
        offset: Pagination offset.
        thread_id: Optional filter by thread.
        source: Optional filter by source type.
        content_type: Optional filter by content type.
        tags: Optional filter by tags (OR logic).
        start_date: Optional filter by created_at >= date (ISO 8601 string).
        end_date: Optional filter by created_at <= date (ISO 8601 string).
        metadata: Simple metadata filters.
        metadata_filters: Advanced metadata filters.
        highlight: Include highlighted snippets in client response.
        _internal_highlight_for_rerank: Generate highlights internally for passage extraction.
        explain_query: Include query execution statistics.

    Returns:
        Tuple of (results, stats). Results include text_content and score.
        When _internal_highlight_for_rerank=True, results also include 'rerank_text' field.

    Raises:
        ToolError: If FTS is not available or fails.
        FtsValidationError: If query or filters are invalid.
    """
    # Check if FTS is enabled
    if not settings.fts.enabled:
        raise ToolError(
            'Full-text search is not available. '
            'Set ENABLE_FTS=true to enable this feature.',
        )

    # Check if migration is in progress
    fts_status = get_fts_migration_status()
    if fts_status.in_progress:
        raise ToolError('FTS migration in progress. Please retry shortly.')

    # Get repositories
    repos = await ensure_repositories()

    # Check if FTS is properly initialized
    if not await repos.fts.is_available():
        raise ToolError(
            'FTS index not found. The database may need migration. '
            'Restart the server with ENABLE_FTS=true to apply migrations.',
        )

    # Import exception here to avoid circular imports
    from app.repositories.fts_repository import FtsValidationError

    # Determine actual highlight setting
    # Generate highlights if client requested OR if we need for passage extraction
    actual_highlight = highlight or _internal_highlight_for_rerank

    try:
        search_results, stats = await repos.fts.search(
            query=query,
            mode=mode,
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
            highlight=actual_highlight,
            language=settings.fts.language,
            explain_query=explain_query,
        )
    except FtsValidationError:
        raise  # Let caller handle validation errors
    except Exception as e:
        logger.error(f'FTS search failed: {e}')
        raise ToolError(f'FTS search failed: {format_exception_message(e)}') from e

    # Post-process for reranking: extract passages from highlighted results
    if _internal_highlight_for_rerank:
        for result in search_results:
            highlighted = result.get('highlighted')
            text_content = result.get('text_content', '')

            # Extract passage for reranking using highlight positions
            result['rerank_text'] = extract_rerank_passage(
                text_content=text_content,
                highlighted=highlighted,
                window_size=settings.fts_passage.rerank_window_size,
                max_passage_size=settings.reranking.max_length * 4,  # ~2000 chars for 512 tokens
                gap_merge_threshold=settings.fts_passage.rerank_gap_merge,
            )

            # Remove 'highlighted' if client didn't request it
            if not highlight:
                result.pop('highlighted', None)

    return search_results, stats


async def search_context(
    limit: Annotated[int, Field(ge=1, le=100, description='Maximum results to return (1-100, default: 30)')] = 30,
    thread_id: Annotated[str | None, Field(min_length=1, description='Filter by thread (indexed)')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Filter by source type (indexed)')] = None,
    tags: Annotated[list[str] | None, Field(description='Filter by any of these tags (OR logic)')] = None,
    content_type: Annotated[Literal['text', 'multimodal'] | None, Field(description='Filter by content type')] = None,
    metadata: Annotated[
        dict[str, str | int | float | bool] | None,
        Field(description='Simple metadata filters (key=value equality)'),
    ] = None,
    metadata_filters: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description='Advanced metadata filters: [{"key": "priority", "operator": "gt", "value": 5}]. '
            'Operators: eq, ne, gt, gte, lt, lte, in, not_in, exists, not_exists, contains, '
            'starts_with, ends_with, is_null, is_not_null, array_contains',
        ),
    ] = None,
    start_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at >= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T10:00:00")',
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at <= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T23:59:59")',
        ),
    ] = None,
    offset: Annotated[int, Field(ge=0, description='Pagination offset (default: 0)')] = 0,
    include_images: Annotated[bool, Field(description='Include image data (only for multimodal entries)')] = False,
    explain_query: Annotated[bool, Field(description='Include query execution statistics')] = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Search context entries with filtering. Returns TRUNCATED text_content (150 chars max).

    Use get_context_by_ids to retrieve full content for specific entries of interest.

    Filtering options:
    - thread_id, source: Indexed for fast filtering (always prefer specifying thread_id)
    - tags: OR logic (matches ANY of provided tags)
    - metadata: Simple key=value equality matching
    - metadata_filters: Advanced operators (gt, lt, contains, exists, etc.)
    - start_date/end_date: Filter by creation timestamp (ISO 8601)

    Performance tips:
    - Always specify thread_id to reduce search space
    - Use indexed metadata fields: status, agent_name, task_name, project, report_type
      (PostgreSQL also indexes: references, technologies)

    Returns:
        Dict with results (list of ContextEntryDict), count (int), and
        stats (dict, only when explain_query=True).

    Raises:
        ToolError: If search operation fails.
    """
    try:
        # Validate date parameters
        start_date = validate_date_param(start_date, 'start_date')
        end_date = validate_date_param(end_date, 'end_date')
        validate_date_range(start_date, end_date)

        if ctx:
            await ctx.info(f'Searching context with filters: thread_id={thread_id}, source={source}')

        # Get repositories
        repos = await ensure_repositories()

        # Use the improved search_contexts method that now supports metadata and date filtering
        result = await repos.context.search_contexts(
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            tags=tags,
            metadata=metadata,
            metadata_filters=metadata_filters,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
            explain_query=explain_query,
        )

        # Always expect tuple from repository
        rows, stats = result

        # Check for validation errors in stats
        if 'error' in stats:
            # Return the error response with validation details
            error_response: dict[str, Any] = {
                'results': [],
                'count': 0,
                'error': stats.get('error', 'Unknown error'),
            }
            if 'validation_errors' in stats:
                error_response['validation_errors'] = stats['validation_errors']
            return error_response

        entries: list[ContextEntryDict] = []

        for row in rows:
            # Create entry dict with proper typing for dynamic fields
            entry = cast(ContextEntryDict, dict(row))

            # Parse JSON metadata - database stores as JSON string
            metadata_raw = entry.get('metadata')
            # Database can return string that needs parsing
            # Using hasattr to check for string-like object avoids unreachable code warning
            if metadata_raw is not None and hasattr(metadata_raw, 'strip'):  # String-like object from DB
                try:
                    entry['metadata'] = json.loads(str(metadata_raw))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    entry['metadata'] = None

            # Get normalized tags
            entry_id_raw = entry.get('id')
            if entry_id_raw is not None:
                entry_id = int(entry_id_raw)
                tags_result = await repos.tags.get_tags_for_context(entry_id)
                entry['tags'] = tags_result
            else:
                entry['tags'] = []

            # Apply text truncation for search_context
            text_content = entry.get('text_content', '')
            truncated_text, is_truncated = truncate_text(text_content)
            entry['text_content'] = truncated_text
            entry['is_truncated'] = is_truncated

            # Fetch images if requested and applicable
            if include_images and entry.get('content_type') == 'multimodal':
                entry_id = int(entry.get('id', 0))
                images_result = await repos.images.get_images_for_context(entry_id, include_data=True)
                entry['images'] = cast(list[dict[str, str]], images_result)

            entries.append(entry)

        # Return dict with results, count, and optional stats
        response: dict[str, Any] = {'results': entries, 'count': len(entries)}
        if explain_query:
            response['stats'] = stats
        return response
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error searching context: {e}')
        raise ToolError(f'Failed to search context: {str(e)}') from e


async def semantic_search_context(
    query: Annotated[str, Field(min_length=1, description='Natural language search query')],
    limit: Annotated[int, Field(ge=1, le=100, description='Maximum results to return (1-100, default: 5)')] = 5,
    offset: Annotated[int, Field(ge=0, description='Pagination offset (default: 0)')] = 0,
    thread_id: Annotated[str | None, Field(min_length=1, description='Optional filter by thread')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Optional filter by source type')] = None,
    content_type: Annotated[
        Literal['text', 'multimodal'] | None, Field(description='Filter by content type (text or multimodal)'),
    ] = None,
    tags: Annotated[list[str] | None, Field(description='Filter by any of these tags (OR logic)')] = None,
    start_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at >= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T10:00:00")',
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at <= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T23:59:59")',
        ),
    ] = None,
    metadata: Annotated[
        dict[str, str | int | float | bool] | None,
        Field(description='Simple metadata filters (key=value equality)'),
    ] = None,
    metadata_filters: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description='Advanced metadata filters: [{"key": "priority", "operator": "gt", "value": 5}]. '
            'Operators: eq, ne, gt, gte, lt, lte, in, not_in, exists, not_exists, contains, '
            'starts_with, ends_with, is_null, is_not_null, array_contains',
        ),
    ] = None,
    include_images: Annotated[bool, Field(description='Include image data (only for multimodal entries)')] = False,
    explain_query: Annotated[bool, Field(description='Include query execution statistics')] = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Find semantically similar context using vector embeddings with optional metadata filtering.

    Unlike keyword search (search_context), this finds entries with similar MEANING
    even without matching keywords. Use for: finding related concepts, similar discussions,
    thematic grouping.

    Filtering options (all combinable):
    - thread_id/source: Basic entry filtering
    - content_type: Filter by text or multimodal entries
    - tags: OR logic (matches ANY of provided tags)
    - start_date/end_date: Date range filtering (ISO 8601)
    - metadata: Simple key=value equality matching
    - metadata_filters: Advanced operators (gt, lt, contains, exists, etc.)

    The `scores` object contains:
    - semantic_distance: L2 Euclidean distance (LOWER = more similar)
    - semantic_rank: Always null for standalone semantic search
    - rerank_score: Cross-encoder relevance (HIGHER = better), present when reranking enabled

    Typical distance interpretation: <0.5 very similar, 0.5-1.0 related, >1.0 less related.

    Returns:
        Dict with query (str), results (list with id, thread_id, source, text_content,
        metadata, scores, tags), count (int), model (str), and stats (only when explain_query=True).

        The `scores` field contains: semantic_distance, semantic_rank, rerank_score.

    Raises:
        ToolError: If semantic search is not available or search operation fails.
    """
    # Validate date parameters
    start_date = validate_date_param(start_date, 'start_date')
    end_date = validate_date_param(end_date, 'end_date')
    validate_date_range(start_date, end_date)

    try:
        if ctx:
            await ctx.info(f'Performing semantic search: "{query[:50]}..."')

        # Calculate overfetch limit for reranking
        # Over-fetch more results to give reranker better candidates
        reranking_provider = get_reranking_provider()
        need_reranking = reranking_provider is not None and settings.reranking.enabled
        overfetch_limit = (
            (limit + offset) * settings.reranking.overfetch if need_reranking else limit + offset
        )

        # Import exception here to avoid circular imports at module level
        from app.repositories.embedding_repository import MetadataFilterValidationError

        try:
            # Call raw search (Layer 1) with overfetch
            # Extract rerank_text (matched chunk) when reranking is enabled
            search_results, search_stats = await _semantic_search_raw(
                query=query,
                limit=overfetch_limit,
                offset=0,  # Offset handled after reranking
                thread_id=thread_id,
                source=source,
                content_type=content_type,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
                metadata=metadata,
                metadata_filters=metadata_filters,
                _extract_rerank_text=need_reranking,
                explain_query=explain_query,
            )
        except MetadataFilterValidationError as e:
            # Return error response (unified with search_context behavior)
            return {
                'query': query,
                'results': [],
                'count': 0,
                'model': settings.embedding.model,
                'error': e.message,
                'validation_errors': e.validation_errors,
            }

        # Transform results to use scores object (before reranking)
        for result in search_results:
            # Move distance into scores object
            distance_value = result.pop('distance', None)
            result['scores'] = {
                'semantic_distance': distance_value,
                'semantic_rank': None,  # Standalone semantic has no ranking
            }

        # Apply reranking (Layer 2) if available
        reranked_results = await _apply_reranking(
            query=query,
            results=search_results,
            limit=limit + offset,  # Get enough for offset + limit
        )

        # Apply offset after reranking
        final_results = reranked_results[offset:][:limit]

        # Clean up internal fields from final results
        for result in final_results:
            result.pop('rerank_text', None)  # Internal field, not for client

        # Enrich results with tags and optionally images
        repos = await ensure_repositories()
        for result in final_results:
            context_id = result.get('id')
            if context_id:
                tags_result = await repos.tags.get_tags_for_context(int(context_id))
                result['tags'] = tags_result
                # Fetch images if requested and applicable
                if include_images and result.get('content_type') == 'multimodal':
                    images_result = await repos.images.get_images_for_context(int(context_id), include_data=True)
                    result['images'] = images_result
            else:
                result['tags'] = []

        logger.info(f'Semantic search found {len(final_results)} results for query: "{query[:50]}..."')

        response: dict[str, Any] = {
            'query': query,
            'results': final_results,
            'count': len(final_results),
            'model': settings.embedding.model,
        }
        if explain_query:
            response['stats'] = search_stats
        return response

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error in semantic search: {e}')
        raise ToolError(f'Semantic search failed: {format_exception_message(e)}') from e


async def fts_search_context(
    query: Annotated[str, Field(min_length=1, description='Full-text search query')],
    limit: Annotated[int, Field(ge=1, le=100, description='Maximum results to return (1-100, default: 5)')] = 5,
    mode: Annotated[
        Literal['match', 'prefix', 'phrase', 'boolean'],
        Field(
            description="Search mode: 'match' (default, natural language), "
            "'prefix' (wildcard with *), 'phrase' (exact phrase), "
            "'boolean' (AND/OR/NOT operators)",
        ),
    ] = 'match',
    thread_id: Annotated[str | None, Field(min_length=1, description='Optional filter by thread')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Optional filter by source type')] = None,
    content_type: Annotated[
        Literal['text', 'multimodal'] | None, Field(description='Filter by content type (text or multimodal)'),
    ] = None,
    tags: Annotated[list[str] | None, Field(description='Filter by any of these tags (OR logic)')] = None,
    start_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at >= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T10:00:00")',
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at <= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T23:59:59")',
        ),
    ] = None,
    metadata: Annotated[
        dict[str, str | int | float | bool] | None,
        Field(description='Simple metadata filters (key=value equality)'),
    ] = None,
    metadata_filters: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description='Advanced metadata filters: [{"key": "priority", "operator": "gt", "value": 5}]. '
            'Operators: eq, ne, gt, gte, lt, lte, in, not_in, exists, not_exists, contains, '
            'starts_with, ends_with, is_null, is_not_null, array_contains',
        ),
    ] = None,
    offset: Annotated[int, Field(ge=0, description='Pagination offset (default: 0)')] = 0,
    highlight: Annotated[bool, Field(description='Include highlighted snippets in results')] = False,
    include_images: Annotated[bool, Field(description='Include image data (only for multimodal entries)')] = False,
    explain_query: Annotated[bool, Field(description='Include query execution statistics')] = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Full-text search with linguistic analysis (stemming, ranking, boolean queries).

    Unlike keyword filtering (search_context) or semantic similarity (semantic_search_context),
    FTS provides:
    - Stemming: "running" matches "run", "runs", "runner"
    - Stop word handling: common words like "the", "is" are ignored
    - Boolean operators: AND, OR, NOT for precise queries
    - BM25/ts_rank relevance scoring

    Search modes:
    - match: Natural language query (default) - words are stemmed and matched
    - prefix: Wildcard search - "search*" matches "searching", "searched"
    - phrase: Exact phrase matching - "exact phrase" must appear as-is
    - boolean: Boolean operators - "python AND (async OR await) NOT blocking"

    Filtering options (all combinable):
    - thread_id/source: Basic entry filtering
    - content_type: Filter by text or multimodal entries
    - tags: OR logic (matches ANY of provided tags)
    - start_date/end_date: Date range filtering (ISO 8601)
    - metadata: Simple key=value equality matching
    - metadata_filters: Advanced operators (gt, lt, contains, exists, etc.)

    The `scores` object contains:
    - fts_score: BM25/ts_rank relevance (HIGHER = better match)
    - fts_rank: Always null for standalone FTS search
    - rerank_score: Cross-encoder relevance (HIGHER = better), present when reranking enabled

    Returns:
        Dict with query (str), mode (str), results (list with id, thread_id, source,
        text_content, metadata, scores, highlighted, tags), count (int), language (str),
        and stats (only when explain_query=True).

        The `scores` field contains: fts_score, fts_rank, rerank_score.

    Raises:
        ToolError: If FTS is not available or search operation fails.
    """
    # Validate date parameters
    start_date = validate_date_param(start_date, 'start_date')
    end_date = validate_date_param(end_date, 'end_date')
    validate_date_range(start_date, end_date)

    # Check if migration is in progress - return informative response for graceful degradation
    fts_status = get_fts_migration_status()
    if fts_status.in_progress:
        if fts_status.started_at is not None and fts_status.estimated_seconds is not None:
            elapsed = (datetime.now(tz=UTC) - fts_status.started_at).total_seconds()
            remaining = max(0, fts_status.estimated_seconds - int(elapsed))
        else:
            remaining = 60  # Default estimate if no timing info available

        old_lang = fts_status.old_language or 'unknown'
        new_lang = fts_status.new_language or settings.fts.language

        return {
            'migration_in_progress': True,
            'message': f'FTS index is being rebuilt with language/tokenizer "{new_lang}". '
            'Search functionality will be available shortly.',
            'started_at': fts_status.started_at.isoformat() if fts_status.started_at else '',
            'estimated_remaining_seconds': remaining,
            'old_language': old_lang,
            'new_language': new_lang,
            'suggestion': f'Please retry in {remaining + 5} seconds.',
        }

    try:
        if ctx:
            await ctx.info(f'Performing FTS search: "{query[:50]}..." (mode={mode})')

        # Calculate overfetch limit for reranking
        reranking_provider = get_reranking_provider()
        if reranking_provider is not None and settings.reranking.enabled:
            overfetch_limit = (limit + offset) * settings.reranking.overfetch
        else:
            overfetch_limit = limit + offset

        # Determine if we need highlights for internal reranking
        need_highlight_for_rerank = reranking_provider is not None and settings.reranking.enabled

        # Import exception here to avoid circular imports
        from app.repositories.fts_repository import FtsValidationError

        try:
            # Call raw search (Layer 1) with overfetch
            search_results, stats = await _fts_search_raw(
                query=query,
                limit=overfetch_limit,
                mode=mode,
                offset=0,  # Offset handled after reranking
                thread_id=thread_id,
                source=source,
                content_type=content_type,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
                metadata=metadata,
                metadata_filters=metadata_filters,
                highlight=highlight,
                _internal_highlight_for_rerank=need_highlight_for_rerank,
                explain_query=explain_query,
            )
        except FtsValidationError as e:
            # Return error response (unified with search_context behavior)
            error_response: dict[str, Any] = {
                'query': query,
                'mode': mode,
                'results': [],
                'count': 0,
                'language': settings.fts.language,
                'error': e.message,
                'validation_errors': e.validation_errors,
            }
            if explain_query:
                error_response['stats'] = {
                    'execution_time_ms': 0.0,
                    'filters_applied': 0,
                    'rows_returned': 0,
                }
            return error_response

        # Transform results to use scores object (before reranking)
        for result in search_results:
            # Move score into scores object
            fts_score_value = result.pop('score', None)
            result['scores'] = {
                'fts_score': fts_score_value,
                'fts_rank': None,  # Standalone FTS has no ranking
            }

        # Apply reranking (Layer 2) if available
        reranked_results = await _apply_reranking(
            query=query,
            results=search_results,
            limit=limit + offset,  # Get enough for offset + limit
        )

        # Apply offset after reranking
        final_results = reranked_results[offset:][:limit]

        # Clean up internal fields from final results
        for result in final_results:
            result.pop('rerank_text', None)  # Internal field, not for client

        # Process results: parse metadata and enrich with tags
        repos = await ensure_repositories()
        for result in final_results:
            # Parse JSON metadata - database stores as JSON string
            metadata_raw = result.get('metadata')
            # Database can return string that needs parsing
            # Using hasattr to check for string-like object avoids unreachable code warning
            if metadata_raw is not None and hasattr(metadata_raw, 'strip'):  # String-like object from DB
                try:
                    result['metadata'] = json.loads(str(metadata_raw))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    result['metadata'] = None

            # Get normalized tags
            context_id = result.get('id')
            if context_id:
                tags_result = await repos.tags.get_tags_for_context(int(context_id))
                result['tags'] = tags_result
                # Fetch images if requested and applicable
                if include_images and result.get('content_type') == 'multimodal':
                    images_result = await repos.images.get_images_for_context(int(context_id), include_data=True)
                    result['images'] = images_result
            else:
                result['tags'] = []

        logger.info(f'FTS search found {len(final_results)} results for query: "{query[:50]}..."')

        response: dict[str, Any] = {
            'query': query,
            'mode': mode,
            'results': final_results,
            'count': len(final_results),
            'language': settings.fts.language,
        }
        if explain_query:
            response['stats'] = stats
        return response

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error in FTS search: {e}')
        raise ToolError(f'FTS search failed: {format_exception_message(e)}') from e


async def hybrid_search_context(
    query: Annotated[str, Field(min_length=1, description='Natural language search query')],
    limit: Annotated[int, Field(ge=1, le=100, description='Maximum results to return (1-100, default: 5)')] = 5,
    offset: Annotated[int, Field(ge=0, description='Pagination offset (default: 0)')] = 0,
    fusion_method: Annotated[
        Literal['rrf'],
        Field(description="Fusion algorithm: 'rrf' (Reciprocal Rank Fusion, default)"),
    ] = 'rrf',
    rrf_k: Annotated[
        int | None,
        Field(
            ge=1,
            le=1000,
            description='RRF smoothing constant (default from HYBRID_RRF_K env var, typically 60). '
            'Higher values give more weight to lower-ranked documents.',
        ),
    ] = None,
    thread_id: Annotated[str | None, Field(min_length=1, description='Optional filter by thread')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Optional filter by source type')] = None,
    content_type: Annotated[
        Literal['text', 'multimodal'] | None, Field(description='Filter by content type (text or multimodal)'),
    ] = None,
    tags: Annotated[list[str] | None, Field(description='Filter by any of these tags (OR logic)')] = None,
    start_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at >= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T10:00:00")',
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at <= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T23:59:59")',
        ),
    ] = None,
    metadata: Annotated[
        dict[str, str | int | float | bool] | None,
        Field(description='Simple metadata filters (key=value equality)'),
    ] = None,
    metadata_filters: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description='Advanced metadata filters: [{"key": "priority", "operator": "gt", "value": 5}]. '
            'Operators: eq, ne, gt, gte, lt, lte, in, not_in, exists, not_exists, contains, '
            'starts_with, ends_with, is_null, is_not_null, array_contains',
        ),
    ] = None,
    include_images: Annotated[bool, Field(description='Include image data (only for multimodal entries)')] = False,
    explain_query: Annotated[bool, Field(description='Include query execution statistics')] = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Hybrid search combining FTS and semantic search with Reciprocal Rank Fusion (RRF).

    Executes both full-text search and semantic search in parallel, then fuses results
    using RRF algorithm. Documents appearing in both result sets score higher.

    RRF Formula: score(d) = sum(1 / (k + rank_i(d))) for each search method i

    Graceful degradation:
    - If only FTS is available, returns FTS results only
    - If only semantic search is available, returns semantic results only
    - If neither is available, raises ToolError

    Filtering options (all combinable):
    - thread_id/source: Basic entry filtering
    - content_type: Filter by text or multimodal entries
    - tags: OR logic (matches ANY of provided tags)
    - start_date/end_date: Date range filtering (ISO 8601)
    - metadata: Simple key=value equality matching
    - metadata_filters: Advanced operators (gt, lt, contains, exists, etc.)

    The `scores` field contains: rrf (combined), fts_rank, semantic_rank,
    fts_score, semantic_distance, rerank_score.

    When explain_query=True, the `stats` field contains:
    - execution_time_ms: Total hybrid search time
    - fts_stats: {execution_time_ms, filters_applied, rows_returned} or None
    - semantic_stats: {execution_time_ms, embedding_generation_ms, filters_applied, rows_returned} or None
    - fusion_stats: {rrf_k, total_unique_documents, documents_in_both, documents_fts_only, documents_semantic_only}

    Returns:
        Dict with query (str), results (list with id, thread_id, source, text_content,
        metadata, scores, tags), count (int), fusion_method (str), search_modes_used (list),
        fts_count (int), semantic_count (int), and stats (only when explain_query=True).

    Raises:
        ToolError: If hybrid search is not available or all search modes fail.
    """
    # Validate date parameters
    start_date = validate_date_param(start_date, 'start_date')
    end_date = validate_date_param(end_date, 'end_date')
    validate_date_range(start_date, end_date)

    # Check if hybrid search is enabled
    if not settings.hybrid_search.enabled:
        raise ToolError(
            'Hybrid search is not available. '
            'Set ENABLE_HYBRID_SEARCH=true to enable this feature. '
            'Also ensure ENABLE_FTS=true and/or ENABLE_SEMANTIC_SEARCH=true.',
        )

    # Use settings default if rrf_k not specified
    effective_rrf_k = rrf_k if rrf_k is not None else settings.hybrid_search.rrf_k

    # Determine available search modes
    fts_available = settings.fts.enabled
    semantic_available = settings.semantic_search.enabled and get_embedding_provider() is not None

    # Collect all available modes
    available_modes: list[str] = []
    if fts_available:
        available_modes.append('fts')
    if semantic_available:
        available_modes.append('semantic')

    if not available_modes:
        unavailable_reasons: list[str] = []
        if not fts_available:
            unavailable_reasons.append('FTS requires ENABLE_FTS=true')
        if not semantic_available:
            unavailable_reasons.append(
                f'Semantic search requires ENABLE_SEMANTIC_SEARCH=true and '
                f'{settings.embedding.provider} provider properly configured',
            )
        raise ToolError(
            f'No search modes available. '
            f'Issues: {"; ".join(unavailable_reasons)}',
        )

    try:
        import time as time_module

        total_start_time = time_module.time()

        if ctx:
            await ctx.info(f'Performing hybrid search: "{query[:50]}..." (modes={available_modes})')

        # Import fusion module
        from app.fusion import count_unique_results
        from app.fusion import reciprocal_rank_fusion

        # Get repositories for tag/image enrichment
        repos = await ensure_repositories()

        # Calculate overfetch limit for hybrid search with reranking
        # Chain: limit * hybrid_rrf_overfetch * reranking.overfetch
        reranking_provider = get_reranking_provider()
        if reranking_provider is not None and settings.reranking.enabled:
            over_fetch_limit = (limit + offset) * settings.hybrid_search.rrf_overfetch * settings.reranking.overfetch
        else:
            over_fetch_limit = (limit + offset) * settings.hybrid_search.rrf_overfetch

        # Determine if we need highlights for internal reranking
        need_highlight_for_rerank = reranking_provider is not None and settings.reranking.enabled

        # Execute searches in parallel using raw functions (Layer 1 - no reranking)
        fts_results: list[dict[str, Any]] = []
        semantic_results: list[dict[str, Any]] = []
        fts_error: str | None = None
        semantic_error: str | None = None

        # Stats collection for explain_query
        fts_stats: dict[str, Any] | None = None
        semantic_stats: dict[str, Any] | None = None

        async def run_fts_search() -> None:
            nonlocal fts_results, fts_error, fts_stats
            try:
                results, stats = await _fts_search_raw(
                    query=query,
                    limit=over_fetch_limit,
                    mode='match',
                    offset=0,
                    thread_id=thread_id,
                    source=source,
                    content_type=content_type,
                    tags=tags,
                    start_date=start_date,
                    end_date=end_date,
                    metadata=metadata,
                    metadata_filters=metadata_filters,
                    highlight=False,  # Client doesn't want highlighted field in hybrid
                    _internal_highlight_for_rerank=need_highlight_for_rerank,
                    explain_query=explain_query,
                )
                fts_results = results
                if explain_query:
                    fts_stats = stats
            except ToolError as e:
                fts_error = str(e)
            except Exception as e:
                fts_error = str(e)

        async def run_semantic_search() -> None:
            nonlocal semantic_results, semantic_error, semantic_stats
            try:
                results, stats = await _semantic_search_raw(
                    query=query,
                    limit=over_fetch_limit,
                    offset=0,
                    thread_id=thread_id,
                    source=source,
                    content_type=content_type,
                    tags=tags,
                    start_date=start_date,
                    end_date=end_date,
                    metadata=metadata,
                    metadata_filters=metadata_filters,
                    _extract_rerank_text=need_highlight_for_rerank,
                    explain_query=explain_query,
                )
                semantic_results = results
                if explain_query:
                    semantic_stats = stats
            except ToolError as e:
                semantic_error = str(e)
            except Exception as e:
                semantic_error = str(e)

        # Run searches in parallel
        tasks: list[Coroutine[Any, Any, None]] = []
        if 'fts' in available_modes:
            tasks.append(run_fts_search())
        if 'semantic' in available_modes:
            tasks.append(run_semantic_search())

        await asyncio.gather(*tasks)

        # Check if both searches failed
        if fts_error and semantic_error:
            raise ToolError(
                f'All search modes failed. FTS: {fts_error}. Semantic: {semantic_error}',
            )

        # Determine which modes actually returned results
        modes_used: list[str] = []
        if fts_results:
            modes_used.append('fts')
        if semantic_results:
            modes_used.append('semantic')

        # Parse FTS metadata (returned as JSON strings from DB)
        for result in fts_results:
            metadata_raw = result.get('metadata')
            if metadata_raw is not None and hasattr(metadata_raw, 'strip'):
                try:
                    result['metadata'] = json.loads(str(metadata_raw))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    result['metadata'] = None

        # Fuse results using RRF (no reranking yet - Layer 1 results only)
        # Get more than needed for reranking
        fused_limit_for_reranking = (limit + offset) * (
            settings.reranking.overfetch if reranking_provider is not None and settings.reranking.enabled else 1
        )
        fused_results = reciprocal_rank_fusion(
            fts_results=fts_results,
            semantic_results=semantic_results,
            k=effective_rrf_k,
            limit=fused_limit_for_reranking,
        )

        # Apply reranking (Layer 3 - single reranking after fusion)
        fused_results_any: list[dict[str, Any]] = cast(list[dict[str, Any]], fused_results)
        reranked_results = await _apply_reranking(
            query=query,
            results=fused_results_any,
            limit=limit + offset,  # Get enough for offset + limit
        )

        # Apply offset after reranking
        final_results = reranked_results[offset:][:limit]

        # Clean up internal fields from final results
        for result in final_results:
            result.pop('rerank_text', None)  # Internal field, not for client

        # Enrich results with tags and optionally images
        for result in final_results:
            context_id = result.get('id')
            if context_id:
                tags_result = await repos.tags.get_tags_for_context(int(context_id))
                result['tags'] = tags_result
                # Fetch images if requested and applicable
                if include_images and result.get('content_type') == 'multimodal':
                    images_result = await repos.images.get_images_for_context(int(context_id), include_data=True)
                    result['images'] = images_result
            else:
                result['tags'] = []

        logger.info(
            f'Hybrid search found {len(final_results)} results for query: "{query[:50]}..." '
            f'(fts={len(fts_results)}, semantic={len(semantic_results)}, modes={modes_used})',
        )

        # Build response
        response: dict[str, Any] = {
            'query': query,
            'results': final_results,
            'count': len(final_results),
            'fusion_method': fusion_method,
            'search_modes_used': modes_used,
            'fts_count': len(fts_results),
            'semantic_count': len(semantic_results),
        }

        # Add stats if explain_query is enabled
        if explain_query:
            # Calculate fusion stats
            fts_only, semantic_only, overlap = count_unique_results(fts_results, semantic_results)
            fusion_stats: dict[str, Any] = {
                'rrf_k': effective_rrf_k,
                'total_unique_documents': fts_only + semantic_only + overlap,
                'documents_in_both': overlap,
                'documents_fts_only': fts_only,
                'documents_semantic_only': semantic_only,
            }

            # Calculate total execution time
            total_execution_time_ms = (time_module.time() - total_start_time) * 1000

            response['stats'] = {
                'execution_time_ms': round(total_execution_time_ms, 2),
                'fts_stats': fts_stats,
                'semantic_stats': semantic_stats,
                'fusion_stats': fusion_stats,
            }

        return response

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error in hybrid search: {e}')
        raise ToolError(f'Hybrid search failed: {format_exception_message(e)}') from e
