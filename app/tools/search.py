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
from app.settings import get_settings
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.startup.validation import truncate_text
from app.startup.validation import validate_date_param
from app.startup.validation import validate_date_range
from app.types import ContextEntryDict

logger = logging.getLogger(__name__)
settings = get_settings()


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
    - Use indexed metadata fields: status, priority, agent_name, task_name, completed

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

    The `distance` field is L2 Euclidean distance - LOWER values mean HIGHER similarity.
    Typical interpretation: <0.5 very similar, 0.5-1.0 related, >1.0 less related.

    Returns:
        Dict with query (str), results (list with id, thread_id, source, text_content,
        metadata, distance, tags), count (int), model (str), and stats (only when explain_query=True).

    Raises:
        ToolError: If semantic search is not available or search operation fails.
    """
    # Validate date parameters
    start_date = validate_date_param(start_date, 'start_date')
    end_date = validate_date_param(end_date, 'end_date')
    validate_date_range(start_date, end_date)

    # Check if semantic search is available
    semantic_embedding_provider = get_embedding_provider()
    if semantic_embedding_provider is None:
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

    try:
        if ctx:
            await ctx.info(f'Performing semantic search: "{query[:50]}..."')

        # Get repositories
        repos = await ensure_repositories()

        # Generate embedding for query
        try:
            query_embedding = await semantic_embedding_provider.embed_query(query)
        except Exception as e:
            logger.error(f'Failed to generate query embedding: {e}')
            raise ToolError(f'Failed to generate embedding for query: {str(e)}') from e

        # Perform similarity search with optional filtering (date and metadata)
        # Import exception here to avoid circular imports at module level
        from app.repositories.embedding_repository import MetadataFilterValidationError

        try:
            # Unpack tuple; stats used when explain_query=True
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
        except Exception as e:
            logger.error(f'Semantic search failed: {e}')
            raise ToolError(f'Semantic search failed: {format_exception_message(e)}') from e

        # Enrich results with tags and optionally images
        for result in search_results:
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

        logger.info(f'Semantic search found {len(search_results)} results for query: "{query[:50]}..."')

        response: dict[str, Any] = {
            'query': query,
            'results': search_results,
            'count': len(search_results),
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

    The `score` field is relevance score - HIGHER values mean BETTER match.

    Returns:
        Dict with query (str), mode (str), results (list with id, thread_id, source,
        text_content, metadata, score, highlighted, tags), count (int), language (str),
        and stats (only when explain_query=True).

    Raises:
        ToolError: If FTS is not available or search operation fails.
    """
    # Validate date parameters
    start_date = validate_date_param(start_date, 'start_date')
    end_date = validate_date_param(end_date, 'end_date')
    validate_date_range(start_date, end_date)

    # Check if FTS is enabled
    if not settings.enable_fts:
        raise ToolError(
            'Full-text search is not available. '
            'Set ENABLE_FTS=true to enable this feature.',
        )

    # Check if migration is in progress - return informative response for graceful degradation
    fts_status = get_fts_migration_status()
    if fts_status.in_progress:
        if fts_status.started_at is not None and fts_status.estimated_seconds is not None:
            elapsed = (datetime.now(tz=UTC) - fts_status.started_at).total_seconds()
            remaining = max(0, fts_status.estimated_seconds - int(elapsed))
        else:
            remaining = 60  # Default estimate if no timing info available

        old_lang = fts_status.old_language or 'unknown'
        new_lang = fts_status.new_language or settings.fts_language

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
                highlight=highlight,
                language=settings.fts_language,
                explain_query=explain_query,
            )
        except FtsValidationError as e:
            # Return error response (unified with search_context behavior)
            error_response: dict[str, Any] = {
                'query': query,
                'mode': mode,
                'results': [],
                'count': 0,
                'language': settings.fts_language,
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
        except Exception as e:
            logger.error(f'FTS search failed: {e}')
            raise ToolError(f'FTS search failed: {format_exception_message(e)}') from e

        # Process results: parse metadata and enrich with tags
        for result in search_results:
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

        logger.info(f'FTS search found {len(search_results)} results for query: "{query[:50]}..."')

        response: dict[str, Any] = {
            'query': query,
            'mode': mode,
            'results': search_results,
            'count': len(search_results),
            'language': settings.fts_language,
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
    search_modes: Annotated[
        list[Literal['fts', 'semantic']] | None,
        Field(
            description="Search modes to use: 'fts' (full-text), 'semantic' (vector similarity), "
            "or both ['fts', 'semantic'] (default). Modes are executed in parallel.",
        ),
    ] = None,
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
    fts_score, semantic_distance.

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
    if not settings.enable_hybrid_search:
        raise ToolError(
            'Hybrid search is not available. '
            'Set ENABLE_HYBRID_SEARCH=true to enable this feature. '
            'Also ensure ENABLE_FTS=true and/or ENABLE_SEMANTIC_SEARCH=true.',
        )

    # Use default search modes if not specified
    if search_modes is None:
        search_modes = ['fts', 'semantic']

    # Use settings default if rrf_k not specified
    effective_rrf_k = rrf_k if rrf_k is not None else settings.hybrid_rrf_k

    # Determine available search modes
    fts_available = settings.enable_fts
    semantic_available = settings.enable_semantic_search and get_embedding_provider() is not None

    # Filter requested modes to available ones
    available_modes: list[str] = []
    if 'fts' in search_modes and fts_available:
        available_modes.append('fts')
    if 'semantic' in search_modes and semantic_available:
        available_modes.append('semantic')

    if not available_modes:
        unavailable_reasons: list[str] = []
        if 'fts' in search_modes and not fts_available:
            unavailable_reasons.append('FTS requires ENABLE_FTS=true')
        if 'semantic' in search_modes and not semantic_available:
            unavailable_reasons.append(
                f'Semantic search requires ENABLE_SEMANTIC_SEARCH=true and '
                f'{settings.embedding.provider} provider properly configured',
            )
        raise ToolError(
            f'No search modes available. Requested: {search_modes}. '
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

        # Get repositories
        repos = await ensure_repositories()

        # Over-fetch for better fusion quality
        # Must account for offset to ensure all entries are fetched for proper pagination
        over_fetch_limit = (limit + offset) * 2

        # Execute searches in parallel
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
                # Check if FTS migration is in progress
                if get_fts_migration_status().in_progress:
                    fts_error = 'FTS migration in progress'
                    return

                # Check if FTS is properly initialized
                if not await repos.fts.is_available():
                    fts_error = 'FTS index not available'
                    return

                from app.repositories.fts_repository import FtsValidationError

                try:
                    results, stats = await repos.fts.search(
                        query=query,
                        mode='match',
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
                        highlight=False,
                        language=settings.fts_language,
                        explain_query=explain_query,
                    )
                    fts_results = results
                    if explain_query:
                        fts_stats = stats
                except FtsValidationError as e:
                    fts_error = str(e.message)
                except Exception as e:
                    fts_error = str(e)
            except Exception as e:
                fts_error = str(e)

        async def run_semantic_search() -> None:
            nonlocal semantic_results, semantic_error, semantic_stats
            try:
                hybrid_embedding_provider = get_embedding_provider()
                if hybrid_embedding_provider is None:
                    semantic_error = 'Embedding service not available'
                    return

                # Track embedding generation time for explain_query
                embedding_start_time = time_module.time() if explain_query else 0.0

                # Generate embedding for query
                try:
                    query_embedding = await hybrid_embedding_provider.embed_query(query)
                except Exception as e:
                    semantic_error = f'Failed to generate embedding: {e}'
                    return

                embedding_generation_ms = (
                    (time_module.time() - embedding_start_time) * 1000 if explain_query else 0.0
                )

                from app.repositories.embedding_repository import MetadataFilterValidationError

                try:
                    # Unpack tuple; stats will be captured for explain_query
                    results, search_stats = await repos.embeddings.search(
                        query_embedding=query_embedding,
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
                        explain_query=explain_query,
                    )
                    semantic_results = results

                    # Build semantic stats with embedding generation time
                    if explain_query:
                        semantic_stats = {
                            'execution_time_ms': round(search_stats.get('execution_time_ms', 0.0), 2),
                            'embedding_generation_ms': round(embedding_generation_ms, 2),
                            'filters_applied': search_stats.get('filters_applied', 0),
                            'rows_returned': search_stats.get('rows_returned', 0),
                            'backend': search_stats.get('backend', 'unknown'),
                            'query_plan': search_stats.get('query_plan'),
                        }
                except MetadataFilterValidationError as e:
                    semantic_error = str(e.message)
                except Exception as e:
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

        # Fuse results using RRF
        # Over-fetch to handle offset, then apply offset after fusion
        fused_results = reciprocal_rank_fusion(
            fts_results=fts_results,
            semantic_results=semantic_results,
            k=effective_rrf_k,
            limit=limit + offset,  # Over-fetch to handle offset
        )

        # Apply offset after fusion
        fused_results = fused_results[offset:]

        # Enrich results with tags and optionally images (cast to Any for mutation compatibility)
        fused_results_any: list[dict[str, Any]] = cast(list[dict[str, Any]], fused_results)
        for result in fused_results_any:
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
            f'Hybrid search found {len(fused_results_any)} results for query: "{query[:50]}..." '
            f'(fts={len(fts_results)}, semantic={len(semantic_results)}, modes={modes_used})',
        )

        # Build response
        response: dict[str, Any] = {
            'query': query,
            'results': fused_results_any,
            'count': len(fused_results_any),
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
