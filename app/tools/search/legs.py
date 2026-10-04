"""Raw semantic and full-text search legs, without reranking.

The standalone semantic and FTS tools and the hybrid tool all run these legs. Each
caller resolves the repositories, the embedding provider and the caller's access scope
once and passes them in; every leg searches only the entries that scope may read.
"""

import json
import logging
import time
from typing import Any
from typing import Literal

from fastmcp.exceptions import ToolError

from app.access_scope import Scope
from app.embeddings.base import EmbeddingProvider
from app.errors import format_exception_message
from app.migrations import get_fts_migration_status
from app.repositories import RepositoryContainer
from app.services.passage_extraction_service import extract_rerank_passage
from app.settings import get_settings
from app.tools.search.limits import structural_filter_errors

logger = logging.getLogger(__name__)
settings = get_settings()


async def semantic_search_raw(
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
    extract_rerank_text: bool = False,
    explain_query: bool = False,
    *,
    repos: RepositoryContainer,
    embedding_provider: EmbeddingProvider | None,
    scope: Scope,
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
        extract_rerank_text: If True, extract matched chunk as rerank_text (internal use).
        explain_query: Include query execution statistics.
        repos: Repository container the search runs against.
        embedding_provider: Provider that embeds the query; None when semantic search is unavailable.
        scope: The caller's scope; only entries it may read are ranked.

    Returns:
        Tuple of (results, stats). Results include text_content and distance.
        If extract_rerank_text is True, results also include rerank_text.

    Raises:
        ToolError: If semantic search is not available or fails.
        MetadataFilterValidationError: If metadata filters are invalid.
    """
    # Check if semantic search is available
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

    # Reject a structurally invalid filter BEFORE the embedding round trip. The
    # repository validates the same inputs, but only inside the read callable that
    # runs after embed_query(), so a request that could never return results still
    # burned provider latency and quota -- and the rejection then reported a zero
    # embedding_generation_ms for the seconds it had just spent. The messages and the
    # exception type are the repository's, so the response a caller builds from them
    # is unchanged.
    from app.repositories.embedding_repository.records import MetadataFilterValidationError

    structural_errors = structural_filter_errors(tags, metadata, metadata_filters)
    if structural_errors is not None:
        raise MetadataFilterValidationError('Metadata filter validation failed', structural_errors)

    # Generate embedding for query, measuring the call so callers can surface the
    # documented embedding_generation_ms stat. When EMBEDDING_QUERY_INSTRUCTION is
    # set, it is prepended verbatim to the text handed to the embedding provider,
    # because instruct-aware models condition query vectors on such a prefix. Only
    # this embed_query call sees the prefix: reranking, the hybrid FTS leg, and the
    # response echo all keep the bare query, and document embeddings on the
    # store/update path are never prefixed.
    query_instruction = settings.embedding.query_instruction
    embedding_input = f'{query_instruction}{query}' if query_instruction else query
    embedding_start = time.perf_counter()
    try:
        query_embedding = await embedding_provider.embed_query(embedding_input)
    except Exception as e:
        logger.error(f'Failed to generate query embedding: {e}')
        raise ToolError(f'Failed to generate embedding for query: {format_exception_message(e)}') from e
    embedding_generation_ms = (time.perf_counter() - embedding_start) * 1000

    # Perform similarity search with optional filtering
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
            scope=scope,
        )
    except MetadataFilterValidationError:
        raise  # Let caller handle validation errors
    except Exception as e:
        logger.error(f'Semantic search failed: {e}')
        raise ToolError(f'Semantic search failed: {format_exception_message(e)}') from e

    # Surface the measured query-embedding duration in the stats contract
    # (HybridSemanticStatsDict.embedding_generation_ms); hybrid search inherits it
    # via semantic_stats.
    search_stats['embedding_generation_ms'] = round(embedding_generation_ms, 2)

    # Post-process for reranking: extract chunk text from boundaries
    if extract_rerank_text:
        for result in search_results:
            text_content = result.get('text_content', '')
            start_idx = result.get('matched_chunk_start')
            end_idx = result.get('matched_chunk_end')

            if start_idx is not None and end_idx is not None and end_idx > start_idx:
                # Extract matched chunk for reranking
                result['rerank_text'] = text_content[start_idx:end_idx]
                logger.debug(
                    f'Extracted rerank_text: {len(result["rerank_text"])} chars '
                    f'from [{start_idx}:{end_idx}]',
                )
            else:
                # Fallback: use the beginning of the document (an embedding stored without chunk boundaries)
                max_rerank_len = int(settings.reranking.max_length * settings.reranking.chars_per_token * 0.95)
                result['rerank_text'] = text_content[:max_rerank_len]
                logger.debug(
                    f'No chunk boundaries, using document beginning '
                    f'({len(result["rerank_text"])} chars)',
                )

    # matched_chunk_start/matched_chunk_end are internal chunk boundaries that
    # exist only to feed the chunk-aware reranking above. Strip them from EVERY
    # result unconditionally so this shared helper never returns implementation-only
    # keys: the result contract declares no such fields, and the tools return
    # dict[str, Any] (no FastMCP output-schema filtering), so leaving them in leaks
    # them to the client whenever reranking did not run (provider disabled,
    # unavailable, or failed). The rerank branch above already consumed the
    # boundaries before this scrub, so removing them here is safe for every caller.
    for result in search_results:
        result.pop('matched_chunk_start', None)
        result.pop('matched_chunk_end', None)

    # Parse JSON metadata to a dict for EVERY semantic result so this shared
    # helper returns the same metadata shape as search_context/fts_search_context.
    # The repository returns metadata as a JSON string on BOTH backends (SQLite
    # TEXT; PostgreSQL JSONB decoded as str because asyncpg registers no jsonb
    # codec), so without this both semantic_search_context and the semantic leg
    # of hybrid_search_context would emit a string where the result contract
    # types metadata as a dict. The hasattr(..., 'strip') guard makes this
    # idempotent: an already-parsed dict has no 'strip' and is left untouched.
    for result in search_results:
        metadata_raw = result.get('metadata')
        if metadata_raw is not None and hasattr(metadata_raw, 'strip'):
            try:
                result['metadata'] = json.loads(str(metadata_raw))
            except (json.JSONDecodeError, ValueError, AttributeError):
                result['metadata'] = None

    return search_results, search_stats


async def fts_search_raw(
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
    internal_highlight_for_rerank: bool = False,
    explain_query: bool = False,
    *,
    repos: RepositoryContainer,
    scope: Scope,
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
        internal_highlight_for_rerank: Generate highlights internally for passage extraction.
        explain_query: Include query execution statistics.
        repos: Repository container the search runs against.
        scope: The caller's scope; only entries it may read match.

    Returns:
        Tuple of (results, stats). Results include text_content and score.
        When internal_highlight_for_rerank=True, results also include 'rerank_text' field.

    Raises:
        ToolError: If FTS is not available or fails.
        FtsValidationError: If query or filters are invalid.
    """
    # Check if FTS is enabled
    if not settings.fts.enabled:
        raise ToolError(
            'Full-text search is not available. '
            'Set ENABLE_FTS to auto (default) or true to enable this feature.',
        )

    # Check if migration is in progress
    fts_status = get_fts_migration_status()
    if fts_status.in_progress:
        raise ToolError('FTS migration in progress. Please retry shortly.')

    # Check if FTS is properly initialized
    if not await repos.fts.is_available():
        raise ToolError(
            'FTS index not found. The database may need migration. '
            'Restart the server with ENABLE_FTS=true to apply migrations.',
        )

    # Import exception here to avoid circular imports
    from app.repositories.fts_repository.faults import FtsValidationError

    # Determine actual highlight setting
    # Generate highlights if client requested OR if we need for passage extraction
    actual_highlight = highlight or internal_highlight_for_rerank

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
            scope=scope,
        )
    except FtsValidationError:
        raise  # Let caller handle validation errors
    except Exception as e:
        logger.error(f'FTS search failed: {e}')
        raise ToolError(f'FTS search failed: {format_exception_message(e)}') from e

    # Post-process for reranking: extract passages from highlighted results
    if internal_highlight_for_rerank:
        for result in search_results:
            highlighted = result.get('highlighted')
            text_content = result.get('text_content', '')

            # Extract passage for reranking using highlight positions
            result['rerank_text'] = extract_rerank_passage(
                text_content=text_content,
                highlighted=highlighted,
                window_size=settings.fts_passage.rerank_window_size,
                max_passage_size=int(settings.reranking.max_length * settings.reranking.chars_per_token * 0.95),
                gap_merge_threshold=settings.fts_passage.rerank_gap_merge,
            )

            # Remove 'highlighted' if client didn't request it
            if not highlight:
                result.pop('highlighted', None)

    return search_results, stats
