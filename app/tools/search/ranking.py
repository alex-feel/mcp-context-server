"""Post-retrieval shaping of search results.

Cross-encoder reranking with an injected provider, and the display format (text
truncation, summary normalization, canonical timestamps) applied to every result a
search tool returns.
"""

import logging
from typing import Any

from app.repositories.base import canonical_timestamp
from app.reranking.base import RerankingProvider
from app.settings import get_settings
from app.startup.validation import truncate_text

logger = logging.getLogger(__name__)
settings = get_settings()


async def apply_reranking(
    query: str,
    results: list[dict[str, Any]],
    limit: int | None = None,
    *,
    reranking_provider: RerankingProvider | None,
) -> list[dict[str, Any]]:
    """Apply cross-encoder reranking to search results.

    This is a helper function used by search tools to rerank results
    after initial retrieval (semantic search, FTS, or hybrid).

    Args:
        query: The search query to score results against.
        results: Search results with 'id' and 'text_content' fields.
        limit: Maximum number of results to return after reranking.
        reranking_provider: Cross-encoder provider; None when reranking is unavailable.

    Returns:
        Reranked results with 'rerank_score' injected into the 'scores' object.
        If reranking is disabled or unavailable, returns original results unchanged.
    """
    # Check if reranking is available
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
        result_by_id: dict[str, dict[str, Any]] = {
            str(r.get('id', '')): r for r in results if r.get('id') is not None
        }

        # Merge rerank scores back into original results (inject into scores object)
        final_results: list[dict[str, Any]] = []
        for reranked_item in reranked:
            item_id = reranked_item.get('id')
            if item_id is not None and str(item_id) in result_by_id:
                merged = result_by_id[str(item_id)].copy()
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


def apply_search_display_format(entry: dict[str, Any]) -> None:
    """Apply unified search display formatting to a single result entry.

    Truncates text_content based on SEARCH_TRUNCATION_LENGTH setting and
    normalizes the summary field. Both fields are always present in the output:
    text_content as a truncated preview, summary as a string (empty when absent).

    Modifies the entry dict in-place.

    Args:
        entry: A single search result dict to format.
    """
    # Normalize summary: always present as string, empty when absent
    summary = entry.get('summary')
    if isinstance(summary, str) and summary.strip():
        entry['summary'] = summary
    else:
        entry['summary'] = ''

    # Always truncate text_content (regardless of summary presence)
    text_content = entry.get('text_content', '')
    truncated_text, is_truncated = truncate_text(text_content, max_length=settings.search.truncation_length)
    entry['text_content'] = truncated_text
    entry['is_text_content_truncated'] = is_truncated

    # Canonical timestamp wire format -- identical across SQLite and PostgreSQL for
    # every search tool (search/fts/semantic/hybrid all route through here). See
    # app.repositories.base.canonical_timestamp.
    for ts_key in ('created_at', 'updated_at'):
        if entry.get(ts_key) is not None:
            entry[ts_key] = canonical_timestamp(entry[ts_key])
