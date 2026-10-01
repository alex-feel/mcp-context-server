"""Argument bounds and filter caps for the search tools.

Holds the result-limit, offset, ranked-depth and query-length bounds, the caps on
tags and metadata filters (shared with the navigation tools), the structural filter
check that runs before any search, and the empty responses those bounds produce.
"""

import logging
from typing import Any
from typing import Literal

from app.errors import format_exception_message
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Maximum allowed search results limit (Postel's Law: accept, clamp, warn)
MAX_SEARCH_LIMIT = 100


# Depth of the ONE ranked ordering that semantic, FTS, and hybrid search serve every
# page from. It is fixed and page-independent by design. Those three tools decide the
# final order AFTER the database returns rows -- cross-encoder reranking, RRF fusion,
# or both -- so sizing the candidate window from the requested page (the old
# (limit + offset) * factors) built page N and page N + 1 from DIFFERENT candidate
# pools: a document that only entered the larger pool could outrank rows already
# served, which pushed those rows onto a later page a second time while other rows
# were never returned by any page. With a fixed depth there is exactly one ordering
# per query and offset/limit merely select a window inside it. MAX_SEARCH_LIMIT is
# that depth because a single request can never return more rows than it, so the
# ordering spans every window a client can ask for; a window reaching past the depth
# is reported to the client through the rank_depth_limit hint. The depth is also the
# candidate pool the cross-encoder scores, and it already exceeds any requested page
# size, so the ranked tools no longer scale that pool by RERANKING_OVERFETCH on top of
# the page -- doing so would have made the pool grow with the page again.
RANKED_SEARCH_DEPTH = MAX_SEARCH_LIMIT


# Maximum allowed pagination offset, rejected at the tool boundary. search_context
# binds the offset straight into its LIMIT/OFFSET SQL, where an unbounded value is
# both an anti-pattern (the database still scans offset + limit rows) and a bind that
# can exceed the int64 column width. That failure surfaces INSIDE connection scope and
# charges the circuit breaker for what is purely invalid client input; bounding the
# offset here keeps it a clean client-facing validation error. The ranked tools slice
# their fixed-depth ordering in Python, so for them any offset past RANKED_SEARCH_DEPTH
# simply returns no rows together with the rank_depth_limit hint.
MAX_SEARCH_OFFSET = 1_000_000


# Wire-schema ceiling for the full-text query string, applied to the two tools whose
# query text is handed to a search ENGINE that builds an expression out of it. Neither
# engine is bounded by the number of terms it will assemble: PostgreSQL aborts a query
# of many thousands of terms while allocating the tsquery, and FTS5 spends real CPU on
# one. Neither failure is dangerous any more (both classify as client input and stay off
# the circuit breaker), but there is no reason to let a client ask for the work at all.
# The bound is roughly a thousand words -- orders of magnitude above any real search
# query and far below the sizes at which the engines misbehave -- and rejecting past it
# is a clean schema-level validation error before the request reaches the server. The
# semantic-only tool is deliberately NOT bounded: its query goes to an embedding
# provider, which has its own documented context handling.
MAX_FTS_QUERY_LENGTH = 10_000


def rank_depth_hint(offset: int, limit: int) -> dict[str, int] | None:
    """Return the hint describing a page that reaches past the fixed ranked depth.

    Ranked search serves every page from ONE ordering of at most
    RANKED_SEARCH_DEPTH rows (see that constant), so a window whose end lies beyond
    the depth is served short -- or empty when the offset alone is past it. The hint
    says so explicitly instead of leaving the client to guess whether the result set
    was exhausted or the window fell outside the ranked ordering.

    Args:
        offset: The requested pagination offset.
        limit: The applied (already clamped) page size.

    Returns:
        A hint carrying the requested window and the ranked depth, or None when the
        whole window fits inside the depth.
    """
    if offset + limit <= RANKED_SEARCH_DEPTH:
        return None
    return {
        'requested_offset': offset,
        'requested_limit': limit,
        'rank_depth': RANKED_SEARCH_DEPTH,
    }


# Maximum member count for the client-supplied tags filter, shared by the four
# search tools and grep_context. Each tag expands into one SQL bind placeholder
# (tag IN (?, ?, ...)) in a single statement on both backends, so an unbounded
# schema-legal tags array could overflow the backend's bind limit (999 variables on
# conservative SQLite builds; asyncpg's wire-protocol argument cap) INSIDE
# connection scope -- a non-ControlFlowError that charges the circuit breaker for
# purely invalid client input. The cap is advertised in the wire schema via
# Field(max_length=MAX_FILTER_TAGS) on every tags parameter and re-checked in each
# tool body (tags_filter_cap_error), so a caller bypassing schema validation gets
# the same structured validation error before any SQL executes. Aligned with
# MAX_SEARCH_LIMIT and the metadata IN/NOT_IN member cap (MAX_IN_LIST_MEMBERS in
# app.metadata_types).
MAX_FILTER_TAGS = 100


def tags_filter_cap_error(tags: list[str] | None) -> str | None:
    """Return a validation message when the tags filter exceeds MAX_FILTER_TAGS.

    Defense in depth behind the Field(max_length=MAX_FILTER_TAGS) wire-schema cap:
    the tool bodies call this before any repository work so an oversized list from
    a caller that bypassed schema validation is rejected as a structured validation
    error instead of reaching a SQL bind (where the placeholder overflow would
    charge the circuit breaker).

    Args:
        tags: The client-supplied tags filter (None when absent).

    Returns:
        A client-facing message when the list has more than MAX_FILTER_TAGS
        members, else None.
    """
    if tags is not None and len(tags) > MAX_FILTER_TAGS:
        return (
            f'Too many tags in filter: {len(tags)} exceeds the maximum of '
            f'{MAX_FILTER_TAGS}. Narrow the filter to at most {MAX_FILTER_TAGS} tags.'
        )
    return None


# Maximum member count for the client-supplied metadata_filters list, shared by the
# four search tools and grep_context. The sibling caps bound the PER-ITEM dimensions
# (MAX_FILTER_TAGS per tags list, MAX_IN_LIST_MEMBERS per IN/NOT_IN value list), but
# without a cap on the filter LIST itself the capped dimensions still multiply: a few
# hundred schema-legal IN filters of 100 members each expand past the backend's
# per-statement bind limit (asyncpg's 32,767-argument wire cap; 32,766 variables on a
# python.org SQLite build) INSIDE connection scope -- a non-ControlFlowError that
# charges the circuit breaker for purely invalid client input, and a cross-backend
# divergence (a Debian-built SQLite with a higher compile-time limit silently
# succeeds where PostgreSQL hard-errors). The cap is advertised in the wire schema
# via Field(max_length=MAX_METADATA_FILTERS) on every metadata_filters parameter and
# re-checked in each tool body (metadata_filters_cap_error), matching the tags-cap
# pattern. Aligned with MAX_FILTER_TAGS and MAX_IN_LIST_MEMBERS (both 100).
MAX_METADATA_FILTERS = 100


# Maximum key count for the simple metadata (key=value equality) filter dict, shared
# by the four search tools. Each key expands into one SQL condition with one bind
# placeholder, so it is the same bind-multiplication dimension as the filter list
# above and carries the identical cap for uniformity.
MAX_METADATA_KEYS = 100


def metadata_filters_cap_error(metadata_filters: list[dict[str, Any]] | None) -> str | None:
    """Return a validation message when the metadata_filters list exceeds MAX_METADATA_FILTERS.

    Defense in depth behind the Field(max_length=MAX_METADATA_FILTERS) wire-schema
    cap, mirroring :func:`tags_filter_cap_error`: the tool bodies call this before
    any repository work so an oversized list from a caller that bypassed schema
    validation is rejected as a structured validation error instead of expanding
    into an oversized single-statement placeholder run (which would charge the
    circuit breaker).

    Args:
        metadata_filters: The client-supplied advanced filter list (None when absent).

    Returns:
        A client-facing message when the list has more than MAX_METADATA_FILTERS
        members, else None.
    """
    if metadata_filters is not None and len(metadata_filters) > MAX_METADATA_FILTERS:
        return (
            f'Too many metadata filters: {len(metadata_filters)} exceeds the maximum of '
            f'{MAX_METADATA_FILTERS}. Narrow the query to at most {MAX_METADATA_FILTERS} filters.'
        )
    return None


def metadata_keys_cap_error(metadata: dict[str, str | int | float | bool] | None) -> str | None:
    """Return a validation message when the simple metadata dict exceeds MAX_METADATA_KEYS.

    Defense in depth behind the Field(max_length=MAX_METADATA_KEYS) wire-schema cap
    (rendered as maxProperties in the JSON Schema), mirroring
    :func:`tags_filter_cap_error` for the simple key=value equality dict.

    Args:
        metadata: The client-supplied simple metadata filter dict (None when absent).

    Returns:
        A client-facing message when the dict has more than MAX_METADATA_KEYS keys,
        else None.
    """
    if metadata is not None and len(metadata) > MAX_METADATA_KEYS:
        return (
            f'Too many metadata keys in filter: {len(metadata)} exceeds the maximum of '
            f'{MAX_METADATA_KEYS}. Narrow the filter to at most {MAX_METADATA_KEYS} keys.'
        )
    return None


def filter_caps_error(
    tags: list[str] | None,
    metadata: dict[str, str | int | float | bool] | None = None,
    metadata_filters: list[dict[str, Any]] | None = None,
) -> str | None:
    """Return the first boundary-cap violation among the client-supplied filter inputs.

    One shared body re-check for the three capped filter dimensions (tags list,
    simple metadata dict, advanced metadata_filters list), so every filter-bearing
    tool rejects an oversized input as the same structured validation error before
    any repository work runs.

    Args:
        tags: The client-supplied tags filter (None when absent).
        metadata: The client-supplied simple metadata filter dict (None when absent).
        metadata_filters: The client-supplied advanced filter list (None when absent).

    Returns:
        The first cap-violation message found, else None.
    """
    return (
        tags_filter_cap_error(tags)
        or metadata_keys_cap_error(metadata)
        or metadata_filters_cap_error(metadata_filters)
    )


def structural_filter_errors(
    tags: list[str] | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
    metadata_filters: list[dict[str, Any]] | None = None,
) -> list[str] | None:
    """Return the filter-validation messages a purely structural check can produce.

    The repositories validate ``tags``/``metadata``/``metadata_filters`` inside their
    per-backend read callables, which on the semantic path runs only AFTER the query
    embedding has been generated -- so a request that is guaranteed to be rejected
    still paid a full round trip to the embedding provider (seconds to tens of
    seconds), and the rejection then reported ``embedding_generation_ms: 0.0`` for
    time it really spent. Running the same checks here, before that call, rejects the
    request for the same reason with the same messages and no provider traffic.

    The check touches no database and is deliberately built without the enclosing
    statement's bind offset or table alias, which makes it strictly MORE permissive
    than the repository's own clause-budget accounting: it can only reject a request
    the repository would also reject, never one the repository would accept.

    Args:
        tags: The client-supplied tags filter (None when absent).
        metadata: The client-supplied simple metadata filter dict (None when absent).
        metadata_filters: The client-supplied advanced filter list (None when absent).

    Returns:
        The validation messages, in the order the repositories report them, or None
        when nothing structurally invalid was found.
    """
    from app.metadata_types import MetadataFilter
    from app.query_builder import MetadataQueryBuilder
    from app.repositories.base import BaseRepository

    if tags:
        try:
            BaseRepository.normalize_tag_filter(tags)
        except ValueError as e:
            return [format_exception_message(e)]

    if not metadata and not metadata_filters:
        return None

    backend_type: Literal['sqlite', 'postgresql'] = (
        'postgresql' if settings.storage.backend_type == 'postgresql' else 'sqlite'
    )
    builder = MetadataQueryBuilder(backend_type=backend_type)
    errors: list[str] = []

    if metadata:
        for key, value in metadata.items():
            try:
                builder.add_simple_filter(key, value)
            except ValueError as e:
                errors.append(f'Invalid metadata key {key!r}: {format_exception_message(e)}')

    if metadata_filters:
        for filter_dict in metadata_filters:
            try:
                # pydantic's ValidationError subclasses ValueError, so one branch
                # covers both the schema rejection (bad operator, missing key) and
                # the builder's own ValueError (unsafe key, clause budgets).
                builder.add_advanced_filter(MetadataFilter(**filter_dict))
            except ValueError as e:
                errors.append(f'Invalid metadata filter {filter_dict}: {format_exception_message(e)}')
            except Exception as e:
                # Parity with the repositories: an unexpected failure is still a
                # structured validation message rather than an opaque tool error.
                errors.append(f'Unexpected error in metadata filter {filter_dict}: {format_exception_message(e)}')
                logger.error(f'Unexpected error processing metadata filter: {e}')

    return errors or None


def empty_stats_for_unexecuted_query(*, include_embedding_ms: bool = False) -> dict[str, Any]:
    """Build the zeroed stats dict for a response whose search never executed.

    The uniform shape under ``explain_query`` for every path that returns without
    running SQL -- a rejected filter, and a page that provably lies outside the ranked
    ordering -- so a client sees the same stats keys whether the search executed or
    not. Zeroed counters plus the always-present ``backend`` key (the active storage
    backend type). ``query_plan`` is included as an explicit ``None``: no plan exists
    for a query that never ran, but omitting the key entirely would break the very
    uniformity this shape promises and make the documented
    ``stats['query_plan']`` access raise ``KeyError`` on those paths.
    ``include_embedding_ms`` adds the semantic shape's
    ``embedding_generation_ms`` counter (see ``HybridSemanticStatsDict``).

    Args:
        include_embedding_ms: Include the semantic-only embedding timing counter.

    Returns:
        The stats dict for the response.
    """
    stats: dict[str, Any] = {
        'execution_time_ms': 0.0,
        'filters_applied': 0,
        'rows_returned': 0,
        'backend': settings.storage.backend_type,
        'query_plan': None,
    }
    if include_embedding_ms:
        stats['embedding_generation_ms'] = 0.0
    return stats


def empty_page_beyond_rank_depth(
    base: dict[str, Any],
    *,
    offset: int,
    limit: int,
    original_limit: int,
    stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """Finish the response for a page that lies entirely past the fixed ranked depth.

    Every ranked page is cut from one ordering of at most RANKED_SEARCH_DEPTH rows, so a
    request whose offset already reaches the depth can only ever return nothing. That is
    decidable from the client's own arguments alone, before any work runs, which is why
    the ranked tools answer it here instead of paying an embedding round trip, a search,
    and a cross-encoder pass over the full window only to discard every row in the slice.

    Args:
        base: The response keys specific to the calling tool (query, model, mode, and
            the hybrid shape's mode counters).
        offset: The requested pagination offset.
        limit: The applied (already clamped) page size.
        original_limit: The page size as requested, for the clamped_limit notice.
        stats: The stats block to attach, or None when explain_query is off.

    Returns:
        The complete empty-page response, carrying the rank_depth_limit hint that
        explains why it is empty.
    """
    response: dict[str, Any] = {**base, 'results': [], 'count': 0}
    if stats is not None:
        response['stats'] = stats
    if original_limit != limit:
        response['clamped_limit'] = {
            'requested': original_limit,
            'applied': limit,
        }
    depth_hint = rank_depth_hint(offset, limit)
    if depth_hint is not None:
        response['rank_depth_limit'] = depth_hint
    return response
