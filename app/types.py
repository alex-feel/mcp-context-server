"""Type definitions for the MCP context server.

This module provides type definitions to replace explicit Any usage
and ensure strict type safety throughout the codebase.
"""

from typing import TypedDict

# JSON value types - recursive union for JSON-like data structures
type JsonValue = str | int | float | bool | None | list['JsonValue'] | dict[str, 'JsonValue']

# Metadata value types - simpler non-recursive type for metadata fields
type MetadataValue = str | int | float | bool | None

# Metadata dictionary type for use in models - supports nested JSON structures
type MetadataDict = dict[str, JsonValue]


# API Response TypedDicts for proper return type annotations
class ImageAttachmentDict(TypedDict):
    """Type definition for image attachment responses."""

    image_id: int
    context_id: int
    mime_type: str
    size_bytes: int


class ContextEntryDict(TypedDict, total=False):
    """Type definition for context entry responses.

    Uses total=False to handle optional fields properly.
    """

    id: int
    thread_id: str
    source: str
    content_type: str
    text_content: str | None
    metadata: MetadataDict | None
    created_at: str
    updated_at: str
    tags: list[str]
    images: list[ImageAttachmentDict] | list[dict[str, str]] | None
    is_truncated: bool | None


class SearchContextResponseDict(TypedDict, total=False):
    """Type definition for search_context response.

    Uses total=False to handle optional fields (stats, error, validation_errors).
    """

    results: list[ContextEntryDict]
    count: int
    stats: dict[str, object] | None  # Only present when explain_query=True
    error: str | None
    validation_errors: list[str] | None


class StoreContextSuccessDict(TypedDict):
    """Type definition for successful store context response."""

    success: bool
    context_id: int
    thread_id: str
    message: str


class ThreadInfoDict(TypedDict):
    """Type definition for individual thread info."""

    thread_id: str
    entry_count: int
    source_types: int
    multimodal_count: int
    first_entry: str
    last_entry: str
    last_id: int


class ThreadListDict(TypedDict):
    """Type definition for thread list response."""

    threads: list[ThreadInfoDict]
    total_threads: int


class ImageDict(TypedDict, total=False):
    """Type definition for image data in API responses."""

    data: str
    mime_type: str
    metadata: dict[str, str] | None


class UpdateContextSuccessDict(TypedDict):
    """Type definition for successful update context response."""

    success: bool
    context_id: int
    updated_fields: list[str]
    message: str


# Bulk operation TypedDicts


class BulkStoreItemDict(TypedDict, total=False):
    """Type definition for a single item in bulk store request.

    Required fields: thread_id, source, text
    Optional fields: metadata, tags, images
    """

    thread_id: str
    source: str
    text: str
    metadata: MetadataDict | None
    tags: list[str] | None
    images: list[dict[str, str]] | None


class BulkStoreResultItemDict(TypedDict):
    """Type definition for a single result in bulk store response."""

    index: int
    success: bool
    context_id: int | None
    error: str | None


class BulkStoreResponseDict(TypedDict):
    """Type definition for bulk store response."""

    success: bool
    total: int
    succeeded: int
    failed: int
    results: list[BulkStoreResultItemDict]
    message: str


class BulkUpdateItemDict(TypedDict, total=False):
    """Type definition for a single item in bulk update request.

    Required field: context_id
    Optional fields: text, metadata, metadata_patch, tags, images
    Note: metadata and metadata_patch are mutually exclusive per entry.
    """

    context_id: int
    text: str | None
    metadata: MetadataDict | None
    metadata_patch: MetadataDict | None
    tags: list[str] | None
    images: list[dict[str, str]] | None


class BulkUpdateResultItemDict(TypedDict):
    """Type definition for a single result in bulk update response."""

    index: int
    context_id: int
    success: bool
    updated_fields: list[str] | None
    error: str | None


class BulkUpdateResponseDict(TypedDict):
    """Type definition for bulk update response."""

    success: bool
    total: int
    succeeded: int
    failed: int
    results: list[BulkUpdateResultItemDict]
    message: str


class BulkDeleteResponseDict(TypedDict):
    """Type definition for bulk delete response."""

    success: bool
    deleted_count: int
    criteria_used: list[str]
    message: str


# FTS (Full-Text Search) TypedDicts


class FtsSearchResultDict(TypedDict, total=False):
    """Type definition for FTS search result entry."""

    id: int
    thread_id: str
    source: str
    content_type: str
    text_content: str
    metadata: MetadataDict | None
    created_at: str
    updated_at: str
    tags: list[str]
    score: float
    highlighted: str | None


class FtsSearchResponseDict(TypedDict, total=False):
    """Type definition for fts_search_context response.

    Uses total=False to handle optional fields (stats, error, validation_errors).
    """

    query: str
    mode: str
    results: list[FtsSearchResultDict]
    count: int
    language: str
    stats: dict[str, object] | None  # Only present when explain_query=True
    error: str | None
    validation_errors: list[str] | None


class SemanticSearchResultDict(TypedDict, total=False):
    """Type definition for semantic search result entry.

    Similar to FtsSearchResultDict but with distance instead of score.
    """

    id: int
    thread_id: str
    source: str
    content_type: str
    text_content: str
    metadata: MetadataDict | None
    created_at: str
    updated_at: str
    tags: list[str]
    distance: float  # L2 Euclidean distance (LOWER = more similar)
    images: list[ImageAttachmentDict] | list[dict[str, str]] | None


class SemanticSearchResponseDict(TypedDict, total=False):
    """Type definition for semantic_search_context response.

    Uses total=False to handle optional fields (stats, error, validation_errors).
    """

    query: str
    results: list[SemanticSearchResultDict]
    count: int
    model: str
    stats: dict[str, object] | None  # Only present when explain_query=True
    error: str | None
    validation_errors: list[str] | None


class FtsMigrationInProgressDict(TypedDict):
    """Type definition for FTS migration in progress response.

    Returned by fts_search_context when the FTS index is being rebuilt
    due to a language/tokenizer change. Provides informative feedback
    to clients with estimated completion time.
    """

    migration_in_progress: bool
    message: str
    started_at: str  # ISO 8601 timestamp
    estimated_remaining_seconds: int
    old_language: str
    new_language: str
    suggestion: str


# Hybrid Search TypedDicts


class HybridScoresDict(TypedDict, total=False):
    """Type definition for hybrid search scores breakdown.

    Contains scores from individual search methods and the combined RRF score.
    """

    rrf: float  # Combined RRF score
    fts_rank: int | None  # Rank in FTS results (1-based), None if not in FTS results
    semantic_rank: int | None  # Rank in semantic results (1-based), None if not in semantic results
    fts_score: float | None  # Original FTS score (BM25/ts_rank)
    semantic_distance: float | None  # Original semantic distance (L2)


class HybridSearchResultDict(TypedDict, total=False):
    """Type definition for hybrid search result entry.

    Combines fields from both FTS and semantic search results with
    hybrid-specific scoring information.
    """

    id: int
    thread_id: str
    source: str
    content_type: str
    text_content: str
    metadata: MetadataDict | None
    created_at: str
    updated_at: str
    tags: list[str]
    scores: HybridScoresDict  # Hybrid scoring breakdown


# Hybrid Search Stats TypedDicts (for explain_query parameter)
# NOTE: Stats types defined before HybridSearchResponseDict to avoid forward reference


class HybridFtsStatsDict(TypedDict, total=False):
    """Type definition for FTS statistics in hybrid search.

    Contains timing and filter information from the FTS portion
    of hybrid search.
    """

    execution_time_ms: float
    filters_applied: int
    rows_returned: int
    query_plan: str | None
    backend: str


class HybridSemanticStatsDict(TypedDict, total=False):
    """Type definition for semantic search statistics in hybrid search.

    Contains timing and filter information from the semantic portion
    of hybrid search.
    """

    execution_time_ms: float
    embedding_generation_ms: float
    filters_applied: int
    rows_returned: int
    backend: str
    query_plan: str | None


class HybridFusionStatsDict(TypedDict):
    """Type definition for RRF fusion statistics in hybrid search.

    Contains overlap and distribution information about how results
    were combined from FTS and semantic search.
    """

    rrf_k: int
    total_unique_documents: int
    documents_in_both: int
    documents_fts_only: int
    documents_semantic_only: int


class HybridSearchStatsDict(TypedDict, total=False):
    """Type definition for complete hybrid search statistics.

    Aggregates stats from FTS, semantic search, and fusion operations.
    Only present in response when explain_query=True.
    """

    execution_time_ms: float  # Total hybrid search time
    fts_stats: HybridFtsStatsDict | None
    semantic_stats: HybridSemanticStatsDict | None
    fusion_stats: HybridFusionStatsDict


class HybridSearchResponseDict(TypedDict, total=False):
    """Type definition for hybrid_search_context response.

    Uses total=False to handle optional stats field.
    """

    query: str
    results: list[HybridSearchResultDict]
    count: int
    fusion_method: str  # 'rrf'
    search_modes_used: list[str]  # Actual modes executed, e.g., ['fts', 'semantic']
    fts_count: int  # Number of results from FTS
    semantic_count: int  # Number of results from semantic search
    stats: HybridSearchStatsDict | None  # Only present when explain_query=True
