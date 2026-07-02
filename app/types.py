"""Type definitions for the MCP context server.

This module provides type definitions to replace explicit Any usage
and ensure strict type safety throughout the codebase.
"""

from typing import NotRequired
from typing import TypedDict

# JSON value types - recursive union for JSON-like data structures
type JsonValue = str | int | float | bool | None | list['JsonValue'] | dict[str, 'JsonValue']

# Metadata value types - simpler non-recursive type for metadata fields
type MetadataValue = str | int | float | bool | None

# Metadata dictionary type for use in models - supports nested JSON structures
type MetadataDict = dict[str, JsonValue]


# API Response TypedDicts for proper return type annotations
class ImageDict(TypedDict):
    """Type definition for image data in API responses.

    ``mime_type`` is always present; ``data`` is present only when images are fetched
    with ``include_data=True``; ``metadata`` is present only when the image carries
    metadata.
    """

    mime_type: str
    data: NotRequired[str]
    metadata: NotRequired[dict[str, str] | None]


class ContextEntryDict(TypedDict, total=False):
    """Type definition for context entry responses.

    Uses total=False to handle optional fields properly.
    """

    id: str
    thread_id: str
    source: str
    content_type: str
    text_content: str | None
    metadata: MetadataDict | None
    summary: str  # Conditional presence handled by TypedDict(total=False); absence == "feature disabled"
    created_at: str
    updated_at: str
    tags: list[str]
    images: list[ImageDict] | None
    is_text_content_truncated: bool | None


class ClampedLimitDict(TypedDict):
    """Type definition for clamped limit hint in search responses."""

    requested: int
    applied: int


class SearchContextResponseDict(TypedDict, total=False):
    """Type definition for search_context response.

    Uses total=False to handle optional fields (stats, error, validation_errors).
    """

    results: list[ContextEntryDict]
    count: int
    stats: dict[str, object] | None  # Only present when explain_query=True
    error: str | None
    validation_errors: list[str] | None
    clamped_limit: ClampedLimitDict | None


class GrepFileDict(TypedDict):
    """A grep result row in ``files_with_matches`` mode (locator only)."""

    context_id: str
    match_count: int


class GrepContentMatchDict(TypedDict):
    """A grep result row in ``content`` mode (one per match).

    Offsets are Unicode code-point indices into the entry's ``text_content`` and
    compose directly with ``read_context_range`` (start_char/end_char).
    """

    context_id: str
    line_number: int
    line: str
    match_start: int
    match_end: int
    before: list[str]
    after: list[str]


class GrepCountDict(TypedDict):
    """A grep result row in ``count`` mode (per-entry match tally)."""

    context_id: str
    count: int


class GrepContextResultDict(TypedDict, total=False):
    """Type definition for the grep_context response.

    ``results`` holds one of the three row shapes depending on ``mode``; ``error``
    and ``validation_errors`` appear only when a metadata filter is invalid.
    ``timed_out_context_ids`` appears only when one or more entries were skipped
    because their regex match exceeded the per-entry timeout (the result is then
    incomplete for those entries).
    """

    mode: str
    total_matches: int
    truncated: bool
    results: list[GrepFileDict] | list[GrepContentMatchDict] | list[GrepCountDict]
    error: str | None
    validation_errors: list[str] | None
    timed_out_context_ids: list[str]


class ReadContextRangeDict(TypedDict):
    """Type definition for the read_context_range response.

    Echoes the RESOLVED span actually returned (after clamping out-of-range
    offsets to ``[0, len]``) so a caller always knows exactly what it received.
    Offsets are Unicode code-point indices.
    """

    context_id: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    text: str


class OutlineNodeDict(TypedDict, total=False):
    """A node in the navigate_context outline tree.

    ``char_start``/``char_end`` are Unicode code-point offsets feeding directly
    into read_context_range. ``summary`` mirrors the entry summary on the root
    (by reference) and carries an optional per-node LLM summary on descendants.
    """

    node_id: str
    level: int
    ordinal: int
    title: str
    char_start: int
    char_end: int
    summary: str | None
    children: list['OutlineNodeDict']


class NavigateContextResultDict(TypedDict, total=False):
    """Type definition for the navigate_context response."""

    context_id: str
    total_chars: int
    node_count: int
    root: OutlineNodeDict


class StoreContextSuccessDict(TypedDict):
    """Type definition for successful store context response."""

    success: bool
    context_id: str
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
    last_id: str


class ThreadListDict(TypedDict):
    """Type definition for thread list response."""

    threads: list[ThreadInfoDict]
    total_threads: int


# Statistics TypedDicts (for get_statistics tool response)


class ConnectionMetricsDict(TypedDict, total=False):
    """Type definition for backend connection metrics.

    Shape varies by backend; the only fields guaranteed across backends are
    ``backend_type`` and ``pool_size``. SQLite adds active_readers,
    writer_busy, write_queue_size, circuit_breaker_state, total_writes,
    total_reads, failed_writes, failed_reads. PostgreSQL adds pool_idle,
    pool_free, total_queries, failed_queries. Treated as an open-shape
    dict; per-backend keys are documented here for reference only.
    """

    backend_type: str
    pool_size: int


class MostActiveThreadDict(TypedDict):
    """One row of ``most_active_threads`` in the stats response."""

    thread_id: str
    count: int


class TopTagDict(TypedDict):
    """One row of ``top_tags`` in the stats response."""

    tag: str
    count: int


class SemanticSearchStatsDict(TypedDict, total=False):
    """Type definition for the ``semantic_search`` sub-block in get_statistics.

    ``enabled`` and ``available`` are ALWAYS present. The remaining fields
    appear only when ``available is True``; when ``available is False`` an
    optional ``message`` field carries the degradation reason.

    The field name ``embedding_count`` refers to the number of stored
    embedding rows on disk (one row per text chunk, since the storage layer
    persists one embedding vector per chunk). The text chunks themselves
    are NOT persisted; chunking is a transient pre-processing step purely
    for embedding quality.
    """

    enabled: bool
    available: bool
    backend: str
    model: str
    dimensions: int
    context_count: int  # COUNT(*) from embedding_metadata (entry count)
    embedding_count: int  # SUM(chunk_count) from embedding_metadata (stored embedding rows)
    average_chunks_per_entry: float
    coverage_percentage: float
    message: str  # Present when available is False


class FtsStatsDict(TypedDict, total=False):
    """Type definition for the ``fts`` sub-block in get_statistics.

    ``enabled`` and ``available`` are ALWAYS present. The remaining fields
    appear only when ``available is True``; when ``available is False`` an
    optional ``message`` field carries the degradation reason.
    """

    enabled: bool
    available: bool
    language: str
    backend: str
    engine: str
    indexed_entries: int
    coverage_percentage: float
    message: str  # Present when available is False


class ChunkingStatsDict(TypedDict):
    """Type definition for the ``chunking`` sub-block in get_statistics.

    All fields ALWAYS present (no conditional shape).
    """

    enabled: bool
    available: bool
    chunk_size: int
    chunk_overlap: int
    aggregation: str


class RerankingStatsDict(TypedDict, total=False):
    """Type definition for the ``reranking`` sub-block in get_statistics.

    ``enabled`` and ``available`` are ALWAYS present. The remaining fields
    appear only when ``available is True``; when ``available is False`` an
    optional ``message`` field carries the degradation reason.
    """

    enabled: bool
    available: bool
    provider: str
    model: str
    message: str  # Present when available is False


class SummaryStatsDict(TypedDict, total=False):
    """Type definition for the ``summary`` sub-block in get_statistics.

    ``enabled`` and ``available`` are ALWAYS present. The remaining fields
    appear only when ``available is True``; when ``available is False`` an
    optional ``message`` field carries the degradation reason.
    """

    enabled: bool
    available: bool
    provider: str
    model: str
    summary_count: int
    coverage_percentage: float
    min_content_length: int
    message: str  # Present when available is False


class CompressionStatsDict(TypedDict, total=False):
    """Type definition for the ``compression`` sub-block in get_statistics.

    ``enabled`` and ``available`` are ALWAYS present. The remaining fields
    appear only when ``available is True``; when ``available is False`` an
    optional ``message`` field carries the degradation reason. Values
    originate from the singleton ``compression_metadata`` row (DB-truth)
    except ``max_concurrent`` which comes from runtime settings.
    """

    enabled: bool
    available: bool
    provider: str
    bits: int
    variant: str
    seed: int
    dim: int
    max_concurrent: int
    message: str  # Present when available is False


class IndexTreeStatsDict(TypedDict, total=False):
    """Type definition for the ``index_tree`` sub-block in get_statistics.

    ``enabled`` reflects ENABLE_INDEX_TREE_NODE_SUMMARIES. ``node_count`` is the
    total stored per-node summaries when the feature is enabled. It is 0 whenever
    the feature is off -- reported without querying, mirroring the compression
    block's enabled-gated pattern, so any residual rows from a previously-enabled
    run are intentionally NOT counted -- and 0 when the table is absent.
    """

    enabled: bool
    node_count: int


class StatisticsResponseDict(TypedDict):
    """Type definition for the get_statistics tool response.

    Top-level fields come from
    ``StatisticsRepository.get_database_statistics``; sub-blocks
    (``semantic_search``, ``fts``, ``chunking``, ``reranking``,
    ``summary``, ``compression``, ``index_tree``) are added by
    ``app.tools.discovery.get_statistics`` and are always present.

    ``database_size_mb`` reflects the whole database on PostgreSQL
    (``pg_database_size``) and the on-disk database file on SQLite. It is
    omitted only for in-memory or missing-file SQLite databases.

    ``embeddings_size_mb`` reflects the active vector payload table and is not
    byte-comparable across backends (on-disk relation size including indexes on
    PostgreSQL; raw payload bytes or an fp32 estimate on SQLite). When the
    SQLite figure is the fp32 estimate, ``embeddings_size_estimated`` is True.
    """

    total_entries: int
    by_source: dict[str, int]
    by_content_type: dict[str, int]
    total_images: int
    unique_tags: int
    total_threads: int
    avg_entries_per_thread: float
    most_active_threads: list[MostActiveThreadDict]
    top_tags: list[TopTagDict]
    backend: str
    database_size_mb: NotRequired[float]  # Whole DB on PostgreSQL; on-disk file on SQLite
    # Embedding vector payload size (not byte-comparable across backends)
    embeddings_size_mb: NotRequired[float]
    embeddings_size_estimated: NotRequired[bool]  # True only for the SQLite fp32 estimate
    connection_metrics: ConnectionMetricsDict
    semantic_search: SemanticSearchStatsDict
    fts: FtsStatsDict
    chunking: ChunkingStatsDict
    reranking: RerankingStatsDict
    summary: SummaryStatsDict
    compression: CompressionStatsDict
    # Always emitted by get_statistics (both toggle branches set it), so required.
    index_tree: IndexTreeStatsDict


class UpdateContextSuccessDict(TypedDict):
    """Type definition for successful update context response."""

    success: bool
    context_id: str
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
    context_id: str | None
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

    context_id: str
    text: str | None
    metadata: MetadataDict | None
    metadata_patch: MetadataDict | None
    tags: list[str] | None
    images: list[dict[str, str]] | None


class BulkUpdateResultItemDict(TypedDict):
    """Type definition for a single result in bulk update response."""

    index: int
    context_id: str
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


class ScoresDict(TypedDict, total=False):
    """Unified scores breakdown for all search tools.

    All search tools return this structure with applicable fields populated:
    - FTS search: fts_score, fts_rank, rerank_score
    - Semantic search: semantic_distance, semantic_rank, rerank_score
    - Hybrid search: All fields

    Score Polarity Reference:
    - fts_score: HIGHER = better match (BM25/ts_rank relevance)
    - fts_rank: LOWER = better (1 = best)
    - semantic_distance: LOWER = better; L2 for fp32/mse, negated inner product for the ip compression variant
    - semantic_rank: LOWER = better (1 = best)
    - rrf: HIGHER = better (combined RRF score)
    - rerank_score: HIGHER = better (cross-encoder relevance, 0.0-1.0)
    """

    # FTS scores
    fts_score: float | None  # BM25/ts_rank relevance (HIGHER = better)
    fts_rank: int | None  # Rank in FTS results (1-based, LOWER = better)

    # Semantic scores
    semantic_distance: float | None  # Lower = better; L2 for fp32/mse, negated inner product for ip variant
    semantic_rank: int | None  # Rank in semantic results (1-based, LOWER = better)

    # RRF score (hybrid only)
    rrf: float | None  # Combined RRF score (HIGHER = better)

    # Rerank score (all tools when reranking enabled)
    rerank_score: float | None  # Cross-encoder relevance (HIGHER = better, 0.0-1.0)


class FtsSearchResultDict(TypedDict, total=False):
    """Type definition for FTS search result entry.

    The `scores` object contains:
    - fts_score: BM25/ts_rank relevance score (HIGHER = better)
    - fts_rank: Always null for standalone FTS (no ranking)
    - rerank_score: Present when reranking is enabled (HIGHER = better)
    """

    id: str
    thread_id: str
    source: str
    content_type: str
    text_content: str
    summary: str
    is_text_content_truncated: bool
    metadata: MetadataDict | None
    created_at: str
    updated_at: str
    tags: list[str]
    scores: ScoresDict
    highlighted: str | None
    images: list[ImageDict] | None


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
    clamped_limit: ClampedLimitDict | None


class SemanticSearchResultDict(TypedDict, total=False):
    """Type definition for semantic search result entry.

    The `scores` object contains:
    - semantic_distance: LOWER = more similar; L2 Euclidean for fp32/mse, negated inner product (~ -1..0) for the ip variant
    - semantic_rank: Always null for standalone semantic (no ranking)
    - rerank_score: Present when reranking is enabled (HIGHER = better)
    """

    id: str
    thread_id: str
    source: str
    content_type: str
    text_content: str
    summary: str
    is_text_content_truncated: bool
    metadata: MetadataDict | None
    created_at: str
    updated_at: str
    tags: list[str]
    scores: ScoresDict
    images: list[ImageDict] | None


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
    clamped_limit: ClampedLimitDict | None


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
    The `rerank_score` is present when reranking is enabled.
    """

    rrf: float  # Combined RRF score (HIGHER = better)
    fts_rank: int | None  # Rank in FTS results (1-based), None if not in FTS results
    semantic_rank: int | None  # Rank in semantic results (1-based), None if not in semantic results
    fts_score: float | None  # Original FTS score (BM25/ts_rank)
    semantic_distance: float | None  # Lower = better; L2 for fp32/mse, negated inner product for ip variant
    rerank_score: float | None  # Cross-encoder reranking score (HIGHER = better, 0.0-1.0)


class HybridSearchResultDict(TypedDict, total=False):
    """Type definition for hybrid search result entry.

    Combines fields from both FTS and semantic search results with
    hybrid-specific scoring information.
    """

    id: str
    thread_id: str
    source: str
    content_type: str
    text_content: str
    summary: str
    is_text_content_truncated: bool
    metadata: MetadataDict | None
    created_at: str
    updated_at: str
    tags: list[str]
    scores: HybridScoresDict  # Hybrid scoring breakdown
    rerank_text: str | None  # Internal: chunk text for reranking (removed before API response)
    images: list[ImageDict] | None


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
    adaptive_fts_mode: str  # 'match' or 'boolean' - indicates FTS mode used


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
    stats: HybridSearchStatsDict | None
    clamped_limit: ClampedLimitDict | None
    warnings: list[str] | None  # Degradation warnings when sub-searches fail (e.g., FTS or semantic)
