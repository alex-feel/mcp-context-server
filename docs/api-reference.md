# API Reference

## Introduction

The MCP Context Server exposes 13 MCP tools for context management, organized into core operations, search tools, and batch operations.

**Tool Categories:**
- **Core Operations**: `store_context`, `search_context`, `get_context_by_ids`, `delete_context`, `update_context`, `list_threads`, `get_statistics`
- **Search Tools**: `semantic_search_context`, `fts_search_context`, `hybrid_search_context`
- **Batch Operations**: `store_context_batch`, `update_context_batch`, `delete_context_batch`

## Core Tools

### store_context

Store a context entry with optional images and flexible metadata.

**Parameters:**
- `thread_id` (str, required): Unique identifier for the conversation/task thread
- `source` (str, required): Either 'user' or 'agent'
- `text` (str, required): Text content to store
- `images` (list, optional): Base64 encoded images with mime_type
- `metadata` (dict, optional): Additional structured data - completely flexible JSON object for your use case
- `tags` (list, optional): Tags for organization (automatically normalized)

**Metadata Flexibility:**
The metadata field accepts any JSON-serializable structure, making the server adaptable to various use cases:
- **Task Management**: Store `status`, `priority`, `assignee`, `due_date`, `completed`
- **Agent Coordination**: Track `agent_name`, `task_name`, `execution_time`, `resource_usage`
- **Knowledge Base**: Include `category`, `relevance_score`, `source_url`, `author`
- **Debugging Context**: Save `error_type`, `stack_trace`, `environment`, `version`
- **Analytics**: Record `user_id`, `session_id`, `event_type`, `timestamp`

**Performance Note:** The following metadata fields are indexed by default for faster filtering:
- `status`: State information (e.g., 'pending', 'active', 'done')
- `agent_name`: Specific agent identifier
- `task_name`: Task title for string searches
- `project`: Project name for filtering
- `report_type`: Report categorization (e.g., 'research', 'implementation')
- `references`: Cross-references object (PostgreSQL GIN index only)
- `technologies`: Technology stack array (PostgreSQL GIN index only)

Indexed fields are configurable via `METADATA_INDEXED_FIELDS` environment variable. See [Metadata Guide](metadata-addition-updating-and-filtering.md#environment-variables) for details.

**Returns:** Dictionary with success status and context_id

### search_context

Search context entries with powerful filtering including metadata queries and date ranges.

**Parameters:**
- `thread_id` (str, optional): Filter by thread
- `source` (str, optional): Filter by source ('user' or 'agent')
- `tags` (list, optional): Filter by tags (OR logic)
- `content_type` (str, optional): Filter by type ('text' or 'multimodal')
- `metadata` (dict, optional): Simple metadata filters (key=value equality)
- `metadata_filters` (list, optional): Advanced metadata filters with operators
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `limit` (int, optional): Maximum results to return (1-100, default: 30)
- `offset` (int, optional): Pagination offset (default: 0)
- `include_images` (bool, optional): Include image data in response
- `explain_query` (bool, optional): Include query execution statistics

**Metadata Filtering:** Supports simple key=value equality and advanced filtering with 16 operators. See [Metadata Guide](metadata-addition-updating-and-filtering.md).

**Date Filtering:** Supports ISO 8601 date filtering. See [Date Filtering](#date-filtering) section below.

**Returns:** List of matching context entries with optional query statistics

### get_context_by_ids

Fetch specific context entries by their IDs.

**Parameters:**
- `context_ids` (list, required): List of context entry IDs
- `include_images` (bool, optional): Include image data (default: True)

**Returns:** List of context entries with full content

### delete_context

Delete context entries by IDs or thread.

**Parameters:**
- `context_ids` (list, optional): Specific IDs to delete
- `thread_id` (str, optional): Delete all entries in a thread

**Returns:** Dictionary with deletion count

### list_threads

List all active threads with statistics.

**Returns:** Dictionary containing:
- List of threads with entry counts
- Source type distribution
- Multimodal content counts
- Timestamp ranges

### get_statistics

Get database statistics and usage metrics.

**Returns:** Dictionary with:
- Total entries count
- Breakdown by source and content type
- Total images count
- Unique tags count
- Database size in MB

### update_context

Update specific fields of an existing context entry.

**Parameters:**
- `context_id` (int, required): ID of the context entry to update
- `text` (str, optional): New text content
- `metadata` (dict, optional): New metadata (full replacement)
- `metadata_patch` (dict, optional): Partial metadata update using RFC 7396 JSON Merge Patch
- `tags` (list, optional): New tags (full replacement)
- `images` (list, optional): New images (full replacement)

**Metadata Update Options:**

Use `metadata` for full replacement or `metadata_patch` for partial updates. These parameters are mutually exclusive.

RFC 7396 JSON Merge Patch semantics (`metadata_patch`):
- New keys are ADDED to existing metadata
- Existing keys are REPLACED with new values
- Null values DELETE keys

```python
# Update single field while preserving others
update_context(context_id=123, metadata_patch={"status": "completed"})

# Add new field and delete another
update_context(context_id=123, metadata_patch={"reviewer": "alice", "draft": None})
```

**Limitations (RFC 7396):** Null values cannot be stored (null means delete key - use full replacement if needed), arrays are replaced entirely (not merged). See [Metadata Guide](metadata-addition-updating-and-filtering.md#partial-metadata-updates-metadata_patch) for details.

**Field Update Rules:**
- **Updatable fields**: text_content, metadata, tags, images
- **Immutable fields**: id, thread_id, source, created_at (preserved for data integrity)
- **Auto-managed fields**: content_type (recalculated based on image presence), updated_at (set to current timestamp)

**Update Behavior:**
- Only provided fields are updated (selective updates)
- Tags and images use full replacement semantics for consistency
- Content type automatically switches between 'text' and 'multimodal' based on image presence
- At least one updatable field must be provided

**Returns:** Dictionary with:
- Success status
- Context ID
- List of updated fields
- Success/error message

## Search Tools

### semantic_search_context

Perform semantic similarity search using vector embeddings.

Note: This tool is only available when semantic search is enabled via `ENABLE_SEMANTIC_SEARCH=true` and all dependencies are installed. The implementation varies by backend:
- **SQLite**: Uses sqlite-vec extension with embedding model via Ollama
- **PostgreSQL**: Uses pgvector extension (pre-installed in pgvector Docker image) with embedding model via Ollama

**Parameters:**
- `query` (str, required): Natural language search query
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `thread_id` (str, optional): Optional filter by thread
- `source` (str, optional): Filter by source type ('user' or 'agent')
- `tags` (list, optional): Filter by any of these tags (OR logic)
- `content_type` (str, optional): Filter by content type ('text' or 'multimodal')
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `metadata` (dict, optional): Simple metadata filters (key=value equality)
- `metadata_filters` (list, optional): Advanced metadata filters with operators
- `include_images` (bool, optional): Include image data in results (default: false)
- `explain_query` (bool, optional): Include query execution statistics (default: false)

**Metadata Filtering:** Supports same filtering syntax as search_context. See [Metadata Guide](metadata-addition-updating-and-filtering.md).

**Returns:** Dictionary with:
- Query string
- List of semantically similar context entries with similarity scores
- Result count
- Model name used for embeddings
- Query execution statistics (only when `explain_query=True`)

**Use Cases:**
- Find related work across different threads based on semantic similarity
- Discover contexts with similar meaning but different wording
- Concept-based retrieval without exact keyword matching
- Find similar content within a specific time period using date filters

**Date Filtering Example:**
```python
# Find similar content from the past week
semantic_search_context(
    query="authentication implementation",
    start_date="2025-11-22",
    end_date="2025-11-29"
)
```

For setup instructions, see the [Semantic Search Guide](semantic-search.md).

### fts_search_context

Perform full-text search with linguistic processing, relevance ranking, and highlighted snippets.

Note: This tool is only available when FTS is enabled via `ENABLE_FTS=true`. The implementation varies by backend:
- **SQLite**: Uses FTS5 with BM25 ranking. Porter stemmer (English) or unicode61 tokenizer (multilingual).
- **PostgreSQL**: Uses tsvector/tsquery with ts_rank_cd ranking. Supports 29 languages with full stemming.

**Parameters:**
- `query` (str, required): Search query
- `mode` (str, optional): Search mode - `match` (default), `prefix`, `phrase`, or `boolean`
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `thread_id` (str, optional): Optional filter by thread
- `source` (str, optional): Filter by source type ('user' or 'agent')
- `tags` (list, optional): Filter by any of these tags (OR logic)
- `content_type` (str, optional): Filter by content type ('text' or 'multimodal')
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `metadata` (dict, optional): Simple metadata filters (key=value equality)
- `metadata_filters` (list, optional): Advanced metadata filters with operators
- `highlight` (bool, optional): Include highlighted snippets in results (default: false)
- `include_images` (bool, optional): Include image data in results (default: false)
- `explain_query` (bool, optional): Include query execution statistics (default: false)

**Search Modes:**
- `match`: Standard word matching with stemming (default)
- `prefix`: Prefix matching for autocomplete-style search
- `phrase`: Exact phrase matching preserving word order
- `boolean`: Boolean operators (AND, OR, NOT) for complex queries

**Metadata Filtering:** Supports same filtering syntax as search_context. See [Metadata Guide](metadata-addition-updating-and-filtering.md).

**Returns:** Dictionary with:
- Query string and search mode
- List of matching entries with relevance scores and highlighted snippets
- Result count
- FTS availability status

**Example:**
```python
# Search with prefix matching
fts_search_context(
    query="auth",
    mode="prefix",
    thread_id="project-123"
)

# Boolean search with metadata filter
fts_search_context(
    query="authentication AND security",
    mode="boolean",
    metadata_filters=[{"key": "status", "operator": "eq", "value": "active"}]
)
```

For detailed configuration, see the [Full-Text Search Guide](full-text-search.md).

### hybrid_search_context

Perform hybrid search combining FTS and semantic search with Reciprocal Rank Fusion (RRF).

Note: This tool is only available when hybrid search is enabled via `ENABLE_HYBRID_SEARCH=true` and at least one of FTS (`ENABLE_FTS=true`) or semantic search (`ENABLE_SEMANTIC_SEARCH=true`) is enabled. The RRF algorithm combines results from available search methods, boosting documents that appear in both.

**Parameters:**
- `query` (str, required): Natural language search query
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `search_modes` (list, optional): Search modes to use - `['fts', 'semantic']` (default: both)
- `fusion_method` (str, optional): Fusion algorithm - `'rrf'` (default)
- `rrf_k` (int, optional): RRF smoothing constant (1-1000, default from HYBRID_RRF_K env var)
- `thread_id` (str, optional): Optional filter by thread
- `source` (str, optional): Filter by source type ('user' or 'agent')
- `tags` (list, optional): Filter by any of these tags (OR logic)
- `content_type` (str, optional): Filter by content type ('text' or 'multimodal')
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `metadata` (dict, optional): Simple metadata filters (key=value equality)
- `metadata_filters` (list, optional): Advanced metadata filters with operators
- `include_images` (bool, optional): Include image data in results (default: false)
- `explain_query` (bool, optional): Include query execution statistics (default: false)

**Metadata Filtering:** Supports same filtering syntax as search_context. See [Metadata Guide](metadata-addition-updating-and-filtering.md).

**Returns:** Dictionary with:
- Query string and fusion method
- List of matching entries with combined RRF scores and individual search rankings
- Result count and counts from each search method
- List of search modes actually used
- Query execution statistics (only when `explain_query=True`)

**Scores Breakdown:**
Each result includes a `scores` object with:
- `rrf`: Combined RRF score (higher = better)
- `fts_rank`: Position in FTS results (1-based), null if not in FTS results
- `semantic_rank`: Position in semantic results (1-based), null if not in semantic results
- `fts_score`: Original FTS relevance score (BM25/ts_rank)
- `semantic_distance`: Original semantic distance (L2, lower = more similar)
- `rerank_score`: Cross-encoder relevance score (higher = better, 0.0-1.0), null if reranking disabled

**Graceful Degradation:**
- If only FTS is available, returns FTS results only
- If only semantic search is available, returns semantic results only
- If neither is available, raises an error

**Example:**
```python
# Full hybrid search
hybrid_search_context(
    query="authentication implementation",
    thread_id="project-123"
)

# Hybrid with metadata filtering
hybrid_search_context(
    query="performance optimization",
    metadata={"status": "completed"},
    metadata_filters=[{"key": "priority", "operator": "gte", "value": 7}]
)

# Single mode through hybrid API (for consistent interface)
hybrid_search_context(
    query="exact phrase",
    search_modes=["fts"]
)
```

For detailed configuration and troubleshooting, see the [Hybrid Search Guide](hybrid-search.md).

## Search Tools Response Structure

All search tools return consistent response structures with common fields and tool-specific additions:

| Field | search_context | semantic_search_context | fts_search_context | hybrid_search_context |
|-------|----------------|------------------------|-------------------|----------------------|
| `results` | List of entries | List of entries | List of entries | List of entries |
| `count` | Yes | Yes | Yes | Yes |
| `query` | No | Yes | Yes | Yes |
| `stats` | explain_query=True | explain_query=True | explain_query=True | explain_query=True |
| `model` | No | Yes (embedding model) | No | No |
| `mode` | No | No | Yes (search mode) | No |
| `language` | No | No | Yes (FTS language) | No |
| `fusion_method` | No | No | No | Yes |
| `search_modes_used` | No | No | No | Yes |
| `fts_count` | No | No | No | Yes |
| `semantic_count` | No | No | No | Yes |

**Entry Fields by Tool:**

| Entry Field                                    | search_context        | semantic_search_context | fts_search_context  | hybrid_search_context |
|------------------------------------------------|-----------------------|-------------------------|---------------------|-----------------------|
| `id`, `thread_id`, `source`, `content_type`    | Yes                   | Yes                     | Yes                 | Yes                   |
| `text_content`                                 | Truncated (150 chars) | Full                    | Full                | Full                  |
| `is_truncated`                                 | Yes                   | No                      | No                  | No                    |
| `metadata`, `tags`, `created_at`, `updated_at` | Yes                   | Yes                     | Yes                 | Yes                   |
| `images`                                       | include_images=True   | include_images=True     | include_images=True | include_images=True   |
| `scores`                                       | No                    | Yes                     | Yes                 | Yes                   |
| `highlighted`                                  | No                    | No                      | highlight=True      | No                    |

**Scores Object Structure:**

All search tools (except `search_context`) return a unified `scores` object with applicable fields:

| Field               | semantic_search | fts_search | hybrid_search | Polarity        |
|---------------------|-----------------|------------|---------------|-----------------|
| `semantic_distance` | Yes             | No         | Yes           | LOWER = better  |
| `semantic_rank`     | null            | No         | Yes           | LOWER = better  |
| `fts_score`         | No              | Yes        | Yes           | HIGHER = better |
| `fts_rank`          | No              | null       | Yes           | LOWER = better  |
| `rrf`               | No              | No         | Yes           | HIGHER = better |
| `rerank_score`      | Yes*            | Yes*       | Yes*          | HIGHER = better |

*`rerank_score` is present when reranking is enabled (`ENABLE_RERANKING=true`, default).

**Notes:**
- `stats` is only included when `explain_query=True` for all search tools
- `search_context` returns truncated text for browsing; use `get_context_by_ids` for full content
- For standalone FTS and semantic searches, rank fields are always `null` (no cross-method ranking)

## Batch Operations

The following tools enable efficient batch processing of context entries.

### store_context_batch

Store multiple context entries in a single batch operation.

**Parameters:**
- `entries` (list, required): List of context entries (max 100). Each entry has:
  - `thread_id` (str, required), `source` (str, required), `text` (str, required)
  - `metadata` (dict, optional), `tags` (list, optional), `images` (list, optional)
- `atomic` (bool, optional): If true, all succeed or all fail (default: true)

**Returns:** Dictionary with success, total, succeeded, failed, results array, message

### update_context_batch

Update multiple context entries in a single batch operation.

**Parameters:**
- `updates` (list, required): List of update operations (max 100). Each update has:
  - `context_id` (int, required)
  - `text` (str, optional), `metadata` (dict, optional), `metadata_patch` (dict, optional)
  - `tags` (list, optional), `images` (list, optional)
- `atomic` (bool, optional): If true, all succeed or all fail (default: true)

**Note:** `metadata_patch` uses RFC 7396 JSON Merge Patch semantics. See [Metadata Guide](metadata-addition-updating-and-filtering.md#partial-metadata-updates-metadata_patch) for details.

**Returns:** Dictionary with success, total, succeeded, failed, results array, message

### delete_context_batch

Delete multiple context entries by various criteria. **IRREVERSIBLE.**

**Parameters:**
- `context_ids` (list, optional): Specific context IDs to delete
- `thread_ids` (list, optional): Delete all entries in these threads
- `source` (str, optional): Filter by source ('user' or 'agent') - must combine with another criterion
- `older_than_days` (int, optional): Delete entries older than N days

At least one criterion must be provided. Cascading delete removes associated tags, images, and embeddings.

**Returns:** Dictionary with success, deleted_count, criteria_used, message

## Filtering Reference

The following filtering options apply to `search_context`, `semantic_search_context`, `fts_search_context`, and `hybrid_search_context` tools.

### Metadata Filtering

*Simple filtering* (exact match):
```python
metadata={'status': 'active', 'priority': 5}
```

*Advanced filtering* with operators:
```python
metadata_filters=[
    {'key': 'priority', 'operator': 'gt', 'value': 3},
    {'key': 'status', 'operator': 'in', 'value': ['active', 'pending']},
    {'key': 'agent_name', 'operator': 'starts_with', 'value': 'gpt'},
    {'key': 'completed', 'operator': 'eq', 'value': False}
]
```

**Supported Operators:**
- `eq`: Equals (case-insensitive for strings by default)
- `ne`: Not equals
- `gt`, `gte`, `lt`, `lte`: Numeric comparisons
- `in`, `not_in`: List membership
- `exists`, `not_exists`: Field presence
- `contains`, `starts_with`, `ends_with`: String operations
- `is_null`, `is_not_null`: Null checks
- `array_contains`: Check if array field contains element

All string operators support `case_sensitive: true/false` option.

For comprehensive documentation on metadata filtering including real-world use cases, operator examples, nested JSON paths, and performance optimization, see the [Metadata Guide](metadata-addition-updating-and-filtering.md).

### Date Filtering

Filter entries by creation timestamp using ISO 8601 format:
```python
# Find entries from a specific day
search_context(thread_id="project-123", start_date="2025-11-29", end_date="2025-11-29")

# Find entries from a date range
search_context(thread_id="project-123", start_date="2025-11-01", end_date="2025-11-30")

# Find entries with precise timestamp
search_context(thread_id="project-123", start_date="2025-11-29T10:00:00")
```

Supported ISO 8601 formats:
- Date-only: `2025-11-29`
- DateTime: `2025-11-29T10:00:00`
- UTC (Z suffix): `2025-11-29T10:00:00Z`
- Timezone offset: `2025-11-29T10:00:00+02:00`

**Note:** Date-only `end_date` values automatically expand to end-of-day (`T23:59:59.999999`) for intuitive "entire day" behavior. Naive datetime (without timezone) is interpreted as UTC.

## Additional Resources

### Related Documentation

- **Database Backends**: [Database Backends Guide](database-backends.md) - database configuration
- **Semantic Search**: [Semantic Search Guide](semantic-search.md) - vector similarity search setup
- **Full-Text Search**: [Full-Text Search Guide](full-text-search.md) - FTS configuration and usage
- **Hybrid Search**: [Hybrid Search Guide](hybrid-search.md) - combined FTS + semantic search
- **Metadata Filtering**: [Metadata Guide](metadata-addition-updating-and-filtering.md) - metadata operators
- **Docker Deployment**: [Docker Deployment Guide](deployment/docker.md) - containerized deployment
- **Authentication**: [Authentication Guide](authentication.md) - HTTP transport authentication
- **Main Documentation**: [README.md](../README.md) - overview and quick start
