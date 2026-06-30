# API Reference

## Introduction

The MCP Context Server exposes 16 MCP tools for context management, organized into core operations, search tools, navigation tools (locate / navigate / extract), and batch operations.

**Tool Categories:**
- **Core Operations**: `store_context`, `search_context`, `get_context_by_ids`, `delete_context`, `update_context`, `list_threads`, `get_statistics`
- **Search Tools**: `semantic_search_context`, `fts_search_context`, `hybrid_search_context`
- **Navigation Tools**: `grep_context`, `navigate_context`, `read_context_range`
- **Batch Operations**: `store_context_batch`, `update_context_batch`, `delete_context_batch`

## Context Identifier Format

Every context entry is identified by a UUIDv7 value. The canonical public form is a 32-character lowercase hexadecimal string with no hyphens (regex `^[0-9a-f]{32}$`).

Example: `0190abcdef1234567890abcdef123456`

**Accepted Input Formats:**

Every tool that accepts a context identifier accepts BOTH of the following at the parameter boundary:

- The canonical 32-character hex form (e.g., `0190abcdef1234567890abcdef123456`)
- The 36-character hyphenated UUID form (e.g., `0190abcd-ef12-3456-7890-abcdef123456`)

Whitespace is stripped and the value is folded to lowercase before validation. Storage canonicalizes to the 32-character lowercase hex form; the `context_id` returned in tool responses is always in canonical form regardless of input format.

**Prefix Lookup:**

Every tool parameter that accepts a context identifier ALSO accepts a hex prefix of 8 to 31 characters, including the bulk and batch list parameters (each element is resolved independently). The server resolves each prefix against stored entries:

- Exactly one match: resolves to that entry's full ID.
- Zero matches: returns an error `No context entry matches prefix '<prefix>'`.
- More than one match: returns an error `Ambiguous prefix '<prefix>' matches multiple entries`.

Prefixes shorter than 8 characters or containing non-hex characters are rejected at validation.

**Input Format Support by Tool:**

| Parameter Kind                                                                                                                                                                      | Accepts Full UUID (32-hex / 36-hyphenated) | Accepts Prefix (8-31 hex)                 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|-------------------------------------------|
| Single-ID parameters (`update_context.context_id`, `navigate_context.context_id`, `read_context_range.context_id`)                                                                  | Yes                                        | Yes                                       |
| Bulk / batch list parameters (`get_context_by_ids.context_ids`, `delete_context.context_ids`, `delete_context_batch.context_ids`, each `update_context_batch` entry's `context_id`) | Yes                                        | Yes (each element resolved independently) |

**Bulk Parameter Example:**

```python
get_context_by_ids(context_ids=[
    "0190abcdef1234567890abcdef123456",
    "0190abcd-ef12-3456-7890-abcdef987654"
])
```

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

**Returns:** Dictionary with success status, `thread_id`, a message, and a 32-character lowercase hex `context_id` identifying the stored entry. Example: `{"success": true, "context_id": "0190abcdef1234567890abcdef123456", "thread_id": "project-123", "message": "..."}`

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

**Returns:** List of matching context entries with truncated `text_content`, `summary`, and `is_text_content_truncated` flag, plus optional query statistics

### get_context_by_ids

Fetch specific context entries by their IDs.

**Parameters:**
- `context_ids` (list[str], required): List of context-entry IDs in canonical 32-character hex or 36-character hyphenated UUID form. Both forms are accepted at the tool boundary; storage canonicalizes to 32-character lowercase hex. An 8-31 character hex prefix is also accepted for each ID and resolved independently (zero matches or an ambiguous prefix returns an error).
- `include_images` (bool, optional): Include image data (default: True)

**Returns:** List of context entries with full untruncated `text_content`. Each entry contains `id`, `thread_id`, `source`, `text_content`, `metadata`, `tags`, `images`, `created_at`, and `updated_at`. The `summary` field follows a tri-state contract controlled by the `GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY` environment variable:

- When disabled (the default), the `summary` key is omitted entirely; consumers reading `entry.get('summary')` will receive `None`, which is the conventional Python signal for "feature disabled, no value to surface".
- When enabled and the stored summary is a non-empty string, the value is returned verbatim.
- When enabled but the stored summary is `NULL` or empty (e.g., generation was skipped because text was shorter than `SUMMARY_MIN_CONTENT_LENGTH`, or no provider is configured), the value is normalized to an empty string `''`. This mirrors the search-tool contract (search tools always emit `summary` as a string, never `None`) and provides an explicit "feature on, no data yet" signal distinct from the "feature disabled" `None`.

### delete_context

Delete context entries by IDs or thread.

**Parameters:**
- `context_ids` (list[str], optional): Specific 32-character hex or 36-character hyphenated UUID IDs to delete. Both forms are accepted at the tool boundary. An 8-31 character hex prefix is also accepted for each ID and resolved independently (zero matches or an ambiguous prefix returns an error).
- `thread_id` (str, optional): Delete all entries in a thread

**Returns:** Dictionary with deletion count

### list_threads

List threads with statistics. Pagination is optional and backward-compatible: with no arguments ALL threads are returned, ordered by most-recent activity first.

**Parameters:**
- `limit` (int, optional): Maximum threads to return (1-100). Omit (the default) to return all threads with no limit.
- `offset` (int, optional): Number of leading threads to skip for pagination (default 0). Ignored when `limit` is omitted.

**Returns:** Dictionary containing:
- `threads`: List of threads for the requested page, each with thread_id, entry_count, source_types, multimodal_count, first_entry/last_entry timestamps, and last_id (a hint for future keyset pagination).
- `total_threads`: Count of threads in THIS response (the returned page), not the whole database.

Threads are ordered by `last_entry` descending, tie-broken by the latest entry id descending; `limit`/`offset` are applied after this ordering. Keyset (cursor) pagination on `last_id` is a possible future enhancement; today `limit`/`offset` is the supported pagination.

### get_statistics

Get database statistics, usage metrics, and feature status.

**Returns:** Dictionary with:
- Total entries count
- Breakdown by source and content type
- Total images count
- Unique tags count
- Database size in MB (`database_size_mb`) — whole database via `pg_database_size(current_database())` on PostgreSQL; on-disk database file size on SQLite (excludes the `-wal`/`-shm` sidecars, so it can transiently under-report under WAL mode). Omitted for in-memory or missing-file SQLite databases.
- Embeddings storage size in MB (`embeddings_size_mb`, with the boolean `embeddings_size_estimated`) — size of the active vector payload table (`vec_context_embeddings_compressed` when compression is enabled, otherwise `vec_context_embeddings`). Present when embedding generation or compression is enabled. NOT byte-comparable across backends: on PostgreSQL it is the on-disk relation size including indexes (`pg_total_relation_size`); on SQLite it is the exact compressed payload bytes when compression is enabled, or a deterministic fp32 estimate when it is not. `embeddings_size_estimated` is `true` only for the SQLite fp32 estimate.
- Connection metrics
- Semantic search status (enabled, available, model, dimensions, embedding count, coverage)
- Full-text search status (enabled, available, language, backend, engine, indexed entries, coverage)
- Chunking configuration (enabled, available, chunk size, overlap, aggregation)
- Reranking status (enabled, available, provider, model)
- Summary generation status (enabled, available, provider, model, summary count, coverage, min content length)
- Compression status (enabled, available, provider, bits, variant, seed, dim, max_concurrent) — present when `ENABLE_EMBEDDING_COMPRESSION=true`; reduced to `{enabled: false, available: false}` shape when disabled.
- Index-tree node-summary status (`index_tree` with `enabled`, `node_count`) — `enabled` reflects `ENABLE_INDEX_TREE_NODE_SUMMARIES`; `node_count` is the total stored per-node summaries (0 when disabled or the table is absent).

### update_context

Update specific fields of an existing context entry.

**Parameters:**
- `context_id` (str, required): ID of the context entry to update. Accepts a 32-character hex UUID, a 36-character hyphenated UUID, or an 8-31 character hex prefix that uniquely identifies an entry. Prefixes shorter than 8 characters are rejected. Ambiguous prefixes (multiple matches) return an error. Whitespace is stripped and case is folded to lowercase at the boundary.
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
update_context(context_id="0190abcdef1234567890abcdef123456", metadata_patch={"status": "completed"})

# Add new field and delete another
update_context(context_id="0190abcdef1234567890abcdef123456", metadata_patch={"reviewer": "alice", "draft": None})
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

Note: This tool is available by default (`ENABLE_SEMANTIC_SEARCH=auto`) whenever an embedding provider is present; set `ENABLE_SEMANTIC_SEARCH=false` to force off. The implementation varies by backend:
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
- List of semantically similar context entries with truncated `text_content`, `summary`, `is_text_content_truncated` flag, and similarity scores
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

Note: This tool is available by default (`ENABLE_FTS=auto`); set `ENABLE_FTS=false` to force off. The implementation varies by backend:
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
- List of matching entries with truncated `text_content`, `summary`, `is_text_content_truncated` flag, relevance scores, and highlighted snippets
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

Note: This tool is available by default (`ENABLE_HYBRID_SEARCH=auto`) when at least one of full-text or semantic search is available; set `ENABLE_HYBRID_SEARCH=false` to force off. The RRF algorithm combines results from available search methods, boosting documents that appear in both.

**Parameters:**
- `query` (str, required): Natural language search query
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
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
- List of matching entries with truncated `text_content`, `summary`, `is_text_content_truncated` flag, combined RRF scores, and individual search rankings
- Result count and counts from each search method
- List of search modes actually used
- Query execution statistics (only when `explain_query=True`)

**Scores Breakdown:**
Each result includes a `scores` object with:
- `rrf`: Combined RRF score (higher = better)
- `fts_rank`: Position in FTS results (1-based), null if not in FTS results
- `semantic_rank`: Position in semantic results (1-based), null if not in semantic results
- `fts_score`: Original FTS relevance score (BM25/ts_rank)
- `semantic_distance`: Original semantic distance, lower = more similar. The underlying metric is Euclidean L2 (>= 0) for uncompressed/`mse` storage, or a negated inner product (~ -1..0 for normalized embeddings, more negative = more similar) when the default `ip` compression variant is active.
- `rerank_score`: Cross-encoder relevance score (higher = better, 0.0-1.0), null if reranking disabled

**Note:** When `ENABLE_RERANKING=true` (default), results are re-ordered by `rerank_score` after initial retrieval. The original scores (`fts_score`, `semantic_distance`) are preserved for debugging but `rerank_score` determines final ordering.

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
```

For detailed configuration and troubleshooting, see the [Hybrid Search Guide](hybrid-search.md).

## Search Tools Response Structure

All search tools return consistent response structures with common fields and tool-specific additions:

| Field               | search_context     | semantic_search_context | fts_search_context | hybrid_search_context |
|---------------------|--------------------|-------------------------|--------------------|-----------------------|
| `results`           | List of entries    | List of entries         | List of entries    | List of entries       |
| `count`             | Yes                | Yes                     | Yes                | Yes                   |
| `query`             | No                 | Yes                     | Yes                | Yes                   |
| `stats`             | explain_query=True | explain_query=True      | explain_query=True | explain_query=True    |
| `model`             | No                 | Yes (embedding model)   | No                 | No                    |
| `mode`              | No                 | No                      | Yes (search mode)  | No                    |
| `language`          | No                 | No                      | Yes (FTS language) | No                    |
| `fusion_method`     | No                 | No                      | No                 | Yes                   |
| `search_modes_used` | No                 | No                      | No                 | Yes                   |
| `fts_count`         | No                 | No                      | No                 | Yes                   |
| `semantic_count`    | No                 | No                      | No                 | Yes                   |

**Entry Fields by Tool:**

| Entry Field                                    | search_context        | semantic_search_context    | fts_search_context         | hybrid_search_context      |
|------------------------------------------------|-----------------------|----------------------------|----------------------------|----------------------------|
| `id`, `thread_id`, `source`, `content_type`    | Yes                   | Yes                        | Yes                        | Yes                        |
| `text_content`                                 | Truncated             | Truncated                  | Truncated                  | Truncated                  |
| `summary`                                      | Yes (string)          | Yes (string)               | Yes (string)               | Yes (string)               |
| `is_text_content_truncated`                    | Yes                   | Yes                        | Yes                        | Yes                        |
| `metadata`, `tags`, `created_at`, `updated_at` | Yes                   | Yes                        | Yes                        | Yes                        |
| `images`                                       | include_images=True   | include_images=True        | include_images=True        | include_images=True        |
| `scores`                                       | No                    | Yes                        | Yes                        | Yes                        |
| `highlighted`                                  | No                    | No                         | highlight=True             | No                         |

**`summary` field**: Present in all search tool results. Populated by automatic LLM-based summary generation (enabled by default with Ollama). Contains a dense summary (token limit controlled by `SUMMARY_MAX_TOKENS`, default 4000) that is more informative than the truncated `text_content`. Empty string when summary generation is disabled or the summary has not yet been generated. See [Summary Generation Guide](summary-generation.md) for configuration.

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
- All search tools return truncated `text_content` (configurable via `SEARCH_TRUNCATION_LENGTH`, default 300 chars) with `summary` and `is_text_content_truncated` flag; use `get_context_by_ids` for full content
- For standalone FTS and semantic searches, rank fields are always `null` (no cross-method ranking)

## Navigation Tools (locate / navigate / extract)

These read-only tools complement search: grep locates exact text, navigate orients within a record, and read extracts a span. They share one Unicode code-point character-offset contract, so a `grep_context` match's offsets and a `navigate_context` node's offsets both feed directly into `read_context_range`. See [Grep, Navigation & Partial Reads](grep-navigation-partial-read.md) for when to use each.

### grep_context

Server-side grep: literal or regular-expression, line-oriented, UNRANKED pattern matching over stored `text_content`. Unlike `fts_search_context` (stemmed, ranked) it matches raw characters and returns precise match locations. Matching runs in Python — the stdlib `re` engine for literal patterns and the third-party `regex` engine for user regular expressions — so results are identical on SQLite and PostgreSQL.

**Parameters:**
- `pattern` (str, required): Literal substring (default) or regular expression to match
- `is_regex` (bool, optional): Treat `pattern` as a Python regular expression (default: False — literal substring, auto-escaped)
- `case_sensitive` (bool, optional): Match case-sensitively (default: False — Unicode-aware case-insensitive)
- `output_mode` (str, optional): `files_with_matches` (default; context_ids + match_count), `content` (each match with line + offsets + context), or `count` (per-entry tally)
- `context_lines` (int, optional): Surrounding lines before/after each match in content mode (0-100; effectively clamped to `GREP_MAX_CONTEXT_LINES`, default 20; like `grep -C`)
- `max_matches` (int, optional): Maximum total matches to return (default 100; clamped to the server cap)
- `max_entries_scanned` (int, optional): Maximum entries the scan visits (clamped to the server cap)
- `thread_id` / `source` / `tags` / `metadata_filters` / `content_type` (optional): Reuse the store's filters to scope the scan (the ripgrep glob/type analog); scoping with `thread_id` is recommended

**Returns:** `{mode, total_matches, truncated, results}`. In `content` mode each result carries `context_id`, `line_number`, `line`, `match_start`/`match_end` (code-point offsets into `text_content`), and `before`/`after` context lines; in `files_with_matches` mode `context_id` + `match_count`; in `count` mode `context_id` + `count`. `truncated` is True when matches or the scan were capped.

### navigate_context

Build a navigable Markdown-heading outline (index_tree) for one record, computed on demand from the current text — never stale, works for every entry. The synthetic root spans the whole document and mirrors the entry summary; each node carries char offsets that feed `read_context_range`.

**Parameters:**
- `context_id` (str, required): Context entry id (32/36-char UUID or 8-31 char hex prefix)
- `max_depth` (int, optional): Deepest Markdown heading level to include (1-6, default 6; deeper headings fold into their section)
- `include_node_summaries` (bool, optional): Attach stored per-node LLM summaries to descendant nodes when that layer is enabled (default: False)

**Returns:** `{context_id, total_chars, node_count, root}` where `root` is a recursive node `{node_id, level, ordinal, title, char_start, char_end, summary, children}`. `node_id` is a heading-path slug with a sibling ordinal (e.g. `setup/install`, `notes-2`); the root's `summary` mirrors the entry summary by reference.

### read_context_range

Read part of one record by character range, line range, or outline `node_id`. Slices the full stored `text_content` directly (works for every entry regardless of embeddings).

**Parameters:**
- `context_id` (str, required): Context entry id (32/36-char UUID or 8-31 char hex prefix)
- `start_char` / `end_char` (int, optional): Character range (Unicode code-point offsets; end exclusive)
- `start_line` / `end_line` (int, optional): Line range (1-based, inclusive)
- `node_id` (str, optional): An outline node id from `navigate_context`

Provide exactly ONE addressing mode. Pair with `grep_context` (content mode) by feeding `match_start`/`match_end` into `start_char`/`end_char`, or with `navigate_context` by passing a section's `node_id`.

**Returns:** `{context_id, start_char, end_char, start_line, end_line, text}` echoing the RESOLVED span. Out-of-range offsets are clamped to `[0, len(text)]`, so a stale offset or node_id from a prior turn degrades gracefully.

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
  - `context_id` (str, required): 32-character hex or 36-character hyphenated UUID, or an 8-31 character hex prefix resolved against stored entries (zero matches or an ambiguous prefix returns an error).
  - `text` (str, optional), `metadata` (dict, optional), `metadata_patch` (dict, optional)
  - `tags` (list, optional), `images` (list, optional)
- `atomic` (bool, optional): If true, all succeed or all fail (default: true)

**Note:** `metadata_patch` uses RFC 7396 JSON Merge Patch semantics. See [Metadata Guide](metadata-addition-updating-and-filtering.md#partial-metadata-updates-metadata_patch) for details.

**Returns:** Dictionary with success, total, succeeded, failed, results array, message

### delete_context_batch

Delete multiple context entries by various criteria. **IRREVERSIBLE.**

**Parameters:**
- `context_ids` (list[str], optional): Specific 32-character hex or 36-character hyphenated UUID context IDs to delete, or 8-31 character hex prefixes resolved independently per element (zero matches or an ambiguous prefix returns an error).
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

- **Summary Generation**: [Summary Generation Guide](summary-generation.md) - LLM-based summary generation setup
- **Database Backends**: [Database Backends Guide](database-backends.md) - database configuration
- **Semantic Search**: [Semantic Search Guide](semantic-search.md) - vector similarity search setup
- **Full-Text Search**: [Full-Text Search Guide](full-text-search.md) - FTS configuration and usage
- **Hybrid Search**: [Hybrid Search Guide](hybrid-search.md) - combined FTS + semantic search
- **Metadata Filtering**: [Metadata Guide](metadata-addition-updating-and-filtering.md) - metadata operators
- **Docker Deployment**: [Docker Deployment Guide](deployment/docker.md) - containerized deployment
- **Authentication**: [Authentication Guide](authentication.md) - HTTP transport authentication
- **Main Documentation**: [README.md](../README.md) - overview and quick start
