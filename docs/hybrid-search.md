# Hybrid Search Guide

## Introduction

Hybrid Search combines the strengths of Full-Text Search (FTS) and Semantic Search using Reciprocal Rank Fusion (RRF), a proven algorithm used by Elasticsearch, Weaviate, and other search platforms. This approach leverages:

- **Full-Text Search**: Exact keyword matching, stemming, phrase search, and boolean queries
- **Semantic Search**: Meaning-based similarity using vector embeddings
- **RRF Fusion**: Rank-based algorithm that combines results without score normalization

This combination is particularly powerful for:

- Finding content that matches both keywords AND meaning
- Boosting results that appear in both search methods (high confidence matches)
- Graceful fallback when one search method is unavailable
- Cross-domain queries where exact terms and related concepts matter

This feature is **optional** and requires enabling hybrid search along with at least one underlying search method.

## Prerequisites

Hybrid Search requires at least one of the following search methods to be enabled:

- **Full-Text Search (FTS)**: Set `ENABLE_FTS=true` (no additional dependencies)
- **Semantic Search**: Set `ENABLE_SEMANTIC_SEARCH=true` (requires Ollama + embedding model)

For maximum effectiveness, enable both:

```bash
ENABLE_FTS=true
ENABLE_SEMANTIC_SEARCH=true
ENABLE_HYBRID_SEARCH=true
```

**Dependencies by Search Mode:**

| Search Mode | Dependencies | Setup Guide |
|-------------|--------------|-------------|
| FTS only | None (built-in) | [Full-Text Search Guide](full-text-search.md) |
| Semantic only | Ollama, embedding model, sqlite-vec/pgvector | [Semantic Search Guide](semantic-search.md) |
| Both (recommended) | All above | Both guides |

## Installation

Hybrid Search uses existing FTS and Semantic Search infrastructure. No additional installation is required beyond the dependencies for your chosen search modes.

For full hybrid search capability:

```bash
# Install semantic search dependencies
uv sync --extra semantic-search

# Pull embedding model
ollama pull embeddinggemma:latest
```

## Configuration

### Environment Variables

Enable hybrid search by setting these environment variables in your MCP configuration:

#### ENABLE_HYBRID_SEARCH (Required)

- **Type**: Boolean
- **Default**: `false`
- **Description**: Master switch for hybrid search functionality
- **Example**: `"ENABLE_HYBRID_SEARCH": "true"`

**Note**: Setting `ENABLE_HYBRID_SEARCH=true` alone is not sufficient. You must also have at least one of `ENABLE_FTS=true` or `ENABLE_SEMANTIC_SEARCH=true` (or both).

#### HYBRID_RRF_K (Optional)

- **Type**: Integer
- **Default**: `60`
- **Range**: 1-1000
- **Description**: RRF smoothing constant controlling how much emphasis is placed on top-ranked vs. lower-ranked documents
- **Example**: `"HYBRID_RRF_K": "60"`

**Understanding RRF k Parameter:**

| k Value | Behavior |
|---------|----------|
| Lower (10-30) | More emphasis on top-ranked documents; larger score differences between ranks |
| Default (60) | Balanced approach; industry standard used by Elasticsearch |
| Higher (100+) | More uniform treatment across all ranks; smaller score differences |

### MCP Configuration Example

Add to your `.mcp.json` file:

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--python",
        "3.12",
        "--with",
        "mcp-context-server[semantic-search]",
        "mcp-context-server"
      ],
      "env": {
        "ENABLE_FTS": "true",
        "ENABLE_SEMANTIC_SEARCH": "true",
        "ENABLE_HYBRID_SEARCH": "true",
        "HYBRID_RRF_K": "60"
      }
    }
  }
}
```

## How RRF Fusion Works

### The RRF Algorithm

Reciprocal Rank Fusion combines results from multiple search methods using a simple yet effective formula:

```
RRF_score(d) = sum(1 / (k + rank_i(d))) for each search method i
```

Where:
- `d` is a document
- `k` is the smoothing constant (default: 60)
- `rank_i(d)` is the rank of document `d` in result list `i` (1-based)

### Why RRF Works

**1. Rank-based, not score-based:** RRF uses positions rather than raw scores, avoiding the need to normalize different scoring systems (BM25 vs L2 distance).

**2. Documents in both lists score higher:** A document ranked #1 in FTS and #1 in semantic search gets:
```
RRF = 1/(60+1) + 1/(60+1) = 0.0328
```
While a document ranked #1 in only one list gets:
```
RRF = 1/(60+1) = 0.0164
```

**3. Graceful handling of unique results:** Documents appearing in only one search method still receive a score and can rank highly if their single-source rank is good.

### Visual Example

```
FTS Results:           Semantic Results:      After RRF Fusion:
1. Doc A (score 2.5)   1. Doc B (dist 0.15)   1. Doc B (rrf 0.0328) [in both]
2. Doc B (score 2.1)   2. Doc C (dist 0.22)   2. Doc A (rrf 0.0164) [FTS only]
3. Doc D (score 1.8)   3. Doc A (dist 0.35)   3. Doc C (rrf 0.0164) [semantic only]
                                              4. Doc D (rrf 0.0159) [FTS only]
```

Doc B appears in both lists (FTS rank 2, semantic rank 1), so it scores highest after fusion.

## Usage

### New Tool: hybrid_search_context

When hybrid search is enabled and at least one underlying search method is available, a new MCP tool becomes available.

**Parameters:**

- `query` (str, required): Natural language search query
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `search_modes` (list, optional): Which search modes to use - `['fts', 'semantic']` (default: both)
- `fusion_method` (str, optional): Fusion algorithm - currently only `'rrf'` supported
- `rrf_k` (int, optional): RRF smoothing constant (1-1000, default from settings)
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

**Metadata Filtering**: The `metadata` and `metadata_filters` parameters work identically to `search_context`. For comprehensive documentation on operators, nested paths, and best practices, see the [Metadata Guide](metadata-addition-updating-and-filtering.md).

**Returns:**

```json
{
  "query": "authentication implementation",
  "results": [
    {
      "id": 123,
      "thread_id": "project-alpha",
      "source": "agent",
      "content_type": "text",
      "text_content": "Implemented JWT authentication...",
      "metadata": {"status": "completed", "priority": 8},
      "created_at": "2025-12-01T10:00:00Z",
      "updated_at": "2025-12-01T10:00:00Z",
      "tags": ["auth", "backend"],
      "scores": {
        "rrf": 0.0328,
        "fts_rank": 2,
        "semantic_rank": 1,
        "fts_score": 2.45,
        "semantic_distance": 0.234
      }
    }
  ],
  "count": 15,
  "fusion_method": "rrf",
  "search_modes_used": ["fts", "semantic"],
  "fts_count": 12,
  "semantic_count": 10,
  "stats": {
    "execution_time_ms": 125.5,
    "fts_stats": {
      "execution_time_ms": 15.2,
      "filters_applied": 2,
      "rows_returned": 12,
      "query_plan": "..."
    },
    "semantic_stats": {
      "execution_time_ms": 85.3,
      "embedding_generation_ms": 45.1,
      "filters_applied": 2,
      "rows_returned": 10,
      "query_plan": "..."
    },
    "fusion_stats": {
      "rrf_k": 60,
      "total_unique_documents": 15,
      "documents_in_both": 7,
      "documents_fts_only": 5,
      "documents_semantic_only": 3
    }
  }
}
```

**Note:** The `stats` field is only included when `explain_query=True`.

### Understanding the Stats Field

When `explain_query=True`, the response includes a `stats` object with detailed execution statistics:

| Field | Type | Description |
|-------|------|-------------|
| `execution_time_ms` | float | Total hybrid search execution time |
| `fts_stats` | object or null | FTS search statistics (null if FTS not used) |
| `semantic_stats` | object or null | Semantic search statistics (null if semantic not used) |
| `fusion_stats` | object | RRF fusion statistics |

**FTS Stats:**
- `execution_time_ms`: FTS search execution time
- `filters_applied`: Number of metadata/date filters applied
- `rows_returned`: Number of FTS results before fusion
- `query_plan`: Query execution plan details (when explain_query=True)

**Semantic Stats:**
- `execution_time_ms`: Semantic search execution time
- `embedding_generation_ms`: Time spent generating query embedding via Ollama
- `filters_applied`: Number of metadata/date filters applied
- `rows_returned`: Number of semantic results before fusion
- `query_plan`: Query execution plan details (when explain_query=True)

**Fusion Stats:**
- `rrf_k`: RRF smoothing constant used
- `total_unique_documents`: Total unique documents after fusion
- `documents_in_both`: Documents found by both FTS and semantic search (high confidence)
- `documents_fts_only`: Documents found only by FTS
- `documents_semantic_only`: Documents found only by semantic search

### Understanding the Scores Field

Each result includes a `scores` object with detailed breakdown:

| Field | Type | Description |
|-------|------|-------------|
| `rrf` | float | Combined RRF score (higher = better) |
| `fts_rank` | int or null | Position in FTS results (1-based), null if not in FTS results |
| `semantic_rank` | int or null | Position in semantic results (1-based), null if not in semantic results |
| `fts_score` | float or null | Original FTS relevance score (BM25/ts_rank) |
| `semantic_distance` | float or null | Original semantic distance (L2, lower = more similar) |

**Interpreting null values:**

- `fts_rank: null, semantic_rank: 3` - Document found only via semantic search
- `fts_rank: 1, semantic_rank: null` - Document found only via FTS
- Both non-null - Document found by both methods (high confidence match)

### Graceful Degradation

Hybrid search automatically adapts when search methods are unavailable:

| FTS Available | Semantic Available | Behavior |
|---------------|-------------------|----------|
| Yes | Yes | Full hybrid search with RRF fusion |
| Yes | No | FTS results only (semantic_rank always null) |
| No | Yes | Semantic results only (fts_rank always null) |
| No | No | Error: "No search modes available" |

**Common scenarios for partial availability:**

- Semantic unavailable: Ollama not running, embedding model not pulled
- FTS unavailable: FTS migration in progress, database corruption

The `search_modes_used` field in the response indicates which modes were actually executed.

### Example Use Cases

**1. High-confidence document discovery:**
Find documents matching both keywords AND meaning:
```python
hybrid_search_context(query="authentication token validation")
# Results with both fts_rank and semantic_rank are high-confidence matches
```

**2. Fallback to available search:**
Use hybrid even when uncertain which search methods are running:
```python
hybrid_search_context(query="error handling patterns")
# Works with whatever is available
```

**3. Filtered hybrid search:**
Combine hybrid search with metadata filtering:
```python
hybrid_search_context(
    query="performance optimization",
    thread_id="project-alpha",
    metadata={"status": "completed"},
    metadata_filters=[{"key": "priority", "operator": "gte", "value": 7}]
)
```

**4. Time-bounded search:**
Find matching content within a specific date range:
```python
hybrid_search_context(
    query="deployment issues",
    start_date="2025-11-01",
    end_date="2025-11-30"
)
```

**5. Single-mode hybrid search:**
Use hybrid infrastructure but restrict to one method:
```python
# FTS only through hybrid API
hybrid_search_context(query="exact phrase match", search_modes=["fts"])

# Semantic only through hybrid API
hybrid_search_context(query="conceptually similar", search_modes=["semantic"])
```

**6. Tuning RRF for your use case:**
Adjust k parameter for different ranking behaviors:
```python
# Emphasize top results more (lower k)
hybrid_search_context(query="critical bug", rrf_k=30)

# More uniform treatment of all ranks (higher k)
hybrid_search_context(query="general information", rrf_k=100)
```

### Performance Characteristics

Hybrid search executes FTS and semantic search in parallel for optimal performance:

| Operation | SQLite | PostgreSQL |
|-----------|--------|------------|
| FTS search | 10-50ms | 5-30ms |
| Semantic search | 50-200ms | 30-100ms |
| Hybrid (parallel) | 55-220ms | 35-115ms |
| RRF fusion | <1ms | <1ms |

**Performance notes:**

- Searches run in parallel via `asyncio.gather`
- Over-fetch strategy (limit * 2) improves fusion quality with minimal overhead
- RRF fusion is extremely fast (simple arithmetic operations)
- Total time dominated by slower search method (usually semantic)

## Verification

### Complete Setup Checklist

1. **Verify FTS is enabled:**
   ```bash
   echo $ENABLE_FTS  # Should show: true
   ```

2. **Verify semantic search is enabled (if using):**
   ```bash
   echo $ENABLE_SEMANTIC_SEARCH  # Should show: true
   curl http://localhost:11434   # Should return: Ollama is running
   ollama list                   # Should show your embedding model
   ```

3. **Verify hybrid search is enabled:**
   ```bash
   echo $ENABLE_HYBRID_SEARCH  # Should show: true
   ```

4. **Start server and check logs:**
   ```bash
   uv run mcp-context-server
   ```
   Look for:
   ```
   [OK] Hybrid search enabled
   [OK] hybrid_search_context registered
   ```

5. **Verify MCP client** - List available tools and confirm `hybrid_search_context` is present

6. **Test functionality:**
   ```python
   hybrid_search_context(query="test query", limit=5)
   ```

### Verify via get_statistics

Call `get_statistics` to check hybrid search availability:

```json
{
  "fts": {
    "available": true,
    "indexed_entries": 1000
  },
  "semantic_search": {
    "available": true,
    "indexed_entries": 1000
  }
}
```

Both FTS and semantic search should show as available for full hybrid functionality.

## Troubleshooting

### Issue 1: hybrid_search_context Not Available

**Error**: Tool not listed or "Hybrid search is not available"

**Diagnostic Steps:**

1. **Check environment variables:**
   ```bash
   echo $ENABLE_HYBRID_SEARCH  # Must show: true
   echo $ENABLE_FTS            # Should show: true
   echo $ENABLE_SEMANTIC_SEARCH # Should show: true (for full hybrid)
   ```

2. **Check server logs** for initialization messages

3. **Call `get_statistics` tool** to verify underlying search methods

**Solution**: Ensure `ENABLE_HYBRID_SEARCH=true` and at least one of `ENABLE_FTS=true` or `ENABLE_SEMANTIC_SEARCH=true`.

### Issue 2: Only FTS or Only Semantic Results

**Symptom**: Results have `semantic_rank: null` for all entries or `fts_rank: null` for all entries

**Cause**: One search method is unavailable

**Diagnostic Steps:**

1. Check `search_modes_used` in response - shows which modes actually executed
2. Check `fts_count` and `semantic_count` in response

**For missing semantic search:**
- Verify Ollama is running: `curl http://localhost:11434`
- Verify model is available: `ollama list`
- Check `ENABLE_SEMANTIC_SEARCH=true`

**For missing FTS:**
- Check `ENABLE_FTS=true`
- Verify FTS migration completed (check server logs)

### Issue 3: Poor Fusion Quality

**Symptom**: Results don't seem to combine well, or single-source results dominate

**Possible Causes:**

1. **Very different result sets**: FTS and semantic may return completely different documents
2. **One search returning few results**: Limited overlap for fusion
3. **Inappropriate k value**: May need tuning

**Solutions:**

1. **Check overlap**: Look at results - are any entries in both search methods?
2. **Increase limit**: More results = more potential overlap
3. **Tune k parameter**:
   - Lower k (30) for more top-heavy ranking
   - Higher k (100) for more uniform treatment
4. **Verify data has embeddings**: Check `get_statistics` for embedding coverage

### Issue 4: Slow Hybrid Search

**Symptom**: Searches taking longer than expected

**Cause**: Usually semantic search is the bottleneck

**Solutions:**

1. **Check Ollama performance**: Ensure model is loaded in memory
   ```bash
   ollama ps  # Shows loaded models
   ```

2. **Increase Ollama keep-alive:**
   ```bash
   export OLLAMA_KEEP_ALIVE=3600  # Keep model loaded
   ```

3. **Consider single-mode search** if only keywords needed:
   ```python
   hybrid_search_context(query="exact match", search_modes=["fts"])
   ```

### Common Error Messages

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `Hybrid search is not available` | Feature not enabled | Set `ENABLE_HYBRID_SEARCH=true` |
| `No search modes available` | Neither FTS nor semantic enabled | Enable at least one search method |
| `FTS requires ENABLE_FTS=true` | Requested FTS but not enabled | Set `ENABLE_FTS=true` |
| `Semantic search requires...` | Requested semantic but dependencies missing | Set up Ollama and semantic search |
| `All search modes failed` | Both FTS and semantic errored | Check individual search method status |

## Comparison: Hybrid vs Individual Search Methods

| Feature | FTS | Semantic | Hybrid |
|---------|-----|----------|--------|
| **Query Type** | Keywords/phrases | Natural language meaning | Both |
| **Result Ranking** | BM25/ts_rank score | L2 distance | RRF combined score |
| **Best For** | Exact matches, known terms | Concept discovery | High-confidence matches |
| **Performance** | Fastest | Slower | Similar to semantic (parallel) |
| **Dependencies** | None | Ollama + model | At least one method |
| **Graceful Degradation** | N/A | N/A | Falls back to available method |

**When to use each:**

- **FTS**: Known exact terms, phrase matching, boolean queries
- **Semantic**: Exploring related concepts, meaning-based retrieval
- **Hybrid**: Best of both, high-confidence discovery, uncertain query type

## Additional Resources

### Related Guides

- [Full-Text Search Guide](full-text-search.md) - FTS configuration and usage
- [Semantic Search Guide](semantic-search.md) - Semantic search setup with Ollama
- [Metadata Guide](metadata-addition-updating-and-filtering.md) - Metadata filtering with operators

### Algorithm References

- **Reciprocal Rank Fusion**: [Cormack et al., 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- **Elasticsearch RRF**: [Elastic documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html)
- **Weaviate Hybrid Search**: [Weaviate documentation](https://weaviate.io/developers/weaviate/search/hybrid)

### Implementation Files

- [`app/fusion.py`](../app/fusion.py) - RRF fusion algorithm implementation
- [`app/server.py`](../app/server.py) - hybrid_search_context tool definition
- [`app/settings.py`](../app/settings.py) - Configuration settings
- [`app/types.py`](../app/types.py) - TypedDict definitions for hybrid search
