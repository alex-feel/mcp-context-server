# Full-Text Search Guide

## Introduction

Full-Text Search (FTS) enables linguistic search with stemming, ranking, and boolean queries. Unlike semantic search which finds content by meaning, FTS performs traditional text search with advanced features:

- Finding exact phrases within context entries
- Prefix matching for partial word searches
- Boolean operators (AND, OR, NOT) for complex queries
- Language-specific stemming (e.g., "running" matches "run")
- BM25/ts_rank relevance scoring
- Highlighted snippets in search results

This feature is **optional** and can be enabled alongside or independently of semantic search.

## Prerequisites

Full-Text Search has minimal requirements as it uses built-in database capabilities:

- **SQLite**: Version 3.35+ with FTS5 extension (included by default in Python 3.9+)
- **PostgreSQL**: Version 10+ with built-in tsvector support (no extensions required)
- **Python**: 3.12+ (already required by MCP Context Server)

No additional Python packages or external services are required.

## Installation

FTS functionality is built into the core MCP Context Server - no additional installation steps are needed. Simply enable it via environment variable.

## Configuration

### Environment Variables

Enable full-text search by setting these environment variables in your MCP configuration:

#### ENABLE_FTS (Required)

- **Type**: Boolean
- **Default**: `false`
- **Description**: Master switch for full-text search functionality
- **Example**: `"ENABLE_FTS": "true"`

#### FTS_LANGUAGE (Optional)

- **Type**: String
- **Default**: `english`
- **Description**: Language for stemming and text search
- **Example**: `"FTS_LANGUAGE": "german"`

**Supported Languages** (29 total):

```
arabic, armenian, basque, catalan, danish, dutch, english, finnish, french,
german, greek, hindi, hungarian, indonesian, irish, italian, lithuanian,
nepali, norwegian, portuguese, romanian, russian, serbian, simple, spanish,
swedish, tamil, turkish, yiddish
```

**Important Notes**:
- PostgreSQL uses this language for full stemming support (all 29 languages)
- SQLite uses `porter unicode61` tokenizer for English (with stemming) or `unicode61` for other languages (no stemming, but proper Unicode tokenization)
- Invalid language values are rejected at startup with a clear error message

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
        "mcp-context-server"
      ],
      "env": {
        "ENABLE_FTS": "true",
        "FTS_LANGUAGE": "english"
      }
    }
  }
}
```

## Backend-Specific Implementation

### SQLite (FTS5)

SQLite uses the FTS5 extension with external content mode for efficient full-text search.

**Features**:
- **BM25 ranking**: Industry-standard relevance scoring algorithm
- **Tokenizer selection**: Porter stemmer for English, unicode61 for other languages
- **External content**: FTS index references main table (no data duplication)
- **Auto-sync triggers**: Index automatically updates on INSERT/UPDATE/DELETE

**Tokenizer Behavior**:

| FTS_LANGUAGE | Tokenizer | Stemming | "running" matches "run" |
|--------------|-----------|----------|-------------------------|
| `english` | `porter unicode61` | Yes | Yes |
| Other languages | `unicode61` | No | No |

**SQLite Stemming Limitation**: SQLite FTS5 only supports English stemming via the Porter stemmer. For non-English languages, use PostgreSQL for full stemming support, or use prefix mode (`mode='prefix'`) to match word beginnings.

### PostgreSQL (tsvector)

PostgreSQL uses native tsvector/tsquery functionality with GIN indexing.

**Features**:
- **ts_rank scoring**: PostgreSQL's built-in relevance ranking
- **29 language support**: Full linguistic stemming for all supported languages
- **GIN index**: Optimized for fast full-text searches
- **Generated column**: tsvector automatically computed and stored

**PostgreSQL Advantages**:
- Full stemming support for all 29 languages
- More sophisticated query parsing with `websearch_to_tsquery`
- Native phrase search support with `phraseto_tsquery`

## Usage

### New Tool: fts_search_context

When FTS is enabled, a new MCP tool becomes available.

**Parameters**:
- `query` (str, required): Search query string
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

**Metadata Filtering**: The `metadata` and `metadata_filters` parameters work identically to `search_context`. For comprehensive documentation on operators, nested paths, and best practices, see the [Metadata Guide](metadata-addition-updating-and-filtering.md).

**Returns**:
```json
{
  "query": "original search query",
  "mode": "match",
  "results": [
    {
      "id": 123,
      "thread_id": "thread-abc",
      "source": "agent",
      "content_type": "text",
      "text_content": "matching context content",
      "metadata": {"status": "completed"},
      "created_at": "2025-12-01T10:00:00Z",
      "updated_at": "2025-12-01T10:00:00Z",
      "tags": ["tag1", "tag2"],
      "score": 2.45,
      "highlighted": "matching <mark>context</mark> content"
    }
  ],
  "count": 1,
  "language": "english",
  "stats": {
    "execution_time_ms": 12.34,
    "filters_applied": 2,
    "rows_returned": 1,
    "query_plan": "..."
  }
}
```

**Note:** The `stats` field is only included when `explain_query=True`.

**Score**: Higher values indicate better relevance (BM25 for SQLite, ts_rank for PostgreSQL).

### Search Modes

#### Match Mode (Default)

Standard word matching with implicit AND between terms.

```
fts_search_context(query="error handling", mode="match")
```

Finds entries containing both "error" AND "handling" (with stemming applied).

#### Prefix Mode

Matches words starting with the query terms. Useful for autocomplete-style searches.

```
fts_search_context(query="implement", mode="prefix")
```

Finds entries containing words starting with "implement" (implements, implementation, implementing).

**SQLite**: Transforms query to `implement*`
**PostgreSQL**: Transforms query to `implement:*`

#### Phrase Mode

Matches exact phrases in order.

```
fts_search_context(query="connection refused", mode="phrase")
```

Finds entries containing the exact phrase "connection refused" (words must appear adjacent and in order).

#### Boolean Mode

Supports complex queries with AND, OR, NOT operators.

```
fts_search_context(query="python OR javascript NOT typescript", mode="boolean")
```

**SQLite syntax**: `word1 AND word2`, `word1 OR word2`, `NOT word`
**PostgreSQL syntax**: Uses `websearch_to_tsquery` which accepts Google-like syntax with quotes, `-` for NOT, and `OR`

### Example Use Cases

1. **Simple search**: Find entries mentioning specific terms
   ```
   fts_search_context(query="authentication", mode="match")
   ```

2. **Prefix autocomplete**: Find entries with partial word matches
   ```
   fts_search_context(query="auth", mode="prefix")
   ```

3. **Exact phrase**: Find specific error messages
   ```
   fts_search_context(query="connection timed out", mode="phrase")
   ```

4. **Boolean query**: Complex search with operators
   ```
   fts_search_context(query="database AND (postgres OR sqlite) NOT memory", mode="boolean")
   ```

5. **Filtered search**: Combine FTS with metadata filtering
   ```
   fts_search_context(
       query="performance",
       thread_id="current-task",
       metadata={"status": "completed"},
       highlight=True
   )
   ```

6. **Time-bounded search**: Find recent content matching terms
   ```
   fts_search_context(query="deployment", start_date="2025-11-01", end_date="2025-11-30")
   ```

7. **Highlighted snippets**: Get marked-up results for display
   ```
   fts_search_context(query="error handling", highlight=True)
   ```
   Returns `highlighted` field with `<mark>` tags around matched terms.

### Performance Characteristics

- **Index Updates**: Automatic via triggers (SQLite) or generated column (PostgreSQL)
- **Search Speed**: O(log n) with proper indexing
- **Storage Impact**: ~10-20% of text_content size for index
- **Concurrent Access**: Full support via WAL (SQLite) or MVCC (PostgreSQL)

## Verification

### Complete Setup Checklist

1. **Verify environment variable**:
   ```bash
   echo $ENABLE_FTS  # Linux/macOS
   echo %ENABLE_FTS% # Windows
   # Should show: true
   ```

2. **Start server with FTS enabled**:
   ```bash
   # Set environment variable
   export ENABLE_FTS=true  # Linux/macOS
   set ENABLE_FTS=true     # Windows

   # Start server
   uv run mcp-context-server
   ```

3. **Check server logs** for:
   ```
   [OK] FTS enabled and available
   [OK] fts_search_context registered
   ```

4. **Verify MCP client** - List available tools and confirm `fts_search_context` is present

5. **Test functionality**:
   ```
   fts_search_context(query="test", mode="match")
   ```

### Verify FTS Index Status

Call `get_statistics` to check FTS availability:

```json
{
  "fts": {
    "available": true,
    "indexed_entries": 1000,
    "total_entries": 1000,
    "coverage_percentage": 100.0,
    "backend": "sqlite",
    "engine": "fts5"
  }
}
```

## Troubleshooting

### Issue 1: fts_search_context Not Available

**Error**: `fts_search_context not available` or tool not listed

**Diagnostic Steps**:

1. **Check environment variable**:
   ```bash
   echo $ENABLE_FTS  # Must show: true
   ```

2. **Check server logs** for FTS initialization messages

3. **Call `get_statistics` tool**: Check `fts.available` field in response

**Solution**: Ensure `ENABLE_FTS=true` is set in your environment or MCP configuration.

### Issue 2: Invalid FTS_LANGUAGE Error

**Error**: `ValueError: FTS_LANGUAGE='xyz' is not a valid PostgreSQL text search configuration`

**Cause**: Invalid language name specified

**Solution**: Use one of the 29 supported language names (see Configuration section above).

### Issue 3: No Stemming with Non-English Language (SQLite)

**Symptom**: Search for "running" doesn't match "run" when using German, French, etc.

**Cause**: SQLite FTS5 only supports English stemming via Porter stemmer

**Solutions**:

1. **Use prefix mode**: Query with `mode='prefix'` to match word beginnings
   ```
   fts_search_context(query="runn", mode="prefix")
   ```

2. **Switch to PostgreSQL**: PostgreSQL supports full stemming for all 29 languages

3. **Accept limitation**: Unicode61 tokenizer still provides proper Unicode tokenization for all languages

### Issue 4: Empty Search Results

**Possible Causes**:
- Query terms don't exist in stored context
- Wrong search mode for query syntax
- Date filters excluding all results

**Diagnostic Steps**:

1. Try broader search without filters:
   ```
   fts_search_context(query="the", mode="match", limit=10)
   ```

2. Check if data exists:
   ```
   search_context(limit=10)
   ```

3. Verify FTS index coverage via `get_statistics`

### Issue 5: Boolean Query Syntax Errors

**Symptom**: Boolean queries not working as expected

**Solution**: Match syntax to backend:

**SQLite** uses FTS5 syntax:
- AND: `word1 AND word2`
- OR: `word1 OR word2`
- NOT: `NOT word`
- Grouping: `(word1 OR word2) AND word3`

**PostgreSQL** uses websearch_to_tsquery syntax:
- OR: `word1 or word2`
- NOT: `-word` (dash prefix)
- Phrase: `"exact phrase"`
- Default is AND between terms

### Common Error Messages

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `fts_search_context not available` | FTS not enabled | Set `ENABLE_FTS=true` |
| `Invalid FTS_LANGUAGE` | Unknown language | Use valid language name |
| `FTS5 table not found` | Migration not applied | Restart server to apply migration |
| `text_search_vector column missing` | PostgreSQL migration incomplete | Restart server to apply migration |

## Changing FTS Language

**Important**: Changing `FTS_LANGUAGE` after initial setup requires re-indexing.

### Migration Behavior

When you change `FTS_LANGUAGE`:

1. **SQLite**: FTS5 virtual table is dropped and recreated with new tokenizer
2. **PostgreSQL**: Generated column is dropped and recreated with new language

Both operations automatically rebuild the index from existing data.

### Steps to Change Language

1. **Stop the MCP server**

2. **Update environment variable**:
   ```bash
   export FTS_LANGUAGE=german  # New language
   ```

3. **Restart the server** - migration runs automatically

4. **Verify via logs**:
   ```
   [INFO] FTS language changed from 'english' to 'german'
   [INFO] Rebuilding FTS index...
   [OK] FTS index rebuilt successfully
   ```

### Data Safety

- Context entries are **never deleted** during language migration
- Only the FTS index is rebuilt
- Queries may return empty results during rebuild

## Comparison: FTS vs Semantic Search

| Feature | Full-Text Search | Semantic Search |
|---------|------------------|-----------------|
| **Query Type** | Keywords/phrases | Natural language meaning |
| **Stemming** | Yes (language-specific) | N/A |
| **Synonym Support** | No | Yes (via embeddings) |
| **Phrase Matching** | Yes | Approximate |
| **Boolean Operators** | Yes | No |
| **External Dependencies** | None | Ollama + embedding model |
| **Storage Overhead** | ~10-20% | ~3KB per entry |
| **Best For** | Exact matches, known terms | Concept discovery, similar content |

**Recommendation**: Enable both for maximum flexibility. Use FTS when you know the exact terms, semantic search when exploring related concepts.

## Additional Resources

### Official Documentation

- **SQLite FTS5**: [sqlite.org/fts5.html](https://www.sqlite.org/fts5.html)
- **PostgreSQL Full Text Search**: [postgresql.org/docs/current/textsearch.html](https://www.postgresql.org/docs/current/textsearch.html)

### Algorithm References

- **BM25**: [en.wikipedia.org/wiki/Okapi_BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- **Porter Stemmer**: [tartarus.org/martin/PorterStemmer](https://tartarus.org/martin/PorterStemmer/)

### Related Guides

- **Metadata Filtering**: [Metadata Guide](metadata-addition-updating-and-filtering.md) - filtering results with metadata operators
- **Semantic Search**: [Semantic Search Guide](semantic-search.md) - meaning-based search with embeddings
- **Hybrid Search**: [Hybrid Search Guide](hybrid-search.md) - combined FTS + semantic search with RRF fusion
