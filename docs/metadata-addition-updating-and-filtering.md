# Metadata Guide

## Introduction

Metadata in the MCP Context Server provides a powerful way to enrich, organize, and query your context entries. This guide covers the complete lifecycle of metadata: **adding** it when storing context, **updating** it as your workflow evolves, and **filtering** by it when searching.

**Key Capabilities:**
- **Flexible Structure**: Store any JSON-serializable data (strings, numbers, booleans, arrays, nested objects)
- **15 Operators**: From simple equality to advanced pattern matching and range queries
- **Performance Optimized**: Strategic indexing for common fields (status, priority, agent_name, task_name, completed)
- **Case Control**: Case-sensitive and case-insensitive string operations
- **Partial Updates**: RFC 7396 JSON Merge Patch for selective metadata modifications
- **Query Statistics**: Execution time and query plan analysis

**Available in Multiple Tools:**
- **`store_context`**: Add metadata when creating context entries
- **`update_context`**: Modify metadata on existing entries (full replacement or partial patch)
- **`search_context`**: Keyword and filter-based search with metadata filtering
- **`semantic_search_context`**: Vector similarity search with metadata filtering (requires semantic search enabled)
- **`fts_search_context`**: Full-text search with metadata filtering (requires FTS enabled)
- **`hybrid_search_context`**: Combined FTS + semantic search with metadata filtering (requires hybrid search enabled)

All search tools support identical metadata filtering syntax and return consistent error responses.

## Metadata Structure

### Flexible JSON Schema

The `metadata` field accepts any JSON-serializable structure with no predefined schema:

```json
{
  "status": "active",
  "priority": 5,
  "assignee": "alice@example.com",
  "due_date": "2025-10-15",
  "tags": ["urgent", "backend"],
  "config": {
    "retries": 3,
    "timeout": 30
  }
}
```

**Supported Types:**
- Strings: `"active"`, `"completed"`
- Numbers: `42`, `3.14`, `0`
- Booleans: `true`, `false`
- Null: `null`
- Lists: `["value1", "value2"]`
- Nested objects: `{"user": {"id": 123}}`

### Indexed Fields for Performance

The following metadata fields are indexed for faster filtering:

| Field | Type | Use Case | Example Values |
|-------|------|----------|----------------|
| `status` | string | State tracking | `"pending"`, `"active"`, `"completed"` |
| `priority` | number | Range queries | `1-10`, `0`, `100` |
| `agent_name` | string | Agent identification | `"planner"`, `"executor"`, `"reviewer"` |
| `task_name` | string | Task identification | `"analyze-data"`, `"generate-report"` |
| `completed` | boolean | Completion state | `true`, `false` |

**Performance Note:** Using indexed fields in filters significantly improves query performance, especially with large datasets.

## Adding Metadata (store_context)

When storing context entries, you can attach metadata to provide structured information about the entry. This metadata can later be used for filtering and organization.

### Basic Usage

```python
# Store context with metadata
store_context(
    thread_id="project-alpha",
    source="agent",
    text="Implement user authentication",
    metadata={
        "status": "active",
        "priority": 8,
        "assignee": "alice@example.com"
    }
)
```

### Using Indexed Fields

For optimal query performance, use the indexed fields when possible:

```python
# Store task with indexed metadata fields
store_context(
    thread_id="project-alpha",
    source="agent",
    text="Implement user authentication",
    metadata={
        "status": "active",           # Indexed - fast filtering
        "priority": 8,                # Indexed - range queries
        "agent_name": "planner",      # Indexed - agent identification
        "task_name": "auth-impl",     # Indexed - task tracking
        "completed": False,           # Indexed - completion state
        "assignee": "alice@example.com",  # Not indexed but still queryable
        "due_date": "2025-10-20"          # Not indexed but still queryable
    }
)
```

### Complex Metadata Structures

Metadata can include nested objects for more complex data:

```python
# Store context with nested metadata
store_context(
    thread_id="config-test",
    source="agent",
    text="Configuration loaded",
    metadata={
        "user": {
            "id": 123,
            "preferences": {
                "theme": "dark",
                "notifications": True
            }
        },
        "settings": {
            "timeout": 30,
            "retries": 3
        }
    }
)
```

### Metadata Use Cases by Domain

**Task Management:**
```python
metadata={
    "status": "active",
    "priority": 8,
    "assignee": "alice@example.com",
    "due_date": "2025-10-20",
    "task_name": "auth-implementation",
    "completed": False,
    "tags": ["backend", "security"]
}
```

**Agent Coordination:**
```python
metadata={
    "agent_name": "data-processor",
    "task_name": "batch-processing",
    "execution_time": 45.3,
    "resource_usage": {
        "cpu_percent": 65.2,
        "memory_mb": 512
    },
    "records_processed": 1000,
    "status": "completed"
}
```

**Knowledge Base:**
```python
metadata={
    "category": "machine-learning",
    "subcategory": "transformers",
    "relevance_score": 8.5,
    "source_url": "https://arxiv.org/...",
    "author": "research-agent",
    "year": 2025,
    "peer_reviewed": True
}
```

**Debugging Context:**
```python
metadata={
    "error_type": "TimeoutError",
    "error_code": "DB_TIMEOUT",
    "stack_trace": "File payment.py, line 42...",
    "environment": "production",
    "version": "v2.3.1",
    "severity": "critical",
    "timestamp": "2025-10-10T08:30:00Z"
}
```

**Analytics and Events:**
```python
metadata={
    "user_id": "user_12345",
    "session_id": "sess_abc123",
    "event_type": "checkout_completed",
    "timestamp": "2025-10-10T10:15:30Z",
    "revenue": 149.99,
    "items_count": 3,
    "platform": "web"
}
```

## Updating Metadata (update_context)

The `update_context` tool provides two ways to modify metadata on existing context entries:

1. **Full Replacement** (`metadata` parameter): Replace the entire metadata object
2. **Partial Update** (`metadata_patch` parameter): Modify specific fields while preserving others

### When to Use Each Method

| Scenario | Use | Parameter |
|----------|-----|-----------|
| Replace entire metadata object | Full replacement | `metadata` |
| Update a single field | Partial update | `metadata_patch` |
| Add new fields while preserving existing | Partial update | `metadata_patch` |
| Delete specific fields | Partial update | `metadata_patch` |
| Clear all metadata | Full replacement | `metadata={}` |

**Mutual Exclusivity:** You cannot use both `metadata` and `metadata_patch` in the same call.

### Full Replacement

Use the `metadata` parameter to completely replace all metadata:

```python
# Replace all metadata
update_context(
    context_id=123,
    metadata={
        "status": "completed",
        "priority": 10,
        "reviewer": "bob"
    }
)
# Result: Only these three fields exist (previous metadata is gone)
```

### Partial Updates (metadata_patch)

The `metadata_patch` parameter implements RFC 7396 JSON Merge Patch semantics:

- **New keys** in patch are ADDED to existing metadata
- **Existing keys** are REPLACED with new values from patch
- **Null values** DELETE keys from metadata

#### Basic Operations

```python
# Original metadata: {"status": "pending", "priority": 5, "assignee": "alice"}

# Add a new field
update_context(context_id=123, metadata_patch={"category": "backend"})
# Result: {"status": "pending", "priority": 5, "assignee": "alice", "category": "backend"}

# Update existing field
update_context(context_id=123, metadata_patch={"status": "completed"})
# Result: {"status": "completed", "priority": 5, "assignee": "alice", "category": "backend"}

# Delete a field using null
update_context(context_id=123, metadata_patch={"assignee": None})
# Result: {"status": "completed", "priority": 5, "category": "backend"}

# Multiple operations in one call
update_context(
    context_id=123,
    metadata_patch={
        "status": "archived",      # Update existing
        "archived_at": "2025-10",  # Add new
        "category": None           # Delete
    }
)
# Result: {"status": "archived", "priority": 5, "archived_at": "2025-10"}
```

#### Combined with Other Updates

The `metadata_patch` can be combined with other update fields:

```python
# Update text and patch metadata in one operation
update_context(
    context_id=123,
    text="Updated analysis results",
    metadata_patch={"status": "reviewed", "reviewer": "bob"}
)

# Update tags and patch metadata
update_context(
    context_id=123,
    tags=["completed", "verified"],
    metadata_patch={"completed": True}
)
```

### Limitations (RFC 7396)

**1. Cannot Set Null Values**

Using `null` in the patch always DELETES the key. If you need to store a null value, use full metadata replacement:

```python
# This DELETES the field, not sets it to null
update_context(context_id=123, metadata_patch={"optional_field": None})
# Result: field is removed

# To store null, use full replacement
current = get_context_by_ids(context_ids=[123])
new_metadata = current[0]["metadata"]
new_metadata["optional_field"] = None
update_context(context_id=123, metadata=new_metadata)
```

**2. Arrays are Replaced Entirely**

Array operations are replace-only - no element-wise add/remove:

```python
# Original: {"tags": ["a", "b", "c"]}

# This replaces the entire array
update_context(context_id=123, metadata_patch={"tags": ["x", "y"]})
# Result: {"tags": ["x", "y"]}  (not ["a", "b", "c", "x", "y"])

# To append, read current array first
current = get_context_by_ids(context_ids=[123])
tags = current[0]["metadata"]["tags"]
tags.append("new_tag")
update_context(context_id=123, metadata_patch={"tags": tags})
```

## Filtering by Metadata

Metadata filtering enables powerful, flexible querying of context entries using structured JSON data. Unlike simple tag-based organization, metadata filtering supports complex queries with 15 operators, nested JSON paths, and performance-optimized indexes for common fields.

All search tools (`search_context`, `semantic_search_context`, `fts_search_context`, and `hybrid_search_context`) support identical metadata filtering syntax.

### Simple Filtering (Exact Match)

Use the `metadata` parameter for straightforward key-value equality matching:

```python
# Find all active contexts
search_context(
    thread_id="project-123",
    metadata={"status": "active"}
)

# Multiple conditions (AND logic)
search_context(
    thread_id="project-123",
    metadata={
        "status": "active",
        "priority": 5,
        "completed": False
    }
)
```

**Characteristics:**
- Case-insensitive string matching by default
- All conditions must match (AND logic)
- Simple syntax for common queries
- Limited to equality comparisons

### Advanced Filtering (Operators)

Use the `metadata_filters` parameter for complex queries with operators:

```python
# Priority greater than 5
search_context(
    thread_id="project-123",
    metadata_filters=[
        {"key": "priority", "operator": "gt", "value": 5}
    ]
)

# Multiple advanced filters
search_context(
    thread_id="project-123",
    metadata_filters=[
        {"key": "priority", "operator": "gte", "value": 3},
        {"key": "status", "operator": "in", "value": ["active", "pending"]},
        {"key": "agent_name", "operator": "starts_with", "value": "executor"}
    ]
)
```

**Characteristics:**
- 15 powerful operators
- Numeric comparisons and range queries
- Pattern matching for strings
- Field existence checks
- Supports nested JSON paths

### Using with Semantic Search

The same metadata filtering syntax works with `semantic_search_context`:

```python
# Semantic search with metadata filtering
semantic_search_context(
    query="authentication implementation",
    metadata={"status": "completed"},  # Simple filter
    metadata_filters=[                 # Advanced filters
        {"key": "priority", "operator": "gte", "value": 5},
        {"key": "agent_name", "operator": "exists"}
    ]
)

# Find similar contexts from a specific agent with high priority
semantic_search_context(
    query="database optimization",
    thread_id="project-123",
    metadata_filters=[
        {"key": "agent_name", "operator": "eq", "value": "research-agent"},
        {"key": "priority", "operator": "gt", "value": 7}
    ]
)
```

This combines the power of semantic similarity search with precise metadata filtering, enabling queries like "find similar content about authentication from completed, high-priority tasks."

### Using with Hybrid Search

The same metadata filtering syntax works with `hybrid_search_context`:

```python
# Hybrid search with metadata filtering
hybrid_search_context(
    query="authentication implementation",
    metadata={"status": "completed"},  # Simple filter
    metadata_filters=[                 # Advanced filters
        {"key": "priority", "operator": "gte", "value": 5},
        {"key": "agent_name", "operator": "exists"}
    ]
)

# Find high-confidence matches from a specific project
hybrid_search_context(
    query="database optimization",
    thread_id="project-123",
    metadata_filters=[
        {"key": "status", "operator": "in", "value": ["completed", "reviewed"]},
        {"key": "priority", "operator": "gt", "value": 7}
    ]
)
```

Hybrid search returns results with combined RRF scores from both FTS and semantic search, plus individual rankings from each method.

### Combining Both Methods

Simple and advanced filters can be combined in a single query:

```python
search_context(
    thread_id="project-123",
    source="agent",
    metadata={"status": "active"},  # Simple filter
    metadata_filters=[              # Advanced filters
        {"key": "priority", "operator": "gt", "value": 5},
        {"key": "agent_name", "operator": "exists"}
    ]
)
```

## Operator Reference

### Equality Operators

#### `eq` - Equals

Match exact values. Case-insensitive for strings by default.

```python
# Find completed tasks
{"key": "status", "operator": "eq", "value": "completed"}

# Case-sensitive matching
{"key": "status", "operator": "eq", "value": "Completed", "case_sensitive": True}

# Numeric equality
{"key": "priority", "operator": "eq", "value": 5}

# Boolean equality
{"key": "completed", "operator": "eq", "value": True}
```

#### `ne` - Not Equals

Match all values except the specified one.

```python
# Exclude archived entries
{"key": "status", "operator": "ne", "value": "archived"}

# Not a specific priority
{"key": "priority", "operator": "ne", "value": 0}
```

### Comparison Operators

#### `gt` - Greater Than

Numeric comparison, exclusive.

```python
# High priority tasks (priority > 5)
{"key": "priority", "operator": "gt", "value": 5}

# Recent scores
{"key": "score", "operator": "gt", "value": 80.5}
```

#### `gte` - Greater Than or Equal

Numeric comparison, inclusive.

```python
# Priority 5 or higher
{"key": "priority", "operator": "gte", "value": 5}
```

#### `lt` - Less Than

Numeric comparison, exclusive.

```python
# Low priority tasks (priority < 3)
{"key": "priority", "operator": "lt", "value": 3}
```

#### `lte` - Less Than or Equal

Numeric comparison, inclusive.

```python
# Priority 3 or lower
{"key": "priority", "operator": "lte", "value": 3}
```

### Membership Operators

#### `in` - Value in List

Check if value matches any item in the provided list. Supports both string and integer arrays.

```python
# Active or pending tasks
{"key": "status", "operator": "in", "value": ["active", "pending", "review"]}

# Specific priority levels (integer arrays)
{"key": "priority", "operator": "in", "value": [1, 5, 10]}

# Case-insensitive by default
{"key": "environment", "operator": "in", "value": ["dev", "staging", "prod"]}
```

#### `not_in` - Value Not in List

Exclude values matching any item in the list.

```python
# Exclude archived and deleted
{"key": "status", "operator": "not_in", "value": ["archived", "deleted"]}
```

### Existence Operators

#### `exists` - Field Exists

Check if a metadata field is present (not null, not missing).

```python
# Contexts with agent assignment
{"key": "agent_name", "operator": "exists"}

# Has error information
{"key": "error_message", "operator": "exists"}
```

#### `not_exists` - Field Does Not Exist

Check if a metadata field is missing or null.

```python
# Contexts without assignee
{"key": "assignee", "operator": "not_exists"}

# No completion date set
{"key": "completed_at", "operator": "not_exists"}
```

### String Pattern Operators

#### `contains` - String Contains Substring

Case-insensitive substring matching by default.

```python
# Description contains "error"
{"key": "description", "operator": "contains", "value": "error"}

# Case-sensitive search
{"key": "log_message", "operator": "contains", "value": "ERROR", "case_sensitive": True}
```

#### `starts_with` - String Starts With Prefix

Match strings beginning with specific prefix.

```python
# Agent names starting with "executor"
{"key": "agent_name", "operator": "starts_with", "value": "executor"}

# Task names starting with "test_"
{"key": "task_name", "operator": "starts_with", "value": "test_"}
```

#### `ends_with` - String Ends With Suffix

Match strings ending with specific suffix.

```python
# Files ending with .json
{"key": "filename", "operator": "ends_with", "value": ".json"}

# Email addresses ending with company domain
{"key": "email", "operator": "ends_with", "value": "@example.com"}
```

### Null Checking Operators

#### `is_null` - Value is JSON Null

Check if field contains JSON `null` value (different from missing field).

```python
# Explicitly set to null
{"key": "deleted_at", "operator": "is_null"}
```

#### `is_not_null` - Value is Not JSON Null

Check if field has a non-null value.

```python
# Has a value (not null)
{"key": "assigned_to", "operator": "is_not_null"}
```

**Note:** `exists` checks for field presence, `is_not_null` checks for non-null values. Use `exists` for missing fields, `is_not_null` for null values.

## Real-World Use Cases

### Task Management System

Track task states, priorities, assignments, and deadlines.

```python
# Store task with metadata
store_context(
    thread_id="project-alpha",
    source="agent",
    text="Implement user authentication",
    metadata={
        "status": "active",
        "priority": 8,
        "assignee": "alice@example.com",
        "due_date": "2025-10-20",
        "task_name": "auth-implementation",
        "completed": False,
        "tags": ["backend", "security"]
    }
)

# Find high-priority incomplete tasks
search_context(
    thread_id="project-alpha",
    metadata_filters=[
        {"key": "priority", "operator": "gte", "value": 7},
        {"key": "completed", "operator": "eq", "value": False},
        {"key": "status", "operator": "in", "value": ["active", "pending"]}
    ]
)

# Find overdue tasks (assignee exists but not completed)
search_context(
    thread_id="project-alpha",
    metadata_filters=[
        {"key": "assignee", "operator": "exists"},
        {"key": "completed", "operator": "eq", "value": False}
    ]
)

# Find Alice's tasks
search_context(
    thread_id="project-alpha",
    metadata={"assignee": "alice@example.com"}
)
```

### Agent Coordination

Track agent activities, resource usage, and execution metrics.

```python
# Store agent execution context
store_context(
    thread_id="data-pipeline",
    source="agent",
    text="Processed 1000 records successfully",
    metadata={
        "agent_name": "data-processor",
        "task_name": "batch-processing",
        "execution_time": 45.3,
        "resource_usage": {
            "cpu_percent": 65.2,
            "memory_mb": 512
        },
        "records_processed": 1000,
        "status": "completed"
    }
)

# Find slow executions (> 60 seconds)
search_context(
    thread_id="data-pipeline",
    metadata_filters=[
        {"key": "execution_time", "operator": "gt", "value": 60}
    ]
)

# Find specific agent activities
search_context(
    thread_id="data-pipeline",
    metadata_filters=[
        {"key": "agent_name", "operator": "starts_with", "value": "data-"}
    ]
)

# Find failed or incomplete tasks
search_context(
    thread_id="data-pipeline",
    metadata_filters=[
        {"key": "status", "operator": "in", "value": ["failed", "error", "timeout"]}
    ]
)
```

### Knowledge Base

Categorize and retrieve information with relevance scoring.

```python
# Store knowledge base entry
store_context(
    thread_id="ml-research",
    source="agent",
    text="GPT-4 architecture uses sparse attention mechanism...",
    metadata={
        "category": "machine-learning",
        "subcategory": "transformers",
        "relevance_score": 8.5,
        "source_url": "https://arxiv.org/...",
        "author": "research-agent",
        "year": 2025,
        "peer_reviewed": True
    }
)

# Find highly relevant ML papers
search_context(
    thread_id="ml-research",
    metadata_filters=[
        {"key": "category", "operator": "eq", "value": "machine-learning"},
        {"key": "relevance_score", "operator": "gte", "value": 7.0},
        {"key": "peer_reviewed", "operator": "eq", "value": True}
    ]
)

# Find recent transformer research
search_context(
    thread_id="ml-research",
    metadata={
        "subcategory": "transformers",
        "year": 2025
    }
)

# Find papers from specific source
search_context(
    thread_id="ml-research",
    metadata_filters=[
        {"key": "source_url", "operator": "contains", "value": "arxiv.org"}
    ]
)
```

### Debugging Context

Save error information, stack traces, environment details.

```python
# Store error context
store_context(
    thread_id="production-debug",
    source="agent",
    text="Database connection timeout in payment processor",
    metadata={
        "error_type": "TimeoutError",
        "error_code": "DB_TIMEOUT",
        "stack_trace": "File payment.py, line 42...",
        "environment": "production",
        "version": "v2.3.1",
        "severity": "critical",
        "timestamp": "2025-10-10T08:30:00Z"
    }
)

# Find critical production errors
search_context(
    thread_id="production-debug",
    metadata_filters=[
        {"key": "environment", "operator": "eq", "value": "production"},
        {"key": "severity", "operator": "in", "value": ["critical", "high"]}
    ]
)

# Find timeout-related errors
search_context(
    thread_id="production-debug",
    metadata_filters=[
        {"key": "error_type", "operator": "contains", "value": "timeout", "case_sensitive": False}
    ]
)

# Find errors in specific version
search_context(
    thread_id="production-debug",
    metadata_filters=[
        {"key": "version", "operator": "starts_with", "value": "v2.3"}
    ]
)
```

### Analytics and Event Tracking

Record user actions, sessions, and event metadata.

```python
# Store analytics event
store_context(
    thread_id="user-analytics",
    source="agent",
    text="User completed checkout process",
    metadata={
        "user_id": "user_12345",
        "session_id": "sess_abc123",
        "event_type": "checkout_completed",
        "timestamp": "2025-10-10T10:15:30Z",
        "revenue": 149.99,
        "items_count": 3,
        "platform": "web"
    }
)

# Find high-value purchases
search_context(
    thread_id="user-analytics",
    metadata_filters=[
        {"key": "event_type", "operator": "eq", "value": "checkout_completed"},
        {"key": "revenue", "operator": "gt", "value": 100}
    ]
)

# Find specific user activity
search_context(
    thread_id="user-analytics",
    metadata={"user_id": "user_12345"}
)

# Find mobile platform events
search_context(
    thread_id="user-analytics",
    metadata_filters=[
        {"key": "platform", "operator": "in", "value": ["ios", "android"]}
    ]
)
```

## Nested JSON Path Queries

### Accessing Nested Fields

Use dot notation to query nested metadata structures:

```python
# Store context with nested metadata
store_context(
    thread_id="config-test",
    source="agent",
    text="Configuration loaded",
    metadata={
        "user": {
            "id": 123,
            "preferences": {
                "theme": "dark",
                "notifications": True
            }
        },
        "settings": {
            "timeout": 30,
            "retries": 3
        }
    }
)

# Query nested fields with dot notation
search_context(
    thread_id="config-test",
    metadata_filters=[
        {"key": "user.preferences.theme", "operator": "eq", "value": "dark"},
        {"key": "settings.timeout", "operator": "gt", "value": 20}
    ]
)
```

**Supported Paths:**
- Single level: `"status"`
- Multi-level: `"user.preferences.theme"`
- Deep nesting: `"config.database.connection.pool.size"`

**Path Restrictions:**
- Only alphanumeric characters, dots, underscores, and hyphens allowed
- No array indexing (e.g., `items[0].name` not supported)
- No special characters or spaces

## Query Performance

### Understanding Query Execution

Use `explain_query=True` to analyze query performance:

```python
result = search_context(
    thread_id="project-123",
    metadata={"status": "active"},
    metadata_filters=[
        {"key": "priority", "operator": "gt", "value": 5}
    ],
    explain_query=True
)

# Returns execution statistics
print(result["stats"])
# {
#     "execution_time_ms": 12.34,
#     "filters_applied": 2,
#     "rows_returned": 15,
#     "query_plan": "..."
# }
```

**Statistics Provided:**
- `execution_time_ms`: Query execution time in milliseconds
- `filters_applied`: Number of metadata filters in the query
- `rows_returned`: Number of matching entries
- `query_plan`: SQLite query execution plan (when explain_query=True)

### Performance Optimization Tips

#### 1. Use Indexed Fields When Possible

Indexed fields (`status`, `priority`, `agent_name`, `task_name`, `completed`) filter significantly faster:

```python
# Fast - uses indexed field
{"key": "status", "operator": "eq", "value": "active"}

# Slower - non-indexed field
{"key": "custom_field", "operator": "eq", "value": "value"}
```

#### 2. Filter on Thread ID First

Always specify `thread_id` to reduce the search space:

```python
# Good - scoped to thread
search_context(
    thread_id="project-123",
    metadata={"status": "active"}
)

# Slower - searches all threads
search_context(
    metadata={"status": "active"}
)
```

#### 3. Order Filters by Selectivity

Place more restrictive filters first for better performance:

```python
# Better - specific agent name filters more
metadata_filters=[
    {"key": "agent_name", "operator": "eq", "value": "specific-agent"},
    {"key": "status", "operator": "in", "value": ["active", "pending", "review"]}
]

# Less optimal - broad status filter first
metadata_filters=[
    {"key": "status", "operator": "in", "value": ["active", "pending", "review"]},
    {"key": "agent_name", "operator": "eq", "value": "specific-agent"}
]
```

#### 4. Avoid Complex String Operations on Large Datasets

Pattern matching operations (`contains`, `starts_with`, `ends_with`) are slower than equality:

```python
# Faster
{"key": "status", "operator": "eq", "value": "active"}

# Slower
{"key": "description", "operator": "contains", "value": "active"}
```

### Performance Benchmarks

Based on test suite results with typical workloads:

| Query Type | Execution Time | Notes |
|------------|----------------|-------|
| Single indexed field | < 10ms | Uses database index |
| Multiple indexed fields | < 20ms | AND conditions, indexed |
| Non-indexed field | < 50ms | Full table scan on filtered results |
| Complex pattern match | < 100ms | String operations |
| Nested JSON path | < 50ms | Uses json_extract() |

**Acceptable Scale:** Up to 100,000 context entries with acceptable performance using indexed fields.

## Case Sensitivity

### Default Behavior

String operations are case-insensitive by default:

```python
# Matches "Active", "active", "ACTIVE"
{"key": "status", "operator": "eq", "value": "active"}
```

### Case-Sensitive Matching

Set `case_sensitive: true` for exact case matching:

```python
# Only matches "Active" (exact case)
{"key": "status", "operator": "eq", "value": "Active", "case_sensitive": True}
```

### Operators Supporting Case Sensitivity

Case sensitivity applies to all string operators:
- `eq`, `ne` - Equality comparisons
- `in`, `not_in` - List membership
- `contains` - Substring search
- `starts_with` - Prefix matching
- `ends_with` - Suffix matching

```python
# Case-insensitive contains (default)
{"key": "description", "operator": "contains", "value": "error"}
# Matches: "Error occurred", "error found", "ERROR: timeout"

# Case-sensitive contains
{"key": "description", "operator": "contains", "value": "ERROR", "case_sensitive": True}
# Matches: "ERROR: timeout", "FATAL ERROR"
# No match: "error found", "Error occurred"
```

## Error Handling

### Unified Error Responses

Both `search_context` and `semantic_search_context` return identical error response formats when metadata filter validation fails. This unified error handling ensures consistent client-side processing.

### Validation Errors

Invalid filters return error responses with validation details:

```python
# Invalid operator
result = search_context(
    thread_id="test",
    metadata_filters=[
        {"key": "status", "operator": "invalid_op", "value": "active"}
    ]
)

# Returns:
# {
#     "entries": [],
#     "error": "Metadata filter validation failed",
#     "validation_errors": [
#         "Invalid metadata filter {...}: ... 'invalid_op' is not a valid MetadataOperator"
#     ],
#     "execution_time_ms": 0.0,
#     "filters_applied": 0,
#     "rows_returned": 0
# }
```

### Common Validation Errors

#### 1. Empty IN/NOT_IN List

```python
# Invalid - empty list
{"key": "status", "operator": "in", "value": []}

# Error: "Operator in requires a non-empty list"
```

#### 2. Invalid Metadata Key

```python
# Invalid - special characters
{"key": "status'; DROP TABLE", "operator": "eq", "value": "x"}

# Error: "Invalid metadata key: ... Only alphanumeric characters, dots, underscores, and hyphens are allowed"
```

#### 3. Wrong Value Type for Operator

```python
# Invalid - string value for IN operator
{"key": "status", "operator": "in", "value": "active"}

# Error: "Operator in requires a list value"
```

#### 4. Multiple Filter Errors

All validation errors are collected and returned together:

```python
metadata_filters=[
    {"key": "status", "operator": "invalid_op", "value": "active"},
    {"key": "priority", "operator": "in", "value": []},
    {"key": "type", "operator": "another_invalid", "value": 5}
]

# Returns validation_errors array with all 3 errors
```

## Best Practices

### 1. Use Indexed Fields for Frequent Queries

Design your metadata schema to leverage indexed fields:

```python
# Good - uses indexed fields
metadata={
    "status": "active",          # Indexed
    "priority": 5,               # Indexed
    "agent_name": "processor",   # Indexed
    "task_name": "data-export",  # Indexed
    "completed": False           # Indexed
}

# Avoid - custom fields for frequently queried data
metadata={
    "my_custom_status": "active",  # Not indexed
    "my_priority_level": 5         # Not indexed
}
```

### 2. Choose the Right Filtering Method

- **Simple filtering** for exact matches
- **Advanced filtering** for complex queries
- **Combine both** for mixed requirements

```python
# Simple - exact match queries
search_context(metadata={"status": "active", "priority": 5})

# Advanced - range and pattern matching
search_context(metadata_filters=[
    {"key": "priority", "operator": "gte", "value": 5},
    {"key": "agent_name", "operator": "starts_with", "value": "exec"}
])

# Combined - best of both
search_context(
    metadata={"status": "active"},
    metadata_filters=[{"key": "priority", "operator": "gt", "value": 5}]
)
```

### 3. Design Consistent Metadata Schemas

Maintain consistency across agents for better filtering:

```python
# Good - consistent schema
metadata={
    "status": "active",      # Always lowercase
    "priority": 5,           # Always number 1-10
    "agent_name": "planner"  # Always lowercase, hyphen-separated
}

# Avoid - inconsistent values
metadata={
    "status": "Active",      # Mixed case
    "priority": "high",      # String instead of number
    "agent_name": "Planner"  # Mixed case
}
```

### 4. Validate Metadata Before Storage

Ensure metadata conforms to your schema:

```python
# Define schema validation
def validate_task_metadata(metadata):
    required_fields = ["status", "priority"]
    valid_statuses = ["pending", "active", "completed", "failed"]

    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Missing required field: {field}")

    if metadata["status"] not in valid_statuses:
        raise ValueError(f"Invalid status: {metadata['status']}")

    if not isinstance(metadata["priority"], int) or not (1 <= metadata["priority"] <= 10):
        raise ValueError("Priority must be an integer between 1 and 10")

# Use validation before storing
metadata = {"status": "active", "priority": 5}
validate_task_metadata(metadata)
store_context(thread_id="task", source="agent", text="...", metadata=metadata)
```

### 5. Use Explain Query for Optimization

Profile slow queries to identify optimization opportunities:

```python
result = search_context(
    thread_id="large-project",
    metadata_filters=[...],
    explain_query=True
)

# Check execution time
if result["stats"]["execution_time_ms"] > 100:
    print("Slow query detected!")
    print(result["stats"]["query_plan"])
```

### 6. Limit Result Sets

Use pagination for large result sets:

```python
# Fetch in batches
search_context(
    thread_id="large-thread",
    metadata={"status": "active"},
    limit=50,    # Results per page
    offset=0     # Start position
)
```

### 7. Combine with Other Filters

Leverage all filtering capabilities for precise queries:

```python
search_context(
    thread_id="project-123",              # Thread filter
    source="agent",                       # Source filter
    tags=["urgent", "backend"],           # Tag filter (OR logic)
    content_type="text",                  # Content type filter
    metadata={"status": "active"},        # Simple metadata filter
    metadata_filters=[                    # Advanced metadata filters
        {"key": "priority", "operator": "gte", "value": 5}
    ],
    start_date="2025-11-01",              # Date range filter
    end_date="2025-11-30"
)
```

**Date Filtering:**

The `start_date` and `end_date` parameters filter entries by creation timestamp using ISO 8601 format:

```python
# Find entries from a specific day
search_context(thread_id="project-123", start_date="2025-11-29", end_date="2025-11-29")

# Find entries from a date range with metadata filters
search_context(
    thread_id="project-123",
    metadata={"status": "active"},
    start_date="2025-11-01",
    end_date="2025-11-30"
)
```

## Integration Examples

### Task Queue System

```python
# Producer agent stores tasks
async def create_task(task_data):
    await store_context(
        thread_id="task-queue",
        source="agent",
        text=task_data["description"],
        metadata={
            "status": "pending",
            "priority": task_data["priority"],
            "task_name": task_data["name"],
            "created_at": datetime.now().isoformat(),
            "completed": False
        }
    )

# Consumer agent fetches high-priority pending tasks
async def get_next_task():
    result = await search_context(
        thread_id="task-queue",
        metadata_filters=[
            {"key": "status", "operator": "eq", "value": "pending"},
            {"key": "priority", "operator": "gte", "value": 5}
        ],
        limit=1
    )
    return result["entries"][0] if result["entries"] else None

# Mark task as completed (preserves existing metadata like priority, task_name)
async def complete_task(context_id):
    await update_context(
        context_id=context_id,
        metadata_patch={
            "status": "completed",
            "completed": True,
            "completed_at": datetime.now().isoformat()
        }
    )
```

### Multi-Agent Workflow

```python
# Planner agent creates tasks
await store_context(
    thread_id="data-pipeline",
    source="agent",
    text="Extract data from API",
    metadata={
        "agent_name": "planner",
        "task_name": "extract-api-data",
        "status": "assigned",
        "assigned_to": "extractor-agent",
        "priority": 7
    }
)

# Extractor agent finds assigned tasks
my_tasks = await search_context(
    thread_id="data-pipeline",
    metadata={
        "assigned_to": "extractor-agent",
        "status": "assigned"
    }
)

# Monitor agent finds stalled tasks
stalled = await search_context(
    thread_id="data-pipeline",
    metadata_filters=[
        {"key": "status", "operator": "in", "value": ["assigned", "in_progress"]},
        {"key": "assigned_to", "operator": "exists"}
    ]
)
```

## Troubleshooting

### Query Returns No Results

**Problem:** Filter returns empty results when data exists.

**Solutions:**

1. **Check case sensitivity:**
   ```python
   # Try case-insensitive (default)
   {"key": "status", "operator": "eq", "value": "active"}

   # If that fails, try explicit case-insensitive
   {"key": "status", "operator": "eq", "value": "active", "case_sensitive": False}
   ```

2. **Verify metadata field names:**
   ```python
   # Use exact field names
   result = search_context(thread_id="test", include_images=False)
   print(result["entries"][0]["metadata"])  # Check actual field names
   ```

3. **Check for null vs missing fields:**
   ```python
   # Field doesn't exist
   {"key": "field", "operator": "not_exists"}

   # Field exists but is null
   {"key": "field", "operator": "is_null"}
   ```

### Slow Query Performance

**Problem:** Queries take longer than expected.

**Solutions:**

1. **Add thread_id filter:**
   ```python
   # Slow - searches all threads
   search_context(metadata={"status": "active"})

   # Fast - scoped to specific thread
   search_context(thread_id="specific-thread", metadata={"status": "active"})
   ```

2. **Use indexed fields:**
   ```python
   # Slow - custom field
   {"key": "my_status", "operator": "eq", "value": "active"}

   # Fast - indexed field
   {"key": "status", "operator": "eq", "value": "active"}
   ```

3. **Profile with explain_query:**
   ```python
   result = search_context(..., explain_query=True)
   print(result["stats"]["execution_time_ms"])
   print(result["stats"]["query_plan"])
   ```

### Validation Errors

**Problem:** Receiving metadata filter validation errors.

**Solutions:**

1. **Check operator spelling:**
   ```python
   # Invalid
   {"key": "priority", "operator": "greater_than", "value": 5}

   # Valid
   {"key": "priority", "operator": "gt", "value": 5}
   ```

2. **Verify value types:**
   ```python
   # Invalid - string for IN operator
   {"key": "status", "operator": "in", "value": "active"}

   # Valid - list for IN operator
   {"key": "status", "operator": "in", "value": ["active"]}
   ```

3. **Check key format:**
   ```python
   # Invalid - special characters
   {"key": "status@field", "operator": "eq", "value": "active"}

   # Valid - alphanumeric, dots, underscores, hyphens only
   {"key": "status_field", "operator": "eq", "value": "active"}
   ```

## Additional Resources

### Related Documentation

- [README.md](../README.md) - Complete API reference for all MCP tools
- [Semantic Search Guide](semantic-search.md) - Optional vector similarity search
- [Full-Text Search Guide](full-text-search.md) - Optional linguistic search with stemming
- [Hybrid Search Guide](hybrid-search.md) - Combined FTS + semantic search with RRF fusion
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Architecture and development guidelines

### Implementation Files

- [`app/metadata_types.py`](../app/metadata_types.py) - MetadataFilter and MetadataOperator definitions
- [`app/query_builder.py`](../app/query_builder.py) - SQL query construction with security validation
- [`app/repositories/context_repository.py`](../app/repositories/context_repository.py) - Database operations for metadata filtering

### Test Examples

- [`tests/test_metadata_filtering.py`](../tests/test_metadata_filtering.py) - Comprehensive operator tests and integration examples
- [`tests/test_metadata_error_handling.py`](../tests/test_metadata_error_handling.py) - Error handling and validation tests

### Performance Tuning

For advanced performance optimization:
- Review indexed fields in [`app/schemas/sqlite_schema.sql`](../app/schemas/sqlite_schema.sql)
- Check query execution plans with `explain_query=True`
- Monitor execution time in query statistics
- Optimize metadata schema design for your use case
