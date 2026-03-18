---
name: context-retrieval-protocol
description: |
  Context retrieval patterns for accessing session context via an MCP-compatible context server.
  Provides patterns for browse and retrieve, hybrid search, semantic search, and full-text search.
  Use when you need to retrieve previous context or search for relevant information.
---

<overview>

# Context Retrieval Best Practices

Retrieving session context before examining your task is recommended for best results. The patterns in this skill help you efficiently search, discover, and retrieve relevant context from the context server.

</overview>

<orchestrator_verification>

## Multi-Agent Workflow: Verify Orchestrator Task Against Context

If you are working within a multi-agent orchestrated workflow (where a coordinator assigns tasks to specialized agents), consider verifying the task against the context server.

**Recommended Verification Steps:**

1. **Retrieve USER messages** - These have highest priority (source="user")
2. **Retrieve AGENT reports** - For context on previous work (source="agent")
3. **Compare orchestrator task** against retrieved context
4. **Identify discrepancies** - User messages override orchestrator instructions

**Why This Is Recommended:**

- Orchestrators can misinterpret, summarize incorrectly, or omit critical details
- User messages are the primary source of truth
- Agent reports provide implementation context and decisions

**If Discrepancies Are Found:**

- User messages take priority over orchestrator instructions
- Flag the discrepancy in your work report
- Execute based on verified user requirements

</orchestrator_verification>

<scoped_retrieval>

## Advanced: Orchestrated Workflows -- Scoped Retrieval

In orchestrated multi-agent workflows, the coordinator may provide a `context_scope` section in your task prompt, specifying exactly which context entries to retrieve. This is an optimization for workflows where the orchestrator has already identified the relevant context.

Check your task prompt for a `context_scope` XML section before executing the standard retrieval sequence below.

### When context_scope is Present

If your task prompt contains a context_scope XML tag (opening tag: `<` + `context_scope` + `>`), this overrides the standard retrieval sequence:

1. **SKIP Steps 1-2** - Do NOT retrieve all user messages or all agent reports
2. **Extract context_ids** from the context_scope section
3. **Retrieve ONLY those specific IDs** using `get_context_by_ids`
4. **Proceed with your task** - Do NOT execute additional broad searches

### Why Scoped Retrieval Exists

Some workflows (like consensus) have already identified the exact context needed:

- Orchestrator tracks specific report IDs throughout the workflow
- Broad retrieval would pollute the context window with irrelevant information
- Agents should trust the scope specified by the orchestrator

### Compliance Note

Using scoped retrieval when context_scope is present is expected behavior. The standard retrieval sequence applies only when no scope is defined.

### Example

If your prompt contains:

```text
<context_scope>
Retrieve ONLY these context_ids:
- Report 1: 4233
- Report 2: 4234
- Report 3: 4235
</context_scope>
```

Your retrieval should be:

```text
get_context_by_ids(context_ids=[4233, 4234, 4235])
```

Do NOT execute search_context for all users or all agents.

</scoped_retrieval>

<mandatory_sequence>

# Context Retrieval Sequence

## Quick Recall (Default Pattern)

For most use cases, two steps are sufficient:

### Step 1: Search for Relevant Context

```text
search_context(thread_id="session-id", source="user", limit=10)
search_context(thread_id="session-id", source="agent", limit=30)
```

Browse truncated previews to identify entries relevant to your task.

### Step 2: Retrieve Full Content

```text
get_context_by_ids(context_ids=[...relevant IDs from Step 1...])
```

Retrieve full content of relevant entries identified in Step 1.

## Comprehensive Multi-Agent Pattern

When working in a multi-agent workflow or when deeper context discovery is needed, extend the quick recall with additional steps:

### Step 3: Hybrid Search for Additional Context (Recommended)

```text
hybrid_search_context(query="relevant search terms", thread_id="session-id", limit=15)
```

Use hybrid search to find conceptually related content. Useful when:

- Metadata filtering alone may miss relevant entries
- You need conceptual matches beyond exact keyword matches
- You are uncertain if you have retrieved all relevant context

### Step 4: Navigate References (Optional)

```text
# Check if retrieved entries have references.context_ids
# If yes, consider retrieving them for deeper understanding
get_context_by_ids(context_ids=[...IDs from metadata.references.context_ids...])
```

When entries retrieved in earlier steps contain `references.context_ids` in their metadata, these represent related entries the original author worked with. Consider following these references when:

- The current context seems incomplete
- You need to understand the reasoning behind decisions
- Referenced entries appear relevant to your task

## Complete Example

```text
# Quick Recall
search_context(thread_id="session-id", source="user", limit=10)
search_context(thread_id="session-id", source="agent", limit=30)
get_context_by_ids(context_ids=[123, 124, 125])

# Extended (if needed)
hybrid_search_context(query="implementation patterns", thread_id="session-id", limit=15)
```

</mandatory_sequence>

<session_id>

# How to Obtain Session ID

The session ID is used as `thread_id` for context server queries. Obtain it using one of these approaches:

1. **Already available** -- If the session ID is provided via context or prompt, use it directly
2. **Session file** -- Check for a session ID file in the project directory (e.g., `.claude/.session_id` in the working directory)
3. **Project directory name** -- If no session ID is available through other means, use the current project directory basename as the thread identifier. This is the same value as described in the project name derivation chain (see below). Using the project name ensures all agents working on the same project write to the same thread, which is essential for multi-agent coordination.

</session_id>

<project_name>

# How to Obtain Canonical Project Name

The project name should be derived using the following fallback chain to ensure consistency across git worktrees:

**Fallback Chain (in priority order):**

1. **Parse from git remote URL** (PREFERRED)
   - Try `origin` remote first: `git remote get-url origin`
   - Parse repository name from URL:
     - `https://github.com/user/my-project.git` -> `my-project`
     - `git@github.com:user/my-project.git` -> `my-project`
   - If `origin` unavailable, try `upstream`, then first available remote

2. **Git toplevel basename** (FALLBACK for repos without remotes)
   - `git rev-parse --show-toplevel` -> extract last path component
   - Example: `/home/user/projects/my-project` -> `my-project`

3. **Current directory basename** (FALLBACK for non-git directories)
   - Extract last directory name from working directory path
   - Example: `/home/user/work/my-project` -> `my-project`

**Why this matters:**

- Different worktrees of the same repository have different directory names
- Using directory name causes context isolation to break across worktrees
- Remote URL provides true canonical identity across all worktrees and users

</project_name>

<worktree_queries>

## Worktree-Aware Context Queries

When working in git worktree environments, use appropriate query patterns based on your search scope.

### Same-Session Queries (Default)

Always use `thread_id` filter for current session context:

```text
search_context(thread_id="session-uuid", source="agent", limit=30)
```

This is the default pattern for all retrieval steps.

### Cross-Session, Same-Worktree Queries

When searching across sessions within the same worktree:

```text
search_context(
  metadata={"project": "canonical-name", "worktree_id": "current-worktree"},
  limit=10
)
```

Use this pattern to find historical work in the same worktree but different sessions.

### Cross-Session, Same-Project Queries

When searching across all worktrees of the same project:

```text
search_context(metadata={"project": "canonical-name"}, limit=20)
hybrid_search_context(query="...", metadata={"project": "canonical-name"})
```

Use this pattern to find work across all worktrees of the repository.

### WARNING: Cross-Worktree Context

When retrieving context from other worktrees, exercise caution:

- **File paths may not exist** - Different worktrees have different branches checked out
- **Implementation status may differ** - Code merged in one worktree may not exist in another
- **Always verify file existence** before referencing paths from cross-worktree context
- **Check branch compatibility** - Features implemented for one branch may conflict with another

**Safe Cross-Worktree Usage:**

1. Use cross-worktree context for **conceptual understanding** (patterns, decisions, rationale)
2. **Verify** that referenced files exist in current worktree before editing
3. **Do not assume** implementation status applies to current branch

</worktree_queries>

<tools>

# Available Context Server Tools

**Note:** Not all tools listed below may be available in your environment. Tool availability depends on server configuration and how the server is connected to your MCP client. Use the tools that are available to you. If a recommended tool is unavailable, use an alternative from this table.

For context storage and update tools, see the counterpart skill (`context-preservation-protocol`).

| Tool                      | Status           | Returns                           | Use For                                                |
|---------------------------|------------------|-----------------------------------|--------------------------------------------------------|
| `search_context`          | RECOMMENDED      | TRUNCATED text + summary          | Browse/discover entries before retrieval               |
| `get_context_by_ids`      | RECOMMENDED      | FULL text + images                | Retrieve specific entries after discovery              |
| `hybrid_search_context`   | RECOMMENDED      | TRUNCATED text + summary + scores | Best overall search (FTS + semantic)                   |
| `semantic_search_context` | Optional         | TRUNCATED text + summary + scores | Meaning-based search (see Score Fields Reference)      |
| `fts_search_context`      | Optional         | TRUNCATED text + summary + scores | Keyword/linguistic search (see Score Fields Reference) |
| `list_threads`            | Optional         | Thread list with statistics       | Discover available threads and their metadata          |
| `get_statistics`          | Optional         | Server statistics                 | Check server health and usage metrics                  |
| `delete_context`          | Use with caution | Confirmation                      | Remove specific context entries during cleanup         |

**Key notes:**

- **ALL search tools return TRUNCATED content** (text + summary). Use `get_context_by_ids` to retrieve full content of relevant entries identified through search. This applies to `search_context`, `hybrid_search_context`, `semantic_search_context`, and `fts_search_context` equally.
- Because results are truncated, you can search more aggressively: use higher limits (10-20+), perform multiple sequential searches with different queries, and iterate to find the best matches before retrieving full content.
- `search_context` is recommended for browsing user and agent entries
- `get_context_by_ids` is recommended for retrieving full content
- `hybrid_search_context` is recommended for conceptual discovery -- use when in doubt
- Specify `thread_id` to search within the current session

## Score Fields Reference

Each search tool returns a `scores` object with different fields:

| Tool                      | Scores Object Fields                                                                 |
|---------------------------|--------------------------------------------------------------------------------------|
| `fts_search_context`      | `fts_score`, `rerank_score`                                                          |
| `semantic_search_context` | `semantic_distance`, `rerank_score`                                                  |
| `hybrid_search_context`   | `rrf`, `fts_rank`, `semantic_rank`, `fts_score`, `semantic_distance`, `rerank_score` |

### Score Polarity

| Field               | Polarity        | Description                       |
|---------------------|-----------------|-----------------------------------|
| `fts_score`         | HIGHER = better | BM25/ts_rank relevance            |
| `fts_rank`          | LOWER = better  | FTS result rank (1 = best)        |
| `semantic_distance` | LOWER = better  | L2 Euclidean distance             |
| `semantic_rank`     | LOWER = better  | Semantic result rank (1 = best)   |
| `rrf`               | HIGHER = better | Combined RRF score                |
| `rerank_score`      | HIGHER = better | Cross-encoder relevance (0.0-1.0) |

</tools>

<metadata_reference>

## Metadata Filtering

For the complete list of metadata fields and allowed values, see `context-preservation-protocol` skill (both skills are always loaded together).

**Quick Reference for Filtering:**

| Filter By   | Use Parameter                            | Example                                                                                  |
|-------------|------------------------------------------|------------------------------------------------------------------------------------------|
| Agent       | `metadata: {"agent_name": "..."}`        | Find all implementation-guide reports                                                    |
| Status      | `metadata: {"status": "done"}`           | Find completed work                                                                      |
| Project     | `metadata: {"project": "..."}`           | Scope to current project                                                                 |
| Report type | `metadata: {"report_type": "research"}`  | Find all research reports                                                                |
| Technology  | Use `array_contains` or tags             | `metadata_filters: [{key: "technologies", operator: "array_contains", value: "python"}]` |
| References  | `metadata_filters` with `array_contains` | `[{key: "references.context_ids", operator: "array_contains", value: 2322}]`             |

**Note:** For technology filtering, you have two options:
- Use `array_contains` operator for exact element match: `metadata_filters: [{key: "technologies", operator: "array_contains", value: "python"}]`
- Use `tags` parameter for OR logic: `tags: ["python", "fastapi"]`

</metadata_reference>

<revision_context_detection>

## Advanced: Revision Context Detection

This section is relevant for multi-agent workflows where agents update each other's prior work. If you are working in a simple single-agent setup, you can skip this section.

### Detecting Revision Mode

When your task prompt contains revision indicators, extract the previous context_id and use `update_context` instead of `store_context`.

**Revision Indicators in Prompt:**

| Pattern                    | Meaning                                 |
|----------------------------|-----------------------------------------|
| `PREVIOUS CONTEXT ID: [N]` | Explicit signal to UPDATE entry N       |
| `PLAN REVISION REQUEST`    | Revision mode - look for context_id     |
| `RESEARCH CONTINUATION`    | Continuation mode - look for context_id |

### Extraction Protocol

1. **Detect revision mode** by scanning prompt for indicators above
2. **Extract the context_id:**
   ```text
   # Look for: PREVIOUS CONTEXT ID: 123
   # Extract: 123
   ```
3. **Retrieve the previous entry:**
   ```text
   get_context_by_ids(context_ids=[extracted_id])
   ```
4. **Store the context_id** for use with `update_context` when saving

### Finding Your Own Prior Entries (for Update)

When you need to update your own prior work but context_id is not provided:

```text
search_context(
  thread_id="session-id",
  source="agent",
  metadata={"agent_name": "[your-agent-name]", "report_type": "research"},
  limit=15
)
```

Then use `update_context(context_id=...)` with the most recent matching entry.

Only update entries where `agent_name` matches your agent identifier. Never update another agent's entries.

</revision_context_detection>

<references_navigation>

## References-Based Navigation

### Understanding `references.context_ids`

When you retrieve context entries, check for `metadata.references.context_ids`. These IDs are NOT random - they represent entries the original agent actually WORKED WITH:

- Implementation guides reference research plans they are based on
- Validation reports reference implementation reports they validated
- Research reports reference prior work they built upon

**These connections form a knowledge graph that you can navigate.**

### When to Follow References

**CONSIDER following `references.context_ids` when:**

- You need deeper understanding of WHY decisions were made
- The current entry references a plan or research you have not yet retrieved
- You want to trace the full history of a task (research -> implementation -> validation)
- The truncated preview suggests related context would be valuable

**You do NOT need to follow references when:**

- You already have sufficient context for your task

### How to Navigate References

1. **Identify references** in retrieved entry's metadata:

   ```json
   "metadata": {
     "references": {
       "context_ids": [3348, 3349, 3352]
     }
   }
   ```

2. **Retrieve related entries** using `get_context_by_ids`:

   ```text
   get_context_by_ids(context_ids=[3348, 3349, 3352])
   ```

3. **Evaluate relevance** - not all referenced entries may be needed for current task

### Navigation Depth Guidance

| Scenario                   | Recommended Depth                       |
|----------------------------|-----------------------------------------|
| Understanding current task | 1 level (direct references)             |
| Tracing decision history   | 2 levels (references of references)     |
| Comprehensive research     | Follow until pattern emerges            |

</references_navigation>

<strategy>

# Retrieval Strategy

- Retrieve relevant user and agent context to understand the current task
- You can query the context server as many times as needed
- You can return to the context server at any point during your work
- Search iteratively and liberally -- all search results are truncated, so you can safely perform multiple searches with higher limits (10-20+) without overwhelming your context window. Use `get_context_by_ids` to retrieve full content only for entries that appear relevant.
- Include `include_images: true` to capture visual context (diagrams, matrices, charts)

</strategy>

<patterns>

# Retrieval Patterns

**When to use each pattern:**

- **Browse and Retrieve**: Default pattern for most use cases
- **Hybrid Search**: Recommended when you need conceptual search
- **Semantic Search**: Optional - alternative to hybrid when only semantic matching is needed
- **Full-Text Search**: Optional - precise keyword matching, boolean queries, exact phrases

## Pattern 1 - Browse and Retrieve (Default)

Use for the default retrieval workflow (finding context by source and metadata):

1. Use `search_context` with `thread_id` and `source="user"` (Step 1)
2. Use `search_context` with `thread_id` and `source="agent"` (Step 2)
3. Browse truncated previews to identify ALL relevant entries
4. Use `get_context_by_ids` to retrieve full content of selected entries (Step 3)

## Pattern 2 - Hybrid Search (Recommended)

Use for iterative discovery (combined FTS + semantic search):

1. Use `hybrid_search_context` with a natural language query describing what you need
2. Specify `thread_id` to search within current session context
3. Documents found by BOTH FTS and semantic methods rank highest
4. Best for finding prior solutions, knowledge, principles, and conceptually related content
5. Search iteratively: start broad with higher limits (10-20+), assess truncated previews + summaries, refine queries, then retrieve full content of best matches via `get_context_by_ids`
6. Results are truncated -- assess relevance from truncated text + summary + metadata, then use `get_context_by_ids` for full content of relevant entries

**Use hybrid search aggressively.** Because results are lightweight (truncated), you can safely perform multiple rounds of searching with different queries and higher limits without overwhelming the context window.

IF IN DOUBT - USE IT!

## Pattern 3 - Semantic Search (Optional)

Use for meaning-based discovery when hybrid search is not needed:

1. Use `semantic_search_context` with a query describing what you need
2. Specify `thread_id` to search within current session context
3. Results are truncated -- assess relevance from truncated text + summary + metadata, then use `get_context_by_ids` for full content of relevant entries

## Pattern 4 - Full-Text Search (Optional)

Use for precise keyword matching:

1. Use `fts_search_context` for keyword-based search
2. Specify `thread_id` to search within current session context
3. Use `boolean` mode for complex queries: `"python AND async NOT deprecated"`
4. Use `phrase` mode for exact matches: `"error handling"`
5. Enable `highlight: true` to see matching snippets
6. Results are truncated -- assess relevance from truncated text + summary + metadata, then use `get_context_by_ids` for full content of relevant entries

## Pattern 5 - References Navigation (Optional)

Use for following knowledge graph links when deeper context is needed:

1. Retrieve entries using Steps 1-2 (or extended Steps 1-4)
2. Use `get_context_by_ids` to retrieve selected referenced entries
3. Repeat if those entries have further relevant references (depth limit: 2-3 levels)

**Example workflow:**

```text
# Step 1-3: Retrieved entry 3357 (validation report)
# Entry 3357 has metadata.references.context_ids: [3349, 3352]

# These are: 3349 (potentially implementation plan), 3352 (potentially implementation report)
# Retrieve them for complete picture

get_context_by_ids(context_ids=[3349, 3352])
```

**When to use Pattern 5:**

- Research plans reference prior research you need to understand
- Validation reports reference implementations you need to verify
- You see a chain of work and need the full picture

</patterns>

<examples>

# Behavioral Examples

<example scenario="complete_mandatory_sequence">
**Input:** Agent starts task, receives instructions from orchestrator
**Correct Approach:** (1) Obtain session ID; (2) Step 1: Call `search_context(thread_id="session-id", source="user", limit=10)` to retrieve user messages; (3) Call `search_context(thread_id="session-id", source="agent", limit=30)` to retrieve agent reports; (4) Step 2: Call `get_context_by_ids(context_ids=[...])` to retrieve full content; (5) If in a multi-agent workflow, verify orchestrator task against retrieved user messages and agent reports; (6) Step 3: Call `hybrid_search_context` if additional context needed
**Result:** Agent has full context of user requirements, verified orchestrator task, and implementation plans
</example>

<example scenario="orchestrator_verification">
**Input:** Orchestrator provides task "Implement feature X with approach A"
**Correct Approach:** (1) Execute Steps 1-2 to retrieve user messages and agent reports; (2) Compare orchestrator task against user messages; (3) Discover user message says "Use approach B, not A"; (4) Flag discrepancy; (5) Execute based on USER requirement (approach B)
**Result:** Agent correctly identifies orchestrator error and follows user's actual requirements
</example>

<example scenario="hybrid_search_usage">
**Input:** Agent completed Steps 1-3 but uncertain if all relevant context was retrieved
**Correct Approach:** (1) Execute Step 4: `hybrid_search_context(query="authentication implementation patterns", thread_id="session-id", limit=15)`; (2) Review truncated previews + summaries to identify relevant entries; (3) Call `get_context_by_ids(context_ids=[...relevant IDs...])` to retrieve full content of promising matches; (4) If needed, search again with refined queries or different terms
**Result:** Agent finds additional conceptually related context that metadata filtering missed, using iterative truncation-aware search
</example>

<example scenario="truncation_aware_retrieval">
**Input:** Agent needs to find prior implementation decisions about database schema design
**Correct Approach:** (1) Search broadly: `hybrid_search_context(query="database schema design decisions", thread_id="session-id", limit=15)`; (2) Review truncated text + summary + metadata of each result to assess relevance; (3) Identify 3-4 entries that appear most relevant based on previews; (4) Retrieve full content: `get_context_by_ids(context_ids=[id1, id2, id3, id4])`; (5) If insufficient, search again with refined query: `hybrid_search_context(query="SQL migration table structure", thread_id="session-id", limit=15)`
**Result:** Agent efficiently discovers relevant context through iterative search without overwhelming the context window, then retrieves full content only for the most promising entries
</example>

<example scenario="protocol_violation">
**Input:** Agent receives orchestrator task and skips context retrieval, trusting orchestrator's summary
**Incorrect Approach:** Agent proceeds directly with task based only on orchestrator-provided information
**Result:** Not recommended - Agent missed critical user requirements and produced incorrect work
**Correct Action:** Agent should execute retrieval Steps 1-2 before examining any task
</example>

<example scenario="references_navigation">
**Input:** Agent retrieves implementation report with metadata showing `references.context_ids: [3322, 3323]`
**Correct Approach:** (1) Note that entry references two prior entries; (2) Call `get_context_by_ids(context_ids=[3322, 3323])` to retrieve BOTH entries;
**Result:** Agent has complete context chain: (3322) -> (3323) -> current context entry, enabling full traceability and verification of decisions
</example>

</examples>

<compliance_checklist>

# Compliance Checklist

Before proceeding with your task, consider verifying the following:

- [ ] **Step 1 completed**: Called `search_context(source="user")` to retrieve user messages
- [ ] **Step 2 completed**: Called `search_context(source="agent")` to retrieve agent reports
- [ ] **Full content retrieved**: Called `get_context_by_ids` for full content of relevant entries
- [ ] **Hybrid search considered**: Evaluated whether `hybrid_search_context` is needed for additional context
- [ ] **References considered**: Checked `metadata.references.context_ids` in retrieved entries; followed references when deeper context needed

Completing this checklist is a best practice for reliable results.

</compliance_checklist>

<error_handling>

# Error Handling

**If a context retrieval step fails:**

1. **Retry once** after a brief pause
2. **Document the failure** in your work report
3. **Continue with remaining steps** of the mandatory sequence
4. **Note limitations** in your analysis due to incomplete context

Even if one step fails, attempt the remaining steps. A single failure does not excuse skipping other steps.

**If ALL context retrieval fails:**

**WARNING: Results will be significantly degraded without context server access.**

1. **Log the failure** with the specific error message
2. **Proceed with available information** if any context was obtained through other means
3. **If no context is available at all**, inform the caller that results may be incomplete:
   ```text
   WARNING: Context server unavailable. Proceeding with limited context.
   Error: [specific error message]
   Impact: Unable to retrieve session history. Results may be incomplete or miss prior decisions.
   ```
4. **Note limitations** in your work report so downstream consumers know context was unavailable

**Rationale:** Context server provides session continuity and coordination. Without it, results may be degraded but work can still proceed with reduced confidence.

</error_handling>
