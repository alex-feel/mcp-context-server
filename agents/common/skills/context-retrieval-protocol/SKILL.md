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

If you are working within a multi-agent orchestrated workflow (where a coordinator assigns tasks to specialized agents), you MUST verify the task against the context server before executing it: retrieve USER messages (source="user", highest priority), retrieve AGENT reports (source="agent") for context on previous work, then compare the orchestrator task against the retrieved context. This verification is mandatory because orchestrators can misinterpret, summarize incorrectly, or omit critical details; user messages are the primary source of truth, and agent reports provide implementation context and decisions.

If discrepancies are found: user messages take priority over orchestrator instructions; flag the discrepancy in your work report; execute based on verified user requirements.

**Conflict resolution rule:** user-message wording is authoritative; any orchestrator-introduced scope criterion, exclusion, exception, qualification, or pre-approval that is NOT traceable to a verbatim user message in the current session MUST be discarded, and the agent MUST execute on the user-message wording only.

</orchestrator_verification>

<scoped_retrieval>

## Advanced: Orchestrated Workflows -- Scoped Retrieval

In orchestrated multi-agent workflows, the coordinator may provide a `context_scope` section in your task prompt, specifying exactly which context entries to retrieve. This is an optimization for workflows (like consensus) where the orchestrator has already tracked the exact relevant context and broad retrieval would pollute the context window with irrelevant information; trust the scope specified by the orchestrator.

Check your task prompt for a context_scope XML tag (opening tag: `<` + `context_scope` + `>`) before executing the standard retrieval sequence below. When present, it overrides that sequence:

1. **SKIP Steps 1-2** - Do NOT retrieve all user messages or all agent reports
2. **Extract context_ids** from the context_scope section
3. **Retrieve ONLY those specific IDs** using `get_context_by_ids`
4. **Proceed with your task** - Do NOT execute additional broad searches

Using scoped retrieval when context_scope is present is expected behavior; the standard retrieval sequence applies only when no scope is defined.

### Sequential Retrieval Fallback

When retrieving multiple context entries via `get_context_by_ids`, the response may be truncated or incomplete if the combined content exceeds MCP response token limits. If this occurs, or if the caller or user provides instructions to retrieve entries sequentially, follow those instructions: retrieve entries ONE AT A TIME using separate `get_context_by_ids` calls, processing each entry before requesting the next.

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

retrieve `get_context_by_ids(context_ids=[4233, 4234, 4235])` and do NOT execute search_context for all users or all agents.

</scoped_retrieval>

<mandatory_sequence>

# Context Retrieval Sequence

## Quick Recall (Default Pattern)

For most use cases, two steps are sufficient.

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

## Comprehensive Multi-Agent Pattern

When working in a multi-agent workflow or when deeper context discovery is needed, extend the quick recall:

### Step 3: Hybrid Search for Additional Context (Recommended)

```text
hybrid_search_context(query="relevant search terms", thread_id="session-id", limit=15)
```

Use hybrid search to find conceptually related content when metadata filtering alone may miss relevant entries, when you need conceptual matches beyond exact keyword matches, or when you are uncertain you have retrieved all relevant context.

### Step 4: Navigate References (Optional)

When entries retrieved in earlier steps contain `references.context_ids` in their metadata, these represent related entries the original author worked with. Retrieve them via `get_context_by_ids(context_ids=[...IDs from metadata.references.context_ids...])` when the current context seems incomplete, when you need to understand the reasoning behind decisions, or when referenced entries appear relevant to your task.

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

<thread_id>

# How to Obtain Thread ID

The thread ID is used as `thread_id` for context server queries. Obtain it using the following search chain:

1. **Already available** -- If the thread ID is provided via context or prompt, use it directly
2. **Thread ID file** -- Check `.context_server/.thread_id` in the project working directory
3. **Project directory name** -- If no thread ID file exists, derive the thread identifier using the canonical project name fallback chain described below (git remote URL preferred, then git toplevel basename, then current directory basename). Using the project name ensures all agents working on the same project write to the same thread, which is essential for multi-agent coordination

</thread_id>

<project_name>

# How to Obtain Canonical Project Name

Derive the project name using the following fallback chain (in priority order) to ensure consistency across git worktrees:

1. **Parse from git remote URL** (PREFERRED) -- Try `origin` first: `git remote get-url origin`; parse the repository name from the URL (`https://github.com/user/my-project.git` -> `my-project`; `git@github.com:user/my-project.git` -> `my-project`). If `origin` is unavailable, try `upstream`, then the first available remote
2. **Git toplevel basename** (FALLBACK for repos without remotes) -- `git rev-parse --show-toplevel`, extract the last path component (`/home/user/projects/my-project` -> `my-project`)
3. **Current directory basename** (FALLBACK for non-git directories) -- Extract the last directory name from the working directory path (`/home/user/work/my-project` -> `my-project`)

Why this matters: different worktrees of the same repository have different directory names, so directory-derived names break context isolation across worktrees; the remote URL provides true canonical identity across all worktrees and users.

</project_name>

<worktree_queries>

## Worktree-Aware Context Queries

When working in git worktree environments, use the query pattern matching your search scope.

### Same-Session Queries (Default)

Always use the `thread_id` filter for current session context; this is the default pattern for all retrieval steps:

```text
search_context(thread_id="session-uuid", source="agent", limit=30)
```

### Cross-Session, Same-Worktree Queries

Find historical work in the same worktree but different sessions:

```text
search_context(
  metadata={"project": "canonical-name", "worktree_id": "current-worktree"},
  limit=10
)
```

### Cross-Session, Same-Project Queries

Find work across all worktrees of the repository:

```text
search_context(metadata={"project": "canonical-name"}, limit=20)
hybrid_search_context(query="...", metadata={"project": "canonical-name"})
```

### WARNING: Cross-Worktree Context

Exercise caution with context from other worktrees: different worktrees have different branches checked out, so referenced file paths may not exist, implementation status may differ (code merged in one worktree may not exist in another), and features implemented for one branch may conflict with another. Safe usage:

1. Use cross-worktree context for **conceptual understanding** (patterns, decisions, rationale)
2. **Always verify** that referenced files exist in the current worktree before referencing or editing them
3. **Do not assume** implementation status applies to the current branch

</worktree_queries>

<environment_integration>

## Environment Integration Patterns

Context retrieval operations can interact with environment-level hooks, validation gates, and orchestration workflows:

- **Hook-aware retrieval:** Environment hooks may verify that agents retrieve context before starting work (enforcing retrieval discipline), compare retrieved context against task instructions to detect discrepancies, or log which context entries were retrieved by which agent for traceability. In such environments, follow the documented sequence (Quick Recall or Comprehensive Multi-Agent Pattern) to avoid triggering validation failures.
- **Metadata patterns for multi-agent coordination:** Filter by `agent_name` to find prior work from a specific agent role, by `status: "done"` or `status: "pending"` to find completed work or work requiring continuation, and by `report_type` to find all validation, research, or implementation reports regardless of which agent produced them; follow `references.context_ids` to reconstruct the full workflow chain (research -> plan -> implementation -> validation).
- **Orchestrated workflows:** Context retrieval serves as the shared memory layer -- agents retrieve prior agent reports to understand completed work before continuing (task handoff), retrieve implementation plans and compare against current task instructions (plan verification), and retrieve user messages to verify orchestrator instructions match user intent (conflict detection; see Orchestrator Verification section).

These patterns are generic and apply to any environment with multi-agent coordination capabilities.

</environment_integration>

<tools>

# Available Context Server Tools

**Note:** Not all tools listed below may be available in your environment; availability depends on server configuration and how the server is connected to your MCP client. Use the tools that are available to you; if a recommended tool is unavailable, use an alternative from this table. The tools below cover retrieval -- for storage and update operations, the context server exposes a parallel set of tools (for example `store_context` and `update_context`); consult the storage section of the server's own documentation.

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

- **ALL search tools return TRUNCATED content** (text + summary): this applies to `search_context`, `hybrid_search_context`, `semantic_search_context`, and `fts_search_context` equally. Use `get_context_by_ids` to retrieve full content of relevant entries identified through search.
- Because results are truncated, you can search more aggressively: use higher limits (10-20+), perform multiple sequential searches with different queries, and iterate to find the best matches before retrieving full content. Use `hybrid_search_context` for conceptual discovery -- use it when in doubt.
- Specify `thread_id` to search within the current session.

## Truncation Discipline (ABSOLUTE)

Truncated text, summaries, and metadata returned by ALL search tools (`search_context`, `hybrid_search_context`, `semantic_search_context`, `fts_search_context`) are FOR RELEVANCE ASSESSMENT ONLY; substance MUST come from `get_context_by_ids` full retrieval. This is an ABSOLUTE rule with NO exception: search-result summaries are AI-generated approximations that MAY omit critical conditions, caveats, or nuance present in the full text, and drawing conclusions about what an entry says, recommends, or decides from search results alone -- before retrieving the full text via `get_context_by_ids` -- is exactly the failure mode this rule prevents. It applies to every search tool, every result field (text, summary, metadata, tags), and every downstream consumer; reasoning about an entry's substance from any source other than `get_context_by_ids` full retrieval is a PROTOCOL VIOLATION.

## Score Fields Reference

Each search tool returns a `scores` object with different fields:

| Tool                      | Scores Object Fields                                                                 |
|---------------------------|--------------------------------------------------------------------------------------|
| `fts_search_context`      | `fts_score`, `rerank_score`                                                          |
| `semantic_search_context` | `semantic_distance`, `rerank_score`                                                  |
| `hybrid_search_context`   | `rrf`, `fts_rank`, `semantic_rank`, `fts_score`, `semantic_distance`, `rerank_score` |

### Score Polarity

| Field               | Polarity        | Description                                                             |
|---------------------|-----------------|-------------------------------------------------------------------------|
| `fts_score`         | HIGHER = better | BM25/ts_rank relevance                                                  |
| `fts_rank`          | LOWER = better  | FTS result rank (1 = best)                                              |
| `semantic_distance` | LOWER = better  | Similarity-ordered: L2 (fp32/mse) or negated inner product (ip variant) |
| `semantic_rank`     | LOWER = better  | Semantic result rank (1 = best)                                         |
| `rrf`               | HIGHER = better | Combined RRF score                                                      |
| `rerank_score`      | HIGHER = better | Cross-encoder relevance (0.0-1.0)                                       |

</tools>

<metadata_reference>

## Metadata Filtering

When filtering search results, use the metadata fields documented by the context server itself. Common fields include `agent_name`, `task_name`, `status`, `project`, `report_type`, `technologies`, and `references` (an object that may contain a `context_ids` array). Supported filter operators include direct equality (via the `metadata` parameter) and the `metadata_filters` advanced operators such as `eq`, `ne`, `gt`, `lt`, `contains`, `array_contains`, `starts_with`, `exists`, and similar comparators.

**Quick Reference for Filtering:**

| Filter By   | Use Parameter                            | Example                                                                                  |
|-------------|------------------------------------------|------------------------------------------------------------------------------------------|
| Agent       | `metadata: {"agent_name": "..."}`        | Find all implementation-guide reports                                                    |
| Status      | `metadata: {"status": "done"}`           | Find completed work                                                                      |
| Project     | `metadata: {"project": "..."}`           | Scope to current project                                                                 |
| Report type | `metadata: {"report_type": "research"}`  | Find all research reports                                                                |
| Technology  | Use `array_contains` or tags             | `metadata_filters: [{key: "technologies", operator: "array_contains", value: "python"}]` |
| References  | `metadata_filters` with `array_contains` | `[{key: "references.context_ids", operator: "array_contains", value: 2322}]`             |

**Note:** For technology filtering, use the `array_contains` operator for exact element match (see table) or the `tags` parameter for OR logic: `tags: ["python", "fastapi"]`.

</metadata_reference>

<revision_context_detection>

## Advanced: Revision Context Detection

This section is relevant for multi-agent workflows where agents update each other's prior work; in a simple single-agent setup, you can skip it.

When your task prompt contains revision indicators, extract the previous context_id and use `update_context` instead of `store_context`.

**Revision Indicators in Prompt:**

| Pattern                    | Meaning                                 |
|----------------------------|-----------------------------------------|
| `PREVIOUS CONTEXT ID: [N]` | Explicit signal to UPDATE entry N       |
| `PLAN REVISION REQUEST`    | Revision mode - look for context_id     |
| `RESEARCH CONTINUATION`    | Continuation mode - look for context_id |

**Extraction protocol:** (1) detect revision mode by scanning the prompt for the indicators above; (2) extract the context_id (e.g., `PREVIOUS CONTEXT ID: 123` -> `123`); (3) retrieve the previous entry with `get_context_by_ids(context_ids=[extracted_id])`; (4) store the context_id for use with `update_context` when saving.

### Finding Your Own Prior Entries (for Update)

When you need to update your own prior work but the context_id is not provided:

```text
search_context(
  thread_id="session-id",
  source="agent",
  metadata={"agent_name": "[your-agent-name]", "report_type": "research"},
  limit=15
)
```

Then use `update_context(context_id=...)` with the most recent matching entry. Only update entries where `agent_name` matches your agent identifier. Never update another agent's entries.

</revision_context_detection>

<references_navigation>

## References-Based Navigation

When you retrieve context entries, check for `metadata.references.context_ids`. These IDs are NOT random - they represent entries the original agent actually WORKED WITH: implementation guides reference the research plans they are based on, validation reports reference the implementation reports they validated, and research reports reference prior work they built upon. These connections form a knowledge graph that you can navigate.

**CONSIDER following references when:** you need deeper understanding of WHY decisions were made; the current entry references a plan or research you have not yet retrieved; you want to trace the full history of a task (research -> implementation -> validation); or the truncated preview suggests related context would be valuable. You do NOT need to follow references when you already have sufficient context for your task.

**How to navigate:** identify references in the retrieved entry's metadata, retrieve the related entries using `get_context_by_ids`, and evaluate relevance -- not all referenced entries may be needed for the current task:

```json
"metadata": {
  "references": {
    "context_ids": [3348, 3349, 3352]
  }
}
```

```text
get_context_by_ids(context_ids=[3348, 3349, 3352])
```

### Navigation Depth Guidance

| Scenario                   | Recommended Depth                       |
|----------------------------|-----------------------------------------|
| Understanding current task | 1 level (direct references)             |
| Tracing decision history   | 2 levels (references of references)     |
| Comprehensive research     | Follow until pattern emerges            |

</references_navigation>

<context_continuity>

## Context Continuity Patterns

These patterns help agents maintain coherence across context window boundaries and long-running tasks.

### Basic Continuity (Default)

Apply these patterns by default in all sessions:

- **Status tracking:** Always set `status: "done"` or `status: "pending"` in metadata to indicate work completion state
- **Session handoff notes:** Before stopping, store a brief summary of work performed, decisions made, and next steps, so the next session (or agent) can resume without re-discovering context
- **Task completion markers:** Use `references.context_ids` to link new work to the prior entries it builds upon, creating a navigable chain of work history
- **Re-retrieval after context loss:** After any context compaction or window reset, re-read key context entries (plans, requirements, prior decisions) from the server to restore working memory. Do not rely on compacted summaries or search-truncated previews for critical details -- always retrieve full content via `get_context_by_ids`

### Advanced: Long-Running Task Continuity (Optional)

For tasks spanning multiple context windows or requiring extended multi-step execution, consider these additional patterns:

- **Progressive summarization:** Periodically condense accumulated context into structured summaries stored on the context server, preserving critical information (architectural decisions, unresolved issues, implementation progress) while reducing context window pressure. Store summaries as new context entries with `references.context_ids` pointing to the original detailed entries
- **Checkpoints:** At defined milestones during multi-step tasks, store intermediate state as a context entry -- current progress (what is completed, what remains), key decisions and their rationale, active blockers or dependencies, and files modified and their purpose. This enables recovery if a session is interrupted and provides a clear starting point for the next session
- **Context window monitoring:** If approaching context limits, proactively store current progress before compaction occurs; after compaction, immediately retrieve critical context entries (plans, requirements) from the server. Treat the context server as persistent memory that survives compaction -- store anything that must not be lost
- **Multi-agent handoff:** For clean agent-to-agent transitions in orchestrated workflows, the completing agent stores a comprehensive handoff report with clear next steps; the receiving agent retrieves the handoff report and referenced context before starting; both agents use consistent metadata (`references.context_ids`) to maintain the work chain; and disagreements between orchestrator instructions and stored context are resolved in favor of stored user messages (see Orchestrator Verification)

</context_continuity>

<strategy>

# Retrieval Strategy

- Retrieve relevant user and agent context to understand the current task
- Query the context server as many times as needed; you can return to it at any point during your work
- Search iteratively and liberally -- all search results are truncated, so you can safely perform multiple searches with higher limits (10-20+) without overwhelming your context window. Use `get_context_by_ids` to retrieve full content only for entries that appear relevant
- Include `include_images: true` to capture visual context (diagrams, matrices, charts)

</strategy>

<patterns>

# Retrieval Patterns

For every search pattern below: specify `thread_id` to search within the current session, and remember that results are truncated -- assess relevance from truncated text + summary + metadata, then use `get_context_by_ids` for full content of relevant entries.

## Pattern 1 - Browse and Retrieve (Default)

Default retrieval workflow (finding context by source and metadata):

1. Use `search_context` with `thread_id` and `source="user"`
2. Use `search_context` with `thread_id` and `source="agent"`
3. Browse truncated previews to identify ALL relevant entries
4. Use `get_context_by_ids` to retrieve full content of selected entries

## Pattern 2 - Hybrid Search (Recommended)

Iterative conceptual discovery (combined FTS + semantic search):

1. Use `hybrid_search_context` with a natural language query describing what you need; documents found by BOTH FTS and semantic methods rank highest
2. Best for finding prior solutions, knowledge, principles, and conceptually related content
3. Search iteratively: start broad with higher limits (10-20+), assess truncated previews + summaries, refine queries, then retrieve full content of best matches via `get_context_by_ids`

**Use hybrid search aggressively.** Because results are lightweight (truncated), you can safely perform multiple rounds of searching with different queries and higher limits without overwhelming the context window. IF IN DOUBT - USE IT!

## Pattern 3 - Semantic Search (Optional)

Meaning-based discovery when hybrid search is not needed: use `semantic_search_context` with a query describing what you need.

## Pattern 4 - Full-Text Search (Optional)

Precise keyword matching with `fts_search_context`:

1. Use `boolean` mode for complex queries: `"python AND async NOT deprecated"`
2. Use `phrase` mode for exact matches: `"error handling"`
3. Enable `highlight: true` to see matching snippets

## Pattern 5 - References Navigation (Optional)

Follow knowledge graph links when deeper context is needed -- research plans reference prior research you need to understand, validation reports reference implementations you need to verify, or you see a chain of work and need the full picture, including the full user intent:

1. Retrieve entries using the retrieval sequence (Steps 1-2, optionally 3-4)
2. Use `get_context_by_ids` to retrieve selected referenced entries
3. Repeat if those entries have further relevant references (depth limit: 2-3 levels)

**Example:** a retrieved validation report (entry 3357) has `metadata.references.context_ids: [3349, 3352]` -- potentially the implementation plan and implementation report. Retrieve them for the complete picture: `get_context_by_ids(context_ids=[3349, 3352])`.

</patterns>

<examples>

# Behavioral Examples

<example scenario="complete_mandatory_sequence">
**Input:** Agent starts task, receives instructions from orchestrator
**Correct Approach:** (1) Obtain thread ID; (2) Step 1: call `search_context(thread_id="session-id", source="user", limit=10)` and `search_context(thread_id="session-id", source="agent", limit=30)`; (3) Step 2: call `get_context_by_ids(context_ids=[...])` to retrieve full content; (4) if in a multi-agent workflow, verify the orchestrator task against retrieved user messages and agent reports; (5) Step 3: call `hybrid_search_context` if additional context is needed
**Result:** Agent has full context of user requirements, verified orchestrator task, and implementation plans
</example>

<example scenario="orchestrator_verification">
**Input:** Orchestrator provides task "Implement feature X with approach A"
**Correct Approach:** (1) Execute Steps 1-2 to retrieve user messages and agent reports; (2) compare the orchestrator task against user messages; (3) discover the user message says "Use approach B, not A"; (4) flag the discrepancy; (5) execute based on the USER requirement (approach B)
**Result:** Agent correctly identifies the orchestrator error and follows the user's actual requirements
</example>

<example scenario="truncation_aware_hybrid_search">
**Input:** Agent completed Steps 1-2 but is uncertain all relevant context was retrieved (e.g., prior decisions about database schema design)
**Correct Approach:** (1) Execute Step 3: `hybrid_search_context(query="database schema design decisions", thread_id="session-id", limit=15)`; (2) review truncated text + summary + metadata of each result to assess POTENTIAL RELEVANCE only -- do not conclude what an entry recommends or decides based on truncated previews; (3) retrieve full content of the entries that appear relevant: `get_context_by_ids(context_ids=[...relevant IDs...])`; (4) only AFTER reading full content, reason about what the entries actually say; (5) if insufficient, search again with refined queries or different terms
**Result:** Agent finds conceptually related context that metadata filtering missed, retrieves full content before drawing any conclusions about substance, and avoids the silent failure mode of acting on truncated approximations
</example>

<example scenario="protocol_violation">
**Input:** Agent receives orchestrator task and skips context retrieval, trusting the orchestrator's summary
**Incorrect Approach:** Agent proceeds directly with the task based only on orchestrator-provided information
**Result:** Not recommended - Agent missed critical user requirements and produced incorrect work
**Correct Action:** Execute retrieval Steps 1-2 before examining any task
</example>

<example scenario="references_navigation">
**Input:** Agent retrieves an implementation report whose metadata shows `references.context_ids: [3322, 3323]`
**Correct Approach:** Call `get_context_by_ids(context_ids=[3322, 3323])` to retrieve BOTH referenced entries
**Result:** Agent has the complete context chain (3322) -> (3323) -> current entry, enabling full traceability and verification of decisions
</example>

</examples>

<compliance_checklist>

# Compliance Checklist

Before proceeding with your task, consider verifying the following:

- [ ] **User messages retrieved**: Called `search_context(source="user")`
- [ ] **Agent reports retrieved**: Called `search_context(source="agent")`
- [ ] **Full content retrieved**: Called `get_context_by_ids` for full content of relevant entries
- [ ] **Hybrid search considered**: Evaluated whether `hybrid_search_context` is needed for additional context
- [ ] **References considered**: Checked `metadata.references.context_ids` in retrieved entries; followed references when deeper context was needed

Completing this checklist is a best practice for reliable results.

</compliance_checklist>

<error_handling>

# Error Handling

**If a context retrieval step fails:** retry once after a brief pause; document the failure in your work report; continue with the remaining steps of the retrieval sequence; and note limitations in your analysis due to incomplete context. A single failure does not excuse skipping other steps.

**If ALL context retrieval fails**, results will be significantly degraded without context server access:

1. **Log the failure** with the specific error message
2. **Proceed with available information** if any context was obtained through other means
3. **If no context is available at all**, inform the caller that results may be incomplete:
   ```text
   WARNING: Context server unavailable. Proceeding with limited context.
   Error: [specific error message]
   Impact: Unable to retrieve session history. Results may be incomplete or miss prior decisions.
   ```
4. **Note limitations** in your work report so downstream consumers know context was unavailable

**Rationale:** The context server provides session continuity and coordination; without it, work can still proceed, but with reduced confidence.

</error_handling>
