---
name: context-retrieval-protocol
description: |
  Retrieval workflow for an MCP-compatible context server: browse and retrieve, hybrid, semantic, and full-text search; retrieving user messages first and agent reports second; verifying a delegated task against the stored user messages, whose wording outranks the orchestrator's; taking a coordinator's context_scope list as a starting point, not a limit; treating every search result as a truncated preview and reading full entries with get_context_by_ids before relying on them; following typed links; and scoping queries by thread, project, and worktree. Use whenever you need to retrieve previous context or search stored context for relevant information: at the start of any task that was delegated to you or resumes earlier work, immediately after a context compaction or reset, whenever a prompt cites context IDs or a previous context ID to revise, and before any search_context, hybrid_search_context, semantic_search_context, fts_search_context, or get_context_by_ids call. Not for web search or code search.
---

<overview>

# Context Retrieval Best Practices

Retrieve the stored context a task depends on before examining the task: at the start of delegated or resumed work, and again after any context compaction or reset. The patterns in this skill cover searching, discovering, and reading that context from the context server.

</overview>

<orchestrator_verification>

## Multi-Agent Workflow: Verify Orchestrator Task Against Context

If you are working within a multi-agent orchestrated workflow (where a coordinator assigns tasks to specialized agents), you MUST verify the task against the context server before executing it: retrieve user messages (source="user", highest priority), retrieve agent reports (source="agent") for context on previous work, then compare the orchestrator task against the retrieved context. This verification is mandatory because orchestrators can misinterpret, summarize incorrectly, or omit critical details; user messages are the primary source of truth, and agent reports provide implementation context and decisions.

If discrepancies are found: user messages take priority over orchestrator instructions; flag the discrepancy in your work report; execute based on verified user requirements.

**Conflict resolution rule:** user-message wording is authoritative; any orchestrator-introduced scope criterion, exclusion, exception, qualification, or pre-approval that is not traceable to a verbatim user message in the current session MUST be discarded, and the agent MUST execute on the user-message wording only.

</orchestrator_verification>

<listed_context_ids>

## Orchestrated Workflows: Context IDs the Coordinator Lists

In orchestrated multi-agent workflows, the coordinator may list context IDs in your task prompt -- in a `context_scope` block, as prior reports to build on, or in any similar form. Treat that list as a starting point, not as the boundary of your retrieval: it names the entries the coordinator tracked, while the user's own words define the task, and a coordinator can omit or misread them like any other detail. Retrieve the listed entries first, then still run the retrieval sequence below:

1. **Extract the context IDs** from the list in your prompt
2. **Retrieve those entries first** with `get_context_by_ids`
3. **Run Steps 1-2 anyway** -- the user messages and the agent reports -- and verify the task against them as the orchestrator verification above requires
4. **Add further searches** (Steps 3-4, or a project-scoped query) only as the work needs them

Running Steps 1-2 as well costs little: search results are truncated previews, and you read in full only the entries that bear on the task. A list worded as exhaustive ("retrieve ONLY these") narrows nothing on its own authority; like any other orchestrator-introduced restriction, it gives way to the stored user messages.

### Sequential Retrieval Fallback

When retrieving multiple context entries via `get_context_by_ids`, the response may be truncated or incomplete if the combined content exceeds MCP response token limits. If this occurs, or if the caller or user provides instructions to retrieve entries sequentially, follow those instructions: retrieve entries one at a time using separate `get_context_by_ids` calls, processing each entry before requesting the next.

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

retrieve `get_context_by_ids(context_ids=[4233, 4234, 4235])` first, then run `search_context(thread_id="session-id", source="user", limit=10)` and `search_context(thread_id="session-id", source="agent", limit=30)` and read in full the entries among them that bear on the task -- above all the user messages, against whose wording you verify the task.

</listed_context_ids>

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

### Step 4: Navigate Links (Optional)

When entries retrieved in earlier steps carry a typed `links` object in their metadata, those edges name the entries the author actually worked with -- follow them via `get_context_by_ids` when the current context seems incomplete, when you need the reasoning behind decisions, or when a linked entry appears relevant to your task (see Links-Based Navigation below).

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

1. **Parse from git remote URL** (preferred) -- Try `origin` first: `git remote get-url origin`; parse the repository name from the URL (`https://github.com/user/my-project.git` -> `my-project`; `git@github.com:user/my-project.git` -> `my-project`). If `origin` is unavailable, try `upstream`, then the first available remote
2. **Git toplevel basename** (fallback for repos without remotes) -- `git rev-parse --show-toplevel`, extract the last path component (`/home/user/projects/my-project` -> `my-project`)
3. **Current directory basename** (fallback for non-git directories) -- Extract the last directory name from the working directory path (e.g., `/home/user/work/my-project` -> `my-project`)

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

### Warning: Cross-Worktree Context

Exercise caution with context from other worktrees: different worktrees have different branches checked out, so referenced file paths may not exist, implementation status may differ (code merged in one worktree may not exist in another), and features implemented for one branch may conflict with another. Safe usage:

1. Use cross-worktree context for **conceptual understanding** (patterns, decisions, rationale)
2. **Always verify** that referenced files exist in the current worktree before referencing or editing them
3. **Do not assume** implementation status applies to the current branch

</worktree_queries>

<environment_integration>

## Environment Integration Patterns

Context retrieval operations can interact with environment-level hooks, validation gates, and orchestration workflows:

- **Hook-aware retrieval:** Environment hooks may verify that agents retrieve context before starting work (enforcing retrieval discipline), compare retrieved context against task instructions to detect discrepancies, or log which context entries were retrieved by which agent for traceability. In such environments, follow the documented sequence (Quick Recall or Comprehensive Multi-Agent Pattern) to avoid triggering validation failures.
- **Metadata patterns for multi-agent coordination:** Filter by `kind` to find records by type (`report`, `plan`, `handoff`, `checkpoint`) regardless of author, by `agent_name` to find prior work from a specific agent role, by `kind` plus `status` to find completed work or work requiring continuation (never filter `status` without `kind` -- the same token can mean different things in different kind vocabularies), and by `report_type` to find validation, research, or implementation reports; follow typed `links` edges to reconstruct the full workflow chain (research -> plan -> implementation -> validation).
- **Orchestrated workflows:** Context retrieval serves as the shared memory layer -- agents retrieve prior agent reports to understand completed work before continuing (task handoff), retrieve implementation plans and compare against current task instructions (plan verification), and retrieve user messages to verify orchestrator instructions match user intent (conflict detection; see Orchestrator Verification section).

These patterns are generic and apply to any environment with multi-agent coordination capabilities.

</environment_integration>

<tools>

# Available Context Server Tools

**Note:** Not all tools listed below may be available in your environment; availability depends on server configuration and how the server is connected to your MCP client. Use the tools that are available to you; if a recommended tool is unavailable, use an alternative from this table. The tools below cover retrieval -- for storage and update operations, the context server exposes a parallel set of tools (for example `store_context` and `update_context`); consult the storage section of the server's own documentation.

| Tool                      | Status           | Returns                           | Use For                                                |
|---------------------------|------------------|-----------------------------------|--------------------------------------------------------|
| `search_context`          | Recommended      | Truncated text + summary          | Browse/discover entries before retrieval               |
| `get_context_by_ids`      | Recommended      | Full text + images                | Retrieve specific entries after discovery              |
| `hybrid_search_context`   | Recommended      | Truncated text + summary + scores | Best overall search (FTS + semantic)                   |
| `semantic_search_context` | Optional         | Truncated text + summary + scores | Meaning-based search (see Score Fields Reference)      |
| `fts_search_context`      | Optional         | Truncated text + summary + scores | Keyword/linguistic search (see Score Fields Reference) |
| `list_threads`            | Optional         | Thread list with statistics       | Discover available threads and their metadata          |
| `get_statistics`          | Optional         | Server statistics                 | Check server health and usage metrics                  |
| `delete_context`          | Use with caution | Confirmation                      | Remove specific context entries during cleanup         |

**Key notes:**

- **All search tools return truncated content** (text + summary): this applies to `search_context`, `hybrid_search_context`, `semantic_search_context`, and `fts_search_context` equally. Use `get_context_by_ids` to retrieve full content of relevant entries identified through search.
- Because results are truncated, you can search more aggressively: use higher limits (10-20+), perform multiple sequential searches with different queries, and iterate to find the best matches before retrieving full content. Use `hybrid_search_context` for conceptual discovery -- use it when in doubt.
- Specify `thread_id` to search within the current session.

## Truncation Discipline

Every search tool (`search_context`, `hybrid_search_context`, `semantic_search_context`, `fts_search_context`) returns previews for judging relevance, not substance: the text is truncated, and the summary is an AI-generated approximation that can drop conditions, caveats, or nuance the full entry carries. Before you state, act on, or pass on what an entry says, recommends, or decides, read it in full with `get_context_by_ids`.

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
| `fts_score`         | Higher = better | BM25/ts_rank relevance                                                  |
| `fts_rank`          | Lower = better  | FTS result rank (1 = best)                                              |
| `semantic_distance` | Lower = better  | Similarity-ordered: L2 (fp32/mse) or negated inner product (ip variant) |
| `semantic_rank`     | Lower = better  | Semantic result rank (1 = best)                                         |
| `rrf`               | Higher = better | Combined RRF score                                                      |
| `rerank_score`      | Higher = better | Cross-encoder relevance (0.0-1.0)                                       |

</tools>

<metadata_reference>

## Metadata Filtering

The metadata vocabulary you filter on is defined normatively by the schema skill -- before composing non-trivial metadata queries, invoke `Skill(skill="context-metadata-schema")` for the full field tables, kind registry, status vocabularies, and filter recipes. Supported filter operators include direct equality (via the `metadata` parameter) and the `metadata_filters` advanced operators such as `eq`, `ne`, `gt`, `lt`, `contains`, `array_contains`, `starts_with`, `exists`, and similar comparators.

**Quick Reference for Filtering:**

| Filter By      | Use Parameter                                             | Example                                                                                                   |
|----------------|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Record kind    | `metadata: {"kind": "report"}`                            | Find records by type regardless of author                                                                 |
| Kind + status  | `metadata: {"kind": "plan", "status": "pending"}`         | Find work requiring continuation (NEVER filter `status` without `kind`)                                   |
| Agent          | `metadata: {"agent_name": "..."}`                         | Find all reports from one agent role                                                                      |
| Project        | `metadata: {"project": "..."}`                            | Scope to current project                                                                                  |
| Report type    | `metadata: {"kind": "report", "report_type": "research"}` | Find all research reports                                                                                 |
| Technology     | `array_contains` or tags                                  | `metadata_filters: [{key: "technologies", operator: "array_contains", value: "python"}]`                  |
| Incoming links | `metadata_filters` with `array_contains`                  | `[{key: "links.derived_from", operator: "array_contains", value: 2322}]` finds entries that build on 2322 |

**Notes:** Entries stored before the schema existed carry no `kind`/`schema_version`; filtering on those fields excludes the legacy corpus by construction, so fall back to `source`, `tags`, and text search when older context matters. When a links filter value is a string ID (hex), add `"case_sensitive": true` for exact, index-friendly matching; integer ID values need no flag. For technology filtering, use `array_contains` for exact element match or the `tags` parameter for OR logic: `tags: ["python", "fastapi"]`.

</metadata_reference>

<revision_context_detection>

## Advanced: Revision Context Detection

This section is relevant for multi-agent workflows where agents update each other's prior work; in a simple single-agent setup, you can skip it.

When your task prompt contains revision indicators, extract the previous context_id and use `update_context` instead of `store_context`.

**Revision Indicators in Prompt:**

| Pattern                    | Meaning                                 |
|----------------------------|-----------------------------------------|
| `PREVIOUS CONTEXT ID: [N]` | Explicit signal to update entry N       |
| `PLAN REVISION REQUEST`    | Revision mode - look for context_id     |
| `RESEARCH CONTINUATION`    | Continuation mode - look for context_id |

**Extraction protocol:** (1) detect revision mode by scanning the prompt for the indicators above; (2) extract the context_id (e.g., `PREVIOUS CONTEXT ID: 123` -> `123`); (3) retrieve the previous entry with `get_context_by_ids(context_ids=[extracted_id])`; (4) store the context_id for use with `update_context` when saving.

### Finding Prior Entries to Update

When you need to update your own prior work but the context_id is not provided:

```text
search_context(
  thread_id="session-id",
  source="agent",
  metadata={"agent_name": "[your-agent-name]", "kind": "report", "report_type": "research"},
  limit=15
)
```

Then use `update_context(context_id=...)` with the most recent matching entry. Which agent wrote an entry does not limit whether you may update it: when your work requires changing another agent's entry -- for example repairing the stale status of an entry that a newer one supersedes -- find it the same way by that agent's `agent_name` and update it.

</revision_context_detection>

<links_navigation>

## Links-Based Navigation

When you retrieve context entries, check for a typed `links` object in metadata. Its edges are not random -- each key states why the author connected the entries: `derived_from` names the entries the work builds upon, `evidence` names the entries backing specific claims, `commissioned_by` names the user message that spawned the work, `supersedes` names the entries this one replaces, and `parent` attaches a sub-issue to its issue or a comment to its target. These connections form a typed knowledge graph you can navigate in both directions.

**Outgoing (follow the entry's own edges):** retrieve the IDs stored under the relevant key via `get_context_by_ids`. Follow `derived_from` for the reasoning behind decisions, `commissioned_by` for the authoritative user intent, `evidence` to verify claims.

**Incoming (query the reverse direction):** edges are stored once, on the active side, so the reverse question is an `array_contains` query. Who built on entry X: `metadata_filters=[{"key": "links.derived_from", "operator": "array_contains", "value": X}]`. Is plan X still current: the same query on `links.supersedes` -- a hit means a newer entry replaced X, and the edge outranks a stale `status`. Comments on X vs sub-issues of X: `array_contains` on `links.parent` with value X, paired with `kind: "comment"` or `kind: "issue"` respectively (always pair a `links.parent` query with `kind`). For `related`, which is symmetric and stored once, the complete answer is the union of X's own `links.related` array and the reverse `array_contains` query.

**Legacy entries:** records stored before this schema carry an untyped `references` object (typically `references.context_ids`). Read it during navigation exactly as you would `derived_from` -- the pointers are real, only untyped -- but never write `references` in new entries.

**How to navigate:** identify the relevant edges in the retrieved entry's metadata, retrieve the linked entries with `get_context_by_ids`, and evaluate relevance -- not all linked entries may be needed for the current task:

```json
"metadata": {
  "links": {
    "derived_from": [3348, 3349],
    "commissioned_by": [3340]
  }
}
```

```text
get_context_by_ids(context_ids=[3348, 3349, 3340])
```

### Navigation Depth Guidance

| Scenario                   | Recommended Depth            |
|----------------------------|------------------------------|
| Understanding current task | 1 level (direct links)       |
| Tracing decision history   | 2 levels (links of links)    |
| Comprehensive research     | Follow until pattern emerges |

</links_navigation>

<context_continuity>

## Context Continuity Patterns

These patterns help agents maintain coherence across context window boundaries and long-running tasks.

### Basic Continuity (Default)

Apply these patterns by default in all sessions:

- **Status tracking:** Always read `kind` together with `status` to interpret completion state -- each kind family has its own vocabulary: `pending`/`done`/`superseded` for work artifacts, and separate lifecycles for `issue` and `upstream_issue` entries
- **Session handoff notes:** When resuming earlier work, retrieve the latest `kind: "handoff"` entry for the task first -- it records the work completed, key decisions, unresolved issues, and next steps the previous session left
- **Task completion markers:** Typed `links` edges (`derived_from`, `commissioned_by`, `supersedes`) connect new work to the prior entries it builds upon, creating a navigable chain of work history
- **Re-retrieval after context loss:** After any context compaction or window reset, re-read key context entries (plans, requirements, prior decisions) from the server to restore working memory. Do not rely on compacted summaries or search-truncated previews for critical details -- always retrieve full content via `get_context_by_ids`

### Advanced: Long-Running Task Continuity (Optional)

For tasks spanning multiple context windows or requiring extended multi-step execution, consider these additional patterns:

- **Progressive summarization:** Periodically condense accumulated context into structured summaries stored on the context server, preserving critical information (architectural decisions, unresolved issues, implementation progress) while reducing context window pressure. Store summaries as new entries whose `links.derived_from` points at the original detailed entries
- **Checkpoints:** When resuming a multi-step task, retrieve its latest `kind: "checkpoint"` entry -- each milestone stores a new one, linked to the previous checkpoint through `links.derived_from` -- for the progress it records (what is completed, what remains), key decisions and their rationale, active blockers or dependencies, and files modified and their purpose; follow that link back to an earlier checkpoint only when the latest one leaves out something you need. This enables recovery if a session is interrupted and provides a clear starting point for the next session
- **Plan currency check:** Before resuming any plan, verify it has not been superseded: query `array_contains` on `links.supersedes` with the plan's ID, and prefer the replacing entry when one exists
- **Context window monitoring:** If approaching context limits, proactively store current progress before compaction occurs; after compaction, immediately retrieve critical context entries (plans, requirements) from the server. Treat the context server as persistent memory that survives compaction -- store anything that must not be lost
- **Multi-agent handoff:** For clean agent-to-agent transitions in orchestrated workflows, the completing agent stores a comprehensive handoff report with clear next steps; the receiving agent retrieves the handoff report and its linked context before starting; both agents use consistent typed links to maintain the work chain; and disagreements between orchestrator instructions and stored context are resolved in favor of stored user messages (see Orchestrator Verification)

</context_continuity>

<strategy>

# Retrieval Strategy

- Retrieve relevant user and agent context to understand the current task
- Query the context server as many times as needed; you can return to it at any point during your work
- Search iteratively (see the tools section's Key notes for how aggressively you can search), and use `get_context_by_ids` to retrieve full content only for entries that appear relevant
- Include `include_images: true` to capture visual context (diagrams, matrices, charts)

</strategy>

<patterns>

# Retrieval Patterns

The Context Retrieval Sequence's Steps 1-4 above are the canonical enumeration of what to run; this section covers only the decision guidance those steps do not already state -- when to reach for one search tool over another. For every pattern below: specify `thread_id` to search within the current session, and remember that results are truncated -- assess relevance from truncated text + summary + metadata, then use `get_context_by_ids` for full content of relevant entries.

## Choosing Among Step 3's Search Tools

Step 3 covers `hybrid_search_context` as the recommended default for conceptual discovery: it is best for finding prior solutions, knowledge, principles, and conceptually related content, because documents found by both FTS and semantic methods rank highest. Reach for `semantic_search_context` instead only when you specifically want meaning-based matches without the FTS half of hybrid search. Reach for `fts_search_context` when you need precise keyword control that hybrid search does not expose: `boolean` mode for complex queries (`"python AND async NOT deprecated"`), `phrase` mode for exact matches (`"error handling"`), and `highlight: true` to see matching snippets.

## Choosing to Navigate Links (Step 4)

Follow the typed knowledge graph when deeper context is needed -- a plan's `derived_from` names the research behind it, a validation report's `evidence` names what it verified, `commissioned_by` restores the full user intent, and reverse `array_contains` queries answer who-builds-on-this and is-this-superseded (see Links-Based Navigation above for the full outgoing/incoming query forms).

**Example:** a retrieved validation report (entry 3357) has `links.evidence: [3349, 3352]` -- the implementation plan and implementation report it verified. Retrieve them for the complete picture: `get_context_by_ids(context_ids=[3349, 3352])`.

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
**Correct Approach:** (1) Execute Steps 1-2 to retrieve user messages and agent reports; (2) compare the orchestrator task against user messages; (3) discover the user message says "Use approach B, not A"; (4) flag the discrepancy; (5) execute based on the user requirement (approach B)
**Result:** Agent correctly identifies the orchestrator error and follows the user's actual requirements
</example>

<example scenario="truncation_aware_hybrid_search">
**Input:** Agent completed Steps 1-2 but is uncertain all relevant context was retrieved (e.g., prior decisions about database schema design)
**Correct Approach:** (1) Execute Step 3: `hybrid_search_context(query="database schema design decisions", thread_id="session-id", limit=15)`; (2) review truncated text + summary + metadata of each result to assess potential relevance only -- do not conclude what an entry recommends or decides based on truncated previews; (3) retrieve full content of the entries that appear relevant: `get_context_by_ids(context_ids=[...relevant IDs...])`; (4) only after reading full content, reason about what the entries actually say; (5) if insufficient, search again with refined queries or different terms
**Result:** Agent finds conceptually related context that metadata filtering missed, retrieves full content before drawing any conclusions about substance, and avoids the silent failure mode of acting on truncated approximations
</example>

<example scenario="protocol_violation">
**Input:** Agent receives orchestrator task and skips context retrieval, trusting the orchestrator's summary
**Incorrect Approach:** Agent proceeds directly with the task based only on orchestrator-provided information
**Result:** Agent missed critical user requirements and produced incorrect work
**Correct Action:** Execute retrieval Steps 1-2 before examining any task
</example>

<example scenario="links_navigation">
**Input:** Agent retrieves an implementation report whose metadata shows `links.derived_from: [3322]` and `links.commissioned_by: [3310]`
**Correct Approach:** Call `get_context_by_ids(context_ids=[3322, 3310])` to read the plan the implementation was built from and the user message that commissioned it; if the report's currency matters, additionally run `metadata_filters=[{"key": "links.supersedes", "operator": "array_contains", "value": <report-id>}]` to confirm nothing replaced it
**Result:** Agent has the complete typed chain (user intent -> plan -> implementation), enabling full traceability and verification of decisions
</example>

</examples>

<error_handling>

# Error Handling

**If a context retrieval step fails:** retry once after a brief pause; document the failure in your work report; continue with the remaining steps of the retrieval sequence; and note limitations in your analysis due to incomplete context. A single failure does not excuse skipping other steps.

**If all context retrieval fails**, results will be significantly degraded without context server access:

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
