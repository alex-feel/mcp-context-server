---
name: context-preservation-protocol
description: |
  Context preservation patterns for storing work results and session context via an MCP-compatible context server.
  Provides patterns for documenting work, storing reports, and ensuring continuity between sessions.
  Use when you need to preserve work results or session context.
---

<overview>

# Context Preservation Best Practices

Storing work documentation and context before stopping is MANDATORY whenever you have context-server store capability and produced substantive work this session: only the durable record survives a context reset or compaction, so an artifact left in an ephemeral channel is lost. The patterns in this skill help you structure and store your work results in the context server.

</overview>

<schema_directive>

# Mandatory First Step: Load the Metadata Schema

Before composing the metadata of ANY entry you store or update, invoke `Skill(skill="context-metadata-schema")`. That skill is the single normative source for the metadata contract: the universal core fields (`schema_version`, `kind`, `project`, `links`), the kind registry (`report`, `plan`, `handoff`, `checkpoint`, `note`, `issue`, `comment`, `user_message`), the kind-scoped status vocabularies, the typed `links` registry with its design rules, the filter recipes, and the operational rules for link appends, supersession, deduplication, and concurrency. This skill covers the storage WORKFLOW; the metadata examples below show the contract in action but do not redefine it -- on any discrepancy, the schema skill wins.

</schema_directive>

<thread_id>

# How to Obtain Thread ID

The thread ID is used as `thread_id` for context server queries. Obtain it using the following search chain:

1. **Already available** -- If the thread ID is provided via context or prompt, use it directly
2. **Thread ID file** -- Check `.context_server/.thread_id` in the project working directory
3. **Project directory name** -- If no thread ID file exists, derive the thread identifier from the project directory basename using the git remote URL fallback chain (remote URL repository name preferred, then git toplevel basename, then current directory basename). Using the project name ensures all agents working on the same project write to the same thread, which is essential for multi-agent coordination

</thread_id>

<tools>

# Available Context Server Tools

**Note:** Not all tools listed below may be available in your environment; availability depends on server configuration and how the server is connected to your MCP client. Use the tools available to you; if a recommended tool is unavailable, use an alternative from this table.

The tools below cover storage and update. For retrieval and search, the context server exposes a parallel set of tools (for example `search_context`, `get_context_by_ids`, `hybrid_search_context`, `semantic_search_context`, and `fts_search_context`) -- consult the retrieval section of the server's own documentation.

| Tool                   | Status           | Use For                                              |
|------------------------|------------------|------------------------------------------------------|
| `store_context`        | RECOMMENDED      | Store NEW entry (standard for fresh work reports)    |
| `update_context`       | RECOMMENDED      | Update EXISTING entry (for revisions/continuations)  |
| `store_context_batch`  | Optional         | Store multiple entries at once (rarely needed)       |
| `update_context_batch` | Optional         | Update multiple entries at once (rarely needed)      |
| `delete_context`       | Use with caution | Delete specific context entries                      |
| `delete_context_batch` | Use with caution | Delete multiple context entries at once              |
| `list_threads`         | Optional         | Discover available threads and their metadata        |
| `get_statistics`       | Optional         | Check server health and usage metrics                |

Use `store_context_batch` ONLY when storing multiple independent entries in a single operation (typically migrations, imports, or bulk data operations) -- NOT for normal work reports (use `store_context` instead).

**Protocol requirements:**

- `metadata`: Required for schema-v1 entries -- the universal core enables every filter recipe other agents rely on
- `tags`: Recommended -- the kind token plus labels enables search and categorization
- `images`: optional

**Deduplication caution:** `store_context` collapses an entry whose (thread, source, text) exactly matches an existing entry into an update of that entry, shallow-overriding metadata. Give every stored body distinguishing content early (concrete specifics, not reusable boilerplate alone) so unrelated records never collapse.

</tools>

<update_strategy>

## Context Update Strategy

### When to Use update_context vs store_context

Use `update_context` when revising a previously stored plan based on user feedback, continuing research that was marked INCOMPLETE, correcting errors in a prior report, or updating status from "pending" to "done". Use `store_context` when creating fresh research/implementation work, when no prior context_id exists for this task, or when starting a new research thread.

### update_context Parameters

| Parameter        | Required | Description                                        |
|------------------|----------|----------------------------------------------------|
| `context_id`     | YES      | ID of the entry to update                          |
| `text`           | NO       | Complete revised text (replaces existing entirely) |
| `metadata`       | NO       | Full metadata replacement (replaces all metadata)  |
| `metadata_patch` | NO       | Partial metadata update (RFC 7396 merge semantics) |
| `tags`           | NO       | Updated tags (replaces existing tags entirely)     |

**Important:** Use `metadata_patch` (not `metadata`) for revisions to preserve fields you do not want to change. The `updated_at` timestamp is set automatically by the server, and embeddings are regenerated when text changes.

### Metadata Merge Semantics (RFC 7396)

With `metadata_patch`: new keys are ADDED, existing keys are UPDATED with new values, keys set to `null` are DELETED, omitted keys are PRESERVED unchanged, and nested objects deep-merge key-by-key -- but ARRAYS are always replaced whole, never merged element-wise.

**Appending to a links array is therefore read-modify-write:** retrieve the entry with `get_context_by_ids`, append the new ID client-side to the touched key's array, then patch ONLY that key with its complete new array (for example `metadata_patch={"links": {"derived_from": [3348, 3349, 3401]}}`). Sibling keys inside `links` survive the deep merge untouched. Never hand-construct an array you did not first read.

**Superseding an earlier entry is a two-write sequence:** FIRST store the replacing entry with `links.supersedes` naming the old ID, THEN patch the old entry to `status: "superseded"`. If the second write is lost, the edge wins -- readers trust `links.supersedes` over a stale status, and any session noticing the mismatch repairs it.

**Concurrency:** `update_context` uses compare-and-set versioning and fails with a version-conflict error when the entry changed underneath you. On conflict, refetch, reapply your change to the fresh state, and retry once.

### Update Protocol for Plan Revisions

When updating an existing entry for plan revision:

1. **Extract context_id** from the prompt (e.g., `PREVIOUS CONTEXT ID: 123`)
2. **Retrieve previous entry:** `get_context_by_ids([context_id])`
3. **Verify ownership:** Check that `agent_name` in metadata matches your agent identifier
4. **Create revised content:** Generate the updated plan as one coherent revision (never an appended addendum)
5. **Call update_context:**
   ```text
   update_context(
       context_id=<extracted_id>,
       text=<revised_report>,
       metadata_patch={
           "revision_count": <current + 1 or 1 if first revision>,
           "status": "done"
       },
       tags=["report", "implementation-guide", "research", ...]
   )
   ```
6. **Return SAME context_id** in status message

</update_strategy>

<environment_integration>

## Environment Integration Patterns

Context preservation operations can interact with environment-level hooks, validation gates, and orchestration workflows. Environment hooks may validate that stored context includes required metadata fields, correct tagging, and proper typed links; log storage operations for traceability, verifying that agents store work results before session completion; or reject entries lacking required structure (e.g., missing `kind`, `status`, or `agent_name`). In such environments, follow the metadata contract and compliance checklist rigorously to avoid validation failures.

### Metadata Patterns for Multi-Agent Coordination

Structured metadata enables sophisticated workflows across multiple agents. These patterns are generic and apply to any environment with multi-agent coordination capabilities:

- **Work chain linking:** Always populate `links.derived_from` with the IDs of entries your work builds upon, `links.commissioned_by` with the user message that spawned the work, and `links.evidence` with entries backing specific claims. Typed edges create navigable chains that other agents and orchestrators can follow and that survive context window resets; the meaning of each pointer is its key, so readers never guess
- **Agent identification:** Always set `agent_name` to enable filtering by agent role. This is critical for orchestrators that need to find specific agent outputs
- **Status signaling:** Use `status: "pending"` to signal that work requires continuation, `status: "done"` to signal completion, and `status: "superseded"` (set by the supersession sequence) to signal replacement. Future sessions, other agents, and orchestrators use this to determine workflow progression -- always alongside `kind`, never as a bare status filter
- **Kind classification:** Set `kind` accurately (`report`, `plan`, `handoff`, `checkpoint`, `note`) so cross-agent discovery works by record type regardless of which agent produced the record; `report_type` further classifies reports
- **Handoff readiness:** In multi-agent orchestrated environments, structure every stored report so that another agent can understand the work without additional context. Include goals, work performed, results, and explicit next steps
- **Tag consistency:** Include the kind token in `tags` plus consistent labels across related entries to enable grouped retrieval (e.g., all entries tagged with a specific feature or task name)

</environment_integration>

<strategy>

# Preservation Strategy

When you have context-server store capability and produced substantive work this session, you MUST complete the following before stopping (if you already stored this report earlier in the same session and it is unchanged, do not store it again):

1. **Create a comprehensive Markdown report** of your work results:

   **FIRST CHECK**: If you have a specific report structure defined in your own agent instructions, use your own STRUCTURE within the Markdown format. **ONLY IF NO SPECIFIC FORMAT EXISTS**, use the following structure:

   ```markdown
   ## Summary
   - Brief overview including key decisions, recommendations, and conclusions

   ## Goals
   - What goals you were tasked to achieve

   ## Work Performed
   - Detailed list of all tasks completed

   ## Results Achieved
   - Detailed documentation, outcomes, deliverables
   - Examples (code snippets, configurations)
   - URIs (URLs, file paths)
   - References (version numbers, filenames, entity names, line numbers)
   - Any other relevant information
   ```

   **Front-load critical information:** Place key findings, decisions, recommendations, and conclusions in the opening section (Summary) of your stored entries. Search tools return truncated previews from the beginning of stored text -- information buried deep in an entry may be invisible during search-based discovery, causing other agents to misjudge relevance and skip retrieval of entries that contain important content.

2. **Always use English** to write the report, REGARDLESS of the language requested by the calling party.

3. **Save the report** using `store_context` with these parameters:
   - `thread_id`: Your thread ID (REQUIRED)
   - `source`: `agent` (REQUIRED)
   - `text`: Your complete Markdown report (REQUIRED)
   - `metadata`: Compose per the schema skill loaded above -- the universal core plus the fields for your kind. A completed work report looks like:
     ```json
     {
       "schema_version": 1,
       "kind": "report",
       "project": "[canonical project name]",
       "agent_name": "[your agent name]",
       "task_name": "[human-readable task description]",
       "status": "done",
       "report_type": "research | implementation | validation | documentation",
       "technologies": ["list", "of", "technologies"],
       "links": {
         "derived_from": [],
         "commissioned_by": []
       }
     }
     ```
   - `tags`: The kind token plus relevant labels (e.g., `["report", ...]`)

4. **After successfully saving**, capture the `context_id` from the `store_context` response and include it in your brief completion status to the calling party -- format: `"[Brief status summary]. Report ID: [context_id]"` (e.g., `"Implementation complete. 3 features implemented. Report ID: 2510"`). The caller can use this ID to retrieve the full report via `get_context_by_ids([context_id])`

This ensures your work is documented, preserved, and **retrievable by other agents** who need your detailed findings. A structured-output return value or any other in-window reply to your caller is SEPARATE from this durable record and does NOT substitute for it; the ephemeral reply is lost on compaction, the stored entry is not. A dispatch instruction that forbids writing report files to disk (for example a swarm or deep-research "do not write files to disk" contract) governs on-disk files only and does NOT relieve you of storing the context-server entry.

</strategy>

<context_continuity>

## Context Continuity Patterns

These patterns help agents preserve state across context window boundaries and long-running tasks. They are the storage-side patterns; the symmetric retrieval-side patterns (search, re-read after compaction, links navigation) belong to the retrieval workflow and follow the same principles applied to retrieval tools.

### Basic Continuity (Default)

Apply these by default when storing context:

- **Status and link chains:** Always set `status` and populate the typed `links` keys (`derived_from`, `commissioned_by`, `evidence`) per the multi-agent coordination patterns above
- **Session handoff notes:** Before ending a session, store a `kind: "handoff"` entry describing work completed, key decisions, unresolved issues, and recommended next steps -- a briefing document for the next session
- **Pre-compaction preservation:** If approaching context window limits during extended work, proactively store current progress to the context server before compaction occurs. Critical details stored externally survive compaction intact

### Advanced: Long-Running Task Continuity (Optional)

For tasks spanning multiple context windows or extended multi-step execution:

- **Checkpoint storage:** At defined milestones, store a `kind: "checkpoint"` entry containing a summary of completed steps and remaining work, key decisions and their rationale, active blockers or dependencies, and the list of modified files and their purpose. Set `status: "pending"` and point `links.derived_from` at the task plan
- **Progressive summarization:** For tasks generating large volumes of context, periodically store condensed summary entries distilling key findings, decisions, and progress. Point `links.derived_from` at the original detailed entries and tag summaries consistently (e.g., with task name) for easy retrieval
- **Plan supersession:** When a revised plan replaces an earlier one as a NEW entry, follow the supersession sequence (store the new plan with `links.supersedes`, then patch the old plan to `status: "superseded"`) so later sessions can always answer "is there a newer plan"
- **Multi-agent handoff reports:** When another agent will continue your work, store a comprehensive handoff report that the receiving agent can understand without additional context: clear sections (Summary, Work Performed, Results, Next Steps, and others) covering goals, work performed, results, and explicit next steps; all relevant typed links so the receiving agent can trace the full work chain; and `kind`, `report_type`, and `agent_name` set accurately for precise filtering

</context_continuity>

<compliance_checklist>

# Compliance Checklist

Before returning to the calling party, verify the following whenever you had store capability and produced substantive work; completing this checklist is mandatory for reliable context preservation:

- [ ] **Schema loaded**: Invoked `Skill(skill="context-metadata-schema")` before composing metadata
- [ ] **Report created**: Comprehensive Markdown report documenting your work
- [ ] **Report saved**: Called `store_context` with thread_id, source="agent", text, metadata, and tags
- [ ] **Universal core complete**: Included `schema_version`, `kind`, `project` (plus `agent_name` and `status` for your kind)
- [ ] **Kind-specific fields**: Populated correctly (e.g., `report_type` and `technologies` per task subject, not execution tools)
- [ ] **Links typed**: Populated `links` with typed keys (`derived_from`, `commissioned_by`, `evidence`, ...); omitted or `{}` if none; never wrote a legacy untyped reference list
- [ ] **Tags included**: Added the kind token plus relevant categorization tags
- [ ] **Storage verified**: Confirmed `store_context` call succeeded before returning
- [ ] **Report ID returned**: Included `context_id` from `store_context` response in status message

</compliance_checklist>

<examples>

# Behavioral Examples

<example scenario="successful_preservation">
**Input:** Agent completed implementation task successfully
**Correct Approach:** (1) Invoke `Skill(skill="context-metadata-schema")`; (2) Create Markdown report following skill format; (3) Call `store_context(thread_id="session-id", source="agent", text="## Summary\n...", metadata={"schema_version": 1, "kind": "report", "project": "my-project", "agent_name": "developer", "task_name": "Implement authentication feature", "status": "done", "report_type": "implementation", "technologies": ["python", "fastapi"], "links": {"commissioned_by": [2481]}}, tags=["report", "implementation"])` and capture returned `context_id`; (4) Verify storage succeeded; (5) Return brief status with Report ID to caller
**store_context Response:** `{"success": true, "context_id": 2510, "thread_id": "session-id", "message": "..."}`
**Returned Status:** "Implementation complete. Auth feature implemented with 3 endpoints. Report ID: 2510"
</example>

<example scenario="partial_completion">
**Input:** Agent completed 2 of 3 tasks, blocked on third
**Correct Approach:** (1) Create report documenting completed work AND blocker; (2) Set status to "pending" in metadata; (3) Store report and capture `context_id`; (4) Return brief status with Report ID explaining blocker
**Returned Status:** "Partial completion. 2/3 tasks done. BLOCKED: Missing API credentials. Report ID: 2511"
</example>

<example scenario="context_server_failure">
**Input:** Agent completed work but `store_context` call fails
**Correct Approach:** (1) Attempt storage; (2) On failure, log error; (3) Return FULL REPORT to caller (not just status); (4) Inform caller of storage failure
**Returned to Caller:** Full Markdown report + "WARNING: Context server storage failed. Full report included above."
</example>

</examples>

<error_handling>

# Error Handling

## Storage Failure Protocol

Context server storage is mandatory for substantive work when you have store capability; failure to store means work results may be lost. If context storage fails (network error, server unavailable, timeout):

1. **Retry once** after 2 seconds for transient errors (timeout, 5xx)
2. **If retry fails or the error is non-transient (4xx, connection refused):** return the FULL REPORT to the caller inline in your response (not just a status summary) and inform the caller of the storage failure so they can decide next steps. Preserving the report inline ensures work is not lost entirely; the caller can manually store it later or take other action. Example fallback message:

   ```text
   WARNING: Context server storage failed. Full report included below.
   Error: [specific error message]
   Impact: Report not persisted to context server. Content preserved in this response only.
   ```

</error_handling>
