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

<thread_id>

# How to Obtain Thread ID

The thread ID is used as `thread_id` for context server queries. Obtain it using the following search chain:

1. **Already available** -- If the thread ID is provided via context or prompt, use it directly
2. **Thread ID file** -- Check `.context_server/.thread_id` in the project working directory
3. **Project directory name** -- If no thread ID file exists, derive the thread identifier from the project directory basename using the git remote URL fallback chain described below. Using the project name ensures all agents working on the same project write to the same thread, which is essential for multi-agent coordination

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

- `metadata`: Recommended - enables filtering by agent_name, task_name, status, project
- `tags`: Recommended - enables search and categorization
- `images`: optional

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

### Update Protocol for Plan Revisions

When updating an existing entry for plan revision:

1. **Extract context_id** from the prompt (e.g., `PREVIOUS CONTEXT ID: 123`)
2. **Retrieve previous entry:** `get_context_by_ids([context_id])`
3. **Verify ownership:** Check that `agent_name` in metadata matches your agent identifier
4. **Create revised content:** Generate the updated plan
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

### Metadata Merge Semantics (RFC 7396)

With `metadata_patch`: new keys are ADDED, existing keys are UPDATED with new values, keys set to `null` are DELETED, and omitted keys are PRESERVED unchanged.

**Example:**
```python
# Original metadata: {"agent_name": "implementation-guide", "status": "pending", "revision_count": 0}
# Patch: {"status": "done", "revision_count": 1}
# Result: {"agent_name": "implementation-guide", "status": "done", "revision_count": 1}
```

</update_strategy>

<environment_integration>

## Environment Integration Patterns

Context preservation operations can interact with environment-level hooks, validation gates, and orchestration workflows. Environment hooks may validate that stored context includes required metadata fields, correct tagging, and proper references; log storage operations for traceability, verifying that agents store work results before session completion; or reject entries lacking required structure (e.g., missing `status`, `agent_name`, or `references`). In such environments, follow the metadata schema and compliance checklist rigorously to avoid validation failures.

### Metadata Patterns for Multi-Agent Coordination

Structured metadata enables sophisticated workflows across multiple agents. These patterns are generic and apply to any environment with multi-agent coordination capabilities:

- **Work chain linking:** Always populate `references.context_ids` with IDs of entries your work builds upon. This creates navigable chains that other agents and orchestrators can follow and that survive context window resets; incomplete references break the traceability chain
- **Agent identification:** Always set `agent_name` to enable filtering by agent role. This is critical for orchestrators that need to find specific agent outputs
- **Status signaling:** Use `status: "pending"` to signal that work requires continuation, and `status: "done"` to signal completion. Future sessions, other agents, and orchestrators use this to determine workflow progression
- **Report type classification:** Use `report_type` consistently to enable cross-agent discovery (e.g., finding all validation reports regardless of which validator produced them)
- **Handoff readiness:** In multi-agent orchestrated environments, structure every stored report so that another agent can understand the work without additional context. Include goals, work performed, results, and explicit next steps
- **Tag consistency:** Use consistent tags across related entries to enable grouped retrieval (e.g., all entries tagged with a specific feature or task name)

</environment_integration>

<metadata_schema>

## Metadata Schema

### Core Fields (Recommended)

| Field          | Type   | Recommended | Description                                                                                                                          |
|----------------|--------|-------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `agent_name`   | string | Yes         | Your agent identifier (defined in your instructions)                                                                                 |
| `task_name`    | string | Yes         | Human-readable task description                                                                                                      |
| `status`       | enum   | Yes         | Completion state: `done` or `pending`                                                                                                |
| `project`      | string | Yes         | Canonical project name (from git remote URL, see Worktree Metadata Fields section)                                                   |
| `technologies` | array  | Yes         | List of technologies involved                                                                                                        |
| `report_type`  | enum   | Yes         | Type of work report                                                                                                                  |
| `references`   | object | Yes         | Cross-system identifiers linking to external resources (use `{}` if none). See [References Field](#references-field) for sub-fields. |

### Standardized Values

#### `status` Field

| Value     | Use When                                                             |
|-----------|----------------------------------------------------------------------|
| `done`    | Work complete, no follow-up required                                 |
| `pending` | Work incomplete, requires continuation (e.g., Research Continuation) |

#### `report_type` Field

| Value            | Description                                 |
|------------------|---------------------------------------------|
| `research`       | Research, analysis, implementation planning |
| `implementation` | Code implementation work                    |
| `validation`     | Quality validation, testing results         |
| `documentation`  | Documentation creation/updates              |

#### `technologies` Field

Array of lowercase technology identifiers representing the **SUBJECT MATTER of the task itself**. The following distinction is **CRITICAL**:

| Include (Task Subject)                 | Exclude (Execution Tools)                       |
|----------------------------------------|-------------------------------------------------|
| Technologies the task is ABOUT         | Tools used to execute the task                  |
| Languages/frameworks being implemented | Linters, formatters, type checkers              |
| Databases being configured or queried  | Version control operations                      |
| APIs being developed or integrated     | Testing frameworks (unless testing IS the task) |
| Infrastructure being designed          | CI/CD tools (unless CI/CD IS the task)          |

**Examples of Correct Usage:**

| Task Description                        | Correct `technologies`       | Why                                  |
|-----------------------------------------|------------------------------|--------------------------------------|
| Fix bug in FastAPI endpoint             | `["python", "fastapi"]`      | Task is about Python/FastAPI code    |
| Fix markdown typo in README             | `["markdown"]`               | Task is about markdown content       |
| Configure PostgreSQL connection pooling | `["postgresql"]`             | Task is about database configuration |
| Write pytest tests for auth module      | `["python", "pytest"]`       | Testing IS the task subject          |
| Set up GitHub Actions CI pipeline       | `["github-actions"]`         | CI/CD IS the task subject            |
| Research LangGraph checkpointing        | `["langchain", "langgraph"]` | Research topic is LangGraph          |

**INCORRECT:** `["python", "pre-commit", "markdown"]` for a markdown typo fix, or `["python", "pytest", "mypy"]` for a hook logic update -- execution, validation, and development tools (pre-commit, pytest, mypy, git, vscode) are not the task subject.

**Example values (illustrative, not exhaustive):** use any lowercase identifier that describes your task's subject matter -- e.g. languages (`python`, `typescript`, `go`, `rust`), frameworks (`fastapi`, `react`, `django`), databases (`postgresql`, `mongodb`, `redis`), infrastructure (`docker`, `kubernetes`, `terraform`).

### Optional Fields

| Field    | Type | Description                                                                                   |
|----------|------|-----------------------------------------------------------------------------------------------|
| `domain` | enum | Technical domain: `backend`, `frontend`, `fullstack`, `devops`, `data`, `security`, `general` |

### Advanced: Worktree Metadata Fields

This section is relevant only for environments that use git worktrees; skip it otherwise. In a git worktree environment, include these fields for proper context isolation:

| Field                | Type    | Required    | Description                                  |
|----------------------|---------|-------------|----------------------------------------------|
| `worktree_id`        | string  | RECOMMENDED | Current worktree directory name              |
| `worktree_path`      | string  | RECOMMENDED | Absolute path to worktree root               |
| `is_linked_worktree` | boolean | OPTIONAL    | True if linked worktree, false for main      |

**Project Name Derivation:**

The `project` field MUST be derived using this fallback chain to ensure consistency across git worktrees. Different worktrees of the same repository have different directory names, so using the directory name breaks context isolation across worktrees; the remote URL provides true canonical identity across all worktrees and users:

1. **Parse from git remote URL** (PREFERRED) -- Try `origin` first (`git remote get-url origin`) and parse the repository name from the URL (`https://github.com/user/my-project.git` -> `my-project`; `git@github.com:user/my-project.git` -> `my-project`). If `origin` is unavailable, try `upstream`, then the first available remote
2. **Git toplevel basename** (FALLBACK for repos without remotes) -- `git rev-parse --show-toplevel` -> extract the last path component (e.g., `/home/user/projects/my-project` -> `my-project`)
3. **Current directory basename** (FALLBACK for non-git directories) -- Extract the last directory name from the working directory path (e.g., `/home/user/work/my-project` -> `my-project`)

**Example with worktree fields:**

```json
{
  "agent_name": "developer",
  "task_name": "Implement feature X",
  "status": "done",
  "project": "my-project",
  "worktree_id": "feature-branch",
  "worktree_path": "/home/user/projects/feature-branch",
  "is_linked_worktree": true,
  "technologies": ["python", "fastapi"],
  "report_type": "implementation",
  "references": {}
}
```

### References Field

The `references` field stores cross-system identifiers for traceability. All values are arrays. Include relevant references when storing context; use an empty object `{}` if no external references exist.

**Core Reference Types:**

| Key           | Value Type | Format                     | Example                                        |
|---------------|------------|----------------------------|------------------------------------------------|
| `context_ids` | integer[]  | Numeric                    | `[2322, 2325]`                                 |
| `urls`        | string[]   | Full URL                   | `["https://github.com/org/repo/pull/445"]`     |
| `git_commits` | string[]   | Full SHA (40/64 hex chars) | `["abc1234def5678901234567890abcdef12345678"]` |

Open for extension following the `{system}_{entity_type}s` convention (e.g., `github_prs`, `jira_issues`).

- `context_ids`: Reference related entries in the same or other threads
- `urls`: Store any external reference as a full URL (issues, PRs, documentation pages, commit URLs, etc.)
- `git_commits`: Use FULL SHA only (40 characters for SHA-1, 64 characters for SHA-256). Within a single-project session where `project` metadata identifies the repository, bare SHAs are unambiguous and directly usable with `git show`

**Cross-Repository Disambiguation:** When context spans multiple repositories (e.g., changes across a backend and frontend repo), bare SHAs in `git_commits` are ambiguous because the hash provides no indication of which repository it belongs to. In such cases, supplement `git_commits` with platform URLs in `urls`. The two fields are complementary: `git_commits` provides typed, validated commit identifiers usable across any git platform (including local repos without hosting); `urls` provides human-readable, clickable links with full repository context:

```json
"references": {
  "git_commits": ["abc1234def5678901234567890abcdef12345678"],
  "urls": ["https://github.com/org/backend-repo/commit/abc1234def5678901234567890abcdef12345678"]
}
```

</metadata_schema>

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
   - `metadata`: **Recommended - include these fields for best discoverability.** Each enables metadata filtering so other agents and sessions can find your context: by agent (`agent_name`), task (`task_name`), completion state (`status`), project (`project`), tech stack (`technologies`, via the `array_contains` operator or tags), and work type (`report_type`):
     ```json
     {
       "agent_name": "[your agent name]",
       "task_name": "[human-readable task description, e.g., 'Implement authentication', 'Fix login bug']",
       "status": "done | pending",
       "project": "[current directory name]",
       "technologies": ["list", "of", "technologies"],
       "report_type": "research | implementation | validation | documentation",
       "references": {}
     }
     ```
   - `tags`: **Recommended** - `["report", ...relevant tags]` for search and categorization

4. **After successfully saving**, capture the `context_id` from the `store_context` response and include it in your brief completion status to the calling party -- format: `"[Brief status summary]. Report ID: [context_id]"` (e.g., `"Implementation complete. 3 features implemented. Report ID: 2510"`). The caller can use this ID to retrieve the full report via `get_context_by_ids([context_id])`

This ensures your work is documented, preserved, and **retrievable by other agents** who need your detailed findings. A structured-output return value or any other in-window reply to your caller is SEPARATE from this durable record and does NOT substitute for it; the ephemeral reply is lost on compaction, the stored entry is not. A dispatch instruction that forbids writing report files to disk (for example a swarm or deep-research "do not write files to disk" contract) governs on-disk files only and does NOT relieve you of storing the context-server entry.

</strategy>

<context_continuity>

## Context Continuity Patterns

These patterns help agents preserve state across context window boundaries and long-running tasks. They are the storage-side patterns; the symmetric retrieval-side patterns (search, re-read after compaction, references navigation) belong to the retrieval workflow and follow the same principles applied to retrieval tools.

### Basic Continuity (Default)

Apply these by default when storing context:

- **Status and reference chains:** Always set `status` and populate `references.context_ids` per the multi-agent coordination patterns above
- **Session handoff notes:** Before ending a session, store a summary entry describing work completed, key decisions, unresolved issues, and recommended next steps -- a briefing document for the next session
- **Pre-compaction preservation:** If approaching context window limits during extended work, proactively store current progress to the context server before compaction occurs. Critical details stored externally survive compaction intact

### Advanced: Long-Running Task Continuity (Optional)

For tasks spanning multiple context windows or extended multi-step execution:

- **Checkpoint storage:** At defined milestones, store a checkpoint entry containing a summary of completed steps and remaining work, key decisions and their rationale, active blockers or dependencies, and the list of modified files and their purpose. Set `status: "pending"` and include `references.context_ids` linking to the task plan
- **Progressive summarization:** For tasks generating large volumes of context, periodically store condensed summary entries distilling key findings, decisions, and progress. Reference the original detailed entries via `references.context_ids` and tag summaries consistently (e.g., with task name) for easy retrieval
- **Multi-agent handoff reports:** When another agent will continue your work, store a comprehensive handoff report that the receiving agent can understand without additional context: clear sections (Summary, Work Performed, Results, Next Steps, and others) covering goals, work performed, results, and explicit next steps; all relevant `references.context_ids` so the receiving agent can trace the full work chain; and `report_type` and `agent_name` set accurately for precise filtering

</context_continuity>

<compliance_checklist>

# Compliance Checklist

Before returning to the calling party, verify the following whenever you had store capability and produced substantive work; completing this checklist is mandatory for reliable context preservation:

- [ ] **Report created**: Comprehensive Markdown report documenting your work
- [ ] **Report saved**: Called `store_context` with thread_id, source="agent", text, metadata, and tags
- [ ] **Metadata complete**: Included agent_name, task_name, status, and project fields
- [ ] **Technologies and report_type**: Populated correctly per task subject (not execution tools)
- [ ] **References included**: Populated `references` field with relevant identifiers (use `{}` if none)
- [ ] **Tags included**: Added "report" tag plus relevant categorization tags
- [ ] **Storage verified**: Confirmed `store_context` call succeeded before returning
- [ ] **Report ID returned**: Included `context_id` from `store_context` response in status message

</compliance_checklist>

<examples>

# Behavioral Examples

<example scenario="successful_preservation">
**Input:** Agent completed implementation task successfully
**Correct Approach:** (1) Create Markdown report following skill format; (2) Call `store_context(thread_id="session-id", source="agent", text="## Summary\n...", metadata={"agent_name": "developer", "task_name": "Implement authentication feature", "status": "done", "project": "my-project"}, tags=["report", "implementation"])` and capture returned `context_id`; (3) Verify storage succeeded; (4) Return brief status with Report ID to caller
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
