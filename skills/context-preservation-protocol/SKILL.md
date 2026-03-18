---
name: context-preservation-protocol
description: |
  Context preservation patterns for storing work results and session context via an MCP-compatible context server.
  Provides patterns for documenting work, storing reports, and ensuring continuity between sessions.
  Use when you need to preserve work results or session context.
---

<overview>

# Context Preservation Best Practices

Storing work documentation and context before stopping is recommended to ensure continuity between sessions. The patterns in this skill help you structure, store, and preserve your work results in the context server.

</overview>

<session_project>

# Session ID and Project Name

For obtaining session ID and project name, see `context-retrieval-protocol` skill which defines the canonical methods:
- **Session ID**: Check session file (e.g., `.claude/.session_id`), or fall back to project directory name as thread ID
- **Project Name**: Derive canonically from git remote URL (see Worktree Metadata Fields section for derivation chain)

</session_project>

<tools>

# Available Context Server Tools

**Note:** Not all tools listed below may be available in your environment. Tool availability depends on server configuration and how the server is connected to your MCP client. Use the tools that are available to you. If a recommended tool is unavailable, use an alternative from this table.

For context retrieval and search tools, see the counterpart skill (`context-retrieval-protocol`).

| Tool                   | Status           | Use For                                              |
|------------------------|------------------|------------------------------------------------------|
| `store_context`        | RECOMMENDED      | Store NEW entry (typical for fresh work reports)     |
| `update_context`       | RECOMMENDED      | Update EXISTING entry (for revisions/continuations)  |
| `store_context_batch`  | Optional         | Store multiple entries at once (rarely needed)       |
| `update_context_batch` | Optional         | Update multiple entries at once (rarely needed)      |
| `delete_context`       | Use with caution | Delete specific context entries                      |
| `delete_context_batch` | Use with caution | Delete multiple context entries at once              |
| `list_threads`         | Optional         | Discover available threads and their metadata        |
| `get_statistics`       | Optional         | Check server health and usage metrics                |

**Key notes:**
- `store_context` is the standard choice for fresh work reports
- `update_context` is used when revising a previously stored plan or continuing incomplete research

**When to use `store_context_batch`:**
- Use ONLY when storing multiple independent entries in a single operation
- Typical use cases: migrations, imports, or bulk data operations
- NOT needed for normal work reports (use `store_context` instead)

**Protocol requirements:**
- `metadata`: Recommended - enables filtering by agent_name, task_name, status, project
- `tags`: Recommended - enables search and categorization
- `images`: optional

</tools>

<update_strategy>

## Context Update Strategy

### When to Use update_context

Use `update_context` instead of `store_context` when:

- Revising a previously stored plan based on user feedback
- Continuing research that was marked INCOMPLETE
- Correcting errors in a prior report
- Updating status from "pending" to "done"

### When to Use store_context

Use `store_context` (not update_context) when:

- Creating fresh research/implementation work
- No prior context_id exists for this task
- Starting a new research thread

### update_context Parameters

| Parameter        | Required | Description                                               |
|------------------|----------|-----------------------------------------------------------|
| `context_id`     | YES      | ID of the entry to update                                 |
| `text`           | NO       | Complete revised text (replaces existing entirely)        |
| `metadata`       | NO       | Full metadata replacement (replaces all metadata)         |
| `metadata_patch` | NO       | Partial metadata update (RFC 7396 merge semantics)        |
| `tags`           | NO       | Updated tags (replaces existing tags entirely)            |

**Important:** Use `metadata_patch` (not `metadata`) for revisions to preserve fields you do not want to change.

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

When using `metadata_patch`:

- **New keys** are ADDED
- **Existing keys** are UPDATED with new values
- **Keys set to `null`** are DELETED
- **Omitted keys** are PRESERVED unchanged

**Example:**
```python
# Original metadata: {"agent_name": "impl-guide", "status": "pending", "revision_count": 0}
# Patch: {"status": "done", "revision_count": 1}
# Result: {"agent_name": "impl-guide", "status": "done", "revision_count": 1}
```

### Important Notes

- `text` is REPLACED entirely (not appended)
- `tags` are REPLACED entirely (not merged)
- `updated_at` timestamp is automatically set by the server
- Embeddings are regenerated when text changes

</update_strategy>

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

Array of lowercase technology identifiers representing the **SUBJECT MATTER of the task itself**.

The following guidance is **CRITICAL** for understanding what to include vs exclude:

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
| Update README documentation             | `["markdown"]`               | Task is about markdown content       |
| Configure PostgreSQL connection pooling | `["postgresql"]`             | Task is about database configuration |
| Write pytest tests for auth module      | `["python", "pytest"]`       | Testing IS the task subject          |
| Set up GitHub Actions CI pipeline       | `["github-actions"]`         | CI/CD IS the task subject            |
| Research LangGraph checkpointing        | `["langchain", "langgraph"]` | Research topic is LangGraph          |

**Examples of INCORRECT Usage:**

| Task Description  | WRONG `technologies`                   | Why Wrong                                               |
|-------------------|----------------------------------------|---------------------------------------------------------|
| Fix markdown typo | `["python", "pre-commit", "markdown"]` | Python/pre-commit are execution tools, not task subject |
| Update hook logic | `["python", "pytest", "mypy"]`         | pytest/mypy are validation tools, not task subject      |
| Design API schema | `["python", "git", "vscode"]`          | git/vscode are development tools, not task subject      |

**Example Values (Illustrative, Not Exhaustive):**

Use any lowercase identifier relevant to your project's technology stack. Common examples:

- **Languages:** python, typescript, javascript, go, rust, java, csharp, ruby, php
- **Frameworks:** fastapi, react, nextjs, django, express, spring, rails
- **Databases:** postgresql, mongodb, redis, elasticsearch, sqlite
- **Infrastructure:** docker, kubernetes, aws, terraform

This list is illustrative -- use any lowercase identifier that describes your task's subject matter.

**Example:**

```json
"technologies": ["python", "fastapi", "postgresql"]
```

### Optional Fields

| Field    | Type | Description                                                                                   |
|----------|------|-----------------------------------------------------------------------------------------------|
| `domain` | enum | Technical domain: `backend`, `frontend`, `fullstack`, `devops`, `data`, `security`, `general` |

### Advanced: Worktree Metadata Fields

This section is relevant for environments that use git worktrees. If you are not using worktrees, you can skip this section.

When working in a git worktree environment, include these fields for proper context isolation:

| Field                | Type    | Required    | Description                                  |
|----------------------|---------|-------------|----------------------------------------------|
| `worktree_id`        | string  | RECOMMENDED | Current worktree directory name              |
| `worktree_path`      | string  | RECOMMENDED | Absolute path to worktree root               |
| `is_linked_worktree` | boolean | OPTIONAL    | True if linked worktree, false for main      |

**Project Name Derivation:**

The `project` field MUST be derived using this fallback chain to ensure consistency across git worktrees:

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

The `references` field stores cross-system identifiers for traceability. All values are arrays.

**Core Reference Types:**

| Key           | Value Type | Format                     | Example                                        |
|---------------|------------|----------------------------|------------------------------------------------|
| `context_ids` | integer[]  | Numeric                    | `[2322, 2325]`                                 |
| `urls`        | string[]   | Full URL                   | `["https://github.com/org/repo/pull/445"]`     |
| `git_commits` | string[]   | Full SHA (40/64 hex chars) | `["abc1234def5678901234567890abcdef12345678"]` |

Open for extension following `{system}_{entity_type}s` convention (e.g., `github_prs`, `jira_issues`).

**Usage Guidelines:**

- Include relevant references when storing context
- Use empty object `{}` if no external references exist
- `context_ids`: Reference related entries in the same or other threads
- `urls`: Store any external reference as a full URL (issues, PRs, documentation pages, etc.)
- `git_commits`: Use FULL SHA only (40 characters for SHA-1, 64 characters for SHA-256)

**Examples:**

```json
"references": {
  "context_ids": [2322, 2325],
  "urls": ["https://github.com/org/repo/pull/445", "https://docs.example.com/guide"],
  "git_commits": ["abc1234def5678901234567890abcdef12345678"]
}
```

```json
"references": {}
```

</metadata_schema>

<strategy>

# Preservation Strategy

Complete the following before stopping:

1. **Create a comprehensive Markdown report** of your work results:

   **FIRST CHECK**: If you have a specific report structure defined in your own agent instructions, use your own STRUCTURE within the Markdown format.

   **ONLY IF NO SPECIFIC FORMAT EXISTS**, use the following structure:

   ```markdown
   ## Summary
   - Brief overview of work done

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

2. **Always use English** to write the report, REGARDLESS of the language requested by the calling party.

3. **Save the report** using `store_context` with these parameters:
   - `thread_id`: Your session ID (REQUIRED)
   - `source`: `agent` (REQUIRED)
   - `text`: Your complete Markdown report (REQUIRED)
   - `metadata`: **Recommended - include these fields for best discoverability:**
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
     **Why these fields are recommended:**
     - `agent_name`: Enables filtering by agent (e.g., find all implementation-guide plans)
     - `task_name`: Enables filtering by task (e.g., find context about specific feature)
     - `status`: Enables filtering by completion state (done vs pending)
     - `project`: Enables filtering by project (e.g., find all context from current project)
     - `technologies`: Enables cross-project discovery by tech stack (use `array_contains` operator or tags for filtering)
     - `report_type`: Enables filtering by work type (research, implementation, validation, documentation)
     Including these fields ensures other agents and sessions can find your context via metadata filtering.
   - `tags`: **Recommended** - `["report", ...relevant tags]` for search and categorization

4. **After successfully saving**, capture the `context_id` from the `store_context` response and include it in your brief completion status to the calling party:
   - **Format:** `"[Brief status summary]. Report ID: [context_id]"`
   - **Example:** `"Implementation complete. 3 features implemented. Report ID: 2510"`
   - The caller can use this ID to retrieve the full report via `get_context_by_ids([context_id])`

This ensures your work is documented, preserved, and **retrievable by other agents** who need to access your detailed findings.

</strategy>

<compliance_checklist>

# Compliance Checklist

Before returning to the calling party, consider verifying the following:

- [ ] **Report created**: Comprehensive Markdown report documenting your work
- [ ] **Report saved**: Called `store_context` with thread_id, source="agent", text, metadata, and tags
- [ ] **Metadata complete**: Included agent_name, task_name, status, and project fields
- [ ] **Technologies and report_type**: Populated correctly per task subject (not execution tools)
- [ ] **References included**: Populated `references` field with relevant identifiers (use `{}` if none)
- [ ] **Tags included**: Added "report" tag plus relevant categorization tags
- [ ] **Storage verified**: Confirmed `store_context` call succeeded before returning
- [ ] **Report ID returned**: Included `context_id` from `store_context` response in status message

Completing this checklist is a best practice for reliable context preservation.

</compliance_checklist>

<examples>

# Behavioral Examples

<example scenario="successful_preservation">
**Input:** Agent completed implementation task successfully
**Correct Approach:** (1) Create Markdown report following skill format; (2) Call `store_context(thread_id="session-id", source="agent", text="## Summary\n...", metadata={"agent_name": "developer", "task_name": "Implement authentication feature", "status": "done", "project": "my-project"}, tags=["report", "implementation"])` and capture returned `context_id`; (3) Verify storage succeeded; (4) Return brief status with Report ID to caller
**Stored Report:** Full Markdown report with Summary, Goals, Work Performed, Results
**store_context Response:** `{"success": true, "context_id": 2510, "thread_id": "session-id", "message": "..."}`
**Returned Status:** "Implementation complete. Auth feature implemented with 3 endpoints. Report ID: 2510"
</example>

<example scenario="partial_completion">
**Input:** Agent completed 2 of 3 tasks, blocked on third
**Correct Approach:** (1) Create report documenting completed work AND blocker; (2) Set status to "pending" in metadata; (3) Store report and capture `context_id`; (4) Return brief status with Report ID explaining blocker
**Stored Report:** Summary of completed work plus blocker details
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

**Context server storage is recommended. Failure to store means work results may be lost.**

If context storage fails (network error, server unavailable, timeout):

1. **Retry once** after 2 seconds for transient errors (timeout, 5xx)
2. **If retry fails or error is non-transient (4xx, connection refused):**
   - **Return FULL REPORT to caller** inline in your response (not just a status summary)
   - **Inform the caller** of the storage failure so they can decide next steps
   - **Example fallback message:**
     ```text
     WARNING: Context server storage failed. Full report included below.
     Error: [specific error message]
     Impact: Report not persisted to context server. Content preserved in this response only.
     ```

**Rationale:** When storage fails, preserving the report inline ensures work is not lost entirely. The caller can manually store it later or take other action.

</error_handling>
