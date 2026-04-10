# MCP Context Server Integration

This rule enables automatic context server usage for persistent memory across Claude Code sessions.

## Activation Check

When MCP Context Server tools are available (any `mcp__context-server__*` tool in your tools list), you MUST follow the session lifecycle instructions below. If no context-server tools are found, proceed normally without context server integration.

## Session Start

At the beginning of every session:

1. **Obtain thread ID** -- Check `.context_server/.thread_id` in the project working directory. If the file does not exist, derive the thread ID from the project's canonical name (parse from `git remote get-url origin`, falling back to `git rev-parse --show-toplevel` basename, then current directory basename).
2. **Obtain project name** -- Parse the canonical project name from the git remote URL (same fallback chain as thread ID). Use this as the `project` field in all metadata.
3. **Retrieve recent context** -- Search for recent user and agent entries to restore session awareness:

   ```text
   search_context(thread_id="<thread-id>", source="user", limit=10)
   search_context(thread_id="<thread-id>", source="agent", limit=10)
   ```

4. **Retrieve full content** of relevant entries identified in step 3:

   ```text
   get_context_by_ids(context_ids=[...relevant IDs...])
   ```

For detailed retrieval patterns, follow the `context-retrieval-protocol` skill instructions.

## During the Session

### Storing Work Results

Before stopping or when completing significant work, store a comprehensive report:

```text
store_context(
  thread_id="<thread-id>",
  source="agent",
  text="<markdown report>",
  metadata={
    "agent_name": "<your role>",
    "task_name": "<task description>",
    "status": "done",
    "project": "<project-name>",
    "technologies": ["<relevant technologies>"],
    "report_type": "<research|implementation|validation|documentation>",
    "references": {}
  },
  tags=["report", "<relevant tags>"]
)
```

For detailed preservation patterns, follow the `context-preservation-protocol` skill instructions.

### Saving User Messages

User messages are automatically saved by the `user_prompt_context_saver` hook. You do not need to manually save user messages.

### Searching Context

Use `hybrid_search_context` for conceptual discovery when you need to find related context:

```text
hybrid_search_context(query="<search terms>", thread_id="<thread-id>", limit=10)
```

## Pre-Compaction Preservation

Before context compaction occurs, preserve critical state:

- **Context IDs** -- Maintain references to all stored entries relevant to current work
- **Current task state** -- What you are working on and what remains
- **User decisions** -- Choices the user made during this session

Do NOT attempt to preserve full content that is already stored in the context server. Use context IDs for retrieval after compaction.

## Subagent Context

When launching subagents (via the Task or Agent tool), you MUST include the current thread ID and timezone/date context directly in the task description so the subagent inherits them.

The `task_thread_id_context` and `task_timezone_date_context` hooks enforce this requirement: they inspect the Task/Agent tool input before the call is executed and block the tool call (returning guidance via stderr) if the required context is missing. When the guidance is returned, revise the task description to include the missing context and retry the tool call.

Additionally, include any relevant context IDs in the task description so the subagent can retrieve the necessary entries from the context server via `get_context_by_ids`.
