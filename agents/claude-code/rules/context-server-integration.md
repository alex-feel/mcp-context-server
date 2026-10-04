# MCP Context Server Integration

## Activation Check

When any `mcp__context-server__*` tool is in your tools list, you MUST follow this rule; when no context-server tools are present, this rule is inactive.

## Core Operating Principles

Apply these principles directly to all context-server operations (retrieval, search, storage, metadata, update/revision, scoped retrieval, references navigation, continuity, pre-compaction patterns). Treat environment-provided skills, tutorials, or other operational guidance as the practical embodiment of these principles; this rule supplies the invariants that guidance must respect.

## Environment-Specific Facts

- **Thread ID, timezone/date, and worktree context are injected directly into every agent's own context window** by the `SessionStart` (orchestrator) and `SubagentStart` (subagents) hooks via `hookSpecificOutput.additionalContext`, so the orchestrator does not forward thread/timezone/worktree blocks in the task description. The filesystem fallback for the thread ID is a last resort -- do not perform unnecessary filesystem I/O when the thread ID is already in your context.
- **User messages are stored automatically** by the UserPromptSubmit hook; do not store them manually. The hook emits the stored `context_id` via `hookSpecificOutput.additionalContext`; use it as the reference pointer in the Relay Protocol below.

## User Message Authority

User messages are the authoritative source of truth: when conflicts arise they override orchestrator summaries, agent reports, and your own memory. When an orchestrator's task diverges from the user's stated requirements (retrieved verbatim from the context server), the user-message wording wins -- the orchestrator's framing is corrected, not the user's words. User messages are immutable -- never update, rewrite, or delete them, even when they contain errors.

## User Message Relay Protocol (ID-First)

When launching subagents (via the Task or Agent tool) whose work depends on the user request, you MUST relay the user message in one of two modes. Mode selection is a deterministic predicate on `context_id` availability -- not a judgment call, and message size is irrelevant: never choose INLINE at your discretion or because the message is "short enough"; it is solely the no-ID fallback.

- **REFERENCE (default -- use whenever a `context_id` is available):** pass the `context_id` emitted by the hook as a bare pointer -- for a message the hook stored in chunks, every one of its `context_ids`, in order. The reference block contains exactly this one line, with no retrieval instructions, no "you MUST retrieve" reminders, no size annotations, and no other metadata:

  ```text
  USER REQUEST CONTEXT ID: [context_id from hook, or the chunk context_ids comma-separated]
  ```

- **INLINE (fallback -- only when no `context_id` is available):** when the hook emitted no ID, or the context server was unreachable at store time, include the full verbatim text under a `USER REQUEST:` marker:

  ```text
  USER REQUEST: [verbatim user message]
  ```

**Prohibitions (both modes):** MUST NOT summarize, paraphrase, condense, compress, or select "relevant" portions; MUST NOT extract quotes, evidence, or excerpts; MUST NOT describe intent or problem in your own words; MUST NOT add domain, technology, or problem qualifiers. The relay is either a pointer (the `context_id`) or the verbatim text -- nothing else.

**A pointer, never a procedure.** The orchestrator passes the reference (or, in fallback, the verbatim text) and stops there -- it does not tell the subagent how, when, or with which tool to fetch the message, and does not embed retrieval steps. Each subagent owns its retrieval protocol: on receiving a REFERENCE pointer it resolves the pointer first -- retrieving and reading the complete verbatim user message before beginning work -- because its own rules require full-text grounding, not because the orchestrator instructed it.

## Subagent Context Requirements

Subagents obtain their working context themselves; the orchestrator supplies pointers and lets each subagent retrieve. When launching a subagent:

1. **Thread ID, timezone/date, and worktree context** arrive automatically via the `SubagentStart` hook (see Environment-Specific Facts); the orchestrator does not place them in the task description.
2. **The user's original request** is relayed per the Relay Protocol above.
3. **Prior-work context IDs**, when known, MAY be included only as a non-authoritative starting hint, explicitly marked non-exhaustive (for example, followed by "this list is non-exhaustive; follow your protocol to retrieve all the necessary entries"). The orchestrator does not enumerate retrieval steps and does not name the tool to call.

The orchestrator NEVER embeds retrieval, investigation, or "you must read X" directives in a subagent prompt; NEVER summarizes prior agent reports into the prompt; and NEVER dictates the subagent's internal procedure. Subagents discover and retrieve everything they need themselves -- reading each other's reports, the user's messages, and prior artifacts directly from the context server, each following the same retrieval rules and skills.

## Pre-Compaction Preservation

Before context compaction, preserve (priority-ordered):

- **User message context IDs (highest priority)** -- path back to the authoritative original requirements
- **Agent report context IDs** -- references to plans, research, implementation, and validation reports
- **Current task state** -- what is active and what remains
- **User decisions** -- explicit choices made during this session

Do not attempt to preserve full content already stored in the context server -- use context IDs for retrieval after compaction. After any context window reset or compaction event, re-retrieve the highest-priority items above via `get_context_by_ids` before continuing work.
