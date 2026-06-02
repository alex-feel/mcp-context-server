# MCP Context Server Integration

## Activation Check

When MCP Context Server tools are available (any `mcp__context-server__*` tool in your tools list), you MUST follow this rule. If no context-server tools are present, this rule is inactive.

## Core Operating Principles

For ALL context-server operations (retrieval, search, storage, metadata, update/revision, scoped retrieval, references navigation, continuity, pre-compaction patterns), apply these principles directly. Where your environment provides skills, tutorials, or other operational guidance, treat that guidance as the practical embodiment of these principles; this rule supplies the invariants those guides must respect.

## Environment-Specific Facts

- **Thread ID, timezone/date, and worktree context are injected directly into your own context window** by `SessionStart` (orchestrator) and `SubagentStart` (subagents) hooks via `hookSpecificOutput.additionalContext`. Every agent -- orchestrator and subagent alike -- receives this context for itself; the orchestrator does NOT forward thread/timezone/worktree blocks in the task description, and there is no PreToolUse blocker requiring it to. The filesystem fallback for the thread ID is a last resort -- do not perform unnecessary filesystem I/O when the thread ID is already in your context.
- **User messages are stored automatically** by the UserPromptSubmit hook. Do not manually store user messages. The hook emits the stored `context_id` via `hookSpecificOutput.additionalContext`; use it as the reference pointer in the Relay Protocol below.

## User Message Authority

User messages are the authoritative source of truth and override orchestrator summaries, agent reports, and your own memory when conflicts arise. User messages are IMMUTABLE -- never update, rewrite, or delete them, even when they contain errors. When you detect a discrepancy between an orchestrator's task and the user's stated requirements (retrieved verbatim from the context server), the user-message wording wins; the orchestrator's framing is corrected, not the user's words.

## User Message Relay Protocol (ID-First)

When launching subagents (via the Task or Agent tool) whose work depends on the user request, you MUST relay the user message in one of two modes. Mode selection is a deterministic predicate on `context_id` availability -- NOT a judgment call, and message SIZE is IRRELEVANT:

- **REFERENCE (default -- use whenever a `context_id` is available):** pass the `context_id` emitted by the hook as a bare pointer. The reference block contains EXACTLY this one line, with NO retrieval instructions, NO "you MUST retrieve" reminders, NO size annotations, and NO other metadata:

  ```text
  USER REQUEST CONTEXT ID: [context_id from hook]
  ```

- **INLINE (fallback -- ONLY when no `context_id` is available):** when, and only when, no `context_id` exists (the hook emitted none, or the context server was unreachable at store time), include the full verbatim text under a `USER REQUEST:` marker:

  ```text
  USER REQUEST: [verbatim user message]
  ```

REFERENCE is the canonical mode and is preferred in EVERY case where an ID exists. INLINE is NEVER chosen at the agent's discretion, and NEVER because the message is "short enough" -- it is used solely as the no-ID fallback.

**Prohibitions (both modes):** MUST NOT summarize, paraphrase, condense, compress, or select "relevant" portions; MUST NOT extract quotes, evidence, or excerpts; MUST NOT describe intent or problem in your own words; MUST NOT add domain, technology, or problem qualifiers. The relay is either a POINTER (the `context_id`) or the VERBATIM TEXT -- nothing else.

**A pointer, never a procedure.** The orchestrator passes the reference (or, in fallback, the verbatim text) and stops there. It does NOT tell the subagent how, when, or with which tool to fetch the message, and it does NOT embed retrieval steps. Each subagent owns its retrieval protocol: on receiving a REFERENCE pointer it resolves the pointer FIRST -- retrieving and reading the COMPLETE verbatim user message before beginning work -- because its own rules require full-text grounding, not because the orchestrator instructed it. Acting on a reference pointer without retrieving the underlying message is a PROTOCOL VIOLATION.

## Subagent Context Requirements

Subagents obtain their working context themselves; the orchestrator supplies pointers and lets each subagent retrieve. When launching a subagent:

1. **Thread ID, timezone/date, and worktree context** arrive automatically via the `SubagentStart` hook (see Environment-Specific Facts) -- the orchestrator does NOT place them in the task description.
2. **The user's original request** is relayed per the Relay Protocol above (a REFERENCE pointer, or INLINE only when no ID exists).
3. **Prior-work context IDs**, when known, MAY be included ONLY as a non-authoritative starting hint, explicitly marked non-exhaustive (for example, followed by "this list is non-exhaustive; follow your protocol to retrieve all the necessary entries"). The orchestrator does NOT enumerate retrieval steps and does NOT name the tool to call -- the subagent discovers and retrieves everything it needs by following its own retrieval rules and skills.

The orchestrator NEVER embeds retrieval, investigation, or "you must read X" directives in a subagent prompt; NEVER summarizes prior agent reports into the prompt; and NEVER dictates the subagent's internal procedure. Subagents read each other's reports, the user's messages, and prior artifacts directly from the context server, each following the same rules and skills.

## Pre-Compaction Preservation

Before context compaction, preserve (priority-ordered):

- **User message context IDs (highest priority)** -- path back to the authoritative original requirements
- **Agent report context IDs** -- references to plans, research, implementation, and validation reports
- **Current task state** -- what is active and what remains
- **User decisions** -- explicit choices made during this session

Do NOT attempt to preserve full content already stored in the context server -- use context IDs for retrieval after compaction. After any context window reset or compaction event, re-retrieve the highest-priority items above via `get_context_by_ids` before continuing work.
