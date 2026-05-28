# MCP Context Server Integration

## Activation Check

When MCP Context Server tools are available (any `mcp__context-server__*` tool in your tools list), you MUST follow this rule. If no context-server tools are present, this rule is inactive.

## Mandatory Skill Delegation

For ALL context-server operations (retrieval, search, storage, metadata, update/revision, scoped retrieval, references navigation, continuity, pre-compaction patterns), you MUST follow these skills as the authoritative source of truth:

- **Retrieval:** `context-retrieval-protocol` skill -- thread ID acquisition, project name derivation, retrieval sequences, hybrid/semantic/FTS search, scoped retrieval (`context_scope`), references navigation, revision context detection, worktree-aware queries, and continuity patterns.
- **Preservation:** `context-preservation-protocol` skill -- storage patterns, metadata schema (including the "task subject vs execution tools" distinction for the `technologies` field), `store_context` vs `update_context` strategy, handoff reports, and continuity patterns.

If these skills are already loaded in your context (via your agent frontmatter `skills:` field or a slash-command invocation), treat them as active. If not, invoke them explicitly via the Skill tool before performing any context-server operation. **Rule vs skill precedence:** if this rule appears to contradict a skill, the SKILL WINS.

## Environment-Specific Facts

- **Thread ID is already in your context.** The SessionStart hook injects thread context into the orchestrator system prompt, and subagents receive it via the orchestrator task description (enforced by a PreToolUse blocker). The filesystem fallback is a last resort -- do not perform unnecessary filesystem I/O when the thread ID is already available.
- **User messages are stored automatically** by the UserPromptSubmit hook. Do not manually store user messages. The hook emits the stored `context_id` via `hookSpecificOutput.additionalContext`; use it as the reference pointer in the Relay Protocol below.

## User Message Authority

User messages are the authoritative source of truth and override orchestrator summaries, agent reports, and your own memory when conflicts arise. User messages are IMMUTABLE -- never update, rewrite, or delete them, even when they contain errors. For discrepancy-handling details, see the retrieval skill's orchestrator-verification section.

## User Message Relay Protocol (ID-First)

When launching subagents (via the Task or Agent tool) whose work depends on the user request, you MUST relay the user message using one of two modes, chosen by a deterministic predicate on `context_id` availability. Message SIZE is IRRELEVANT to mode selection.

- **Default Mode -- REFERENCE (context_id only):** This is the ALWAYS-PREFERRED mode regardless of message size. Use it whenever the UserPromptSubmit hook has emitted a `context_id` for the current user message. The Reference Block contains EXACTLY this one line and nothing else (no retrieval instructions, no CRITICAL reminders, no size annotations, no format descriptors):

  ```text
  USER REQUEST CONTEXT ID: [context_id from hook]
  ```

- **Fallback Mode -- INLINE (verbatim text):** Use ONLY when the `context_id` is unavailable -- the UserPromptSubmit hook did not emit one (hook failure or upstream error), or the context-server is unreachable at message-store time. Include the full verbatim text under a `USER REQUEST:` marker:

  ```text
  USER REQUEST: [verbatim user message text]
  ```

**Prohibitions (both modes):** MUST NOT summarize, paraphrase, condense, compress, or select "relevant" portions; MUST NOT extract quotes, evidence, or excerpts; MUST NOT describe intent or problem in your own words; MUST NOT add domain, technology, or problem qualifiers. Relay the complete message (REFERENCE) or its complete verbatim text (INLINE). When a subagent receives a Reference Block, it resolves the pointer per the retrieval skill's Pattern 6 (User Request Resolution) before starting work.

**Image Path Relay (extension to BOTH modes):** Images attached via the terminal are NOT stored in the context-server; subagents can only analyze them by reading the absolute paths directly with the `Read` tool. Whenever the task involves visual analysis, the orchestrator MUST forward image paths verbatim: in Fallback (INLINE) mode the paths flow naturally with the verbatim text; in Default (REFERENCE) mode the orchestrator MUST append an explicit `IMAGE PATHS` block after the Reference Block, because image paths cannot be retrieved from the context-server. The orchestrator MUST NOT summarize, describe, interpret, or redact image contents or paths.

## Subagent Context Requirements

When launching subagents via the Task or Agent tool, the task description MUST include:

1. **Thread ID** -- enforced by a PreToolUse blocker
2. **Timezone / date context** -- enforced by a PreToolUse blocker
3. **User original request** -- per the Relay Protocol above (Default REFERENCE pointer, or INLINE fallback when no `context_id`)
4. **Relevant context IDs** -- so the subagent can retrieve prior work via `get_context_by_ids`

Items 1 and 2 are hook-enforced: if missing, the tool call is blocked with guidance via stderr -- revise the task description and retry. Items 3 and 4 are your responsibility.

## Pre-Compaction Preservation

Before context compaction, preserve (priority-ordered):

- **User message context IDs (highest priority)** -- path back to the authoritative original requirements
- **Agent report context IDs** -- references to plans, research, implementation, and validation reports
- **Current task state** -- what is active and what remains
- **User decisions** -- explicit choices made during this session

Do NOT attempt to preserve full content already stored in the context server -- use context IDs for retrieval after compaction. For detailed continuity patterns, follow the continuity sections of both skills.
