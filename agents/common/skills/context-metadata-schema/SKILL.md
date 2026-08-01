---
name: context-metadata-schema
description: |
  Normative metadata schema v1 for all context-server records. Defines the kind registry (user_message, report, plan, handoff, checkpoint, note, issue, comment), required and optional metadata fields per kind, status vocabularies, the typed links object with its design rules, and the filter recipes that make entries reliably queryable. Use whenever composing or updating context-server entry metadata, choosing kind or status values, linking entries to each other, or writing metadata / metadata_filters queries -- even when another skill already guides the surrounding workflow, this skill is the single source of truth for field names, value vocabularies, and link semantics.
---

<overview>

# Context-Server Metadata Schema v1

This skill is the single normative definition of the metadata contract for every entry stored on the context server. Workflow skills teach WHEN and HOW to store, retrieve, or track work; this skill defines WHAT the metadata of any entry must look like so that every record is discoverable by filters, navigable by typed links, and ready for future server-side schema enforcement. When any guidance conflicts with this skill on field names, value vocabularies, or link semantics, this skill wins.

The schema is versioned by the `schema_version` field. This document defines version 1. Evolution is additive-only (see the Evolution section), so version 1 is expected to remain current for a long time.

</overview>

<universal_core>

## Universal Core (Every Entry, Every Kind)

Every new entry stored on the context server carries these metadata fields:

| Field            | Type   | Requirement | Meaning                                                                          |
|------------------|--------|-------------|----------------------------------------------------------------------------------|
| `schema_version` | int    | REQUIRED    | Literal `1` (JSON number, not a string).                                         |
| `kind`           | string | REQUIRED    | The record-kind discriminator. See the Kind Registry below.                      |
| `project`        | string | REQUIRED    | Canonical project name (derivation chain below).                                 |
| `links`          | object | OPTIONAL    | Typed connections to other entries and external systems. Omit or `{}` when none. |

### Kind Registry v1

| Kind           | What it is                                                                                                                          |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `user_message` | A user prompt captured verbatim (normally written by an automatic hook). IMMUTABLE: never update, rewrite, or delete these entries. |
| `report`       | A completed-work report: research, implementation, validation, or documentation results.                                            |
| `plan`         | An executable plan or work-state artifact that a session is following or will follow.                                               |
| `handoff`      | A session-handoff briefing: work completed, decisions, unresolved issues, recommended next steps.                                   |
| `checkpoint`   | A mid-task milestone snapshot: progress, remaining work, blockers, modified files.                                                  |
| `note`         | A durable knowledge-base entry (fact, how-to, convention) not tied to a single task.                                                |
| `issue`        | A task-tracker issue in the unified `issues` thread.                                                                                |
| `comment`      | A comment attached to any non-comment entry via `links.parent`.                                                                     |

The registry is open and additive: new kinds are legitimate, and readers MUST tolerate kind values they do not recognize (treat them as opaque record types, never as errors).

### Legacy Scope Rule

The REQUIRED markers above apply to schema-v1 entries prospectively. Entries stored before this schema existed (the large legacy corpus) carry none of these fields, and at least some carry no metadata at all. Readers MUST tolerate absent `kind` and `schema_version` entirely, and MUST NOT assume an `exists` filter on these fields reaches legacy records -- filtering on `kind` or `schema_version` excludes the pre-v1 corpus by construction.

### Tags Convention

Server-level `tags` (a separate first-class field, not metadata) follow one rule: agent-authored entries include their kind token as a tag (`report`, `plan`, `handoff`, `checkpoint`, `note`, `issue`, `comment`) plus any labels or topics. Hook-written `user_message` entries are exempt and carry no tags (`source='user'` plus metadata `kind` already discriminate them). Tags are the ONLY label mechanism -- never duplicate labels into a metadata field.

### Project Name Derivation

Derive `project` with this fallback chain, which always resolves: (1) parse the repository name from the git remote URL (`origin` first, then `upstream`, then the first available remote; `https://github.com/user/my-project.git` -> `my-project`); (2) if the repository has no remote, use the basename of `git rev-parse --show-toplevel`; (3) outside any git repository, use the current directory basename. The remote URL is preferred because different worktrees of one repository have different directory names, and only the remote gives one canonical identity across all worktrees.

When you need the canonical name of a project you cannot inspect (for example, filing an issue for another project), first try that project's git remote if accessible; otherwise search existing entries for its established `project` value (e.g., `search_context(thread_id="issues", limit=30)` and inspect metadata); only if neither works, ask the user.

</universal_core>

<agent_artifact_core>

## Agent-Artifact Core (Kinds report, plan, handoff, checkpoint, note, issue, comment)

- `agent_name` (string) REQUIRED for agent-authored entries: your agent identifier from your instructions; a main agent without a defined identifier uses the fallback `main-agent`. Human-authored entries (`source='user'`, e.g., a user filing an issue directly) omit `agent_name`.
- `task_name` (string) RECOMMENDED for `report`, `plan`, `handoff`, `checkpoint`: a human-readable task description. Issues carry `title` instead.
- `status` (string) REQUIRED for work artifacts and issues, with KIND-SCOPED vocabularies:

| Kind family                                                | Allowed `status` values                                                                                                                                  |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Work artifacts (`report`, `plan`, `handoff`, `checkpoint`) | `pending` (work continues or plan is being executed), `done` (complete), `superseded` (replaced by a newer entry that links here via `links.supersedes`) |
| Issues (`issue`)                                           | `triage`, `backlog`, `todo`, `in_progress`, `in_review`, `done`, `canceled`, `duplicate`                                                                 |

The two vocabularies never mix: never use `triage`/`todo` on a report, never use `pending` on an issue. Note that `done` legitimately appears in BOTH vocabularies with different lifecycle meanings, which is why the compound-filter rule below is a MUST.

**MUST rule -- never filter `status` without `kind`.** A bare `metadata={"status": "done"}` query silently merges finished reports with closed issues (this exact collision exists in live data). Always pair them: `metadata={"kind": "report", "status": "done"}` or `metadata={"kind": "issue", "status": "done"}`.

Design note on the flat issue enum: terminal states (`canceled`, `duplicate`) are first-class statuses rather than a separate `status_reason` field. This follows the live tracker schema; the cross-vendor lesson that closed enums break strict readers is answered by the tolerant-reader MUST (readers treat unknown status values as opaque, never as errors), not by splitting the field. A `duplicate` status is always accompanied by a `links.duplicate_of` edge naming the canonical issue.

</agent_artifact_core>

<kind_reference>

## Per-Kind Field Reference

### issue

| Field            | Type   | Requirement | Meaning                                                                                                                                                                                               |
|------------------|--------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `title`          | string | REQUIRED    | Short imperative summary, duplicated from the body's H1 so listings need no text parsing.                                                                                                             |
| `status`         | string | REQUIRED    | Issue vocabulary above. Default is `triage` when filing into a project the filer does not own; the owning project's sessions accept it into `backlog`/`todo`. A project owner may file at any status. |
| `priority`       | int    | REQUIRED    | Linear numeric convention: `0` none, `1` urgent, `2` high, `3` medium, `4` low. WARNING: `0` means NONE (unset), not top priority; "urgent or high" is `priority gt 0 AND priority lt 3`.             |
| `source_project` | string | REQUIRED    | Canonical name of the project where the need was DISCOVERED (equals `project` for self-filed issues).                                                                                                 |
| `assignee`       | string | OPTIONAL    | Agent identifier or human name.                                                                                                                                                                       |
| `due_date`       | string | OPTIONAL    | ISO 8601 date.                                                                                                                                                                                        |
| `estimate`       | int    | OPTIONAL    | Points.                                                                                                                                                                                               |
| `milestone`      | string | OPTIONAL    | Milestone name.                                                                                                                                                                                       |
| `completed_at`   | string | OPTIONAL    | ISO 8601 datetime; set when status becomes `done` or `canceled`.                                                                                                                                      |

Issue body (the entry `text`): Markdown, English, H1 title first line, then Problem/Goal, Context/Evidence, and Acceptance criteria sections; front-load the substance because search previews truncate from the start.

### comment

A comment attaches to exactly one NON-comment entry of any kind (issue, report, plan, note, ...). Required shape: `links.parent` = a one-element array holding the target's context ID. Comment-on-comment is FORBIDDEN in schema v1 -- discussions are flat, like a timeline; if threading is ever needed, it will arrive as an additive extension. A comment is stored in the SAME THREAD as its parent (issues thread for issues, the session or knowledge thread for reports and notes). Tags are `["comment"]` with no labels. The body SHOULD open by naming its parent (for example `Re issue 27074:`) -- this keeps the comment readable standalone AND prevents the server's identical-text deduplication from collapsing short bodies like "Done." filed under different parents into one entry.

### report

| Field                                                | Type               | Requirement | Meaning                                                                                                                                             |
|------------------------------------------------------|--------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `report_type`                                        | string             | RECOMMENDED | Core values: `research`, `implementation`, `validation`, `documentation`. Open vocabulary; readers tolerate other values.                           |
| `technologies`                                       | array              | RECOMMENDED | Lowercase identifiers for the task's SUBJECT MATTER (what the task is about), never the execution tools used to do it (linters, test runners, git). |
| `worktree_id`, `worktree_path`, `is_linked_worktree` | string/string/bool | OPTIONAL    | Git-worktree context when working in a worktree environment.                                                                                        |

Synonym mapping for `report_type` (write the core value, read the synonyms): audit, review, verification, verdict, adversarial-review, and review-verdict all mean `validation`; synthesis means `research`; module-completion means `implementation`. Progress or status reports are not a `report_type` -- use kind `checkpoint` (mid-task) or `handoff` (session end) instead.

### plan, handoff, checkpoint, note

These kinds need no fields beyond the universal and agent-artifact cores. A `plan` holds the authoritative work plan or work-state its author is executing (store it before acting, update it as work proceeds, mark `done` when executed or `superseded` when replaced). A `handoff` is a session-end briefing for the next session or agent. A `checkpoint` is a mid-task snapshot for recovery. A `note` is durable knowledge; when a note has no meaningful completion lifecycle, `status` may be omitted.

### user_message

Written automatically by the user-prompt hook with the universal core plus worktree fields. User messages are the authoritative record of user intent: IMMUTABLE, never updated, never deleted, even when they contain errors.

</kind_reference>

<links_registry>

## The links Object

`links` holds EVERY connection an entry has -- workflow edges, provenance, and external pointers -- as a map from typed relation keys to flat arrays.

### ID Typing Rule

In-server targets are context IDs stored EXACTLY as the server returns them: JSON numbers on a server issuing integer IDs, hex strings on a server issuing UUIDv7 IDs. Filters must use the same JSON type as the stored element, because the server performs zero type coercion (a string filter never matches a stored number, and vice versa). Arrays mixing numbers and hex strings are a bounded migration transition state, never a design target.

### Registry v1

| Key                       | Targets                   | Direction semantics (store ONCE, on the active side)                                               |
|---------------------------|---------------------------|----------------------------------------------------------------------------------------------------|
| `parent`                  | context IDs (max one)     | Containment: a sub-issue points at its parent issue; a comment points at the entry it comments on. |
| `blocks`                  | context IDs               | This issue blocks the targets. There is deliberately NO `blocked_by`.                              |
| `duplicate_of`            | context IDs (max one)     | This issue duplicates the canonical target; set together with status `duplicate`.                  |
| `related`                 | context IDs               | Symmetric relation, stored once by whichever side discovers it.                                    |
| `derived_from`            | context IDs               | Entries this work builds upon (the successor of the legacy untyped reference list).                |
| `evidence`                | context IDs               | Entries backing specific claims made in this entry.                                                |
| `commissioned_by`         | context IDs               | The user message(s) that spawned this work.                                                        |
| `supersedes`              | context IDs               | Entries this one replaces; each target's `status` becomes `superseded`.                            |
| `urls`                    | strings (full URLs)       | External web pointers.                                                                             |
| `git_commits`             | strings (full 40/64 SHAs) | Commit identifiers usable with any git platform.                                                   |
| `{system}_{entity_type}s` | strings                   | Open extension for external systems (e.g., `github_prs`, `jira_issues`).                           |

### Three Design Rules

1. **Flat arrays under typed keys, never lists of edge objects.** Dotted-path `metadata_filters` with `array_contains` (`{"key": "links.blocks", "operator": "array_contains", "value": X}`) is the server's native strength; edge objects would break it.
2. **Single-writer direction discipline.** Each edge is stored ONCE on the semantically active side. There are no passive pair keys (`blocked_by`, `subissues`, `duplicates`, `superseded_by` do not exist), because a second stored direction inevitably drifts from the first. The reverse question is a QUERY, not a field (recipes below).
3. **Typed provenance over prose.** The meaning of a pointer is its KEY (`derived_from`, `evidence`, `commissioned_by`), never a guess the reader makes from an untyped ID list.

Legacy note: entries predating this schema carry an untyped `references` object (`references.context_ids` and similar). READ it when navigating old entries; NEVER write it in new entries -- `links` fully replaces it.

</links_registry>

<filter_recipes>

## Filter Recipes

These recipes are the supported query surface of the schema. All of them compose with `thread_id` and each other (filters AND-combine).

- **By kind and status (the compound MUST):** `metadata={"kind": "issue", "status": "todo"}`. Never filter `status` alone.
- **Sub-issues of X vs comments on X** (the `parent` key is shared, so `kind` disambiguates -- always pair them): sub-issues are `metadata={"kind": "issue"}` plus `metadata_filters=[{"key": "links.parent", "operator": "array_contains", "value": X}]`; the discussion is the same `array_contains` with `metadata={"kind": "comment"}`.
- **Who blocks X / what does X block:** what blocks X is `metadata_filters=[{"key": "links.blocks", "operator": "array_contains", "value": X}]` (plus `kind: issue`); what X blocks is X's own `links.blocks` array (read the entry).
- **All entries related to X (two-sided union):** the union of X's own `links.related` array AND the results of `array_contains` on `links.related` with value X. The symmetric relation is stored once, so one query alone is incomplete.
- **Is this plan current (supersession check):** before trusting a plan whose `status` is `pending` or `done`, run `array_contains` on `links.supersedes` with the plan's ID; a hit means a newer entry replaced it (the edge wins over a stale status; see the supersession sequence below).
- **Priority bands (numeric):** urgent-or-high open work is `metadata={"kind": "issue"}` plus `metadata_filters=[{"key": "priority", "operator": "gt", "value": 0}, {"key": "priority", "operator": "lt", "value": 3}]`. Works only because `priority` is stored as a JSON number.
- **By technology:** `metadata_filters=[{"key": "technologies", "operator": "array_contains", "value": "python"}]`, or the server-level `tags` parameter for OR-logic.
- **String-ID links filters -- always pass `case_sensitive: true`.** On PostgreSQL, `array_contains` is index-accelerated for integers, booleans, and case-SENSITIVE strings; the default case-insensitive string match falls back to a full function scan. Context-ID hex strings and other exact tokens are always exact-case, so add `"case_sensitive": true` to every string-valued links filter: `{"key": "links.parent", "operator": "array_contains", "value": "019cbd61...", "case_sensitive": true}`. Integer IDs need no flag. (SQLite backends scan object/array containment regardless; keep such queries narrow.)
- **Enum values are lowercase ASCII** because simple `metadata={...}` equality is ASCII-case-insensitive with no override; write and filter enums in lowercase and exact-match behavior follows on every backend.

</filter_recipes>

<operational_rules>

## Operational Rules

### Appending to a links Array (Read-Modify-Write)

`update_context`'s `metadata_patch` is RFC 7396 JSON Merge Patch: nested OBJECTS deep-merge key-by-key, but ARRAYS are always replaced whole -- there is no append operator. To add one edge: (1) `get_context_by_ids` the entry and read its current `links`; (2) append the new ID client-side to the touched key's array; (3) patch ONLY the touched key with its complete new array: `metadata_patch={"links": {"derived_from": [8944, 9044, 27331]}}`. Sibling keys inside `links` survive the deep merge untouched -- never resend keys you did not change, and never hand-construct an array you did not first read. Patching per key is last-writer-wins for that key, so refetch and reapply if a concurrent writer got there first.

### Supersession Sequence (Non-Atomic Dual Write)

Superseding an entry is two writes on two rows with no transaction: FIRST store the new entry with `links.supersedes` naming the old ID, THEN patch the old entry to `status: "superseded"`. If the second write is lost, the EDGE wins: readers resolve inconsistency in favor of the `links.supersedes` edge (that is why the plan-currency recipe queries the edge rather than trusting status alone), and any session noticing the mismatch repairs the stale status.

### Identical-Text Deduplication Caution

`store_context` collapses an entry whose (thread, source, text) exactly matches an existing entry into an UPDATE of that entry, shallow-overriding metadata keys. Short generic bodies ("Done.", "LGTM") and reused issue templates are collision-prone: a collapsed comment silently reparents instead of creating a new record. Defense: give every stored body distinguishing content -- comments open with their parent reference, issues include their concrete specifics before any boilerplate.

### Concurrent Updates

`update_context` uses compare-and-set versioning and raises a version-conflict error when the entry changed underneath you. On conflict: refetch the entry, reapply your change to the fresh state, and retry once; if it conflicts again, surface the contention instead of looping.

</operational_rules>

<evolution>

## Evolution Rules

Schema v1 evolves additively only: new optional fields, new enum values, new kinds, and new link keys are legitimate additions; removing, renaming, retyping, or repurposing anything that exists is prohibited. Readers MUST tolerate unknown fields, kinds, statuses, and link keys (opaque, never an error). `schema_version` stays `1` under additive evolution; it increments only for an unavoidable breaking change, which the additive-only rule exists to prevent.

</evolution>

<examples>

## Canonical Examples

A stored implementation report:

```json
{
  "schema_version": 1,
  "kind": "report",
  "project": "my-project",
  "agent_name": "developer",
  "task_name": "Implement JWT authentication",
  "status": "done",
  "report_type": "implementation",
  "technologies": ["python", "fastapi"],
  "links": {
    "derived_from": [3348],
    "commissioned_by": [3340],
    "git_commits": ["abc1234def5678901234567890abcdef12345678"]
  }
}
```

An issue filed for another project (note `status: "triage"` and distinct `source_project`):

```json
{
  "schema_version": 1,
  "kind": "issue",
  "project": "mcp-context-server",
  "source_project": "claude-code-artifacts",
  "agent_name": "main-agent",
  "title": "Scope identical-text dedup by kind",
  "status": "triage",
  "priority": 3,
  "links": {
    "commissioned_by": [27318],
    "related": [27074]
  }
}
```

A comment on that issue (same thread as its parent; body opens with `Re issue <id>:`):

```json
{
  "schema_version": 1,
  "kind": "comment",
  "project": "mcp-context-server",
  "agent_name": "main-agent",
  "links": {
    "parent": [27401]
  }
}
```

A plan superseding an earlier plan (the old plan then gets `status: "superseded"`):

```json
{
  "schema_version": 1,
  "kind": "plan",
  "project": "my-project",
  "agent_name": "main-agent",
  "task_name": "Migration plan, revised after review",
  "status": "pending",
  "links": {
    "supersedes": [27210],
    "derived_from": [27198, 27204]
  }
}
```

</examples>
