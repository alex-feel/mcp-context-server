---
name: issue-tracking-protocol
description: |
  Cross-project task tracker hosted on the context server. Use whenever you need to file an issue, bug report, feature request, improvement, or follow-up task for ANY project (your own or another); triage, prioritize, assign, or transition an issue through its lifecycle; pick up an issue and implement it; comment on an issue; mark duplicates or blockers; or query the tracker (open work for a project, urgent issues across projects, sub-issues, blocked work, triage backlog). Trigger on phrases like "file an issue", "create a task", "track this", "заведи задачу", "add it to the backlog", "start on issue N", or any request to record work for later or to act on work already recorded -- even when no tracker is named explicitly.
---

<overview>

# Context-Server Issue Tracker

ONE unified tracker thread named `issues` holds every issue of every project -- one workspace-wide system of record, the Linear model. The project dimension lives in metadata (`project`), never in the thread name: any agent files an issue for any project into the same thread and filters by project when querying. One issue = one context entry; the entry's context ID IS the issue number, globally unique across all projects by construction.

Why this shape: `project` metadata already encodes the project dimension, so cross-project queries are one `thread_id` plus metadata filters and agents need zero thread-name derivation; context IDs are global, so linking works identically within and across projects; the server's first-class `tags` are the single label mechanism; and one typed `links` object holds every connection an issue has, so no fact is ever stored twice.

</overview>

<schema_directive>

## Mandatory First Step: Load the Schema

Before composing or updating ANY tracker entry, invoke `Skill(skill="context-metadata-schema")`. That skill is the single normative source for the field contract this tracker relies on: the universal core (`schema_version`, `kind`, `project`, `links`), the issue and comment field tables, the status and priority vocabularies, the links registry with its three design rules, the filter recipes (including the compound kind+status MUST and the `case_sensitive` rule for string-ID filters), and the operational rules (read-modify-write link appends, supersession sequence, dedup caution, concurrency retry). The examples below show the contract in action but do not redefine it; on any discrepancy, the schema skill wins.

</schema_directive>

<filing>

## Filing an Issue

Store a new entry in the `issues` thread. The body is Markdown: H1 title first line, then Problem/Goal, Context/Evidence, and Acceptance criteria sections, substance front-loaded because search previews truncate from the start. Give every body concrete specifics early -- identical-text deduplication collapses template-only bodies filed twice.

```text
store_context(
    thread_id="issues",
    source="agent",
    text="# Fix flaky teardown in integration tests\n\n## Problem\n...\n\n## Context\n...\n\n## Acceptance criteria\n...",
    metadata={
        "schema_version": 1,
        "kind": "issue",
        "project": "target-project",
        "source_project": "project-where-discovered",
        "agent_name": "main-agent",
        "title": "Fix flaky teardown in integration tests",
        "status": "triage",
        "priority": 3,
        "links": {"commissioned_by": [27318], "evidence": [27326]}
    },
    tags=["issue", "bug", "tests"]
)
```

Rules that matter at filing time: `status` defaults to `triage` when you file into a project you do not own (the owning project's sessions accept it into `backlog`/`todo`); a project owner may file at any status. `priority` is numeric with `0` meaning NONE, not top. `tags` are `["issue"]` plus labels -- recommended core label vocabulary, lowercase kebab-case: `bug`, `feature`, `improvement`, `tech-debt`, `docs`, `security`, `dx`, `question`; free extension is fine. Report the returned context ID as the issue number.

</filing>

<lifecycle>

## Updating an Issue

Patch only what changes, via RFC 7396 `metadata_patch`:

```text
update_context(context_id=27401, metadata_patch={"status": "in_progress", "assignee": "main-agent"})
update_context(context_id=27401, metadata_patch={"status": "done", "completed_at": "2026-08-01T15:00:00Z"})
```

Set `completed_at` whenever status becomes `done` or `canceled`. Closing as duplicate is a pair: `metadata_patch={"status": "duplicate", "links": {"duplicate_of": [<canonical-id>]}}`. Adding any link edge is read-modify-write: read the issue's current `links`, append client-side, patch the touched key with its complete new array (sibling link keys survive the deep merge). Label changes go through the `tags` parameter, which REPLACES the whole list -- resend every tag including `issue`. Text edits replace the whole body. On a version-conflict error, refetch and reapply once.

</lifecycle>

<commenting>

## Commenting on an Issue

A comment is a NEW entry in the SAME thread -- never append discussion into the issue's own text:

```text
store_context(
    thread_id="issues",
    source="agent",
    text="Re issue 27401: reproduced on Windows only; the teardown races the file-lock release. Evidence in report 27455.",
    metadata={
        "schema_version": 1,
        "kind": "comment",
        "project": "target-project",
        "agent_name": "main-agent",
        "links": {"parent": [27401], "evidence": [27455]}
    },
    tags=["comment"]
)
```

Comments attach to exactly one parent, never to another comment (discussions are flat in schema v1), and always open the body by naming the parent (`Re issue 27401:`) -- that keeps them readable standalone and prevents short identical bodies from dedup-collapsing across different parents. Reading a discussion: filter kind `comment` plus `array_contains` on `links.parent` with the issue ID.

</commenting>

<queries>

## Query Recipes

All queries run against `thread_id="issues"` unless noted; search results are truncated previews for relevance triage -- retrieve full bodies with `get_context_by_ids` before acting on substance.

- **A project's open work:** open means the five non-terminal statuses, so ask for all of them at once -- `search_context(thread_id="issues", metadata={"kind": "issue", "project": "<name>"}, metadata_filters=[{"key": "status", "operator": "in", "value": ["triage", "backlog", "todo", "in_progress", "in_review"]}])`. Narrowing to one status (`metadata={"kind": "issue", "project": "<name>", "status": "todo"}`) answers a different question -- what is already accepted and scheduled -- and is right only when that is the question you have.
- **Cross-project urgency:** kind `issue` plus `priority gt 0 AND priority lt 3`, no project filter.
- **By label:** the `tags` parameter (OR semantics); for a rare label-AND, run the narrower tag query and intersect client-side.
- **By assignee:** `metadata={"kind": "issue", "assignee": "<name>"}`.
- **Provenance:** `metadata={"kind": "issue", "source_project": "<name>"}` finds issues discovered while working in a project, wherever they target.
- **Sub-issues of X vs discussion of X:** same `array_contains` on `links.parent` with value X, disambiguated by `kind` (`issue` = sub-issues, `comment` = discussion). Never run a `links.parent` query without a `kind` filter.
- **What blocks X:** `array_contains` on `links.blocks` with value X (plus kind `issue`); what X blocks is X's own `links.blocks` array.
- **Duplicates of X:** `array_contains` on `links.duplicate_of` with value X.
- On a server issuing string (hex) context IDs, every links filter above adds `"case_sensitive": true`; integer IDs need no flag.

</queries>

<triage>

## Triage Discipline

**The trigger is a query, not a schedule.** Whenever you query your own project's open work and the result holds entries with status `triage`, those entries are the queue, and dispositioning them is part of that query -- do it before choosing what to work on. An entry sitting in `triage` is a request nobody has answered yet, and a queue you have not dispositioned cannot tell you what is worth doing next. A session that never consults the tracker owes nothing here; the obligation attaches to the moment you look.

The queue you triage is your OWN project's. A `triage` entry for any other project -- one you filed there, or one you passed while querying across projects -- belongs to that project's queue, its sessions meet the same trigger there, and reporting its existence from here changes nothing, so leave it alone and say nothing about it.

**Which transitions are yours to make.** The line runs between a disposition you can justify with EVIDENCE and one that rests on your PREFERENCE. Accepting needs no justification, because the filer already made the case: move the entry to `todo` (work the project intends to reach) or `backlog` (accepted, not scheduled) on your own initiative. `duplicate` is a factual finding -- when an existing issue demonstrably covers the same need, patch `{"status": "duplicate", "links": {"duplicate_of": [<canonical-id>]}}` and comment naming the canonical entry, because the need survives there rather than being discarded. `canceled` splits in two: cancel on your own initiative ONLY when the reason is verifiable and you verified it (the defect no longer reproduces, the capability has since shipped), stating that check in the closing comment; when the reason is instead that the work looks not worth doing, the decision is the user's, because the session that filed it saw something you cannot see from here.

**When the decision is not yours,** leave the entry in `triage` and put the proposal to the user through the structured question tool, naming each entry and what you would do with it, batched into one question rather than a stream of them. An entry still in `triage` because nobody has answered is in the right state -- unaccepted, visible to the next query, not lost -- so never cancel by default to empty a queue.

**What triage produces is the transitions.** Report what you changed. Do not additionally warn that unaccepted entries exist, or that a narrower query would have hidden them: you ran the query that does not hide them, and the entries it surfaced are now handled. A warning repeated every session in place of a disposition is the failure this section exists to prevent.

</triage>

<picking_up>

## Picking Up an Issue

**Re-verify what the work will rest on before implementing it.** An issue body is a point-in-time observation written in the present tense, and that tense is exactly what makes a stale reading look like current state: an entry saying that a queue is empty, that an endpoint answers in some shape, or that a file still carries a line reads identically on the day it was filed and a month after it stopped being true. So take the live claims the implementation will stand on -- states read from an API, a queue, or the filesystem, timestamps, and anything else phrased in the present tense -- and check them against the artifact itself before building anything on them. Proportionate means exactly that set: the claims the work RESTS on, never every sentence in the body, and never the background it merely passes.

**Acceptance criteria inherit the errors of the evidence they were derived from.** Criteria are written out of those same observations, so a stale or misread one yields a criterion that still looks satisfiable while prescribing the wrong thing -- and an agent that meets it faithfully ships something the artifact itself refutes, with every check green. A criterion whose ground has moved is therefore NOT satisfied by following it literally. Implement what is true and let the criteria follow the evidence, never the reverse.

**Record the divergence instead of quietly absorbing it.** A comment carries it -- when found, or in the closing one at the latest -- naming what the body claimed, what the artifact actually shows, how that was established, and what was built instead. Leave the body's observation as filed: it records what was seen at that moment, and rewriting it to match the newer reading destroys the only trace that the reading ever changed.

**When re-verification leaves nothing to implement, the entry is closed rather than built.** A defect that no longer reproduces and a capability that has since shipped both clear the evidence bar the triage section above sets for canceling an entry on your own initiative -- verifiable, and verified by you, with that check stated in the closing comment. When instead the premise is gone but the need behind it may survive, the call is the user's: leave the entry open and put it to them through the structured question tool.

</picking_up>

<anti_patterns>

## Anti-Patterns (Forbidden)

- Never store work reports, session context, or knowledge-base notes in the `issues` thread: it holds only kind `issue` entries and their kind `comment` discussion.
- Never create per-project tracker threads -- the project dimension is metadata.
- Never duplicate labels into a metadata field -- tags are the single label mechanism.
- Never store both directions of an edge, and never split connections across a second field -- one typed `links` object holds every connection, and the reverse direction is a query.
- Never mix status vocabularies -- issue statuses never appear on reports, and `pending` never appears on issues.
- Never comment on a comment, and never append discussion into an issue's body.
- Never renumber or reuse issue IDs -- context IDs are immutable.
- Never file secrets, tokens, or credentials into issue bodies or comments.

</anti_patterns>

<walkthrough>

## Lifecycle Walkthrough

1. File: `store_context` into `issues` with kind `issue`, status `triage` (foreign project), priority 3, `links.commissioned_by` pointing at the user message -- returns ID 27401.
2. Accept: a session working in the owning project queries its own open work, finds 27401 sitting in `triage`, and patches `{"status": "todo"}`.
3. Start: re-verify the live claims the work will rest on, then patch `{"status": "in_progress", "assignee": "main-agent"}`.
4. Discuss: a kind `comment` entry with `links.parent: [27401]`, body opening `Re issue 27401:`.
5. Link: discovery that 27401 blocks 27130 -- read 27401's `links`, patch `{"links": {"blocks": [27130]}}`.
6. Close: `{"status": "done", "completed_at": "..."}`, plus a closing comment citing the shipping commit in `links.git_commits` and stating any divergence from the body's evidence.

</walkthrough>
