# Knowledge Base

The server stores session context for agents, and it also hosts document corpora such as a handbook, a set of runbooks or a design archive. Agents read a corpus with the same search, navigation and retrieval tools they use for session context. The `mcp-context-ingest` command uploads a corpus from a directory and keeps the stored entries in sync with the files.

## Corpus Conventions

- **One thread per corpus.** A corpus lives in the thread `kb:<corpus>`, where `<corpus>` is a short, stable, lowercase name such as `kb:handbook`. Session threads keep their own ids. The corpus, its thread and the `scope` in each managed entry's ingest block are one and the same value.
- **Managed and unmanaged entries.** An entry the sync writes carries the ingest block described below. A corpus thread may also hold entries without a block, stored by any other client; the sync never plans, updates or deletes them.
- **Source.** Corpus entries use `source='user'`. The value records provenance and changes nothing about reading or syncing.
- **Owner.** Every entry belongs to the principal the client authenticates as: `ACCESS_CONTROL_DEFAULT_PRINCIPAL` over stdio and with the `none` and `simple_token` providers, and the token subject under `jwt`. For a corpus several people maintain, sync as a dedicated service principal.
- **One owner per corpus.** A sync manages only the entries its own principal owns. When a second principal syncs the same thread, it builds its own copy of the corpus, and each principal's plans count the other's readable entries as foreign entries.

## Visibility

## The Ingest Metadata Block

Every entry the sync manages carries the reserved top-level metadata key `ingest`:

```json
{"ingest": {"scope": "kb:handbook", "source_id": "guides/setup.md", "meta_digest": "<64 lowercase hexadecimal characters>", "synced_at": "2026-10-18T12:00:00Z"}}
```

| Field         | Value                                         | Meaning                                                                     |
|---------------|-----------------------------------------------|-----------------------------------------------------------------------------|
| `scope`       | a string of 1 to 256 characters               | The corpus; a member's `scope` equals its `thread_id`                       |
| `source_id`   | a string of 1 to 256 characters               | The source document inside the corpus                                       |
| `meta_digest` | exactly 64 characters from `0-9a-f`           | The client's digest of everything it controls except the text               |
| `synced_at`   | `YYYY-MM-DDTHH:MM:SS[.ffffff]Z`, ASCII digits | When a sync last wrote the entry, in UTC; it must name a real date and time |

Lengths count Unicode code points, so a character outside the Basic Multilingual Plane counts once. The 256-character bound on `source_id` keeps the corpus index key within PostgreSQL's index size limit.

**Membership.** An entry belongs to the corpus in thread `T` exactly when its block is valid and its `scope` equals `T`. The server decides this each time it reads the block, so a block whose `scope` names another thread makes its entry a member of no corpus.

**Document identity.** `mcp-context-ingest` writes as `source_id` the file's path relative to the corpus root, with `/` separators, in Unicode NFC, case preserved. The server treats `source_id` as an opaque string and compares it code point by code point, with no case folding and no Unicode normalization, on SQLite and PostgreSQL alike; renaming a file only in case therefore deletes one entry and creates another. To read the entry of one document, filter the corpus thread with `metadata={"ingest.source_id": "guides/setup.md"}`, which matches exactly and case-sensitively.

**Digests.** The server stores a text without its leading and trailing whitespace (Python `str.strip()`) and records the SHA-256 hex digest of the stored text's UTF-8 bytes as the entry's content hash. A client computes the same digest over the text it would send, after the same strip, and the sync plan compares the two, so the block carries no text digest. `mcp-context-ingest` computes `meta_digest` as the SHA-256 hex digest of the canonical JSON of `{"metadata": <the entry metadata without ingest>, "tags": <the tags normalized as the server stores them (trimmed, lower-cased, blanks and duplicates dropped), sorted>, "visibility": <the visibility it sends>}`. Canonical JSON sorts object keys at every depth, uses no whitespace between tokens and keeps non-ASCII characters as UTF-8. A change to a file's metadata, to the tags of a run or to the visibility therefore makes the next sync update the entry.

**Validation on every write.** `store_context`, `store_context_batch`, `update_context` and `update_context_batch` check the block before any embedding or summary generation and refuse a malformed one with a message naming the field: a single-entry tool fails the call, a batch fails that entry, and an atomic batch aborts naming the entry. A block holds exactly the four fields. `metadata_patch` accepts `"ingest": null`, which removes the block, or a complete valid block, which replaces it, and refuses a partial block. A full `metadata` replacement keeps plain replacement semantics: a replacement without `ingest` removes the block, the entry stops being managed and no sync deletes it, and the next sync stores the document again as a new entry. `mcp-context-ingest` always writes the complete block.

**Access.** The block never decides who may read or write an entry; ownership, visibility and grants do. A client that writes a wrong digest affects only its own plans, because a sync plans only the entries its principal owns. A principal holding a write grant on an entry can patch the entry's block; that changes no one's access and at most makes the owner's next sync update or store the document again.

**Edits made outside the sync.** The server compares `meta_digest` values and never recomputes them, so a sync cannot see metadata another client changed on a managed entry. Such an edit persists until the source file's metadata or text changes; the update that follows replaces the metadata with the file's.

## Sync Planning

## Listing Ingest State

## Storing Without Deduplication

## The mcp-context-ingest CLI

## Concurrency and Recovery

## Limits

## Reading a Corpus

## Access Scope

## Backend Parity

## Configuration

## Related Documentation
