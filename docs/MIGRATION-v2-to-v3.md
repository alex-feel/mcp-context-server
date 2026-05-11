# Migrating to the UUIDv7 Schema

This guide applies to the v2 -> v3 transition. Future major-version migrations will be documented in separate guides.

The context-entry primary key is a UUIDv7 value. On SQLite the `context_entries.id` column is `TEXT NOT NULL UNIQUE`; on PostgreSQL it is a native `UUID NOT NULL PRIMARY KEY`. Public IDs are exchanged at MCP tool boundaries as 32-character lowercase hex strings (no hyphens).

If you previously ran an installation whose `context_entries.id` column was an INTEGER (SQLite) or `BIGSERIAL` (PostgreSQL) and you want to keep your existing data, you must run the `mcp-context-server-migrate` CLI on a backup of your source database before pointing the new server at it. This file walks through that process.

## When You Need to Migrate

You need to migrate if all of the following apply:

- You ran a previous installation of this server and have a SQLite database file or a PostgreSQL instance with stored context entries.
- The `context_entries.id` column in that database is INTEGER (SQLite) or BIGINT / BIGSERIAL (PostgreSQL).
- You want to keep your existing data when upgrading to the UUIDv7-keyed schema.

If you are starting fresh (no existing data) you do NOT need to migrate. Install the new server, point it at an empty database location, and proceed normally.

## Pre-Migration Checklist

Before running the CLI:

- **Stop the server.** The migration tool requires exclusive read access to the source database.
- **Make a backup copy of the source database.** The migration reads the source in read-only mode (SQLite uses the `mode=ro` URI parameter; PostgreSQL uses `BEGIN TRANSACTION READ ONLY`), but a backup is still essential in case of an interrupted run or operator error.
- **Provision an empty target database.** For SQLite, a non-existent file path is fine -- the CLI creates the file and applies the current schema. For PostgreSQL, create an empty database manually (for example, `CREATE DATABASE mcp_context_v3;`); the CLI does not create PostgreSQL databases.
- **For SQLite source databases that use semantic search or FTS**, ensure the `sqlite-vec` extension is available in the Python environment running the CLI. The CLI loads the extension on both source and target connections when vector tables are present.
- **For PostgreSQL targets that should use semantic search after migration**, ensure the `pgvector` extension is available on the target server; the migration copies row data but does not install database extensions.
- **Protect PostgreSQL credentials.** Avoid placing passwords directly on the command line. Prefer environment-variable substitution or a `.pgpass` file. URLs printed by the CLI are masked in stdout (`postgresql://user:***@host/db`), but passwords on the original command line remain visible in process listings and shell history.

## CLI Usage

### Installation

The CLI is installed alongside the server. If you installed `mcp-context-server` via `uv` or `pip`, `mcp-context-server-migrate` is on your PATH (it is registered in `[project.scripts]`). Verify with:

```bash
mcp-context-server-migrate --help
```

### Same-Backend Migration (SQLite to SQLite)

```bash
mcp-context-server-migrate \
  --source-url sqlite:////path/to/source.db \
  --target-url sqlite:////path/to/target.db
```

The bare filesystem path form is also accepted:

```bash
mcp-context-server-migrate \
  --source-url /path/to/source.db \
  --target-url /path/to/target.db
```

Note that `sqlite:///` requires three slashes followed by an absolute path. The CLI also accepts plain filesystem paths without a scheme.

### Same-Backend Migration (PostgreSQL to PostgreSQL)

```bash
mcp-context-server-migrate \
  --source-url postgresql://user:password@host:5432/old_db \
  --target-url postgresql://user:password@host:5432/new_db
```

Passwords in printed URLs are masked in stdout output (`postgresql://user:***@host/db`).

### Cross-Backend Migration

The CLI supports SQLite -> PostgreSQL and PostgreSQL -> SQLite. Vector embeddings (the binary `embedding` BLOBs in SQLite's `vec_context_embeddings` table and the corresponding PostgreSQL `pgvector` values) are NOT portable across backends because their binary representations differ. When cross-backend migration is requested, the CLI emits a warning and DROPS the vector embeddings -- you must re-embed the target database after migration (either by storing entries again or by running the standard server startup path with embedding generation enabled, which fills in missing embeddings for entries that lack them).

### Optional Flags

- `--dry-run` -- run the full migration logic in memory (read source, build the integer-to-UUIDv7 mapping, compute target rows) but issue zero INSERT statements against the target. Useful for validating that the source database is well-formed before committing to a real run.
- `--report PATH` -- after the run completes, write the run's statistics (rows migrated, references rewritten, orphan references detected, warnings, errors) as JSON to `PATH`. If omitted, a human-readable summary is printed to stdout.

## What the CLI Does

The CLI performs the following steps in order:

1. **Read source rows in deterministic order.** Reads from `context_entries` ordered by `(created_at ASC, id ASC)`. The deterministic ordering ensures the in-memory integer-to-UUIDv7 mapping is reproducible across runs.

2. **Generate a UUIDv7 per row from each row's `created_at`.** Uses `uuid_utils.uuid7(timestamp=int(created_at.timestamp()), nanos=created_at.microsecond * 1000)`. The `timestamp` parameter is in UNIX SECONDS with an optional nanosecond fraction; passing milliseconds in the `timestamp` slot would shift the embedded timestamp roughly 1000x further into the future, producing UUIDs whose decoded year is around 50,000 AD. Upstream tracker documenting this confusion source: [`aminalaee/uuid-utils#73`](https://github.com/aminalaee/uuid-utils/issues/73). The resulting UUIDs preserve the source's chronological order: lex-string comparison on the 32-character lowercase hex form is monotonic with respect to `created_at` at millisecond granularity. Sub-millisecond ordering is determined by the UUID random tail, which is acceptable for the server's deduplication interleaving check.

3. **Rewrite `metadata.references.context_ids` arrays.** For each row, walks the parsed `metadata` JSON recursively, finding every `references.context_ids` array. Integer entries are replaced with the mapped UUIDv7 hex string. String entries are preserved unchanged (treated as already-UUID values). Integer entries that do not match any source row are kept as integers and flagged in the run summary as `orphan_references` (see Troubleshooting below). Malformed shapes (for example, a non-array `context_ids` value) are flagged as `malformed_references` errors, and the row's metadata is copied verbatim.

4. **Copy free-form text byte-for-byte.** The `text_content` and `summary` columns are copied unchanged. Any mention of an integer ID inside the prose (for example, "see entry 8944") is left as-is. The CLI does not attempt to rewrite free-form text -- such substrings become inert references after migration, which is the intended outcome (eliminating false positives from regex-style ID detection across the corpus).

5. **Copy tags, image attachments, embedding metadata, embedding chunks, and vector rows.** Only the `context_id` foreign-key column is remapped to the new UUIDv7 hex value. Internal embedding identifiers (`embedding_chunks.id`, `embedding_chunks.vec_rowid`, vec0 rowids, `vec_context_embeddings.id`) are preserved verbatim because the project's chunking-layer architecture uses INTEGER bridges to vec0; only the outer `context_id` foreign key changes type.

6. **Rebuild the SQLite FTS5 index.** When the source has `context_entries_fts`, the CLI issues `INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild')` against the target to re-index the copied rows. The FTS5 rowid alignment with `context_entries.rowid_int` is preserved.

7. **Print or write the run report.** A summary of counters (rows migrated, references rewritten, orphan references, malformed references, tags/images/embeddings copied, FTS rebuild status, warnings, errors) is printed to stdout. If `--report PATH` is set, the same data is also written as JSON.

## Optional: PostgreSQL 18+ Server-Side UUIDv7 Generation

If you are running PostgreSQL 18 or later, you have the OPTION to use the server-side `uuidv7()` built-in for the `id` column default:

```sql
ALTER TABLE context_entries ALTER COLUMN id SET DEFAULT uuidv7();
```

This is a pure operator-side optimization and is NOT required for correct operation. The application generates UUIDs Python-side via the `app.ids` module by default; the column default would only matter if external clients ever insert rows without specifying an `id`. Most installations should leave this column default unset. PostgreSQL 17 and earlier do not have the `uuidv7()` function; users on those versions cannot apply this optimization.

## Troubleshooting

### Orphan References

If the run summary shows `orphan_references > 0`, your source database contained `metadata.references.context_ids` entries pointing to IDs that do not exist in `context_entries` (most commonly from older rows that were deleted before migration). The orphan IDs are preserved as integers in the migrated metadata; downstream consumers that interpret `references.context_ids` should be prepared to encounter both string UUIDs and stray integers in legacy data. To find and clean orphan references, query the migrated database after the run -- the JSON `references.context_ids` arrays are still queryable with `array_contains` operators.

### Source Database File Missing

If the CLI prints `source database file does not exist: <path>`, the path passed to `--source-url` does not exist on disk. For SQLite, ensure the path is correct and absolute. For `sqlite:///` URLs, note that three slashes are required: `sqlite:///C:/Users/me/db.sqlite` on Windows or `sqlite:////home/me/db.sqlite` on Linux.

### Target Database Already Has Data

The CLI refuses to write to a non-empty SQLite target. If you see `target database already contains context_entries rows: <path>`, delete the target file (or specify a different `--target-url`) and rerun.

### Cross-Backend Vector Embedding Warning

If you ran a cross-backend migration (SQLite to PostgreSQL or vice versa) and saw the warning `cross-backend migration drops vector embeddings; re-embed the target after migration`, your target database contains all rows and metadata but no vector embeddings. To restore embeddings:

- Re-store the affected entries (each successful store triggers embedding generation when generation is enabled), OR
- Run `mcp-context-server` against the target with `ENABLE_EMBEDDING_GENERATION=true`. The embedding pipeline fills in missing embeddings for entries that lack them.

### Second-Precision Source Data

If the run summary or log shows that the source database has second-precision `created_at` (no microsecond data), the CLI prints an informational log line. Same-second entries in the source are still uniquely identified (the UUIDv7 random tail differentiates them), but the resulting ordering of same-second entries may differ from the source's `id`-based ordering. This is acceptable per RFC 9562 and the server's deduplication invariants.

### `writable_schema` In-Place ALTER (do NOT use)

Some online guides suggest using SQLite's `writable_schema` pragma to ALTER the `context_entries.id` column type in-place from INTEGER to TEXT. **Do not do this.** It is documented as unsafe by the SQLite project, and it would not migrate the actual integer values to UUIDs anyway -- it would only change the column declaration while leaving every row's `id` as an integer, breaking referential integrity. The export-then-import path implemented by `mcp-context-server-migrate` is the correct approach.

## Post-Migration Verification

Suggested verification steps after a successful migration:

1. Start the server pointed at the target database. Confirm it starts cleanly (no schema-mismatch errors at startup).
2. Run a smoke test query: `list_threads` or a simple `search_context` call. Confirm the response includes 32-character hex `context_id` values.
3. If FTS was enabled on the source, run a small `fts_search_context` query and confirm results are returned.
4. If semantic search was enabled on the source AND the migration was same-backend (no embeddings dropped), confirm a `semantic_search_context` query returns expected results.
5. Spot-check that `metadata.references.context_ids` arrays contain UUIDv7 hex strings (not integers) in a few migrated rows.

A one-liner SQLite sanity check that confirms the `id` column is TEXT in the target:

```bash
python -c "import sqlite3; c = sqlite3.connect('target.db'); rows = c.execute(\"SELECT typeof(id) FROM context_entries LIMIT 1\").fetchall(); assert rows and rows[0][0] == 'text', rows"
```

## Reverting

If the migration outcome is unsatisfactory, the source database (untouched by the CLI's read-only connection) remains intact. Point the previous server version at the source database to revert. The target database can be deleted.
