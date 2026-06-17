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
- **Provision an empty target database.** For SQLite, a non-existent file path is fine -- the CLI creates the file and applies the current schema. For PostgreSQL, create an empty database manually (for example, `CREATE DATABASE mcp_context_v3;`); the CLI does not create the PostgreSQL database itself, but it now **auto-initializes the target schema** (tables, indexes, and functions) on an empty target, mirroring the SQLite path. You no longer need to start the server once against the target to create the schema -- if the target has no `context_entries` table, the CLI builds it before copying. When the source carries embeddings, the auto-initialized target gets the **fp32** vector layout (the CLI never enables compression during initialization); enable compression afterward with the separate `--compress` step (see [Compressing an Existing Database](#compressing-an-existing-database)).
- **For SQLite source databases that use semantic search or FTS**, ensure the `sqlite-vec` extension is available in the Python environment running the CLI. The CLI loads the extension on both source and target connections when vector tables are present.
- **For PostgreSQL targets that should use semantic search after migration**, ensure the `pgvector` extension is enabled. When the source has embeddings, the CLI runs `CREATE EXTENSION IF NOT EXISTS vector` on the target during auto-initialization; this requires sufficient privileges. On managed services that restrict `CREATE EXTENSION` (notably Supabase), enable pgvector first via the database management interface (Supabase: Dashboard -> Database -> Extensions -> vector), then rerun.
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

### PostgreSQL Connection Modes (Poolers, Schema, SSL)

The CLI honors the same `POSTGRESQL_*` environment variables the server uses for every PostgreSQL connection it opens (source, target, and the auto-initialization connection):

- **Schema (`POSTGRESQL_SCHEMA`).** The CLI applies `search_path = "<POSTGRESQL_SCHEMA>", public` on every connection, so a non-default schema is resolved correctly for both reads and the auto-initialized target. Set `POSTGRESQL_SCHEMA` to the same value the server will use.
- **Connection poolers (`POSTGRESQL_STATEMENT_CACHE_SIZE`).** The CLI uses prepared statements by default (cache size 100). Transaction-mode poolers -- PgBouncer transaction mode, Pgpool-II, AWS RDS Proxy, and the **Supabase Transaction Pooler (port 6543)** -- break prepared statements. For those, either run the migration through a session-capable endpoint (a Direct connection or the Supabase **Session Pooler**, both on port 5432) or set `POSTGRESQL_STATEMENT_CACHE_SIZE=0`. For Supabase specifically, prefer the Direct connection or Session Pooler URL on port 5432.
- **SSL.** SSL is taken from the connection URL. Put `?sslmode=require` (or your chosen mode) directly in `--source-url` / `--target-url`; the CLI does not inject `sslmode` from settings because the source and target may be different databases than the running server.

### Cross-Backend Migration

The CLI supports SQLite -> PostgreSQL and PostgreSQL -> SQLite. Vector embeddings (the binary `embedding` BLOBs in SQLite's `vec_context_embeddings` table and the corresponding PostgreSQL `pgvector` values) are NOT portable across backends because their binary representations differ. When cross-backend migration is requested, the CLI emits a warning and DROPS **only** the vector embeddings -- you must re-embed the target database after migration (either by storing entries again or by running the standard server startup path with embedding generation enabled, which fills in missing embeddings for entries that lack them).

All other data is copied: `context_entries` (with `summary` and `content_hash`), `tags`, and `image_attachments` (image payloads are portable -- BYTEA on PostgreSQL maps to BLOB on SQLite). The target schema is auto-initialized when absent (the SQLite target via the existing initializer; the PostgreSQL target via the base schema, without the vector tables, which the server creates at your configured `EMBEDDING_DIM` when you re-embed). For PostgreSQL -> SQLite, the SQLite target's FTS5 index is rebuilt from the copied rows, so full-text search works on the target even though FTS is not portable from PostgreSQL.

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

### Recovering From an Interrupted Migration

The CLI does not currently support resume-from-checkpoint. If a migration is interrupted mid-run (Ctrl+C, process kill, machine power loss, network drop on PostgreSQL), the source database is unaffected -- it was opened read-only -- and the target database is left in one of two recoverable states described below.

The recovery procedure is the same in both cases: delete the target database and rerun the same `mcp-context-server-migrate` command. The source is unchanged, so a fresh run reads the same input data and produces the same target. Note that the lower 74 random bits of each UUIDv7 will differ from any prior interrupted run, but the embedded millisecond timestamps and the row-by-row chronological ordering are identical, which is the only property the server depends on.

**State 1: Empty or schema-only target.**

The data-copy transaction rolled back. The target file (SQLite) or database (PostgreSQL) may contain the empty schema (tables defined but no rows) or may not exist at all, depending on how far the CLI got before the interrupt. Recovery:

- SQLite: `rm /path/to/target.db` -- then rerun. Alternatively, pass a different `--target-url` to the rerun.
- PostgreSQL: `DROP DATABASE new_db; CREATE DATABASE new_db;` -- then rerun. Alternatively, point `--target-url` at a different empty database.

**State 2: Data committed but FTS index stale (SQLite only).**

There is a narrow window after the main data transaction commits but before the FTS5 rebuild commits. If the interrupt happens inside that window, the target contains all migrated rows but its `context_entries_fts` virtual table is stale or empty. This is intentional: the FTS rebuild runs outside the main data transaction so that a failing FTS rebuild does not destroy the row data. Two recovery options:

- Clean rerun: `rm /path/to/target.db` and rerun the CLI. This is the simplest option and is always correct.
- In-place FTS rebuild: run the SQLite shell command below to re-index the existing target without redoing the row copy:

  ```bash
  sqlite3 /path/to/target.db "INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild')"
  ```

  This is the same idempotent rebuild directive that the CLI itself issues, so running it manually is safe.

PostgreSQL does not have a State 2 equivalent. Its `text_search_vector` column is a generated column maintained by the row INSERT, so a successful main transaction also produces a complete tsvector index.

**Preflight tip.** If you want to verify a migration will succeed before committing to a real run, pass `--dry-run` on a first invocation. The dry run reads the source, builds the integer-to-UUIDv7 mapping, and computes target rows in memory; it does NOT INSERT into the target. On SQLite, the target schema is still persisted by the dry-run path (because schema creation runs outside the data transaction), so a subsequent real run against the same target path will be rejected by the empty-target gate -- delete or rename the target before the real run. PostgreSQL dry runs leave the target database empty.

### Cross-Backend Vector Embedding Warning

If you ran a cross-backend migration (SQLite to PostgreSQL or vice versa) and saw the warning `cross-backend migration drops vector embeddings; re-embed the target after migration`, your target database contains all rows and metadata but no vector embeddings. To restore embeddings, in order of preference:

- **Recommended:** Run `mcp-context-server-migrate --source-url <target-url> --embed-missing` against the target database. The `--embed-missing` flag invokes the configured embedding provider (`EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`) for every entry that lacks an `embedding_metadata` row. When `ENABLE_EMBEDDING_COMPRESSION=true` is set (the default in v3.0.0), the new embeddings land directly in `vec_context_embeddings_compressed`; otherwise they land in `vec_context_embeddings`. Run with `--dry-run` first to count missing entries without calling the provider. See the [Backfilling missing embeddings](embedding-compression.md#backfilling-missing-embeddings) section of the Embedding Compression Guide for details.
- Re-store the affected entries via the live server (each successful `store_context` call triggers embedding generation when generation is enabled), OR
- Run `mcp-context-server` against the target with `ENABLE_EMBEDDING_GENERATION=true` and trigger any operation that touches the affected entries.

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

## Compressing an Existing Database

If you want to enable the optional `ENABLE_EMBEDDING_COMPRESSION` feature on a database that already contains fp32 embeddings, run the `mcp-context-server-migrate` CLI with the `--compress` flag. This step is independent of the v2-to-v3 schema migration described above: if your database came from a v2 installation, run the v2-to-v3 migration first, then optionally run `--compress` on the resulting v3 database.

For background on what compression does, how it is configured, and the multi-pod seed-locked invariant, see the [Embedding Compression Guide](embedding-compression.md).

### Backup Required

> [!WARNING]
> The `--compress` flag is destructive: it permanently drops the `vec_context_embeddings` fp32 table and replaces it with `vec_context_embeddings_compressed`. The original fp32 vectors are NOT recoverable from the compressed payload (the compression is lossy by design). **Back up your database before running the CLI.**
>
> - SQLite: take a filesystem copy of the `.db` file.
> - PostgreSQL: run `pg_dump` to a portable archive.

### Dry Run First

Always run `--dry-run` before the real execution to preview the operation:

```bash
mcp-context-server-migrate \
  --source-url sqlite:////path/to/db.sqlite \
  --compress \
  --dry-run
```

The dry run prints the BACKUP REQUIRED warning, the source URL (with credentials masked), the row count, the destination table name, the singleton provenance values that will be recorded, and an estimated execution time computed from a probe batch. No writes are issued.

### Execute the Compression

After verifying the dry-run plan, export the compression env vars and re-run without `--dry-run`:

```bash
# All four export lines are optional in v3.0.0; the values shown are the defaults.
# Override only if you intentionally want a different configuration.
# export ENABLE_EMBEDDING_COMPRESSION=true
# export COMPRESSION_BITS=4
# export COMPRESSION_VARIANT=ip
# export COMPRESSION_SEED=0  # immutable after first compressed row is written

mcp-context-server-migrate \
  --source-url sqlite:////path/to/db.sqlite \
  --compress
```

The CLI streams fp32 rows in batches of 10 000, encodes each batch through the compression provider, and INSERTs each batch into `vec_context_embeddings_compressed`. The singleton `compression_metadata` INSERT and the source table DROP run LAST, all inside the same single atomic transaction. Peak working set is bounded by `O(batch_size * dim * 4)` bytes for the fp32 read window (approximately 40 MB at the default `batch_size=10000` and `dim=1024`), independent of total row count. On PostgreSQL the `idx_vec_context_embeddings_hnsw` HNSW index is dropped before the source table. If any step fails before COMMIT, the entire transaction rolls back end-to-end: no partial state, no resume marker, and the source `vec_context_embeddings` table remains intact.

Top-K recall against fp32 ground truth is at least `0.85` per query at the default configuration (the project's regression gate; measured recall is `1.0` on the test corpus).

### After Compression

Start the server with the same compression env vars exported. The server reads the seed from the `compression_metadata` row and serves search through the compressed read path. A mismatch between the env vars and the stored row raises `ConfigurationError` (exit 78) and refuses startup.

### Reversing the Compression

The `--decompress` flag reverses the operation, decoding the compressed payload back to fp32 and recreating the fp32 vec table. Reconstruction is **lossy**: the decoded vectors approximate the original MSE component (for `variant='mse'`) and cannot perfectly reconstruct the inner-product information (for `variant='ip'`). Use it only when you intend to abandon compression on a given database.

`--decompress` follows the same streaming + single-transaction shape as `--compress`: compressed rows are read in batches of 10 000, decoded per batch, and INSERTed into the recreated `vec_context_embeddings` table. The source compressed table DROP and (on PostgreSQL) HNSW index recreation run LAST inside the same atomic transaction. The same peak-memory bound and the same all-or-nothing rollback contract apply.

Unset `ENABLE_EMBEDDING_COMPRESSION` first so the post-decompression startup validator does not reject the disabled state:

```bash
unset ENABLE_EMBEDDING_COMPRESSION

mcp-context-server-migrate \
  --source-url sqlite:////path/to/db.sqlite \
  --decompress
```

On PostgreSQL the `idx_vec_context_embeddings_hnsw` HNSW index is recreated after the fp32 table is rebuilt.

### Cross-Backend Note

The compression CLI is single-backend. If you are migrating from SQLite to PostgreSQL (or vice versa), run the v2-to-v3 cross-backend migration first, then run `--compress` on the target.

### Idempotency

Running `--compress` against an already-compressed database is a no-op: the CLI detects the singleton `compression_metadata` row, prints an informational message, and exits successfully. The same applies to `--decompress` when there is no compression to undo.

### Further Reading

- Feature reference: [Embedding Compression Guide](embedding-compression.md)
- Configuration: [Environment Variables Reference](environment-variables.md#embedding-compression-settings)

## Changing the Embedding Model or Dimensions

There are two distinct scenarios, and they have different procedures because changing the model and changing the dimension affect the vector storage differently.

### Changing the model at the same dimension (use `--re-embed`)

Switching to a different embedding model that produces vectors of the SAME dimension (for example, replacing `qwen3-embedding:0.6b` with another 1024-dimensional model) is a one-command operation. `--re-embed` regenerates embeddings for EVERY `context_entries` row using the currently configured `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`, deleting the old vectors first and backfilling any entries that were missing embeddings along the way (it is a superset of `--embed-missing`). It works on both the fp32 and compressed layouts.

Steps:

1. **Stop the server** and **back up the database** (`--re-embed` deletes and rewrites every embedding).
2. **Set the new `EMBEDDING_MODEL`** in the environment, leaving `EMBEDDING_DIM` unchanged, and ensure `ENABLE_EMBEDDING_GENERATION=true`.
3. **Preview** with `--dry-run` -- it reports how many entries will be re-embedded and the existing model(s) being replaced, without calling the provider:

   ```bash
   mcp-context-server-migrate --source-url sqlite:////path/to/db.sqlite --re-embed --dry-run
   ```

4. **Execute** by re-running without `--dry-run`. Each entry's delete + regenerate runs inside one transaction, so an entry is never left without embeddings. On a compressed database the new payloads land in `vec_context_embeddings_compressed`; on an fp32 database they land in `vec_context_embeddings`.
5. **Restart the server** with the new `EMBEDDING_MODEL` (and the same `COMPRESSION_*` values, if compression is enabled).

`--re-embed` requires `ENABLE_EMBEDDING_GENERATION=true`, is mutually exclusive with `--compress` / `--decompress`, and supersedes `--embed-missing`. Related: the `--embed-missing` flag now refuses to backfill into a database whose existing embeddings use a different model or dimension than the configured values (mixing embedding spaces would corrupt semantic search); it points you here when it detects a model change.

### Changing the dimension (destructive rebuild)

Changing `EMBEDDING_DIM` is heavier and `--re-embed` deliberately refuses it. The dimension is baked into the vector-storage geometry: the fp32 `vec_context_embeddings` column width is fixed at table creation, and under compression the dimension is part of the seed-locked `compression_metadata` codebook (immutable by design). A dimension change therefore requires recreating the vector storage from scratch, not an in-place re-embed. Follow the [Changing Embedding Dimensions](semantic-search.md#changing-embedding-dimensions) procedure: back up, update the configuration, delete the database (or drop and recreate the vector tables at the new dimension), restart the server to create fresh tables, and re-store the data so embeddings are generated at the new dimension.

## End-to-End Checklist: Supabase v2 to v3 With Compression

This is the recommended order for moving an existing v2 deployment to a v3.0.0 Supabase (PostgreSQL) database with TurboQuant compression enabled. The key ordering rule is that compression is always the LAST step: the migration auto-initializes the target with the fp32 layout, and `--compress` converts it afterward.

1. **Provision the empty target database** in Supabase and **enable the `pgvector` extension** (Dashboard -> Database -> Extensions -> vector). The CLI does not run `CREATE DATABASE`, and managed services restrict `CREATE EXTENSION`.
2. **Configure the PostgreSQL connection** for the CLI and server: set `STORAGE_BACKEND=postgresql` and `POSTGRESQL_CONNECTION_STRING`. Use the Direct connection or Session Pooler (port 5432); if you must use the Transaction Pooler (port 6543), also set `POSTGRESQL_STATEMENT_CACHE_SIZE=0`. Set `POSTGRESQL_SCHEMA` to the schema the server will use. Match `EMBEDDING_MODEL` / `EMBEDDING_DIM` to the source data.
3. **Keep compression OFF for the migration**: leave `ENABLE_EMBEDDING_COMPRESSION=false` (or unset) so the target is auto-initialized with the fp32 vector layout the migration copies into.
4. **Run the v2-to-v3 migration** (preview first): `mcp-context-server-migrate --source-url <v2-url> --target-url <v3-supabase-url> --dry-run`, then without `--dry-run`. A same-backend (PostgreSQL -> PostgreSQL) migration copies the embeddings; a cross-backend migration (SQLite -> Supabase) drops them.
5. **Restore embeddings if they were dropped** (cross-backend only): run `mcp-context-server-migrate --source-url <v3-supabase-url> --embed-missing` to regenerate them with the configured provider. (If you also intend to switch models, run `--re-embed` instead.)
6. **Enable compression as the final step**: export `ENABLE_EMBEDDING_COMPRESSION=true` and the `COMPRESSION_*` values, then run `mcp-context-server-migrate --source-url <v3-supabase-url> --compress --dry-run` followed by `--compress`. In multi-pod Kubernetes deployments every pod MUST inherit the same `COMPRESSION_SEED`.
7. **Launch the v3.0.0 server** with `ENABLE_EMBEDDING_COMPRESSION=true` and the same `COMPRESSION_SEED` / `COMPRESSION_BITS` / `COMPRESSION_VARIANT`. The server reads the seed from the singleton `compression_metadata` row; a mismatch raises `ConfigurationError` (exit 78) and refuses to start.

## Reverting

If the migration outcome is unsatisfactory, the source database (untouched by the CLI's read-only connection) remains intact. Point the previous server version at the source database to revert. The target database can be deleted.
