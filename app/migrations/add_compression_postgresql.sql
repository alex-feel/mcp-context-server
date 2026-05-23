-- Embedding compression migration for PostgreSQL.
-- Replaces the fp32 pgvector table with a BYTEA-backed compressed table when
-- ENABLE_EMBEDDING_COMPRESSION=true.
--
-- Idempotency: the HNSW index idx_vec_context_embeddings_hnsw is created in
-- two source migration files (add_semantic_search_postgresql.sql and
-- add_chunking_postgresql.sql). DROP INDEX IF EXISTS handles every prior
-- state (only one, both, or neither present) and re-runs of this migration.
-- Atomicity: the migration loader runs every statement inside a single
-- execute_write() -> conn.transaction(); partial failures roll back.
--
-- Singleton provenance row: compression_metadata uses CHECK (id = 1) for the
-- same reasons documented in add_compression_sqlite.sql.
-- NOTE: Table and index DDL uses BARE names; operators with a
-- non-default POSTGRESQL_SCHEMA must configure search_path so the
-- migration creates tables in the intended schema. This matches the
-- BARE convention used by app/repositories/embedding_repository.py
-- and the TABLE DDL in app/schemas/postgresql_schema.sql. The
-- migration loader (app/migrations/compression.py) no longer
-- substitutes {SCHEMA} for this file.

-- Transaction-level advisory lock for multi-pod DDL safety. Auto-releases on
-- COMMIT or ROLLBACK; matches the pattern used by every other DDL migration.
SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'));

-- Drop the HNSW index BEFORE the table it references to avoid orphaned-index
-- errors. Idempotent; tolerates the index being absent.
DROP INDEX IF EXISTS idx_vec_context_embeddings_hnsw;

-- Drop the fp32 physical table created by add_semantic_search_postgresql.sql.
-- This is the irreversible step.
DROP TABLE IF EXISTS vec_context_embeddings;

-- Compressed-vector physical table. BIGSERIAL keeps insert ordering
-- deterministic. context_id matches the UUID type used elsewhere;
-- ON DELETE CASCADE keeps cleanup consistent with the fp32 table it replaces.
CREATE TABLE IF NOT EXISTS vec_context_embeddings_compressed (
    id BIGSERIAL PRIMARY KEY,
    context_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_index INTEGER NOT NULL DEFAULT 0,
    end_index INTEGER NOT NULL DEFAULT 0,
    payload BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

-- B-tree index on context_id for deletes, deduplication, retrieval.
CREATE INDEX IF NOT EXISTS idx_vec_compressed_context
    ON vec_context_embeddings_compressed(context_id);

-- Singleton provenance table.
CREATE TABLE IF NOT EXISTS compression_metadata (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    provider TEXT NOT NULL,
    bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),
    variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),
    seed BIGINT NOT NULL CHECK (seed >= 0),
    dim INTEGER NOT NULL CHECK (dim > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
