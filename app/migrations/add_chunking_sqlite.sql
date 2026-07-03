-- Chunking migration for SQLite: Enable 1:N embedding relationship
-- This migration creates embedding_chunks table to map context_id to vec0 rowid
-- Includes chunk boundaries for chunk-aware reranking
-- NOTE: vec0 virtual tables cannot be altered, so we use a mapping table
-- NOTE: This migration is idempotent (safe to run multiple times)

-- Step 1: Create embedding_chunks table for 1:N mapping with chunk boundaries
-- This table maps context_id to vec_context_embeddings.rowid
-- Multiple rows per context_id enables chunking
-- start_index/end_index track character boundaries in original document
--
-- Column types (dual-key bridge between TEXT context IDs and INTEGER vec0 rowids):
--   - context_id is TEXT (UUIDv7 hex FK to context_entries(id)).
--   - vec_rowid is INTEGER because the sqlite-vec virtual table
--     vec_context_embeddings uses INTEGER rowids internally; the bridge
--     between the public TEXT context_id and that INTEGER rowid lives in
--     this column.
--   - id is INTEGER PRIMARY KEY AUTOINCREMENT because SQLite
--     AUTOINCREMENT requires INTEGER.
CREATE TABLE IF NOT EXISTS embedding_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id TEXT NOT NULL,
    vec_rowid INTEGER NOT NULL,  -- Links to vec_context_embeddings.rowid
    start_index INTEGER NOT NULL DEFAULT 0,  -- Character offset where chunk starts in original text
    end_index INTEGER NOT NULL DEFAULT 0,    -- Character offset where chunk ends in original text
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

-- Step 2: Create index for fast context_id lookups (deduplication queries)
CREATE INDEX IF NOT EXISTS idx_embedding_chunks_context
    ON embedding_chunks(context_id);

-- Step 3: Create index for vec_rowid lookups (reverse mapping for deletes)
CREATE INDEX IF NOT EXISTS idx_embedding_chunks_vec_rowid
    ON embedding_chunks(vec_rowid);

-- NOTE: chunk_count column is added to embedding_metadata in Python
-- because SQLite doesn't support ADD COLUMN IF NOT EXISTS
-- (see app/migrations/chunking.py)
