-- Chunking migration for SQLite: Enable 1:N embedding relationship
-- This migration creates embedding_chunks table to map context_id to vec0 rowid
-- Includes chunk boundaries for chunk-aware reranking
-- NOTE: vec0 virtual tables cannot be altered, so we use a mapping table
-- NOTE: This migration is idempotent (safe to run multiple times)

-- Step 1: Create embedding_chunks table for 1:N mapping with chunk boundaries
-- This table maps context_id to vec_context_embeddings.rowid
-- Multiple rows per context_id enables chunking
-- start_index/end_index track character boundaries in original document
CREATE TABLE IF NOT EXISTS embedding_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id INTEGER NOT NULL,
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

-- Step 4: Migrate existing data from embedding_metadata to embedding_chunks
-- Existing embeddings have rowid = context_id (1:1 relationship from old schema)
-- They become single-chunk entries in the new 1:N relationship
-- start_index=0 and end_index=0 indicates legacy data without boundaries
INSERT OR IGNORE INTO embedding_chunks (context_id, vec_rowid, start_index, end_index)
SELECT context_id, context_id, 0, 0
FROM embedding_metadata
WHERE NOT EXISTS (
    SELECT 1 FROM embedding_chunks ec
    WHERE ec.context_id = embedding_metadata.context_id
);

-- NOTE: chunk_count column is added to embedding_metadata in Python
-- because SQLite doesn't support ADD COLUMN IF NOT EXISTS
