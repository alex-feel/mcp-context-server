-- Semantic search migration: Add vector embeddings support
-- This migration adds tables for semantic search using sqlite-vec

-- Virtual table for vector embeddings (sqlite-vec)
-- NOTE: This table requires sqlite-vec extension to be loaded
-- NOTE: Dimension is templated and replaced during migration (see server.py)
CREATE VIRTUAL TABLE IF NOT EXISTS vec_context_embeddings USING vec0(
    embedding float[{EMBEDDING_DIM}]
);

-- Metadata table for tracking embeddings
CREATE TABLE IF NOT EXISTS embedding_metadata (
    context_id INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

-- Index for fast model-based queries
CREATE INDEX IF NOT EXISTS idx_embedding_metadata_model
ON embedding_metadata(model_name);
