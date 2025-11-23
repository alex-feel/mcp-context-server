-- Semantic search migration for PostgreSQL: Add vector embeddings support
-- This migration adds tables for semantic search using pgvector extension
-- NOTE: This migration requires pgvector extension to be installed
-- NOTE: Dimension is templated and replaced during migration (see server.py)
-- NOTE: pgvector extension is created during backend initialization (postgresql_backend.py)

-- Table for vector embeddings using native vector type
CREATE TABLE IF NOT EXISTS vec_context_embeddings (
    context_id BIGINT PRIMARY KEY,
    embedding vector({EMBEDDING_DIM}),
    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

-- Metadata table for tracking embeddings
CREATE TABLE IF NOT EXISTS embedding_metadata (
    context_id BIGINT PRIMARY KEY,
    model_name TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

-- HNSW index for fast approximate nearest neighbor search
-- Using L2 distance (Euclidean distance) via vector_l2_ops
-- Parameters: m=16 (max connections per layer), ef_construction=64 (build quality)
CREATE INDEX IF NOT EXISTS idx_vec_context_embeddings_hnsw
ON vec_context_embeddings
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- Index for fast model-based queries
CREATE INDEX IF NOT EXISTS idx_embedding_metadata_model
ON embedding_metadata(model_name);

-- Trigger to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_embedding_metadata_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_embedding_metadata_updated_at ON embedding_metadata;
CREATE TRIGGER trigger_embedding_metadata_updated_at
BEFORE UPDATE ON embedding_metadata
FOR EACH ROW
EXECUTE FUNCTION update_embedding_metadata_timestamp();
