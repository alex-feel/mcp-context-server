-- Chunking migration for PostgreSQL: Enable 1:N embedding relationship
-- This migration converts vec_context_embeddings from 1:1 to 1:N with context_entries
-- Includes chunk boundaries for chunk-aware reranking
-- NOTE: This migration is idempotent (safe to run multiple times)
-- NOTE: Schema is templated and replaced during migration

-- Step 1: Check if migration already applied by checking for 'id' column
-- The DO block ensures idempotency - if id column exists, skip schema changes
DO $$
DECLARE
    id_exists BOOLEAN;
BEGIN
    -- Check if 'id' column already exists (migration already applied)
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{SCHEMA}'
          AND table_name = 'vec_context_embeddings'
          AND column_name = 'id'
    ) INTO id_exists;

    IF NOT id_exists THEN
        -- Step 2: Drop existing PRIMARY KEY constraint on context_id
        ALTER TABLE {SCHEMA}.vec_context_embeddings
            DROP CONSTRAINT IF EXISTS vec_context_embeddings_pkey;

        -- Step 3: Add id BIGSERIAL column as new primary key
        -- This enables multiple rows per context_id (1:N relationship)
        ALTER TABLE {SCHEMA}.vec_context_embeddings
            ADD COLUMN id BIGSERIAL;

        -- Step 4: Add chunk boundary columns for chunk-aware reranking
        -- start_index: Character offset where chunk starts in original document
        -- end_index: Character offset where chunk ends in original document
        ALTER TABLE {SCHEMA}.vec_context_embeddings
            ADD COLUMN start_index INTEGER NOT NULL DEFAULT 0;

        ALTER TABLE {SCHEMA}.vec_context_embeddings
            ADD COLUMN end_index INTEGER NOT NULL DEFAULT 0;

        -- Step 5: Create new PRIMARY KEY on id
        ALTER TABLE {SCHEMA}.vec_context_embeddings
            ADD PRIMARY KEY (id);

        -- Step 6: Create index for context_id lookups
        -- Required for deduplication queries and cascading deletes
        CREATE INDEX IF NOT EXISTS idx_vec_embeddings_context_id
            ON {SCHEMA}.vec_context_embeddings(context_id);

        RAISE NOTICE 'Chunking migration: vec_context_embeddings schema updated (1:N with boundaries)';
    ELSE
        -- If id exists but start_index doesn't, add boundary columns only
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = '{SCHEMA}'
              AND table_name = 'vec_context_embeddings'
              AND column_name = 'start_index'
        ) THEN
            ALTER TABLE {SCHEMA}.vec_context_embeddings
                ADD COLUMN start_index INTEGER NOT NULL DEFAULT 0;

            ALTER TABLE {SCHEMA}.vec_context_embeddings
                ADD COLUMN end_index INTEGER NOT NULL DEFAULT 0;

            RAISE NOTICE 'Chunking migration: chunk boundary columns added';
        ELSE
            RAISE NOTICE 'Chunking migration: already applied (all columns exist)';
        END IF;
    END IF;
END $$;

-- Step 7: Drop existing HNSW index (may reference old schema)
DROP INDEX IF EXISTS {SCHEMA}.idx_vec_context_embeddings_hnsw;

-- Step 8: Recreate HNSW index for vector search
-- Using L2 distance (Euclidean distance) via vector_l2_ops
-- Parameters: m=16 (max connections per layer), ef_construction=64 (build quality)
CREATE INDEX IF NOT EXISTS idx_vec_context_embeddings_hnsw
ON {SCHEMA}.vec_context_embeddings
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- Step 9: Add chunk_count column to embedding_metadata
-- This tracks how many chunks exist for each context_id
ALTER TABLE {SCHEMA}.embedding_metadata
    ADD COLUMN IF NOT EXISTS chunk_count INTEGER NOT NULL DEFAULT 1;
