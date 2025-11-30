-- Full-Text Search migration for PostgreSQL: Add tsvector support
-- This migration adds generated tsvector column and GIN index
-- NOTE: Language is templated and replaced during migration (see server.py)

-- Add generated tsvector column for full-text search
-- GENERATED ALWAYS AS ... STORED means:
-- - Column value is automatically computed from text_content
-- - Value is stored on disk (not computed on read)
-- - Automatically updates when text_content changes
ALTER TABLE context_entries
ADD COLUMN IF NOT EXISTS text_search_vector tsvector
GENERATED ALWAYS AS (to_tsvector('{FTS_LANGUAGE}', COALESCE(text_content, ''))) STORED;

-- Create GIN index for fast full-text searching
-- GIN (Generalized Inverted Index) is optimized for tsvector
-- Provides faster searches compared to GiST for full-text
CREATE INDEX IF NOT EXISTS idx_text_search_gin
ON context_entries USING GIN(text_search_vector);
