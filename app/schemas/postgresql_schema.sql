-- PostgreSQL Schema for MCP Context Server
-- Converted from SQLite schema with PostgreSQL-specific optimizations
-- NOTE: Schema is templated and replaced during initialization (see server.py)

-- Function to automatically update updated_at timestamp
-- SET search_path for security (CVE-2018-1058 mitigation)
CREATE OR REPLACE FUNCTION {SCHEMA}.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, pg_temp;

-- Main context storage table
CREATE TABLE IF NOT EXISTS context_entries (
    id UUID NOT NULL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
    content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
    text_content TEXT,
    metadata JSONB,
    summary TEXT,
    content_hash TEXT,
    -- Monotonic optimistic-concurrency token. Bumped on every text/metadata
    -- update (see ContextRepository.update_context_entry); a conditional
    -- `WHERE id = ? AND version = ?` makes concurrent same-entry updates
    -- last-writer-by-submission instead of last-writer-by-completion, so an
    -- older-text update can never silently overwrite a newer one (and its
    -- index_tree node rows can never describe stale text).
    version BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Trigger to auto-update updated_at on row modification
DROP TRIGGER IF EXISTS update_context_entries_updated_at ON context_entries;
CREATE TRIGGER update_context_entries_updated_at
    BEFORE UPDATE ON context_entries
    FOR EACH ROW
    EXECUTE FUNCTION {SCHEMA}.update_updated_at_column();

-- Standard indexes
CREATE INDEX IF NOT EXISTS idx_thread_id ON context_entries(thread_id);
CREATE INDEX IF NOT EXISTS idx_source ON context_entries(source);
CREATE INDEX IF NOT EXISTS idx_created_at ON context_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_thread_source ON context_entries(thread_id, source);
-- Deduplication lookup index (mirrors apply_content_hash_migration). Kept in the
-- base schema so every initialization path -- server startup AND the migration CLI
-- target init -- provisions it from inception, not only on a later server start.
CREATE INDEX IF NOT EXISTS idx_context_entries_dedup_hash ON context_entries(thread_id, source, content_hash);

-- Tags table (many-to-many relationship)
CREATE TABLE IF NOT EXISTS tags (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id UUID NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_entry ON tags(context_entry_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

-- Image attachments table
CREATE TABLE IF NOT EXISTS image_attachments (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id UUID NOT NULL,
    image_data BYTEA NOT NULL,
    mime_type TEXT NOT NULL,
    image_metadata JSONB,
    position INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_context ON image_attachments(context_entry_id);

-- Scalar metadata field expression indexes are NOT declared here. They are the
-- single responsibility of handle_metadata_indexes (app/migrations/metadata.py),
-- which provisions exactly the typed btree indexes named in
-- METADATA_INDEXED_FIELDS at server startup (the default field set reproduces
-- the former five scalar indexes: status, agent_name, task_name, project,
-- report_type). Keeping them out of the base schema makes the settings-driven
-- sync layer the single source of truth, so a fresh database with a CUSTOM
-- METADATA_INDEXED_FIELDS does not boot with the five defaults that strict-mode
-- reconciliation would then reject as "extra" (and auto-mode would
-- create-then-drop on every fresh init).
--
-- Consequence: a database initialized ONLY by the migration CLI (which does not
-- run handle_metadata_indexes) has no scalar metadata expression indexes until
-- its first server startup provisions them. The always-present GIN index below
-- is NOT managed by the sync layer, so it stays declared in the base schema.

-- GIN index for full JSONB search (enables containment queries)
-- This allows efficient queries like: metadata @> '{"key": "value"}'
-- NOTE: 'technologies' (array) and 'references' (object) fields use this GIN index
-- for containment queries like: metadata @> '{"technologies": ["python"]}'
CREATE INDEX IF NOT EXISTS idx_metadata_gin
ON context_entries USING GIN (metadata jsonb_path_ops);

-- Additional composite index for thread-based queries
CREATE INDEX IF NOT EXISTS idx_thread_created
ON context_entries(thread_id, created_at DESC);
