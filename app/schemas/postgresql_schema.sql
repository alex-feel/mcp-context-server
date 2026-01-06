-- PostgreSQL Schema for MCP Context Server
-- Converted from SQLite schema with PostgreSQL-specific optimizations

-- Function to automatically update updated_at timestamp
-- SET search_path for security (CVE-2018-1058 mitigation)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, pg_temp;

-- Main context storage table
CREATE TABLE IF NOT EXISTS context_entries (
    id BIGSERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
    content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
    text_content TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Trigger to auto-update updated_at on row modification
DROP TRIGGER IF EXISTS update_context_entries_updated_at ON context_entries;
CREATE TRIGGER update_context_entries_updated_at
    BEFORE UPDATE ON context_entries
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Standard indexes
CREATE INDEX IF NOT EXISTS idx_thread_id ON context_entries(thread_id);
CREATE INDEX IF NOT EXISTS idx_source ON context_entries(source);
CREATE INDEX IF NOT EXISTS idx_created_at ON context_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_thread_source ON context_entries(thread_id, source);

-- Tags table (many-to-many relationship)
CREATE TABLE IF NOT EXISTS tags (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id BIGINT NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_entry ON tags(context_entry_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

-- Image attachments table
CREATE TABLE IF NOT EXISTS image_attachments (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id BIGINT NOT NULL,
    image_data BYTEA NOT NULL,
    mime_type TEXT NOT NULL,
    image_metadata JSONB,
    position INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_context ON image_attachments(context_entry_id);

-- Functional indexes for common metadata patterns using JSONB operators
-- These indexes extract specific JSON fields for faster querying
-- Note: Index configuration is managed via METADATA_INDEXED_FIELDS environment variable

-- Status-based filtering (most common use case)
CREATE INDEX IF NOT EXISTS idx_metadata_status
ON context_entries((metadata->>'status'))
WHERE metadata->>'status' IS NOT NULL;

-- Agent name filtering (identify specific agents)
CREATE INDEX IF NOT EXISTS idx_metadata_agent_name
ON context_entries((metadata->>'agent_name'))
WHERE metadata->>'agent_name' IS NOT NULL;

-- Task name filtering (search by task title/name)
CREATE INDEX IF NOT EXISTS idx_metadata_task_name
ON context_entries((metadata->>'task_name'))
WHERE metadata->>'task_name' IS NOT NULL;

-- Project filtering (filter by project name)
CREATE INDEX IF NOT EXISTS idx_metadata_project
ON context_entries((metadata->>'project'))
WHERE metadata->>'project' IS NOT NULL;

-- Report type filtering (filter by report type)
CREATE INDEX IF NOT EXISTS idx_metadata_report_type
ON context_entries((metadata->>'report_type'))
WHERE metadata->>'report_type' IS NOT NULL;

-- Composite indexes for common filter combinations
CREATE INDEX IF NOT EXISTS idx_thread_metadata_status
ON context_entries(thread_id, (metadata->>'status'));

-- GIN index for full JSONB search (enables containment queries)
-- This allows efficient queries like: metadata @> '{"key": "value"}'
-- NOTE: 'technologies' (array) and 'references' (object) fields use this GIN index
-- for containment queries like: metadata @> '{"technologies": ["python"]}'
CREATE INDEX IF NOT EXISTS idx_metadata_gin
ON context_entries USING GIN (metadata jsonb_path_ops);

-- Additional composite index for thread-based queries
CREATE INDEX IF NOT EXISTS idx_thread_created
ON context_entries(thread_id, created_at DESC);
