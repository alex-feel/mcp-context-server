-- Main context storage table
-- NOTE on the dual key design (SQLite ONLY):
--   - `rowid_int` is the PRIVATE INTEGER PRIMARY KEY AUTOINCREMENT (the SQLite
--     rowid alias). It is the FTS5 external-content rowid AND the AUTOINCREMENT
--     monotonic counter. It is NEVER exposed via MCP tool boundaries.
--     The name is intentionally `rowid_int` (not `rowid`) because SQLite's
--     `ROWID`/`rowid` are reserved implicit column names; a user-defined
--     column named `rowid` would shadow the implicit one.
--   - `id` is the PUBLIC UUIDv7 hex string (32 lowercase hex chars). It is the
--     externally-visible primary key (via the UNIQUE constraint) and the FK
--     target for `tags`, `image_attachments`, `embedding_metadata`, and
--     `embedding_chunks`. PostgreSQL has no equivalent surrogate (its UUID
--     column is the true PRIMARY KEY).
CREATE TABLE IF NOT EXISTS context_entries (
    rowid_int INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
    content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
    text_content TEXT,
    metadata JSON,
    summary TEXT,
    content_hash TEXT,
    -- Monotonic optimistic-concurrency token. Bumped on every text/metadata
    -- update (see ContextRepository.update_context_entry); a conditional
    -- `WHERE id = ? AND version = ?` makes concurrent same-entry updates
    -- last-writer-by-submission instead of last-writer-by-completion, so an
    -- older-text update can never silently overwrite a newer one (and its
    -- index_tree node rows can never describe stale text).
    version INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_thread_id ON context_entries(thread_id);
CREATE INDEX IF NOT EXISTS idx_source ON context_entries(source);
CREATE INDEX IF NOT EXISTS idx_created_at ON context_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_thread_source ON context_entries(thread_id, source);
-- Deduplication lookup index (mirrors apply_content_hash_migration). Kept in the
-- base schema so every initialization path -- server startup AND the migration CLI
-- target init -- provisions it from inception, not only on a later server start.
CREATE INDEX IF NOT EXISTS idx_context_entries_dedup_hash ON context_entries(thread_id, source, content_hash);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_entry ON tags(context_entry_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

CREATE TABLE IF NOT EXISTS image_attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id TEXT NOT NULL,
    image_data BLOB NOT NULL,
    mime_type TEXT NOT NULL,
    image_metadata JSON,
    position INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_context ON image_attachments(context_entry_id);

-- Functional indexes for common metadata patterns (improves metadata filtering performance)
-- These indexes extract specific JSON fields for faster querying
-- Note: Index configuration is managed via METADATA_INDEXED_FIELDS environment variable

-- Status-based filtering (most common use case)
CREATE INDEX IF NOT EXISTS idx_metadata_status
ON context_entries(json_extract(metadata, '$.status'))
WHERE json_extract(metadata, '$.status') IS NOT NULL;

-- Agent name filtering (identify specific agents)
CREATE INDEX IF NOT EXISTS idx_metadata_agent_name
ON context_entries(json_extract(metadata, '$.agent_name'))
WHERE json_extract(metadata, '$.agent_name') IS NOT NULL;

-- Task name filtering (search by task title/name)
CREATE INDEX IF NOT EXISTS idx_metadata_task_name
ON context_entries(json_extract(metadata, '$.task_name'))
WHERE json_extract(metadata, '$.task_name') IS NOT NULL;

-- Project filtering (filter by project name)
CREATE INDEX IF NOT EXISTS idx_metadata_project
ON context_entries(json_extract(metadata, '$.project'))
WHERE json_extract(metadata, '$.project') IS NOT NULL;

-- Report type filtering (filter by report type)
CREATE INDEX IF NOT EXISTS idx_metadata_report_type
ON context_entries(json_extract(metadata, '$.report_type'))
WHERE json_extract(metadata, '$.report_type') IS NOT NULL;

-- NOTE: 'references' (object) and 'technologies' (array) are NOT indexed in SQLite
-- Array/object fields cannot be efficiently indexed with expression indexes in SQLite
-- Queries on these fields will use full table scan with json_each()
-- For high-performance queries on these fields, use PostgreSQL with GIN index
