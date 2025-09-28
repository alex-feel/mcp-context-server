-- Main context storage table
CREATE TABLE IF NOT EXISTS context_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
    content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
    text_content TEXT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_thread_id ON context_entries(thread_id);
CREATE INDEX IF NOT EXISTS idx_source ON context_entries(source);
CREATE INDEX IF NOT EXISTS idx_created_at ON context_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_thread_source ON context_entries(thread_id, source);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_entry ON tags(context_entry_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

CREATE TABLE IF NOT EXISTS image_attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id INTEGER NOT NULL,
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

-- Status-based filtering (most common use case)
CREATE INDEX IF NOT EXISTS idx_metadata_status
ON context_entries(json_extract(metadata, '$.status'))
WHERE json_extract(metadata, '$.status') IS NOT NULL;

-- Priority-based filtering (numeric comparisons)
CREATE INDEX IF NOT EXISTS idx_metadata_priority
ON context_entries(json_extract(metadata, '$.priority'))
WHERE json_extract(metadata, '$.priority') IS NOT NULL;

-- Agent type filtering (categorical data)
CREATE INDEX IF NOT EXISTS idx_metadata_agent_type
ON context_entries(json_extract(metadata, '$.agent_type'))
WHERE json_extract(metadata, '$.agent_type') IS NOT NULL;

-- Task type filtering
CREATE INDEX IF NOT EXISTS idx_metadata_task_type
ON context_entries(json_extract(metadata, '$.task_type'))
WHERE json_extract(metadata, '$.task_type') IS NOT NULL;

-- Composite indexes for common filter combinations
CREATE INDEX IF NOT EXISTS idx_thread_metadata_status
ON context_entries(thread_id, json_extract(metadata, '$.status'));

CREATE INDEX IF NOT EXISTS idx_thread_metadata_priority
ON context_entries(thread_id, json_extract(metadata, '$.priority'));

-- Boolean flag indexes (for flags like 'complete', 'error', etc.)
CREATE INDEX IF NOT EXISTS idx_metadata_complete
ON context_entries(json_extract(metadata, '$.complete'))
WHERE json_extract(metadata, '$.complete') IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_metadata_error
ON context_entries(json_extract(metadata, '$.error'))
WHERE json_extract(metadata, '$.error') IS NOT NULL;
