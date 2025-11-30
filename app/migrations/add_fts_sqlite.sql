-- Full-Text Search migration: Add FTS5 support for SQLite
-- This migration adds FTS5 virtual table for fast full-text search
--
-- NOTE: Tokenizer is templated and replaced during migration (see server.py)
-- {TOKENIZER} is replaced with:
--   - 'porter unicode61' for English (enables stemming: "running" matches "run")
--   - 'unicode61' for other languages (multilingual support, no stemming)

-- Create FTS5 virtual table with external content
CREATE VIRTUAL TABLE IF NOT EXISTS context_entries_fts USING fts5(
    text_content,
    content='context_entries',
    content_rowid='id',
    tokenize='{TOKENIZER}'
);

-- Trigger to keep FTS in sync: INSERT
CREATE TRIGGER IF NOT EXISTS context_fts_insert AFTER INSERT ON context_entries
BEGIN
    INSERT INTO context_entries_fts(rowid, text_content)
    VALUES (new.id, new.text_content);
END;

-- Trigger to keep FTS in sync: DELETE
CREATE TRIGGER IF NOT EXISTS context_fts_delete AFTER DELETE ON context_entries
BEGIN
    INSERT INTO context_entries_fts(context_entries_fts, rowid, text_content)
    VALUES('delete', old.id, old.text_content);
END;

-- Trigger to keep FTS in sync: UPDATE
-- Only trigger when text_content actually changes
CREATE TRIGGER IF NOT EXISTS context_fts_update AFTER UPDATE OF text_content ON context_entries
BEGIN
    INSERT INTO context_entries_fts(context_entries_fts, rowid, text_content)
    VALUES('delete', old.id, old.text_content);
    INSERT INTO context_entries_fts(rowid, text_content)
    VALUES (new.id, new.text_content);
END;

-- Populate FTS index with existing data
-- This rebuilds the entire index from the content table
INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild');
