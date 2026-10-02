"""Integer-keyed source databases shared by the UUIDv7 migration tests.

The schemas below are the integer-keyed shape the migration CLI accepts as
input. Production code under ``app/`` does not include this shape; these
definitions are the canonical reference for the source layout the migration
tests seed.
"""

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from app.cli.migrate_uuid.records import MigrationOptions

HEX_32_RE = re.compile(r'^[0-9a-f]{32}$')


INTEGER_KEYED_SCHEMA_SQL = '''
CREATE TABLE IF NOT EXISTS context_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
    content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
    text_content TEXT,
    metadata JSON,
    summary TEXT,
    content_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

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
'''


# The context_entries table alone, for sources that need no tags or image attachments.
INTEGER_KEYED_ENTRIES_SCHEMA_SQL = '''
CREATE TABLE IF NOT EXISTS context_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
    content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
    text_content TEXT,
    metadata JSON,
    summary TEXT,
    content_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
'''


def seed_source_db(path: Path, rows: list[dict[str, object]]) -> None:
    """Create an integer-keyed source DB at ``path`` and seed it.

    Each row dict must include ``id``, ``thread_id``, ``source``,
    ``content_type``, ``text_content``, ``metadata`` (or None), and
    ``created_at`` (str or datetime). Optional keys: ``summary``,
    ``content_hash``, ``updated_at``.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(INTEGER_KEYED_SCHEMA_SQL)
        for row in rows:
            metadata_text = row.get('metadata')
            if isinstance(metadata_text, (dict, list)):
                metadata_text = json.dumps(metadata_text)
            created_at = row['created_at']
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            updated_at = row.get('updated_at', created_at)
            if isinstance(updated_at, datetime):
                updated_at = updated_at.isoformat()
            text_content = row.get('text_content')
            content_hash = row.get('content_hash')
            if content_hash is None and isinstance(text_content, str):
                content_hash = hashlib.sha256(text_content.encode('utf-8')).hexdigest()
            conn.execute(
                'INSERT INTO context_entries '
                '(id, thread_id, source, content_type, text_content, metadata, '
                'summary, content_hash, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    row['id'],
                    row['thread_id'],
                    row['source'],
                    row['content_type'],
                    text_content,
                    metadata_text,
                    row.get('summary'),
                    content_hash,
                    created_at,
                    updated_at,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def build_sqlite_options(source: Path, target: Path, *, dry_run: bool = False,
                         report: Path | None = None) -> MigrationOptions:
    """Build a :class:`MigrationOptions` for direct unit-test invocation."""
    return MigrationOptions(
        source_url=f'sqlite:///{source.as_posix()}',
        target_url=f'sqlite:///{target.as_posix()}',
        dry_run=dry_run,
        report_path=report,
    )
