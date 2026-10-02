"""Source and target database fixtures for the UUIDv7 migration tests."""

from pathlib import Path

import pytest

from tests.cli.migrate_uuid._sources import seed_source_db


@pytest.fixture
def legacy_source_db(tmp_path: Path) -> Path:
    """Create a temporary integer-keyed source DB with a small sample set."""
    path = tmp_path / 'source.db'
    rows: list[dict[str, object]] = [
        {
            'id': 1,
            'thread_id': 'thread-alpha',
            'source': 'user',
            'content_type': 'text',
            'text_content': 'Hello world',
            'metadata': {'agent_name': 'orchestrator'},
            'created_at': '2024-06-15 12:00:00',
        },
        {
            'id': 2,
            'thread_id': 'thread-alpha',
            'source': 'agent',
            'content_type': 'text',
            'text_content': 'See Report ID: 8944 and entries 9044, 14226',
            'metadata': {
                'agent_name': 'analyst',
                'references': {'context_ids': [1]},
            },
            'summary': 'A reply that mentions ID 8944 inline.',
            'created_at': '2025-01-01 09:30:00',
        },
        {
            'id': 3,
            'thread_id': 'thread-beta',
            'source': 'user',
            'content_type': 'text',
            'text_content': 'Third entry',
            'metadata': {
                'references': {'context_ids': [1, 2]},
            },
            'created_at': '2026-02-20 17:45:00',
        },
    ]
    seed_source_db(path, rows)
    return path


@pytest.fixture
def new_target_db_path(tmp_path: Path) -> Path:
    """Return a path to a non-existent target SQLite file."""
    return tmp_path / 'target.db'
