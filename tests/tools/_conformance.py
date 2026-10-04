"""Shared helpers for the batch versus single-entry conformance tests.

A minimal PNG payload, the thread-name prefix, readers that normalize a stored entry, and the field-by-field
comparator.
"""

import base64
import json
from typing import Any

from app.startup import ensure_repositories
from tests.helpers import LOCAL_SCOPE


def require_context_id(value: str | None) -> str:
    """Return value as str, asserting it is non-None.

    Batch store/update result items declare context_id as Optional because
    failed entries omit the identifier, but successful conformance results
    always carry a string ID.

    Returns:
        The non-None string identifier.
    """
    assert value is not None, 'batch result missing context_id'
    return value


# Minimal valid 1x1 PNG for conformance tests
CONFORMANCE_PNG_DATA = base64.b64encode(bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
    0x54, 0x08, 0x99, 0x01, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x01, 0x7B, 0xDB, 0x56, 0x61, 0x00,
    0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
    0x42, 0x60, 0x82,
])).decode('utf-8')

THREAD_PREFIX = 'conformance'


async def read_db_entry(context_id: str) -> dict[str, Any]:
    """Read a context entry from the database and return a normalized dict for state comparison."""
    repos = await ensure_repositories()
    rows = await repos.context.get_by_ids([context_id], scope=LOCAL_SCOPE)
    assert len(rows) == 1, f'Expected 1 row for id {context_id}, got {len(rows)}'
    row = rows[0]

    tags = await repos.tags.get_tags_for_context(context_id)
    sorted_tags = sorted(tags)
    image_count = await repos.images.count_images_for_context(context_id)

    raw_metadata = row['metadata']
    if isinstance(raw_metadata, str):
        metadata = json.loads(raw_metadata)
    elif raw_metadata is None:
        metadata = None
    else:
        metadata = raw_metadata

    return {
        'thread_id': row['thread_id'],
        'source': row['source'],
        'content_type': row['content_type'],
        'text_content': row['text_content'],
        'metadata': metadata,
        'summary': row['summary'],
        'tags': sorted_tags,
        'image_count': image_count,
    }


async def count_entries_in_thread(thread_id: str) -> int:
    """Count the number of context entries in a thread."""
    repos = await ensure_repositories()
    rows, _ = await repos.context.search_contexts(
        thread_id=thread_id, limit=10000, offset=0, explain_query=False,
        scope=LOCAL_SCOPE,
    )
    return len(rows)


def assert_db_states_equal(
    nb_state: dict[str, Any],
    b_state: dict[str, Any],
    *,
    ignore_thread: bool = True,
) -> None:
    """Assert two database entry states are equal, ignoring thread_id if specified."""
    fields = ['source', 'content_type', 'text_content', 'metadata', 'summary', 'tags', 'image_count']
    if not ignore_thread:
        fields.insert(0, 'thread_id')
    for field in fields:
        assert nb_state[field] == b_state[field], (
            f'DB state mismatch on field {field!r}: '
            f'nonbatch={nb_state[field]!r} vs batch={b_state[field]!r}'
        )
