"""Entry seeding and dict-returning call wrappers shared by the navigation tool tests."""

import sqlite3
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import cast

from app.backends import StorageBackend
from app.ids import generate_id_with_timestamp
from app.tools.navigation import grep_context
from app.tools.navigation import navigate_context

_SEQ = datetime(2024, 1, 1, tzinfo=UTC)


async def store_entry(
    backend: StorageBackend,
    text: str,
    *,
    thread_id: str = 't',
    offset_seconds: int = 0,
    owner: str = 'local',
) -> str:
    """Insert one private entry owned by ``owner`` (the default principal unless given) and return its id."""
    cid = generate_id_with_timestamp(_SEQ + timedelta(seconds=offset_seconds))

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (cid, thread_id, 'agent', 'text', text, owner),
        )

    await backend.execute_write(_write)
    return cid


async def grep_as_dict(**kwargs: Any) -> dict[str, Any]:
    """Call grep_context, returning a plain dict for ergonomic test assertions."""
    return cast(dict[str, Any], await grep_context(**kwargs))


async def navigate_as_dict(**kwargs: Any) -> dict[str, Any]:
    """Call navigate_context, returning a plain dict for ergonomic test assertions."""
    return cast(dict[str, Any], await navigate_context(**kwargs))
