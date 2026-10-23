"""The ranked search tools report each result's visibility and whether the caller owns it, never the owner.

Runs semantic_search_context, fts_search_context and hybrid_search_context against a real
SQLite database as two principals. The repositories compute the ownership flag in SQL as an
integer; every tool returns it as a bool, because clients validate it against the boolean
output schema.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

import pytest

from app.tools.search.fts import fts_search_context
from app.tools.search.hybrid import hybrid_search_context
from app.tools.search.semantic import semantic_search_context
from tests.conftest import requires_semantic_search
from tests.helpers import as_principal
from tests.tools.search.conftest import RANKED_KEYWORD
from tests.tools.search.conftest import RANKED_THREAD

type RankedTool = Callable[..., Awaitable[dict[str, Any]]]

_TOOLS = pytest.mark.parametrize(
    'tool',
    [semantic_search_context, fts_search_context, hybrid_search_context],
    ids=['semantic', 'fts', 'hybrid'],
)


async def _read_back(tool: RankedTool, principal_id: str) -> dict[str, tuple[object, object]]:
    """Search as the principal, checking that every result carries a bool ownership flag and no owner.

    Returns:
        Each result's ``(visibility, is_owner)`` keyed by its id.
    """
    with as_principal(principal_id):
        response = await tool(query=RANKED_KEYWORD, thread_id=RANKED_THREAD, limit=10)
    results: list[dict[str, Any]] = response['results']
    for entry in results:
        assert type(entry['is_owner']) is bool, entry
        assert 'owner_id' not in entry
    return {entry['id']: (entry['visibility'], entry['is_owner']) for entry in results}


@requires_semantic_search
@pytest.mark.asyncio
@_TOOLS
async def test_owner_reads_back_ownership_of_its_entries(tool: RankedTool, ranked_entries: dict[str, str]) -> None:
    """Alice reads both of her entries as their owner and never sees bob's private one."""
    assert await _read_back(tool, 'alice') == {
        ranked_entries['alice_private']: ('private', True),
        ranked_entries['alice_public']: ('public', True),
    }


@requires_semantic_search
@pytest.mark.asyncio
@_TOOLS
async def test_reader_reads_back_another_owners_public_entry(tool: RankedTool, ranked_entries: dict[str, str]) -> None:
    """Bob reads alice's public entry without owning it, and his own private entry as its owner."""
    assert await _read_back(tool, 'bob') == {
        ranked_entries['alice_public']: ('public', False),
        ranked_entries['bob_private']: ('private', True),
    }
