"""Tool-boundary rejection of PostgreSQL-unstorable strings (NUL / lone surrogate).

A string carrying an embedded NUL (U+0000) or an unpaired UTF-16 surrogate stores
and matches on SQLite but is rejected by PostgreSQL (a TEXT bind or the jsonb
parser), and the driver error -- not a ControlFlowError -- charges the circuit
breaker. The store/update/search/grep/delete tools reject such a value at the
input-validation phase, before any generation or connection scope, so both
backends fail fast and identically with a clean client error and zero breaker
charge. These tests exercise the SQLite server through the real tool functions;
the cross-backend circuit-breaker behavior is covered by the integration harness.
"""

import pytest
from fastmcp.exceptions import ToolError

NUL = '\x00'
SURROGATE = '\ud800'


@pytest.mark.usefixtures('initialized_server')
class TestStoreContextNulRejection:
    """store_context rejects a NUL / surrogate in thread_id, text, tags, or metadata."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('bad', [NUL, SURROGATE])
    async def test_thread_id_rejected(self, bad: str) -> None:
        """A NUL/surrogate thread_id is rejected before generation."""
        from app.tools.context import store_context

        with pytest.raises(ToolError):
            await store_context(thread_id=f'thread{bad}', source='user', text='hello')

    @pytest.mark.asyncio
    async def test_text_rejected(self) -> None:
        """A NUL in text is rejected before generation."""
        from app.tools.context import store_context

        with pytest.raises(ToolError):
            await store_context(thread_id='nul-thread', source='user', text=f'he{NUL}llo')

    @pytest.mark.asyncio
    async def test_tag_rejected(self) -> None:
        """A NUL in a tag is rejected before generation."""
        from app.tools.context import store_context

        with pytest.raises(ToolError):
            await store_context(thread_id='nul-thread', source='user', text='hello', tags=[f'ta{NUL}g'])

    @pytest.mark.asyncio
    async def test_metadata_value_and_key_rejected(self) -> None:
        """A NUL in a metadata value OR key is rejected before generation."""
        from app.tools.context import store_context

        with pytest.raises(ToolError):
            await store_context(
                thread_id='nul-thread', source='user', text='hello', metadata={'note': f'a{NUL}b'},
            )
        with pytest.raises(ToolError):
            await store_context(
                thread_id='nul-thread', source='user', text='hello', metadata={f'k{NUL}ey': 'v'},
            )


@pytest.mark.usefixtures('initialized_server')
class TestUpdateContextNulRejection:
    """update_context rejects a NUL in text on an existing entry."""

    @pytest.mark.asyncio
    async def test_text_rejected_on_existing_entry(self) -> None:
        """A NUL-bearing text update on a real entry is rejected before generation."""
        from app.tools.context import store_context
        from app.tools.context import update_context

        stored = await store_context(thread_id='update-nul', source='user', text='original text')
        context_id = stored['context_id']

        with pytest.raises(ToolError):
            await update_context(context_id=context_id, text=f'new{NUL}text')


@pytest.mark.usefixtures('initialized_server')
class TestReadPathNulRejection:
    """search / grep / delete reject a NUL thread_id at the tool boundary (no breaker charge)."""

    @pytest.mark.asyncio
    async def test_search_context_thread_id_rejected(self) -> None:
        """search_context rejects a NUL thread_id before it reaches the filter bind."""
        from app.tools.search import search_context

        with pytest.raises(ToolError):
            await search_context(thread_id=f'thread{NUL}')

    @pytest.mark.asyncio
    async def test_search_context_tag_rejected(self) -> None:
        """search_context rejects a NUL tag filter before it reaches the filter bind."""
        from app.tools.search import search_context

        with pytest.raises(ToolError):
            await search_context(tags=[f'ta{NUL}g'])

    @pytest.mark.asyncio
    async def test_grep_context_thread_id_rejected(self) -> None:
        """grep_context rejects a NUL thread_id before it reaches the filter bind."""
        from app.tools.navigation import grep_context

        with pytest.raises(ToolError):
            await grep_context(pattern='needle', thread_id=f'thread{NUL}')

    @pytest.mark.asyncio
    async def test_delete_context_thread_id_rejected(self) -> None:
        """delete_context rejects a NUL thread_id before it reaches delete_by_thread's bind."""
        from app.tools.context import delete_context

        with pytest.raises(ToolError):
            await delete_context(thread_id=f'thread{NUL}')
