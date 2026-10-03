"""Tests for the ``delete_context`` tool."""

import base64
from typing import get_args
from typing import get_type_hints

import pytest
from annotated_types import MaxLen
from fastmcp.exceptions import ToolError
from fastmcp.exceptions import ValidationError as FastMCPValidationError
from pydantic.fields import FieldInfo

import app.tools
from tests.helpers import argument_errors

# The tool functions are plain coroutines that lifespan() registers with FastMCP at startup; tests call them directly.
store_context = app.tools.store_context
search_context = app.tools.search_context
get_context_by_ids = app.tools.get_context_by_ids
delete_context = app.tools.delete_context


@pytest.mark.usefixtures('initialized_server')
class TestDeleteContext:
    """Test the delete_context MCP tool."""

    @pytest.mark.asyncio
    async def test_delete_by_ids(self) -> None:
        """Test deleting specific contexts by IDs."""
        # Create test contexts
        result1 = await store_context(
            thread_id='delete_test',
            source='user',
            text='Entry 1',
        )
        result2 = await store_context(
            thread_id='delete_test',
            source='agent',
            text='Entry 2',
        )
        result3 = await store_context(
            thread_id='delete_test',
            source='user',
            text='Entry 3',
        )

        ids_to_delete = [result1['context_id'], result2['context_id']]

        # Delete first two
        delete_result = await delete_context(context_ids=ids_to_delete)

        assert delete_result['success'] is True
        assert delete_result['deleted_count'] == 2

        # Verify third still exists
        remaining = await search_context(limit=50, thread_id='delete_test')
        assert isinstance(remaining, dict)
        assert len(remaining['results']) == 1
        assert remaining['results'][0]['id'] == result3['context_id']

    @pytest.mark.asyncio
    async def test_delete_by_thread(self) -> None:
        """Test deleting all contexts in a thread."""
        thread_id = 'thread_to_delete'

        # Create multiple contexts
        for i in range(5):
            await store_context(
                thread_id=thread_id,
                source='user' if i % 2 == 0 else 'agent',
                text=f'Entry {i}',
            )

        # Delete entire thread
        delete_result = await delete_context(thread_id=thread_id)

        assert delete_result['success'] is True
        assert delete_result['deleted_count'] == 5

        # Verify all deleted
        remaining = await search_context(limit=50, thread_id=thread_id)
        assert isinstance(remaining, dict)
        assert remaining['results'] == []

    @pytest.mark.asyncio
    async def test_delete_no_parameters(self) -> None:
        """Test error when no delete parameters provided."""
        with pytest.raises(ToolError, match='Must provide either context_ids or thread_id'):
            await delete_context()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_ids(self) -> None:
        """Test deleting non-existent IDs."""
        result = await delete_context(context_ids=['0190abcdef1234567890abcd0000270f', '0190abcdef1234567890abcd00002710'])

        assert result['success'] is True
        assert result['deleted_count'] == 0

    @pytest.mark.asyncio
    async def test_delete_cascades(self) -> None:
        """Test that deleting context also deletes tags and images."""
        # Store context with tags and images
        result = await store_context(
            thread_id='cascade_test',
            source='user',
            text='With tags and images',
            tags=['tag1', 'tag2'],
            images=[{'data': base64.b64encode(b'img').decode('utf-8')}],
        )

        context_id = result['context_id']

        # Delete the context
        delete_result = await delete_context(context_ids=[context_id])
        assert delete_result['success'] is True

        # Verify context and related data are gone
        remaining = await get_context_by_ids(context_ids=[context_id])
        assert remaining == []


class TestDeleteContextIdsCap:
    """delete_context.context_ids carries the same 100-ID cap as every sibling ID list.

    Without the cap an unbounded schema-legal list flows into per-ID prefix
    resolution plus the serial per-ID embedding-delete loop, monopolizing the
    SQLite write queue or pinning a PostgreSQL pool connection, while the identical
    operation via delete_context_batch is rejected cleanly at the boundary.
    """

    def test_context_ids_declares_max_length_cap(self) -> None:
        """The context_ids Field declares max_length=100 (parity with get_context_by_ids)."""
        hints = get_type_hints(app.tools.delete_context, include_extras=True)
        field_info = next(
            meta for meta in get_args(hints['context_ids'])[1:] if isinstance(meta, FieldInfo)
        )
        max_values = [constraint.max_length for constraint in field_info.metadata if isinstance(constraint, MaxLen)]
        assert max_values == [100]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_context_ids_over_cap_rejected_at_boundary(self) -> None:
        """A context_ids list above the cap is rejected by the wire-schema validation with a FastMCP ValidationError."""
        from fastmcp.tools import Tool

        validated = Tool.from_function(app.tools.delete_context)
        oversized = [f'{i:032x}' for i in range(101)]
        with pytest.raises(FastMCPValidationError) as exc_info:
            await validated.run({'context_ids': oversized})
        errors = argument_errors(exc_info)
        assert any(err['type'] == 'too_long' for err in errors), errors
