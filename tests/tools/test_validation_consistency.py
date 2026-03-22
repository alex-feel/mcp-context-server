"""
Test validation consistency between store_context and update_context.

Ensures both tools have consistent validation behavior.
"""

from typing import cast

import pytest
from fastmcp.exceptions import ToolError

import app.server

# Get the actual async functions - they are no longer wrapped by @mcp.tool() at import time
store_context = app.server.store_context
update_context = app.server.update_context


@pytest.mark.usefixtures('initialized_server')
class TestValidationConsistency:
    """Test that validation is consistent across all MCP tools."""

    @pytest.mark.asyncio
    async def test_store_context_empty_text_validation(
        self,
    ) -> None:
        """Test store_context rejects empty text."""

        # Test empty string
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='test-thread',
                source='user',
                text='',  # Empty string
            )
        assert 'text cannot be empty' in str(exc_info.value)

        # Test whitespace only
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='test-thread',
                source='user',
                text='   ',  # Whitespace only
            )
        assert 'text cannot be empty' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_context_empty_text_validation(
        self,
    ) -> None:
        """Test update_context rejects empty text when provided."""

        # First create a context
        result = await store_context(
            thread_id='test-thread',
            source='user',
            text='Initial text',
        )
        assert result['success'] is True
        context_id = result['context_id']

        # Try to update with empty string
        with pytest.raises(ToolError) as exc_info:
            await update_context(
                context_id=context_id,
                text='',  # Empty string
            )
        assert 'text cannot be empty' in str(exc_info.value)

        # Try to update with whitespace only
        with pytest.raises(ToolError) as exc_info:
            await update_context(
                context_id=context_id,
                text='   ',  # Whitespace only
            )
        assert 'text cannot be empty or contain only whitespace' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_store_context_requires_text(
        self,
    ) -> None:
        """Test that store_context requires text field."""

        # Test with None text (simulated by not providing it)
        # Since text is required, we need to test the validation
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='test-thread',
                source='user',
                text=cast(str, None),  # Force None to bypass type checking
            )
        assert 'text is required' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_context_text_is_optional(
        self,
    ) -> None:
        """Test that update_context allows None for text (optional)."""

        # First create a context
        result = await store_context(
            thread_id='test-thread',
            source='user',
            text='Initial text',
        )
        assert result['success'] is True
        context_id = result['context_id']

        # Update with only metadata (text=None is OK)
        result = await update_context(
            context_id=context_id,
            metadata={'status': 'updated'},
        )

        assert result['success'] is True
        assert 'Successfully updated' in result['message']

    @pytest.mark.asyncio
    async def test_validation_consistency_thread_id(
        self,
    ) -> None:
        """Test thread_id validation is consistent."""

        # Test empty thread_id
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='',
                source='user',
                text='Some text',
            )
        assert 'thread_id cannot be empty' in str(exc_info.value)

        # Test whitespace-only thread_id
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='   ',
                source='user',
                text='Some text',
            )
        assert 'thread_id cannot be empty' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_both_tools_accept_valid_text(
        self,
    ) -> None:
        """Test both tools accept valid non-empty text."""

        valid_texts = [
            'a',  # Single character
            'Hello, world!',  # Normal text
            '   Leading and trailing spaces   ',  # Spaces preserved
            '123',  # Numbers as string
            '!@#$%^&*()',  # Special characters
        ]

        for text in valid_texts:
            # Test store_context
            result = await store_context(
                thread_id=f'test-{text[:10]}',
                source='user',
                text=text,
            )
            assert result['success'] is True, f'store_context failed for: {text!r}'
            context_id = result['context_id']

            # Test update_context
            result = await update_context(
                context_id=context_id,
                text=text + ' updated',
            )
            assert result['success'] is True, f'update_context failed for: {text!r}'

    @pytest.mark.asyncio
    async def test_validation_error_format_consistency(
        self,
    ) -> None:
        """Test that validation error responses have consistent format."""

        # Store context error format - should raise ToolError
        with pytest.raises(ToolError) as store_exc:
            await store_context(
                thread_id='test',
                source='user',
                text='',
            )
        assert isinstance(str(store_exc.value), str)
        assert 'text' in str(store_exc.value).lower()

        # First create valid context for update test
        valid_result = await store_context(
            thread_id='test',
            source='user',
            text='Valid text',
        )
        context_id = valid_result['context_id']

        # Update context error format - should raise ToolError
        with pytest.raises(ToolError) as update_exc:
            await update_context(
                context_id=context_id,
                text='',
            )
        assert isinstance(str(update_exc.value), str)
        assert 'text' in str(update_exc.value).lower()

    @pytest.mark.asyncio
    async def test_whitespace_handling_consistency(
        self,
    ) -> None:
        """Test that whitespace is handled consistently."""

        # Test that pure whitespace is rejected by both
        whitespace_variants = [
            ' ',
            '  ',
            '\t',
            '\n',
            '\r\n',
            ' \t\n ',
        ]

        for ws in whitespace_variants:
            # Test store_context
            with pytest.raises(ToolError) as exc_info:
                await store_context(
                    thread_id='test-ws',
                    source='user',
                    text=ws,
                )
            assert 'text cannot be empty' in str(exc_info.value), f'store_context should reject: {ws!r}'

            # Create valid entry for update test
            valid = await store_context(
                thread_id='test-ws-update',
                source='user',
                text='Valid',
            )
            context_id = valid['context_id']

            # Test update_context
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=context_id,
                    text=ws,
                )
            assert 'text cannot be empty' in str(exc_info.value), f'update_context should reject: {ws!r}'

    @pytest.mark.asyncio
    async def test_none_vs_empty_string_handling(
        self,
    ) -> None:
        """Test distinction between None (omitted) and empty string."""

        # Create a context
        result = await store_context(
            thread_id='test-none',
            source='user',
            text='Initial text',
        )
        context_id = result['context_id']

        # Update with None text (omitted) - should succeed
        result = await update_context(
            context_id=context_id,
            metadata={'status': 'test'},
            # text is omitted (None)
        )
        assert result['success'] is True

        # Update with empty string - should fail
        with pytest.raises(ToolError) as exc_info:
            await update_context(
                context_id=context_id,
                text='',  # Explicitly empty
            )
        assert 'cannot be empty' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_text_with_only_spaces_but_other_content(
        self,
    ) -> None:
        """Test text that has spaces and actual content."""

        valid_spaced_texts = [
            ' a',  # Leading space
            'a ',  # Trailing space
            ' a ',  # Both
            '  multiple  spaces  ',
            '\ta\t',  # Tabs
            '\na\n',  # Newlines
        ]

        for text in valid_spaced_texts:
            # Should succeed for store_context
            result = await store_context(
                thread_id=f'test-{len(text)}',
                source='user',
                text=text,
            )
            assert result['success'] is True, f'Failed for: {text!r}'
            context_id = result['context_id']

            # Should succeed for update_context
            result = await update_context(
                context_id=context_id,
                text=text + 'x',  # Add something to make it different
            )
            assert result['success'] is True, f'Update failed for: {text!r}'
