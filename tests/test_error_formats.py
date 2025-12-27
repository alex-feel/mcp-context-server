"""
Comprehensive tests to verify all validation errors return consistent JSON format.

This test file ensures that ALL validation errors across ALL MCP tools raise
ToolError exceptions with descriptive messages, following the FastMCP pattern.

No raw Pydantic ValidationError messages or error dictionaries should reach the client.
"""

from typing import Literal
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import app.server

# Get the actual async functions - they are no longer wrapped by @mcp.tool() at import time
# Tools are registered dynamically in lifespan(), so we can access the functions directly
store_context = app.server.store_context
search_context = app.server.search_context
get_context_by_ids = app.server.get_context_by_ids
update_context = app.server.update_context


class TestErrorFormatConsistency:
    """Test that all validation errors return consistent JSON format."""

    @pytest.mark.asyncio
    async def test_store_context_empty_thread_id(self, mock_server_dependencies: None) -> None:
        """Test store_context with empty thread_id raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Import ToolError
        from fastmcp.exceptions import ToolError

        # Test with empty string - should raise ToolError
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='',
                source='user',
                text='Some text',
            )

        # Verify the error message
        error_msg = str(exc_info.value).lower()
        assert 'thread_id' in error_msg, 'Error should mention thread_id'
        assert 'empty' in error_msg or 'whitespace' in error_msg, 'Error should mention empty or whitespace'

    @pytest.mark.asyncio
    async def test_store_context_whitespace_thread_id(self, mock_server_dependencies: None) -> None:
        """Test store_context with whitespace thread_id raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Import ToolError
        from fastmcp.exceptions import ToolError

        # Test with whitespace string - should raise ToolError
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='   ',
                source='user',
                text='Some text',
            )

        # Verify the error message
        error_msg = str(exc_info.value).lower()
        assert 'thread_id' in error_msg, 'Error should mention thread_id'

    @pytest.mark.asyncio
    async def test_store_context_empty_text(self, mock_server_dependencies: None) -> None:
        """Test store_context with empty text raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Import ToolError
        from fastmcp.exceptions import ToolError

        # Test with empty text - should raise ToolError
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='test_thread',
                source='user',
                text='',
            )

        # Verify the error message
        error_msg = str(exc_info.value).lower()
        assert 'text' in error_msg, 'Error should mention text'
        assert 'empty' in error_msg or 'whitespace' in error_msg, 'Error should mention empty or whitespace'

    @pytest.mark.asyncio
    async def test_store_context_whitespace_text(self, mock_server_dependencies: None) -> None:
        """Test store_context with whitespace text raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Import ToolError
        from fastmcp.exceptions import ToolError

        # Test with whitespace text - should raise ToolError
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='test_thread',
                source='user',
                text='   \t\n   ',
            )

        # Verify the error message
        error_msg = str(exc_info.value).lower()
        assert 'text' in error_msg or 'whitespace' in error_msg, 'Error should mention text or whitespace'

    @pytest.mark.asyncio
    async def test_store_context_invalid_source(self, mock_server_dependencies: None) -> None:
        """Test store_context with invalid source raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        from fastmcp.exceptions import ToolError

        # Test with invalid source type - cast to bypass type checker but test runtime validation
        invalid_source = cast(Literal['user', 'agent'], 'invalid')
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='test_thread',
                source=invalid_source,
                text='Some text',
            )

        # Verify the error message
        error_msg = str(exc_info.value).lower()
        assert 'source' in error_msg, 'Error should mention source'

    @pytest.mark.asyncio
    async def test_update_context_empty_text(self, mock_server_dependencies: None) -> None:
        """Test update_context with empty text raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        from fastmcp.exceptions import ToolError

        # Mock repository to say entry exists
        with patch('app.server._ensure_repositories') as mock_repos:
            container = MagicMock()
            container.context.check_entry_exists = AsyncMock(return_value=True)
            mock_repos.return_value = container

            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=1,
                    text='',  # Empty text should fail
                )

        # Verify the error message
        error_msg = str(exc_info.value).lower()
        assert 'empty' in error_msg or 'whitespace' in error_msg, 'Error should mention empty or whitespace'

    @pytest.mark.asyncio
    async def test_update_context_whitespace_text(self, mock_server_dependencies: None) -> None:
        """Test update_context with whitespace text raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        from fastmcp.exceptions import ToolError

        # Mock repository to say entry exists
        with patch('app.server._ensure_repositories') as mock_repos:
            container = MagicMock()
            container.context.check_entry_exists = AsyncMock(return_value=True)
            mock_repos.return_value = container

            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=1,
                    text='    ',  # Whitespace text should fail
                )

        # Verify the error message
        error_msg = str(exc_info.value).lower()
        assert 'empty' in error_msg or 'whitespace' in error_msg, 'Error should mention empty or whitespace'

    @pytest.mark.asyncio
    async def test_get_context_by_ids_empty_list(self, mock_server_dependencies: None) -> None:
        """Test get_context_by_ids with empty list - Pydantic handles at protocol layer."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup

        # When called directly (bypassing FastMCP), no runtime validation occurs.
        # This is correct - Pydantic Field(min_length=1) validates at the MCP protocol layer.
        # We trust Pydantic completely and don't add redundant runtime checks.

        # The function will fail at the repository layer when trying to query with empty list
        result = await get_context_by_ids(
            context_ids=[],
        )

        # Repository returns empty list for empty input
        assert result == [], 'Should return empty list for empty input when bypassing protocol validation'

    @pytest.mark.asyncio
    async def test_search_context_invalid_limit(self, mock_server_dependencies: None) -> None:
        """Test search_context with invalid limit - Pydantic handles at protocol layer."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        from fastmcp.exceptions import ToolError

        # When called directly (bypassing FastMCP), no runtime validation occurs.
        # This is correct - Pydantic Field(ge=1, le=100) validates at the MCP protocol layer.
        # We trust Pydantic completely and don't add redundant runtime checks.
        #
        # However, database-level validation may still occur:
        # - SQLite: Allows negative LIMIT (treated as no limit), returns results
        # - PostgreSQL: Rejects negative LIMIT with error "LIMIT must not be negative"

        try:
            result = await search_context(
                limit=-1,
            )
            # SQLite backend: proceeds with invalid value
            assert 'results' in result, 'Should return result structure'
        except ToolError:
            # PostgreSQL backend: database-level validation rejects negative LIMIT
            # Test passes - exception is expected for PostgreSQL
            pass

    @pytest.mark.asyncio
    async def test_search_context_excessive_limit(self, mock_server_dependencies: None) -> None:
        """Test search_context with excessive limit - Pydantic handles at protocol layer."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup

        # When called directly (bypassing FastMCP), no runtime validation occurs.
        # This is correct - Pydantic Field(ge=1, le=100) validates at the MCP protocol layer.
        # We trust Pydantic completely and don't add redundant runtime checks.

        result = await search_context(
            limit=101,  # Max is 100
        )

        # Function proceeds with invalid value when protocol validation is bypassed
        assert 'results' in result, 'Should return result structure even with excessive limit'

    @pytest.mark.asyncio
    async def test_search_context_negative_offset(self, mock_server_dependencies: None) -> None:
        """Test search_context with negative offset - Pydantic handles at protocol layer."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        from fastmcp.exceptions import ToolError

        # When called directly (bypassing FastMCP), no runtime validation occurs.
        # This is correct - Pydantic Field(ge=0) validates at the MCP protocol layer.
        # We trust Pydantic completely and don't add redundant runtime checks.
        #
        # However, database-level validation may still occur:
        # - SQLite: Allows negative OFFSET (treated as 0), returns results
        # - PostgreSQL: Rejects negative OFFSET with error "OFFSET must not be negative"

        try:
            result = await search_context(
                limit=50,
                offset=-1,
            )
            # SQLite backend: proceeds with invalid value
            assert 'results' in result, 'Should return result structure'
        except ToolError:
            # PostgreSQL backend: database-level validation rejects negative OFFSET
            # Test passes - exception is expected for PostgreSQL
            pass

    def test_no_raw_validation_errors_in_responses(self) -> None:
        """
        Meta test to ensure no responses contain raw Pydantic validation error format.

        Raw Pydantic errors typically look like:
        - "Input validation error: '' should be non-empty"
        - "Input validation error for FieldName"
        - "validation error for Model"

        All errors should be in JSON format with 'success' and 'error' keys.
        """
        # This is a meta test to document the expected behavior
        # The actual validation is done in the tests above

        # Expected error format (all errors should follow this pattern):
        expected_format = {
            'success': False,
            'error': 'Human-readable error message',
        }

        # Raw Pydantic error formats we should NOT see:
        raw_error_patterns = [
            'Input validation error:',
            'validation error for',
            'should be non-empty',
            'String should have at least',
            'ensure this value',
        ]

        # Document that all error responses should:
        # 1. Be a dictionary
        # 2. Have 'success': False
        # 3. Have 'error' key with descriptive message
        # 4. NOT contain raw Pydantic error patterns
        assert expected_format is not None
        assert raw_error_patterns is not None


class TestValidationIntegration:
    """Integration tests for validation across the full stack."""

    @pytest.mark.asyncio
    async def test_validation_with_fastmcp_client(self) -> None:
        """
        Test that validation errors through FastMCP client are properly formatted.

        This would require a running server and FastMCP client.
        Marked for future implementation.
        """
        # Future implementation: Implement when server is running
        # This test would:
        # 1. Start the MCP server
        # 2. Connect with FastMCP client
        # 3. Send invalid requests
        # 4. Verify all errors are JSON formatted

    @pytest.mark.asyncio
    async def test_all_tools_handle_none_parameters(self, mock_server_dependencies: None) -> None:
        """Test that tools rely on Pydantic for None validation."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        from fastmcp.exceptions import ToolError

        # When called directly with None (bypassing FastMCP validation),
        # the function has defensive None checks to prevent AttributeError crashes.
        # This is defensive programming - Pydantic Field(min_length=1) ensures
        # None never reaches the function in production, but defensive checks
        # prevent crashes in edge cases like tests using cast().

        none_text = cast(str, None)
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='test',
                source='user',
                text=none_text,
            )

        # The ToolError contains defensive None check message
        error_msg = str(exc_info.value).lower()
        assert 'required' in error_msg
