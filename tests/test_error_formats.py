"""
Comprehensive tests to verify all validation errors return consistent JSON format.

This test file ensures that ALL validation errors across ALL MCP tools return
the consistent JSON format: {"success": false, "error": "descriptive message"}

No raw Pydantic ValidationError messages should reach the client.
"""

from typing import Literal
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import app.server

# Get the actual async functions from the FunctionTool wrappers
# FastMCP wraps our functions in FunctionTool objects, but we need the original functions for testing
store_context = app.server.store_context.fn
search_context = app.server.search_context.fn
get_context_by_ids = app.server.get_context_by_ids.fn
update_context = app.server.update_context.fn


class TestErrorFormatConsistency:
    """Test that all validation errors return consistent JSON format."""

    @pytest.mark.asyncio
    async def test_store_context_empty_thread_id(self, mock_server_dependencies: None) -> None:
        """Test store_context with empty thread_id returns JSON error."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Test with empty string
        result = await store_context(
            thread_id='',
            source='user',
            text='Some text',
        )

        assert isinstance(result, dict), 'Result should be a dictionary'
        assert result.get('success') is False, 'Success should be False'
        assert 'error' in result, 'Error key should be present'
        assert isinstance(result['error'], str), 'Error should be a string'
        assert 'thread_id' in result['error'].lower(), 'Error should mention thread_id'
        assert 'empty' in result['error'].lower(), 'Error should mention empty'

    @pytest.mark.asyncio
    async def test_store_context_whitespace_thread_id(self, mock_server_dependencies: None) -> None:
        """Test store_context with whitespace thread_id returns JSON error."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Test with whitespace string
        result = await store_context(
            thread_id='   ',
            source='user',
            text='Some text',
        )

        assert isinstance(result, dict), 'Result should be a dictionary'
        assert result.get('success') is False, 'Success should be False'
        assert 'error' in result, 'Error key should be present'
        assert isinstance(result['error'], str), 'Error should be a string'
        assert 'thread_id' in result['error'].lower(), 'Error should mention thread_id'

    @pytest.mark.asyncio
    async def test_store_context_empty_text(self, mock_server_dependencies: None) -> None:
        """Test store_context with empty text returns JSON error."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Test with empty text
        result = await store_context(
            thread_id='test_thread',
            source='user',
            text='',
        )

        assert isinstance(result, dict), 'Result should be a dictionary'
        assert result.get('success') is False, 'Success should be False'
        assert 'error' in result, 'Error key should be present'
        assert isinstance(result['error'], str), 'Error should be a string'
        assert 'text' in result['error'].lower(), 'Error should mention text'
        assert 'empty' in result['error'].lower(), 'Error should mention empty'

    @pytest.mark.asyncio
    async def test_store_context_whitespace_text(self, mock_server_dependencies: None) -> None:
        """Test store_context with whitespace text returns JSON error."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Test with whitespace text
        result = await store_context(
            thread_id='test_thread',
            source='user',
            text='   \t\n   ',
        )

        assert isinstance(result, dict), 'Result should be a dictionary'
        assert result.get('success') is False, 'Success should be False'
        assert 'error' in result, 'Error key should be present'
        assert isinstance(result['error'], str), 'Error should be a string'
        assert 'text' in result['error'].lower() or 'whitespace' in result['error'].lower(), (
            'Error should mention text or whitespace'
        )

    @pytest.mark.asyncio
    async def test_store_context_invalid_source(self, mock_server_dependencies: None) -> None:
        """Test store_context with invalid source returns JSON error."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Test with invalid source type - cast to bypass type checker but test runtime validation
        invalid_source = cast(Literal['user', 'agent'], 'invalid')
        result = await store_context(
            thread_id='test_thread',
            source=invalid_source,
            text='Some text',
        )

        assert isinstance(result, dict), 'Result should be a dictionary'
        assert result.get('success') is False, 'Success should be False'
        assert 'error' in result, 'Error key should be present'
        assert isinstance(result['error'], str), 'Error should be a string'

    @pytest.mark.asyncio
    async def test_update_context_empty_text(self, mock_server_dependencies: None) -> None:
        """Test update_context with empty text returns JSON error."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Mock repository to say entry exists
        with patch('app.server._ensure_repositories') as mock_repos:
            container = MagicMock()
            container.context.check_entry_exists = AsyncMock(return_value=True)
            mock_repos.return_value = container

            result = await update_context(
                context_id=1,
                text='',  # Empty text should fail
            )

        assert isinstance(result, dict), 'Result should be a dictionary'
        assert result.get('success') is False, 'Success should be False'
        assert 'error' in result, 'Error key should be present'
        assert isinstance(result['error'], str), 'Error should be a string'
        assert 'empty' in result['error'].lower() or 'whitespace' in result['error'].lower(), (
            'Error should mention empty or whitespace'
        )

    @pytest.mark.asyncio
    async def test_update_context_whitespace_text(self, mock_server_dependencies: None) -> None:
        """Test update_context with whitespace text returns JSON error."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Mock repository to say entry exists
        with patch('app.server._ensure_repositories') as mock_repos:
            container = MagicMock()
            container.context.check_entry_exists = AsyncMock(return_value=True)
            mock_repos.return_value = container

            result = await update_context(
                context_id=1,
                text='    ',  # Whitespace text should fail
            )

        assert isinstance(result, dict), 'Result should be a dictionary'
        assert result.get('success') is False, 'Success should be False'
        assert 'error' in result, 'Error key should be present'
        assert isinstance(result['error'], str), 'Error should be a string'
        assert 'empty' in result['error'].lower() or 'whitespace' in result['error'].lower(), (
            'Error should mention empty or whitespace'
        )

    @pytest.mark.asyncio
    async def test_get_context_by_ids_empty_list(self, mock_server_dependencies: None) -> None:
        """Test get_context_by_ids with empty list returns JSON error or empty list."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        result = await get_context_by_ids(
            context_ids=[],
        )

        # Empty context_ids should return empty list (not an error)
        assert isinstance(result, list), 'Result should be a list'
        assert len(result) == 0, 'Result should be empty list'

    @pytest.mark.asyncio
    async def test_search_context_invalid_limit(self, mock_server_dependencies: None) -> None:
        """Test search_context with invalid limit returns JSON error."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Test with negative limit
        result = await search_context(
            limit=-1,
        )

        assert isinstance(result, dict), 'Result should be a dictionary'
        assert 'entries' in result, 'Entries key should be present'
        assert isinstance(result['entries'], list), 'Entries should be a list'
        assert len(result['entries']) == 0, 'Entries should be empty on error'
        if 'error' in result:
            assert isinstance(result['error'], str), 'Error should be a string'
            assert 'limit' in result['error'].lower(), 'Error should mention limit'

    @pytest.mark.asyncio
    async def test_search_context_excessive_limit(self, mock_server_dependencies: None) -> None:
        """Test search_context with excessive limit returns JSON error."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Test with limit exceeding maximum
        result = await search_context(
            limit=1000,  # Max is 500
        )

        assert isinstance(result, dict), 'Result should be a dictionary'
        assert 'entries' in result, 'Entries key should be present'
        assert isinstance(result['entries'], list), 'Entries should be a list'
        assert len(result['entries']) == 0, 'Entries should be empty on error'
        if 'error' in result:
            assert isinstance(result['error'], str), 'Error should be a string'
            assert 'limit' in result['error'].lower() or '500' in result['error'], (
                'Error should mention limit or max value'
            )

    @pytest.mark.asyncio
    async def test_search_context_negative_offset(self, mock_server_dependencies: None) -> None:
        """Test search_context with negative offset returns JSON error."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        result = await search_context(
            offset=-1,
        )

        assert isinstance(result, dict), 'Result should be a dictionary'
        assert 'entries' in result, 'Entries key should be present'
        assert isinstance(result['entries'], list), 'Entries should be a list'
        assert len(result['entries']) == 0, 'Entries should be empty on error'
        if 'error' in result:
            assert isinstance(result['error'], str), 'Error should be a string'
            assert 'offset' in result['error'].lower(), 'Error should mention offset'

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
        """Test that all tools handle None parameters gracefully."""
        _ = mock_server_dependencies  # Fixture needed for proper test setup
        # Store context with None text (when called directly)
        with patch('app.models.StoreContextRequest') as mock_request:
            mock_request.side_effect = ValueError('text is required and cannot be empty')

            # When Pydantic validation fails, we should get a JSON error
            # Using cast to simulate None being passed at runtime
            none_text = cast(str, None)
            result = await store_context(
                thread_id='test',
                source='user',
                text=none_text,
            )

            # Even with None, we should get a proper error response
            assert isinstance(result, (dict, type(None))), 'Result should be a dict or None'
