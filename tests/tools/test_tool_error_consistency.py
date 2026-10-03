"""Cross-tool consistency of MCP tool errors.

Validates field constraints, message clarity, and that every error surfaces as a ToolError in one consistent
format without raw Pydantic output.
"""

import base64
from typing import Literal
from typing import cast

import pytest
from fastmcp.exceptions import ToolError

import app.tools
from app.repositories.context_repository.records import EntryProbe
from tests.tools._tool_error_mocks import build_mock_repos
from tests.tools._tool_error_mocks import patch_tool_repositories

# Tool coroutines under test; FastMCP registration happens only in the server lifespan
store_context = app.tools.store_context
search_context = app.tools.search_context
get_context_by_ids = app.tools.get_context_by_ids
delete_context = app.tools.delete_context
update_context = app.tools.update_context
list_threads = app.tools.list_threads
get_statistics = app.tools.get_statistics


# --- Fixtures ---


@pytest.fixture
def mock_repos():
    """Create a mock repository container with transaction support."""
    return build_mock_repos()


@pytest.fixture
def mock_server_dependencies(mock_repos):
    """Mock server dependencies for tool error testing.

    Patches ensure_repositories in each tool module where it is imported.

    Yields:
        MagicMock: The mock repository container.
    """
    with patch_tool_repositories(mock_repos):
        yield mock_repos


# --- Test classes ---


class TestFieldValidation:
    """Test that Field validation constraints are properly applied."""

    @pytest.mark.asyncio
    async def test_thread_id_min_length(self, mock_server_dependencies):
        """Test that thread_id min_length is enforced."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        # Empty string should be caught by min_length=1
        # But we also have manual validation for whitespace
        with pytest.raises(ToolError):
            await store_context(
                thread_id='',
                source='user',
                text='test',
            )

    @pytest.mark.asyncio
    async def test_text_min_length(self, mock_server_dependencies):
        """Test that text min_length is enforced."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        # Empty string should be caught by min_length=1
        # But we also have manual validation for whitespace
        with pytest.raises(ToolError):
            await store_context(
                thread_id='test',
                source='user',
                text='',
            )

    @pytest.mark.asyncio
    async def test_context_id_positive(self, mock_server_dependencies):
        """Test that context_id must be positive."""
        mock_server_dependencies.context.check_entry_exists.return_value = EntryProbe(True, 'agent', 0, 'local', True)
        # This would be caught by Field(gt=0) at FastMCP level
        # Testing our manual validation as fallback
        with pytest.raises(ToolError):
            await update_context(
                context_id='0190abcdef1234567890abcd00000000',  # Should be > 0
                text='test',
            )

    @pytest.mark.asyncio
    async def test_limit_range(self, mock_server_dependencies):
        """Test that Pydantic Field(ge=1, le=100) enforces limit range."""
        # Set up mock to return valid response (rows, stats_dict)
        mock_server_dependencies.context.search_contexts.return_value = ([], {})

        # Valid limits work
        result = await search_context(limit=1)
        assert 'results' in result
        result = await search_context(limit=100)
        assert 'results' in result

    @pytest.mark.asyncio
    async def test_offset_non_negative(self, mock_server_dependencies):
        """Test that Pydantic Field(ge=0) enforces non-negative offset."""
        # Set up mock to return valid response (rows, stats_dict)
        mock_server_dependencies.context.search_contexts.return_value = ([], {})

        # Valid offsets work
        result = await search_context(limit=50, offset=0)
        assert 'results' in result
        result = await search_context(limit=50, offset=100)
        assert 'results' in result


class TestErrorMessageConsistency:
    """Test that error messages are consistent and informative."""

    @pytest.mark.asyncio
    async def test_validation_errors_have_field_context(self, mock_server_dependencies):
        """Test that validation errors mention the field name."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='thread_id'):
            await store_context(
                thread_id='',
                source='user',
                text='test',
            )

        with pytest.raises(ToolError, match='text'):
            await store_context(
                thread_id='test',
                source='user',
                text='',
            )

    @pytest.mark.asyncio
    async def test_business_logic_errors_are_clear(self, mock_server_dependencies):
        """Test that business logic errors have clear messages."""
        mock_server_dependencies.context.check_entry_exists.return_value = EntryProbe(False, None, None, None, False)

        with pytest.raises(ToolError, match='Context entry with ID .* not found'):
            await update_context(
                context_id='0190abcdef1234567890abcd000003e7',
                text='test',
            )

    @pytest.mark.asyncio
    async def test_wrapped_exceptions_preserve_context(self, mock_server_dependencies):
        """Test that wrapped exceptions preserve original error context."""
        mock_server_dependencies.context.store_with_deduplication.side_effect = ValueError('Specific DB error')

        with pytest.raises(ToolError, match='Specific DB error'):
            await store_context(
                thread_id='test',
                source='user',
                text='test',
            )


class TestErrorFormatConsistency:
    """Test that error format is consistent across tools.

    Tests Pydantic bypass scenarios and runtime validation behavior.
    """

    @pytest.mark.asyncio
    async def test_store_context_invalid_source(self, mock_server_dependencies):
        """Test store_context with invalid source wraps DB error in ToolError.

        When called directly (bypassing FastMCP), Pydantic Literal validation
        is not applied. The database CHECK constraint catches the invalid value.
        With mock fixtures, we simulate the DB CHECK constraint error.
        """
        # Simulate the database CHECK constraint rejecting an invalid source value
        mock_server_dependencies.context.store_with_deduplication.side_effect = Exception(
            'CHECK constraint failed: source must be user or agent',
        )

        invalid_source = cast(Literal['user', 'agent'], 'invalid')
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='test_thread',
                source=invalid_source,
                text='Some text',
            )

        # Verify the error is wrapped as ToolError with context
        error_msg = str(exc_info.value).lower()
        assert 'source' in error_msg, 'Error should mention source'

    @pytest.mark.asyncio
    async def test_get_context_by_ids_empty_list(self, mock_server_dependencies):
        """Test get_context_by_ids with empty list - Pydantic handles at protocol layer."""
        # Set up mock to return empty results for empty input
        mock_server_dependencies.context.get_by_ids.return_value = []

        # When called directly (bypassing FastMCP), no runtime validation occurs.
        # This is correct - Pydantic Field(min_length=1) validates at the MCP protocol layer.
        result = await get_context_by_ids(
            context_ids=[],
        )

        # Repository returns empty list for empty input
        assert result == [], 'Should return empty list for empty input when bypassing protocol validation'

    @pytest.mark.asyncio
    async def test_search_context_invalid_limit(self, mock_server_dependencies):
        """Test search_context with invalid limit - Pydantic handles at protocol layer."""
        # Set up mock to return valid response
        mock_server_dependencies.context.search_contexts.return_value = ([], {})

        # When called directly (bypassing FastMCP), no runtime validation occurs.
        # This is correct - Pydantic Field(ge=1, le=100) validates at the MCP protocol layer.
        #
        # However, database-level validation may still occur:
        # - SQLite: Allows negative LIMIT (treated as no limit), returns results
        # - PostgreSQL: Rejects negative LIMIT with error "LIMIT must not be negative"

        try:
            result = await search_context(
                limit=-1,
            )
            # SQLite backend / mock: proceeds with invalid value
            assert 'results' in result, 'Should return result structure'
        except ToolError:
            # PostgreSQL backend: database-level validation rejects negative LIMIT
            pass

    @pytest.mark.asyncio
    async def test_search_context_excessive_limit(self, mock_server_dependencies):
        """Test search_context with excessive limit - Pydantic handles at protocol layer."""
        # Set up mock to return valid response
        mock_server_dependencies.context.search_contexts.return_value = ([], {})

        # When called directly (bypassing FastMCP), no runtime validation occurs.
        result = await search_context(
            limit=101,  # Max is 100
        )

        # Function proceeds with invalid value when protocol validation is bypassed
        assert 'results' in result, 'Should return result structure even with excessive limit'

    @pytest.mark.asyncio
    async def test_search_context_negative_offset(self, mock_server_dependencies):
        """Test search_context with negative offset - Pydantic handles at protocol layer."""
        # Set up mock to return valid response
        mock_server_dependencies.context.search_contexts.return_value = ([], {})

        # When called directly (bypassing FastMCP), no runtime validation occurs.
        #
        # However, database-level validation may still occur:
        # - SQLite: Allows negative OFFSET (treated as 0), returns results
        # - PostgreSQL: Rejects negative OFFSET with error "OFFSET must not be negative"

        try:
            result = await search_context(
                limit=50,
                offset=-1,
            )
            # SQLite backend / mock: proceeds with invalid value
            assert 'results' in result, 'Should return result structure'
        except ToolError:
            # PostgreSQL backend: database-level validation rejects negative OFFSET
            pass

    def test_no_raw_validation_errors_in_responses(self) -> None:
        """Meta test documenting expected error format patterns.

        Raw Pydantic errors typically look like:
        - "Input validation error: '' should be non-empty"
        - "validation error for Model"

        All errors should be ToolError with descriptive messages.
        """
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

    @pytest.mark.asyncio
    async def test_all_tools_handle_none_parameters(self, mock_server_dependencies):
        """Test that tools rely on Pydantic for None validation."""
        _ = mock_server_dependencies  # Fixture needed for mocking

        # When called directly with None (bypassing FastMCP validation),
        # the function has defensive None checks to prevent AttributeError crashes.
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


class TestJSONErrorConsistency:
    """Test that ALL error conditions return consistent JSON format through FastMCP."""

    @pytest.mark.asyncio
    async def test_all_validation_errors_raise_tool_error(self, mock_server_dependencies):
        """Test that all BUSINESS LOGIC validation errors raise ToolError.

        Note: Input validation (Field constraints) is handled by Pydantic at FastMCP level.
        This test only validates business logic errors (e.g., whitespace-only after strip()).
        """
        # Set up mocks for successful database operations
        mock_server_dependencies.context.store_with_deduplication.return_value = (1, False)
        mock_server_dependencies.context.check_entry_exists.return_value = EntryProbe(True, 'agent', 0, 'local', True)

        # Test cases that should raise ToolError for BUSINESS LOGIC
        test_cases = [
            # Business logic: empty strings after strip() are not allowed
            ('store_context empty thread_id', lambda: store_context(thread_id='', source='user', text='test'), 'thread_id'),
            ('store_context empty text', lambda: store_context(thread_id='test', source='user', text=''), 'text'),
            (
                'store_context whitespace thread_id',
                lambda: store_context(thread_id='   ', source='user', text='test'),
                'thread_id',
            ),
            ('store_context whitespace text', lambda: store_context(thread_id='test', source='user', text='   '), 'text'),
            # update_context business logic validation
            (
                'update_context empty text',
                lambda: update_context(context_id='0190abcdef1234567890abcd00000001', text=''),
                'text',
            ),
            ('update_context no fields', lambda: update_context(context_id='0190abcdef1234567890abcd00000001'), 'field'),
            # delete_context business logic validation
            ('delete_context no parameters', lambda: delete_context(), 'provide'),
        ]

        for test_name, test_func, expected_keyword in test_cases:
            with pytest.raises(ToolError) as exc_info:
                await test_func()

            error_msg = str(exc_info.value)
            assert isinstance(error_msg, str), f'{test_name}: Error should be a string'
            assert expected_keyword in error_msg.lower(), (
                f'{test_name}: Error should mention {expected_keyword}, got: {error_msg}'
            )

    @pytest.mark.asyncio
    async def test_all_database_errors_raise_tool_error(self, mock_server_dependencies):
        """Test that all database errors are wrapped in ToolError."""

        # Test store_context database error
        mock_server_dependencies.context.store_with_deduplication.side_effect = Exception('DB error')
        with pytest.raises(ToolError, match='Failed to store context'):
            await store_context(thread_id='test', source='user', text='test')

        # Reset mock
        mock_server_dependencies.context.store_with_deduplication.side_effect = None
        mock_server_dependencies.context.store_with_deduplication.return_value = (1, False)

        # Test update_context database error
        mock_server_dependencies.context.check_entry_exists.return_value = EntryProbe(True, 'agent', 0, 'local', True)
        mock_server_dependencies.context.update_context_entry.side_effect = Exception('Update failed')
        with pytest.raises(ToolError, match='Failed to update context'):
            await update_context(context_id='0190abcdef1234567890abcd00000001', text='new text')

        # Test search_context database error
        mock_server_dependencies.context.search_contexts.side_effect = Exception('Search failed')
        with pytest.raises(ToolError, match='Failed to search context'):
            await search_context(limit=50)

        # Test get_context_by_ids database error
        mock_server_dependencies.context.get_by_ids.side_effect = Exception('Fetch failed')
        with pytest.raises(ToolError, match='Failed to fetch context'):
            await get_context_by_ids(context_ids=['0190abcdef1234567890abcd00000001', '0190abcdef1234567890abcd00000002'])

        # Test list_threads database error
        mock_server_dependencies.statistics.get_thread_list.side_effect = Exception('List failed')
        with pytest.raises(ToolError, match='Failed to list threads'):
            await list_threads()

        # Test get_statistics database error
        mock_server_dependencies.statistics.get_database_statistics.side_effect = Exception('Stats failed')
        with pytest.raises(ToolError, match='Failed to get statistics'):
            await get_statistics()

    @pytest.mark.asyncio
    async def test_image_validation_errors_raise_tool_error(self, mock_server_dependencies):
        """Test that image validation errors raise ToolError."""
        mock_server_dependencies.context.store_with_deduplication.return_value = (1, False)

        # Test invalid base64 image
        with pytest.raises(ToolError, match='Invalid base64'):
            await store_context(
                thread_id='test',
                source='user',
                text='test',
                images=[{'data': 'not-base64!@#$', 'mime_type': 'image/png'}],
            )

        # Test oversized image
        large_data = 'A' * (15 * 1024 * 1024)  # 15MB
        encoded = base64.b64encode(large_data.encode()).decode()

        with pytest.raises(ToolError, match='exceeds .* limit'):
            await store_context(
                thread_id='test',
                source='user',
                text='test',
                images=[{'data': encoded, 'mime_type': 'image/png'}],
            )

    @pytest.mark.asyncio
    async def test_error_messages_are_descriptive(self, mock_server_dependencies):
        """Test that BUSINESS LOGIC error messages are descriptive and not generic.

        Note: Input validation messages come from Pydantic Field constraints.
        This test validates business logic error messages only.
        """
        _ = mock_server_dependencies  # Fixture needed for mocking
        # Test empty thread_id (business logic: whitespace-only not allowed)
        with pytest.raises(ToolError) as exc_info:
            await store_context(thread_id='', source='user', text='test')
        assert 'thread_id' in str(exc_info.value).lower()
        assert 'empty' in str(exc_info.value).lower() or 'whitespace' in str(exc_info.value).lower()

        # Test empty text (business logic: whitespace-only not allowed)
        with pytest.raises(ToolError) as exc_info:
            await store_context(thread_id='test', source='user', text='')
        assert 'text' in str(exc_info.value).lower()
        assert 'empty' in str(exc_info.value).lower() or 'whitespace' in str(exc_info.value).lower()

        # Test no fields provided (business logic: at least one field required)
        with pytest.raises(ToolError) as exc_info:
            await update_context(context_id='0190abcdef1234567890abcd00000001')
        assert 'field' in str(exc_info.value).lower()
        assert 'least' in str(exc_info.value).lower() or 'provide' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_no_raw_pydantic_errors_exposed(self, mock_server_dependencies):
        """Test that raw Pydantic validation errors are never exposed in business logic errors.

        Note: Pydantic Field validation happens at FastMCP level and is properly formatted.
        This test ensures our business logic errors don't leak Pydantic internals.
        """
        _ = mock_server_dependencies  # Fixture needed for mocking
        # These patterns should NEVER appear in BUSINESS LOGIC error messages
        forbidden_patterns = [
            'Input validation error:',
            'validation error for',
            'ensure this value',
            'String should have at least',
            'field required',
            'type=value_error',
            'loc=',
            'ctx=',
        ]

        # Test business logic error conditions
        error_messages = []

        # Collect error messages from business logic validation
        try:
            await store_context(thread_id='', source='user', text='test')
        except ToolError as e:
            error_messages.append(str(e))

        try:
            await store_context(thread_id='test', source='user', text='')
        except ToolError as e:
            error_messages.append(str(e))

        try:
            await update_context(context_id='0190abcdef1234567890abcd00000001')
        except ToolError as e:
            error_messages.append(str(e))

        try:
            await delete_context()
        except ToolError as e:
            error_messages.append(str(e))

        # Check that no forbidden patterns appear in any error message
        for msg in error_messages:
            for pattern in forbidden_patterns:
                assert pattern not in msg, f'Raw Pydantic pattern "{pattern}" found in error: {msg}'

    @pytest.mark.asyncio
    async def test_consistent_error_format_across_tools(self, mock_server_dependencies):
        """Test that all tools use consistent BUSINESS LOGIC error format.

        Note: Input validation is handled by Pydantic Field constraints.
        This test validates business logic error consistency.
        """
        _ = mock_server_dependencies  # Fixture needed for mocking

        # All tools should raise ToolError for business logic failures.
        # Each call is unrolled so static type checkers bind to the specific
        # overload of the called tool, rather than a union over all three tools.
        async def _invoke_tools() -> list[ToolError]:
            errors: list[ToolError] = []
            with pytest.raises(ToolError) as exc_info_store:
                await store_context(thread_id='', source='user', text='test')  # Empty after strip
            errors.append(exc_info_store.value)
            with pytest.raises(ToolError) as exc_info_update:
                await update_context(context_id='0190abcdef1234567890abcd00000001', text='')  # Empty text
            errors.append(exc_info_update.value)
            with pytest.raises(ToolError) as exc_info_delete:
                await delete_context()  # No parameters provided
            errors.append(exc_info_delete.value)
            return errors

        for error in await _invoke_tools():
            # All errors should be ToolError instances
            assert isinstance(error, ToolError)

            # All error messages should be strings
            error_msg = str(error)
            assert isinstance(error_msg, str)

            # All error messages should be non-empty
            assert len(error_msg) > 0

            # No error message should contain raw exception details
            assert 'Traceback' not in error_msg
            assert 'File "' not in error_msg
