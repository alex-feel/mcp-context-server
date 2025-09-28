"""Test metadata filtering error handling."""

import pytest
from app.server import search_context
from fastmcp import Context as MockFastMCPContext


@pytest.mark.asyncio
class TestMetadataErrorHandling:
    """Test error handling for metadata filtering."""

    @pytest.mark.usefixtures('initialized_server')
    async def test_invalid_operator_returns_error(self, mock_context: MockFastMCPContext) -> None:
        """Test that invalid operators return proper error responses."""
        # Try to use an invalid operator
        result = await search_context.fn(
            thread_id='test',
            metadata_filters=[
                {'key': 'status', 'operator': 'invalid_operator', 'value': 'active'},
            ],
            ctx=mock_context,
        )

        # Should return an error response
        assert isinstance(result, dict)
        assert 'entries' in result
        assert result['entries'] == []
        assert 'error' in result
        assert 'Metadata filter validation failed' in result['error']
        assert 'validation_errors' in result
        assert len(result['validation_errors']) > 0
        assert 'invalid_operator' in result['validation_errors'][0]

    @pytest.mark.usefixtures('initialized_server')
    async def test_empty_in_list_returns_error(self, mock_context: MockFastMCPContext) -> None:
        """Test that empty IN operator lists return proper error responses."""
        # Try to use an empty list with IN operator
        result = await search_context.fn(
            thread_id='test',
            metadata_filters=[
                {'key': 'status', 'operator': 'in', 'value': []},
            ],
            ctx=mock_context,
        )

        # Should return an error response
        assert isinstance(result, dict)
        assert 'entries' in result
        assert result['entries'] == []
        assert 'error' in result
        assert 'Metadata filter validation failed' in result['error']
        assert 'validation_errors' in result
        assert len(result['validation_errors']) > 0
        assert 'non-empty list' in result['validation_errors'][0]

    @pytest.mark.usefixtures('initialized_server')
    async def test_multiple_invalid_filters_collect_all_errors(self, mock_context: MockFastMCPContext) -> None:
        """Test that multiple invalid filters collect all errors."""
        # Try multiple invalid filters
        result = await search_context.fn(
            thread_id='test',
            metadata_filters=[
                {'key': 'status', 'operator': 'invalid_op', 'value': 'active'},
                {'key': 'priority', 'operator': 'in', 'value': []},
                {'key': 'type', 'operator': 'another_invalid', 'value': 5},
            ],
            ctx=mock_context,
        )

        # Should return an error response with all validation errors
        assert isinstance(result, dict)
        assert 'entries' in result
        assert result['entries'] == []
        assert 'error' in result
        assert 'validation_errors' in result
        assert len(result['validation_errors']) == 3  # All three errors collected

    @pytest.mark.usefixtures('initialized_server')
    async def test_valid_filters_work_correctly(self, mock_context: MockFastMCPContext) -> None:
        """Test that valid filters still work correctly after error handling changes."""
        # Use valid filters
        result = await search_context.fn(
            thread_id='test',
            metadata_filters=[
                {'key': 'status', 'operator': 'eq', 'value': 'active'},
                {'key': 'priority', 'operator': 'gt', 'value': 5},
                {'key': 'tags', 'operator': 'in', 'value': ['urgent', 'important']},
            ],
            ctx=mock_context,
        )

        # Should NOT return an error
        assert isinstance(result, dict)
        assert 'entries' in result
        assert 'error' not in result
        assert 'validation_errors' not in result
        assert 'stats' in result

    @pytest.mark.usefixtures('initialized_server')
    async def test_case_sensitivity_flag_works(self, mock_context: MockFastMCPContext) -> None:
        """Test that case_sensitive flag is properly handled."""
        # Store test data first
        from app.server import store_context

        await store_context.fn(
            thread_id='test_case',
            source='user',
            text='Test entry',
            metadata={'name': 'TestCase'},
            ctx=mock_context,
        )

        # Search with case-insensitive (default)
        result1 = await search_context.fn(
            thread_id='test_case',
            metadata_filters=[
                {'key': 'name', 'operator': 'eq', 'value': 'testcase', 'case_sensitive': False},
            ],
            ctx=mock_context,
        )
        assert 'entries' in result1
        assert len(result1['entries']) == 1  # Should find the entry

        # Search with case-sensitive
        result2 = await search_context.fn(
            thread_id='test_case',
            metadata_filters=[
                {'key': 'name', 'operator': 'eq', 'value': 'testcase', 'case_sensitive': True},
            ],
            ctx=mock_context,
        )
        assert 'entries' in result2
        assert len(result2['entries']) == 0  # Should NOT find the entry (case mismatch)

        # Search with correct case
        result3 = await search_context.fn(
            thread_id='test_case',
            metadata_filters=[
                {'key': 'name', 'operator': 'eq', 'value': 'TestCase', 'case_sensitive': True},
            ],
            ctx=mock_context,
        )
        assert 'entries' in result3
        assert len(result3['entries']) == 1  # Should find the entry (exact match)
