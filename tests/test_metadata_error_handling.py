"""Test metadata filtering error handling."""

import pytest
from fastmcp import Context as MockFastMCPContext

from app.server import search_context


@pytest.mark.asyncio
class TestMetadataErrorHandling:
    """Test error handling for metadata filtering."""

    @pytest.mark.usefixtures('initialized_server')
    async def test_invalid_operator_returns_error(self, mock_context: MockFastMCPContext) -> None:
        """Test that invalid operators return proper error responses."""
        # Try to use an invalid operator
        result = await search_context(
            limit=50,
            thread_id='test',
            metadata_filters=[
                {'key': 'status', 'operator': 'invalid_operator', 'value': 'active'},
            ],
            ctx=mock_context,
        )

        # Should return an error response
        assert isinstance(result, dict)
        assert 'results' in result
        assert result['results'] == []
        assert 'error' in result
        assert 'Metadata filter validation failed' in result['error']
        assert 'validation_errors' in result
        assert len(result['validation_errors']) > 0
        assert 'invalid_operator' in result['validation_errors'][0]

    @pytest.mark.usefixtures('initialized_server')
    async def test_empty_in_list_returns_error(self, mock_context: MockFastMCPContext) -> None:
        """Test that empty IN operator lists return proper error responses."""
        # Try to use an empty list with IN operator
        result = await search_context(
            limit=50,
            thread_id='test',
            metadata_filters=[
                {'key': 'status', 'operator': 'in', 'value': []},
            ],
            ctx=mock_context,
        )

        # Should return an error response
        assert isinstance(result, dict)
        assert 'results' in result
        assert result['results'] == []
        assert 'error' in result
        assert 'Metadata filter validation failed' in result['error']
        assert 'validation_errors' in result
        assert len(result['validation_errors']) > 0
        assert 'non-empty list' in result['validation_errors'][0]

    @pytest.mark.usefixtures('initialized_server')
    async def test_multiple_invalid_filters_collect_all_errors(self, mock_context: MockFastMCPContext) -> None:
        """Test that multiple invalid filters collect all errors."""
        # Try multiple invalid filters
        result = await search_context(
            limit=50,
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
        assert 'results' in result
        assert result['results'] == []
        assert 'error' in result
        assert 'validation_errors' in result
        assert len(result['validation_errors']) == 3  # All three errors collected

    @pytest.mark.usefixtures('initialized_server')
    async def test_valid_filters_work_correctly(self, mock_context: MockFastMCPContext) -> None:
        """Test that valid filters still work correctly after error handling changes."""
        # Use valid filters
        result = await search_context(
            limit=50,
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
        assert 'results' in result
        assert 'error' not in result
        assert 'validation_errors' not in result

    @pytest.mark.usefixtures('initialized_server')
    async def test_case_sensitivity_flag_works(self, mock_context: MockFastMCPContext) -> None:
        """Test that case_sensitive flag is properly handled."""
        # Store test data first
        from app.server import store_context

        await store_context(
            thread_id='test_case',
            source='user',
            text='Test entry',
            metadata={'name': 'TestCase'},
            ctx=mock_context,
        )

        # Search with case-insensitive (default)
        result1 = await search_context(
            limit=50,
            thread_id='test_case',
            metadata_filters=[
                {'key': 'name', 'operator': 'eq', 'value': 'testcase', 'case_sensitive': False},
            ],
            ctx=mock_context,
        )
        assert 'results' in result1
        assert len(result1['results']) == 1  # Should find the entry

        # Search with case-sensitive
        result2 = await search_context(
            limit=50,
            thread_id='test_case',
            metadata_filters=[
                {'key': 'name', 'operator': 'eq', 'value': 'testcase', 'case_sensitive': True},
            ],
            ctx=mock_context,
        )
        assert 'results' in result2
        assert len(result2['results']) == 0  # Should NOT find the entry (case mismatch)

        # Search with correct case
        result3 = await search_context(
            limit=50,
            thread_id='test_case',
            metadata_filters=[
                {'key': 'name', 'operator': 'eq', 'value': 'TestCase', 'case_sensitive': True},
            ],
            ctx=mock_context,
        )
        assert 'results' in result3
        assert len(result3['results']) == 1  # Should find the entry (exact match)


class TestMetadataFilterValidation:
    """Test MetadataFilter pydantic model validation directly."""

    def test_empty_key_raises_validation_error(self) -> None:
        """Test that empty metadata key raises validation error."""
        from pydantic import ValidationError

        from app.metadata_types import MetadataFilter
        from app.metadata_types import MetadataOperator

        with pytest.raises(ValidationError, match='Metadata key cannot be empty'):
            MetadataFilter(key='', operator=MetadataOperator.EQ, value='test')

    def test_whitespace_only_key_raises_validation_error(self) -> None:
        """Test that whitespace-only metadata key raises validation error."""
        from pydantic import ValidationError

        from app.metadata_types import MetadataFilter
        from app.metadata_types import MetadataOperator

        with pytest.raises(ValidationError, match='Metadata key cannot be empty'):
            MetadataFilter(key='   ', operator=MetadataOperator.EQ, value='test')

    def test_invalid_key_pattern_raises_validation_error(self) -> None:
        """Test that invalid key pattern raises validation error."""
        from pydantic import ValidationError

        from app.metadata_types import MetadataFilter
        from app.metadata_types import MetadataOperator

        # Special characters not allowed
        with pytest.raises(ValidationError, match='Invalid metadata key'):
            MetadataFilter(key='status@field', operator=MetadataOperator.EQ, value='test')

        # Spaces not allowed
        with pytest.raises(ValidationError, match='Invalid metadata key'):
            MetadataFilter(key='my field', operator=MetadataOperator.EQ, value='test')

        # SQL injection attempts blocked
        with pytest.raises(ValidationError, match='Invalid metadata key'):
            MetadataFilter(key="status'; DROP TABLE", operator=MetadataOperator.EQ, value='test')

    def test_valid_key_patterns_accepted(self) -> None:
        """Test that valid key patterns are accepted."""
        from app.metadata_types import MetadataFilter
        from app.metadata_types import MetadataOperator

        # Simple keys
        f1 = MetadataFilter(key='status', operator=MetadataOperator.EQ, value='active')
        assert f1.key == 'status'

        # Nested paths with dots
        f2 = MetadataFilter(key='user.preferences.theme', operator=MetadataOperator.EQ, value='dark')
        assert f2.key == 'user.preferences.theme'

        # Underscores and hyphens
        f3 = MetadataFilter(key='task_name', operator=MetadataOperator.EQ, value='test')
        assert f3.key == 'task_name'

        f4 = MetadataFilter(key='agent-name', operator=MetadataOperator.EQ, value='test')
        assert f4.key == 'agent-name'

    def test_in_operator_requires_list(self) -> None:
        """Test that IN operator requires list value."""
        from pydantic import ValidationError

        from app.metadata_types import MetadataFilter
        from app.metadata_types import MetadataOperator

        with pytest.raises(ValidationError, match='requires a list value'):
            MetadataFilter(key='status', operator=MetadataOperator.IN, value='active')

        with pytest.raises(ValidationError, match='requires a list value'):
            MetadataFilter(key='priority', operator=MetadataOperator.NOT_IN, value=5)

    def test_string_operators_require_string(self) -> None:
        """Test that string operators require string value."""
        from pydantic import ValidationError

        from app.metadata_types import MetadataFilter
        from app.metadata_types import MetadataOperator

        with pytest.raises(ValidationError, match='requires a string value'):
            MetadataFilter(key='name', operator=MetadataOperator.CONTAINS, value=123)

        with pytest.raises(ValidationError, match='requires a string value'):
            MetadataFilter(key='name', operator=MetadataOperator.STARTS_WITH, value=['list'])

        with pytest.raises(ValidationError, match='requires a string value'):
            MetadataFilter(key='name', operator=MetadataOperator.ENDS_WITH, value=True)

    def test_existence_operators_ignore_value(self) -> None:
        """Test that existence operators ignore provided value."""
        from app.metadata_types import MetadataFilter
        from app.metadata_types import MetadataOperator

        # Value is set to None for existence operators
        f1 = MetadataFilter(key='status', operator=MetadataOperator.EXISTS, value='ignored')
        assert f1.value is None

        f2 = MetadataFilter(key='status', operator=MetadataOperator.NOT_EXISTS, value=123)
        assert f2.value is None

        f3 = MetadataFilter(key='status', operator=MetadataOperator.IS_NULL)
        assert f3.value is None

        f4 = MetadataFilter(key='status', operator=MetadataOperator.IS_NOT_NULL, value=['ignored'])
        assert f4.value is None
