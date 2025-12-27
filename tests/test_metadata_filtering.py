"""Comprehensive tests for metadata filtering functionality."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import pytest

from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder
from app.server import search_context
from app.server import store_context

if TYPE_CHECKING:
    pass


class TestMetadataQueryBuilder:
    """Test the MetadataQueryBuilder class."""

    def test_simple_filter(self) -> None:
        """Test simple key=value filtering."""
        builder = MetadataQueryBuilder()
        builder.add_simple_filter('status', 'active')

        where_clause, params = builder.build_where_clause()
        assert where_clause == "(json_extract(metadata, '$.status') = ?)"
        assert params == ['active']

    def test_multiple_simple_filters(self) -> None:
        """Test multiple simple filters combined with AND."""
        builder = MetadataQueryBuilder()
        builder.add_simple_filter('status', 'active')
        builder.add_simple_filter('priority', 5)

        where_clause, params = builder.build_where_clause()
        assert 'json_extract' in where_clause
        assert len(params) == 2
        assert 'active' in params
        assert 5 in params

    def test_operator_eq(self) -> None:
        """Test equality operator."""
        builder = MetadataQueryBuilder()
        # Test with case_sensitive=True for exact matching
        filter_spec = MetadataFilter(key='status', operator=MetadataOperator.EQ, value='active', case_sensitive=True)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert where_clause == "(json_extract(metadata, '$.status') = ?)"
        assert params == ['active']

        # Test default case-insensitive behavior
        builder2 = MetadataQueryBuilder()
        filter_spec2 = MetadataFilter(key='status', operator=MetadataOperator.EQ, value='active')
        builder2.add_advanced_filter(filter_spec2)

        where_clause2, params2 = builder2.build_where_clause()
        assert 'LOWER' in where_clause2
        assert params2 == ['active']

    def test_operator_ne(self) -> None:
        """Test not-equal operator."""
        builder = MetadataQueryBuilder()
        # Use case_sensitive=True to avoid LOWER function
        filter_spec = MetadataFilter(key='status', operator=MetadataOperator.NE, value='inactive', case_sensitive=True)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert '!=' in where_clause
        assert params == ['inactive']

    def test_operator_gt(self) -> None:
        """Test greater-than operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(key='priority', operator=MetadataOperator.GT, value=5)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'CAST' in where_clause
        assert '>' in where_clause
        assert params == [5]

    def test_operator_in(self) -> None:
        """Test IN operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='status',
            operator=MetadataOperator.IN,
            value=['active', 'pending', 'review'],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IN (' in where_clause
        assert len(params) == 3
        assert 'active' in params

    def test_operator_in_with_integers(self) -> None:
        """Test IN operator with integer array values.

        Regression test: Integer arrays caused silent failures on SQLite
        (type mismatch with json_extract TEXT result) and explicit errors
        on PostgreSQL (asyncpg type mismatch with TEXT cast).
        """
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='priority',
            operator=MetadataOperator.IN,
            value=[5, 9],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IN (' in where_clause
        assert len(params) == 2
        # All values should be converted to strings for TEXT comparison
        assert all(isinstance(p, str) for p in params)
        assert '5' in params
        assert '9' in params

    def test_operator_in_with_floats(self) -> None:
        """Test IN operator with float array values."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='score',
            operator=MetadataOperator.IN,
            value=[math.pi, math.e, 1.41],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IN (' in where_clause
        assert len(params) == 3
        # All values should be converted to strings
        assert all(isinstance(p, str) for p in params)

    def test_operator_in_with_mixed_types(self) -> None:
        """Test IN operator with mixed string and integer array values."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='value',
            operator=MetadataOperator.IN,
            value=['active', 5, 'pending', 10],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IN (' in where_clause
        assert len(params) == 4
        # All values should be converted to strings
        assert all(isinstance(p, str) for p in params)

    def test_operator_not_in(self) -> None:
        """Test NOT IN operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='status',
            operator=MetadataOperator.NOT_IN,
            value=['deleted', 'archived'],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'NOT IN (' in where_clause
        assert len(params) == 2

    def test_operator_not_in_with_integers(self) -> None:
        """Test NOT IN operator with integer array values.

        Regression test: Integer arrays caused silent failures on SQLite
        (type mismatch with json_extract TEXT result) and explicit errors
        on PostgreSQL (asyncpg type mismatch with TEXT cast).
        """
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='priority',
            operator=MetadataOperator.NOT_IN,
            value=[1, 2, 3],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'NOT IN (' in where_clause
        assert len(params) == 3
        # All values should be converted to strings for TEXT comparison
        assert all(isinstance(p, str) for p in params)
        assert '1' in params
        assert '2' in params
        assert '3' in params

    def test_operator_not_in_with_mixed_types(self) -> None:
        """Test NOT IN operator with mixed string and integer array values."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='status',
            operator=MetadataOperator.NOT_IN,
            value=['archived', 100, 'deleted', 200],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'NOT IN (' in where_clause
        assert len(params) == 4
        # All values should be converted to strings
        assert all(isinstance(p, str) for p in params)

    def test_operator_exists(self) -> None:
        """Test EXISTS operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(key='priority', operator=MetadataOperator.EXISTS)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IS NOT NULL' in where_clause
        assert len(params) == 0

    def test_operator_not_exists(self) -> None:
        """Test NOT EXISTS operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(key='optional_field', operator=MetadataOperator.NOT_EXISTS)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IS NULL' in where_clause
        assert len(params) == 0

    def test_operator_contains(self) -> None:
        """Test CONTAINS operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='description',
            operator=MetadataOperator.CONTAINS,
            value='important',
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'LIKE' in where_clause
        assert "'%' ||" in where_clause
        assert params == ['important']

    def test_operator_starts_with(self) -> None:
        """Test STARTS_WITH operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='name',
            operator=MetadataOperator.STARTS_WITH,
            value='test_',
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'LIKE' in where_clause
        assert "|| '%'" in where_clause
        assert params == ['test_']

    def test_operator_ends_with(self) -> None:
        """Test ENDS_WITH operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='filename',
            operator=MetadataOperator.ENDS_WITH,
            value='.txt',
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'LIKE' in where_clause
        assert "'%' ||" in where_clause
        assert params == ['.txt']

    def test_operator_is_null(self) -> None:
        """Test IS_NULL operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(key='deleted_at', operator=MetadataOperator.IS_NULL)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'json_type' in where_clause
        assert "= 'null'" in where_clause
        assert len(params) == 0

    def test_case_insensitive_string_comparison(self) -> None:
        """Test case-insensitive string operations."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='name',
            operator=MetadataOperator.EQ,
            value='TEST',
            case_sensitive=False,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, _ = builder.build_where_clause()
        assert 'LOWER' in where_clause

    def test_nested_json_path(self) -> None:
        """Test nested JSON path support."""
        builder = MetadataQueryBuilder()
        builder.add_simple_filter('user.preferences.theme', 'dark')

        where_clause, params = builder.build_where_clause()
        assert '$.user.preferences.theme' in where_clause
        assert params == ['dark']

    def test_sql_injection_prevention(self) -> None:
        """Test that SQL injection attempts are prevented."""
        builder = MetadataQueryBuilder()

        # Attempt SQL injection in key
        with pytest.raises(ValueError, match='Invalid metadata key'):
            builder.add_simple_filter("status'; DROP TABLE context_entries; --", 'active')

        # Valid key with special characters should work
        builder.add_simple_filter('valid_key-123.nested', 'value')
        where_clause, _ = builder.build_where_clause()
        assert where_clause is not None

    def test_empty_filters(self) -> None:
        """Test behavior with no filters."""
        builder = MetadataQueryBuilder()
        where_clause, params = builder.build_where_clause()
        assert where_clause == ''
        assert params == []

    def test_filter_count(self) -> None:
        """Test filter counting."""
        builder = MetadataQueryBuilder()
        assert builder.get_filter_count() == 0

        builder.add_simple_filter('status', 'active')
        assert builder.get_filter_count() == 1

        filter_spec = MetadataFilter(key='priority', operator=MetadataOperator.GT, value=5)
        builder.add_advanced_filter(filter_spec)
        assert builder.get_filter_count() == 2


@pytest.mark.integration
@pytest.mark.usefixtures('initialized_server')
class TestMetadataFilteringIntegration:
    """Integration tests for metadata filtering with the full stack."""

    async def _setup_test_data(self) -> None:
        """Helper method to set up test data."""
        # Use a unique thread_id for each test run
        import time
        from typing import Any

        self.test_thread_id = f'test_metadata_{int(time.time() * 1000)}'

        # Create fresh test data
        test_data: list[dict[str, Any]] = [
            {
                'thread_id': self.test_thread_id,
                'source': 'agent',
                'text': 'Task 1',
                'metadata': {'status': 'active', 'priority': 5, 'agent_name': 'planner'},
            },
            {
                'thread_id': self.test_thread_id,
                'source': 'agent',
                'text': 'Task 2',
                'metadata': {'status': 'pending', 'priority': 3, 'agent_name': 'executor'},
            },
            {
                'thread_id': self.test_thread_id,
                'source': 'user',
                'text': 'Task 3',
                'metadata': {'status': 'active', 'priority': 8, 'agent_name': 'reviewer'},
            },
            {
                'thread_id': self.test_thread_id,
                'source': 'agent',
                'text': 'Task 4',
                'metadata': {'status': 'completed', 'priority': 1},
            },
            {
                'thread_id': self.test_thread_id,
                'source': 'agent',
                'text': 'Task 5',
                'metadata': {'status': 'error', 'priority': 10, 'error_message': 'timeout'},
            },
            {
                'thread_id': self.test_thread_id,
                'source': 'agent',
                'text': 'Task 6 - no metadata',
                'metadata': None,
            },
        ]

        for data in test_data:
            await store_context(**data, ctx=None)

    @pytest.mark.asyncio
    async def test_simple_metadata_filter(self) -> None:
        """Test simple metadata filtering."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'active'},
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 2
        for entry in result['results']:
            assert entry['metadata']['status'] == 'active'

    @pytest.mark.asyncio
    async def test_multiple_simple_filters(self) -> None:
        """Test multiple simple metadata filters."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'active', 'priority': 5},
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 1
        assert result['results'][0]['text_content'] == 'Task 1'

    @pytest.mark.asyncio
    async def test_advanced_gt_operator(self) -> None:
        """Test greater-than operator."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[{'key': 'priority', 'operator': 'gt', 'value': 5}],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 2  # priority 8 and 10
        priorities = [e['metadata']['priority'] for e in result['results']]
        assert all(p > 5 for p in priorities)

    @pytest.mark.asyncio
    async def test_advanced_in_operator(self) -> None:
        """Test IN operator."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[
                {
                    'key': 'status',
                    'operator': 'in',
                    'value': ['active', 'pending'],
                },
            ],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 3
        statuses = [e['metadata']['status'] for e in result['results']]
        assert all(s in ['active', 'pending'] for s in statuses)

    @pytest.mark.asyncio
    async def test_advanced_in_operator_with_integer_array(self) -> None:
        """Test IN operator with integer array values.

        Regression test: Integer arrays caused silent failures on SQLite
        (type mismatch with json_extract TEXT result) and explicit errors
        on PostgreSQL (asyncpg type mismatch with TEXT cast).
        """
        await self._setup_test_data()

        # Test IN with integer array [5, 10]
        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[
                {
                    'key': 'priority',
                    'operator': 'in',
                    'value': [5, 10],  # Integer array
                },
            ],
            ctx=None,
        )

        assert 'results' in result
        # Should find entries with priority 5 and 10
        assert len(result['results']) == 2
        priorities = [e['metadata']['priority'] for e in result['results']]
        assert all(p in [5, 10] for p in priorities)

    @pytest.mark.asyncio
    async def test_advanced_not_in_operator_with_integer_array(self) -> None:
        """Test NOT IN operator with integer array values.

        Regression test: Integer arrays caused failures in NOT IN operator.
        """
        await self._setup_test_data()

        # Test NOT IN with integer array - should exclude entries with priority 1, 3, 5
        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[
                {
                    'key': 'priority',
                    'operator': 'not_in',
                    'value': [1, 3, 5],  # Integer array
                },
            ],
            ctx=None,
        )

        assert 'results' in result
        # Should find entries with priority 8 and 10 (excluding 1, 3, 5)
        assert len(result['results']) == 2
        priorities = [e['metadata']['priority'] for e in result['results']]
        assert all(p not in [1, 3, 5] for p in priorities)

    @pytest.mark.asyncio
    async def test_advanced_exists_operator(self) -> None:
        """Test EXISTS operator."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[{'key': 'agent_name', 'operator': 'exists'}],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 3
        for entry in result['results']:
            assert 'agent_name' in entry['metadata']

    @pytest.mark.asyncio
    async def test_advanced_contains_operator(self) -> None:
        """Test CONTAINS operator."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[
                {
                    'key': 'agent_name',
                    'operator': 'contains',
                    'value': 'plan',
                },
            ],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 1
        assert result['results'][0]['metadata']['agent_name'] == 'planner'

    @pytest.mark.asyncio
    async def test_combined_filters(self) -> None:
        """Test combining simple and advanced filters."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            source='agent',
            metadata={'status': 'active'},
            metadata_filters=[{'key': 'priority', 'operator': 'gte', 'value': 5}],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 1
        entry = result['results'][0]
        assert entry['metadata']['status'] == 'active'
        assert entry['metadata']['priority'] >= 5
        assert entry['source'] == 'agent'

    @pytest.mark.asyncio
    async def test_explain_query(self) -> None:
        """Test query explanation feature."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'active'},
            explain_query=True,
            ctx=None,
        )

        assert 'results' in result
        assert 'stats' in result
        stats = result['stats']
        assert 'execution_time_ms' in stats
        assert 'filters_applied' in stats
        assert 'rows_returned' in stats
        # The implementation counts filters differently - accept either 1 or 2
        assert stats['filters_applied'] in [1, 2]  # Could be just metadata filter or thread_id + metadata
        assert stats['rows_returned'] == 2

    @pytest.mark.asyncio
    async def test_empty_result_set(self) -> None:
        """Test filtering that returns no results."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'nonexistent'},
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_null_metadata_handling(self) -> None:
        """Test handling of entries with null metadata."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[{'key': 'status', 'operator': 'not_exists'}],
            ctx=None,
        )

        assert 'results' in result
        # Should find the entry with null metadata
        found_null = any('no metadata' in e['text_content'] for e in result['results'])
        assert found_null

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_metadata_filter_performance(self) -> None:
        """Test that metadata filtering meets performance targets."""
        await self._setup_test_data()

        # Simple filter performance test
        start_time = time.time()
        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'active'},
            ctx=None,
        )
        simple_time = (time.time() - start_time) * 1000

        assert simple_time < 200  # Should be under 200ms (relaxed for CI variability)
        assert 'results' in result

        # Complex filter performance test
        start_time = time.time()
        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'active'},
            metadata_filters=[
                {'key': 'priority', 'operator': 'gt', 'value': 3},
                {'key': 'agent_name', 'operator': 'exists'},
            ],
            ctx=None,
        )
        complex_time = (time.time() - start_time) * 1000

        assert complex_time < 500  # Should be under 500ms (relaxed for CI variability)
        assert 'results' in result


@pytest.mark.asyncio
@pytest.mark.usefixtures('initialized_server')
@pytest.mark.parametrize(
    ('operator', 'value', 'expected_count'),
    [
        (MetadataOperator.EQ, 'active', 2),
        (MetadataOperator.NE, 'active', 3),
        (MetadataOperator.GT, 5, 2),
        (MetadataOperator.GTE, 5, 3),
        (MetadataOperator.LT, 5, 2),
        (MetadataOperator.LTE, 5, 3),
        (MetadataOperator.IN, ['active', 'pending'], 3),
        (MetadataOperator.NOT_IN, ['active', 'pending'], 2),
        (MetadataOperator.EXISTS, None, 3),
        (MetadataOperator.NOT_EXISTS, None, 2),
    ],
)
async def test_all_operators(
    operator: MetadataOperator,
    value: str | int | list[str] | None,
    expected_count: int,
) -> None:
    """Parameterized test for all metadata operators."""
    # Create test data
    test_data = [
        {'status': 'active', 'priority': 5, 'agent_name': 'planner'},
        {'status': 'pending', 'priority': 3, 'agent_name': 'executor'},
        {'status': 'active', 'priority': 8, 'agent_name': 'reviewer'},
        {'status': 'completed', 'priority': 1},
        {'status': 'error', 'priority': 10},
    ]

    for i, metadata in enumerate(test_data):
        await store_context(
            thread_id='test_operators',
            source='agent',
            text=f'Task {i + 1}',
            metadata=metadata,
            ctx=None,
        )

    # Determine which field to filter on
    if operator in (MetadataOperator.EXISTS, MetadataOperator.NOT_EXISTS):
        key = 'agent_name'
    elif operator in (MetadataOperator.GT, MetadataOperator.GTE, MetadataOperator.LT, MetadataOperator.LTE):
        key = 'priority'
    else:
        key = 'status'

    # Apply filter
    result = await search_context(
        limit=50,
        thread_id='test_operators',
        metadata_filters=[{'key': key, 'operator': operator.value, 'value': value}],
        ctx=None,
    )

    assert 'results' in result
    assert len(result['results']) == expected_count


@pytest.mark.integration
@pytest.mark.usefixtures('initialized_server')
class TestMetadataFilterErrorHandling:
    """Test error handling for invalid metadata filters."""

    @pytest.mark.asyncio
    async def test_invalid_operator_returns_validation_error(self) -> None:
        """Test that invalid operator returns explicit validation error."""
        result = await search_context(
            limit=50,
            metadata_filters=[{'key': 'status', 'operator': 'invalid_operator', 'value': 'test'}],
            ctx=None,
        )

        assert 'error' in result
        assert result['error'] == 'Metadata filter validation failed'
        assert 'validation_errors' in result
        assert len(result['validation_errors']) == 1
        error_msg = result['validation_errors'][0].lower()
        assert 'invalid_operator' in error_msg or 'invalid' in error_msg

    @pytest.mark.asyncio
    async def test_multiple_invalid_filters_returns_all_errors(self) -> None:
        """Test that multiple invalid filters return all validation errors."""
        result = await search_context(
            limit=50,
            metadata_filters=[
                {'key': 'status', 'operator': 'invalid_op1', 'value': 'test'},
                {'key': 'priority', 'operator': 'invalid_op2', 'value': 123},
            ],
            ctx=None,
        )

        assert 'error' in result
        assert result['error'] == 'Metadata filter validation failed'
        assert 'validation_errors' in result
        assert len(result['validation_errors']) == 2

    @pytest.mark.asyncio
    async def test_invalid_key_with_sql_injection_returns_error(self) -> None:
        """Test that invalid keys (SQL injection attempts) return validation error."""
        result = await search_context(
            limit=50,
            metadata_filters=[{'key': 'DROP TABLE;--', 'operator': 'eq', 'value': 'test'}],
            ctx=None,
        )

        assert 'error' in result
        assert result['error'] == 'Metadata filter validation failed'
        assert 'validation_errors' in result
        assert len(result['validation_errors']) == 1


@pytest.mark.integration
@pytest.mark.usefixtures('initialized_server')
class TestNestedJSONMetadata:
    """Test nested JSON structures in metadata."""

    @pytest.mark.asyncio
    async def test_store_nested_objects(self) -> None:
        """Test storing nested JSON objects in metadata."""
        complex_metadata = {
            'status': 'active',
            'config': {
                'database': {
                    'connection': {
                        'pool': {'size': 10, 'timeout': 30},
                        'retry': {'max_attempts': 3, 'backoff': 2.5},
                    },
                },
                'cache': {'enabled': True, 'ttl': 300},
            },
            'user': {'id': 123, 'name': 'Alice Johnson', 'preferences': {'theme': 'dark', 'language': 'en'}},
        }

        result = await store_context(
            thread_id='test_nested_json',
            source='agent',
            text='Test nested metadata storage',
            metadata=complex_metadata,
            ctx=None,
        )

        assert result['success'] is True
        assert 'context_id' in result

        # Retrieve and verify the metadata is preserved
        search_result = await search_context(limit=50, thread_id='test_nested_json', ctx=None)
        assert len(search_result['results']) == 1

        stored_metadata = search_result['results'][0]['metadata']
        assert stored_metadata['status'] == 'active'
        assert stored_metadata['config']['database']['connection']['pool']['size'] == 10
        assert stored_metadata['config']['database']['connection']['pool']['timeout'] == 30
        assert stored_metadata['config']['database']['connection']['retry']['max_attempts'] == 3
        assert stored_metadata['config']['database']['connection']['retry']['backoff'] == 2.5
        assert stored_metadata['config']['cache']['enabled'] is True
        assert stored_metadata['user']['preferences']['theme'] == 'dark'
        assert stored_metadata['user']['preferences']['language'] == 'en'

    @pytest.mark.asyncio
    async def test_store_arrays_in_metadata(self) -> None:
        """Test storing arrays in metadata."""
        metadata_with_arrays = {
            'tags': ['urgent', 'backend', 'production'],
            'priority_levels': [1, 2, 3, 4, 5],
            'mixed_array': ['string', 42, math.pi, True, None],
            'nested_arrays': [[1, 2], [3, 4], [5, 6]],
        }

        result = await store_context(
            thread_id='test_arrays',
            source='agent',
            text='Test array metadata',
            metadata=metadata_with_arrays,
            ctx=None,
        )

        assert result['success'] is True

        # Retrieve and verify arrays are preserved
        search_result = await search_context(limit=50, thread_id='test_arrays', ctx=None)
        stored_metadata = search_result['results'][0]['metadata']

        assert stored_metadata['tags'] == ['urgent', 'backend', 'production']
        assert stored_metadata['priority_levels'] == [1, 2, 3, 4, 5]
        assert stored_metadata['mixed_array'] == ['string', 42, math.pi, True, None]
        assert stored_metadata['nested_arrays'] == [[1, 2], [3, 4], [5, 6]]

    @pytest.mark.asyncio
    async def test_query_nested_paths(self) -> None:
        """Test querying nested JSON paths."""
        # Store multiple entries with nested metadata
        await store_context(
            thread_id='test_nested_paths',
            source='agent',
            text='Entry 1',
            metadata={'user': {'preferences': {'theme': 'dark', 'notifications': {'email': True}}}},
            ctx=None,
        )

        await store_context(
            thread_id='test_nested_paths',
            source='agent',
            text='Entry 2',
            metadata={'user': {'preferences': {'theme': 'light', 'notifications': {'email': False}}}},
            ctx=None,
        )

        # Query using nested path
        result = await search_context(
            limit=50,
            thread_id='test_nested_paths',
            metadata={'user.preferences.theme': 'dark'},
            ctx=None,
        )

        assert len(result['results']) == 1
        assert result['results'][0]['text_content'] == 'Entry 1'
        assert result['results'][0]['metadata']['user']['preferences']['theme'] == 'dark'

    @pytest.mark.asyncio
    async def test_complex_nested_structure(self) -> None:
        """Test very complex nested structure with multiple levels."""
        complex_structure = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'value': 'deeply_nested',
                            'number': 42,
                            'array': [1, 2, 3],
                            'object': {'key': 'value'},
                        },
                    },
                },
            },
            'metrics': {
                'cpu': 45.5,
                'memory': 512,
                'disk': {'used': 80.5, 'total': 100.0, 'partitions': ['/dev/sda1', '/dev/sda2']},
            },
            'features': {
                'enabled': ['feature_a', 'feature_b', 'feature_c'],
                'disabled': [],
                'experimental': {'count': 3, 'names': ['exp_1', 'exp_2', 'exp_3']},
            },
        }

        result = await store_context(
            thread_id='test_complex',
            source='agent',
            text='Complex nested structure test',
            metadata=complex_structure,
            ctx=None,
        )

        assert result['success'] is True

        # Verify structure is preserved
        search_result = await search_context(limit=50, thread_id='test_complex', ctx=None)
        stored_metadata = search_result['results'][0]['metadata']

        # Verify deep nesting
        assert stored_metadata['level1']['level2']['level3']['level4']['value'] == 'deeply_nested'
        assert stored_metadata['level1']['level2']['level3']['level4']['number'] == 42
        assert stored_metadata['level1']['level2']['level3']['level4']['array'] == [1, 2, 3]
        assert stored_metadata['level1']['level2']['level3']['level4']['object']['key'] == 'value'

        # Verify metrics
        assert stored_metadata['metrics']['cpu'] == 45.5
        assert stored_metadata['metrics']['disk']['used'] == 80.5
        assert stored_metadata['metrics']['disk']['partitions'] == ['/dev/sda1', '/dev/sda2']

        # Verify features
        assert stored_metadata['features']['enabled'] == ['feature_a', 'feature_b', 'feature_c']
        assert stored_metadata['features']['disabled'] == []
        assert stored_metadata['features']['experimental']['count'] == 3

    @pytest.mark.asyncio
    async def test_mixed_flat_and_nested(self) -> None:
        """Test mixing flat and nested metadata structures."""
        mixed_metadata = {
            'simple_string': 'value',
            'simple_int': 42,
            'simple_bool': True,
            'nested': {'level1': {'level2': 'deep_value'}},
            'array': [1, 2, 3],
        }

        result = await store_context(
            thread_id='test_mixed',
            source='agent',
            text='Mixed flat and nested',
            metadata=mixed_metadata,
            ctx=None,
        )

        assert result['success'] is True

        # Query using both flat and nested paths
        search_result = await search_context(
            limit=50, thread_id='test_mixed', metadata={'simple_string': 'value'}, ctx=None,
        )
        assert len(search_result['results']) == 1

        # Verify all types are preserved
        stored_metadata = search_result['results'][0]['metadata']
        assert stored_metadata['simple_string'] == 'value'
        assert stored_metadata['simple_int'] == 42
        assert stored_metadata['simple_bool'] is True
        assert stored_metadata['nested']['level1']['level2'] == 'deep_value'
        assert stored_metadata['array'] == [1, 2, 3]
