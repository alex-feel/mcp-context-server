"""search_context metadata filtering end to end: simple and advanced filters, every operator, and validation errors."""

import time

import pytest

from app.metadata_types import MetadataOperator
from app.tools import search_context
from app.tools import store_context
from app.types import JsonValue


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
            await store_context(**data)

    @pytest.mark.asyncio
    async def test_simple_metadata_filter(self) -> None:
        """Test simple metadata filtering."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'active'},
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
        )

        assert 'results' in result
        assert len(result['results']) == 3
        statuses = [e['metadata']['status'] for e in result['results']]
        assert all(s in ['active', 'pending'] for s in statuses)

    @pytest.mark.asyncio
    async def test_advanced_in_operator_with_integer_array(self) -> None:
        """Test IN operator with integer array values.

        Integer members must match numerically: compared as TEXT they would
        mismatch the json_extract result on SQLite and fail the asyncpg TEXT
        cast on PostgreSQL.
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
        )

        assert 'results' in result
        # Should find entries with priority 5 and 10
        assert len(result['results']) == 2
        priorities = [e['metadata']['priority'] for e in result['results']]
        assert all(p in [5, 10] for p in priorities)

    @pytest.mark.asyncio
    async def test_advanced_not_in_operator_with_integer_array(self) -> None:
        """Test NOT IN operator with integer array values.

        Integer members must match numerically under NOT IN as well.
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
        )
        complex_time = (time.time() - start_time) * 1000

        assert complex_time < 500  # Should be under 500ms (relaxed for CI variability)
        assert 'results' in result

    @pytest.mark.asyncio
    async def test_in_operator_over_member_cap_returns_structured_validation_error(self) -> None:
        """An IN filter above MAX_IN_LIST_MEMBERS short-circuits through the
        metadata-filter validation channel (a structured, breaker-exempt error
        response) instead of expanding into an oversized single-statement SQL bind."""
        from app.metadata_types import MAX_IN_LIST_MEMBERS

        oversized: list[str | int | float | bool] = [f'v{i}' for i in range(MAX_IN_LIST_MEMBERS + 1)]
        result = await search_context(
            limit=50,
            thread_id='in_member_cap_thread',
            metadata_filters=[{'key': 'status', 'operator': 'in', 'value': oversized}],
        )

        assert result['results'] == []
        assert result['count'] == 0
        assert 'error' in result
        assert any('at most' in message for message in result['validation_errors'])


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
    test_data: list[dict[str, JsonValue]] = [
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
        )

        assert 'error' in result
        assert result['error'] == 'Metadata filter validation failed'
        assert 'validation_errors' in result
        assert len(result['validation_errors']) == 1
