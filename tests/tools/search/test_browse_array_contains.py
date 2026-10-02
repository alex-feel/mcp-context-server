"""search_context ARRAY_CONTAINS filtering end to end over array, scalar, object, number and null fields."""

import time

import pytest

from app.tools import search_context
from app.tools import store_context


@pytest.mark.integration
@pytest.mark.usefixtures('initialized_server')
class TestArrayContainsOperator:
    """Tests for the ARRAY_CONTAINS operator."""

    test_thread_id: str

    async def _setup_test_data(self) -> None:
        """Set up test data with array metadata fields."""
        self.test_thread_id = f'test_array_contains_{int(time.time() * 1000)}'

        # Entry with string array
        await store_context(
            thread_id=self.test_thread_id,
            source='agent',
            text='Python and FastAPI project',
            metadata={
                'technologies': ['python', 'fastapi', 'postgresql'],
                'tags': ['backend', 'api', 'production'],
            },
        )

        # Entry with different technologies
        await store_context(
            thread_id=self.test_thread_id,
            source='agent',
            text='JavaScript frontend',
            metadata={
                'technologies': ['javascript', 'react', 'typescript'],
                'tags': ['frontend', 'ui'],
            },
        )

        # Entry with numeric array
        await store_context(
            thread_id=self.test_thread_id,
            source='agent',
            text='Priority levels test',
            metadata={
                'priority_levels': [1, 3, 5, 7, 9],
                'scores': [85.5, 90.0, 78.3],
            },
        )

        # Entry with nested array
        await store_context(
            thread_id=self.test_thread_id,
            source='agent',
            text='Nested references',
            metadata={
                'references': {
                    'context_ids': [100, 200, 300],
                    'youtrack': ['AI-100', 'AI-200'],
                },
            },
        )

    @pytest.mark.asyncio
    async def test_array_contains_string_value(self) -> None:
        """Test array_contains with string value."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'technologies', 'operator': 'array_contains', 'value': 'python'},
            ],
        )

        assert len(result['results']) == 1
        assert 'Python and FastAPI' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_case_insensitive(self) -> None:
        """Test array_contains with case-insensitive string matching."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'technologies', 'operator': 'array_contains', 'value': 'PYTHON', 'case_sensitive': False},
            ],
        )

        assert len(result['results']) == 1
        assert 'Python and FastAPI' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_case_sensitive_no_match(self) -> None:
        """Test array_contains with case-sensitive string (no match expected)."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'technologies', 'operator': 'array_contains', 'value': 'PYTHON', 'case_sensitive': True},
            ],
        )

        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_array_contains_integer_value(self) -> None:
        """Test array_contains with integer value."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'priority_levels', 'operator': 'array_contains', 'value': 5},
            ],
        )

        assert len(result['results']) == 1
        assert 'Priority levels' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_float_value(self) -> None:
        """Test array_contains with float value."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'scores', 'operator': 'array_contains', 'value': 90.0},
            ],
        )

        assert len(result['results']) == 1
        assert 'Priority levels' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_nested_path(self) -> None:
        """Test array_contains with nested JSON path."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'references.context_ids', 'operator': 'array_contains', 'value': 200},
            ],
        )

        assert len(result['results']) == 1
        assert 'Nested references' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_nested_string_array(self) -> None:
        """Test array_contains with nested string array."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'references.youtrack', 'operator': 'array_contains', 'value': 'AI-100'},
            ],
        )

        assert len(result['results']) == 1
        assert 'Nested references' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_no_match(self) -> None:
        """Test array_contains returns empty when element not found."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'technologies', 'operator': 'array_contains', 'value': 'rust'},
            ],
        )

        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_array_contains_combined_with_other_filters(self) -> None:
        """Test array_contains combined with other metadata filters."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'technologies', 'operator': 'array_contains', 'value': 'python'},
                {'key': 'tags', 'operator': 'array_contains', 'value': 'production'},
            ],
        )

        assert len(result['results']) == 1
        assert 'Python and FastAPI' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_non_existent_field_returns_empty(self) -> None:
        """Test array_contains on non-existent field returns empty (graceful handling)."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'nonexistent', 'operator': 'array_contains', 'value': 'test'},
            ],
        )

        # Should return empty, not error
        assert 'results' in result
        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_array_contains_scalar_field_returns_empty(self) -> None:
        """Test array_contains on scalar string field returns empty (graceful handling, not error).

        PostgreSQL jsonb_array_elements_text() raises "cannot extract elements
        from a scalar" on a non-array field; the documented behavior is an empty
        result, not an error.
        """
        test_thread_id = f'test_array_contains_scalar_{int(time.time() * 1000)}'
        await store_context(
            thread_id=test_thread_id,
            source='agent',
            text='Entry with scalar category',
            metadata={
                'category': 'backend',  # Scalar string, NOT an array
                'technologies': ['python', 'fastapi'],  # This IS an array
            },
        )

        # This should return empty results, NOT throw an error
        result = await search_context(
            thread_id=test_thread_id,
            metadata_filters=[
                {'key': 'category', 'operator': 'array_contains', 'value': 'backend'},
            ],
        )

        # Should return empty results, not error
        assert 'results' in result
        assert len(result['results']) == 0

        # Verify the array field still works correctly
        result2 = await search_context(
            thread_id=test_thread_id,
            metadata_filters=[
                {'key': 'technologies', 'operator': 'array_contains', 'value': 'python'},
            ],
        )
        assert len(result2['results']) == 1

    @pytest.mark.asyncio
    async def test_array_contains_object_field_returns_empty(self) -> None:
        """Test array_contains on object field returns empty (graceful handling)."""
        test_thread_id = f'test_array_contains_object_{int(time.time() * 1000)}'
        await store_context(
            thread_id=test_thread_id,
            source='agent',
            text='Entry with object config field',
            metadata={
                'config': {'timeout': 30, 'retries': 3},  # Object, NOT an array
            },
        )

        result = await search_context(
            thread_id=test_thread_id,
            metadata_filters=[
                {'key': 'config', 'operator': 'array_contains', 'value': 30},
            ],
        )

        assert 'results' in result
        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_array_contains_number_field_returns_empty(self) -> None:
        """Test array_contains on number field returns empty (graceful handling)."""
        test_thread_id = f'test_array_contains_number_{int(time.time() * 1000)}'
        await store_context(
            thread_id=test_thread_id,
            source='agent',
            text='Entry with number priority field',
            metadata={
                'priority': 5,  # Number scalar, NOT an array
            },
        )

        result = await search_context(
            thread_id=test_thread_id,
            metadata_filters=[
                {'key': 'priority', 'operator': 'array_contains', 'value': 5},
            ],
        )

        assert 'results' in result
        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_array_contains_null_field_returns_empty(self) -> None:
        """Test array_contains on null field returns empty (graceful handling)."""
        test_thread_id = f'test_array_contains_null_{int(time.time() * 1000)}'
        await store_context(
            thread_id=test_thread_id,
            source='agent',
            text='Entry with null field',
            metadata={
                'tags': None,  # Explicit null, NOT an array
            },
        )

        result = await search_context(
            thread_id=test_thread_id,
            metadata_filters=[
                {'key': 'tags', 'operator': 'array_contains', 'value': 'test'},
            ],
        )

        assert 'results' in result
        assert len(result['results']) == 0
