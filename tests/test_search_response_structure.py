"""Tests for search tool response structure consistency.

Validates that all search tools return consistent response structures
matching their TypedDict definitions.
"""

from __future__ import annotations

import pytest

from app.server import search_context
from app.server import store_context


class TestSearchContextResponseStructure:
    """Test search_context response structure."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_response_has_results_key(self) -> None:
        """Response must have 'results' key (not 'entries')."""
        # Store a test entry
        await store_context(
            thread_id='test_response_structure',
            source='agent',
            text='Test entry for response structure',
        )

        result = await search_context(thread_id='test_response_structure', limit=10)
        assert 'results' in result
        assert 'entries' not in result  # Verify old key not present

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_response_has_count_field(self) -> None:
        """Response must have 'count' field."""
        # Store a test entry
        await store_context(
            thread_id='test_count_field',
            source='agent',
            text='Test entry for count field',
        )

        result = await search_context(thread_id='test_count_field', limit=10)
        assert 'count' in result
        assert result['count'] == len(result['results'])

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_stats_only_with_explain_query(self) -> None:
        """Stats should only be present when explain_query=True."""
        # Store a test entry
        await store_context(
            thread_id='test_stats_conditional',
            source='agent',
            text='Test entry for stats',
        )

        result_without = await search_context(
            thread_id='test_stats_conditional',
            limit=10,
            explain_query=False,
        )
        assert 'stats' not in result_without

        result_with = await search_context(
            thread_id='test_stats_conditional',
            limit=10,
            explain_query=True,
        )
        assert 'stats' in result_with

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_entry_has_is_truncated_field(self) -> None:
        """Each entry must have is_truncated field."""
        # Store a test entry with long content
        long_text = 'A' * 200  # Longer than 150 chars truncation limit
        await store_context(
            thread_id='test_truncated_field',
            source='agent',
            text=long_text,
        )

        result = await search_context(thread_id='test_truncated_field', limit=10)
        for entry in result['results']:
            assert 'is_truncated' in entry

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_empty_results(self) -> None:
        """Empty search should return correct structure."""
        result = await search_context(thread_id='nonexistent_thread_xyz', limit=10)
        assert 'results' in result
        assert 'count' in result
        assert result['results'] == []
        assert result['count'] == 0


class TestSearchToolsConsistency:
    """Test consistency across search tools."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_uses_results_key(self) -> None:
        """search_context must use 'results' key."""
        await store_context(
            thread_id='consistency_test',
            source='agent',
            text='Test for consistency',
        )

        result = await search_context(thread_id='consistency_test', limit=5)
        assert 'results' in result
        assert 'entries' not in result

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_has_count(self) -> None:
        """search_context must have count field."""
        await store_context(
            thread_id='count_consistency_test',
            source='agent',
            text='Test for count consistency',
        )

        result = await search_context(thread_id='count_consistency_test', limit=5)
        assert 'count' in result
        assert isinstance(result['count'], int)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_explain_query_adds_stats(self) -> None:
        """explain_query=True should add stats to search_context."""
        await store_context(
            thread_id='explain_query_test',
            source='agent',
            text='Test for explain_query',
        )

        result = await search_context(
            thread_id='explain_query_test',
            limit=5,
            explain_query=True,
        )
        assert 'stats' in result

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_multiple_results_count_matches(self) -> None:
        """count field should match actual results length."""
        thread_id = 'multi_results_test'
        # Store multiple entries
        for i in range(5):
            await store_context(
                thread_id=thread_id,
                source='agent',
                text=f'Test entry {i}',
            )

        result = await search_context(thread_id=thread_id, limit=10)
        assert result['count'] == len(result['results'])
        assert result['count'] == 5
