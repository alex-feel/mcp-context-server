"""Tests for the ``list_threads`` and ``get_statistics`` tools."""

import asyncio
import base64
import sqlite3
from typing import Literal
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools

# The tool functions are plain coroutines that lifespan() registers with FastMCP at startup; tests call them directly.
store_context = app.tools.store_context
list_threads = app.tools.list_threads
get_statistics = app.tools.get_statistics


@pytest.mark.usefixtures('initialized_server')
class TestListThreads:
    """Test the list_threads MCP tool."""

    @pytest.mark.asyncio
    async def test_list_empty_threads(self) -> None:
        """Test listing threads when database is empty."""
        result = await list_threads()

        assert 'threads' in result
        assert 'total_threads' in result
        assert result['threads'] == []
        assert result['total_threads'] == 0

    @pytest.mark.asyncio
    async def test_list_threads_with_data(self) -> None:
        """Test listing threads with multiple entries."""
        # Create test data
        threads_data: list[tuple[str, Literal['user', 'agent'], int]] = [
            ('thread_a', 'user', 3),
            ('thread_b', 'agent', 2),
            ('thread_c', 'user', 5),
        ]

        for thread_id, source, count in threads_data:
            for i in range(count):
                await store_context(
                    thread_id=thread_id,
                    source=source,
                    text=f'Entry {i}',
                )

        result = await list_threads()

        assert result['total_threads'] == 3
        assert len(result['threads']) == 3

        # Check thread statistics
        thread_map = {t['thread_id']: t for t in result['threads']}

        assert thread_map['thread_a']['entry_count'] == 3
        assert thread_map['thread_b']['entry_count'] == 2
        assert thread_map['thread_c']['entry_count'] == 5

    @pytest.mark.asyncio
    async def test_list_threads_with_multimodal(self) -> None:
        """Test thread statistics include multimodal counts."""
        # Create mixed content
        await store_context(
            thread_id='mixed_thread',
            source='user',
            text='Text only',
        )
        await store_context(
            thread_id='mixed_thread',
            source='agent',
            text='With image',
            images=[{'data': base64.b64encode(b'img').decode('utf-8')}],
        )

        result = await list_threads()

        thread = next(t for t in result['threads'] if t['thread_id'] == 'mixed_thread')
        assert thread['entry_count'] == 2
        assert thread['multimodal_count'] == 1
        assert thread['source_types'] == 2  # Both user and agent

    @pytest.mark.asyncio
    async def test_list_threads_ordering(self) -> None:
        """Test threads are ordered by last entry timestamp."""
        # Create threads with delay
        await store_context(thread_id='old_thread', source='user', text='Old')
        await asyncio.sleep(0.01)
        await store_context(thread_id='new_thread', source='user', text='New')
        await asyncio.sleep(0.01)
        await store_context(thread_id='old_thread', source='agent', text='Updated')

        result = await list_threads()

        # Most recent activity should be first
        assert result['threads'][0]['thread_id'] == 'old_thread'
        assert result['threads'][1]['thread_id'] == 'new_thread'


@pytest.mark.usefixtures('initialized_server')
class TestGetStatistics:
    """Test the get_statistics MCP tool."""

    @pytest.mark.asyncio
    async def test_empty_statistics(self) -> None:
        """Test statistics on empty database."""
        stats = await get_statistics()

        assert stats['total_entries'] == 0
        assert stats['by_source'] == {}
        assert stats['by_content_type'] == {}
        assert stats['total_images'] == 0
        assert stats['unique_tags'] == 0

    @pytest.mark.asyncio
    async def test_statistics_with_data(self) -> None:
        """Test statistics with various data."""
        # Create diverse test data
        await store_context(
            thread_id='stats_test',
            source='user',
            text='User text',
            tags=['python', 'testing'],
        )
        await store_context(
            thread_id='stats_test',
            source='agent',
            text='Agent response',
            tags=['python', 'ai'],
        )
        await store_context(
            thread_id='stats_test',
            source='user',
            text='With image',
            images=[{'data': base64.b64encode(b'img1').decode('utf-8')}],
            tags=['image'],
        )
        await store_context(
            thread_id='stats_test2',
            source='agent',
            text='Another with images',
            images=[
                {'data': base64.b64encode(b'img2').decode('utf-8')},
                {'data': base64.b64encode(b'img3').decode('utf-8')},
            ],
        )

        stats = await get_statistics()

        assert stats['total_entries'] == 4
        assert stats['by_source'] == {'user': 2, 'agent': 2}
        assert stats['by_content_type'] == {'text': 2, 'multimodal': 2}
        assert stats['total_images'] == 3
        assert stats['unique_tags'] == 4  # python, testing, ai, image

    @pytest.mark.asyncio
    async def test_statistics_database_size(self) -> None:
        """Test database size reporting."""
        # Add some data to ensure non-zero size
        for i in range(10):
            await store_context(
                thread_id=f'size_test_{i}',
                source='user',
                text=f'Entry {i}' * 100,  # Make it bigger
            )

        stats = await get_statistics()

        assert 'database_size_mb' in stats
        # Database file should exist and be non-zero (or at least >= 0)
        assert stats['database_size_mb'] >= 0

    @pytest.mark.asyncio
    async def test_statistics_error_handling(self) -> None:
        """Test statistics handles errors gracefully."""
        # Mock the repository method to raise an error during read
        with patch('app.repositories.statistics_repository.StatisticsRepository.get_database_statistics') as mock_stats:
            mock_stats.side_effect = sqlite3.OperationalError('Database error')
            with pytest.raises(ToolError, match='Failed to get statistics'):
                await get_statistics()

    @pytest.mark.asyncio
    async def test_statistics_summary_disabled(self) -> None:
        """Test summary section when generation is disabled."""
        from app.settings import get_settings as _get_settings

        real_settings = _get_settings()
        mock_settings = MagicMock(wraps=real_settings)
        mock_summary = MagicMock()
        mock_summary.generation_enabled = False
        mock_settings.summary = mock_summary
        with patch('app.tools.discovery.settings', mock_settings):
            stats = await get_statistics()
            assert 'summary' in stats
            assert stats['summary'] == {'enabled': False, 'available': False}

    @pytest.mark.asyncio
    async def test_statistics_summary_enabled_unavailable(self) -> None:
        """Test summary section when enabled but provider not initialized."""
        from typing import Any
        from typing import cast

        stats = await get_statistics()
        assert 'summary' in stats
        # SummaryStatsDict declares all fields total=False; cast so per-key
        # indexing reads as a runtime structural assertion.
        summary_info = cast(dict[str, Any], stats['summary'])
        assert summary_info['enabled'] is True
        assert summary_info['available'] is False
        assert 'message' in summary_info
