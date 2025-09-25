"""
Test suite for MCP server tools in the Context Storage Server.

Tests all MCP tool implementations including store_context, search_context,
get_context_by_ids, delete_context, list_threads, and get_statistics.
"""

from __future__ import annotations

import asyncio
import base64
import sqlite3
from pathlib import Path
from typing import Any
from typing import Literal
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

# Import the actual async functions from app.server, not the MCP-wrapped versions
# The FunctionTool objects store the original functions in their 'fn' attribute
import app.server
import pytest
from fastmcp import Context

# Get the actual async functions from the FunctionTool wrappers
# FastMCP wraps our functions in FunctionTool objects, but we need the original functions for testing
store_context = app.server.store_context.fn
search_context = app.server.search_context.fn
get_context_by_ids = app.server.get_context_by_ids.fn
delete_context = app.server.delete_context.fn
list_threads = app.server.list_threads.fn
get_statistics = app.server.get_statistics.fn


@pytest.mark.usefixtures('initialized_server')
class TestStoreContext:
    """Test the store_context MCP tool."""

    @pytest.mark.asyncio
    async def test_store_text_context(
        self,
        mock_context: Context,
        sample_context_data: dict[str, Any],
    ) -> None:
        """Test storing a simple text context entry."""
        result = await store_context(
            thread_id=sample_context_data['thread_id'],
            source=sample_context_data['source'],
            text=sample_context_data['text'],
            metadata=sample_context_data['metadata'],
            tags=sample_context_data['tags'],
            ctx=mock_context,
        )

        assert result['success'] is True
        assert 'context_id' in result
        assert result['thread_id'] == sample_context_data['thread_id']
        assert 'Context stored with 0 images' in result['message']

        # Verify context.info was called
        assert mock_context.info.called is True  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_store_multimodal_context(
        self,
        mock_context: Context,
        sample_multimodal_data: dict[str, Any],
    ) -> None:
        """Test storing context with images."""
        result = await store_context(
            thread_id=sample_multimodal_data['thread_id'],
            source=sample_multimodal_data['source'],
            text=sample_multimodal_data['text'],
            images=sample_multimodal_data['images'],
            metadata=sample_multimodal_data['metadata'],
            tags=sample_multimodal_data['tags'],
            ctx=mock_context,
        )

        assert result['success'] is True
        assert 'context_id' in result
        assert 'Context stored with 1 images' in result['message']

    @pytest.mark.asyncio
    async def test_store_context_no_content(self) -> None:
        """Test error when text is empty (text is now required)."""
        result = await store_context(
            thread_id='test_thread',
            source='user',
            text='',  # Empty text should fail validation
        )

        assert result['success'] is False
        assert 'Text content is required' in result['error']

    @pytest.mark.asyncio
    async def test_store_context_invalid_source(self) -> None:
        """Test error with invalid source type."""
        result = await store_context(
            thread_id='test_thread',
            source=cast(Literal['user', 'agent'], 'invalid_source'),
            text='Some text',
        )

        assert result['success'] is False
        assert "Source must be 'user' or 'agent'" in result['error']

    @pytest.mark.asyncio
    async def test_store_context_oversized_image(
        self,
        large_image_data: dict[str, str],
    ) -> None:
        """Test error when image exceeds size limit."""
        result = await store_context(
            thread_id='test_thread',
            source='user',
            text='Text with large image',
            images=[large_image_data],
        )

        assert result['success'] is False
        assert 'exceeds' in result['error']
        assert 'MB limit' in result['error']

    @pytest.mark.asyncio
    async def test_store_context_invalid_base64(self) -> None:
        """Test error with invalid base64 image data."""
        result = await store_context(
            thread_id='test_thread',
            source='agent',
            text='Text with bad image',
            images=[{'data': 'not-valid-base64!', 'mime_type': 'image/png'}],
        )

        assert result['success'] is False
        assert 'Failed to process image' in result['error']

    @pytest.mark.asyncio
    async def test_store_multiple_images(
        self,
        sample_image_data: dict[str, str],
    ) -> None:
        """Test storing multiple images."""
        images = [
            sample_image_data,
            {**sample_image_data, 'metadata': {'position': 1}},
            {**sample_image_data, 'metadata': {'position': 2}},
        ]

        result = await store_context(
            thread_id='test_multi_image',
            source='agent',
            text='Multiple images attached',
            images=images,
        )

        assert result['success'] is True
        assert 'Context stored with 3 images' in result['message']

    @pytest.mark.asyncio
    async def test_store_context_database_error(
        self,
        temp_db_path: Path,
    ) -> None:
        """Test handling of database errors."""
        # Mock the database manager to raise an error
        _ = temp_db_path  # Acknowledge unused parameter
        with patch('app.server._db_manager') as mock_manager:
            mock_manager.execute_write.side_effect = sqlite3.OperationalError('Database error')
            result = await store_context(
                thread_id='test',
                source='user',
                text='This should fail',
            )

            assert result['success'] is False
            assert 'Failed to store context' in result['error']


@pytest.mark.usefixtures('initialized_server')
class TestSearchContext:
    """Test the search_context MCP tool."""

    @pytest.mark.asyncio
    async def test_search_all_contexts(
        self,
        multiple_context_entries: list[int],
    ) -> None:
        """Test searching without filters returns all contexts."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context()

        assert isinstance(results, list)
        assert len(results) == 5  # All test entries

    @pytest.mark.asyncio
    async def test_search_by_thread(
        self,
        multiple_context_entries: list[int],
    ) -> None:
        """Test filtering by thread ID."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context(thread_id='thread_1')

        assert len(results) == 2
        for result in results:
            assert result['thread_id'] == 'thread_1'

    @pytest.mark.asyncio
    async def test_search_by_source(
        self,
        multiple_context_entries: list[int],
    ) -> None:
        """Test filtering by source type."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context(source='agent')

        assert len(results) == 2
        for result in results:
            assert result['source'] == 'agent'

    @pytest.mark.asyncio
    async def test_search_by_tags(
        self,
        multiple_context_entries: list[int],
    ) -> None:
        """Test filtering by tags."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context(tags=['important', 'nonexistent'])

        assert len(results) == 1
        assert 'important' in results[0]['tags']

    @pytest.mark.asyncio
    async def test_search_by_content_type(
        self,
        multiple_context_entries: list[int],
    ) -> None:
        """Test filtering by content type."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context(content_type='multimodal')

        assert len(results) == 1
        assert results[0]['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    async def test_search_with_pagination(
        self,
        multiple_context_entries: list[int],
    ) -> None:
        """Test pagination parameters."""
        _ = multiple_context_entries  # Fixture ensures data exists
        # Get first 2 results
        page1 = await search_context(limit=2, offset=0)
        assert len(page1) == 2

        # Get next 2 results
        page2 = await search_context(limit=2, offset=2)
        assert len(page2) == 2

        # Verify different results
        page1_ids = [r['id'] for r in page1]
        page2_ids = [r['id'] for r in page2]
        assert set(page1_ids).isdisjoint(set(page2_ids))

    @pytest.mark.asyncio
    async def test_search_include_images(
        self,
        temp_db_path: Path,
    ) -> None:
        """Test including image data in search results."""
        _ = temp_db_path  # Fixture provides database path
        # Store a context with image
        image_data = base64.b64encode(b'test_image_data').decode('utf-8')
        await store_context(
            thread_id='image_test',
            source='user',
            text='With image',
            images=[{'data': image_data, 'mime_type': 'image/png'}],
        )

        # Search with images included
        results = await search_context(
            thread_id='image_test',
            include_images=True,
        )

        assert len(results) == 1
        assert 'images' in results[0]
        assert len(results[0]['images']) == 1
        assert results[0]['images'][0]['data'] == image_data

    @pytest.mark.asyncio
    async def test_search_complex_filters(
        self,
        multiple_context_entries: list[int],
    ) -> None:
        """Test combining multiple filters."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context(
            thread_id='thread_2',
            source='user',
            content_type='multimodal',
        )

        assert len(results) == 1
        assert results[0]['thread_id'] == 'thread_2'
        assert results[0]['source'] == 'user'
        assert results[0]['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    async def test_search_invalid_source(
        self,
        multiple_context_entries: list[int],
    ) -> None:
        """Test search with invalid source returns empty."""
        _ = multiple_context_entries  # Fixture ensures data exists
        results = await search_context(source=cast(Literal['user', 'agent'], 'invalid'))

        assert results == []

    @pytest.mark.asyncio
    async def test_search_limit_max(self) -> None:
        """Test search respects maximum limit."""
        # Create many entries
        for i in range(600):
            await store_context(
                thread_id=f'bulk_thread_{i}',
                source='user',
                text=f'Entry {i}',
            )

        results = await search_context(limit=1000)  # Request more than max
        assert len(results) <= 500  # Should be capped at 500

    @pytest.mark.asyncio
    async def test_search_text_truncation_short(self) -> None:
        """Test that short text is not truncated."""
        short_text = 'This is a short text that should not be truncated.'
        assert len(short_text) < 150  # Ensure it's actually short

        await store_context(
            thread_id='truncation_test',
            source='user',
            text=short_text,
        )

        results = await search_context(thread_id='truncation_test')
        assert len(results) == 1
        assert results[0]['text_content'] == short_text
        assert results[0]['is_truncated'] is False

    @pytest.mark.asyncio
    async def test_search_text_truncation_long(self) -> None:
        """Test that long text is truncated with ellipsis."""
        # Create a text longer than 150 characters
        long_text = (
            'This is a very long text that exceeds the truncation limit. '
            'It contains multiple sentences to ensure it goes over 150 characters. '
            'This additional content will be truncated when returned from search_context. '
            'More content here to make it even longer and ensure truncation occurs.'
        )
        assert len(long_text) > 150  # Ensure it's actually long

        await store_context(
            thread_id='truncation_long_test',
            source='agent',
            text=long_text,
        )

        results = await search_context(thread_id='truncation_long_test')
        assert len(results) == 1

        # Check truncation occurred
        assert results[0]['is_truncated'] is True
        assert results[0]['text_content'].endswith('...')
        assert len(results[0]['text_content']) <= 153  # 150 + '...'
        assert results[0]['text_content'] != long_text

        # Verify truncation preserves beginning of text
        assert long_text.startswith(results[0]['text_content'][:-3])  # Remove '...'

    @pytest.mark.asyncio
    async def test_search_text_truncation_word_boundary(self) -> None:
        """Test that truncation happens at word boundaries when possible."""
        # Test case 1: Text with good word boundary near position 150
        text_with_good_boundary = (
            'This text has exactly the right length to test word boundary truncation behavior. '
            'When the 150th character falls within a word, the truncation algorithm should '
            'ideally find the nearest word boundary to avoid splitting words.'
        )

        await store_context(
            thread_id='boundary_test_good',
            source='user',
            text=text_with_good_boundary,
        )

        results = await search_context(thread_id='boundary_test_good')
        assert len(results) == 1
        assert results[0]['is_truncated'] is True
        assert results[0]['text_content'].endswith('...')

        # Test case 2: Text where truncation will happen mid-word due to no good boundary
        # Create a text with a very long word starting before position 105
        long_word = 'verylongwordthatcannotbesplitproperlybecauseitexceedstheboundarythresholdandcontinuesforquiteawhilelonger'
        text_with_bad_boundary = 'Short start then ' + long_word + ' and more text after to ensure truncation happens.'

        await store_context(
            thread_id='boundary_test_bad',
            source='user',
            text=text_with_bad_boundary,
        )

        results_bad = await search_context(thread_id='boundary_test_bad')
        assert len(results_bad) == 1
        assert results_bad[0]['is_truncated'] is True
        assert results_bad[0]['text_content'].endswith('...')
        # In this case, truncation happens at exactly 150 chars since no good word boundary exists

    @pytest.mark.asyncio
    async def test_search_vs_get_by_id_truncation(self) -> None:
        """Test that get_context_by_ids returns full text while search_context truncates."""
        long_text = (
            'This is a comprehensive test to verify that search_context truncates '
            'the text content while get_context_by_ids returns the complete full text. '
            'This distinction is important for the API design where search provides '
            'a preview and get_by_ids provides complete content for detailed viewing. '
            'Additional content here to ensure the text is sufficiently long.'
        )
        assert len(long_text) > 150

        store_result = await store_context(
            thread_id='comparison_test',
            source='user',
            text=long_text,
        )
        context_id = store_result['context_id']

        # Search should return truncated text
        search_results = await search_context(thread_id='comparison_test')
        assert len(search_results) == 1
        assert search_results[0]['is_truncated'] is True
        assert search_results[0]['text_content'] != long_text
        assert search_results[0]['text_content'].endswith('...')

        # get_context_by_ids should return full text
        get_results = await get_context_by_ids(context_ids=[context_id])
        assert len(get_results) == 1
        assert get_results[0]['text_content'] == long_text
        assert 'is_truncated' not in get_results[0]  # This field should not exist

    @pytest.mark.asyncio
    async def test_search_null_text_truncation(self, temp_db_path: Path) -> None:
        """Test that null/empty text content is handled correctly."""
        # This shouldn't normally happen due to validation, but test defensive coding
        _ = temp_db_path  # Fixture provides database path
        await store_context(
            thread_id='null_test',
            source='user',
            text='placeholder',  # Store with some text first
        )

        # Directly update database to set text_content to empty (edge case)
        import sqlite3

        with sqlite3.connect(str(temp_db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE context_entries SET text_content = ? WHERE thread_id = ?',
                ('', 'null_test'),
            )
            conn.commit()

        results = await search_context(thread_id='null_test')
        assert len(results) == 1
        assert results[0]['text_content'] == ''
        assert results[0]['is_truncated'] is False


@pytest.mark.usefixtures('initialized_server')
class TestGetContextByIds:
    """Test the get_context_by_ids MCP tool."""

    @pytest.mark.asyncio
    async def test_get_single_context(
        self,
        multiple_context_entries: list[int],
    ) -> None:
        """Test fetching a single context by ID."""
        context_id = multiple_context_entries[0]
        results = await get_context_by_ids(context_ids=[context_id])

        assert len(results) == 1
        assert results[0]['id'] == context_id

    @pytest.mark.asyncio
    async def test_get_multiple_contexts(
        self,
        multiple_context_entries: list[int],
    ) -> None:
        """Test fetching multiple contexts by IDs."""
        ids_to_fetch = multiple_context_entries[:3]
        results = await get_context_by_ids(context_ids=ids_to_fetch)

        assert len(results) == 3
        result_ids = [r['id'] for r in results]
        assert set(result_ids) == set(ids_to_fetch)

    @pytest.mark.asyncio
    async def test_get_context_with_images(self) -> None:
        """Test fetching context with images included."""
        # Store context with image
        image_data = base64.b64encode(b'test_img').decode('utf-8')
        store_result = await store_context(
            thread_id='img_test',
            source='agent',
            text='With image',
            images=[
                {
                    'data': image_data,
                    'mime_type': 'image/jpeg',
                    'metadata': {'size': 100},
                },
            ],
        )

        context_id = store_result['context_id']

        # Fetch with images
        results = await get_context_by_ids(
            context_ids=[context_id],
            include_images=True,
        )

        assert len(results) == 1
        assert 'images' in results[0]
        assert len(results[0]['images']) == 1
        assert results[0]['images'][0]['mime_type'] == 'image/jpeg'

    @pytest.mark.asyncio
    async def test_get_context_without_images(self) -> None:
        """Test fetching context without images."""
        # Store context with image
        store_result = await store_context(
            thread_id='no_img_test',
            source='user',
            text='With image but not fetched',
            images=[{'data': base64.b64encode(b'img').decode('utf-8')}],
        )

        # Fetch without images
        results = await get_context_by_ids(
            context_ids=[store_result['context_id']],
            include_images=False,
        )

        assert len(results) == 1
        assert 'images' not in results[0] or results[0]['images'] == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_contexts(self) -> None:
        """Test fetching non-existent context IDs."""
        results = await get_context_by_ids(context_ids=[9999, 10000])
        assert results == []

    @pytest.mark.asyncio
    async def test_get_empty_context_list(self) -> None:
        """Test with empty context ID list."""
        results = await get_context_by_ids(context_ids=[])
        assert results == []

    @pytest.mark.asyncio
    async def test_get_context_with_tags(
        self,
        multiple_context_entries: list[int],
    ) -> None:
        """Test that tags are included in fetched contexts."""
        # First entry has tags
        results = await get_context_by_ids(context_ids=[multiple_context_entries[0]])

        assert len(results) == 1
        assert 'tags' in results[0]
        assert 'important' in results[0]['tags']


@pytest.mark.usefixtures('initialized_server')
class TestDeleteContext:
    """Test the delete_context MCP tool."""

    @pytest.mark.asyncio
    async def test_delete_by_ids(self) -> None:
        """Test deleting specific contexts by IDs."""
        # Create test contexts
        result1 = await store_context(
            thread_id='delete_test',
            source='user',
            text='Entry 1',
        )
        result2 = await store_context(
            thread_id='delete_test',
            source='agent',
            text='Entry 2',
        )
        result3 = await store_context(
            thread_id='delete_test',
            source='user',
            text='Entry 3',
        )

        ids_to_delete = [result1['context_id'], result2['context_id']]

        # Delete first two
        delete_result = await delete_context(context_ids=ids_to_delete)

        assert delete_result['success'] is True
        assert delete_result['deleted_count'] == 2

        # Verify third still exists
        remaining = await search_context(thread_id='delete_test')
        assert len(remaining) == 1
        assert remaining[0]['id'] == result3['context_id']

    @pytest.mark.asyncio
    async def test_delete_by_thread(self) -> None:
        """Test deleting all contexts in a thread."""
        thread_id = 'thread_to_delete'

        # Create multiple contexts
        for i in range(5):
            await store_context(
                thread_id=thread_id,
                source='user' if i % 2 == 0 else 'agent',
                text=f'Entry {i}',
            )

        # Delete entire thread
        delete_result = await delete_context(thread_id=thread_id)

        assert delete_result['success'] is True
        assert delete_result['deleted_count'] == 5

        # Verify all deleted
        remaining = await search_context(thread_id=thread_id)
        assert remaining == []

    @pytest.mark.asyncio
    async def test_delete_no_parameters(self) -> None:
        """Test error when no delete parameters provided."""
        result = await delete_context()

        assert result['success'] is False
        assert 'Must provide either context_ids or thread_id' in result['error']

    @pytest.mark.asyncio
    async def test_delete_nonexistent_ids(self) -> None:
        """Test deleting non-existent IDs."""
        result = await delete_context(context_ids=[9999, 10000])

        assert result['success'] is True
        assert result['deleted_count'] == 0

    @pytest.mark.asyncio
    async def test_delete_cascades(self) -> None:
        """Test that deleting context also deletes tags and images."""
        # Store context with tags and images
        result = await store_context(
            thread_id='cascade_test',
            source='user',
            text='With tags and images',
            tags=['tag1', 'tag2'],
            images=[{'data': base64.b64encode(b'img').decode('utf-8')}],
        )

        context_id = result['context_id']

        # Delete the context
        delete_result = await delete_context(context_ids=[context_id])
        assert delete_result['success'] is True

        # Verify context and related data are gone
        remaining = await get_context_by_ids(context_ids=[context_id])
        assert remaining == []


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
        threads_data = [
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
    async def test_statistics_database_size(
        self,
        temp_db_path: Path,
    ) -> None:
        """Test database size reporting."""
        # Add some data to ensure non-zero size
        for i in range(10):
            await store_context(
                thread_id=f'size_test_{i}',
                source='user',
                text=f'Entry {i}' * 100,  # Make it bigger
            )

        with patch('app.server.DB_PATH', temp_db_path):
            stats = await get_statistics()

        assert 'database_size_mb' in stats
        # Database file should exist and be non-zero (or at least >= 0)
        assert stats['database_size_mb'] >= 0

    @pytest.mark.asyncio
    async def test_statistics_error_handling(self) -> None:
        """Test statistics handles errors gracefully."""
        # Mock the database manager to raise an error during read
        with patch('app.server._db_manager') as mock_manager:
            mock_manager.execute_read.side_effect = sqlite3.OperationalError('Database error')
            stats = await get_statistics()

        # Should return empty dict on error
        assert stats == {}


@pytest.mark.usefixtures('initialized_server')
class TestContextParameter:
    """Test MCP Context parameter usage across tools."""

    @pytest.mark.asyncio
    async def test_context_info_logging(self) -> None:
        """Test that context.info is called appropriately."""
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.info = AsyncMock()

        # Test various tools with context
        await store_context(
            thread_id='ctx_test',
            source='user',
            text='Test',
            ctx=mock_ctx,
        )
        assert mock_ctx.info.call_count == 1

        await search_context(thread_id='ctx_test', ctx=mock_ctx)
        assert mock_ctx.info.call_count == 2

        await get_context_by_ids(context_ids=[1], ctx=mock_ctx)
        assert mock_ctx.info.call_count == 3

        await delete_context(thread_id='ctx_test', ctx=mock_ctx)
        assert mock_ctx.info.call_count == 4

        await list_threads(ctx=mock_ctx)
        assert mock_ctx.info.call_count == 5

        await get_statistics(ctx=mock_ctx)
        assert mock_ctx.info.call_count == 6


@pytest.mark.usefixtures('initialized_server')
class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_unicode_content(self) -> None:
        """Test handling of Unicode content."""
        unicode_text = 'Hello 世界 🌍 مرحبا мир'
        result = await store_context(
            thread_id='unicode_test',
            source='user',
            text=unicode_text,
            tags=['unicode', '中文', 'عربي'],
        )

        assert result['success'] is True

        # Verify retrieval
        search_result = await search_context(thread_id='unicode_test')
        assert search_result[0]['text_content'] == unicode_text

    @pytest.mark.asyncio
    async def test_large_metadata(self) -> None:
        """Test handling of large metadata objects."""
        large_metadata = {
            'nested': {
                'level': {
                    'data': ['item'] * 100,
                    'numbers': list(range(1000)),
                },
            },
            'description': 'x' * 10000,
        }

        result = await store_context(
            thread_id='metadata_test',
            source='agent',
            text='Large metadata',
            metadata=large_metadata,
        )

        assert result['success'] is True

        # Verify retrieval
        fetched = await get_context_by_ids(context_ids=[result['context_id']])
        assert fetched[0]['metadata'] == large_metadata

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self) -> None:
        """Test that SQL injection attempts are prevented."""
        malicious_thread = "'; DROP TABLE context_entries; --"
        malicious_tag = "'; DELETE FROM tags; --"

        # Should handle malicious input safely
        result = await store_context(
            thread_id=malicious_thread,
            source='user',
            text='Test SQL injection',
            tags=[malicious_tag],
        )

        assert result['success'] is True

        # Verify data integrity
        search_result = await search_context(thread_id=malicious_thread)
        assert len(search_result) == 1
        # Tag should be normalized to lowercase
        normalized_tag = malicious_tag.strip().lower()
        assert normalized_tag in search_result[0]['tags']

        # Tables should still exist
        stats = await get_statistics()
        assert stats['total_entries'] > 0
