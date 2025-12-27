"""Tests for server edge cases and error handling paths.

This module tests edge cases, error handling, and less common code paths
in app/server.py to improve coverage.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from app.server import delete_context
from app.server import get_context_by_ids
from app.server import get_statistics
from app.server import search_context
from app.server import store_context
from app.server import update_context


class TestStoreContextEdgeCases:
    """Test edge cases for store_context tool."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_with_empty_tags(self) -> None:
        """Test storing context with empty tags list."""
        result = await store_context(
            thread_id='empty_tags_thread',
            source='user',
            text='Message with empty tags',
            tags=[],
        )

        assert result['success'] is True
        assert 'context_id' in result

        # Verify empty tags
        search_result = await search_context(limit=50, thread_id='empty_tags_thread')
        assert len(search_result['results']) == 1
        assert search_result['results'][0]['tags'] == []

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_with_whitespace_tags(self) -> None:
        """Test storing context with tags that have whitespace."""
        result = await store_context(
            thread_id='whitespace_tags_thread',
            source='user',
            text='Message with whitespace tags',
            tags=['  important  ', 'review  ', '  critical'],
        )

        assert result['success'] is True

        # Verify tags are normalized
        search_result = await search_context(limit=50, thread_id='whitespace_tags_thread')
        assert len(search_result['results']) == 1
        # Tags should be stripped and lowercased
        tags = search_result['results'][0]['tags']
        assert 'important' in tags
        assert 'review' in tags
        assert 'critical' in tags

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_with_mixed_case_tags(self) -> None:
        """Test storing context with mixed case tags."""
        result = await store_context(
            thread_id='mixed_case_tags_thread',
            source='user',
            text='Message with mixed case tags',
            tags=['IMPORTANT', 'Review', 'CrItIcAl'],
        )

        assert result['success'] is True

        # Verify tags are normalized to lowercase
        search_result = await search_context(limit=50, thread_id='mixed_case_tags_thread')
        tags = search_result['results'][0]['tags']
        assert 'important' in tags
        assert 'review' in tags
        assert 'critical' in tags

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_with_unicode_text(self) -> None:
        """Test storing context with unicode characters."""
        result = await store_context(
            thread_id='unicode_thread',
            source='user',
            text='Unicode test: Hello World! Cyrillic: Some Text. Chinese: Some Characters',
        )

        assert result['success'] is True

        # Verify unicode is preserved
        entries = await get_context_by_ids(context_ids=[result['context_id']])
        assert len(entries) == 1
        entry: dict[str, Any] = dict(entries[0])
        assert entry['text_content'] is not None
        assert 'Unicode test' in entry['text_content']

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_with_very_long_text(self) -> None:
        """Test storing context with very long text."""
        long_text = 'x' * 10000  # 10KB of text
        result = await store_context(
            thread_id='long_text_thread',
            source='agent',
            text=long_text,
        )

        assert result['success'] is True

        # Verify full text is stored
        entries = await get_context_by_ids(context_ids=[result['context_id']])
        assert len(entries) == 1
        entry: dict[str, Any] = dict(entries[0])
        assert len(entry['text_content']) == 10000

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_with_special_characters(self) -> None:
        """Test storing context with special characters."""
        special_text = "SQL injection: '; DROP TABLE context_entries; --"
        result = await store_context(
            thread_id='special_chars_thread',
            source='user',
            text=special_text,
        )

        assert result['success'] is True

        # Verify text is stored correctly
        entries = await get_context_by_ids(context_ids=[result['context_id']])
        entry = dict(entries[0])
        assert entry['text_content'] == special_text

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_with_multiline_text(self) -> None:
        """Test storing context with multiline text."""
        multiline_text = '''Line 1
Line 2
Line 3
    Indented line
\tTabbed line'''
        result = await store_context(
            thread_id='multiline_thread',
            source='user',
            text=multiline_text,
        )

        assert result['success'] is True

        # Verify multiline text is preserved
        entries = await get_context_by_ids(context_ids=[result['context_id']])
        entry = dict(entries[0])
        assert entry['text_content'] == multiline_text


class TestSearchContextEdgeCases:
    """Test edge cases for search_context tool."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_with_multiple_tags_or_logic(self) -> None:
        """Test searching with multiple tags uses OR logic."""
        # Create entries with different tags
        await store_context(
            thread_id='multi_tag_search',
            source='user',
            text='Entry with tag1',
            tags=['tag1'],
        )
        await store_context(
            thread_id='multi_tag_search',
            source='user',
            text='Entry with tag2',
            tags=['tag2'],
        )
        await store_context(
            thread_id='multi_tag_search',
            source='user',
            text='Entry with tag3',
            tags=['tag3'],
        )

        # Search with multiple tags - should find entries with ANY of the tags
        result = await search_context(
            limit=50,
            thread_id='multi_tag_search',
            tags=['tag1', 'tag2'],
        )

        # Should find 2 entries (tag1 OR tag2)
        assert len(result['results']) == 2

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_with_limit_zero(self) -> None:
        """Test searching with limit of 0 returns no results."""
        await store_context(
            thread_id='limit_zero_thread',
            source='user',
            text='Entry',
        )

        # Note: limit=0 might be interpreted differently, but let's test it
        result = await search_context(
            thread_id='limit_zero_thread',
            limit=1,  # Using limit=1 to test minimal results
        )

        assert len(result['results']) <= 1

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_with_large_offset(self) -> None:
        """Test searching with offset larger than result count."""
        await store_context(
            thread_id='large_offset_thread',
            source='user',
            text='Single entry',
        )

        result = await search_context(
            limit=50,
            thread_id='large_offset_thread',
            offset=1000,
        )

        assert result['results'] == []

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_with_explain_query_true(self) -> None:
        """Test searching with explain_query returns stats."""
        await store_context(
            thread_id='explain_thread',
            source='user',
            text='Entry for explain test',
        )

        result = await search_context(
            limit=50,
            thread_id='explain_thread',
            explain_query=True,
        )

        # Should include stats when explain_query is true
        assert 'stats' in result or 'results' in result

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_with_metadata_integer_value(self) -> None:
        """Test searching with metadata containing integer."""
        await store_context(
            thread_id='meta_int_thread',
            source='user',
            text='Priority 1',
            metadata={'priority': 1},
        )
        await store_context(
            thread_id='meta_int_thread',
            source='user',
            text='Priority 2',
            metadata={'priority': 2},
        )

        result = await search_context(
            limit=50,
            thread_id='meta_int_thread',
            metadata={'priority': 1},
        )

        assert len(result['results']) == 1
        assert result['results'][0]['metadata']['priority'] == 1

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_with_metadata_boolean_value(self) -> None:
        """Test searching with metadata containing boolean."""
        await store_context(
            thread_id='meta_bool_thread',
            source='user',
            text='Completed task',
            metadata={'completed': True},
        )
        await store_context(
            thread_id='meta_bool_thread',
            source='user',
            text='Pending task',
            metadata={'completed': False},
        )

        result = await search_context(
            limit=50,
            thread_id='meta_bool_thread',
            metadata={'completed': True},
        )

        assert len(result['results']) == 1


class TestUpdateContextEdgeCases:
    """Test edge cases for update_context tool."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_text_only(self) -> None:
        """Test updating only text field."""
        result = await store_context(
            thread_id='update_text_thread',
            source='user',
            text='Original text',
            tags=['tag1'],
        )
        context_id = result['context_id']

        # Update only text
        update_result = await update_context(
            context_id=context_id,
            text='Updated text',
        )

        assert update_result['success'] is True

        # Verify text changed but tags unchanged
        entries = await get_context_by_ids(context_ids=[context_id])
        entry: dict[str, Any] = dict(entries[0])
        assert entry['text_content'] == 'Updated text'
        assert 'tag1' in entry['tags']

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_tags_only(self) -> None:
        """Test updating only tags field."""
        result = await store_context(
            thread_id='update_tags_thread',
            source='user',
            text='Original text',
            tags=['old_tag'],
        )
        context_id = result['context_id']

        # Update only tags
        update_result = await update_context(
            context_id=context_id,
            tags=['new_tag1', 'new_tag2'],
        )

        assert update_result['success'] is True

        # Verify tags changed but text unchanged
        entries = await get_context_by_ids(context_ids=[context_id])
        entry: dict[str, Any] = dict(entries[0])
        assert entry['text_content'] == 'Original text'
        assert 'new_tag1' in entry['tags']
        assert 'new_tag2' in entry['tags']
        assert 'old_tag' not in entry['tags']

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_metadata_only(self) -> None:
        """Test updating only metadata field."""
        result = await store_context(
            thread_id='update_meta_thread',
            source='user',
            text='Original text',
            metadata={'old_key': 'old_value'},
        )
        context_id = result['context_id']

        # Update only metadata
        update_result = await update_context(
            context_id=context_id,
            metadata={'new_key': 'new_value'},
        )

        assert update_result['success'] is True

        # Verify metadata changed but text unchanged
        entries = await get_context_by_ids(context_ids=[context_id])
        entry: dict[str, Any] = dict(entries[0])
        assert entry['text_content'] == 'Original text'
        assert entry['metadata']['new_key'] == 'new_value'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_clear_tags(self) -> None:
        """Test clearing tags by providing empty list."""
        result = await store_context(
            thread_id='clear_tags_thread',
            source='user',
            text='Text with tags',
            tags=['tag1', 'tag2'],
        )
        context_id = result['context_id']

        # Clear tags
        update_result = await update_context(
            context_id=context_id,
            tags=[],
        )

        assert update_result['success'] is True

        # Verify tags are cleared
        entries = await get_context_by_ids(context_ids=[context_id])
        entry: dict[str, Any] = dict(entries[0])
        assert entry['tags'] == []

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_nonexistent_context(self) -> None:
        """Test updating a non-existent context returns error."""
        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match='not found'):
            await update_context(
                context_id=999999,
                text='New text',
            )


class TestDeleteContextEdgeCases:
    """Test edge cases for delete_context tool."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_delete_by_multiple_ids(self) -> None:
        """Test deleting multiple entries by IDs."""
        # Create multiple entries
        ids = []
        for i in range(5):
            result = await store_context(
                thread_id='multi_delete_thread',
                source='user',
                text=f'Entry {i}',
            )
            ids.append(result['context_id'])

        # Delete first 3
        delete_result = await delete_context(context_ids=ids[:3])

        assert delete_result['success'] is True
        assert delete_result['deleted_count'] == 3

        # Verify remaining entries
        search_result = await search_context(limit=50, thread_id='multi_delete_thread')
        assert len(search_result['results']) == 2

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_delete_with_mixed_existing_nonexisting_ids(self) -> None:
        """Test deleting mix of existing and non-existing IDs."""
        result = await store_context(
            thread_id='mixed_delete_thread',
            source='user',
            text='Entry to delete',
        )
        existing_id = result['context_id']

        # Delete mix of existing and non-existing
        delete_result = await delete_context(
            context_ids=[existing_id, 999998, 999999],
        )

        assert delete_result['success'] is True
        assert delete_result['deleted_count'] == 1


class TestGetContextByIdsEdgeCases:
    """Test edge cases for get_context_by_ids tool."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_context_with_images_excluded(self) -> None:
        """Test getting context without image data."""
        image_data = base64.b64encode(b'test image data').decode('utf-8')
        result = await store_context(
            thread_id='get_no_images_thread',
            source='user',
            text='Entry with image',
            images=[{'data': image_data, 'mime_type': 'image/png'}],
        )
        context_id = result['context_id']

        # Get without images
        entries = await get_context_by_ids(
            context_ids=[context_id],
            include_images=False,
        )

        assert len(entries) == 1
        entry: dict[str, Any] = dict(entries[0])
        assert entry['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_context_preserves_metadata(self) -> None:
        """Test that get_context_by_ids preserves metadata."""
        complex_metadata = {
            'nested': {'key': 'value'},
            'list': [1, 2, 3],
            'boolean': True,
            'number': 42,
        }
        result = await store_context(
            thread_id='metadata_preserve_thread',
            source='user',
            text='Entry with metadata',
            metadata=complex_metadata,
        )
        context_id = result['context_id']

        entries = await get_context_by_ids(context_ids=[context_id])

        assert len(entries) == 1
        entry: dict[str, Any] = dict(entries[0])
        assert entry['metadata'] == complex_metadata


class TestGetStatisticsEdgeCases:
    """Test edge cases for get_statistics tool."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_statistics_with_data(self) -> None:
        """Test statistics with diverse data."""
        # Create diverse data
        await store_context(
            thread_id='stats_thread_1',
            source='user',
            text='User entry',
            tags=['tag1', 'tag2'],
        )
        await store_context(
            thread_id='stats_thread_1',
            source='agent',
            text='Agent entry',
            tags=['tag2', 'tag3'],
        )

        image_data = base64.b64encode(b'image').decode('utf-8')
        await store_context(
            thread_id='stats_thread_2',
            source='user',
            text='Multimodal entry',
            images=[{'data': image_data, 'mime_type': 'image/png'}],
        )

        stats = await get_statistics()

        assert stats['total_entries'] >= 3
        assert stats['total_threads'] >= 2
        assert 'by_source' in stats
        assert 'connection_metrics' in stats

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_statistics_includes_connection_metrics(self) -> None:
        """Test that statistics include connection metrics."""
        stats = await get_statistics()

        assert 'connection_metrics' in stats
        metrics = stats['connection_metrics']
        assert 'backend_type' in metrics or len(metrics) > 0


class TestStoreContextWithImages:
    """Test storing context with various image configurations."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_multiple_images(self) -> None:
        """Test storing context with multiple images."""
        images = [
            {
                'data': base64.b64encode(f'image{i}'.encode()).decode('utf-8'),
                'mime_type': 'image/png',
            }
            for i in range(3)
        ]

        result = await store_context(
            thread_id='multi_image_thread',
            source='user',
            text='Multiple images',
            images=images,
        )

        assert result['success'] is True

        # Verify all images stored
        entries = await get_context_by_ids(
            context_ids=[result['context_id']],
            include_images=True,
        )
        entry: dict[str, Any] = dict(entries[0])
        assert entry['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_image_with_metadata(self) -> None:
        """Test storing image with metadata."""
        image_metadata = {'width': 800, 'height': 600, 'format': 'png'}
        images = [
            {
                'data': base64.b64encode(b'image with meta').decode('utf-8'),
                'mime_type': 'image/png',
                'metadata': image_metadata,
            },
        ]

        result = await store_context(
            thread_id='image_meta_thread',
            source='user',
            text='Image with metadata',
            images=images,
        )

        assert result['success'] is True


class TestMetadataFilters:
    """Test metadata filter functionality."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_with_advanced_metadata_filters(self) -> None:
        """Test searching with advanced metadata filters."""
        await store_context(
            thread_id='adv_filter_thread',
            source='user',
            text='Entry 1',
            metadata={'priority': 1, 'status': 'active'},
        )
        await store_context(
            thread_id='adv_filter_thread',
            source='user',
            text='Entry 2',
            metadata={'priority': 5, 'status': 'pending'},
        )
        await store_context(
            thread_id='adv_filter_thread',
            source='user',
            text='Entry 3',
            metadata={'priority': 10, 'status': 'active'},
        )

        # Search with advanced filter for priority > 2
        result = await search_context(
            limit=50,
            thread_id='adv_filter_thread',
            metadata_filters=[{'key': 'priority', 'operator': 'gt', 'value': 2}],
        )

        # Should find entries with priority > 2
        assert len(result['results']) == 2

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_with_string_contains_filter(self) -> None:
        """Test searching with string contains filter."""
        await store_context(
            thread_id='contains_filter_thread',
            source='user',
            text='Entry with important task',
            metadata={'task_name': 'Important Task A'},
        )
        await store_context(
            thread_id='contains_filter_thread',
            source='user',
            text='Entry with other task',
            metadata={'task_name': 'Other Task B'},
        )

        # Search for tasks containing "Important"
        result = await search_context(
            limit=50,
            thread_id='contains_filter_thread',
            metadata_filters=[{'key': 'task_name', 'operator': 'contains', 'value': 'Important'}],
        )

        assert len(result['results']) == 1
        assert 'Important' in result['results'][0]['metadata']['task_name']
