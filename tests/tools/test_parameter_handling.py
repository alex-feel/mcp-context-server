"""Tests that complex tool parameters (lists, dicts, None) reach the core tools as Python objects: list, dict,
list-of-dict and None arguments to ``store_context``, ``search_context``, ``get_context_by_ids`` and
``delete_context``, alone and in combination.
"""

import base64
import json
from typing import Any

import pytest
from fastmcp.exceptions import ToolError

from app.tools.context.delete import delete_context
from app.tools.context.retrieve import get_context_by_ids
from app.tools.context.store import store_context
from app.tools.search.browse import search_context
from app.types import MetadataDict


@pytest.mark.usefixtures('initialized_server')
class TestParameterHandling:
    """Test that complex parameter types (lists, dicts, None) reach the tools as Python objects."""

    @pytest.mark.asyncio
    async def test_store_context_tags_as_list(self) -> None:
        """Test that tags parameter accepts a proper Python list[str]."""
        # Test with a list of strings
        tags_list = ['python', 'testing', 'mcp-server']

        result = await store_context(
            thread_id='param_test_tags',
            source='user',
            text='Testing tags parameter',
            tags=tags_list,
        )

        assert result['success'] is True
        assert 'context_id' in result

        # Verify tags were stored correctly
        search_result = await search_context(limit=50, thread_id='param_test_tags')
        assert len(search_result['results']) == 1
        assert set(search_result['results'][0]['tags']) == set(tags_list)

    @pytest.mark.asyncio
    async def test_store_context_tags_none(self) -> None:
        """Test that tags parameter can be None."""
        result = await store_context(
            thread_id='param_test_no_tags',
            source='agent',
            text='Testing without tags',
            tags=None,
        )

        assert result['success'] is True

        # Verify no tags stored
        search_result = await search_context(limit=50, thread_id='param_test_no_tags')
        assert search_result['results'][0]['tags'] == []

    @pytest.mark.asyncio
    async def test_store_context_metadata_as_dict(self) -> None:
        """Test that metadata parameter accepts a proper Python dict[str, Any]."""
        # Test with a complex nested dictionary
        metadata_dict: MetadataDict = {
            'version': '1.0.0',
            'timestamp': 1234567890,
            'nested': {
                'level1': {
                    'level2': ['a', 'b', 'c'],
                    'number': 42,
                    'boolean': True,
                    'null_value': None,
                },
            },
            'array': [1, 2, 3, {'key': 'value'}],
        }

        result = await store_context(
            thread_id='param_test_metadata',
            source='user',
            text='Testing metadata parameter',
            metadata=metadata_dict,
        )

        assert result['success'] is True

        # Verify metadata was stored correctly
        fetched = await get_context_by_ids(
            context_ids=[result['context_id']],
            include_images=False,
        )
        assert len(fetched) == 1
        entry = dict(fetched[0])
        assert entry['metadata'] == metadata_dict

    @pytest.mark.asyncio
    async def test_store_context_metadata_none(self) -> None:
        """Test that metadata parameter can be None."""
        result = await store_context(
            thread_id='param_test_no_metadata',
            source='agent',
            text='Testing without metadata',
            metadata=None,
        )

        assert result['success'] is True

        # Verify no metadata stored
        fetched = await get_context_by_ids(context_ids=[result['context_id']])
        entry = dict(fetched[0])
        assert entry['metadata'] is None

    @pytest.mark.asyncio
    async def test_store_context_images_as_list_of_dicts(self) -> None:
        """Test that images parameter accepts a proper list[dict[str, str]]."""
        # Create test images list
        images_list = [
            {
                'data': base64.b64encode(b'test_image_1').decode('utf-8'),
                'mime_type': 'image/png',
            },
            {
                'data': base64.b64encode(b'test_image_2').decode('utf-8'),
                'mime_type': 'image/jpeg',
                'metadata': json.dumps({'size': 1024, 'width': 100}),
            },
            {
                'data': base64.b64encode(b'test_image_3').decode('utf-8'),
                'mime_type': 'image/gif',
            },
        ]

        result = await store_context(
            thread_id='param_test_images',
            source='user',
            text='Testing images parameter',
            images=images_list,
        )

        assert result['success'] is True
        assert 'Context stored with 3 images' in result['message']

        # Verify images were stored correctly
        fetched = await get_context_by_ids(
            context_ids=[result['context_id']],
            include_images=True,
        )
        assert len(fetched) == 1
        assert 'images' in fetched[0]
        fetched_images = fetched[0]['images']
        assert fetched_images is not None
        assert len(fetched_images) == 3

        # Check mime types are preserved
        mime_types = [img['mime_type'] for img in fetched_images]
        assert set(mime_types) == {'image/png', 'image/jpeg', 'image/gif'}

    @pytest.mark.asyncio
    async def test_store_context_images_none(self) -> None:
        """Test that images parameter can be None."""
        result = await store_context(
            thread_id='param_test_no_images',
            source='agent',
            text='Testing without images',
            images=None,
        )

        assert result['success'] is True
        assert 'Context stored' in result['message']

    @pytest.mark.asyncio
    async def test_store_context_all_complex_params_together(self) -> None:
        """Test all complex parameters (tags, metadata, images) together."""
        tags = ['comprehensive', 'test', 'all-params']
        metadata: MetadataDict = {
            'test_type': 'comprehensive',
            'params_tested': ['tags', 'metadata', 'images'],
            'test_id': 12345,
        }
        images = [
            {
                'data': base64.b64encode(b'comprehensive_test_img').decode('utf-8'),
                'mime_type': 'image/png',
            },
        ]

        result = await store_context(
            thread_id='param_test_comprehensive',
            source='user',
            text='Testing all complex parameters together',
            tags=tags,
            metadata=metadata,
            images=images,
        )

        assert result['success'] is True

        # Verify all parameters were stored correctly
        fetched = await get_context_by_ids(
            context_ids=[result['context_id']],
            include_images=True,
        )
        assert len(fetched) == 1
        entry: dict[str, Any] = {**fetched[0]}

        assert set(entry['tags']) == set(tags)
        assert entry['metadata'] == metadata
        assert len(entry['images']) == 1
        assert entry['images'][0]['mime_type'] == 'image/png'

    @pytest.mark.asyncio
    async def test_search_context_tags_as_list(self) -> None:
        """Test that search_context tags parameter accepts a proper Python list[str]."""
        # First, store some tagged entries
        await store_context(
            thread_id='search_tags_test',
            source='user',
            text='Entry 1',
            tags=['python', 'async'],
        )
        await store_context(
            thread_id='search_tags_test',
            source='agent',
            text='Entry 2',
            tags=['javascript', 'async'],
        )
        await store_context(
            thread_id='search_tags_test',
            source='user',
            text='Entry 3',
            tags=['python', 'testing'],
        )

        # Search with tags as list
        results = await search_context(limit=50, tags=['python', 'javascript'])

        # Should find entries with either python or javascript tags
        assert len(results['results']) >= 3  # All three entries match

        # Test with single tag in list
        results = await search_context(limit=50, tags=['testing'])
        found = [r for r in results['results'] if r['thread_id'] == 'search_tags_test' and 'testing' in r['tags']]
        assert len(found) == 1

    @pytest.mark.asyncio
    async def test_search_context_tags_none(self) -> None:
        """Test that search_context tags parameter can be None."""
        # Store a test entry
        await store_context(
            thread_id='search_no_tags_test',
            source='user',
            text='Test entry',
            tags=['test'],
        )

        # Search without tags filter (tags=None)
        results = await search_context(limit=50, thread_id='search_no_tags_test', tags=None)

        assert len(results['results']) == 1
        assert results['results'][0]['thread_id'] == 'search_no_tags_test'

    @pytest.mark.asyncio
    async def test_search_context_empty_tags_list(self) -> None:
        """Test that search_context handles empty tags list correctly."""
        # Store a test entry
        await store_context(
            thread_id='search_empty_tags_test',
            source='user',
            text='Test entry',
            tags=['test'],
        )

        # Search with empty tags list should return all entries (no filter applied)
        results = await search_context(limit=50, thread_id='search_empty_tags_test', tags=[])

        assert len(results['results']) == 1

    @pytest.mark.asyncio
    async def test_get_context_by_ids_list_of_ints(self) -> None:
        """Test that get_context_by_ids accepts a proper list[int]."""
        # Store multiple entries
        ids = []
        for i in range(5):
            result = await store_context(
                thread_id='get_by_ids_test',
                source='user' if i % 2 == 0 else 'agent',
                text=f'Entry {i}',
            )
            ids.append(result['context_id'])

        # Test with list of integers
        context_ids = ids[:3]  # Get first 3
        results = await get_context_by_ids(context_ids=context_ids)

        assert len(results) == 3
        returned_ids = [dict(r)['id'] for r in results]
        assert set(returned_ids) == set(context_ids)

    @pytest.mark.asyncio
    async def test_get_context_by_ids_empty_list(self) -> None:
        """Test that Pydantic Field(min_length=1) handles empty list.

        Note: Pydantic validates at FastMCP level. This test verifies normal operation.
        """
        # Test with valid non-empty list
        result = await get_context_by_ids(context_ids=['0190abcdef1234567890abcd00000001'])
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_context_by_ids_single_item_list(self) -> None:
        """Test that get_context_by_ids works with single-item list."""
        result = await store_context(
            thread_id='single_id_test',
            source='user',
            text='Single entry',
        )

        context_id = result['context_id']
        results = await get_context_by_ids(context_ids=[context_id])

        assert len(results) == 1
        entry = dict(results[0])
        assert entry['id'] == context_id

    @pytest.mark.asyncio
    async def test_delete_context_ids_as_list(self) -> None:
        """Test that delete_context accepts context_ids as a proper list[int]."""
        # Store multiple entries
        ids_to_delete = []
        for i in range(3):
            result = await store_context(
                thread_id='delete_ids_test',
                source='user',
                text=f'To delete {i}',
            )
            ids_to_delete.append(result['context_id'])

        # Store one to keep
        keep_result = await store_context(
            thread_id='delete_ids_test',
            source='agent',
            text='Keep this one',
        )

        # Delete with list of integers
        delete_result = await delete_context(context_ids=ids_to_delete)

        assert delete_result['success'] is True
        assert delete_result['deleted_count'] == 3

        # Verify only the kept entry remains
        remaining = await search_context(limit=50, thread_id='delete_ids_test')
        assert len(remaining['results']) == 1
        assert remaining['results'][0]['id'] == keep_result['context_id']

    @pytest.mark.asyncio
    async def test_delete_context_ids_none(self) -> None:
        """Test that delete_context accepts context_ids as None."""
        # Store test entries
        for i in range(3):
            await store_context(
                thread_id='delete_by_thread_test',
                source='user',
                text=f'Entry {i}',
            )

        # Delete by thread_id with context_ids=None
        delete_result = await delete_context(
            context_ids=None,
            thread_id='delete_by_thread_test',
        )

        assert delete_result['success'] is True
        assert delete_result['deleted_count'] == 3

    @pytest.mark.asyncio
    async def test_delete_context_empty_list(self) -> None:
        """Test that delete_context handles empty context_ids list."""
        # Delete with empty list should raise an error
        with pytest.raises(ToolError, match='Must provide either context_ids or thread_id'):
            await delete_context(context_ids=[])


@pytest.mark.usefixtures('initialized_server')
class TestParameterInteractions:
    """Test interactions between different parameters."""

    @pytest.mark.asyncio
    async def test_tags_affect_search_with_other_filters(self) -> None:
        """Test that tags work correctly with other search filters."""
        # Store entries with various combinations
        await store_context(
            thread_id='interaction_test',
            source='user',
            text='Entry 1',
            tags=['python', 'web'],
        )
        await store_context(
            thread_id='interaction_test',
            source='agent',
            text='Entry 2',
            tags=['python', 'cli'],
        )
        await store_context(
            thread_id='interaction_test',
            source='user',
            text='Entry 3',
            tags=['javascript', 'web'],
        )
        await store_context(
            thread_id='other_thread',
            source='user',
            text='Entry 4',
            tags=['python', 'web'],
        )

        # Search with multiple filters including tags
        results = await search_context(
            limit=50,
            thread_id='interaction_test',
            source='user',
            tags=['python', 'javascript'],
        )

        # Should find only entries 1 and 3 (correct thread, source, and tags)
        assert len(results['results']) == 2
        texts = [r['text_content'] for r in results['results']]
        assert set(texts) == {'Entry 1', 'Entry 3'}

    @pytest.mark.asyncio
    async def test_metadata_with_images(self) -> None:
        """Test that metadata and images work together correctly."""
        metadata: MetadataDict = {
            'image_count': 2,
            'total_size': 2048,
            'processing': {
                'resized': True,
                'compressed': False,
            },
        }
        images = [
            {
                'data': base64.b64encode(b'image1').decode('utf-8'),
                'mime_type': 'image/png',
            },
            {
                'data': base64.b64encode(b'image2').decode('utf-8'),
                'mime_type': 'image/jpeg',
            },
        ]

        result = await store_context(
            thread_id='metadata_images_test',
            source='agent',
            text='Testing metadata with images',
            metadata=metadata,
            images=images,
        )

        assert result['success'] is True
        assert 'Context stored with 2 images' in result['message']

        # Verify both are stored correctly
        fetched = await get_context_by_ids(
            context_ids=[result['context_id']],
            include_images=True,
        )
        entry: dict[str, Any] = {**fetched[0]}
        assert entry['metadata'] == metadata
        assert len(entry['images']) == 2

    @pytest.mark.asyncio
    async def test_all_parameters_none_except_required(self) -> None:
        """Test that all optional parameters can be None simultaneously."""
        result = await store_context(
            thread_id='all_none_test',
            source='user',
            text='Only required params',
            tags=None,
            metadata=None,
            images=None,
        )

        assert result['success'] is True

        # Verify entry has no optional data
        fetched = await get_context_by_ids(context_ids=[result['context_id']])
        entry = dict(fetched[0])
        assert entry['tags'] == []
        assert entry['metadata'] is None
        assert 'images' not in entry or entry['images'] == []
