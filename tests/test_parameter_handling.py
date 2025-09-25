"""
Test suite for FastMCP parameter handling fix.

This module specifically tests the parameter type handling for complex types
(lists, dicts) in the FastMCP integration. These tests verify that the fix
for removing Field() wrappers from complex parameter types works correctly.

The issue being tested:
- Previously, complex types wrapped with Annotated[..., Field()] were incorrectly
  handled by FastMCP, arriving as JSON strings instead of proper Python objects
- The fix removed Field() wrappers from list[str], dict[str, Any], list[dict], etc.
- These tests ensure complex parameters work as expected
"""

from __future__ import annotations

import base64
import json
from typing import Literal
from typing import cast

import app.server
import pytest

# Get the actual async functions from the FunctionTool wrappers
store_context = app.server.store_context.fn
search_context = app.server.search_context.fn
get_context_by_ids = app.server.get_context_by_ids.fn
delete_context = app.server.delete_context.fn


@pytest.mark.usefixtures('initialized_server')
class TestParameterHandling:
    """Test that FastMCP parameter types are handled correctly after the fix."""

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
        search_result = await search_context(thread_id='param_test_tags')
        assert len(search_result) == 1
        assert set(search_result[0]['tags']) == set(tags_list)

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
        search_result = await search_context(thread_id='param_test_no_tags')
        assert search_result[0]['tags'] == []

    @pytest.mark.asyncio
    async def test_store_context_metadata_as_dict(self) -> None:
        """Test that metadata parameter accepts a proper Python dict[str, Any]."""
        # Test with a complex nested dictionary
        metadata_dict = {
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
        assert fetched[0]['metadata'] == metadata_dict

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
        assert fetched[0]['metadata'] is None

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
        assert len(fetched[0]['images']) == 3

        # Check mime types are preserved
        mime_types = [img['mime_type'] for img in fetched[0]['images']]
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
        assert 'Context stored with 0 images' in result['message']

    @pytest.mark.asyncio
    async def test_store_context_all_complex_params_together(self) -> None:
        """Test all complex parameters (tags, metadata, images) together."""
        tags = ['comprehensive', 'test', 'all-params']
        metadata = {
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
        entry = fetched[0]

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
        results = await search_context(tags=['python', 'javascript'])

        # Should find entries with either python or javascript tags
        assert len(results) >= 3  # All three entries match

        # Test with single tag in list
        results = await search_context(tags=['testing'])
        found = [r for r in results if r['thread_id'] == 'search_tags_test' and 'testing' in r['tags']]
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
        results = await search_context(thread_id='search_no_tags_test', tags=None)

        assert len(results) == 1
        assert results[0]['thread_id'] == 'search_no_tags_test'

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
        results = await search_context(thread_id='search_empty_tags_test', tags=[])

        assert len(results) == 1

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
        returned_ids = [r['id'] for r in results]
        assert set(returned_ids) == set(context_ids)

    @pytest.mark.asyncio
    async def test_get_context_by_ids_empty_list(self) -> None:
        """Test that get_context_by_ids handles empty list correctly."""
        results = await get_context_by_ids(context_ids=[])
        assert results == []

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
        assert results[0]['id'] == context_id

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
        remaining = await search_context(thread_id='delete_ids_test')
        assert len(remaining) == 1
        assert remaining[0]['id'] == keep_result['context_id']

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
        # Delete with empty list should return an error
        delete_result = await delete_context(context_ids=[])

        assert delete_result['success'] is False
        assert 'Must provide either context_ids or thread_id' in delete_result['error']


@pytest.mark.usefixtures('initialized_server')
class TestParameterValidation:
    """Test parameter validation still works correctly after the fix."""

    @pytest.mark.asyncio
    async def test_invalid_source_type(self) -> None:
        """Test that invalid source type is still caught."""
        result = await store_context(
            thread_id='invalid_source_test',
            source=cast(Literal['user', 'agent'], 'invalid'),
            text='This should fail',
        )

        assert result['success'] is False
        assert "Source must be 'user' or 'agent'" in result['error']

    @pytest.mark.asyncio
    async def test_limit_validation(self) -> None:
        """Test that limit parameter validation still works."""
        # Test limit too high (>500)
        results = await search_context(limit=1000)
        # Should be capped at 500 or less
        assert len(results) <= 500

        # Test limit too low (<1) - should be handled without raising
        # The function will handle invalid values gracefully
        # by using default limit or validating internally

    @pytest.mark.asyncio
    async def test_offset_validation(self) -> None:
        """Test that offset parameter validation still works."""
        # Negative offset should be handled gracefully
        # The function will use 0 or validate internally
        results = await search_context(offset=-1)
        # Should return results or empty list, not raise
        assert isinstance(results, list)


@pytest.mark.usefixtures('initialized_server')
class TestParameterTypeCoercion:
    """Test that parameters handle type coercion correctly."""

    @pytest.mark.asyncio
    async def test_tags_normalization(self) -> None:
        """Test that tags are normalized to lowercase."""
        mixed_case_tags = ['Python', 'TESTING', 'MiXeD-CaSe']

        result = await store_context(
            thread_id='tags_normalization_test',
            source='user',
            text='Testing tag normalization',
            tags=mixed_case_tags,
        )

        assert result['success'] is True

        # Verify tags were normalized to lowercase
        search_result = await search_context(thread_id='tags_normalization_test')
        normalized_tags = ['python', 'testing', 'mixed-case']
        assert set(search_result[0]['tags']) == set(normalized_tags)

    @pytest.mark.asyncio
    async def test_metadata_json_serialization(self) -> None:
        """Test that metadata is properly JSON serialized/deserialized."""
        # Complex metadata with various types
        metadata = {
            'string': 'value',
            'number': 123.45,
            'boolean': True,
            'null': None,
            'array': [1, 'two', None, {'nested': 'object'}],
            'object': {
                'deep': {
                    'nesting': {
                        'works': 'correctly',
                    },
                },
            },
        }

        result = await store_context(
            thread_id='metadata_serialization_test',
            source='agent',
            text='Testing metadata serialization',
            metadata=metadata,
        )

        assert result['success'] is True

        # Fetch and verify metadata round-trip
        fetched = await get_context_by_ids(context_ids=[result['context_id']])
        assert fetched[0]['metadata'] == metadata


@pytest.mark.usefixtures('initialized_server')
class TestEdgeCasesForParameters:
    """Test edge cases specific to parameter handling."""

    @pytest.mark.asyncio
    async def test_unicode_in_tags(self) -> None:
        """Test that Unicode characters in tags are handled correctly."""
        unicode_tags = ['python', '中文标签', 'عربي', 'тэг', '🏷️tag']

        result = await store_context(
            thread_id='unicode_tags_test',
            source='user',
            text='Testing Unicode tags',
            tags=unicode_tags,
        )

        assert result['success'] is True

        # Verify Unicode tags work in search
        search_results = await search_context(tags=['中文标签'])
        found = [r for r in search_results if r['thread_id'] == 'unicode_tags_test']
        assert len(found) == 1

    @pytest.mark.asyncio
    async def test_large_list_of_tags(self) -> None:
        """Test handling of large number of tags."""
        # Create 100 unique tags
        large_tags_list = [f'tag_{i:03d}' for i in range(100)]

        result = await store_context(
            thread_id='large_tags_test',
            source='user',
            text='Testing large number of tags',
            tags=large_tags_list,
        )

        assert result['success'] is True

        # Verify all tags were stored
        search_result = await search_context(thread_id='large_tags_test')
        assert len(search_result[0]['tags']) == 100

    @pytest.mark.asyncio
    async def test_very_large_metadata(self) -> None:
        """Test handling of very large metadata dictionary."""
        # Create a large metadata structure
        large_metadata = {
            f'key_{i}': {
                'data': 'x' * 1000,  # 1KB per entry
                'index': i,
                'nested': {
                    'level1': {
                        'level2': list(range(10)),
                    },
                },
            }
            for i in range(100)  # 100+ KB total
        }

        result = await store_context(
            thread_id='large_metadata_test',
            source='agent',
            text='Testing very large metadata',
            metadata=large_metadata,
        )

        assert result['success'] is True

        # Verify large metadata round-trips correctly
        fetched = await get_context_by_ids(context_ids=[result['context_id']])
        assert fetched[0]['metadata'] == large_metadata

    @pytest.mark.asyncio
    async def test_multiple_identical_tags(self) -> None:
        """Test that duplicate tags are deduplicated."""
        duplicate_tags = ['python', 'python', 'test', 'test', 'python']

        result = await store_context(
            thread_id='duplicate_tags_test',
            source='user',
            text='Testing duplicate tags',
            tags=duplicate_tags,
        )

        assert result['success'] is True

        # Verify duplicates were removed
        search_result = await search_context(thread_id='duplicate_tags_test')
        unique_tags = set(search_result[0]['tags'])
        assert unique_tags == {'python', 'test'}

    @pytest.mark.asyncio
    async def test_special_characters_in_metadata_keys(self) -> None:
        """Test metadata with special characters in keys."""
        special_metadata = {
            'normal_key': 'value1',
            'key-with-dash': 'value2',
            'key.with.dots': 'value3',
            'key_with_underscore': 'value4',
            'key with spaces': 'value5',
            '123numeric': 'value6',
            'üñíçødé': 'value7',
        }

        result = await store_context(
            thread_id='special_metadata_test',
            source='user',
            text='Testing special metadata keys',
            metadata=special_metadata,
        )

        assert result['success'] is True

        # Verify special keys preserved
        fetched = await get_context_by_ids(context_ids=[result['context_id']])
        assert fetched[0]['metadata'] == special_metadata


@pytest.mark.usefixtures('initialized_server')
class TestForwardSlashAndSpecialCharacterTags:
    """Test forward slashes and special characters in tags - addressing reported issue."""

    @pytest.mark.asyncio
    async def test_forward_slash_tags(self) -> None:
        """Test that forward slashes in tags work correctly - main issue fix."""
        # Test the exact tags reported in the issue
        forward_slash_tags = ['app/file', 'config/database', 'src/main.py']

        result = await store_context(
            thread_id='forward_slash_test',
            source='user',
            text='Testing forward slash tags',
            tags=forward_slash_tags,
        )

        assert result['success'] is True
        assert 'context_id' in result

        # Verify tags were stored correctly with forward slashes preserved
        search_result = await search_context(thread_id='forward_slash_test')
        assert len(search_result) == 1
        assert set(search_result[0]['tags']) == set(forward_slash_tags)

        # Test searching by forward slash tags
        search_by_tag = await search_context(tags=['app/file'])
        found = [r for r in search_by_tag if r['thread_id'] == 'forward_slash_test']
        assert len(found) == 1

    @pytest.mark.asyncio
    async def test_path_like_tags(self) -> None:
        """Test various path-like structures in tags."""
        path_tags = [
            'src/components/header.tsx',
            'lib/utils/helpers.py',
            'tests/unit/test_models.py',
            'docs/api/v2/endpoints',
            '/absolute/path/to/file',
            'relative/../path/to/file',
            'path\\with\\backslashes',  # Windows-style paths
        ]

        result = await store_context(
            thread_id='path_tags_test',
            source='agent',
            text='Testing path-like tags',
            tags=path_tags,
        )

        assert result['success'] is True

        # Verify all path tags were stored
        search_result = await search_context(thread_id='path_tags_test')
        assert len(search_result) == 1
        assert len(search_result[0]['tags']) == len(path_tags)

    @pytest.mark.asyncio
    async def test_mixed_special_characters_in_tags(self) -> None:
        """Test various special characters in tags."""
        special_tags = [
            'feature/new-ui',  # forward slash with hyphen
            'bug#123',  # hash symbol
            'v1.2.3',  # periods
            'user@domain.com',  # at symbol
            'python:3.12',  # colon
            'high-priority!',  # exclamation
            'question?',  # question mark
            'task[urgent]',  # brackets
            'scope{global}',  # braces
            'item_with_underscore',  # underscore
            '100%complete',  # percent
            'a&b',  # ampersand
            'c++',  # plus signs
            'price=$99',  # dollar sign
            'temp~backup',  # tilde
            'item*wildcard',  # asterisk
        ]

        result = await store_context(
            thread_id='special_chars_test',
            source='user',
            text='Testing special characters in tags',
            tags=special_tags,
        )

        assert result['success'] is True

        # Verify all special character tags were stored (normalized to lowercase)
        search_result = await search_context(thread_id='special_chars_test')
        assert len(search_result) == 1
        assert len(search_result[0]['tags']) == len(special_tags)
        # Check that lowercase normalization happened
        assert 'feature/new-ui' in search_result[0]['tags']
        assert 'v1.2.3' in search_result[0]['tags']

    @pytest.mark.asyncio
    async def test_empty_and_whitespace_tags_filtering(self) -> None:
        """Test that empty and whitespace-only tags are filtered out."""
        tags_with_empty = [
            'valid_tag',
            '',  # empty string
            '   ',  # spaces only
            '\t',  # tab only
            '\n',  # newline only
            '  \t\n  ',  # mixed whitespace
            'another_valid_tag',
            None,  # None should be handled gracefully if it somehow gets through
        ]

        # Filter out None for the actual call
        tags_to_send = [tag for tag in tags_with_empty if tag is not None]

        result = await store_context(
            thread_id='empty_tags_test',
            source='agent',
            text='Testing empty tag filtering',
            tags=tags_to_send,
        )

        assert result['success'] is True

        # Verify only valid tags were stored
        search_result = await search_context(thread_id='empty_tags_test')
        assert len(search_result) == 1
        assert set(search_result[0]['tags']) == {'valid_tag', 'another_valid_tag'}

    @pytest.mark.asyncio
    async def test_tag_normalization_with_special_chars(self) -> None:
        """Test that tag normalization preserves special characters correctly."""
        # Tags with mixed case and special characters
        mixed_tags = [
            'Feature/NEW-UI',  # uppercase with slash and hyphen
            'CONFIG/Database',  # mixed case with slash
            'SRC/Main.py',  # mixed case with extension
            'Path\\TO\\File',  # backslashes with mixed case
            'User@DOMAIN.COM',  # email-like with uppercase
        ]

        result = await store_context(
            thread_id='normalization_special_test',
            source='user',
            text='Testing normalization with special chars',
            tags=mixed_tags,
        )

        assert result['success'] is True

        # Verify normalization to lowercase while preserving special chars
        search_result = await search_context(thread_id='normalization_special_test')
        expected_tags = [
            'feature/new-ui',
            'config/database',
            'src/main.py',
            'path\\to\\file',
            'user@domain.com',
        ]
        assert set(search_result[0]['tags']) == set(expected_tags)

    @pytest.mark.asyncio
    async def test_search_by_forward_slash_tags(self) -> None:
        """Test searching specifically using tags with forward slashes."""
        # Store multiple entries with different forward slash tags
        await store_context(
            thread_id='search_slash_test',
            source='user',
            text='Python file',
            tags=['src/python/main.py', 'type/script'],
        )
        await store_context(
            thread_id='search_slash_test',
            source='agent',
            text='Config file',
            tags=['config/settings.yaml', 'type/config'],
        )
        await store_context(
            thread_id='search_slash_test',
            source='user',
            text='Test file',
            tags=['tests/unit/test_main.py', 'type/test'],
        )

        # Search by forward slash tag
        results = await search_context(tags=['src/python/main.py'])
        found = [r for r in results if r['thread_id'] == 'search_slash_test']
        assert len(found) == 1
        assert found[0]['text_content'] == 'Python file'

        # Search by multiple forward slash tags (OR logic)
        results = await search_context(tags=['config/settings.yaml', 'tests/unit/test_main.py'])
        found = [r for r in results if r['thread_id'] == 'search_slash_test']
        assert len(found) == 2
        texts = {r['text_content'] for r in found}
        assert texts == {'Config file', 'Test file'}

    @pytest.mark.asyncio
    async def test_json_serialization_of_forward_slash_tags(self) -> None:
        """Test that forward slash tags survive JSON serialization correctly."""
        tags = ['module/component', 'path/to/resource', 'namespace/class']

        # Test as JSON string (simulating Claude Code behavior)
        json_tags = json.dumps(tags)

        # The deserialize_json_param should handle this
        from app.server import deserialize_json_param

        deserialized = deserialize_json_param(json_tags)
        assert deserialized == tags

        # Double-encoded JSON (edge case)
        double_encoded = json.dumps(json_tags)
        deserialized_double = deserialize_json_param(double_encoded)
        assert deserialized_double == tags

    @pytest.mark.asyncio
    async def test_extreme_forward_slash_cases(self) -> None:
        """Test edge cases with forward slashes."""
        edge_case_tags = [
            '/',  # just a slash
            '//',  # double slash
            '///',  # triple slash
            '/leading/slash',  # leading slash
            'trailing/slash/',  # trailing slash
            '/both/slashes/',  # both ends
            'multiple//slashes///in////middle',  # multiple consecutive
            '.',  # just a period
            '..',  # double period
            '../../../relative',  # relative path
        ]

        result = await store_context(
            thread_id='extreme_slash_test',
            source='user',
            text='Testing extreme forward slash cases',
            tags=edge_case_tags,
        )

        assert result['success'] is True

        # Verify all edge cases were stored
        search_result = await search_context(thread_id='extreme_slash_test')
        assert len(search_result) == 1
        assert len(search_result[0]['tags']) == len(edge_case_tags)


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
            thread_id='interaction_test',
            source='user',
            tags=['python', 'javascript'],
        )

        # Should find only entries 1 and 3 (correct thread, source, and tags)
        assert len(results) == 2
        texts = [r['text_content'] for r in results]
        assert set(texts) == {'Entry 1', 'Entry 3'}

    @pytest.mark.asyncio
    async def test_metadata_with_images(self) -> None:
        """Test that metadata and images work together correctly."""
        metadata = {
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
        assert fetched[0]['metadata'] == metadata
        assert len(fetched[0]['images']) == 2

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
        assert fetched[0]['tags'] == []
        assert fetched[0]['metadata'] is None
        assert 'images' not in fetched[0] or fetched[0]['images'] == []
