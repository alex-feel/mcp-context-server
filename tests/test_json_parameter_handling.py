"""
Tests for handling JSON string parameters from Claude Code.

Claude Code sends complex types as JSON strings, which need to be handled
properly by the server. This test module ensures both native types and
JSON string types are accepted and processed correctly.
"""

import json
import math
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from app.server import deserialize_json_param
from app.types import JsonValue


class TestDeserializeJsonParam:
    """Test the JSON parameter deserialization utility."""

    def test_deserialize_list_from_json_string(self):
        """Test deserializing a list from JSON string."""
        json_str = '["tag1", "tag2", "tag3"]'
        result = deserialize_json_param(json_str)
        assert result == ['tag1', 'tag2', 'tag3']

    def test_deserialize_dict_from_json_string(self):
        """Test deserializing a dict from JSON string."""
        json_str = '{"key": "value", "number": 42}'
        result = deserialize_json_param(json_str)
        assert result == {'key': 'value', 'number': 42}

    def test_deserialize_list_of_dicts_from_json_string(self):
        """Test deserializing a list of dicts from JSON string."""
        json_str = '[{"data": "base64data", "mime_type": "image/png"}]'
        result = deserialize_json_param(json_str)
        assert result == [{'data': 'base64data', 'mime_type': 'image/png'}]

    def test_deserialize_returns_native_list(self):
        """Test that native lists are returned as-is."""
        native_list = ['tag1', 'tag2']
        result = deserialize_json_param(cast(JsonValue, native_list))
        assert result is native_list

    def test_deserialize_returns_native_dict(self):
        """Test that native dicts are returned as-is."""
        native_dict = {'key': 'value'}
        result = deserialize_json_param(cast(JsonValue, native_dict))
        assert result is native_dict

    def test_deserialize_invalid_json_returns_string(self):
        """Test that invalid JSON strings are returned as-is."""
        invalid_json = 'not a json string'
        result = deserialize_json_param(invalid_json)
        assert result == 'not a json string'

    def test_deserialize_none_returns_none(self):
        """Test that None is returned as-is."""
        result = deserialize_json_param(None)
        assert result is None

    def test_deserialize_numeric_json(self):
        """Test deserializing numbers from JSON strings."""
        assert deserialize_json_param('42') == 42
        # Test high-precision float deserialization with actual value of e
        assert deserialize_json_param('2.718281828459045') == pytest.approx(math.e, abs=1e-10)
        assert deserialize_json_param('[1, 2, 3]') == [1, 2, 3]


class TestStoreContextParameterHandling:
    """Test store_context with different parameter formats."""

    @pytest.mark.asyncio
    async def test_store_context_with_json_string_tags(self):
        """Test store_context accepts tags as JSON string."""
        # Import the function directly within the test
        from app.server import store_context

        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.tags.store_tags = AsyncMock(return_value=None)
            mock_repos.images.store_images = AsyncMock(return_value=None)
            mock_ensure_repos.return_value = mock_repos

            # Send tags as JSON string (how Claude Code sends it)
            # Call the wrapped function directly
            result = await store_context.fn(
                thread_id='test-thread',
                source='user',
                text='Test content',
                tags='["tag1", "tag2", "tag3"]',  # JSON string
            )

            assert result['success'] is True
            assert result['context_id'] == 1

    @pytest.mark.asyncio
    async def test_store_context_with_native_tags(self):
        """Test store_context accepts tags as native list."""
        from app.server import store_context

        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.tags.store_tags = AsyncMock(return_value=None)
            mock_repos.images.store_images = AsyncMock(return_value=None)
            mock_ensure_repos.return_value = mock_repos

            # Send tags as native list
            result = await store_context.fn(
                thread_id='test-thread',
                source='user',
                text='Test content',
                tags=['tag1', 'tag2', 'tag3'],  # Native list
            )

            assert result['success'] is True
            assert result['context_id'] == 1

    @pytest.mark.asyncio
    async def test_store_context_with_json_string_metadata(self):
        """Test store_context accepts metadata as JSON string."""
        from app.server import store_context

        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.tags.store_tags = AsyncMock(return_value=None)
            mock_repos.images.store_images = AsyncMock(return_value=None)
            mock_ensure_repos.return_value = mock_repos

            # Send metadata as JSON string
            result = await store_context.fn(
                thread_id='test-thread',
                source='agent',
                text='Test content',
                metadata='{"key": "value", "number": 42}',  # JSON string
            )

            assert result['success'] is True
            assert result['context_id'] == 1

    @pytest.mark.asyncio
    async def test_store_context_with_json_string_images(self):
        """Test store_context accepts images as JSON string."""
        from app.server import store_context

        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.tags.store_tags = AsyncMock(return_value=None)
            mock_repos.images.store_images = AsyncMock(return_value=None)
            mock_ensure_repos.return_value = mock_repos

            # Send images as JSON string
            images_json = json.dumps([
                {'data': 'dGVzdCBpbWFnZQ==', 'mime_type': 'image/png'},
            ])
            result = await store_context.fn(
                thread_id='test-thread',
                source='user',
                text='Test with image',
                images=images_json,  # JSON string
            )

            assert result['success'] is True
            assert 'with 1 images' in result['message']

    @pytest.mark.asyncio
    async def test_store_context_with_all_json_strings(self):
        """Test store_context with all complex parameters as JSON strings."""
        from app.server import store_context

        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.tags.store_tags = AsyncMock(return_value=None)
            mock_repos.images.store_images = AsyncMock(return_value=None)
            mock_ensure_repos.return_value = mock_repos

            # Send all complex types as JSON strings
            result = await store_context.fn(
                thread_id='test-thread',
                source='agent',
                text='Complex test',
                tags='["tag1", "tag2"]',
                metadata='{"source": "test", "version": 1}',
                images='[{"data": "dGVzdA==", "mime_type": "image/jpeg"}]',
            )

            assert result['success'] is True
            assert result['context_id'] == 1


class TestSearchContextParameterHandling:
    """Test search_context with different parameter formats."""

    @pytest.mark.asyncio
    async def test_search_context_with_json_string_tags(self):
        """Test search_context accepts tags as JSON string."""
        from app.server import search_context

        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.search_contexts = AsyncMock(return_value=[])
            mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
            mock_repos.images.get_images_for_context = AsyncMock(return_value=[])
            mock_ensure_repos.return_value = mock_repos

            # Send tags as JSON string
            result = await search_context.fn(
                thread_id='test-thread',
                tags='["tag1", "tag2"]',  # JSON string
            )

            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_context_with_native_tags(self):
        """Test search_context accepts tags as native list."""
        from app.server import search_context

        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.search_contexts = AsyncMock(return_value=[])
            mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
            mock_repos.images.get_images_for_context = AsyncMock(return_value=[])
            mock_ensure_repos.return_value = mock_repos

            # Send tags as native list
            result = await search_context.fn(
                thread_id='test-thread',
                tags=['tag1', 'tag2'],  # Native list
            )

            assert isinstance(result, list)


class TestGetContextByIdsParameterHandling:
    """Test get_context_by_ids with different parameter formats."""

    @pytest.mark.asyncio
    async def test_get_context_by_ids_with_json_string(self):
        """Test get_context_by_ids accepts IDs as JSON string."""
        from app.server import get_context_by_ids

        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.get_by_ids = AsyncMock(return_value=[])
            mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
            mock_repos.images.get_images_for_context = AsyncMock(return_value=[])
            mock_ensure_repos.return_value = mock_repos

            # Send context_ids as JSON string
            result = await get_context_by_ids.fn(
                context_ids='[1, 2, 3]',  # JSON string
            )

            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_context_by_ids_with_native_list(self):
        """Test get_context_by_ids accepts IDs as native list."""
        from app.server import get_context_by_ids

        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.get_by_ids = AsyncMock(return_value=[])
            mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
            mock_repos.images.get_images_for_context = AsyncMock(return_value=[])
            mock_ensure_repos.return_value = mock_repos

            # Send context_ids as native list
            result = await get_context_by_ids.fn(
                context_ids=[1, 2, 3],  # Native list
            )

            assert isinstance(result, list)


class TestDeleteContextParameterHandling:
    """Test delete_context with different parameter formats."""

    @pytest.mark.asyncio
    async def test_delete_context_with_json_string_ids(self):
        """Test delete_context accepts IDs as JSON string."""
        from app.server import delete_context

        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.delete_by_ids = AsyncMock(return_value=3)
            mock_repos.context.delete_by_thread = AsyncMock(return_value=3)
            mock_ensure_repos.return_value = mock_repos

            # Send context_ids as JSON string
            result = await delete_context.fn(
                context_ids='[1, 2, 3]',  # JSON string
            )

            assert result['success'] is True
            assert result['deleted_count'] == 3

    @pytest.mark.asyncio
    async def test_delete_context_with_native_list(self):
        """Test delete_context accepts IDs as native list."""
        from app.server import delete_context

        with patch('app.server._db_manager') as mock_manager:
            mock_manager.execute_write = AsyncMock(return_value=2)

            # Send context_ids as native list
            result = await delete_context.fn(
                context_ids=[1, 2],  # Native list
            )

            assert result['success'] is True
            assert result['deleted_count'] == 2


class TestMixedParameterFormats:
    """Test functions with mixed parameter formats to ensure robustness."""

    @pytest.mark.asyncio
    async def test_store_context_mixed_formats(self, async_db_initialized):
        """Test store_context with some params as JSON strings, others native."""
        from app.server import store_context

        # Ensure fixture is used (it initializes the database)
        assert async_db_initialized is not None

        # Mix of formats
        result = await store_context.fn(
            thread_id='test-thread',
            source='user',
            text='Mixed format test',
            tags='["json", "string"]',  # JSON string
            metadata={'native': 'dict'},  # Native dict
        )

        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases like empty strings, empty lists, etc."""
        # Empty JSON array string
        assert deserialize_json_param('[]') == []

        # Empty JSON object string
        assert deserialize_json_param('{}') == {}

        # Null JSON value
        assert deserialize_json_param('null') is None

        # Boolean JSON values
        assert deserialize_json_param('true') is True
        assert deserialize_json_param('false') is False

        # Nested structures
        nested = '{"list": [1, 2, {"nested": true}], "key": "value"}'
        result = deserialize_json_param(nested)
        assert result == {'list': [1, 2, {'nested': True}], 'key': 'value'}
