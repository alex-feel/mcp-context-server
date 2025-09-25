"""Test JSON string parameter handling from Claude Code and other clients."""

import json
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import app.server
import pytest
from app.server import deserialize_json_param
from app.types import JsonValue

# Get the actual async functions from the FunctionTool wrappers
store_context = app.server.store_context.fn
search_context = app.server.search_context.fn
delete_context = app.server.delete_context.fn
get_context_by_ids = app.server.get_context_by_ids.fn


class TestJSONStringDeserialization:
    """Test the deserialize_json_param helper function."""

    def test_deserialize_list_as_string(self):
        """Test deserializing a list passed as JSON string."""
        json_string = '["test", "validation"]'
        result = deserialize_json_param(json_string)
        assert result == ['test', 'validation']

    def test_deserialize_dict_as_string(self):
        """Test deserializing a dict passed as JSON string."""
        json_string = '{"key": "value", "number": 42}'
        result = deserialize_json_param(json_string)
        assert result == {'key': 'value', 'number': 42}

    def test_deserialize_list_of_ints_as_string(self):
        """Test deserializing a list of integers passed as JSON string."""
        json_string = '[1, 2, 3, 4]'
        result = deserialize_json_param(json_string)
        assert result == [1, 2, 3, 4]

    def test_deserialize_native_list(self):
        """Test that native Python lists are passed through unchanged."""
        native_list = ['test', 'validation']
        result = deserialize_json_param(cast(JsonValue, native_list))
        assert result is native_list

    def test_deserialize_native_dict(self):
        """Test that native Python dicts are passed through unchanged."""
        native_dict = {'key': 'value', 'number': 42}
        result = deserialize_json_param(native_dict)
        assert result is native_dict

    def test_deserialize_none(self):
        """Test that None is passed through unchanged."""
        result = deserialize_json_param(None)
        assert result is None

    def test_deserialize_regular_string(self):
        """Test that regular strings are passed through unchanged."""
        regular_string = 'just a normal string'
        result = deserialize_json_param(regular_string)
        assert result == regular_string

    def test_deserialize_invalid_json(self):
        """Test that invalid JSON strings are passed through as-is."""
        invalid_json = '{"key": invalid}'
        result = deserialize_json_param(invalid_json)
        assert result == invalid_json

    def test_deserialize_empty_string(self):
        """Test that empty strings are passed through."""
        result = deserialize_json_param('')
        assert result == ''


class TestStoreContextWithJSONStrings:
    """Test store_context with JSON string parameters."""

    @pytest.mark.asyncio
    async def test_store_context_with_json_string_tags(self):
        """Test store_context with tags as JSON string (from Claude Code)."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.tags.store_tags = AsyncMock(return_value=None)
            mock_repos.images.store_images = AsyncMock(return_value=None)
            mock_ensure_repos.return_value = mock_repos
            # Simulate Claude Code sending tags as JSON string
            result = await store_context(
                    thread_id='test-thread',
                    source='user',
                    text='Test message',
                    tags='["python", "testing", "claude"]',  # JSON string
                )

            assert result['success'] is True
            assert result['context_id'] == 1

    @pytest.mark.asyncio
    async def test_store_context_with_json_string_metadata(self):
        """Test store_context with metadata as JSON string (from Claude Code)."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.tags.store_tags = AsyncMock(return_value=None)
            mock_repos.images.store_images = AsyncMock(return_value=None)
            mock_ensure_repos.return_value = mock_repos

            # Simulate Claude Code sending metadata as JSON string
            result = await store_context(
                thread_id='test-thread',
                source='agent',
                text='Test message',
                metadata='{"key": "value", "count": 42}',  # JSON string
            )

            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_store_context_with_json_string_images(self):
        """Test store_context with images as JSON string (from Claude Code)."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.tags.store_tags = AsyncMock(return_value=None)
            mock_repos.images.store_images = AsyncMock(return_value=None)
            mock_ensure_repos.return_value = mock_repos

            # Simulate Claude Code sending images as JSON string
            images_json = json.dumps([
                {'data': 'aGVsbG8=', 'mime_type': 'image/png'},  # base64 for 'hello'
            ])

            result = await store_context(
                thread_id='test-thread',
                source='user',
                text='Test with image',
                images=images_json,  # JSON string
            )

            assert result['success'] is True
            assert 'with 1 images' in result['message']

    @pytest.mark.asyncio
    async def test_store_context_with_native_types(self):
        """Test store_context still works with native Python types."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.tags.store_tags = AsyncMock(return_value=None)
            mock_repos.images.store_images = AsyncMock(return_value=None)
            mock_ensure_repos.return_value = mock_repos
            # Native Python types (from other clients)
            result = await store_context(
                    thread_id='test-thread',
                    source='user',
                    text='Test message',
                    tags=['python', 'testing'],  # Native list
                    metadata={'key': 'value'},  # Native dict
                )

            assert result['success'] is True


class TestSearchContextWithJSONStrings:
    """Test search_context with JSON string parameters."""

    @pytest.mark.asyncio
    async def test_search_context_with_json_string_tags(self):
        """Test search_context with tags as JSON string (from Claude Code)."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.search_contexts = AsyncMock(return_value=[])
            mock_repos.context.get_by_ids = AsyncMock(return_value=[])
            mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
            mock_repos.images.get_images_for_context = AsyncMock(return_value=[])
            mock_ensure_repos.return_value = mock_repos

            # Simulate Claude Code sending tags as JSON string
            result = await search_context(
                thread_id='test-thread',
                tags='["python", "testing"]',  # JSON string
            )

            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_context_with_native_tags(self):
        """Test search_context still works with native Python list."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.search_contexts = AsyncMock(return_value=[])
            mock_repos.context.get_by_ids = AsyncMock(return_value=[])
            mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
            mock_repos.images.get_images_for_context = AsyncMock(return_value=[])
            mock_ensure_repos.return_value = mock_repos

            # Native Python list (from other clients)
            result = await search_context(
                thread_id='test-thread',
                tags=['python', 'testing'],  # Native list
            )

            assert isinstance(result, list)


class TestDeleteContextWithJSONStrings:
    """Test delete_context with JSON string parameters."""

    @pytest.mark.asyncio
    async def test_delete_context_with_json_string_ids(self):
        """Test delete_context with context_ids as JSON string (from Claude Code)."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.delete_by_ids = AsyncMock(return_value=3)
            mock_repos.context.delete_by_thread = AsyncMock(return_value=3)
            mock_ensure_repos.return_value = mock_repos

            # Simulate Claude Code sending context_ids as JSON string
            result = await delete_context(
                context_ids='[1, 2, 3]',  # JSON string
            )

            assert result['success'] is True
            assert result['deleted_count'] == 3

    @pytest.mark.asyncio
    async def test_delete_context_with_native_list(self):
        """Test delete_context still works with native Python list."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.delete_by_ids = AsyncMock(return_value=2)
            mock_repos.context.delete_by_thread = AsyncMock(return_value=2)
            mock_ensure_repos.return_value = mock_repos

            # Native Python list (from other clients)
            result = await delete_context(
                context_ids=[1, 2],  # Native list
            )

            assert result['success'] is True
            assert result['deleted_count'] == 2


class TestGetContextByIdsWithJSONStrings:
    """Test get_context_by_ids with JSON string parameters."""

    @pytest.mark.asyncio
    async def test_get_context_by_ids_with_json_string(self):
        """Test get_context_by_ids with context_ids as JSON string (from Claude Code)."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.search_contexts = AsyncMock(return_value=[])
            mock_repos.context.get_by_ids = AsyncMock(return_value=[])
            mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
            mock_repos.images.get_images_for_context = AsyncMock(return_value=[])
            mock_ensure_repos.return_value = mock_repos

            # Simulate Claude Code sending context_ids as JSON string
            result = await get_context_by_ids(
                context_ids='[1, 2, 3]',  # JSON string
            )

            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_context_by_ids_with_native_list(self):
        """Test get_context_by_ids still works with native Python list."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.search_contexts = AsyncMock(return_value=[])
            mock_repos.context.get_by_ids = AsyncMock(return_value=[])
            mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
            mock_repos.images.get_images_for_context = AsyncMock(return_value=[])
            mock_ensure_repos.return_value = mock_repos

            # Native Python list (from other clients)
            result = await get_context_by_ids(
                context_ids=[1, 2],  # Native list
            )

            assert isinstance(result, list)


class TestMixedScenarios:
    """Test mixed scenarios with both JSON strings and native types."""

    @pytest.mark.asyncio
    async def test_all_json_strings_together(self):
        """Test with all parameters as JSON strings simultaneously."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.tags.store_tags = AsyncMock(return_value=None)
            mock_repos.images.store_images = AsyncMock(return_value=None)
            mock_ensure_repos.return_value = mock_repos

            # All complex parameters as JSON strings (Claude Code style)
            result = await store_context(
                    thread_id='test-thread',
                    source='agent',
                    text='Test with all JSON strings',
                    tags='["tag1", "tag2", "tag3"]',  # JSON string
                    metadata='{"key": "value", "nested": {"inner": 42}}',  # JSON string
                    images='[{"data": "aGVsbG8=", "mime_type": "image/png"}]',  # JSON string
                )

            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_partial_json_strings(self):
        """Test with some parameters as JSON strings and others as native types."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.tags.store_tags = AsyncMock(return_value=None)
            mock_repos.images.store_images = AsyncMock(return_value=None)
            mock_ensure_repos.return_value = mock_repos

            # Mixed: tags as JSON string, metadata as native dict
            result = await store_context(
                    thread_id='test-thread',
                    source='user',
                    text='Mixed parameters test',
                    tags='["json", "string"]',  # JSON string
                    metadata={'native': 'dict'},  # Native dict
                )

            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_invalid_json_string_handling(self):
        """Test that invalid JSON strings don't crash the server."""
        with patch('app.server._ensure_repositories') as mock_ensure_repos:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.tags.store_tags = AsyncMock(return_value=None)
            mock_repos.images.store_images = AsyncMock(return_value=None)
            mock_ensure_repos.return_value = mock_repos

            # Invalid JSON string for tags (will be treated as a regular string)
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text='Test with invalid JSON',
                tags='[invalid json',  # Invalid JSON
                metadata={'valid': 'dict'},  # Valid native dict
            )

            # Should still succeed but tags will be treated as a string
            assert result['success'] is True
