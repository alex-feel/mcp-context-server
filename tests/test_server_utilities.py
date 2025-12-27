"""Tests for server utility functions.

This module tests utility functions in app/server.py including
text truncation, JSON deserialization, and helper functions.
"""

from __future__ import annotations

import json
import math
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.server import deserialize_json_param
from app.server import truncate_text


class TestTruncateText:
    """Test the truncate_text utility function."""

    def test_truncate_none_text(self) -> None:
        """Test that None text returns (None, False)."""
        result, is_truncated = truncate_text(None)
        assert result is None
        assert is_truncated is False

    def test_truncate_empty_text(self) -> None:
        """Test that empty text returns ('', False)."""
        result, is_truncated = truncate_text('')
        assert result == ''
        assert is_truncated is False

    def test_truncate_short_text(self) -> None:
        """Test that short text is not truncated."""
        short_text = 'This is a short text.'
        result, is_truncated = truncate_text(short_text)
        assert result == short_text
        assert is_truncated is False

    def test_truncate_text_at_default_length(self) -> None:
        """Test text at exactly 150 characters is not truncated."""
        text_150 = 'x' * 150
        result, is_truncated = truncate_text(text_150)
        assert result == text_150
        assert is_truncated is False

    def test_truncate_long_text(self) -> None:
        """Test that long text is truncated with ellipsis."""
        long_text = 'x' * 200
        result, is_truncated = truncate_text(long_text)
        assert is_truncated is True
        assert result is not None
        assert result.endswith('...')
        assert len(result) <= 153  # 150 + '...'

    def test_truncate_custom_max_length(self) -> None:
        """Test truncation with custom max_length."""
        text = 'This is a test text that should be truncated.'
        result, is_truncated = truncate_text(text, max_length=20)
        assert is_truncated is True
        assert result is not None
        assert result.endswith('...')
        assert len(result) <= 23  # 20 + '...'

    def test_truncate_at_word_boundary(self) -> None:
        """Test that truncation happens at word boundaries when possible."""
        # Text with a space after position 105 (70% of 150)
        text = 'This is a sentence with several words ' + 'x' * 115 + ' end'
        result, is_truncated = truncate_text(text)
        assert is_truncated is True
        assert result is not None
        assert result.endswith('...')

    def test_truncate_no_word_boundary(self) -> None:
        """Test truncation when no good word boundary exists."""
        # Text without spaces within the 70% threshold
        text = 'x' * 200  # No spaces at all
        result, is_truncated = truncate_text(text)
        assert is_truncated is True
        assert result is not None
        assert result.endswith('...')
        assert len(result) == 153  # Exact truncation at 150 + '...'

    def test_truncate_word_boundary_threshold(self) -> None:
        """Test word boundary threshold (70% of max_length)."""
        # max_length=150, 70% = 105
        # Place a space at position 100 (below threshold) and another at 110 (above)
        text = 'x' * 100 + ' ' + 'y' * 9 + ' ' + 'z' * 100
        result, is_truncated = truncate_text(text)
        assert is_truncated is True
        # Should truncate at position 110 (the space above threshold)
        assert result is not None
        assert result.endswith('...')


class TestDeserializeJsonParam:
    """Test the deserialize_json_param utility function."""

    def test_deserialize_none(self) -> None:
        """Test that None returns None."""
        result = deserialize_json_param(None)
        assert result is None

    def test_deserialize_non_string_passthrough(self) -> None:
        """Test that non-string values pass through unchanged."""
        # Integer - primitive JsonValue type
        int_val: int = 42
        result = deserialize_json_param(int_val)
        assert result == int_val

        # Boolean - primitive JsonValue type
        bool_val: bool = True
        result = deserialize_json_param(bool_val)
        assert result is True

        # Float - primitive JsonValue type
        float_val: float = math.pi
        result = deserialize_json_param(float_val)
        assert result == float_val

    def test_deserialize_list_passthrough(self) -> None:
        """Test that list values pass through unchanged."""
        # Test with JSON string that deserializes to list
        json_list = '["a", "b", "c"]'
        result = deserialize_json_param(json_list)
        assert result == ['a', 'b', 'c']

    def test_deserialize_dict_passthrough(self) -> None:
        """Test that dict values pass through unchanged."""
        # Test with JSON string that deserializes to dict
        json_dict = '{"key": 1}'
        result = deserialize_json_param(json_dict)
        assert result == {'key': 1}

    def test_deserialize_json_string_to_list(self) -> None:
        """Test deserializing JSON string to list."""
        json_str = '["tag1", "tag2", "tag3"]'
        result = deserialize_json_param(json_str)
        assert result == ['tag1', 'tag2', 'tag3']

    def test_deserialize_json_string_to_dict(self) -> None:
        """Test deserializing JSON string to dict."""
        json_str = '{"key": "value", "number": 42}'
        result = deserialize_json_param(json_str)
        assert result == {'key': 'value', 'number': 42}

    def test_deserialize_invalid_json_string(self) -> None:
        """Test that invalid JSON strings return as-is."""
        invalid_json = 'not valid json {'
        result = deserialize_json_param(invalid_json)
        assert result == invalid_json

    def test_deserialize_plain_string(self) -> None:
        """Test that plain strings return as-is."""
        plain_str = 'just a plain string'
        result = deserialize_json_param(plain_str)
        assert result == plain_str

    def test_deserialize_double_encoded_json(self) -> None:
        """Test handling of double-encoded JSON strings."""
        # This simulates: json.dumps(json.dumps(['a', 'b']))
        inner = json.dumps(['a', 'b'])
        double_encoded = json.dumps(inner)
        result = deserialize_json_param(double_encoded)
        assert result == ['a', 'b']

    def test_deserialize_json_number_string(self) -> None:
        """Test deserializing JSON number string."""
        json_str = '42'
        result = deserialize_json_param(json_str)
        assert result == 42

    def test_deserialize_json_boolean_string(self) -> None:
        """Test deserializing JSON boolean string."""
        result_true = deserialize_json_param('true')
        assert result_true is True

        result_false = deserialize_json_param('false')
        assert result_false is False

    def test_deserialize_json_null_string(self) -> None:
        """Test deserializing JSON null string."""
        result = deserialize_json_param('null')
        assert result is None

    def test_deserialize_whitespace_string(self) -> None:
        """Test that whitespace-only strings return as-is."""
        result = deserialize_json_param('   ')
        assert result == '   '

    def test_deserialize_string_with_special_chars(self) -> None:
        """Test strings with special characters that aren't valid JSON."""
        special = 'path/to/file.txt'
        result = deserialize_json_param(special)
        assert result == special

    def test_deserialize_complex_nested_json(self) -> None:
        """Test deserializing complex nested JSON."""
        complex_obj: dict[str, Any] = {
            'nested': {'level': {'deep': [1, 2, 3]}},
            'array': ['a', 'b'],
            'number': 42.5,
        }
        json_str = json.dumps(complex_obj)
        result = deserialize_json_param(json_str)
        assert result == complex_obj


class TestServerEnsureFunctions:
    """Test _ensure_backend and _ensure_repositories functions."""

    @pytest.mark.asyncio
    async def test_ensure_backend_creates_new(self) -> None:
        """Test that _ensure_backend creates a new backend when none exists."""
        import app.server as server

        # Store original
        original_backend = server._backend

        try:
            # Clear the backend
            server._backend = None

            # Mock create_backend and initialize
            mock_backend = MagicMock()
            mock_backend.initialize = AsyncMock()

            with patch('app.server.create_backend', return_value=mock_backend) as mock_create:
                backend = await server._ensure_backend()

                # Verify backend was created and initialized
                mock_create.assert_called_once()
                mock_backend.initialize.assert_awaited_once()
                assert backend == mock_backend
        finally:
            # Restore original
            server._backend = original_backend

    @pytest.mark.asyncio
    async def test_ensure_backend_returns_existing(self) -> None:
        """Test that _ensure_backend returns existing backend."""
        import app.server as server

        # Store original
        original_backend = server._backend

        try:
            # Set a mock backend
            mock_backend = MagicMock()
            server._backend = mock_backend

            with patch('app.server.create_backend') as mock_create:
                backend = await server._ensure_backend()

                # Should return existing, not create new
                mock_create.assert_not_called()
                assert backend == mock_backend
        finally:
            # Restore original
            server._backend = original_backend

    @pytest.mark.asyncio
    async def test_ensure_repositories_creates_new(self) -> None:
        """Test that _ensure_repositories creates new when none exists."""
        import app.server as server
        from app.repositories import RepositoryContainer

        # Store originals
        original_backend = server._backend
        original_repos = server._repositories

        try:
            # Set up a mock backend
            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            server._backend = mock_backend
            server._repositories = None

            repos = await server._ensure_repositories()

            # Verify repositories were created
            assert isinstance(repos, RepositoryContainer)
            assert server._repositories is not None
        finally:
            # Restore originals
            server._backend = original_backend
            server._repositories = original_repos

    @pytest.mark.asyncio
    async def test_ensure_repositories_returns_existing(self) -> None:
        """Test that _ensure_repositories returns existing."""
        import app.server as server

        # Store originals
        original_repos = server._repositories

        try:
            # Set a mock repository container
            mock_repos = MagicMock()
            server._repositories = mock_repos

            repos = await server._ensure_repositories()

            # Should return existing
            assert repos == mock_repos
        finally:
            # Restore original
            server._repositories = original_repos


class TestStoreContextEdgeCases:
    """Test edge cases for store_context function."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_whitespace_thread_id(self) -> None:
        """Test that whitespace-only thread_id is rejected."""
        from fastmcp.exceptions import ToolError

        from app.server import store_context

        with pytest.raises(ToolError, match='thread_id cannot be empty or whitespace'):
            await store_context(
                thread_id='   ',  # Whitespace only
                source='user',
                text='Some text',
            )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_whitespace_text(self) -> None:
        """Test that whitespace-only text is rejected."""
        from fastmcp.exceptions import ToolError

        from app.server import store_context

        with pytest.raises(ToolError, match='text cannot be empty or whitespace'):
            await store_context(
                thread_id='test_thread',
                source='user',
                text='   ',  # Whitespace only
            )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_missing_image_data(self) -> None:
        """Test that images without 'data' field are rejected."""
        from fastmcp.exceptions import ToolError

        from app.server import store_context

        with pytest.raises(ToolError, match='missing required "data" field'):
            await store_context(
                thread_id='test_thread',
                source='user',
                text='Some text',
                images=[{'mime_type': 'image/png'}],  # Missing 'data'
            )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_empty_image_data(self) -> None:
        """Test that images with empty 'data' field are rejected."""
        from fastmcp.exceptions import ToolError

        from app.server import store_context

        with pytest.raises(ToolError, match='empty "data" field'):
            await store_context(
                thread_id='test_thread',
                source='user',
                text='Some text',
                images=[{'data': '', 'mime_type': 'image/png'}],
            )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_default_mime_type(self) -> None:
        """Test that missing mime_type defaults to image/png."""
        import base64

        from app.server import get_context_by_ids
        from app.server import store_context

        image_data = base64.b64encode(b'test_image').decode('utf-8')
        result = await store_context(
            thread_id='test_default_mime',
            source='user',
            text='Test image',
            images=[{'data': image_data}],  # No mime_type specified
        )

        assert result['success'] is True

        # Verify the image was stored
        context = await get_context_by_ids(
            context_ids=[result['context_id']],
            include_images=True,
        )
        assert len(context) == 1
        # Convert TypedDict to regular dict for test assertion
        entry = dict(context[0])
        assert entry['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_failed_store_returns_null_id(self) -> None:
        """Test that failed store with null context_id raises ToolError."""
        from unittest.mock import AsyncMock

        from fastmcp.exceptions import ToolError

        from app.server import store_context

        # Mock the repository to return (None, False)
        with patch('app.server._ensure_repositories') as mock_ensure:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(None, False))
            mock_ensure.return_value = mock_repos

            with pytest.raises(ToolError, match='Failed to store context'):
                await store_context(
                    thread_id='test_thread',
                    source='user',
                    text='Some text',
                )


class TestUpdateContextEdgeCases:
    """Test edge cases for update_context function."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_no_fields_provided(self) -> None:
        """Test that update with no fields raises ToolError."""
        from fastmcp.exceptions import ToolError

        from app.server import store_context
        from app.server import update_context

        # First create a context
        result = await store_context(
            thread_id='test_update',
            source='user',
            text='Original text',
        )
        context_id = result['context_id']

        # Try to update with no fields
        with pytest.raises(ToolError, match='At least one field must be provided'):
            await update_context(context_id=context_id)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_whitespace_text(self) -> None:
        """Test that update with whitespace-only text is rejected."""
        from fastmcp.exceptions import ToolError

        from app.server import store_context
        from app.server import update_context

        # First create a context
        result = await store_context(
            thread_id='test_update_ws',
            source='user',
            text='Original text',
        )
        context_id = result['context_id']

        # Try to update with whitespace text
        with pytest.raises(ToolError, match='text cannot be empty'):
            await update_context(context_id=context_id, text='   ')

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_nonexistent_id(self) -> None:
        """Test that update of nonexistent context raises ToolError."""
        from fastmcp.exceptions import ToolError

        from app.server import update_context

        with pytest.raises(ToolError, match='not found'):
            await update_context(context_id=999999, text='New text')

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_clear_images(self) -> None:
        """Test that setting images to empty list removes all images."""
        import base64

        from app.server import get_context_by_ids
        from app.server import store_context
        from app.server import update_context

        image_data = base64.b64encode(b'test').decode('utf-8')

        # Create with image
        result = await store_context(
            thread_id='test_clear_images',
            source='user',
            text='With image',
            images=[{'data': image_data, 'mime_type': 'image/png'}],
        )
        context_id = result['context_id']

        # Update with empty images list
        update_result = await update_context(context_id=context_id, images=[])

        assert update_result['success'] is True
        assert 'images' in update_result['updated_fields']
        assert 'content_type' in update_result['updated_fields']

        # Verify images were removed
        context = await get_context_by_ids(
            context_ids=[context_id],
            include_images=True,
        )
        # Convert TypedDict to regular dict for test assertion
        entry = dict(context[0])
        assert entry['content_type'] == 'text'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_image_size_limit(self) -> None:
        """Test that oversized images in update are rejected."""
        import base64

        from fastmcp.exceptions import ToolError

        from app.server import store_context
        from app.server import update_context

        # Create a context
        result = await store_context(
            thread_id='test_image_limit',
            source='user',
            text='Original',
        )
        context_id = result['context_id']

        # Create oversized image data (6MB)
        large_data = base64.b64encode(b'x' * (6 * 1024 * 1024)).decode('utf-8')

        with pytest.raises(ToolError, match='exceeds size limit'):
            await update_context(
                context_id=context_id,
                images=[{'data': large_data, 'mime_type': 'image/png'}],
            )


class TestDeleteContextEdgeCases:
    """Test edge cases for delete_context function."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_delete_empty_thread(self) -> None:
        """Test deleting from nonexistent thread returns 0."""
        from app.server import delete_context

        result = await delete_context(thread_id='nonexistent_thread_xyz')

        assert result['success'] is True
        assert result['deleted_count'] == 0


class TestSemanticSearchNotAvailable:
    """Test semantic search tool when not available."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_semantic_search_not_available(self) -> None:
        """Test that semantic_search_context raises error when service not available."""
        from fastmcp.exceptions import ToolError

        import app.server as server
        from app.server import semantic_search_context

        # Store original
        original_service = server._embedding_service

        try:
            # Clear the embedding service
            server._embedding_service = None

            with pytest.raises(ToolError, match='Semantic search is not available'):
                await semantic_search_context(query='test query', limit=20)
        finally:
            # Restore original
            server._embedding_service = original_service
