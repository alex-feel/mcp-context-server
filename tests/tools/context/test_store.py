"""Tests for the ``store_context`` tool: text and multimodal stores, input validation, database errors, and
unusual input (unicode, large metadata, SQL metacharacters).
"""

import sqlite3
from pathlib import Path
from typing import Any
from typing import Literal
from typing import cast
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools

# The tool functions are plain coroutines that lifespan() registers with FastMCP at startup; tests call them directly.
store_context = app.tools.store_context
search_context = app.tools.search_context
get_context_by_ids = app.tools.get_context_by_ids
get_statistics = app.tools.get_statistics


@pytest.mark.usefixtures('initialized_server')
class TestStoreContext:
    """Test the store_context MCP tool."""

    @pytest.mark.asyncio
    async def test_store_text_context(
        self,
        sample_context_data: dict[str, Any],
    ) -> None:
        """Test storing a simple text context entry."""
        result = await store_context(
            thread_id=sample_context_data['thread_id'],
            source=sample_context_data['source'],
            text=sample_context_data['text'],
            metadata=sample_context_data['metadata'],
            tags=sample_context_data['tags'],
        )

        assert result['success'] is True
        assert 'context_id' in result
        assert result['thread_id'] == sample_context_data['thread_id']
        assert 'Context stored' in result['message']

    @pytest.mark.asyncio
    async def test_store_multimodal_context(
        self,
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
        )

        assert result['success'] is True
        assert 'context_id' in result
        assert 'Context stored with 1 images' in result['message']

    @pytest.mark.asyncio
    async def test_store_context_no_content(self) -> None:
        """Test that empty text is properly validated in the function body.

        We test validation in the function body, not Pydantic.
        """
        with pytest.raises(ToolError, match='text cannot be empty or whitespace'):
            await store_context(
                thread_id='test_thread',
                source='user',
                text='',  # Empty text should fail in function validation
            )

    @pytest.mark.asyncio
    async def test_store_context_invalid_source(self) -> None:
        """Test that invalid source bypasses Pydantic and hits database CHECK constraint.

        Note: Pydantic Literal['user', 'agent'] handles validation at FastMCP level.
        This test uses cast() to bypass Pydantic and verify database constraint works.
        """
        with pytest.raises(ToolError, match='CHECK constraint failed|source'):
            await store_context(
                thread_id='test_thread',
                source=cast(Literal['user', 'agent'], 'invalid_source'),
                text='Some text',
            )

    @pytest.mark.asyncio
    async def test_store_context_oversized_image(
        self,
        large_image_data: dict[str, str],
    ) -> None:
        """Test error when image exceeds size limit."""
        with pytest.raises(ToolError, match='exceeds.*MB limit'):
            await store_context(
                thread_id='test_thread',
                source='user',
                text='Text with large image',
                images=[large_image_data],
            )

    @pytest.mark.asyncio
    async def test_store_context_invalid_base64(self) -> None:
        """Test error with invalid base64 image data."""
        with pytest.raises(ToolError, match='Image 0 has invalid base64 encoding'):
            await store_context(
                thread_id='test_thread',
                source='agent',
                text='Text with bad image',
                images=[{'data': 'not-valid-base64!', 'mime_type': 'image/png'}],
            )

    @pytest.mark.asyncio
    async def test_store_multiple_images(
        self,
        sample_image_data: dict[str, str],
    ) -> None:
        """Test storing multiple images."""
        import json as _json

        images = [
            sample_image_data,
            {**sample_image_data, 'metadata': _json.dumps({'position': 1})},
            {**sample_image_data, 'metadata': _json.dumps({'position': 2})},
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
        # Mock the repository method to raise an error
        _ = temp_db_path  # Acknowledge unused parameter
        with patch('app.repositories.context_repository.ContextRepository.store_with_deduplication') as mock_store:
            mock_store.side_effect = sqlite3.OperationalError('Database error')
            with pytest.raises(ToolError, match='Failed to store context'):
                await store_context(
                    thread_id='test',
                    source='user',
                    text='This should fail',
                )


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
        search_result = await search_context(limit=50, thread_id='unicode_test')
        assert isinstance(search_result, dict)
        assert search_result['results'][0]['text_content'] == unicode_text

    @pytest.mark.asyncio
    async def test_large_metadata(self) -> None:
        """Test handling of large metadata objects."""
        from app.types import JsonValue

        large_metadata = cast(
            'dict[str, JsonValue]',
            {
                'nested': {
                    'level': {
                        'data': ['item'] * 100,
                        'numbers': list(range(1000)),
                    },
                },
                'description': 'x' * 10000,
            },
        )

        result = await store_context(
            thread_id='metadata_test',
            source='agent',
            text='Large metadata',
            metadata=large_metadata,
        )

        assert result['success'] is True

        # Verify retrieval
        fetched = await get_context_by_ids(context_ids=[result['context_id']])
        entry = dict(fetched[0])
        assert entry['metadata'] == large_metadata

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
        search_result = await search_context(limit=50, thread_id=malicious_thread)
        assert isinstance(search_result, dict)
        assert len(search_result['results']) == 1
        # Tag should be normalized to lowercase
        normalized_tag = malicious_tag.strip().lower()
        assert normalized_tag in search_result['results'][0]['tags']

        # Tables should still exist
        stats = await get_statistics()
        assert stats['total_entries'] > 0
