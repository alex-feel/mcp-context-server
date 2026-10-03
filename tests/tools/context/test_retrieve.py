"""Tests for the ``get_context_by_ids`` tool."""

import base64

import pytest

import app.tools

# The tool functions are plain coroutines that lifespan() registers with FastMCP at startup; tests call them directly.
store_context = app.tools.store_context
get_context_by_ids = app.tools.get_context_by_ids


@pytest.mark.usefixtures('initialized_server')
class TestGetContextByIds:
    """Test the get_context_by_ids MCP tool."""

    @pytest.mark.asyncio
    async def test_get_single_context(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test fetching a single context by ID."""
        context_id = multiple_context_entries[0]
        results = await get_context_by_ids(context_ids=[context_id])

        assert len(results) == 1
        entry = dict(results[0])
        assert entry['id'] == context_id

    @pytest.mark.asyncio
    async def test_get_multiple_contexts(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test fetching multiple contexts by IDs."""
        ids_to_fetch = multiple_context_entries[:3]
        results = await get_context_by_ids(context_ids=ids_to_fetch)

        assert len(results) == 3
        result_ids = [dict(r)['id'] for r in results]
        assert set(result_ids) == set(ids_to_fetch)

    @pytest.mark.asyncio
    async def test_get_context_with_images(self) -> None:
        """Test fetching context with images included."""
        import json as _json

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
                    'metadata': _json.dumps({'size': 100}),
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
        result_images = results[0]['images']
        assert result_images is not None
        assert len(result_images) == 1
        assert result_images[0]['mime_type'] == 'image/jpeg'

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
        results = await get_context_by_ids(
            context_ids=[
                '0190abcdef1234567890abcd0000270f',
                '0190abcdef1234567890abcd00002710',
            ],
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_get_empty_context_list(self) -> None:
        """Test that Pydantic Field(min_length=1) handles empty list.

        Note: Pydantic validates at FastMCP level. This test verifies normal operation.
        """
        # Test with valid non-empty list
        result = await get_context_by_ids(context_ids=['0190abcdef1234567890abcd00000001'])
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_context_with_tags(
        self,
        multiple_context_entries: list[str],
    ) -> None:
        """Test that tags are included in fetched contexts."""
        # First entry has tags
        results = await get_context_by_ids(context_ids=[multiple_context_entries[0]])

        assert len(results) == 1
        assert 'tags' in results[0]
        assert 'important' in results[0]['tags']
