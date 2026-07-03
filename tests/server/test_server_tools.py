"""Tests for server MCP tool functions.

This module tests the MCP tool handlers in app/server.py including
list_threads, get_statistics, search_context, and delete_context.
"""

import base64
from typing import cast

import pytest

from app.server import delete_context
from app.server import get_context_by_ids
from app.server import get_statistics
from app.server import list_threads
from app.server import search_context
from app.server import store_context


class TestListThreads:
    """Test the list_threads MCP tool."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_list_threads_empty_database(self) -> None:
        """Test listing threads from empty database."""
        result = await list_threads()

        assert result['total_threads'] == 0
        assert result['threads'] == []

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_list_threads_single_thread(self) -> None:
        """Test listing threads with single thread."""
        # Create some context
        await store_context(
            thread_id='test_thread_1',
            source='user',
            text='Test message',
        )

        result = await list_threads()

        assert result['total_threads'] == 1
        assert len(result['threads']) == 1
        assert result['threads'][0]['thread_id'] == 'test_thread_1'
        assert result['threads'][0]['entry_count'] == 1

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_list_threads_multiple_threads(self) -> None:
        """Test listing multiple threads."""
        # Create context in multiple threads
        await store_context(thread_id='thread_a', source='user', text='Message A1')
        await store_context(thread_id='thread_a', source='agent', text='Message A2')
        await store_context(thread_id='thread_b', source='user', text='Message B1')
        await store_context(thread_id='thread_c', source='user', text='Message C1')

        result = await list_threads()

        assert result['total_threads'] == 3
        assert len(result['threads']) == 3

        # Check thread info
        thread_ids = {t['thread_id'] for t in result['threads']}
        assert thread_ids == {'thread_a', 'thread_b', 'thread_c'}

        # Find thread_a and check entry count
        thread_a = next(t for t in result['threads'] if t['thread_id'] == 'thread_a')
        assert thread_a['entry_count'] == 2


class TestGetStatistics:
    """Test the get_statistics MCP tool."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_statistics_empty_database(self) -> None:
        """Test getting statistics from empty database."""
        result = await get_statistics()

        assert result['total_entries'] == 0
        assert result['total_threads'] == 0
        assert result['total_images'] == 0
        assert result['unique_tags'] == 0

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_statistics_with_data(self) -> None:
        """Test getting statistics with populated data."""
        # Create diverse data
        await store_context(
            thread_id='stats_thread',
            source='user',
            text='User message',
            tags=['tag1', 'tag2'],
        )
        await store_context(
            thread_id='stats_thread',
            source='agent',
            text='Agent response',
            tags=['tag2', 'tag3'],
        )

        result = await get_statistics()

        assert result['total_entries'] == 2
        assert result['total_threads'] == 1
        assert result['unique_tags'] == 3  # tag1, tag2, tag3
        assert 'by_source' in result
        assert result['by_source'] == {'user': 1, 'agent': 1}


class TestSearchContext:
    """Test the search_context MCP tool."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_empty_database(self) -> None:
        """Test searching empty database."""
        result = await search_context(limit=50)

        assert result['results'] == []

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_by_thread_id(self) -> None:
        """Test searching by thread_id."""
        await store_context(thread_id='search_thread', source='user', text='Message 1')
        await store_context(thread_id='search_thread', source='agent', text='Message 2')
        await store_context(thread_id='other_thread', source='user', text='Message 3')

        result = await search_context(limit=50, thread_id='search_thread')

        assert len(result['results']) == 2
        for entry in result['results']:
            assert entry['thread_id'] == 'search_thread'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_by_source(self) -> None:
        """Test searching by source."""
        await store_context(thread_id='src_thread', source='user', text='User msg')
        await store_context(thread_id='src_thread', source='agent', text='Agent msg')

        result = await search_context(limit=50, source='user')

        assert len(result['results']) == 1
        assert result['results'][0]['source'] == 'user'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_by_tags(self) -> None:
        """Test searching by tags."""
        await store_context(
            thread_id='tag_thread',
            source='user',
            text='Message with tags',
            tags=['important', 'review'],
        )
        await store_context(
            thread_id='tag_thread',
            source='user',
            text='Another message',
            tags=['other'],
        )

        result = await search_context(limit=50, tags=['important'])

        assert len(result['results']) == 1
        assert 'important' in result['results'][0]['tags']

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_with_limit(self) -> None:
        """Test searching with limit parameter."""
        # Create several entries
        for i in range(5):
            await store_context(
                thread_id='limit_thread',
                source='user',
                text=f'Message {i}',
            )

        result = await search_context(thread_id='limit_thread', limit=3)

        assert len(result['results']) == 3

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_with_offset(self) -> None:
        """Test searching with offset parameter."""
        # Create entries
        for i in range(5):
            await store_context(
                thread_id='offset_thread',
                source='user',
                text=f'Message {i}',
            )

        result = await search_context(thread_id='offset_thread', limit=2, offset=2)

        assert len(result['results']) == 2

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_by_content_type(self) -> None:
        """Test searching by content type."""
        # Create text-only entry
        await store_context(
            thread_id='content_type_thread',
            source='user',
            text='Text only',
        )
        # Create multimodal entry
        image_data = base64.b64encode(b'fake_image').decode('utf-8')
        await store_context(
            thread_id='content_type_thread',
            source='user',
            text='With image',
            images=[{'data': image_data, 'mime_type': 'image/png'}],
        )

        # Search for multimodal
        result = await search_context(limit=50, content_type='multimodal')

        assert len(result['results']) == 1
        assert result['results'][0]['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_search_context_combined_filters(self) -> None:
        """Test searching with multiple filters combined."""
        await store_context(
            thread_id='combined_thread',
            source='user',
            text='User message',
            tags=['tag_a'],
        )
        await store_context(
            thread_id='combined_thread',
            source='agent',
            text='Agent message',
            tags=['tag_a'],
        )
        await store_context(
            thread_id='other_combined',
            source='user',
            text='Other thread',
            tags=['tag_a'],
        )

        result = await search_context(
            limit=50,
            thread_id='combined_thread',
            source='user',
            tags=['tag_a'],
        )

        assert len(result['results']) == 1
        assert result['results'][0]['thread_id'] == 'combined_thread'
        assert result['results'][0]['source'] == 'user'


class TestGetContextByIds:
    """Test the get_context_by_ids MCP tool."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_context_by_ids_single(self) -> None:
        """Test getting single context by ID."""
        result = await store_context(
            thread_id='ids_thread',
            source='user',
            text='Test message',
        )
        context_id = result['context_id']

        entries = await get_context_by_ids(context_ids=[context_id])

        assert len(entries) == 1
        entry = dict(entries[0])
        assert entry['id'] == context_id
        assert entry['text_content'] == 'Test message'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_context_by_ids_multiple(self) -> None:
        """Test getting multiple contexts by IDs."""
        ids = []
        for i in range(3):
            result = await store_context(
                thread_id='multi_ids_thread',
                source='user',
                text=f'Message {i}',
            )
            ids.append(result['context_id'])

        entries = await get_context_by_ids(context_ids=ids)

        assert len(entries) == 3
        returned_ids = {dict(e)['id'] for e in entries}
        assert returned_ids == set(ids)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_context_by_ids_nonexistent(self) -> None:
        """Test getting nonexistent context ID returns empty."""
        entries = await get_context_by_ids(context_ids=['0190abcdef1234567890abcd000f423f'])

        assert entries == []

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_context_by_ids_with_images(self) -> None:
        """Test getting context with images."""
        image_data = base64.b64encode(b'test_image_data').decode('utf-8')
        result = await store_context(
            thread_id='image_ids_thread',
            source='user',
            text='With image',
            images=[{'data': image_data, 'mime_type': 'image/png'}],
        )
        context_id = result['context_id']

        entries = await get_context_by_ids(
            context_ids=[context_id],
            include_images=True,
        )

        assert len(entries) == 1
        entry = dict(entries[0])
        assert entry['content_type'] == 'multimodal'
        # Images should be included when requested

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_context_by_ids_partial_match(self) -> None:
        """Test getting mix of existing and nonexistent IDs."""
        result = await store_context(
            thread_id='partial_ids',
            source='user',
            text='Existing entry',
        )
        existing_id = result['context_id']

        entries = await get_context_by_ids(
            context_ids=[existing_id, '0' * 32, 'f' * 32],
        )

        # Should only return the existing entry
        assert len(entries) == 1
        entry = dict(entries[0])
        assert entry['id'] == existing_id

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_context_by_ids_omits_summary_by_default(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """By default, get_context_by_ids must NOT include the summary key."""
        import app.tools.context as context_module
        from app.settings import get_settings
        monkeypatch.delenv('GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY', raising=False)
        get_settings.cache_clear()
        # Refresh the module-level binding to pick up the new setting
        monkeypatch.setattr(context_module, 'settings', get_settings())
        assert context_module.settings.retrieval.include_summary is False

        result = await store_context(
            thread_id='retrieval_default_thread',
            source='user',
            text='Default-omit-summary test',
        )
        context_id = result['context_id']

        entries = await get_context_by_ids(context_ids=[context_id])
        assert len(entries) == 1
        entry = dict(entries[0])
        assert 'summary' not in entry
        # All other fields must still be present
        assert entry['id'] == context_id
        assert entry['text_content'] == 'Default-omit-summary test'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_context_by_ids_omits_summary_when_explicit_false(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY=false must omit summary."""
        import app.tools.context as context_module
        from app.settings import get_settings
        monkeypatch.setenv('GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY', 'false')
        get_settings.cache_clear()
        monkeypatch.setattr(context_module, 'settings', get_settings())

        result = await store_context(
            thread_id='retrieval_explicit_false_thread',
            source='user',
            text='Explicit-false-summary test',
        )
        context_id = result['context_id']

        entries = await get_context_by_ids(context_ids=[context_id])
        assert len(entries) == 1
        assert 'summary' not in dict(entries[0])

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_context_by_ids_includes_summary_when_explicit_true(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY=true: summary key present, empty string when DB NULL.

        Verifies the second leg of the tri-state contract: when the toggle is enabled
        AND the stored summary is NULL (no provider configured in the test env),
        the wire value MUST be normalized to an empty string ''. This mirrors
        the search-tools contract and provides an explicit "feature on, no data yet"
        signal distinct from the "feature disabled" key-omission.
        """
        import app.tools.context as context_module
        from app.settings import get_settings
        monkeypatch.setenv('GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY', 'true')
        get_settings.cache_clear()
        monkeypatch.setattr(context_module, 'settings', get_settings())

        result = await store_context(
            thread_id='retrieval_explicit_true_thread',
            source='user',
            text='Explicit-true-summary test',
        )
        context_id = result['context_id']

        entries = await get_context_by_ids(context_ids=[context_id])
        assert len(entries) == 1
        entry = dict(entries[0])
        assert 'summary' in entry, 'summary key MUST be present when toggle is enabled'
        # No summary provider in test env => DB row has NULL summary => normalize to ''.
        assert isinstance(entry['summary'], str), (
            f'summary must be a string when present, got {type(entry["summary"]).__name__}'
        )
        assert entry['summary'] == '', (
            f'NULL DB summary must normalize to empty string, got {entry["summary"]!r}'
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_context_by_ids_summary_passes_through_when_stored(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY=true: stored non-empty summary returned verbatim.

        Verifies the third leg of the tri-state contract: when the toggle is enabled
        AND the database has a non-empty summary string, the value is surfaced
        verbatim (no normalization, no transformation).
        """
        import app.tools.context as context_module
        from app.settings import get_settings
        from app.startup import ensure_repositories
        monkeypatch.setenv('GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY', 'true')
        get_settings.cache_clear()
        monkeypatch.setattr(context_module, 'settings', get_settings())

        store_result = await store_context(
            thread_id='retrieval_summary_passthrough_thread',
            source='user',
            text='Pass-through summary test',
        )
        context_id = store_result['context_id']

        # Inject a non-empty summary directly via the repository, bypassing the
        # (absent) summary provider. This simulates the production state where
        # generation succeeded and the DB row holds a real LLM-produced summary.
        repos = await ensure_repositories()
        injected_summary = 'Manually injected summary for test'
        success, updated_fields = await repos.context.update_context_entry(
            context_id=context_id,
            summary=injected_summary,
        )
        assert success, f'Failed to inject summary; updated_fields={updated_fields}'
        assert 'summary' in updated_fields, f'summary not in updated_fields: {updated_fields}'

        entries = await get_context_by_ids(context_ids=[context_id])
        assert len(entries) == 1
        entry = dict(entries[0])
        assert 'summary' in entry, 'summary key MUST be present when toggle is enabled'
        assert entry['summary'] == injected_summary, (
            f'Expected verbatim pass-through {injected_summary!r}, got {entry["summary"]!r}'
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_get_context_by_ids_other_fields_unaffected(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Strip must not affect id, text_content, metadata, tags, or content_type."""
        import app.tools.context as context_module
        from app.settings import get_settings
        monkeypatch.delenv('GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY', raising=False)
        get_settings.cache_clear()
        monkeypatch.setattr(context_module, 'settings', get_settings())

        result = await store_context(
            thread_id='retrieval_other_fields_thread',
            source='agent',
            text='Other fields untouched',
            tags=['retrieval-test'],
            metadata={'k': 'v'},
        )
        context_id = result['context_id']

        entries = await get_context_by_ids(context_ids=[context_id])
        entry = dict(entries[0])
        assert entry['id'] == context_id
        assert entry['source'] == 'agent'
        assert entry['text_content'] == 'Other fields untouched'
        entry_tags = cast(list[str], entry['tags'])
        assert 'retrieval-test' in entry_tags
        assert entry['metadata'] == {'k': 'v'}
        assert 'summary' not in entry


class TestDeleteContext:
    """Test the delete_context MCP tool."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_delete_context_by_thread_id(self) -> None:
        """Test deleting all context by thread_id."""
        # Create entries
        await store_context(thread_id='delete_thread', source='user', text='Msg 1')
        await store_context(thread_id='delete_thread', source='agent', text='Msg 2')
        await store_context(thread_id='keep_thread', source='user', text='Keep')

        result = await delete_context(thread_id='delete_thread')

        assert result['success'] is True
        assert result['deleted_count'] == 2

        # Verify deletion
        search = await search_context(limit=50, thread_id='delete_thread')
        assert len(search['results']) == 0

        # Verify other thread untouched
        search_keep = await search_context(limit=50, thread_id='keep_thread')
        assert len(search_keep['results']) == 1

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_delete_context_by_ids(self) -> None:
        """Test deleting specific context entries by IDs."""
        result1 = await store_context(thread_id='id_del_thread', source='user', text='User')
        result2 = await store_context(thread_id='id_del_thread', source='agent', text='Agent')

        # Delete only the first entry by ID
        result = await delete_context(context_ids=[result1['context_id']])

        assert result['success'] is True
        assert result['deleted_count'] == 1

        # Verify only the specified entry was deleted
        search = await search_context(limit=50, thread_id='id_del_thread')
        assert len(search['results']) == 1
        assert search['results'][0]['id'] == result2['context_id']

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_delete_context_nonexistent_thread(self) -> None:
        """Test deleting from nonexistent thread."""
        result = await delete_context(thread_id='nonexistent_delete_thread')

        assert result['success'] is True
        assert result['deleted_count'] == 0

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_delete_context_cascades_to_images(self) -> None:
        """Test that deleting context also deletes associated images."""
        image_data = base64.b64encode(b'image_to_delete').decode('utf-8')
        await store_context(
            thread_id='cascade_thread',
            source='user',
            text='With image',
            images=[{'data': image_data, 'mime_type': 'image/png'}],
        )

        result = await delete_context(thread_id='cascade_thread')

        assert result['success'] is True
        assert result['deleted_count'] == 1

        # Images should be cascade deleted (verified by searching)
        search = await search_context(limit=50, thread_id='cascade_thread')
        assert len(search['results']) == 0

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_delete_context_cascades_to_tags(self) -> None:
        """Test that deleting context also deletes associated tags."""
        await store_context(
            thread_id='tag_cascade_thread',
            source='user',
            text='Tagged entry',
            tags=['will_be_deleted'],
        )

        result = await delete_context(thread_id='tag_cascade_thread')

        assert result['success'] is True
        assert result['deleted_count'] == 1


class TestStoreContextWithMetadata:
    """Test store_context with metadata filtering support."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_and_search_with_metadata(self) -> None:
        """Test storing and searching with metadata."""
        await store_context(
            thread_id='meta_thread',
            source='user',
            text='Entry with metadata',
            metadata={'priority': 1, 'status': 'active'},
        )
        await store_context(
            thread_id='meta_thread',
            source='user',
            text='Another entry',
            metadata={'priority': 2, 'status': 'pending'},
        )

        # Search with simple metadata filter
        result = await search_context(
            limit=50,
            thread_id='meta_thread',
            metadata={'status': 'active'},
        )

        assert len(result['results']) == 1
        assert result['results'][0]['metadata']['status'] == 'active'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_with_nested_metadata(self) -> None:
        """Test storing context with nested metadata."""
        await store_context(
            thread_id='nested_meta_thread',
            source='user',
            text='Nested metadata test',
            metadata={'user': {'name': 'test', 'settings': {'theme': 'dark'}}},
        )

        result = await search_context(limit=50, thread_id='nested_meta_thread')

        assert len(result['results']) == 1
        assert result['results'][0]['metadata']['user']['name'] == 'test'
        assert result['results'][0]['metadata']['user']['settings']['theme'] == 'dark'
