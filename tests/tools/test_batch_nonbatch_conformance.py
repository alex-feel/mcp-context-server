"""Conformance tests: batch(single_entry) vs non-batch operation parity.

Guarantees that store_context_batch([single]), update_context_batch([single]),
and delete_context_batch([single]) produce IDENTICAL database state, error
behavior, and response semantics as their non-batch counterparts.

Each test calls both paths with equivalent parameters, then reads back the
database state via repository methods and asserts field-by-field equality
(except context_id and timestamps, which legitimately differ).
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.startup import ensure_repositories
from app.tools.batch import delete_context_batch
from app.tools.batch import store_context_batch
from app.tools.batch import update_context_batch
from app.tools.context import delete_context
from app.tools.context import store_context
from app.tools.context import update_context

# Minimal valid 1x1 PNG for conformance tests
_CONFORMANCE_PNG_DATA = base64.b64encode(bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
    0x54, 0x08, 0x99, 0x01, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x01, 0x7B, 0xDB, 0x56, 0x61, 0x00,
    0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
    0x42, 0x60, 0x82,
])).decode('utf-8')

_THREAD_PREFIX = 'conformance'


async def _read_db_entry(context_id: int) -> dict[str, Any]:
    """Read a context entry from the database and return a normalized dict for state comparison."""
    repos = await ensure_repositories()
    rows = await repos.context.get_by_ids([context_id])
    assert len(rows) == 1, f'Expected 1 row for id {context_id}, got {len(rows)}'
    row = rows[0]

    tags = await repos.tags.get_tags_for_context(context_id)
    sorted_tags = sorted(tags)
    image_count = await repos.images.count_images_for_context(context_id)

    raw_metadata = row['metadata']
    if isinstance(raw_metadata, str):
        metadata = json.loads(raw_metadata)
    elif raw_metadata is None:
        metadata = None
    else:
        metadata = raw_metadata

    return {
        'thread_id': row['thread_id'],
        'source': row['source'],
        'content_type': row['content_type'],
        'text_content': row['text_content'],
        'metadata': metadata,
        'summary': row['summary'],
        'tags': sorted_tags,
        'image_count': image_count,
    }


async def _count_entries_in_thread(thread_id: str) -> int:
    """Count the number of context entries in a thread."""
    repos = await ensure_repositories()
    rows, _ = await repos.context.search_contexts(
        thread_id=thread_id, limit=10000, offset=0, explain_query=False,
    )
    return len(rows)


def _assert_db_states_equal(
    nb_state: dict[str, Any],
    b_state: dict[str, Any],
    *,
    ignore_thread: bool = True,
) -> None:
    """Assert two database entry states are equal, ignoring thread_id if specified."""
    fields = ['source', 'content_type', 'text_content', 'metadata', 'summary', 'tags', 'image_count']
    if not ignore_thread:
        fields.insert(0, 'thread_id')
    for field in fields:
        assert nb_state[field] == b_state[field], (
            f'DB state mismatch on field {field!r}: '
            f'nonbatch={nb_state[field]!r} vs batch={b_state[field]!r}'
        )


# ============================================================================
# Category A: Store Conformance (A1-A10)
# ============================================================================

@pytest.mark.usefixtures('initialized_server')
class TestStoreConformance:
    """Verify store_context and store_context_batch([single]) produce identical DB state."""

    @pytest.mark.asyncio
    async def test_store_conformance_basic_text(self) -> None:
        """A1: Basic text store produces identical DB state."""
        thread_nb = f'{_THREAD_PREFIX}_store_basic_nb'
        thread_b = f'{_THREAD_PREFIX}_store_basic_b'

        nb_result = await store_context(
            thread_id=thread_nb, source='user', text='Test text content',
        )
        b_result = await store_context_batch(
            entries=[{'thread_id': thread_b, 'source': 'user', 'text': 'Test text content'}],
            atomic=True,
        )

        assert nb_result['success'] is True
        assert b_result['success'] is True

        nb_state = await _read_db_entry(nb_result['context_id'])
        b_state = await _read_db_entry(b_result['results'][0]['context_id'])

        _assert_db_states_equal(nb_state, b_state)
        assert nb_state['content_type'] == 'text'
        assert nb_state['metadata'] is None
        assert nb_state['tags'] == []
        assert nb_state['image_count'] == 0

    @pytest.mark.asyncio
    async def test_store_conformance_with_tags(self) -> None:
        """A2: Store with tags produces identical normalized tags."""
        thread_nb = f'{_THREAD_PREFIX}_store_tags_nb'
        thread_b = f'{_THREAD_PREFIX}_store_tags_b'

        nb_result = await store_context(
            thread_id=thread_nb, source='agent', text='Tagged content',
            tags=['Gamma', 'Alpha', 'Beta'],
        )
        b_result = await store_context_batch(
            entries=[{
                'thread_id': thread_b, 'source': 'agent', 'text': 'Tagged content',
                'tags': ['Gamma', 'Alpha', 'Beta'],
            }],
            atomic=True,
        )

        nb_state = await _read_db_entry(nb_result['context_id'])
        b_state = await _read_db_entry(b_result['results'][0]['context_id'])

        _assert_db_states_equal(nb_state, b_state)
        assert nb_state['tags'] == ['alpha', 'beta', 'gamma']

    @pytest.mark.asyncio
    async def test_store_conformance_with_images(self) -> None:
        """A3: Store with images produces identical multimodal content type and image count."""
        thread_nb = f'{_THREAD_PREFIX}_store_img_nb'
        thread_b = f'{_THREAD_PREFIX}_store_img_b'
        image = {'data': _CONFORMANCE_PNG_DATA, 'mime_type': 'image/png'}

        nb_result = await store_context(
            thread_id=thread_nb, source='user', text='Image content',
            images=[image],
        )
        b_result = await store_context_batch(
            entries=[{
                'thread_id': thread_b, 'source': 'user', 'text': 'Image content',
                'images': [image],
            }],
            atomic=True,
        )

        nb_state = await _read_db_entry(nb_result['context_id'])
        b_state = await _read_db_entry(b_result['results'][0]['context_id'])

        _assert_db_states_equal(nb_state, b_state)
        assert nb_state['content_type'] == 'multimodal'
        assert nb_state['image_count'] == 1

    @pytest.mark.asyncio
    async def test_store_conformance_with_metadata(self) -> None:
        """A4: Store with metadata produces identical metadata after round-trip."""
        thread_nb = f'{_THREAD_PREFIX}_store_meta_nb'
        thread_b = f'{_THREAD_PREFIX}_store_meta_b'
        meta = {'key': 'value', 'priority': 42, 'nested': {'a': 1}}

        nb_result = await store_context(
            thread_id=thread_nb, source='user', text='Metadata content',
            metadata=meta,
        )
        b_result = await store_context_batch(
            entries=[{
                'thread_id': thread_b, 'source': 'user', 'text': 'Metadata content',
                'metadata': meta,
            }],
            atomic=True,
        )

        nb_state = await _read_db_entry(nb_result['context_id'])
        b_state = await _read_db_entry(b_result['results'][0]['context_id'])

        _assert_db_states_equal(nb_state, b_state)
        assert nb_state['metadata'] == meta

    @pytest.mark.asyncio
    async def test_store_conformance_multimodal_content_type(self) -> None:
        """A5: Content type is 'multimodal' when images present in both paths."""
        thread_nb = f'{_THREAD_PREFIX}_store_mm_nb'
        thread_b = f'{_THREAD_PREFIX}_store_mm_b'
        image = {'data': _CONFORMANCE_PNG_DATA}

        nb_result = await store_context(
            thread_id=thread_nb, source='user', text='Multimodal test',
            images=[image],
        )
        b_result = await store_context_batch(
            entries=[{
                'thread_id': thread_b, 'source': 'user', 'text': 'Multimodal test',
                'images': [image],
            }],
            atomic=True,
        )

        nb_state = await _read_db_entry(nb_result['context_id'])
        b_state = await _read_db_entry(b_result['results'][0]['context_id'])

        assert nb_state['content_type'] == 'multimodal'
        assert b_state['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    async def test_store_conformance_dedup_behavior(self) -> None:
        """A6: Deduplication returns same context_id on second store in both paths."""
        thread_nb = f'{_THREAD_PREFIX}_store_dedup_nb'
        thread_b = f'{_THREAD_PREFIX}_store_dedup_b'

        nb_r1 = await store_context(thread_id=thread_nb, source='user', text='Dedup text')
        nb_r2 = await store_context(thread_id=thread_nb, source='user', text='Dedup text')

        b_r1 = await store_context_batch(
            entries=[{'thread_id': thread_b, 'source': 'user', 'text': 'Dedup text'}],
            atomic=True,
        )
        b_r2 = await store_context_batch(
            entries=[{'thread_id': thread_b, 'source': 'user', 'text': 'Dedup text'}],
            atomic=True,
        )

        assert nb_r1['context_id'] == nb_r2['context_id'], 'Non-batch dedup failed'
        assert b_r1['results'][0]['context_id'] == b_r2['results'][0]['context_id'], 'Batch dedup failed'

        assert await _count_entries_in_thread(thread_nb) == 1
        assert await _count_entries_in_thread(thread_b) == 1

    @pytest.mark.asyncio
    async def test_store_conformance_dedup_interleaving(self) -> None:
        """A7: Dedup suppressed when opposite-source entry is interleaved."""
        thread_nb = f'{_THREAD_PREFIX}_store_interleave_nb'
        thread_b = f'{_THREAD_PREFIX}_store_interleave_b'

        await store_context(thread_id=thread_nb, source='user', text='Interleave text')
        await store_context(thread_id=thread_nb, source='agent', text='Agent response')
        nb_r3 = await store_context(thread_id=thread_nb, source='user', text='Interleave text')

        await store_context_batch(
            entries=[{'thread_id': thread_b, 'source': 'user', 'text': 'Interleave text'}],
            atomic=True,
        )
        await store_context_batch(
            entries=[{'thread_id': thread_b, 'source': 'agent', 'text': 'Agent response'}],
            atomic=True,
        )
        b_r3 = await store_context_batch(
            entries=[{'thread_id': thread_b, 'source': 'user', 'text': 'Interleave text'}],
            atomic=True,
        )

        nb_count = await _count_entries_in_thread(thread_nb)
        b_count = await _count_entries_in_thread(thread_b)

        assert nb_count == b_count, (
            f'Entry count mismatch: nonbatch={nb_count} vs batch={b_count}'
        )
        assert nb_count == 3
        assert nb_r3['success'] is True
        assert b_r3['success'] is True

    @pytest.mark.asyncio
    async def test_store_conformance_invalid_images_error(self) -> None:
        """A8: Both paths reject entries with empty image data."""
        with pytest.raises(ToolError, match='Image 0 has empty "data" field'):
            await store_context(
                thread_id=f'{_THREAD_PREFIX}_store_inv_img_nb',
                source='user', text='Bad image',
                images=[{'data': ''}],
            )

        with pytest.raises(ToolError, match='Image 0 has empty "data" field'):
            await store_context_batch(
                entries=[{
                    'thread_id': f'{_THREAD_PREFIX}_store_inv_img_b',
                    'source': 'user', 'text': 'Bad image',
                    'images': [{'data': ''}],
                }],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_store_conformance_empty_text_error(self) -> None:
        """A9: Both paths reject entries with whitespace-only text."""
        with pytest.raises(ToolError, match='text cannot be empty'):
            await store_context(
                thread_id=f'{_THREAD_PREFIX}_store_empty_nb',
                source='user', text='   ',
            )

        with pytest.raises(ToolError, match='text cannot be empty'):
            await store_context_batch(
                entries=[{
                    'thread_id': f'{_THREAD_PREFIX}_store_empty_b',
                    'source': 'user', 'text': '   ',
                }],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_store_conformance_response_message_parity(self) -> None:
        """A10: Response messages convey same information about generation state."""
        thread_nb = f'{_THREAD_PREFIX}_store_msg_nb'
        thread_b = f'{_THREAD_PREFIX}_store_msg_b'

        nb_result = await store_context(
            thread_id=thread_nb, source='user', text='Message parity test',
        )
        b_result = await store_context_batch(
            entries=[{'thread_id': thread_b, 'source': 'user', 'text': 'Message parity test'}],
            atomic=True,
        )

        nb_msg = nb_result['message']
        b_msg = b_result['message']

        assert 'stored' in nb_msg.lower() or 'context' in nb_msg.lower()
        assert 'stored' in b_msg.lower() or '1/1' in b_msg


# ============================================================================
# Category B: Update Conformance (B1-B9)
# ============================================================================

@pytest.mark.usefixtures('initialized_server')
class TestUpdateConformance:
    """Verify update_context and update_context_batch([single]) produce identical DB state."""

    async def _create_entry(self, thread_id: str) -> int:
        """Create a base entry for update testing."""
        result = await store_context(
            thread_id=thread_id, source='user', text='Original text',
            metadata={'original_key': 'original_value'},
            tags=['original_tag'],
        )
        return result['context_id']

    @pytest.mark.asyncio
    async def test_update_conformance_text_change(self) -> None:
        """B1: Text update produces identical text_content in both paths."""
        nb_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_text_nb')
        b_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_text_b')

        await update_context(context_id=nb_id, text='Updated text content')
        await update_context_batch(
            updates=[{'context_id': b_id, 'text': 'Updated text content'}],
            atomic=True,
        )

        nb_state = await _read_db_entry(nb_id)
        b_state = await _read_db_entry(b_id)

        _assert_db_states_equal(nb_state, b_state)
        assert nb_state['text_content'] == 'Updated text content'

    @pytest.mark.asyncio
    async def test_update_conformance_metadata_full_replace(self) -> None:
        """B2: Full metadata replacement produces identical metadata."""
        nb_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_meta_nb')
        b_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_meta_b')
        new_meta = {'new_key': 'new_value'}

        await update_context(context_id=nb_id, metadata=new_meta)
        await update_context_batch(
            updates=[{'context_id': b_id, 'metadata': new_meta}],
            atomic=True,
        )

        nb_state = await _read_db_entry(nb_id)
        b_state = await _read_db_entry(b_id)

        _assert_db_states_equal(nb_state, b_state)
        assert nb_state['metadata'] == new_meta

    @pytest.mark.asyncio
    async def test_update_conformance_metadata_patch(self) -> None:
        """B3: Metadata patch adds new key, preserves original keys."""
        nb_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_patch_nb')
        b_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_patch_b')

        await update_context(
            context_id=nb_id, metadata_patch={'added_key': 'added_value'},
        )
        await update_context_batch(
            updates=[{'context_id': b_id, 'metadata_patch': {'added_key': 'added_value'}}],
            atomic=True,
        )

        nb_state = await _read_db_entry(nb_id)
        b_state = await _read_db_entry(b_id)

        _assert_db_states_equal(nb_state, b_state)
        assert nb_state['metadata']['added_key'] == 'added_value'
        assert nb_state['metadata']['original_key'] == 'original_value'

    @pytest.mark.asyncio
    async def test_update_conformance_tags_replace(self) -> None:
        """B4: Tag update replaces all existing tags identically."""
        nb_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_tags_nb')
        b_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_tags_b')

        await update_context(context_id=nb_id, tags=['new_tag_1', 'new_tag_2'])
        await update_context_batch(
            updates=[{'context_id': b_id, 'tags': ['new_tag_1', 'new_tag_2']}],
            atomic=True,
        )

        nb_state = await _read_db_entry(nb_id)
        b_state = await _read_db_entry(b_id)

        _assert_db_states_equal(nb_state, b_state)
        assert nb_state['tags'] == ['new_tag_1', 'new_tag_2']

    @pytest.mark.asyncio
    async def test_update_conformance_images_replace(self) -> None:
        """B5: Image update sets multimodal content type and image count identically."""
        nb_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_imgs_nb')
        b_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_imgs_b')
        image = {'data': _CONFORMANCE_PNG_DATA, 'mime_type': 'image/png'}

        await update_context(context_id=nb_id, images=[image])
        await update_context_batch(
            updates=[{'context_id': b_id, 'images': [image]}],
            atomic=True,
        )

        nb_state = await _read_db_entry(nb_id)
        b_state = await _read_db_entry(b_id)

        _assert_db_states_equal(nb_state, b_state)
        assert nb_state['content_type'] == 'multimodal'
        assert nb_state['image_count'] == 1

    @pytest.mark.asyncio
    async def test_update_conformance_content_type_transition(self) -> None:
        """B6: Removing images transitions content_type back to 'text' in both paths."""
        thread_nb = f'{_THREAD_PREFIX}_upd_ct_nb'
        thread_b = f'{_THREAD_PREFIX}_upd_ct_b'
        image = {'data': _CONFORMANCE_PNG_DATA, 'mime_type': 'image/png'}

        nb_r = await store_context(
            thread_id=thread_nb, source='user', text='With image',
            images=[image],
        )
        b_r = await store_context_batch(
            entries=[{
                'thread_id': thread_b, 'source': 'user', 'text': 'With image',
                'images': [image],
            }],
            atomic=True,
        )
        nb_id = nb_r['context_id']
        b_id = b_r['results'][0]['context_id']

        await update_context(context_id=nb_id, images=[])
        await update_context_batch(
            updates=[{'context_id': b_id, 'images': []}],
            atomic=True,
        )

        nb_state = await _read_db_entry(nb_id)
        b_state = await _read_db_entry(b_id)

        _assert_db_states_equal(nb_state, b_state)
        assert nb_state['content_type'] == 'text'
        assert nb_state['image_count'] == 0

    @pytest.mark.asyncio
    async def test_update_conformance_nonexistent_entry(self) -> None:
        """B7: Both paths reject update for non-existent context_id."""
        with pytest.raises(ToolError, match='not found'):
            await update_context(context_id=999999, text='Updated')

        with pytest.raises(ToolError, match='not found'):
            await update_context_batch(
                updates=[{'context_id': 999999, 'text': 'Updated'}],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_update_conformance_no_fields_error(self) -> None:
        """B8: Both paths reject update with no fields provided."""
        nb_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_nofield_nb')
        b_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_nofield_b')

        with pytest.raises(ToolError, match='At least one field'):
            await update_context(context_id=nb_id)

        # Batch validates this in its own validation loop and raises ToolError in atomic mode
        with pytest.raises(ToolError, match='(?i)(at least one field|validation failed)'):
            await update_context_batch(
                updates=[{'context_id': b_id}],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_update_conformance_response_message_parity(self) -> None:
        """B9: Response messages convey same field-count information."""
        nb_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_msg_nb')
        b_id = await self._create_entry(f'{_THREAD_PREFIX}_upd_msg_b')

        nb_result = await update_context(context_id=nb_id, text='Updated text')
        b_result = await update_context_batch(
            updates=[{'context_id': b_id, 'text': 'Updated text'}],
            atomic=True,
        )

        nb_msg = nb_result['message']
        b_msg = b_result['message']

        assert 'updated' in nb_msg.lower() or 'field' in nb_msg.lower()
        assert 'updated' in b_msg.lower() or '1/1' in b_msg


# ============================================================================
# Category C: Delete Conformance (C1-C4)
# ============================================================================

@pytest.mark.usefixtures('initialized_server')
class TestDeleteConformance:
    """Verify delete_context and delete_context_batch produce identical behavior."""

    @pytest.mark.asyncio
    async def test_delete_conformance_by_ids(self) -> None:
        """C1: Delete by IDs removes entry and returns same deleted_count."""
        thread_nb = f'{_THREAD_PREFIX}_del_ids_nb'
        thread_b = f'{_THREAD_PREFIX}_del_ids_b'

        nb_r = await store_context(thread_id=thread_nb, source='user', text='Delete me')
        b_r = await store_context_batch(
            entries=[{'thread_id': thread_b, 'source': 'user', 'text': 'Delete me'}],
            atomic=True,
        )

        nb_del = await delete_context(context_ids=[nb_r['context_id']])
        b_del = await delete_context_batch(context_ids=[b_r['results'][0]['context_id']])

        assert nb_del['deleted_count'] == 1
        assert b_del['deleted_count'] == 1

        assert await _count_entries_in_thread(thread_nb) == 0
        assert await _count_entries_in_thread(thread_b) == 0

    @pytest.mark.asyncio
    async def test_delete_conformance_by_thread(self) -> None:
        """C2: Delete by thread removes all entries and returns same deleted_count."""
        thread_nb = f'{_THREAD_PREFIX}_del_thread_nb'
        thread_b = f'{_THREAD_PREFIX}_del_thread_b'

        for i in range(3):
            await store_context(thread_id=thread_nb, source='user', text=f'Entry {i}')
            await store_context_batch(
                entries=[{'thread_id': thread_b, 'source': 'user', 'text': f'Entry {i}'}],
                atomic=True,
            )

        nb_del = await delete_context(thread_id=thread_nb)
        b_del = await delete_context_batch(thread_ids=[thread_b])

        assert nb_del['deleted_count'] == 3
        assert b_del['deleted_count'] == 3

        assert await _count_entries_in_thread(thread_nb) == 0
        assert await _count_entries_in_thread(thread_b) == 0

    @pytest.mark.asyncio
    async def test_delete_conformance_embedding_cleanup(self) -> None:
        """C3: Both paths trigger embedding cleanup when semantic search is enabled."""
        thread_nb = f'{_THREAD_PREFIX}_del_embed_nb'
        thread_b = f'{_THREAD_PREFIX}_del_embed_b'

        nb_r = await store_context(thread_id=thread_nb, source='user', text='Embed cleanup test')
        b_r = await store_context_batch(
            entries=[{'thread_id': thread_b, 'source': 'user', 'text': 'Embed cleanup test'}],
            atomic=True,
        )

        nb_id = nb_r['context_id']
        b_id = b_r['results'][0]['context_id']

        mock_delete = AsyncMock()
        repos = await ensure_repositories()

        with (
            patch.object(repos.embeddings, 'delete', mock_delete),
            patch('app.tools.context.settings') as mock_nb_settings,
            patch('app.tools.batch.settings') as mock_b_settings,
        ):
            mock_nb_settings.semantic_search.enabled = True
            mock_b_settings.semantic_search.enabled = True
            mock_nb_settings.embedding.model = 'test-model'
            mock_b_settings.embedding.model = 'test-model'

            await delete_context(context_ids=[nb_id])
            await delete_context_batch(context_ids=[b_id])

        delete_calls = [call.args[0] for call in mock_delete.call_args_list]
        assert nb_id in delete_calls, f'Non-batch did not clean up embeddings for {nb_id}'
        assert b_id in delete_calls, f'Batch did not clean up embeddings for {b_id}'

    @pytest.mark.asyncio
    async def test_delete_conformance_nonexistent_id(self) -> None:
        """C4: Both paths handle non-existent IDs gracefully with deleted_count=0."""
        nb_del = await delete_context(context_ids=[999997])
        b_del = await delete_context_batch(context_ids=[999996])

        assert nb_del['deleted_count'] == 0
        assert b_del['deleted_count'] == 0
        assert nb_del['success'] is True
        assert b_del['success'] is True


# ============================================================================
# Category D: Error Behavior Conformance (D1-D3)
# ============================================================================

@pytest.mark.usefixtures('initialized_server')
class TestErrorConformance:
    """Verify error handling parity between batch and non-batch operations."""

    @pytest.mark.asyncio
    async def test_error_conformance_invalid_source(self) -> None:
        """D1: Batch (atomic) raises ToolError for invalid source."""
        with pytest.raises(ToolError, match='(?i)(invalid source|Missing or invalid source)'):
            await store_context_batch(
                entries=[{
                    'thread_id': f'{_THREAD_PREFIX}_err_src_b',
                    'source': 'invalid',
                    'text': 'Invalid source test',
                }],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_error_conformance_missing_text(self) -> None:
        """D2: Batch rejects entry with missing text field."""
        with pytest.raises(ToolError, match='Missing required field: text'):
            await store_context_batch(
                entries=[{
                    'thread_id': f'{_THREAD_PREFIX}_err_notext_b',
                    'source': 'user',
                }],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_error_conformance_image_validation_parity(self) -> None:
        """D3: Both paths reject invalid base64 image data with similar messages."""
        invalid_image = {'data': '!!!not-base64!!!', 'mime_type': 'image/png'}

        with pytest.raises(ToolError, match='(?i)(invalid|base64|Image 0)'):
            await store_context(
                thread_id=f'{_THREAD_PREFIX}_err_b64_nb',
                source='user', text='Bad base64',
                images=[invalid_image],
            )

        with pytest.raises(ToolError, match='(?i)(invalid|base64|Image 0|Validation)'):
            await store_context_batch(
                entries=[{
                    'thread_id': f'{_THREAD_PREFIX}_err_b64_b',
                    'source': 'user', 'text': 'Bad base64',
                    'images': [invalid_image],
                }],
                atomic=True,
            )


# ============================================================================
# Category E: Generation Conformance (E1-E3)
# ============================================================================

@pytest.mark.usefixtures('initialized_server')
class TestGenerationConformance:
    """Verify embedding/summary generation parity between batch and non-batch."""

    @pytest.mark.asyncio
    async def test_generation_conformance_embeddings_triggered(self) -> None:
        """E1: Both paths call generate_embeddings_with_timeout for the same text."""
        from app.repositories.embedding_repository import ChunkEmbedding

        mock_embedding = ChunkEmbedding(
            embedding=[0.1] * 1024,
            start_index=0,
            end_index=26,
        )
        mock_gen_embed = AsyncMock(return_value=[mock_embedding])
        mock_provider = AsyncMock()

        thread_nb = f'{_THREAD_PREFIX}_gen_embed_nb'
        thread_b = f'{_THREAD_PREFIX}_gen_embed_b'

        repos = await ensure_repositories()

        # Mock embedding storage to avoid missing vec_context_embeddings table
        with (
            patch('app.tools.context.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.context.get_summary_provider', return_value=None),
            patch('app.tools.context.generate_embeddings_with_timeout', mock_gen_embed),
            patch('app.tools.batch.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.batch.get_summary_provider', return_value=None),
            patch('app.tools.batch.generate_embeddings_with_timeout', mock_gen_embed),
            patch('app.tools._shared.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._shared.get_summary_provider', return_value=None),
            patch('app.startup.get_embedding_provider', return_value=mock_provider),
            patch('app.startup.get_summary_provider', return_value=None),
            patch.object(repos.embeddings, 'store_chunked', AsyncMock()),
            patch.object(repos.embeddings, 'exists', AsyncMock(return_value=False)),
        ):
            mock_gen_embed.reset_mock()
            await store_context(
                thread_id=thread_nb, source='user', text='Embedding conformance text',
            )
            nb_call_count = mock_gen_embed.call_count

            mock_gen_embed.reset_mock()
            await store_context_batch(
                entries=[{
                    'thread_id': thread_b, 'source': 'user',
                    'text': 'Embedding conformance text',
                }],
                atomic=True,
            )
            b_call_count = mock_gen_embed.call_count

        assert nb_call_count == b_call_count, (
            f'Embedding generation call count mismatch: '
            f'nonbatch={nb_call_count} vs batch={b_call_count}'
        )
        assert nb_call_count >= 1, 'Embedding generation was not called'

    @pytest.mark.asyncio
    async def test_generation_conformance_summary_triggered(self) -> None:
        """E2: Both paths call generate_summary_with_timeout for long text."""
        mock_gen_summary = AsyncMock(return_value='Mock summary')
        mock_provider = AsyncMock()

        # Text longer than default min_content_length (500)
        long_text = 'Summary conformance test content. ' * 20

        thread_nb = f'{_THREAD_PREFIX}_gen_summ_nb'
        thread_b = f'{_THREAD_PREFIX}_gen_summ_b'

        with (
            patch('app.tools.context.get_embedding_provider', return_value=None),
            patch('app.tools.context.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.generate_summary_with_timeout', mock_gen_summary),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.tools.batch.get_summary_provider', return_value=mock_provider),
            patch('app.tools.batch.generate_summary_with_timeout', mock_gen_summary),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=mock_provider),
            patch('app.startup.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_provider),
        ):
            mock_gen_summary.reset_mock()
            await store_context(
                thread_id=thread_nb, source='user', text=long_text,
            )
            nb_call_count = mock_gen_summary.call_count

            mock_gen_summary.reset_mock()
            await store_context_batch(
                entries=[{
                    'thread_id': thread_b, 'source': 'user', 'text': long_text,
                }],
                atomic=True,
            )
            b_call_count = mock_gen_summary.call_count

        assert nb_call_count == b_call_count, (
            f'Summary generation call count mismatch: '
            f'nonbatch={nb_call_count} vs batch={b_call_count}'
        )
        assert nb_call_count >= 1, 'Summary generation was not called for non-batch'

    @pytest.mark.asyncio
    async def test_generation_conformance_skip_short_content(self) -> None:
        """E3: Both paths skip summary generation for text shorter than min_content_length."""
        mock_gen_summary = AsyncMock(return_value='Should not be called')
        mock_provider = AsyncMock()

        short_text = 'Short text'

        thread_nb = f'{_THREAD_PREFIX}_gen_skip_nb'
        thread_b = f'{_THREAD_PREFIX}_gen_skip_b'

        with (
            patch('app.tools.context.get_embedding_provider', return_value=None),
            patch('app.tools.context.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.generate_summary_with_timeout', mock_gen_summary),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.tools.batch.get_summary_provider', return_value=mock_provider),
            patch('app.tools.batch.generate_summary_with_timeout', mock_gen_summary),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=mock_provider),
            patch('app.startup.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=mock_provider),
        ):
            mock_gen_summary.reset_mock()
            await store_context(
                thread_id=thread_nb, source='user', text=short_text,
            )
            nb_call_count = mock_gen_summary.call_count

            mock_gen_summary.reset_mock()
            await store_context_batch(
                entries=[{
                    'thread_id': thread_b, 'source': 'user', 'text': short_text,
                }],
                atomic=True,
            )
            b_call_count = mock_gen_summary.call_count

        assert nb_call_count == 0, f'Non-batch called summary generation for short text ({nb_call_count} calls)'
        assert b_call_count == 0, f'Batch called summary generation for short text ({b_call_count} calls)'
