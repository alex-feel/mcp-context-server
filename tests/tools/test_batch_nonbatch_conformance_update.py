"""Conformance of update_context_batch([single]) with update_context.

Both paths produce identical database state, error behavior and response semantics.
"""

import pytest
from fastmcp.exceptions import ToolError

from app.tools.batch.store import store_context_batch
from app.tools.batch.update import update_context_batch
from app.tools.context.store import store_context
from app.tools.context.update import update_context
from app.types import JsonValue
from tests.tools._conformance import CONFORMANCE_PNG_DATA
from tests.tools._conformance import THREAD_PREFIX
from tests.tools._conformance import assert_db_states_equal
from tests.tools._conformance import read_db_entry
from tests.tools._conformance import require_context_id


@pytest.mark.usefixtures('initialized_server')
class TestUpdateConformance:
    """Verify update_context and update_context_batch([single]) produce identical DB state."""

    async def _create_entry(self, thread_id: str) -> str:
        """Create a base entry for update testing."""
        result = await store_context(
            thread_id=thread_id, source='user', text='Original text',
            metadata={'original_key': 'original_value'},
            tags=['original_tag'],
        )
        return result['context_id']

    @pytest.mark.asyncio
    async def test_update_conformance_text_change(self) -> None:
        """Text update produces identical text_content in both paths."""
        nb_id = await self._create_entry(f'{THREAD_PREFIX}_upd_text_nb')
        b_id = await self._create_entry(f'{THREAD_PREFIX}_upd_text_b')

        await update_context(context_id=nb_id, text='Updated text content')
        await update_context_batch(
            updates=[{'context_id': b_id, 'text': 'Updated text content'}],
            atomic=True,
        )

        nb_state = await read_db_entry(nb_id)
        b_state = await read_db_entry(b_id)

        assert_db_states_equal(nb_state, b_state)
        assert nb_state['text_content'] == 'Updated text content'

    @pytest.mark.asyncio
    async def test_update_conformance_metadata_full_replace(self) -> None:
        """Full metadata replacement produces identical metadata."""
        nb_id = await self._create_entry(f'{THREAD_PREFIX}_upd_meta_nb')
        b_id = await self._create_entry(f'{THREAD_PREFIX}_upd_meta_b')
        new_meta: dict[str, JsonValue] = {'new_key': 'new_value'}

        await update_context(context_id=nb_id, metadata=new_meta)
        await update_context_batch(
            updates=[{'context_id': b_id, 'metadata': new_meta}],
            atomic=True,
        )

        nb_state = await read_db_entry(nb_id)
        b_state = await read_db_entry(b_id)

        assert_db_states_equal(nb_state, b_state)
        assert nb_state['metadata'] == new_meta

    @pytest.mark.asyncio
    async def test_update_conformance_metadata_patch(self) -> None:
        """Metadata patch adds new key, preserves original keys."""
        nb_id = await self._create_entry(f'{THREAD_PREFIX}_upd_patch_nb')
        b_id = await self._create_entry(f'{THREAD_PREFIX}_upd_patch_b')

        await update_context(
            context_id=nb_id, metadata_patch={'added_key': 'added_value'},
        )
        await update_context_batch(
            updates=[{'context_id': b_id, 'metadata_patch': {'added_key': 'added_value'}}],
            atomic=True,
        )

        nb_state = await read_db_entry(nb_id)
        b_state = await read_db_entry(b_id)

        assert_db_states_equal(nb_state, b_state)
        assert nb_state['metadata']['added_key'] == 'added_value'
        assert nb_state['metadata']['original_key'] == 'original_value'

    @pytest.mark.asyncio
    async def test_update_conformance_tags_replace(self) -> None:
        """Tag update replaces all existing tags identically."""
        nb_id = await self._create_entry(f'{THREAD_PREFIX}_upd_tags_nb')
        b_id = await self._create_entry(f'{THREAD_PREFIX}_upd_tags_b')

        await update_context(context_id=nb_id, tags=['new_tag_1', 'new_tag_2'])
        await update_context_batch(
            updates=[{'context_id': b_id, 'tags': ['new_tag_1', 'new_tag_2']}],
            atomic=True,
        )

        nb_state = await read_db_entry(nb_id)
        b_state = await read_db_entry(b_id)

        assert_db_states_equal(nb_state, b_state)
        assert nb_state['tags'] == ['new_tag_1', 'new_tag_2']

    @pytest.mark.asyncio
    async def test_update_conformance_images_replace(self) -> None:
        """Image update sets multimodal content type and image count identically."""
        nb_id = await self._create_entry(f'{THREAD_PREFIX}_upd_imgs_nb')
        b_id = await self._create_entry(f'{THREAD_PREFIX}_upd_imgs_b')
        image = {'data': CONFORMANCE_PNG_DATA, 'mime_type': 'image/png'}

        await update_context(context_id=nb_id, images=[image])
        await update_context_batch(
            updates=[{'context_id': b_id, 'images': [image]}],
            atomic=True,
        )

        nb_state = await read_db_entry(nb_id)
        b_state = await read_db_entry(b_id)

        assert_db_states_equal(nb_state, b_state)
        assert nb_state['content_type'] == 'multimodal'
        assert nb_state['image_count'] == 1

    @pytest.mark.asyncio
    async def test_update_conformance_content_type_transition(self) -> None:
        """Removing images transitions content_type back to 'text' in both paths."""
        thread_nb = f'{THREAD_PREFIX}_upd_ct_nb'
        thread_b = f'{THREAD_PREFIX}_upd_ct_b'
        image = {'data': CONFORMANCE_PNG_DATA, 'mime_type': 'image/png'}

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
        b_id = require_context_id(b_r['results'][0]['context_id'])

        await update_context(context_id=nb_id, images=[])
        await update_context_batch(
            updates=[{'context_id': b_id, 'images': []}],
            atomic=True,
        )

        nb_state = await read_db_entry(nb_id)
        b_state = await read_db_entry(b_id)

        assert_db_states_equal(nb_state, b_state)
        assert nb_state['content_type'] == 'text'
        assert nb_state['image_count'] == 0

    @pytest.mark.asyncio
    async def test_update_conformance_nonexistent_entry(self) -> None:
        """Both paths reject update for non-existent context_id."""
        with pytest.raises(ToolError, match='not found'):
            await update_context(context_id='0190abcdef1234567890abcd000f423f', text='Updated')

        with pytest.raises(ToolError, match='not found'):
            await update_context_batch(
                updates=[{'context_id': '0190abcdef1234567890abcd000f423f', 'text': 'Updated'}],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_update_conformance_no_fields_error(self) -> None:
        """Both paths reject update with no fields provided."""
        nb_id = await self._create_entry(f'{THREAD_PREFIX}_upd_nofield_nb')
        b_id = await self._create_entry(f'{THREAD_PREFIX}_upd_nofield_b')

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
        """Response messages convey same field-count information."""
        nb_id = await self._create_entry(f'{THREAD_PREFIX}_upd_msg_nb')
        b_id = await self._create_entry(f'{THREAD_PREFIX}_upd_msg_b')

        nb_result = await update_context(context_id=nb_id, text='Updated text')
        b_result = await update_context_batch(
            updates=[{'context_id': b_id, 'text': 'Updated text'}],
            atomic=True,
        )

        nb_msg = nb_result['message']
        b_msg = b_result['message']

        assert 'updated' in nb_msg.lower() or 'field' in nb_msg.lower()
        assert 'updated' in b_msg.lower() or '1/1' in b_msg
