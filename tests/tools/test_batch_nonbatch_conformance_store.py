"""Conformance of store_context_batch([single]) with store_context.

Both paths run with equivalent parameters; the stored rows are read back through the repositories and compared
field by field (except context_id and timestamps, which legitimately differ). Both paths reject the same invalid
input.
"""

import pytest
from fastmcp.exceptions import ToolError

from app.tools.batch.store import store_context_batch
from app.tools.context.store import store_context
from app.types import JsonValue
from tests.tools._conformance import CONFORMANCE_PNG_DATA
from tests.tools._conformance import THREAD_PREFIX
from tests.tools._conformance import assert_db_states_equal
from tests.tools._conformance import count_entries_in_thread
from tests.tools._conformance import read_db_entry
from tests.tools._conformance import require_context_id


@pytest.mark.usefixtures('initialized_server')
class TestStoreConformance:
    """Verify store_context and store_context_batch([single]) produce identical DB state."""

    @pytest.mark.asyncio
    async def test_store_conformance_basic_text(self) -> None:
        """Basic text store produces identical DB state."""
        thread_nb = f'{THREAD_PREFIX}_store_basic_nb'
        thread_b = f'{THREAD_PREFIX}_store_basic_b'

        nb_result = await store_context(
            thread_id=thread_nb, source='user', text='Test text content',
        )
        b_result = await store_context_batch(
            entries=[{'thread_id': thread_b, 'source': 'user', 'text': 'Test text content'}],
            atomic=True,
        )

        assert nb_result['success'] is True
        assert b_result['success'] is True

        nb_state = await read_db_entry(nb_result['context_id'])
        b_state = await read_db_entry(require_context_id(b_result['results'][0]['context_id']))

        assert_db_states_equal(nb_state, b_state)
        assert nb_state['content_type'] == 'text'
        assert nb_state['metadata'] is None
        assert nb_state['tags'] == []
        assert nb_state['image_count'] == 0

    @pytest.mark.asyncio
    async def test_store_conformance_with_tags(self) -> None:
        """Store with tags produces identical normalized tags."""
        thread_nb = f'{THREAD_PREFIX}_store_tags_nb'
        thread_b = f'{THREAD_PREFIX}_store_tags_b'

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

        nb_state = await read_db_entry(nb_result['context_id'])
        b_state = await read_db_entry(require_context_id(b_result['results'][0]['context_id']))

        assert_db_states_equal(nb_state, b_state)
        assert nb_state['tags'] == ['alpha', 'beta', 'gamma']

    @pytest.mark.asyncio
    async def test_store_conformance_with_images(self) -> None:
        """Store with images produces identical multimodal content type and image count."""
        thread_nb = f'{THREAD_PREFIX}_store_img_nb'
        thread_b = f'{THREAD_PREFIX}_store_img_b'
        image = {'data': CONFORMANCE_PNG_DATA, 'mime_type': 'image/png'}

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

        nb_state = await read_db_entry(nb_result['context_id'])
        b_state = await read_db_entry(require_context_id(b_result['results'][0]['context_id']))

        assert_db_states_equal(nb_state, b_state)
        assert nb_state['content_type'] == 'multimodal'
        assert nb_state['image_count'] == 1

    @pytest.mark.asyncio
    async def test_store_conformance_with_metadata(self) -> None:
        """Store with metadata produces identical metadata after round-trip."""
        thread_nb = f'{THREAD_PREFIX}_store_meta_nb'
        thread_b = f'{THREAD_PREFIX}_store_meta_b'
        meta: dict[str, JsonValue] = {'key': 'value', 'priority': 42, 'nested': {'a': 1}}

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

        nb_state = await read_db_entry(nb_result['context_id'])
        b_state = await read_db_entry(require_context_id(b_result['results'][0]['context_id']))

        assert_db_states_equal(nb_state, b_state)
        assert nb_state['metadata'] == meta

    @pytest.mark.asyncio
    async def test_store_conformance_multimodal_content_type(self) -> None:
        """Content type is 'multimodal' when images present in both paths."""
        thread_nb = f'{THREAD_PREFIX}_store_mm_nb'
        thread_b = f'{THREAD_PREFIX}_store_mm_b'
        image = {'data': CONFORMANCE_PNG_DATA}

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

        nb_state = await read_db_entry(nb_result['context_id'])
        b_state = await read_db_entry(require_context_id(b_result['results'][0]['context_id']))

        assert nb_state['content_type'] == 'multimodal'
        assert b_state['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    async def test_store_conformance_dedup_behavior(self) -> None:
        """Deduplication returns same context_id on second store in both paths."""
        thread_nb = f'{THREAD_PREFIX}_store_dedup_nb'
        thread_b = f'{THREAD_PREFIX}_store_dedup_b'

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

        assert await count_entries_in_thread(thread_nb) == 1
        assert await count_entries_in_thread(thread_b) == 1

    @pytest.mark.asyncio
    async def test_store_conformance_dedup_interleaving(self) -> None:
        """Dedup suppressed when opposite-source entry is interleaved."""
        thread_nb = f'{THREAD_PREFIX}_store_interleave_nb'
        thread_b = f'{THREAD_PREFIX}_store_interleave_b'

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

        nb_count = await count_entries_in_thread(thread_nb)
        b_count = await count_entries_in_thread(thread_b)

        assert nb_count == b_count, (
            f'Entry count mismatch: nonbatch={nb_count} vs batch={b_count}'
        )
        assert nb_count == 3
        assert nb_r3['success'] is True
        assert b_r3['success'] is True

    @pytest.mark.asyncio
    async def test_store_conformance_invalid_images_error(self) -> None:
        """Both paths reject entries with empty image data."""
        with pytest.raises(ToolError, match='Image 0 has empty "data" field'):
            await store_context(
                thread_id=f'{THREAD_PREFIX}_store_inv_img_nb',
                source='user', text='Bad image',
                images=[{'data': ''}],
            )

        with pytest.raises(ToolError, match='Image 0 has empty "data" field'):
            await store_context_batch(
                entries=[{
                    'thread_id': f'{THREAD_PREFIX}_store_inv_img_b',
                    'source': 'user', 'text': 'Bad image',
                    'images': [{'data': ''}],
                }],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_store_conformance_empty_text_error(self) -> None:
        """Both paths reject entries with whitespace-only text."""
        with pytest.raises(ToolError, match='text cannot be empty'):
            await store_context(
                thread_id=f'{THREAD_PREFIX}_store_empty_nb',
                source='user', text='   ',
            )

        with pytest.raises(ToolError, match='text cannot be empty'):
            await store_context_batch(
                entries=[{
                    'thread_id': f'{THREAD_PREFIX}_store_empty_b',
                    'source': 'user', 'text': '   ',
                }],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_store_conformance_response_message_parity(self) -> None:
        """Response messages convey same information about generation state."""
        thread_nb = f'{THREAD_PREFIX}_store_msg_nb'
        thread_b = f'{THREAD_PREFIX}_store_msg_b'

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


@pytest.mark.usefixtures('initialized_server')
class TestErrorConformance:
    """Verify error handling parity between batch and non-batch operations."""

    @pytest.mark.asyncio
    async def test_error_conformance_invalid_source(self) -> None:
        """Batch (atomic) raises ToolError for invalid source."""
        with pytest.raises(ToolError, match='(?i)(invalid source|Missing or invalid source)'):
            await store_context_batch(
                entries=[{
                    'thread_id': f'{THREAD_PREFIX}_err_src_b',
                    'source': 'invalid',
                    'text': 'Invalid source test',
                }],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_error_conformance_missing_text(self) -> None:
        """Batch rejects entry with missing text field."""
        with pytest.raises(ToolError, match='Missing required field: text'):
            await store_context_batch(
                entries=[{
                    'thread_id': f'{THREAD_PREFIX}_err_notext_b',
                    'source': 'user',
                }],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_error_conformance_image_validation_parity(self) -> None:
        """Both paths reject invalid base64 image data with similar messages."""
        invalid_image = {'data': '!!!not-base64!!!', 'mime_type': 'image/png'}

        with pytest.raises(ToolError, match='(?i)(invalid|base64|Image 0)'):
            await store_context(
                thread_id=f'{THREAD_PREFIX}_err_b64_nb',
                source='user', text='Bad base64',
                images=[invalid_image],
            )

        with pytest.raises(ToolError, match='(?i)(invalid|base64|Image 0|Validation)'):
            await store_context_batch(
                entries=[{
                    'thread_id': f'{THREAD_PREFIX}_err_b64_b',
                    'source': 'user', 'text': 'Bad base64',
                    'images': [invalid_image],
                }],
                atomic=True,
            )
