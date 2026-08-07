"""Per-image ``metadata`` has ONE canonical wire shape: a JSON-encoded string.

The typed single-entry tools declare ``images`` as ``list[dict[str, str]]``, so
per-image metadata can only be a string; the write path json.dumps that already
stringified value and the read path json.loads it straight back. Two layers used
to disagree with that contract:

* ``ImageDict.metadata`` declared ``dict[str, str] | None``, and FastMCP derives
  the strict ``get_context_by_ids`` outputSchema from it -- so an image stored
  through the documented string contract came back as a string the tool's own
  schema rejected, making the entry unreadable for any spec-conformant client.
* The untyped batch path (``list[dict[str, Any]]``) never checked the shape, so
  it accepted and stored a dict the single-entry tool refuses -- producing the
  same unreadable entry from the other direction.
"""

import base64

import pytest
from pydantic import TypeAdapter

from app.tools.batch import store_context_batch
from app.tools.context import get_context_by_ids
from app.tools.context import store_context
from app.types import ContextEntryDict

# Minimal valid 1x1 PNG.
_PNG = base64.b64encode(bytes([
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


@pytest.mark.usefixtures('initialized_server')
class TestImageMetadataWireShape:
    """Round-trip and rejection behavior for per-image metadata."""

    @pytest.mark.asyncio
    async def test_string_image_metadata_round_trips_through_declared_type(self) -> None:
        """An image stored with JSON-string metadata reads back and satisfies the schema.

        The declared return type is what FastMCP publishes as the tool's
        outputSchema, and a spec-conformant client validates the response against
        it -- which is where the mismatch surfaced: the entry stored fine and only
        became unreadable on retrieval. Validating the real response against the
        declared type reproduces that check without a live client.
        """
        stored = await store_context(
            thread_id='image-metadata-contract',
            source='user',
            text='Entry with an annotated image',
            images=[{'data': _PNG, 'mime_type': 'image/png', 'metadata': '{"iso": 100}'}],
        )

        entries = await get_context_by_ids(context_ids=[stored['context_id']], include_images=True)
        images = entries[0].get('images')
        assert images is not None
        assert images[0].get('metadata') == '{"iso": 100}'

        # The response must satisfy the tool's own declared return type.
        TypeAdapter(list[ContextEntryDict]).validate_python(entries)

    @pytest.mark.asyncio
    async def test_batch_rejects_dict_image_metadata(self) -> None:
        """The untyped batch path refuses what the typed single-entry path refuses.

        A dict here previously stored successfully and then failed the retrieval
        output schema, leaving the entry permanently unreadable.
        """
        result = await store_context_batch(
            entries=[{
                'thread_id': 'image-metadata-contract-batch',
                'source': 'user',
                'text': 'Entry with dict image metadata',
                'images': [{'data': _PNG, 'mime_type': 'image/png',
                            'metadata': {'width': 1, 'height': 1}}],
            }],
            atomic=False,
        )

        assert result['succeeded'] == 0
        error = result['results'][0]['error']
        assert error is not None
        assert 'metadata must be a JSON-encoded string' in error

    @pytest.mark.asyncio
    async def test_batch_accepts_string_image_metadata(self) -> None:
        """The canonical string shape is accepted on the batch path."""
        result = await store_context_batch(
            entries=[{
                'thread_id': 'image-metadata-contract-batch-ok',
                'source': 'user',
                'text': 'Entry with string image metadata',
                'images': [{'data': _PNG, 'mime_type': 'image/png',
                            'metadata': '{"width": 1}'}],
            }],
            atomic=False,
        )

        assert result['succeeded'] == 1
        context_id = result['results'][0]['context_id']
        assert context_id is not None
        entries = await get_context_by_ids(context_ids=[context_id], include_images=True)
        images = entries[0].get('images')
        assert images is not None
        assert images[0].get('metadata') == '{"width": 1}'
