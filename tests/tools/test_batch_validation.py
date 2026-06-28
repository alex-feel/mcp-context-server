"""Per-entry type validation for store_context_batch / update_context_batch.

The single-entry tools are Pydantic-typed; the batch tools take untyped dicts, so
they must reject a non-object metadata and a non-list tags per entry (a non-dict
metadata breaks search/metadata_filters; a bare-string tags would otherwise be
stored one character per tag). The same parity guard applies to images: a non-list
images (or a list with a non-dict element) is rejected per entry instead of reaching
the image normalizer and raising a raw AttributeError that aborts the whole batch.
"""

import base64

import pytest
from fastmcp.exceptions import ToolError


@pytest.mark.usefixtures('initialized_server')
class TestStoreBatchEntryTypeValidation:
    @pytest.mark.asyncio
    async def test_rejects_non_dict_metadata(self) -> None:
        from app.tools.batch import store_context_batch

        result = await store_context_batch(
            entries=[{'thread_id': 't', 'source': 'user', 'text': 'hi', 'metadata': 'not-a-dict'}],
            atomic=False,
        )
        assert result['results'][0]['success'] is False
        assert 'metadata must be a JSON object' in (result['results'][0]['error'] or '')

    @pytest.mark.asyncio
    async def test_rejects_string_tags(self) -> None:
        from app.tools.batch import store_context_batch

        result = await store_context_batch(
            entries=[{'thread_id': 't', 'source': 'user', 'text': 'hi', 'tags': 'notalist'}],
            atomic=False,
        )
        assert result['results'][0]['success'] is False
        assert 'tags must be a list of strings' in (result['results'][0]['error'] or '')

    @pytest.mark.asyncio
    async def test_rejects_non_string_tag_element(self) -> None:
        from app.tools.batch import store_context_batch

        result = await store_context_batch(
            entries=[{'thread_id': 't', 'source': 'user', 'text': 'hi', 'tags': ['ok', 123]}],
            atomic=False,
        )
        assert result['results'][0]['success'] is False
        assert 'tags must be a list of strings' in (result['results'][0]['error'] or '')


# A minimal but well-formed base64 PNG signature: enough to pass image normalization
# so the only reason an entry can fail is the deliberately malformed images SHAPE.
_VALID_IMAGE_B64 = base64.b64encode(b'\x89PNG\r\n\x1a\n').decode()


@pytest.mark.usefixtures('initialized_server')
class TestStoreBatchImagesShapeValidation:
    """A dict-as-images (or any non-list images) is a per-entry shape error, not a batch abort.

    A client may mistakenly pass a single image object as `images` instead of a
    one-element list. Without the shape guard that dict reaches the image normalizer,
    which iterates the dict's string keys and calls .get() on a str, raising a raw
    AttributeError that aborts the WHOLE non-atomic batch and silently loses the valid
    sibling entries. The guard turns it into a per-entry validation error so siblings
    still store, matching how metadata/tags shape errors behave.
    """

    @pytest.mark.asyncio
    async def test_nonatomic_dict_images_fails_only_that_entry_siblings_stored(self) -> None:
        from app.tools.batch import store_context_batch

        result = await store_context_batch(
            entries=[
                {'thread_id': 'images-shape', 'source': 'user', 'text': 'first valid entry'},
                {
                    'thread_id': 'images-shape',
                    'source': 'user',
                    'text': 'malformed images entry',
                    # A bare image object instead of a list of image objects.
                    'images': {'data': _VALID_IMAGE_B64},
                },
                {'thread_id': 'images-shape', 'source': 'user', 'text': 'second valid entry'},
            ],
            atomic=False,
        )

        by_index = {r['index']: r for r in result['results']}
        # The malformed entry is reported as a per-entry failure, not a batch abort.
        assert by_index[1]['success'] is False
        assert 'images must be a list of objects' in (by_index[1]['error'] or '')
        # The two valid siblings are stored; the batch is not aborted.
        assert by_index[0]['success'] is True
        assert by_index[0]['context_id'] is not None
        assert by_index[2]['success'] is True
        assert by_index[2]['context_id'] is not None
        assert result['total'] == 3
        assert result['succeeded'] == 2
        assert result['failed'] == 1

    @pytest.mark.asyncio
    async def test_atomic_dict_images_reports_shape_error_not_attributeerror(self) -> None:
        from app.tools.batch import store_context_batch

        with pytest.raises(ToolError, match='images must be a list of objects') as exc_info:
            await store_context_batch(
                entries=[{
                    'thread_id': 'images-shape-atomic',
                    'source': 'user',
                    'text': 'malformed images entry',
                    'images': {'data': _VALID_IMAGE_B64},
                }],
                atomic=True,
            )
        # The clear shape error, not a raw "'str' object has no attribute 'get'" AttributeError.
        assert 'has no attribute' not in str(exc_info.value)


@pytest.mark.usefixtures('initialized_server')
class TestUpdateBatchImagesShapeValidation:
    """update_context_batch applies the same per-entry images shape guard as store."""

    @pytest.mark.asyncio
    async def test_nonatomic_dict_images_fails_only_that_entry_siblings_updated(self) -> None:
        from app.tools.batch import store_context_batch
        from app.tools.batch import update_context_batch

        store_result = await store_context_batch(
            entries=[
                {'thread_id': 'images-shape-upd', 'source': 'user', 'text': 'first original'},
                {'thread_id': 'images-shape-upd', 'source': 'user', 'text': 'second original'},
                {'thread_id': 'images-shape-upd', 'source': 'user', 'text': 'third original'},
            ],
        )
        cid_0 = store_result['results'][0]['context_id']
        cid_1 = store_result['results'][1]['context_id']
        cid_2 = store_result['results'][2]['context_id']
        assert cid_0 is not None
        assert cid_1 is not None
        assert cid_2 is not None

        result = await update_context_batch(
            updates=[
                {'context_id': cid_0, 'tags': ['updated']},
                # A bare image object instead of a list of image objects.
                {'context_id': cid_1, 'images': {'data': _VALID_IMAGE_B64}},
                {'context_id': cid_2, 'tags': ['updated']},
            ],
            atomic=False,
        )

        by_index = {r['index']: r for r in result['results']}
        # The malformed update is a per-entry failure, not a batch abort.
        assert by_index[1]['success'] is False
        assert 'images must be a list of objects' in (by_index[1]['error'] or '')
        # The two valid sibling updates are applied; the batch is not aborted.
        assert by_index[0]['success'] is True
        assert by_index[2]['success'] is True
        assert result['total'] == 3
        assert result['succeeded'] == 2
        assert result['failed'] == 1

    @pytest.mark.asyncio
    async def test_atomic_dict_images_reports_shape_error_not_attributeerror(self) -> None:
        from app.tools.batch import store_context_batch
        from app.tools.batch import update_context_batch

        store_result = await store_context_batch(
            entries=[{'thread_id': 'images-shape-upd-atomic', 'source': 'user', 'text': 'original text'}],
        )
        cid = store_result['results'][0]['context_id']
        assert cid is not None

        with pytest.raises(ToolError, match='images must be a list of objects') as exc_info:
            await update_context_batch(
                updates=[{'context_id': cid, 'images': {'data': _VALID_IMAGE_B64}}],
                atomic=True,
            )
        # The clear shape error, not a raw "'str' object has no attribute 'get'" AttributeError.
        assert 'has no attribute' not in str(exc_info.value)
