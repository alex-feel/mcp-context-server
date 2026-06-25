"""Per-entry type validation for store_context_batch / update_context_batch.

The single-entry tools are Pydantic-typed; the batch tools take untyped dicts, so
they must reject a non-object metadata and a non-list tags per entry (a non-dict
metadata breaks search/metadata_filters; a bare-string tags would otherwise be
stored one character per tag).
"""

import pytest


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
