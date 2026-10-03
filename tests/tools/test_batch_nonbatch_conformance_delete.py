"""Conformance of delete_context_batch with delete_context.

Both paths report identical deleted counts, run the same embedding cleanup, and handle unknown IDs the same way.
"""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

import app.tools._delete_cleanup as delete_cleanup_module
from app.settings import AppSettings
from app.startup import ensure_repositories
from app.tools.batch.delete import delete_context_batch
from app.tools.batch.store import store_context_batch
from app.tools.context.delete import delete_context
from app.tools.context.store import store_context
from tests.tools._conformance import THREAD_PREFIX
from tests.tools._conformance import count_entries_in_thread
from tests.tools._conformance import require_context_id


def _fp32_settings() -> AppSettings:
    """Build a settings object with embedding compression turned off.

    The explicit per-entry embedding cleanup on the delete paths is only needed
    for the fp32 layout, whose vec0 virtual table has no foreign key; the
    compressed payload table cascades with the context row instead. A test that
    asserts the cleanup RUNS therefore has to select the fp32 layout.

    Returns:
        An AppSettings instance with compression disabled.
    """
    settings = AppSettings()
    compression = settings.compression.model_copy(update={'enabled': False})
    return settings.model_copy(update={'compression': compression})


@pytest.mark.usefixtures('initialized_server')
class TestDeleteConformance:
    """Verify delete_context and delete_context_batch produce identical behavior."""

    @pytest.mark.asyncio
    async def test_delete_conformance_by_ids(self) -> None:
        """Delete by IDs removes entry and returns same deleted_count."""
        thread_nb = f'{THREAD_PREFIX}_del_ids_nb'
        thread_b = f'{THREAD_PREFIX}_del_ids_b'

        nb_r = await store_context(thread_id=thread_nb, source='user', text='Delete me')
        b_r = await store_context_batch(
            entries=[{'thread_id': thread_b, 'source': 'user', 'text': 'Delete me'}],
            atomic=True,
        )

        nb_del = await delete_context(context_ids=[nb_r['context_id']])
        b_del = await delete_context_batch(
            context_ids=[require_context_id(b_r['results'][0]['context_id'])],
        )

        assert nb_del['deleted_count'] == 1
        assert b_del['deleted_count'] == 1

        assert await count_entries_in_thread(thread_nb) == 0
        assert await count_entries_in_thread(thread_b) == 0

    @pytest.mark.asyncio
    async def test_delete_conformance_by_thread(self) -> None:
        """Delete by thread removes all entries and returns same deleted_count."""
        thread_nb = f'{THREAD_PREFIX}_del_thread_nb'
        thread_b = f'{THREAD_PREFIX}_del_thread_b'

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

        assert await count_entries_in_thread(thread_nb) == 0
        assert await count_entries_in_thread(thread_b) == 0

    @pytest.mark.asyncio
    async def test_delete_conformance_embedding_cleanup(self) -> None:
        """Both paths trigger embedding cleanup when the embedding tables exist."""
        thread_nb = f'{THREAD_PREFIX}_del_embed_nb'
        thread_b = f'{THREAD_PREFIX}_del_embed_b'

        nb_r = await store_context(thread_id=thread_nb, source='user', text='Embed cleanup test')
        b_r = await store_context_batch(
            entries=[{'thread_id': thread_b, 'source': 'user', 'text': 'Embed cleanup test'}],
            atomic=True,
        )

        nb_id = nb_r['context_id']
        b_id = require_context_id(b_r['results'][0]['context_id'])

        mock_delete = AsyncMock(return_value=0)
        repos = await ensure_repositories()

        # The explicit per-entry cleanup is gated on TWO things: the embedding
        # tables having been provisioned (embedding_tables_exist), and the active
        # layout still needing it. Compression replaces the FK-less fp32 vec0 table
        # with a cascading one, so force the fp32 layout to exercise the loop, and
        # force the table signal True, then assert BOTH paths clean up.
        with (
            patch.object(delete_cleanup_module, 'settings', _fp32_settings()),
            patch.object(repos.embeddings, 'delete_all_chunks_bulk', mock_delete),
            patch.object(
                repos.embeddings, 'embedding_tables_exist', AsyncMock(return_value=True),
            ),
        ):
            await delete_context(context_ids=[nb_id])
            await delete_context_batch(context_ids=[b_id])

        cleaned = [cid for call in mock_delete.call_args_list for cid in call.args[0]]
        assert nb_id in cleaned, f'Non-batch did not clean up embeddings for {nb_id}'
        assert b_id in cleaned, f'Batch did not clean up embeddings for {b_id}'

    @pytest.mark.asyncio
    async def test_delete_conformance_nonexistent_id(self) -> None:
        """Both paths handle non-existent IDs gracefully with deleted_count=0."""
        nb_del = await delete_context(context_ids=['0190abcdef1234567890abcd000f423d'])
        b_del = await delete_context_batch(context_ids=['0190abcdef1234567890abcd000f423c'])

        assert nb_del['deleted_count'] == 0
        assert b_del['deleted_count'] == 0
        assert nb_del['success'] is True
        assert b_del['success'] is True
