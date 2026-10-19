"""Tests for store_context_batch response messages: preserved summaries and stored-versus-generated embedding counts."""

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.access_scope import AccessScope
from app.repositories.context_repository.records import DuplicateCandidate
from app.tools.discovery import get_statistics
from tests.helpers import as_principal
from tests.tools.batch._mocks import make_mock_txn


@pytest.mark.usefixtures('initialized_server')
class TestBatchStoreResponseParity:
    """Batch store response message parity."""

    @pytest.mark.asyncio
    async def test_store_batch_reports_summary_preserved(self):
        """Batch store reports 'summaries preserved' for duplicates with summaries."""
        from app.tools.batch.store import store_context_batch

        mock_summary = 'Test summary content'
        mock_txn, mock_begin_transaction = make_mock_txn()

        mock_settings = MagicMock()
        mock_settings.embedding.enabled = False
        mock_settings.semantic_search.enabled = False
        mock_settings.summary.enabled = True
        mock_settings.summary.min_content_length = 0

        with (
            patch('app.tools.batch.store.settings', mock_settings),
            patch('app.tools.batch.store.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=MagicMock()),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            # Simulate dedup pre-check finding existing summary
            mock_repos.context.check_latest_is_duplicate = AsyncMock(
                return_value=DuplicateCandidate(context_id='42', summary=mock_summary),
            )
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(42, True))

            result = await store_context_batch(
                entries=[{
                    'thread_id': 'test-summary-preserved',
                    'source': 'user',
                    'text': 'Some text that is a duplicate',
                }],
                atomic=False,
            )
            assert result['results'][0]['success'] is True
            assert 'summaries preserved' in result['message']

    @pytest.mark.asyncio
    async def test_store_batch_reports_embedding_stored_vs_generated(self):
        """Batch store distinguishes 'embeddings generated' from 'not stored - duplicates'."""
        from app.tools.batch.store import store_context_batch

        mock_txn, mock_begin_transaction = make_mock_txn()

        mock_settings = MagicMock()
        mock_settings.embedding.enabled = True
        mock_settings.semantic_search.enabled = True
        mock_settings.summary.enabled = False
        mock_settings.summary.min_content_length = 500
        mock_settings.embedding.model = 'test-model'
        mock_settings.embedding.timeout_s = 30
        mock_settings.embedding.max_concurrent = 3

        mock_chunk_embeddings = [('chunk-1', [0.1, 0.2, 0.3])]

        with (
            patch('app.tools.batch.store.settings', mock_settings),
            patch('app.tools.batch.store.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.store.get_embedding_provider') as mock_emb_provider_fn,
            patch('app.tools._generation.get_embedding_provider') as mock_shared_emb_provider_fn,
            patch('app.tools.batch.store.get_summary_provider', return_value=None),
            patch(
                'app.tools._generation._generate_embeddings_for_text',
                new_callable=AsyncMock,
                return_value=mock_chunk_embeddings,
            ),
        ):
            mock_emb_provider = MagicMock()
            mock_emb_provider_fn.return_value = mock_emb_provider
            mock_shared_emb_provider_fn.return_value = mock_emb_provider

            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            # Entry 1: new entry (not a duplicate) -> embedding stored
            # Entry 2: duplicate -> embedding already exists -> not stored
            call_count = 0

            async def mock_store_with_dedup(**_kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return (100, False)  # new entry
                return (101, True)  # deduplicated entry

            mock_repos.context.store_with_deduplication = AsyncMock(
                side_effect=mock_store_with_dedup,
            )
            mock_repos.context.check_latest_is_duplicate = AsyncMock(return_value=None)

            # For entry 2 (deduplicated), embeddings already exist.
            # Accept the optional txn and the scope the store path passes.
            async def mock_embedding_exists(context_id, txn=None, *, scope):
                _ = txn, scope
                return context_id == 101  # entry 2 has existing embeddings

            mock_repos.embeddings.exists = AsyncMock(side_effect=mock_embedding_exists)
            mock_repos.embeddings.store_chunked = AsyncMock()

            result = await store_context_batch(
                entries=[
                    {
                        'thread_id': 'test-emb-stored',
                        'source': 'user',
                        'text': 'First entry text content',
                    },
                    {
                        'thread_id': 'test-emb-stored',
                        'source': 'user',
                        'text': 'Second entry text content',
                    },
                ],
                atomic=False,
            )
            assert result['results'][0]['success'] is True
            assert result['results'][1]['success'] is True
            # 2 embeddings generated, 1 stored (entry 1), 1 not stored (entry 2 = duplicate)
            assert 'not stored - duplicates' in result['message']


@pytest.mark.usefixtures('initialized_server')
class TestBatchStoreScope:
    """The batch store runs its pre-check and every store under the caller's scope."""

    @pytest.mark.parametrize('atomic', [True, False])
    @pytest.mark.asyncio
    async def test_scope_reaches_the_precheck_and_the_store(self, atomic: bool) -> None:
        """One scope per call reaches the pre-check and the deduplicating store on both batch paths."""
        from app.tools.batch.store import store_context_batch

        _, mock_begin_transaction = make_mock_txn()
        mock_settings = MagicMock()
        mock_settings.summary.min_content_length = 0
        mock_settings.access_control.default_visibility = 'private'
        mock_settings.access_control.default_group_grants = 'none'

        with (
            as_principal('bob', groups=['team-x']),
            patch('app.tools.batch.store.settings', mock_settings),
            patch('app.tools.batch.store.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.batch.store.get_summary_provider', return_value=MagicMock()),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend
            precheck = AsyncMock(return_value=DuplicateCandidate(context_id='42', summary='Stored summary'))
            store = AsyncMock(return_value=('42', True))
            mock_repos.context.check_latest_is_duplicate = precheck
            mock_repos.context.store_with_deduplication = store

            result = await store_context_batch(
                entries=[{'thread_id': 'scope-thread', 'source': 'user', 'text': 'Re-sent text'}],
                atomic=atomic,
            )

        expected = AccessScope('bob', frozenset({'team-x'}), may_publish=True)
        precheck_call = precheck.await_args
        store_call = store.await_args
        assert result['results'][0]['success'] is True
        assert precheck_call is not None
        assert precheck_call.kwargs['scope'] == expected
        assert store_call is not None
        assert store_call.kwargs['scope'] == expected


_INGEST_BLOCK: dict[str, object] = {
    'scope': 'kb:handbook',
    'source_id': 'guides/setup.md',
    'meta_digest': 'ab' * 32,
    'synced_at': '2026-10-18T12:00:00Z',
}
_PARTIAL_BLOCK_ERROR = 'metadata.ingest must be an object with exactly the keys meta_digest, scope, source_id, synced_at'


@pytest.mark.usefixtures('initialized_server')
class TestBatchStoreIngestBlock:
    """store_context_batch refuses a malformed ingest block per entry, before any generation."""

    @staticmethod
    def _entries() -> list[dict[str, Any]]:
        return [
            {'thread_id': 'kb:handbook', 'source': 'user', 'text': 'Setup guide', 'metadata': {'ingest': _INGEST_BLOCK}},
            {'thread_id': 'kb:handbook', 'source': 'user', 'text': 'FAQ', 'metadata': {'ingest': {'scope': 'kb:handbook'}}},
            {'thread_id': 'kb:handbook', 'source': 'user', 'text': 'Release notes'},
        ]

    @pytest.mark.asyncio
    async def test_nonatomic_fails_only_the_malformed_entry(self) -> None:
        """The malformed entry fails with the validator's message; its siblings are stored."""
        from app.tools.batch.store import store_context_batch

        result = await store_context_batch(entries=self._entries(), atomic=False)

        by_index = {item['index']: item for item in result['results']}
        assert by_index[1] == {'index': 1, 'success': False, 'context_id': None, 'error': _PARTIAL_BLOCK_ERROR}
        assert by_index[0]['success'] is True
        assert by_index[2]['success'] is True
        assert (result['succeeded'], result['failed']) == (2, 1)

    @pytest.mark.asyncio
    async def test_atomic_aborts_at_the_malformed_index_before_generation(self) -> None:
        """The atomic batch aborts naming the index; no provider is reached and nothing is stored."""
        from app.tools.batch.store import store_context_batch

        embedding_provider = AsyncMock()
        summary_provider = AsyncMock()
        with (
            patch('app.tools.batch.store.get_embedding_provider', return_value=embedding_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=embedding_provider),
            patch('app.tools.batch.store.get_summary_provider', return_value=summary_provider),
            pytest.raises(ToolError) as exc_info,
        ):
            await store_context_batch(entries=self._entries(), atomic=True)

        assert str(exc_info.value) == f'Validation failed for 1 entries. First error at index 1: {_PARTIAL_BLOCK_ERROR}'
        assert embedding_provider.mock_calls == []
        assert summary_provider.mock_calls == []
        stats = await get_statistics()
        assert stats['total_entries'] == 0
