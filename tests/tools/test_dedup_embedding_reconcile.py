"""Tests for the dedup pre-check / transaction divergence reconciliation.

When the read-only deduplication pre-check skips embedding generation for a
likely duplicate that already has embeddings, a concurrent same-thread write can
flip the in-transaction outcome to a genuine INSERT. Committing then would leave
a row with no embeddings while generation is enabled, violating the
generation-first guarantee. execute_store_in_transaction raises
EmbeddingsReconcileRequiredError in that case; store_context and
store_context_batch catch it, regenerate embeddings OUTSIDE the transaction, and
retry. These tests exercise the catch/regenerate/retry loop wiring.

Also guards the variant-aware semantic_distance contract wording so the
"L2 Euclidean (LOWER = better)" label with fixed thresholds does not regress
(it is wrong under the default ip compression variant, which returns a negated
inner product).
"""

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.tools._shared import EmbeddingsReconcileRequiredError


def _make_mock_txn() -> tuple[MagicMock, object]:
    """Create a mock transaction context manager (mirrors test_batch_bug_fixes)."""
    mock_txn = MagicMock()
    mock_txn.connection = MagicMock()
    mock_txn.backend_type = 'sqlite'

    @asynccontextmanager
    async def mock_begin_transaction():
        yield mock_txn

    return mock_txn, mock_begin_transaction


def _fake_chunk_embeddings() -> list[tuple[str, list[float]]]:
    """Stand-in embeddings; execute_store_in_transaction is mocked so shape is irrelevant."""
    return [('chunk-0', [0.1, 0.2, 0.3])]


@pytest.mark.usefixtures('initialized_server')
class TestStoreContextReconcile:
    """store_context regenerates skipped embeddings when the store inserts a new row."""

    @pytest.mark.asyncio
    async def test_reconcile_regenerates_and_retries(self) -> None:
        """A reconcile signal triggers one regeneration and a successful retry."""
        from app.tools.context import store_context

        _, mock_begin_transaction = _make_mock_txn()

        with (
            patch('app.tools.context.ensure_repositories') as mock_repos_fn,
            patch('app.tools.context.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.context.get_summary_provider', return_value=None),
            patch('app.tools.context.execute_store_in_transaction') as mock_exec,
            patch(
                'app.tools._shared.generate_embeddings_with_timeout',
                new_callable=AsyncMock,
                return_value=_fake_chunk_embeddings(),
            ) as mock_gen_emb,
            patch(
                'app.tools._shared.generate_compression_with_timeout',
                new_callable=AsyncMock,
                side_effect=lambda emb: emb,
            ),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            # Pre-check sees a duplicate WITH embeddings -> generation skipped.
            mock_repos.context.check_latest_is_duplicate = AsyncMock(return_value='dup-id')
            mock_repos.embeddings.exists = AsyncMock(return_value=True)

            # First transaction diverges (INSERT); second succeeds after regeneration.
            mock_exec.side_effect = [
                EmbeddingsReconcileRequiredError('reconcile this entry'),
                ('ctx-1', False, True),
            ]

            result = await store_context(
                thread_id='reconcile-thread',
                source='user',
                text='reconcile this entry',
            )

        assert result['success'] is True
        assert result['context_id'] == 'ctx-1'
        # Exactly one retry of the transaction.
        assert mock_exec.call_count == 2
        # Embeddings were regenerated exactly once (only in the reconcile path,
        # since the pre-check skipped the initial generation).
        assert mock_gen_emb.await_count == 1

    @pytest.mark.asyncio
    async def test_reconcile_does_not_loop_forever(self) -> None:
        """A repeated reconcile signal surfaces a ToolError rather than looping."""
        from app.tools.context import store_context

        _, mock_begin_transaction = _make_mock_txn()

        with (
            patch('app.tools.context.ensure_repositories') as mock_repos_fn,
            patch('app.tools.context.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.context.get_summary_provider', return_value=None),
            patch('app.tools.context.execute_store_in_transaction') as mock_exec,
            patch(
                'app.tools._shared.generate_embeddings_with_timeout',
                new_callable=AsyncMock,
                return_value=_fake_chunk_embeddings(),
            ),
            patch(
                'app.tools._shared.generate_compression_with_timeout',
                new_callable=AsyncMock,
                side_effect=lambda emb: emb,
            ),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            mock_repos.context.check_latest_is_duplicate = AsyncMock(return_value='dup-id')
            mock_repos.embeddings.exists = AsyncMock(return_value=True)

            # Always diverges -- the one-shot guard must stop the loop.
            mock_exec.side_effect = EmbeddingsReconcileRequiredError('reconcile this entry')

            with pytest.raises(ToolError, match='Failed to reconcile embeddings'):
                await store_context(
                    thread_id='reconcile-thread',
                    source='user',
                    text='reconcile this entry',
                )


@pytest.mark.usefixtures('initialized_server')
class TestStoreBatchReconcile:
    """store_context_batch regenerates skipped embeddings on dedup divergence."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('atomic', [True, False])
    async def test_batch_reconcile_regenerates_and_retries(self, atomic: bool) -> None:
        """Atomic and non-atomic batch store both reconcile a diverged INSERT."""
        from app.tools.batch import store_context_batch

        _, mock_begin_transaction = _make_mock_txn()

        with (
            patch('app.tools.batch.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.batch.get_summary_provider', return_value=None),
            patch('app.tools.batch.execute_store_in_transaction') as mock_exec,
            patch(
                'app.tools.batch.generate_embeddings_with_timeout',
                new_callable=AsyncMock,
                return_value=_fake_chunk_embeddings(),
            ) as mock_gen_emb,
            patch(
                'app.tools.batch.generate_compression_with_timeout',
                new_callable=AsyncMock,
                side_effect=lambda emb: emb,
            ),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            mock_repos.context.check_latest_is_duplicate = AsyncMock(return_value='dup-id')
            mock_repos.embeddings.exists = AsyncMock(return_value=True)

            mock_exec.side_effect = [
                EmbeddingsReconcileRequiredError('batch reconcile entry'),
                ('ctx-1', False, True),
            ]

            result = await store_context_batch(
                entries=[{
                    'thread_id': 'batch-reconcile-thread',
                    'source': 'user',
                    'text': 'batch reconcile entry',
                }],
                atomic=atomic,
            )

        assert result['results'][0]['success'] is True
        assert result['results'][0]['context_id'] == 'ctx-1'
        assert mock_exec.call_count == 2
        assert mock_gen_emb.await_count == 1


class TestSemanticDistanceContractWording:
    """Guards the variant-aware semantic_distance contract (Bug 2 regression)."""

    def test_search_tool_drops_fixed_threshold_heuristic(self) -> None:
        """The misleading fixed-band heuristic is gone; variant wording is present."""
        content = Path('app/tools/search.py').read_text(encoding='utf-8')
        assert '<0.5 very similar' not in content
        assert 'negated inner product' in content

    def test_types_describe_variant_aware_distance(self) -> None:
        """TypedDicts no longer assert an unconditional L2 Euclidean distance."""
        content = Path('app/types.py').read_text(encoding='utf-8')
        assert 'negated inner product' in content
        assert 'L2 Euclidean distance (LOWER = better)' not in content
