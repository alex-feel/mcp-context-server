"""Unit tests for the node-summary reconcile trigger in execute_store_in_transaction.

A reconcile-divergence INSERT must regenerate index_tree per-node summaries even
when embedding generation is disabled, so the node layer is not silently dropped.
The trigger is decoupled from embeddings via the ``nodes_pending`` flag.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.tools._shared import EmbeddingsReconcileRequiredError
from app.tools._shared import execute_store_in_transaction


def _mock_repos(*, was_updated: bool) -> MagicMock:
    repos = MagicMock()
    repos.context.store_with_deduplication = AsyncMock(return_value=('ctx-new', was_updated))
    repos.embeddings.exists = AsyncMock(return_value=False)
    repos.index_nodes.replace_nodes_for_context = AsyncMock()
    return repos


def _txn() -> MagicMock:
    txn = MagicMock()
    txn.connection = MagicMock()
    txn.backend_type = 'sqlite'
    return txn


async def _run(repos: MagicMock, *, nodes_pending: bool) -> tuple[str, bool, bool]:
    return await execute_store_in_transaction(
        repos, _txn(),
        thread_id='t', source='user', content_type='text',
        text_content='body', metadata_str=None, summary=None,
        tags=None, validated_images=[],
        chunk_embeddings=None, embedding_model='m',
        embedding_generation_enabled=False,
        index_nodes=None, nodes_pending=nodes_pending,
    )


@pytest.mark.asyncio
async def test_node_pending_divergence_insert_triggers_reconcile() -> None:
    """Embeddings off + nodes pending + INSERT divergence raises the reconcile signal."""
    repos = _mock_repos(was_updated=False)
    with pytest.raises(EmbeddingsReconcileRequiredError):
        await _run(repos, nodes_pending=True)


@pytest.mark.asyncio
async def test_node_pending_dedup_update_does_not_reconcile() -> None:
    """A genuine dedup UPDATE does not reconcile even with nodes pending."""
    repos = _mock_repos(was_updated=True)
    with patch('app.tools._shared.transaction_heartbeat', new_callable=AsyncMock):
        context_id, was_updated, embedding_stored = await _run(repos, nodes_pending=True)
    assert context_id == 'ctx-new'
    assert was_updated is True
    assert embedding_stored is False


@pytest.mark.asyncio
async def test_no_nodes_pending_insert_does_not_reconcile() -> None:
    """Embeddings off + nodes NOT pending + INSERT keeps prior (no-reconcile) behavior."""
    repos = _mock_repos(was_updated=False)
    with patch('app.tools._shared.transaction_heartbeat', new_callable=AsyncMock):
        context_id, was_updated, embedding_stored = await _run(repos, nodes_pending=False)
    assert context_id == 'ctx-new'
    assert was_updated is False
    assert embedding_stored is False


@pytest.mark.asyncio
async def test_empty_node_list_clears_reconcile_gate() -> None:
    """index_nodes=[] (the post-regeneration value on total degradation) does NOT
    re-fire the reconcile gate, so a node-summary outage cannot abort the store.

    The reconcile handlers coerce a None node result to [] before retrying; this
    asserts [] clears the gate, preserving the never-raise node-layer contract.
    """
    repos = _mock_repos(was_updated=False)
    with patch('app.tools._shared.transaction_heartbeat', new_callable=AsyncMock):
        context_id, was_updated, embedding_stored = await execute_store_in_transaction(
            repos, _txn(),
            thread_id='t', source='user', content_type='text',
            text_content='body', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=None, embedding_model='m',
            embedding_generation_enabled=False,
            index_nodes=[], nodes_pending=True,
        )
    assert context_id == 'ctx-new'
    assert was_updated is False
    assert embedding_stored is False
    repos.index_nodes.replace_nodes_for_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_node_list_does_not_wipe_on_dedup_update() -> None:
    """index_nodes=[] on a dedup UPDATE must NOT clear an existing entry's nodes.

    A reconcile retry can flip to a dedup UPDATE; the [] coerced from total node
    degradation must be suppressed at the write site so it does not delete the
    concurrently-created entry's good node rows.
    """
    repos = _mock_repos(was_updated=True)
    with patch('app.tools._shared.transaction_heartbeat', new_callable=AsyncMock):
        await execute_store_in_transaction(
            repos, _txn(),
            thread_id='t', source='user', content_type='text',
            text_content='body', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=None, embedding_model='m',
            embedding_generation_enabled=False,
            index_nodes=[], nodes_pending=False,
        )
    repos.index_nodes.replace_nodes_for_context.assert_not_awaited()
