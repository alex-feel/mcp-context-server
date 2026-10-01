"""Tests for execute_store_in_transaction in app.tools._transactions."""

from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastmcp.exceptions import ToolError

from app.repositories.embedding_repository.records import ChunkEmbedding
from app.tools._transactions import EmbeddingsReconcileRequiredError
from app.tools._transactions import execute_store_in_transaction


class TestExecuteStoreInTransaction:
    """Test execute_store_in_transaction shared function."""

    @pytest.fixture
    def mock_repos(self) -> MagicMock:
        """Create a mock RepositoryContainer with all required sub-repositories."""
        repos = MagicMock()
        repos.context.store_with_deduplication = AsyncMock(return_value=('42', False))
        repos.tags.store_tags = AsyncMock()
        repos.tags.replace_tags_for_context = AsyncMock()
        repos.images.store_images = AsyncMock()
        repos.images.replace_images_for_context = AsyncMock()
        repos.embeddings.exists = AsyncMock(return_value=False)
        repos.embeddings.store_chunked = AsyncMock()
        return repos

    @pytest.fixture
    def mock_txn(self) -> MagicMock:
        """Create a mock transaction context."""
        txn = MagicMock()
        txn.backend_type = 'sqlite'
        return txn

    @pytest.mark.asyncio
    async def test_basic_store_new_entry(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Store a new entry with no tags, images, or embeddings."""
        context_id, was_updated, embedding_stored = await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Hello world',
            metadata_str=None,
            summary=None,
            tags=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='test-model',
        )
        assert context_id == '42'
        assert was_updated is False
        assert embedding_stored is False
        mock_repos.context.store_with_deduplication.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_tags_new_entry(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """New entry stores tags via store_tags (not replace)."""
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=['tag1', 'tag2'], validated_images=[],
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.tags.store_tags.assert_called_once_with('42', ['tag1', 'tag2'], txn=mock_txn)
        mock_repos.tags.replace_tags_for_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_tags_dedup_entry(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Deduplicated entry replaces tags via replace_tags_for_context."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=(42, True))
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=['tag1'], validated_images=[],
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.tags.replace_tags_for_context.assert_called_once_with(42, ['tag1'], txn=mock_txn)
        mock_repos.tags.store_tags.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_empty_tags_dedup_entry_clears(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """An explicitly provided empty tags list CLEARS tags on a dedup UPDATE.

        The documented replacement contract distinguishes provided from None:
        [] is a provided value and must replace (clear), matching update_context
        semantics; only None preserves existing tags.
        """
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=('42', True))
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=[], validated_images=[],
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.tags.replace_tags_for_context.assert_called_once_with('42', [], txn=mock_txn)
        mock_repos.tags.store_tags.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_none_tags_dedup_entry_preserves(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """tags=None preserves existing tags on a dedup UPDATE (no tag write)."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=('42', True))
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.tags.replace_tags_for_context.assert_not_called()
        mock_repos.tags.store_tags.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_provided_empty_images_dedup_entry_clears(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """images provided as an empty list CLEARS images on a dedup UPDATE."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=('42', True))
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[], images_provided=True,
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.images.replace_images_for_context.assert_called_once_with('42', [], txn=mock_txn)
        mock_repos.images.store_images.assert_not_called()
        # Providing images (even []) means content_type is NOT preserved.
        dedup_kwargs = mock_repos.context.store_with_deduplication.call_args.kwargs
        assert dedup_kwargs['preserve_content_type_on_dedup'] is False

    @pytest.mark.asyncio
    async def test_store_with_absent_images_dedup_entry_preserves(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """images not provided (None from the caller) preserves existing images."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=('42', True))
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[], images_provided=False,
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.images.replace_images_for_context.assert_not_called()
        mock_repos.images.store_images.assert_not_called()
        dedup_kwargs = mock_repos.context.store_with_deduplication.call_args.kwargs
        assert dedup_kwargs['preserve_content_type_on_dedup'] is True

    @pytest.mark.asyncio
    async def test_store_summary_pending_divergence_raises_reconcile(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """A divergence INSERT with a reused (summary_pending) summary aborts.

        The reused summary was read from a candidate that has since diverged and
        may describe different text; the transaction must abort via the
        reconcile signal so the caller regenerates it for THIS text.
        """
        from app.tools._transactions import EmbeddingsReconcileRequiredError

        mock_repos.context.store_with_deduplication = AsyncMock(return_value=('42', False))
        with pytest.raises(EmbeddingsReconcileRequiredError):
            await execute_store_in_transaction(
                mock_repos, mock_txn,
                owner_id='local',
                visibility='private',
                thread_id='t', source='user', content_type='text',
                text_content='text', metadata_str=None, summary='reused summary',
                tags=None, validated_images=[],
                chunk_embeddings=None, embedding_model='m',
                summary_pending=True,
            )

    @pytest.mark.asyncio
    async def test_store_with_embeddings_new_entry(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """New entry stores embeddings and returns embedding_stored=True."""
        chunk_embeddings = cast(list[ChunkEmbedding], [MagicMock()])
        context_id, was_updated, embedding_stored = await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=chunk_embeddings, embedding_model='m',
        )
        assert embedding_stored is True
        mock_repos.embeddings.store_chunked.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_embeddings_skipped_for_dedup_with_existing(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Deduplicated entry with existing embeddings skips storage."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=(42, True))
        mock_repos.embeddings.exists = AsyncMock(return_value=True)
        chunk_embeddings = cast(list[ChunkEmbedding], [MagicMock()])
        context_id, was_updated, embedding_stored = await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=chunk_embeddings, embedding_model='m',
        )
        assert embedding_stored is False
        mock_repos.embeddings.store_chunked.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_raises_on_failed_dedup(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Raises ToolError when store_with_deduplication returns falsy context_id."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=(0, False))
        with pytest.raises(ToolError, match='Failed to store context'):
            await execute_store_in_transaction(
                mock_repos, mock_txn,
                owner_id='local',
                visibility='private',
                thread_id='t', source='user', content_type='text',
                text_content='text', metadata_str=None, summary=None,
                tags=None, validated_images=[],
                chunk_embeddings=None, embedding_model='m',
            )

    @pytest.mark.asyncio
    async def test_store_raises_reconcile_when_insert_skipped_embeddings(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """INSERT with skipped embeddings + generation enabled raises reconcile signal.

        Models the dedup pre-check / transaction divergence: the caller skipped
        embedding generation expecting an UPDATE, but store_with_deduplication
        inserted a new entry. The transaction must abort so the caller can
        regenerate embeddings outside the transaction and retry.
        """
        # Default fixture returns ('42', False) -- a genuine INSERT.
        with pytest.raises(EmbeddingsReconcileRequiredError) as exc_info:
            await execute_store_in_transaction(
                mock_repos, mock_txn,
                owner_id='local',
                visibility='private',
                thread_id='t', source='user', content_type='text',
                text_content='reconcile me', metadata_str=None, summary=None,
                tags=None, validated_images=[],
                chunk_embeddings=None, embedding_model='m',
                embedding_generation_enabled=True,
            )
        assert exc_info.value.text_content == 'reconcile me'
        # Transaction aborted before any embedding write.
        mock_repos.embeddings.store_chunked.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_no_reconcile_on_dedup_update(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """A dedup UPDATE with skipped embeddings does NOT trigger reconciliation."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=('42', True))
        _, was_updated, embedding_stored = await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=None, embedding_model='m',
            embedding_generation_enabled=True,
        )
        assert was_updated is True
        assert embedding_stored is False

    @pytest.mark.asyncio
    async def test_store_no_reconcile_when_embeddings_present(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """A new INSERT that already carries embeddings does NOT reconcile."""
        chunk_embeddings = cast(list[ChunkEmbedding], [MagicMock()])
        _, was_updated, embedding_stored = await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=chunk_embeddings, embedding_model='m',
            embedding_generation_enabled=True,
        )
        assert was_updated is False
        assert embedding_stored is True
        mock_repos.embeddings.store_chunked.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_no_reconcile_when_generation_disabled(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """With generation disabled (default), a new INSERT with no embeddings is allowed."""
        _, was_updated, embedding_stored = await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=None, embedding_model='m',
            embedding_generation_enabled=False,
        )
        assert was_updated is False
        assert embedding_stored is False
        mock_repos.embeddings.store_chunked.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_images_new_entry(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """New entry stores images via store_images."""
        images = [{'data': 'abc', 'mime_type': 'image/png'}]
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            owner_id='local',
            visibility='private',
            thread_id='t', source='user', content_type='multimodal',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=images,
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.images.store_images.assert_called_once_with('42', images, txn=mock_txn)
        mock_repos.images.replace_images_for_context.assert_not_called()
