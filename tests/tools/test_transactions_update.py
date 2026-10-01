"""Tests for execute_update_in_transaction in app.tools._transactions."""

from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from app.repositories.embedding_repository.records import ChunkEmbedding
from app.tools._transactions import EntryNotFoundError
from app.tools._transactions import execute_update_in_transaction


class TestExecuteUpdateInTransaction:
    """Test execute_update_in_transaction shared function."""

    @pytest.fixture
    def mock_repos(self) -> MagicMock:
        """Create a mock RepositoryContainer with all required sub-repositories."""
        repos = MagicMock()
        repos.context.update_context_entry = AsyncMock(return_value=(True, ['text']))
        repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
        repos.context.update_content_type = AsyncMock()
        repos.context.touch_updated_at = AsyncMock(return_value=True)
        repos.context.get_content_type = AsyncMock(return_value='text')
        repos.context.entry_exists = AsyncMock(return_value=True)
        repos.tags.replace_tags_for_context = AsyncMock()
        repos.images.replace_images_for_context = AsyncMock()
        repos.images.count_images_for_context = AsyncMock(return_value=0)
        repos.embeddings.delete_all_chunks = AsyncMock()
        repos.embeddings.embedding_tables_exist = AsyncMock(return_value=False)
        repos.embeddings.store_chunked = AsyncMock()
        return repos

    @pytest.fixture
    def mock_txn(self) -> MagicMock:
        """Create a mock transaction context."""
        txn = MagicMock()
        txn.backend_type = 'sqlite'
        return txn

    @pytest.mark.asyncio
    async def test_basic_text_update(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Update text field returns updated_fields with text."""
        updated_fields, summary_cleared = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text='New text',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )
        assert 'text' in updated_fields
        assert summary_cleared is False
        mock_repos.context.update_context_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_patch_update(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Apply metadata_patch calls patch_metadata."""
        updated_fields, _ = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text=None,
            metadata=None,
            metadata_patch={'key': 'value'},
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )
        assert 'metadata' in updated_fields
        mock_repos.context.patch_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_tags_replacement(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Tags provided triggers replace_tags_for_context."""
        updated_fields, _ = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text=None,
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=['new-tag'],
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )
        assert 'tags' in updated_fields
        mock_repos.tags.replace_tags_for_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_images_removal(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Empty images list removes all images and sets content_type to text."""
        updated_fields, _ = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text=None,
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=[],
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )
        assert 'images' in updated_fields
        assert 'content_type' in updated_fields
        mock_repos.images.replace_images_for_context.assert_called_once_with(
            '0190abcdef1234567890abcd00000001', [], txn=mock_txn,
        )
        mock_repos.context.update_content_type.assert_called_once_with(
            '0190abcdef1234567890abcd00000001', 'text', txn=mock_txn,
        )

    @pytest.mark.asyncio
    async def test_embeddings_regeneration(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Chunk embeddings provided triggers delete+store cycle."""
        chunk_embeddings = cast(list[ChunkEmbedding], [MagicMock()])
        updated_fields, _ = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text='New text',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=chunk_embeddings,
            embedding_model='m',
        )
        assert 'embedding' in updated_fields
        mock_repos.embeddings.delete_all_chunks.assert_called_once_with(
            '0190abcdef1234567890abcd00000001', txn=mock_txn,
        )
        mock_repos.embeddings.store_chunked.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_change_without_provider_clears_stale_embeddings(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Text changed, no new embeddings, tables exist -> stale chunks deleted.

        When an update changes text but no embedding provider regenerates vectors,
        the stored chunks describe the REPLACED text and must be DELETEd so semantic
        search cannot match the old content. Guarded by embedding_tables_exist.
        """
        mock_repos.images.count_images_for_context = AsyncMock(return_value=0)
        mock_repos.context.get_content_type = AsyncMock(return_value='text')
        mock_repos.context.update_content_type = AsyncMock()
        mock_repos.embeddings.embedding_tables_exist = AsyncMock(return_value=True)
        mock_repos.embeddings.delete_all_chunks = AsyncMock(return_value=True)

        updated_fields, _ = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text='Replaced text',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )

        assert 'embedding' in updated_fields
        mock_repos.embeddings.delete_all_chunks.assert_called_once_with(
            '0190abcdef1234567890abcd00000001', txn=mock_txn,
        )
        mock_repos.embeddings.store_chunked.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_change_without_provider_tables_absent_is_noop(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Same text-change-without-provider path but embeddings were never
        provisioned -> safe no-op (no delete, 'embedding' not in updated_fields)."""
        mock_repos.images.count_images_for_context = AsyncMock(return_value=0)
        mock_repos.context.get_content_type = AsyncMock(return_value='text')
        mock_repos.context.update_content_type = AsyncMock()
        mock_repos.embeddings.embedding_tables_exist = AsyncMock(return_value=False)
        mock_repos.embeddings.delete_all_chunks = AsyncMock(return_value=True)

        updated_fields, _ = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text='Replaced text',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )

        assert 'embedding' not in updated_fields
        mock_repos.embeddings.delete_all_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_raises_on_failed_entry_update(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Raises EntryNotFoundError when update_context_entry reports no such row."""
        mock_repos.context.update_context_entry = AsyncMock(return_value=(False, []))
        with pytest.raises(EntryNotFoundError, match='not found'):
            await execute_update_in_transaction(
                mock_repos, mock_txn,
                context_id='0190abcdef1234567890abcd00000001',
                text='New text',
                metadata=None,
                metadata_patch=None,
                summary=None,
                clear_summary=False,
                tags=None,
                images=None,
                validated_images=[],
                chunk_embeddings=None,
                embedding_model='m',
            )

    @pytest.mark.asyncio
    async def test_update_raises_on_failed_metadata_patch(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Raises EntryNotFoundError when patch_metadata reports no such row."""
        mock_repos.context.patch_metadata = AsyncMock(return_value=(False, []))
        with pytest.raises(EntryNotFoundError, match='not found'):
            await execute_update_in_transaction(
                mock_repos, mock_txn,
                context_id='0190abcdef1234567890abcd00000001',
                text=None,
                metadata=None,
                metadata_patch={'key': 'value'},
                summary=None,
                clear_summary=False,
                tags=None,
                images=None,
                validated_images=[],
                chunk_embeddings=None,
                embedding_model='m',
            )

    @pytest.mark.asyncio
    async def test_tags_only_update_missing_parent_raises_not_found(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """A tags-only update against a vanished parent row raises EntryNotFoundError.

        The parent can disappear between the pre-generation existence check and this
        transaction (concurrent delete). Without the guard the tags write would fire a
        foreign-key insert against a missing parent, charging the circuit breaker.
        """
        mock_repos.context.entry_exists = AsyncMock(return_value=False)
        with pytest.raises(EntryNotFoundError, match='not found'):
            await execute_update_in_transaction(
                mock_repos, mock_txn,
                context_id='0190abcdef1234567890abcd00000001',
                text=None,
                metadata=None,
                metadata_patch=None,
                summary=None,
                clear_summary=False,
                tags=['new-tag'],
                images=None,
                validated_images=[],
                chunk_embeddings=None,
                embedding_model='m',
            )
        mock_repos.tags.replace_tags_for_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_images_only_update_missing_parent_raises_not_found(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """An images-only update against a vanished parent row raises EntryNotFoundError.

        Same concurrent-delete race as the tags-only path; the guard stops the image
        write from touching a missing parent and charging the circuit breaker.
        """
        mock_repos.context.entry_exists = AsyncMock(return_value=False)
        with pytest.raises(EntryNotFoundError, match='not found'):
            await execute_update_in_transaction(
                mock_repos, mock_txn,
                context_id='0190abcdef1234567890abcd00000001',
                text=None,
                metadata=None,
                metadata_patch=None,
                summary=None,
                clear_summary=False,
                tags=None,
                images=[],
                validated_images=[],
                chunk_embeddings=None,
                embedding_model='m',
            )
        mock_repos.images.replace_images_for_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_summary_cleared_flag(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """clear_summary=True is returned as summary_cleared."""
        _, summary_cleared = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text='Short',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=True,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )
        assert summary_cleared is True
