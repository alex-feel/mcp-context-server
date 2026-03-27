"""Tests for batch and non-batch operation bug fixes.

Covers:
- Silent failure handling in update_context_batch
- Image validation parity (empty data, mime_type defaults, index in errors)
- Embedding cleanup for non-ID batch deletes on SQLite
- Connection retry in non-atomic batch operations
- Error formatting (format_exception_message usage)
- Response message parity (summaries preserved, embedding stored vs generated, summaries cleared)
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError


# Error formatting conformance
class TestErrorFormatting:
    """Verify all tool files use format_exception_message instead of str(e)."""

    @pytest.mark.parametrize('file_path', [
        'app/tools/context.py',
        'app/tools/search.py',
        'app/tools/discovery.py',
        'app/tools/batch.py',
    ])
    def test_no_str_e_in_tool_errors(self, file_path: str) -> None:
        """Verify no tool file uses str(e) in error contexts."""
        content = Path(file_path).read_text()
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if 'str(e)' in stripped:
                # Allow in logger calls (internal diagnostics)
                if stripped.startswith('logger.'):
                    continue
                pytest.fail(f'{file_path}:{i}: Found str(e) in non-logger context: {stripped}')


# update_context image validation parity
@pytest.mark.usefixtures('initialized_server')
class TestUpdateContextImageValidation:
    """Empty data check and per-image index in update_context."""

    @pytest.mark.asyncio
    async def test_update_context_rejects_empty_image_data(self):
        """update_context rejects images with empty data field."""
        from app.tools.batch import store_context_batch
        from app.tools.context import update_context

        store_result = await store_context_batch(
            entries=[{'thread_id': 't', 'source': 'user', 'text': 'hello'}],
        )
        cid = store_result['results'][0]['context_id']

        with pytest.raises(ToolError, match='Image 0 has empty "data" field'):
            await update_context(context_id=cid, images=[{'data': ''}])

    @pytest.mark.asyncio
    async def test_update_context_rejects_whitespace_image_data(self):
        """update_context rejects images with whitespace-only data."""
        from app.tools.batch import store_context_batch
        from app.tools.context import update_context

        store_result = await store_context_batch(
            entries=[{'thread_id': 't', 'source': 'user', 'text': 'hello'}],
        )
        cid = store_result['results'][0]['context_id']

        with pytest.raises(ToolError, match='Image 0 has empty "data" field'):
            await update_context(context_id=cid, images=[{'data': '   '}])

    @pytest.mark.asyncio
    async def test_update_context_image_errors_include_index(self):
        """Error messages include per-image index."""
        from app.tools.batch import store_context_batch
        from app.tools.context import update_context

        store_result = await store_context_batch(
            entries=[{'thread_id': 't', 'source': 'user', 'text': 'hello'}],
        )
        cid = store_result['results'][0]['context_id']

        valid_image = base64.b64encode(b'\x89PNG\r\n').decode()
        with pytest.raises(ToolError, match='Image 1') as exc_info:
            await update_context(
                context_id=cid,
                images=[
                    {'data': valid_image},
                    {'data': 'not-valid-base64!!!'},
                ],
            )
        assert '1' in str(exc_info.value)


# Batch image validation parity
@pytest.mark.usefixtures('initialized_server')
class TestBatchImageValidation:
    """mime_type defaults and empty data checks in batch operations."""

    @pytest.mark.asyncio
    async def test_store_batch_rejects_empty_image_data(self):
        """store_context_batch rejects entries with empty image data."""
        from app.tools.batch import store_context_batch

        # atomic=True raises ToolError on validation failure
        with pytest.raises(ToolError, match='empty "data" field'):
            await store_context_batch(
                entries=[{
                    'thread_id': 'test-thread',
                    'source': 'user',
                    'text': 'Test entry',
                    'images': [{'data': '', 'mime_type': 'image/png'}],
                }],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_update_batch_rejects_empty_image_data(self):
        """update_context_batch rejects entries with empty image data."""
        from app.tools.batch import store_context_batch
        from app.tools.batch import update_context_batch

        # Create an entry first
        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-thread',
                'source': 'user',
                'text': 'Original text',
            }],
        )
        context_id = store_result['results'][0]['context_id']

        # atomic=True raises ToolError on validation failure
        with pytest.raises(ToolError, match='empty "data" field'):
            await update_context_batch(
                updates=[{
                    'context_id': context_id,
                    'images': [{'data': '   '}],
                }],
                atomic=True,
            )

    @pytest.mark.asyncio
    async def test_store_batch_defaults_mime_type(self):
        """store_context_batch defaults mime_type to image/png."""
        from app.tools.batch import store_context_batch
        from app.tools.context import get_context_by_ids

        valid_image = base64.b64encode(b'\x89PNG\r\n\x1a\n').decode()
        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-mime',
                'source': 'user',
                'text': 'Test entry',
                'images': [{'data': valid_image}],
            }],
        )
        assert store_result['results'][0]['success'] is True
        cid = store_result['results'][0]['context_id']

        # get_context_by_ids returns list[ContextEntryDict]
        entries = await get_context_by_ids(context_ids=[cid], include_images=True)
        images = entries[0].get('images', [])
        assert len(images) >= 1
        assert images[0]['mime_type'] == 'image/png'

    @pytest.mark.asyncio
    async def test_update_batch_defaults_mime_type(self):
        """update_context_batch defaults mime_type to image/png."""
        from app.tools.batch import store_context_batch
        from app.tools.batch import update_context_batch
        from app.tools.context import get_context_by_ids

        # Create entry without images
        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-mime-upd',
                'source': 'user',
                'text': 'Original text',
            }],
        )
        context_id = store_result['results'][0]['context_id']

        # Update with image lacking mime_type
        valid_image = base64.b64encode(b'\x89PNG\r\n\x1a\n').decode()
        update_result = await update_context_batch(
            updates=[{
                'context_id': context_id,
                'images': [{'data': valid_image}],
            }],
        )
        assert update_result['results'][0]['success'] is True

        # get_context_by_ids returns list[ContextEntryDict]
        entries = await get_context_by_ids(context_ids=[context_id], include_images=True)
        images = entries[0].get('images', [])
        assert len(images) >= 1
        assert images[0]['mime_type'] == 'image/png'


def _make_mock_txn():
    """Create a properly configured mock transaction context manager."""
    mock_txn = MagicMock()
    mock_txn.connection = MagicMock()
    mock_txn.backend_type = 'sqlite'

    @asynccontextmanager
    async def mock_begin_transaction():
        yield mock_txn

    return mock_txn, mock_begin_transaction


# Silent failure handling
@pytest.mark.usefixtures('initialized_server')
class TestUpdateBatchFailureHandling:
    """Silent failure swallowing in update_context_batch."""

    @pytest.mark.asyncio
    async def test_update_batch_atomic_raises_on_update_entry_failure(self):
        """Atomic mode raises ToolError on update_context_entry failure."""
        from app.tools.batch import store_context_batch
        from app.tools.batch import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-atomic-fail',
                'source': 'user',
                'text': 'Original text',
            }],
        )
        context_id = store_result['results'][0]['context_id']

        mock_txn, mock_begin_transaction = _make_mock_txn()

        with patch('app.tools.batch.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=(True, 'user'))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            mock_repos.context.update_context_entry = AsyncMock(return_value=(False, []))

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            with pytest.raises(ToolError, match='Failed to update context entry'):
                await update_context_batch(
                    updates=[{
                        'context_id': context_id,
                        'text': 'Updated text',
                    }],
                    atomic=True,
                )

    @pytest.mark.asyncio
    async def test_update_batch_atomic_raises_on_patch_metadata_failure(self):
        """Atomic mode raises ToolError on patch_metadata failure."""
        from app.tools.batch import store_context_batch
        from app.tools.batch import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-atomic-patch-fail',
                'source': 'user',
                'text': 'Original text',
            }],
        )
        context_id = store_result['results'][0]['context_id']

        mock_txn, mock_begin_transaction = _make_mock_txn()

        with patch('app.tools.batch.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=(True, 'user'))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            mock_repos.context.patch_metadata = AsyncMock(return_value=(False, []))

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            with pytest.raises(ToolError, match='Failed to patch metadata'):
                await update_context_batch(
                    updates=[{
                        'context_id': context_id,
                        'metadata_patch': {'key': 'value'},
                    }],
                    atomic=True,
                )

    @pytest.mark.asyncio
    async def test_update_batch_nonatomic_reports_update_entry_failure(self):
        """Non-atomic mode records per-entry failure on update_context_entry failure."""
        from app.tools.batch import store_context_batch
        from app.tools.batch import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-nonatomic-fail',
                'source': 'user',
                'text': 'Original text',
            }],
        )
        context_id = store_result['results'][0]['context_id']

        mock_txn, mock_begin_transaction = _make_mock_txn()

        with patch('app.tools.batch.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=(True, 'user'))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            mock_repos.context.update_context_entry = AsyncMock(return_value=(False, []))

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            result = await update_context_batch(
                updates=[{
                    'context_id': context_id,
                    'text': 'Updated text',
                }],
                atomic=False,
            )
            assert result['failed'] == 1
            failed_entry = result['results'][0]
            assert failed_entry['success'] is False
            assert failed_entry['error'] is not None

    @pytest.mark.asyncio
    async def test_update_batch_nonatomic_reports_patch_metadata_failure(self):
        """Non-atomic mode records per-entry failure on patch_metadata failure."""
        from app.tools.batch import store_context_batch
        from app.tools.batch import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-nonatomic-patch-fail',
                'source': 'user',
                'text': 'Original text',
            }],
        )
        context_id = store_result['results'][0]['context_id']

        mock_txn, mock_begin_transaction = _make_mock_txn()

        with patch('app.tools.batch.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=(True, 'user'))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            mock_repos.context.patch_metadata = AsyncMock(return_value=(False, []))

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            result = await update_context_batch(
                updates=[{
                    'context_id': context_id,
                    'metadata_patch': {'key': 'value'},
                }],
                atomic=False,
            )
            assert result['failed'] == 1
            failed_entry = result['results'][0]
            assert failed_entry['success'] is False
            assert failed_entry['error'] is not None
            assert 'Failed to patch metadata' in str(failed_entry['error'])


# Connection retry
@pytest.mark.usefixtures('initialized_server')
class TestNonAtomicBatchRetry:
    """Connection retry in non-atomic batch operations."""

    @pytest.mark.asyncio
    async def test_store_batch_nonatomic_retries_on_connection_error(self):
        """Non-atomic store retries on ConnectionResetError."""
        from app.tools.batch import store_context_batch

        call_count = 0
        real_txn = MagicMock()
        real_txn.connection = MagicMock()
        real_txn.backend_type = 'sqlite'

        @asynccontextmanager
        async def failing_begin_transaction():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionResetError('Connection lost')
            yield real_txn

        with patch('app.tools.batch.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = failing_begin_transaction
            mock_repos.context.backend = mock_backend

            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
            mock_repos.embeddings.exists = AsyncMock(return_value=False)

            result = await store_context_batch(
                entries=[{
                    'thread_id': 'test-retry',
                    'source': 'user',
                    'text': 'Retry test entry',
                }],
                atomic=False,
            )
            assert result['results'][0]['success'] is True
            assert call_count >= 2

    @pytest.mark.asyncio
    async def test_update_batch_nonatomic_retries_on_connection_error(self):
        """Non-atomic update retries on ConnectionResetError."""
        from app.tools.batch import store_context_batch
        from app.tools.batch import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-retry-update',
                'source': 'user',
                'text': 'Original text',
            }],
        )
        context_id = store_result['results'][0]['context_id']

        call_count = 0
        real_txn = MagicMock()
        real_txn.connection = MagicMock()
        real_txn.backend_type = 'sqlite'

        @asynccontextmanager
        async def failing_begin_transaction():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionResetError('Connection lost')
            yield real_txn

        with patch('app.tools.batch.ensure_repositories') as mock_repos_fn:
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=(True, 'user'))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            mock_repos.context.update_context_entry = AsyncMock(return_value=(True, ['text']))
            mock_repos.images.count_images_for_context = AsyncMock(return_value=0)

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = failing_begin_transaction
            mock_repos.context.backend = mock_backend

            result = await update_context_batch(
                updates=[{
                    'context_id': context_id,
                    'text': 'Updated text',
                }],
                atomic=False,
            )
            assert result['results'][0]['success'] is True
            assert call_count >= 2


# Summary preserved reporting
@pytest.mark.usefixtures('initialized_server')
class TestBatchStoreResponseParity:
    """Batch store response message parity."""

    @pytest.mark.asyncio
    async def test_store_batch_reports_summary_preserved(self):
        """Batch store reports 'summaries preserved' for duplicates with summaries."""
        from app.tools.batch import store_context_batch

        mock_summary = 'Test summary content'
        mock_txn, mock_begin_transaction = _make_mock_txn()

        mock_settings = MagicMock()
        mock_settings.embedding.enabled = False
        mock_settings.semantic_search.enabled = False
        mock_settings.summary.enabled = True
        mock_settings.summary.min_content_length = 0

        with (
            patch('app.tools.batch.settings', mock_settings),
            patch('app.tools.batch.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools.batch.get_summary_provider', return_value=MagicMock()),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            # Simulate dedup pre-check finding existing summary
            mock_repos.context.check_latest_is_duplicate = AsyncMock(return_value=42)
            mock_repos.context.get_summary = AsyncMock(return_value=mock_summary)
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
        from app.tools.batch import store_context_batch

        mock_txn, mock_begin_transaction = _make_mock_txn()

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
            patch('app.tools.batch.settings', mock_settings),
            patch('app.tools.batch.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.get_embedding_provider') as mock_emb_provider_fn,
            patch('app.tools._shared.get_embedding_provider') as mock_shared_emb_provider_fn,
            patch('app.tools.batch.get_summary_provider', return_value=None),
            patch(
                'app.tools._shared._generate_embeddings_for_text',
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
            mock_repos.context.get_summary = AsyncMock(return_value=None)

            # For entry 2 (deduplicated), embeddings already exist
            async def mock_embedding_exists(context_id):
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


# Batch update summaries cleared
@pytest.mark.usefixtures('initialized_server')
class TestBatchUpdateResponseParity:
    """Batch update response includes 'summaries cleared'."""

    @pytest.mark.asyncio
    async def test_update_batch_reports_summaries_cleared(self):
        """Batch update reports 'summaries cleared' when summaries are removed."""
        from app.tools.batch import store_context_batch
        from app.tools.batch import update_context_batch

        store_result = await store_context_batch(
            entries=[{
                'thread_id': 'test-summary-cleared',
                'source': 'user',
                'text': 'A' * 1000,
            }],
        )
        context_id = store_result['results'][0]['context_id']

        mock_txn, mock_begin_transaction = _make_mock_txn()

        mock_settings = MagicMock()
        mock_settings.embedding.enabled = False
        mock_settings.semantic_search.enabled = False
        mock_settings.summary.enabled = True
        mock_settings.summary.min_content_length = 500

        with (
            patch('app.tools.batch.settings', mock_settings),
            patch('app.tools.batch.ensure_repositories') as mock_repos_fn,
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools.batch.get_summary_provider', return_value=MagicMock()),
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos
            mock_repos.context.check_entry_exists = AsyncMock(return_value=(True, 'user'))
            mock_repos.context.get_content_type = AsyncMock(return_value='text')
            mock_repos.context.update_context_entry = AsyncMock(return_value=(True, ['text']))
            mock_repos.images.count_images_for_context = AsyncMock(return_value=0)

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_backend.begin_transaction = mock_begin_transaction
            mock_repos.context.backend = mock_backend

            # Update with short text (below min_content_length) triggers summary clearing
            result = await update_context_batch(
                updates=[{
                    'context_id': context_id,
                    'text': 'Short',
                }],
                atomic=False,
            )
            assert result['results'][0]['success'] is True
            assert 'summaries cleared' in result['message']


# Embedding cleanup for non-ID batch deletes on SQLite
@pytest.mark.usefixtures('initialized_server')
class TestBatchDeleteEmbeddingCleanup:
    """Embedding cleanup for non-ID criteria on SQLite."""

    @pytest.mark.asyncio
    async def test_delete_batch_by_thread_cleans_embeddings_sqlite(self):
        """Verify embeddings.delete is called when deleting by thread_ids on SQLite."""
        from app.tools.batch import delete_context_batch

        mock_settings = MagicMock()
        mock_settings.semantic_search.enabled = True

        with (
            patch('app.tools.batch.settings', mock_settings),
            patch('app.tools.batch.ensure_repositories') as mock_repos_fn,
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_repos.context.backend = mock_backend

            # Pre-query returns affected IDs
            mock_repos.context.get_ids_matching_batch_criteria = AsyncMock(
                return_value=[10, 20, 30],
            )
            mock_repos.embeddings.delete = AsyncMock()
            mock_repos.context.delete_contexts_batch = AsyncMock(
                return_value=(3, ['thread_ids']),
            )

            result = await delete_context_batch(thread_ids=['thread-abc'])

            assert result['success'] is True
            assert result['deleted_count'] == 3

            # Verify embeddings.delete was called for each affected ID
            assert mock_repos.embeddings.delete.call_count == 3
            mock_repos.embeddings.delete.assert_any_call(10)
            mock_repos.embeddings.delete.assert_any_call(20)
            mock_repos.embeddings.delete.assert_any_call(30)

            # Verify get_ids_matching_batch_criteria was called with correct args
            mock_repos.context.get_ids_matching_batch_criteria.assert_called_once_with(
                thread_ids=['thread-abc'],
                source=None,
                older_than_days=None,
            )

    @pytest.mark.asyncio
    async def test_delete_batch_by_older_than_cleans_embeddings_sqlite(self):
        """Verify embeddings.delete is called when deleting by older_than_days on SQLite."""
        from app.tools.batch import delete_context_batch

        mock_settings = MagicMock()
        mock_settings.semantic_search.enabled = True

        with (
            patch('app.tools.batch.settings', mock_settings),
            patch('app.tools.batch.ensure_repositories') as mock_repos_fn,
        ):
            mock_repos = AsyncMock()
            mock_repos_fn.return_value = mock_repos

            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            mock_repos.context.backend = mock_backend

            # Pre-query returns affected IDs
            mock_repos.context.get_ids_matching_batch_criteria = AsyncMock(
                return_value=[5, 15],
            )
            mock_repos.embeddings.delete = AsyncMock()
            mock_repos.context.delete_contexts_batch = AsyncMock(
                return_value=(2, ['older_than_days']),
            )

            result = await delete_context_batch(older_than_days=30)

            assert result['success'] is True
            assert result['deleted_count'] == 2

            # Verify embeddings.delete was called for each affected ID
            assert mock_repos.embeddings.delete.call_count == 2
            mock_repos.embeddings.delete.assert_any_call(5)
            mock_repos.embeddings.delete.assert_any_call(15)

            # Verify get_ids_matching_batch_criteria was called with correct args
            mock_repos.context.get_ids_matching_batch_criteria.assert_called_once_with(
                thread_ids=None,
                source=None,
                older_than_days=30,
            )
