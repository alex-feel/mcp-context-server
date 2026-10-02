"""Generation-first transactional integrity of update_context_batch.

An atomic batch with a failed embedding modifies nothing, a non-atomic batch applies the updates that
succeeded, and a missing embedding provider applies every update.
"""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio

from app.backends.sqlite_backend import SQLiteBackend
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.schemas import load_schema


class TestUpdateContextBatchEmbeddingFirst:
    """Tests for update_context_batch embedding-first pattern."""

    @pytest_asyncio.fixture
    async def setup_with_existing_entries(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer, list[str]], None]:
        """Set up backend with existing entries for batch update tests."""
        db_path = tmp_path / 'test_batch_update.db'

        # Create schema
        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            # Insert existing entries
            conn.execute(
                '''INSERT INTO context_entries
                   (id, thread_id, source, text_content, content_type, owner_id)
                   VALUES (?, ?, ?, ?, ?, 'local')''',
                (generate_id(), 'batch-update-test', 'agent', 'Original 1', 'text'),
            )
            conn.execute(
                '''INSERT INTO context_entries
                   (id, thread_id, source, text_content, content_type, owner_id)
                   VALUES (?, ?, ?, ?, ?, 'local')''',
                (generate_id(), 'batch-update-test', 'agent', 'Original 2', 'text'),
            )
            conn.execute(
                '''INSERT INTO context_entries
                   (id, thread_id, source, text_content, content_type, owner_id)
                   VALUES (?, ?, ?, ?, ?, 'local')''',
                (generate_id(), 'batch-update-test', 'agent', 'Original 3', 'text'),
            )
            conn.commit()

        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()

        repos = RepositoryContainer(backend)

        # Get the existing entry IDs
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT id FROM context_entries WHERE thread_id = ? ORDER BY id',
                ('batch-update-test',),
            )
            entry_ids = [row[0] for row in cursor.fetchall()]

        yield backend, repos, entry_ids

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_batch_update_atomic_embedding_failure_preserves_all(
        self, setup_with_existing_entries: tuple[SQLiteBackend, RepositoryContainer, list[str]],
    ) -> None:
        """Test that atomic batch update preserves all data when embedding fails."""
        backend, repos, entry_ids = setup_with_existing_entries

        # Create a mock embedding provider that fails on second entry
        call_count = 0

        async def mock_embed_query(_text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception('Embedding failed on second entry')
            return [0.1] * 1024

        mock_provider = MagicMock()
        mock_provider.embed_query = AsyncMock(side_effect=mock_embed_query)

        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        with (
            patch('app.tools.batch.update.ensure_repositories', return_value=repos),
            patch('app.tools.batch.update.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
        ):
            from fastmcp.exceptions import ToolError

            from app.tools.batch.update import update_context_batch

            updates = [
                {'context_id': entry_ids[0], 'text': 'Updated 1'},
                {'context_id': entry_ids[1], 'text': 'Updated 2 - will fail'},
                {'context_id': entry_ids[2], 'text': 'Updated 3'},
            ]

            # Attempt atomic batch update - should fail completely
            with pytest.raises(ToolError, match='Generation failed'):
                await update_context_batch(updates=updates, atomic=True)

        # Verify ALL entries retain original content (atomic rollback)
        async with backend.get_connection(readonly=True) as conn:
            for idx, entry_id in enumerate(entry_ids, 1):
                cursor = conn.execute(
                    'SELECT text_content FROM context_entries WHERE id = ?',
                    (entry_id,),
                )
                row = cursor.fetchone()
                assert row is not None
                assert row[0] == f'Original {idx}', f'Entry {idx} should retain original content'

    @pytest.mark.asyncio
    async def test_batch_update_non_atomic_partial_success(
        self, setup_with_existing_entries: tuple[SQLiteBackend, RepositoryContainer, list[str]],
    ) -> None:
        """Test that non-atomic batch update allows partial success."""
        backend, repos, entry_ids = setup_with_existing_entries

        # Create a mock embedding provider that fails on second entry
        call_count = 0

        async def mock_embed_query(_text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception('Embedding failed on second entry')
            return [0.1] * 1024

        mock_provider = MagicMock()
        mock_provider.embed_query = AsyncMock(side_effect=mock_embed_query)

        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        with (
            patch('app.tools.batch.update.ensure_repositories', return_value=repos),
            patch('app.tools.batch.update.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
            # Mock embedding repository to avoid vec_context_embeddings table issues
            patch.object(repos.embeddings, 'store_chunked', new=AsyncMock(return_value=None)),
            patch.object(repos.embeddings, 'delete_all_chunks', new=AsyncMock(return_value=None)),
        ):
            from app.tools.batch.update import update_context_batch

            updates = [
                {'context_id': entry_ids[0], 'text': 'Updated 1'},
                {'context_id': entry_ids[1], 'text': 'Updated 2 - fail'},
                {'context_id': entry_ids[2], 'text': 'Updated 3'},
            ]

            # Non-atomic batch update - should allow partial success
            result = await update_context_batch(updates=updates, atomic=False)

            # Entry 2 failed embedding
            assert result['succeeded'] == 2
            assert result['failed'] == 1

        # Verify partial updates
        async with backend.get_connection(readonly=True) as conn:
            # Entry 1 should be updated
            cursor = conn.execute(
                'SELECT text_content FROM context_entries WHERE id = ?',
                (entry_ids[0],),
            )
            assert cursor.fetchone()[0] == 'Updated 1'

            # Entry 2 should retain original (embedding failed)
            cursor = conn.execute(
                'SELECT text_content FROM context_entries WHERE id = ?',
                (entry_ids[1],),
            )
            assert cursor.fetchone()[0] == 'Original 2', 'Failed entry should retain original'

            # Entry 3 should be updated
            cursor = conn.execute(
                'SELECT text_content FROM context_entries WHERE id = ?',
                (entry_ids[2],),
            )
            assert cursor.fetchone()[0] == 'Updated 3'

    @pytest.mark.asyncio
    async def test_batch_update_no_embedding_provider_saves_all_updates(
        self, setup_with_existing_entries: tuple[SQLiteBackend, RepositoryContainer, list[str]],
    ) -> None:
        """Test that update_context_batch saves all updates when embedding provider is disabled."""
        backend, repos, entry_ids = setup_with_existing_entries

        with (
            patch('app.tools.batch.update.ensure_repositories', return_value=repos),
            patch('app.tools.batch.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
        ):
            from app.tools.batch.update import update_context_batch

            updates = [
                {'context_id': entry_ids[0], 'text': 'Updated 1 without embedding'},
                {'context_id': entry_ids[1], 'text': 'Updated 2 without embedding'},
                {'context_id': entry_ids[2], 'text': 'Updated 3 without embedding'},
            ]

            # Update batch without embedding provider - should succeed
            result = await update_context_batch(updates=updates, atomic=True)

            assert result['success'] is True
            assert result['succeeded'] == 3
            assert result['failed'] == 0

            # Verify all items succeeded without embedding in updated_fields
            for item in result['results']:
                assert item['success'] is True
                assert item['error'] is None
                # Embedding should NOT be in updated_fields since provider is None
                if item['updated_fields'] is not None:
                    assert 'embedding' not in item['updated_fields']

        # Verify all entries were updated
        async with backend.get_connection(readonly=True) as conn:
            for idx, entry_id in enumerate(entry_ids, 1):
                cursor = conn.execute(
                    'SELECT text_content FROM context_entries WHERE id = ?',
                    (entry_id,),
                )
                row = cursor.fetchone()
                assert row is not None
                assert row[0] == f'Updated {idx} without embedding', f'Entry {idx} should be updated'
