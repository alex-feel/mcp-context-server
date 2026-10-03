"""Generation-first transactional integrity of store_context_batch.

An atomic batch with a failed embedding stores nothing, a non-atomic batch stores the entries that succeeded,
and a missing embedding provider stores every entry.
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
from app.repositories import RepositoryContainer
from app.schemas import load_schema


class TestStoreContextBatchEmbeddingFirst:
    """Tests for store_context_batch embedding-first pattern."""

    @pytest_asyncio.fixture
    async def setup_backend_and_repos(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer], None]:
        """Set up backend and repositories for testing."""
        db_path = tmp_path / 'test_batch_store.db'

        # Create schema
        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()

        repos = RepositoryContainer(backend)

        yield backend, repos

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_batch_store_atomic_embedding_failure_no_entries_saved(
        self, setup_backend_and_repos: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Test that atomic batch store saves nothing when embedding fails."""
        backend, repos = setup_backend_and_repos

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
        mock_provider.embed_documents = AsyncMock(side_effect=Exception('Should use embed_query'))

        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        with (
            patch('app.tools.batch.store.ensure_repositories', return_value=repos),
            patch('app.tools.batch.store.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
        ):
            from fastmcp.exceptions import ToolError

            from app.tools.batch.store import store_context_batch

            entries = [
                {'thread_id': 'batch-test', 'source': 'agent', 'text': 'Entry 1'},
                {'thread_id': 'batch-test', 'source': 'agent', 'text': 'Entry 2 - will fail'},
                {'thread_id': 'batch-test', 'source': 'agent', 'text': 'Entry 3'},
            ]

            # Attempt atomic batch store - should fail completely
            with pytest.raises(ToolError, match='Generation failed'):
                await store_context_batch(entries=entries, atomic=True)

        # Verify NO entries were saved (atomic rollback)
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                ('batch-test',),
            )
            count = cursor.fetchone()[0]
            assert count == 0, 'No entries should be saved when atomic batch embedding fails'

    @pytest.mark.asyncio
    async def test_batch_store_non_atomic_partial_success(
        self, setup_backend_and_repos: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Test that non-atomic batch store allows partial success."""
        backend, repos = setup_backend_and_repos

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
            patch('app.tools.batch.store.ensure_repositories', return_value=repos),
            patch('app.tools.batch.store.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
            # Mock embedding repository to avoid vec_context_embeddings table issues
            patch.object(repos.embeddings, 'store_chunked', new=AsyncMock(return_value=None)),
        ):
            from app.tools.batch.store import store_context_batch

            entries = [
                {'thread_id': 'partial-test', 'source': 'agent', 'text': 'Entry 1 - success'},
                {'thread_id': 'partial-test', 'source': 'agent', 'text': 'Entry 2 - fail'},
                {'thread_id': 'partial-test', 'source': 'agent', 'text': 'Entry 3 - success'},
            ]

            # Non-atomic batch store - should allow partial success
            result = await store_context_batch(entries=entries, atomic=False)

            # Entry 2 failed embedding, so only 2 succeeded
            assert result['succeeded'] == 2
            assert result['failed'] == 1

        # Verify 2 entries were saved
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                ('partial-test',),
            )
            count = cursor.fetchone()[0]
            assert count == 2, 'Only successful entries should be saved in non-atomic mode'

    @pytest.mark.asyncio
    async def test_batch_store_no_embedding_provider_saves_all_entries(
        self, setup_backend_and_repos: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Test that store_context_batch saves all entries when embedding provider is disabled."""
        backend, repos = setup_backend_and_repos

        with (
            patch('app.tools.batch.store.ensure_repositories', return_value=repos),
            patch('app.tools.batch.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
        ):
            from app.tools.batch.store import store_context_batch

            entries = [
                {'thread_id': 'no-embed-batch', 'source': 'agent', 'text': 'Entry 1 without embedding'},
                {'thread_id': 'no-embed-batch', 'source': 'agent', 'text': 'Entry 2 without embedding'},
                {'thread_id': 'no-embed-batch', 'source': 'agent', 'text': 'Entry 3 without embedding'},
            ]

            # Store batch without embedding provider - should succeed
            result = await store_context_batch(entries=entries, atomic=True)

            assert result['success'] is True
            assert result['succeeded'] == 3
            assert result['failed'] == 0

            # Verify all items succeeded
            for item in result['results']:
                assert item['success'] is True
                assert item['context_id'] is not None
                assert item['error'] is None

        # Verify all entries were saved
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                ('no-embed-batch',),
            )
            count = cursor.fetchone()[0]
            assert count == 3, 'All 3 entries should be saved when embedding is disabled'
