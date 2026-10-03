"""Generation-first transactional integrity of store_context.

A failed embedding stores nothing, a missing embedding provider still stores the entry, and the entry and its
tags commit in one transaction.
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


class TestStoreContextEmbeddingFirst:
    """Tests for store_context embedding-first pattern."""

    @pytest_asyncio.fixture
    async def setup_backend_and_repos(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer], None]:
        """Set up backend and repositories for testing."""
        db_path = tmp_path / 'test_store.db'

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
    async def test_store_context_embedding_failure_no_data_saved(
        self, setup_backend_and_repos: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Test that store_context saves no data when embedding generation fails."""
        backend, repos = setup_backend_and_repos

        # Create a mock embedding provider that fails
        mock_provider = MagicMock()
        mock_provider.embed_query = AsyncMock(side_effect=Exception('Embedding service unavailable'))
        mock_provider.embed_documents = AsyncMock(side_effect=Exception('Embedding service unavailable'))

        # Mock the chunking service as disabled
        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        with (
            patch('app.tools.context.store.ensure_repositories', return_value=repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
        ):
            from fastmcp.exceptions import ToolError

            from app.tools.context.store import store_context

            # Attempt to store context - should fail due to embedding error
            with pytest.raises(ToolError, match='Generation failed'):
                await store_context(
                    thread_id='test-thread',
                    source='agent',
                    text='Test content that should not be saved',
                )

        # Verify no data was saved
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                ('test-thread',),
            )
            count = cursor.fetchone()[0]
            assert count == 0, 'No context entries should be saved when embedding fails'

    @pytest.mark.asyncio
    async def test_store_context_no_embedding_provider_saves_data(
        self, setup_backend_and_repos: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Test that store_context saves data when embedding provider is disabled."""
        backend, repos = setup_backend_and_repos

        with (
            patch('app.tools.context.store.ensure_repositories', return_value=repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
        ):
            from app.tools.context.store import store_context

            # Store context without embedding provider
            result = await store_context(
                thread_id='test-no-embed',
                source='agent',
                text='Test content without embedding',
            )

            assert result['success'] is True
            assert result['context_id'] is not None

        # Verify data was saved
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                ('test-no-embed',),
            )
            count = cursor.fetchone()[0]
            assert count == 1, 'Context should be saved when embedding is disabled'


class TestTransactionAtomicityIntegration:
    """Integration tests for transaction atomicity across multiple operations."""

    @pytest_asyncio.fixture
    async def setup_backend_and_repos(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer], None]:
        """Set up backend and repositories for testing."""
        db_path = tmp_path / 'test_atomicity.db'

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
    async def test_store_context_all_operations_in_single_transaction(
        self, setup_backend_and_repos: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Test that store_context commits all operations atomically.

        This test verifies context + tags atomicity without embeddings.
        Embedding atomicity is tested separately in TestStoreContextEmbeddingFirst.
        """
        backend, repos = setup_backend_and_repos

        # Use None for embedding provider to test context + tags atomicity
        # (embedding storage is tested separately with proper mocks)
        with (
            patch('app.tools.context.store.ensure_repositories', return_value=repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
        ):
            from app.tools.context.store import store_context

            result = await store_context(
                thread_id='atomic-test',
                source='agent',
                text='Test content with tags',
                tags=['tag1', 'tag2'],
                metadata={'key': 'value'},
            )

            assert result['success'] is True
            context_id = result['context_id']

        # Verify all data was committed together
        async with backend.get_connection(readonly=True) as conn:
            # Check context entry
            cursor = conn.execute(
                'SELECT text_content, metadata FROM context_entries WHERE id = ?',
                (context_id,),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == 'Test content with tags'
            assert 'value' in row[1]

            # Check tags (committed in same transaction as context entry)
            cursor = conn.execute(
                'SELECT tag FROM tags WHERE context_entry_id = ? ORDER BY tag',
                (context_id,),
            )
            tags = [r[0] for r in cursor.fetchall()]
            assert tags == ['tag1', 'tag2']
