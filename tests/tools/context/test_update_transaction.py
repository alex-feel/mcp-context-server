"""Generation-first transactional integrity of update_context.

A failed embedding leaves the original entry unchanged; metadata-only updates and a missing embedding provider
save without embeddings.
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


class TestUpdateContextEmbeddingFirst:
    """Tests for update_context embedding-first pattern."""

    @pytest_asyncio.fixture
    async def setup_with_existing_entry(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer, str], None]:
        """Set up backend with an existing entry for update tests."""
        db_path = tmp_path / 'test_update.db'

        # Create schema
        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            # Insert an existing entry
            existing_id = generate_id()
            conn.execute(
                '''INSERT INTO context_entries
                   (id, thread_id, source, text_content, content_type, metadata, owner_id)
                   VALUES (?, ?, ?, ?, ?, ?, 'local')''',
                (existing_id, 'existing-thread', 'agent', 'Original content', 'text', '{"status": "original"}'),
            )
            conn.commit()

        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()

        repos = RepositoryContainer(backend)

        # Get the existing entry ID
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT id FROM context_entries WHERE thread_id = ?',
                ('existing-thread',),
            )
            entry_id = cursor.fetchone()[0]

        yield backend, repos, entry_id

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_update_context_embedding_failure_preserves_original(
        self, setup_with_existing_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """Test that update_context preserves original data when embedding fails."""
        backend, repos, entry_id = setup_with_existing_entry

        # Create a mock embedding provider that fails
        mock_provider = MagicMock()
        mock_provider.embed_query = AsyncMock(side_effect=Exception('Embedding service unavailable'))
        mock_provider.embed_documents = AsyncMock(side_effect=Exception('Embedding service unavailable'))

        # Mock the chunking service as disabled
        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        with (
            patch('app.tools.context.update.ensure_repositories', return_value=repos),
            patch('app.tools.context.update.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
        ):
            from fastmcp.exceptions import ToolError

            from app.tools.context.update import update_context

            # Attempt to update context - should fail due to embedding error
            with pytest.raises(ToolError, match='Generation failed'):
                await update_context(
                    context_id=entry_id,
                    text='Updated content that should not be saved',
                )

        # Verify original data is preserved
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT text_content FROM context_entries WHERE id = ?',
                (entry_id,),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == 'Original content', 'Original content should be preserved when embedding fails'

    @pytest.mark.asyncio
    async def test_update_context_metadata_only_no_embedding_required(
        self, setup_with_existing_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """Test that metadata-only updates work without embedding generation."""
        backend, repos, entry_id = setup_with_existing_entry

        # Mock provider that would fail if called (but shouldn't be called)
        mock_provider = MagicMock()
        mock_provider.embed_query = AsyncMock(side_effect=Exception('Should not be called'))

        with (
            patch('app.tools.context.update.ensure_repositories', return_value=repos),
            patch('app.tools.context.update.get_embedding_provider', return_value=mock_provider),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_provider),
        ):
            from app.tools.context.update import update_context

            # Update only metadata - should not trigger embedding generation
            result = await update_context(
                context_id=entry_id,
                metadata={'status': 'updated'},
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']

        # Verify metadata was updated but text unchanged
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT text_content, metadata FROM context_entries WHERE id = ?',
                (entry_id,),
            )
            row = cursor.fetchone()
            assert row[0] == 'Original content', 'Text should remain unchanged'
            assert 'updated' in row[1], 'Metadata should be updated'

    @pytest.mark.asyncio
    async def test_update_context_no_embedding_provider_saves_data(
        self, setup_with_existing_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """Test that update_context saves data when embedding provider is disabled."""
        backend, repos, entry_id = setup_with_existing_entry

        with (
            patch('app.tools.context.update.ensure_repositories', return_value=repos),
            patch('app.tools.context.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
        ):
            from app.tools.context.update import update_context

            # Update context without embedding provider - should succeed
            result = await update_context(
                context_id=entry_id,
                text='Updated content without embedding',
                metadata={'status': 'updated'},
            )

            assert result['success'] is True
            assert 'text_content' in result['updated_fields']
            assert 'metadata' in result['updated_fields']
            # Embedding should NOT be in updated_fields since provider is None
            assert 'embedding' not in result['updated_fields']

        # Verify data was updated
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT text_content, metadata FROM context_entries WHERE id = ?',
                (entry_id,),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == 'Updated content without embedding', 'Text should be updated'
            assert 'updated' in row[1], 'Metadata should be updated'
