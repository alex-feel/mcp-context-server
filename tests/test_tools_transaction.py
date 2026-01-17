"""Tests for tool-level transactional integrity (Embedding-First Pattern).

This module tests the embedding-first pattern in MCP tools:
- store_context: embedding failure = no data saved
- update_context: embedding failure = original data preserved
- store_context_batch: atomic mode embedding failure = no entries saved
- update_context_batch: atomic mode embedding failure = no entries modified

Phase 3 of the Transactional Integrity Fix:
- Tool Layer Refactoring with Embedding-First Pattern
"""

from __future__ import annotations

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
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
        ):
            from fastmcp.exceptions import ToolError

            from app.tools.context import store_context

            # Attempt to store context - should fail due to embedding error
            with pytest.raises(ToolError, match='Embedding generation failed'):
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
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=None),
        ):
            from app.tools.context import store_context

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


class TestUpdateContextEmbeddingFirst:
    """Tests for update_context embedding-first pattern."""

    @pytest_asyncio.fixture
    async def setup_with_existing_entry(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer, int], None]:
        """Set up backend with an existing entry for update tests."""
        db_path = tmp_path / 'test_update.db'

        # Create schema
        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            # Insert an existing entry
            conn.execute(
                '''INSERT INTO context_entries
                   (thread_id, source, text_content, content_type, metadata)
                   VALUES (?, ?, ?, ?, ?)''',
                ('existing-thread', 'agent', 'Original content', 'text', '{"status": "original"}'),
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
        self, setup_with_existing_entry: tuple[SQLiteBackend, RepositoryContainer, int],
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
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
        ):
            from fastmcp.exceptions import ToolError

            from app.tools.context import update_context

            # Attempt to update context - should fail due to embedding error
            with pytest.raises(ToolError, match='Embedding generation failed'):
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
        self, setup_with_existing_entry: tuple[SQLiteBackend, RepositoryContainer, int],
    ) -> None:
        """Test that metadata-only updates work without embedding generation."""
        backend, repos, entry_id = setup_with_existing_entry

        # Mock provider that would fail if called (but shouldn't be called)
        mock_provider = MagicMock()
        mock_provider.embed_query = AsyncMock(side_effect=Exception('Should not be called'))

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_provider),
        ):
            from app.tools.context import update_context

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
        self, setup_with_existing_entry: tuple[SQLiteBackend, RepositoryContainer, int],
    ) -> None:
        """Test that update_context saves data when embedding provider is disabled."""
        backend, repos, entry_id = setup_with_existing_entry

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=None),
        ):
            from app.tools.context import update_context

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
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=mock_provider),
            patch('app.startup.get_chunking_service', return_value=mock_chunking),
        ):
            from fastmcp.exceptions import ToolError

            from app.tools.batch import store_context_batch

            entries = [
                {'thread_id': 'batch-test', 'source': 'agent', 'text': 'Entry 1'},
                {'thread_id': 'batch-test', 'source': 'agent', 'text': 'Entry 2 - will fail'},
                {'thread_id': 'batch-test', 'source': 'agent', 'text': 'Entry 3'},
            ]

            # Attempt atomic batch store - should fail completely
            with pytest.raises(ToolError, match='Embedding generation failed'):
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
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=mock_provider),
            patch('app.startup.get_chunking_service', return_value=mock_chunking),
            # Mock embedding repository to avoid vec_context_embeddings table issues
            patch.object(repos.embeddings, 'store', new=AsyncMock(return_value=None)),
            patch.object(repos.embeddings, 'store_chunked', new=AsyncMock(return_value=None)),
        ):
            from app.tools.batch import store_context_batch

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
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
        ):
            from app.tools.batch import store_context_batch

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


class TestUpdateContextBatchEmbeddingFirst:
    """Tests for update_context_batch embedding-first pattern."""

    @pytest_asyncio.fixture
    async def setup_with_existing_entries(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer, list[int]], None]:
        """Set up backend with existing entries for batch update tests."""
        db_path = tmp_path / 'test_batch_update.db'

        # Create schema
        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            # Insert existing entries
            conn.execute(
                '''INSERT INTO context_entries
                   (thread_id, source, text_content, content_type)
                   VALUES (?, ?, ?, ?)''',
                ('batch-update-test', 'agent', 'Original 1', 'text'),
            )
            conn.execute(
                '''INSERT INTO context_entries
                   (thread_id, source, text_content, content_type)
                   VALUES (?, ?, ?, ?)''',
                ('batch-update-test', 'agent', 'Original 2', 'text'),
            )
            conn.execute(
                '''INSERT INTO context_entries
                   (thread_id, source, text_content, content_type)
                   VALUES (?, ?, ?, ?)''',
                ('batch-update-test', 'agent', 'Original 3', 'text'),
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
        self, setup_with_existing_entries: tuple[SQLiteBackend, RepositoryContainer, list[int]],
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
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=mock_provider),
            patch('app.startup.get_chunking_service', return_value=mock_chunking),
        ):
            from fastmcp.exceptions import ToolError

            from app.tools.batch import update_context_batch

            updates = [
                {'context_id': entry_ids[0], 'text': 'Updated 1'},
                {'context_id': entry_ids[1], 'text': 'Updated 2 - will fail'},
                {'context_id': entry_ids[2], 'text': 'Updated 3'},
            ]

            # Attempt atomic batch update - should fail completely
            with pytest.raises(ToolError, match='Embedding generation failed'):
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
        self, setup_with_existing_entries: tuple[SQLiteBackend, RepositoryContainer, list[int]],
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
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=mock_provider),
            patch('app.startup.get_chunking_service', return_value=mock_chunking),
            # Mock embedding repository to avoid vec_context_embeddings table issues
            patch.object(repos.embeddings, 'store', new=AsyncMock(return_value=None)),
            patch.object(repos.embeddings, 'store_chunked', new=AsyncMock(return_value=None)),
            patch.object(repos.embeddings, 'delete_all_chunks', new=AsyncMock(return_value=None)),
        ):
            from app.tools.batch import update_context_batch

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
        self, setup_with_existing_entries: tuple[SQLiteBackend, RepositoryContainer, list[int]],
    ) -> None:
        """Test that update_context_batch saves all updates when embedding provider is disabled."""
        backend, repos, entry_ids = setup_with_existing_entries

        with (
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
        ):
            from app.tools.batch import update_context_batch

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
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=None),
        ):
            from app.tools.context import store_context

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
