"""Tests for embedding deduplication race condition fix.

Regression tests for the constraint violation error that occurred when
store_context was called with identical content (triggering deduplication)
while embeddings already existed for that entry.

Root cause: was_updated flag from store_with_deduplication() was ignored
when storing embeddings, causing duplicate INSERT attempts.

Fix: Skip embedding storage when entry was deduplicated AND embeddings
already exist. UPSERT defense-in-depth in store_chunked().
"""

from __future__ import annotations

import importlib.util
import sqlite3
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from app.backends import StorageBackend

# Conditional skip marker for tests requiring sqlite-vec package
requires_sqlite_vec = pytest.mark.skipif(
    importlib.util.find_spec('sqlite_vec') is None,
    reason='sqlite-vec package not installed',
)


def _make_mock_embedding_provider(dim: int) -> MagicMock:
    """Create a mock embedding provider returning vectors of the given dimension."""
    provider = MagicMock()
    provider.embed_query = AsyncMock(return_value=[0.1] * dim)
    provider.embed_documents = AsyncMock(return_value=[[0.1] * dim])
    return provider


def _make_mock_chunking_service(*, enabled: bool = False) -> MagicMock:
    """Create a mock chunking service."""
    service = MagicMock()
    service.is_enabled = enabled
    return service


@pytest.mark.asyncio
class TestStoreContextEmbeddingDeduplication:
    """Tests for store_context embedding deduplication fix."""

    @requires_sqlite_vec
    async def test_duplicate_store_with_existing_embeddings_no_error(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Storing duplicate content with existing embeddings does not raise error.

        This is the primary regression test. Scenario:
        1. Store context entry with embeddings (first call)
        2. Store identical context entry (triggers deduplication, was_updated=True)
        3. Should skip embedding storage, NOT attempt duplicate INSERT
        """
        from app.repositories import RepositoryContainer

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        mock_provider = _make_mock_embedding_provider(embedding_dim)
        mock_chunking = _make_mock_chunking_service()

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
        ):
            from app.tools.context import store_context

            # First store - should succeed and create embeddings
            result1 = await store_context(
                thread_id='test-dedup-race',
                source='agent',
                text='Test content for deduplication race condition',
            )

            assert result1['success'] is True
            context_id = result1['context_id']

            # Verify embeddings were created
            embedding_exists = await repos.embeddings.exists(context_id)
            assert embedding_exists is True, 'Embeddings should exist after first store'

            # Second store - identical content triggers deduplication
            # Without the fix this would raise a constraint violation
            result2 = await store_context(
                thread_id='test-dedup-race',
                source='agent',
                text='Test content for deduplication race condition',
            )

            assert result2['success'] is True
            assert result2['context_id'] == context_id, 'Should return same context_id (deduplication)'

    @requires_sqlite_vec
    async def test_duplicate_store_without_embeddings_creates_embeddings(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Deduplication with missing embeddings still creates embeddings.

        Edge case: Entry exists (deduplication triggers) but embeddings are missing.
        This can happen if entry was created before embeddings were enabled.
        Expected: should_store_embedding = True (because embedding doesn't exist).
        """
        from app.repositories import RepositoryContainer

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)

        # Pre-create entry WITHOUT embeddings via direct repository call
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-no-embed',
            source='agent',
            content_type='text',
            text_content='Content without embedding',
            metadata=None,
        )

        # Verify no embeddings exist
        embedding_exists = await repos.embeddings.exists(context_id)
        assert embedding_exists is False, 'Pre-condition: no embeddings should exist'

        mock_provider = _make_mock_embedding_provider(embedding_dim)
        mock_chunking = _make_mock_chunking_service()

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
        ):
            from app.tools.context import store_context

            # Store identical content - should deduplicate AND create embeddings
            result = await store_context(
                thread_id='test-no-embed',
                source='agent',
                text='Content without embedding',
            )

            assert result['success'] is True
            assert result['context_id'] == context_id, 'Should return same context_id'

            # Verify embeddings were created (even though entry was deduplicated)
            embedding_exists = await repos.embeddings.exists(context_id)
            assert embedding_exists is True, 'Embeddings should be created for dedup entry without embeddings'

    @requires_sqlite_vec
    async def test_new_entry_always_creates_embeddings(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """New entries (not deduplicated) always create embeddings.

        Baseline test: Verify normal behavior is not affected by the fix.
        """
        from app.repositories import RepositoryContainer

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        mock_provider = _make_mock_embedding_provider(embedding_dim)
        mock_chunking = _make_mock_chunking_service()

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
        ):
            from app.tools.context import store_context

            result = await store_context(
                thread_id='test-new-entry',
                source='agent',
                text='Completely new content',
            )

            assert result['success'] is True
            context_id = result['context_id']

            embedding_exists = await repos.embeddings.exists(context_id)
            assert embedding_exists is True, 'New entries should always have embeddings created'


@pytest.mark.asyncio
class TestBatchStoreEmbeddingDeduplication:
    """Tests for store_context_batch embedding deduplication fix."""

    @requires_sqlite_vec
    async def test_batch_store_atomic_duplicate_with_embeddings_no_error(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Atomic batch store with duplicate content does not raise embedding constraint error.

        Calls batch store twice with identical content. The second call triggers
        deduplication (was_updated=True) and should NOT attempt to insert duplicate
        embeddings.

        This is the primary regression test for atomic batch mode.
        """
        from app.repositories import RepositoryContainer

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)
        mock_provider = _make_mock_embedding_provider(embedding_dim)
        mock_chunking = _make_mock_chunking_service()

        with (
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=mock_provider),
            patch('app.startup.get_chunking_service', return_value=mock_chunking),
        ):
            from app.tools.batch import store_context_batch

            entries = [
                {'thread_id': 'batch-atomic-dedup', 'source': 'agent', 'text': 'Atomic entry A'},
                {'thread_id': 'batch-atomic-dedup', 'source': 'agent', 'text': 'Atomic entry B'},
            ]

            # First batch store - creates entries with embeddings
            result1 = await store_context_batch(entries=entries, atomic=True)
            assert result1['success'] is True
            assert result1['succeeded'] == 2

            context_id_a = result1['results'][0]['context_id']
            context_id_b = result1['results'][1]['context_id']
            assert context_id_a is not None
            assert context_id_b is not None

            # Verify embeddings were created
            assert await repos.embeddings.exists(context_id_a) is True
            assert await repos.embeddings.exists(context_id_b) is True

            # Second batch store - identical content
            # Without the fix this would raise a constraint violation on embeddings
            # The key assertion is that this succeeds, not that deduplication works
            result2 = await store_context_batch(entries=entries, atomic=True)
            assert result2['success'] is True
            assert result2['succeeded'] == 2

            # Verify embeddings still exist and operation completed successfully
            # Note: We don't assert on context_id matching as deduplication behavior
            # depends on the repository implementation; we only verify no constraint error
            result_ids = [r['context_id'] for r in result2['results']]
            for cid in result_ids:
                assert cid is not None
                assert await repos.embeddings.exists(cid) is True

    @requires_sqlite_vec
    async def test_batch_store_non_atomic_duplicate_with_embeddings_no_error(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """Non-atomic batch store with pre-existing entry does not raise embedding constraint error.

        Pre-creates entry with embeddings via repository, then calls batch store
        with the same content to trigger deduplication and test the fix.
        """
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import ChunkEmbedding

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)

        # Pre-create entry WITH embeddings via direct repository calls
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='batch-non-atomic',
            source='agent',
            content_type='text',
            text_content='Non-atomic entry',
            metadata=None,
        )
        chunk_emb = ChunkEmbedding(embedding=[0.1] * embedding_dim, start_index=0, end_index=16)
        await repos.embeddings.store_chunked(context_id, [chunk_emb], 'test-model')

        # Verify embedding exists
        assert await repos.embeddings.exists(context_id) is True

        mock_provider = _make_mock_embedding_provider(embedding_dim)
        mock_chunking = _make_mock_chunking_service()

        with (
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=mock_provider),
            patch('app.startup.get_chunking_service', return_value=mock_chunking),
        ):
            from app.tools.batch import store_context_batch

            # Batch store with same content - should trigger deduplication
            # Without the fix this would raise a constraint violation
            entries = [
                {'thread_id': 'batch-non-atomic', 'source': 'agent', 'text': 'Non-atomic entry'},
            ]

            result = await store_context_batch(entries=entries, atomic=False)
            assert result['success'] is True
            assert result['results'][0]['context_id'] == context_id


@pytest.mark.asyncio
class TestEmbeddingRepositoryUpsert:
    """Tests for EmbeddingRepository.store_chunked() UPSERT functionality."""

    @requires_sqlite_vec
    async def test_store_chunked_upsert_false_raises_on_duplicate(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """upsert=False (default) raises error on duplicate context_id.

        Verifies the default behavior is unchanged.
        """
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import ChunkEmbedding

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)

        # Create context entry
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-upsert',
            source='agent',
            content_type='text',
            text_content='Test content',
            metadata=None,
        )

        # First store - should succeed
        chunk_emb = ChunkEmbedding(embedding=[0.1] * embedding_dim, start_index=0, end_index=12)
        await repos.embeddings.store_chunked(context_id, [chunk_emb], 'test-model')

        # Second store with upsert=False (default) - should raise IntegrityError
        with pytest.raises(sqlite3.IntegrityError, match='UNIQUE constraint failed'):
            await repos.embeddings.store_chunked(context_id, [chunk_emb], 'test-model', upsert=False)

    @requires_sqlite_vec
    async def test_store_chunked_upsert_true_succeeds_on_duplicate(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """upsert=True handles duplicate context_id gracefully.

        Defense-in-depth test - verifies UPSERT works via delete-then-insert.
        """
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import ChunkEmbedding

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-upsert',
            source='agent',
            content_type='text',
            text_content='Test content',
            metadata=None,
        )

        # First store
        chunk_emb1 = ChunkEmbedding(embedding=[0.1] * embedding_dim, start_index=0, end_index=12)
        await repos.embeddings.store_chunked(context_id, [chunk_emb1], 'model-v1')

        assert await repos.embeddings.exists(context_id) is True

        # Second store with upsert=True - should succeed (delete-then-insert)
        chunk_emb2 = ChunkEmbedding(embedding=[0.2] * embedding_dim, start_index=0, end_index=12)
        await repos.embeddings.store_chunked(context_id, [chunk_emb2], 'model-v2', upsert=True)

        assert await repos.embeddings.exists(context_id) is True

    @requires_sqlite_vec
    async def test_store_chunked_upsert_updates_model_and_dimensions(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """upsert=True correctly updates metadata (model, dimensions)."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import ChunkEmbedding

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-upsert-meta',
            source='agent',
            content_type='text',
            text_content='Test content',
            metadata=None,
        )

        # First store with model-v1
        chunk_emb1 = ChunkEmbedding(embedding=[0.1] * embedding_dim, start_index=0, end_index=12)
        await repos.embeddings.store_chunked(context_id, [chunk_emb1], 'model-v1')

        # Check initial metadata
        def _check_model_v1(conn: sqlite3.Connection) -> str:
            cursor = conn.execute(
                'SELECT model_name FROM embedding_metadata WHERE context_id = ?',
                (context_id,),
            )
            row = cursor.fetchone()
            return row[0]

        model_name = await backend.execute_read(_check_model_v1)
        assert model_name == 'model-v1'

        # UPSERT with model-v2
        chunk_emb2 = ChunkEmbedding(embedding=[0.2] * embedding_dim, start_index=0, end_index=12)
        await repos.embeddings.store_chunked(context_id, [chunk_emb2], 'model-v2', upsert=True)

        # Check updated metadata
        def _check_model_v2(conn: sqlite3.Connection) -> str:
            cursor = conn.execute(
                'SELECT model_name FROM embedding_metadata WHERE context_id = ?',
                (context_id,),
            )
            row = cursor.fetchone()
            return row[0]

        model_name = await backend.execute_read(_check_model_v2)
        assert model_name == 'model-v2'

    @requires_sqlite_vec
    async def test_store_chunked_upsert_handles_chunk_count_change(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """upsert=True correctly handles changing chunk count."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import ChunkEmbedding

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-chunk-change',
            source='agent',
            content_type='text',
            text_content='Test content',
            metadata=None,
        )

        # First store with 2 chunks
        chunks_v1 = [
            ChunkEmbedding(embedding=[0.1] * embedding_dim, start_index=0, end_index=50),
            ChunkEmbedding(embedding=[0.2] * embedding_dim, start_index=50, end_index=100),
        ]
        await repos.embeddings.store_chunked(context_id, chunks_v1, 'model')

        # Check initial chunk count
        def _check_initial_counts(conn: sqlite3.Connection) -> tuple[int, int]:
            cursor = conn.execute(
                'SELECT chunk_count FROM embedding_metadata WHERE context_id = ?',
                (context_id,),
            )
            metadata_count = cursor.fetchone()[0]

            cursor = conn.execute(
                'SELECT COUNT(*) FROM embedding_chunks WHERE context_id = ?',
                (context_id,),
            )
            chunk_count = cursor.fetchone()[0]
            return metadata_count, chunk_count

        metadata_count, chunk_count = await backend.execute_read(_check_initial_counts)
        assert metadata_count == 2
        assert chunk_count == 2

        # UPSERT with 3 chunks
        chunks_v2 = [
            ChunkEmbedding(embedding=[0.3] * embedding_dim, start_index=0, end_index=33),
            ChunkEmbedding(embedding=[0.4] * embedding_dim, start_index=33, end_index=66),
            ChunkEmbedding(embedding=[0.5] * embedding_dim, start_index=66, end_index=100),
        ]
        await repos.embeddings.store_chunked(context_id, chunks_v2, 'model', upsert=True)

        # Check updated chunk count
        def _check_updated_counts(conn: sqlite3.Connection) -> tuple[int, int]:
            cursor = conn.execute(
                'SELECT chunk_count FROM embedding_metadata WHERE context_id = ?',
                (context_id,),
            )
            metadata_count = cursor.fetchone()[0]

            cursor = conn.execute(
                'SELECT COUNT(*) FROM embedding_chunks WHERE context_id = ?',
                (context_id,),
            )
            chunk_count = cursor.fetchone()[0]
            return metadata_count, chunk_count

        metadata_count, chunk_count = await backend.execute_read(_check_updated_counts)
        assert metadata_count == 3
        assert chunk_count == 3


@pytest.mark.asyncio
class TestEmbeddingExistsMethod:
    """Tests for EmbeddingRepository.exists() method used by the fix."""

    @requires_sqlite_vec
    async def test_exists_returns_false_for_nonexistent(
        self, async_db_with_embeddings: StorageBackend,
    ) -> None:
        """exists() returns False for context without embeddings."""
        from app.repositories import RepositoryContainer

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)

        # Create entry without embeddings
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-exists',
            source='agent',
            content_type='text',
            text_content='No embedding',
            metadata=None,
        )

        result = await repos.embeddings.exists(context_id)
        assert result is False

    @requires_sqlite_vec
    async def test_exists_returns_true_for_existing(
        self, async_db_with_embeddings: StorageBackend, embedding_dim: int,
    ) -> None:
        """exists() returns True for context with embeddings."""
        from app.repositories import RepositoryContainer
        from app.repositories.embedding_repository import ChunkEmbedding

        backend = async_db_with_embeddings
        repos = RepositoryContainer(backend)

        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='test-exists',
            source='agent',
            content_type='text',
            text_content='Has embedding',
            metadata=None,
        )

        # Store embedding
        chunk_emb = ChunkEmbedding(embedding=[0.1] * embedding_dim, start_index=0, end_index=13)
        await repos.embeddings.store_chunked(context_id, [chunk_emb], 'test-model')

        result = await repos.embeddings.exists(context_id)
        assert result is True
