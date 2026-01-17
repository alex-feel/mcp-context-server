"""Tests for concurrent operations with embedding failures.

This module tests the system's behavior under concurrent load with
simulated embedding failures, verifying:
1. No orphaned entries exist after concurrent operations
2. Atomic transactions work correctly under concurrent access
3. Partial failures do not corrupt database state

Phase 5 of the Transactional Integrity Fix:
- Testing and Validation
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from random import Random
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio

from app.backends.sqlite_backend import SQLiteBackend
from app.repositories import RepositoryContainer
from app.schemas import load_schema


class TestConcurrentEmbeddingOperations:
    """Tests for concurrent operations with embedding failures.

    Verifies:
    - No orphaned entries after concurrent operations with failures
    - Atomic batches maintain all-or-nothing under concurrent load
    - Concurrent updates preserve data isolation
    """

    @pytest_asyncio.fixture
    async def setup_backend_and_repos(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer], None]:
        """Set up backend and repositories for testing."""
        db_path = tmp_path / 'test_concurrent.db'

        # Create schema
        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            # SQLite optimizations for concurrent access
            conn.execute('PRAGMA journal_mode = WAL')
            conn.execute('PRAGMA busy_timeout = 30000')

        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()

        repos = RepositoryContainer(backend)

        yield backend, repos

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_store_with_10_percent_embedding_failures(
        self, setup_backend_and_repos: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Test 50+ concurrent store_context calls with 10% embedding failures.

        CRITICAL VERIFICATION (User Requirement ID 3995):
        - No orphaned entries (entries without embeddings when embeddings enabled)
        - Successful entries are stored
        - Failed entries have no data in database
        """
        backend, repos = setup_backend_and_repos

        total_operations = 50
        failure_rate = 0.10  # 10% failure rate
        rng = Random(42)  # Deterministic for reproducibility

        # Track which operations should fail
        should_fail = [rng.random() < failure_rate for _ in range(total_operations)]
        expected_failures = sum(1 for f in should_fail if f)

        call_counter = {'count': 0}
        lock = asyncio.Lock()

        async def mock_embed_query(_text: str) -> list[float]:
            """Mock embedding that fails for selected operations."""
            async with lock:
                idx = call_counter['count']
                call_counter['count'] += 1

            if idx < len(should_fail) and should_fail[idx]:
                raise Exception(f'Simulated embedding failure at index {idx}')
            return [0.1] * 1024

        mock_provider = MagicMock()
        mock_provider.embed_query = AsyncMock(side_effect=mock_embed_query)

        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        results: list[tuple[int, bool, str | None]] = []  # (index, success, error)
        results_lock = asyncio.Lock()

        async def store_one(idx: int) -> None:
            """Store a single context entry."""
            from fastmcp.exceptions import ToolError

            from app.tools.context import store_context

            try:
                await store_context(
                    thread_id='concurrent-test',
                    source='agent',
                    text=f'Test content {idx}',
                )
                async with results_lock:
                    results.append((idx, True, None))
            except ToolError as e:
                async with results_lock:
                    results.append((idx, False, str(e)))

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            # Mock embedding repository to avoid vec_context_embeddings table issues
            patch.object(repos.embeddings, 'store_chunked', new=AsyncMock(return_value=None)),
        ):
            # Run all operations concurrently
            tasks = [store_one(i) for i in range(total_operations)]
            await asyncio.gather(*tasks)

        # Verify results
        actual_successes = sum(1 for _, success, _ in results if success)
        actual_failures = sum(1 for _, success, _ in results if not success)

        # Count entries in database
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                ('concurrent-test',),
            )
            db_count = cursor.fetchone()[0]

        # CRITICAL ASSERTIONS (User Requirement: no orphaned entries)
        assert db_count == actual_successes, (
            f'Database should have exactly {actual_successes} entries, '
            f'but found {db_count}. This indicates orphaned entries exist!'
        )

        # Verify failure rate is roughly as expected (allow some variance)
        assert actual_failures >= expected_failures - 2, (
            f'Expected ~{expected_failures} failures but got {actual_failures}'
        )

        # Verify no orphaned entries (all entries have expected content)
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT text_content FROM context_entries WHERE thread_id = ?',
                ('concurrent-test',),
            )
            stored_texts = [row[0] for row in cursor.fetchall()]
            for text in stored_texts:
                assert 'Test content' in text, f'Unexpected content: {text}'

    @pytest.mark.asyncio
    async def test_concurrent_batch_atomic_all_or_nothing(
        self, setup_backend_and_repos: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Test concurrent batch operations maintain atomicity.

        Multiple atomic batches submitted concurrently should each
        be fully committed or fully rolled back - no partial state.
        """
        backend, repos = setup_backend_and_repos

        batch_count = 5
        entries_per_batch = 5

        async def mock_embed_always_succeed(_text: str) -> list[float]:
            return [0.1] * 1024

        mock_provider = MagicMock()
        mock_provider.embed_query = AsyncMock(side_effect=mock_embed_always_succeed)

        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        results: list[tuple[int, bool, int | None]] = []  # (batch_idx, success, entries_count)
        results_lock = asyncio.Lock()

        async def store_batch(batch_idx: int) -> None:
            """Store a batch of entries."""
            from app.tools.batch import store_context_batch

            entries = [
                {
                    'thread_id': f'batch-{batch_idx}',
                    'source': 'agent',
                    'text': f'Batch {batch_idx} Entry {i}',
                }
                for i in range(entries_per_batch)
            ]

            try:
                result = await store_context_batch(entries=entries, atomic=True)
                async with results_lock:
                    results.append((batch_idx, True, result.get('succeeded', 0)))
            except Exception:
                async with results_lock:
                    results.append((batch_idx, False, None))

        with (
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=mock_provider),
            patch('app.startup.get_chunking_service', return_value=mock_chunking),
            patch.object(repos.embeddings, 'store_chunked', new=AsyncMock(return_value=None)),
        ):
            # Run all batches concurrently
            tasks = [store_batch(i) for i in range(batch_count)]
            await asyncio.gather(*tasks)

        # Verify each batch is fully stored or not stored at all
        for batch_idx in range(batch_count):
            async with backend.get_connection(readonly=True) as conn:
                cursor = conn.execute(
                    'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                    (f'batch-{batch_idx}',),
                )
                count = cursor.fetchone()[0]

            # Atomic: should be all or nothing
            assert count in (0, entries_per_batch), (
                f'Batch {batch_idx} has {count} entries, expected 0 or {entries_per_batch}. '
                'This indicates atomic transaction was violated!'
            )

    @pytest.mark.asyncio
    async def test_concurrent_updates_preserve_isolation(
        self, setup_backend_and_repos: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Test concurrent updates to different entries maintain isolation.

        Updating multiple entries concurrently should not cause
        cross-contamination or data corruption.
        """
        backend, repos = setup_backend_and_repos

        # Pre-create entries
        entry_count = 10
        entry_ids = []

        with sqlite3.connect(str(backend.db_path)) as conn:
            for i in range(entry_count):
                conn.execute(
                    '''INSERT INTO context_entries
                       (thread_id, source, text_content, content_type)
                       VALUES (?, ?, ?, ?)''',
                    (f'update-test-{i}', 'agent', f'Original {i}', 'text'),
                )
            conn.commit()

            cursor = conn.execute(
                "SELECT id FROM context_entries WHERE thread_id LIKE 'update-test-%' ORDER BY id",
            )
            entry_ids = [row[0] for row in cursor.fetchall()]

        async def mock_embed(_text: str) -> list[float]:
            return [0.1] * 1024

        mock_provider = MagicMock()
        mock_provider.embed_query = AsyncMock(side_effect=mock_embed)

        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        results: list[bool] = []
        results_lock = asyncio.Lock()

        async def update_one(idx: int) -> None:
            """Update a single entry."""
            from app.tools.context import update_context

            try:
                await update_context(
                    context_id=entry_ids[idx],
                    text=f'Updated {idx}',
                )
                async with results_lock:
                    results.append(True)
            except Exception:
                async with results_lock:
                    results.append(False)

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            patch.object(repos.embeddings, 'delete_all_chunks', new=AsyncMock(return_value=None)),
            patch.object(repos.embeddings, 'store_chunked', new=AsyncMock(return_value=None)),
        ):
            # Run all updates concurrently
            tasks = [update_one(i) for i in range(entry_count)]
            await asyncio.gather(*tasks)

        # Verify all updates succeeded
        success_count = sum(1 for r in results if r)
        assert success_count == entry_count, f'Expected all {entry_count} updates to succeed'

        # Verify each entry has unique content (no cross-contamination)
        async with backend.get_connection(readonly=True) as conn:
            for idx, entry_id in enumerate(entry_ids):
                cursor = conn.execute(
                    'SELECT text_content FROM context_entries WHERE id = ?',
                    (entry_id,),
                )
                row = cursor.fetchone()
                content = row[0] if row else ''
                assert f'Updated {idx}' == content, (
                    f'Entry {entry_id} has wrong content: {content}. '
                    f'Expected "Updated {idx}". This indicates data cross-contamination!'
                )


class TestTransactionRollbackComprehensive:
    """Additional transaction rollback tests for edge cases.

    Verifies rollback behavior in scenarios not covered by basic tests.
    """

    @pytest_asyncio.fixture
    async def setup_backend_and_repos(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer], None]:
        """Set up backend and repositories for testing."""
        db_path = tmp_path / 'test_rollback.db'

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()

        repos = RepositoryContainer(backend)

        yield backend, repos

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_store_context_tag_storage_failure_rollback(
        self, setup_backend_and_repos: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Test rollback when tag storage fails after embedding success.

        If embedding succeeds but subsequent DB operation (tag storage) fails,
        no data should be saved (embedding-first + atomic transaction).
        """
        backend, repos = setup_backend_and_repos

        async def mock_embed(_text: str) -> list[float]:
            return [0.1] * 1024

        mock_provider = MagicMock()
        mock_provider.embed_query = AsyncMock(side_effect=mock_embed)

        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        async def failing_store_tags(*_args: object, **_kwargs: object) -> None:
            raise Exception('Simulated tag storage failure')

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_provider),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            patch.object(repos.embeddings, 'store_chunked', new=AsyncMock(return_value=None)),
            patch.object(repos.tags, 'store_tags', new=AsyncMock(side_effect=failing_store_tags)),
        ):
            from fastmcp.exceptions import ToolError

            from app.tools.context import store_context

            with pytest.raises(ToolError):
                await store_context(
                    thread_id='tag-failure-test',
                    source='agent',
                    text='Should be rolled back',
                    tags=['test-tag'],
                )

        # Verify NO data was saved (transaction rolled back)
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                ('tag-failure-test',),
            )
            count = cursor.fetchone()[0]
            assert count == 0, (
                'Context should be rolled back when tag storage fails. '
                'This indicates transaction atomicity was violated!'
            )

    @pytest.mark.asyncio
    async def test_update_context_batch_mixed_embedding_failures(
        self, setup_backend_and_repos: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Test update_context_batch with mixed embedding successes and failures.

        In non-atomic mode, only entries with successful embeddings should be updated.
        Failed entries should retain their original content.
        """
        backend, repos = setup_backend_and_repos

        # Pre-create entries
        entry_ids = []
        with sqlite3.connect(str(backend.db_path)) as conn:
            for i in range(5):
                conn.execute(
                    '''INSERT INTO context_entries
                       (thread_id, source, text_content, content_type)
                       VALUES (?, ?, ?, ?)''',
                    (f'mixed-update-{i}', 'agent', f'Original {i}', 'text'),
                )
            conn.commit()
            cursor = conn.execute(
                "SELECT id FROM context_entries WHERE thread_id LIKE 'mixed-update-%' ORDER BY id",
            )
            entry_ids = [row[0] for row in cursor.fetchall()]

        call_count = 0
        lock = asyncio.Lock()

        async def mock_embed(_text: str) -> list[float]:
            nonlocal call_count
            async with lock:
                call_count += 1
                current = call_count
            # Fail on entries 2 and 4 (1-indexed)
            if current in (2, 4):
                raise Exception(f'Embedding failure at call {current}')
            return [0.1] * 1024

        mock_provider = MagicMock()
        mock_provider.embed_query = AsyncMock(side_effect=mock_embed)

        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        with (
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=mock_provider),
            patch('app.startup.get_chunking_service', return_value=mock_chunking),
            patch.object(repos.embeddings, 'delete_all_chunks', new=AsyncMock(return_value=None)),
            patch.object(repos.embeddings, 'store_chunked', new=AsyncMock(return_value=None)),
        ):
            from app.tools.batch import update_context_batch

            updates = [
                {'context_id': entry_ids[i], 'text': f'Updated {i}'}
                for i in range(5)
            ]

            result = await update_context_batch(updates=updates, atomic=False)

        # Non-atomic: 3 should succeed, 2 should fail
        assert result['succeeded'] == 3, f"Expected 3 successes, got {result['succeeded']}"
        assert result['failed'] == 2, f"Expected 2 failures, got {result['failed']}"

        # Verify database state
        async with backend.get_connection(readonly=True) as conn:
            for idx, entry_id in enumerate(entry_ids):
                cursor = conn.execute(
                    'SELECT text_content FROM context_entries WHERE id = ?',
                    (entry_id,),
                )
                content = cursor.fetchone()[0]

                # Entries at call positions 2 and 4 (0-indexed: 1 and 3) should retain original
                if idx in (1, 3):
                    assert content == f'Original {idx}', (
                        f'Entry {idx} should retain original content but got: {content}'
                    )
                else:
                    assert content == f'Updated {idx}', (
                        f'Entry {idx} should be updated but got: {content}'
                    )
