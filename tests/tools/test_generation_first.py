"""Tests for Generation-First Transactional Integrity pattern.

Validates the uniform asyncio.gather(return_exceptions=True) pattern across
all 4 tools: store_context, update_context, store_context_batch, update_context_batch.

Key behavior under test:
- Both embedding and summary tasks run in parallel via asyncio.gather
- return_exceptions=True prevents one failure from cancelling the other
- If ANY generation task fails, NO data is saved (atomic guarantee)
- Error messages include all failed task details combined
- Retry budgets are fully managed by tenacity wrappers; no re-invocation at gather level
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastmcp.exceptions import ToolError

import app.server
from app.backends.sqlite_backend import SQLiteBackend
from app.repositories import RepositoryContainer
from app.schemas import load_schema
from app.tools.context import store_context
from app.tools.context import update_context

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _create_mock_repositories() -> MagicMock:
    """Create mock repositories with transaction support for batch tool tests."""
    repos = MagicMock()

    mock_backend = MagicMock()

    @asynccontextmanager
    async def mock_begin_transaction():
        txn = MagicMock()
        txn.backend_type = 'sqlite'
        txn.connection = MagicMock()
        yield txn

    mock_backend.begin_transaction = mock_begin_transaction

    repos.context = MagicMock()
    repos.context.backend = mock_backend
    repos.context.store_with_deduplication = AsyncMock(return_value=(100, False))
    repos.context.check_entry_exists = AsyncMock(return_value=(True, 'agent'))
    repos.context.update_context_entry = AsyncMock(
        return_value=(True, ['text_content', 'summary']),
    )
    repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
    repos.context.update_content_type = AsyncMock(return_value=True)
    repos.context.get_summary = AsyncMock(return_value=None)

    repos.tags = MagicMock()
    repos.tags.store_tags = AsyncMock()
    repos.tags.replace_tags_for_context = AsyncMock()

    repos.images = MagicMock()
    repos.images.store_images = AsyncMock()
    repos.images.replace_images_for_context = AsyncMock()

    repos.embeddings = MagicMock()
    repos.embeddings.exists = AsyncMock(return_value=False)
    repos.embeddings.store_chunked = AsyncMock()
    repos.embeddings.delete_all_chunks = AsyncMock()

    return repos


# ---------------------------------------------------------------------------
# store_context tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures('mock_server_dependencies')
class TestStoreContextGenerationFirst:
    """Tests for store_context generation-first pattern with return_exceptions=True."""

    @pytest_asyncio.fixture
    async def setup_backend(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer], None]:
        db_path = tmp_path / 'test_gen_first_store.db'
        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()
        repos = RepositoryContainer(backend)
        yield backend, repos
        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_embedding_fails_summary_succeeds_no_data_saved(
        self, setup_backend: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """PRIMARY BUG FIX: embedding fails but summary succeeds -- no data saved."""
        backend, repos = setup_backend

        mock_emb = MagicMock()
        mock_emb.embed_query = AsyncMock(
            side_effect=Exception('Embedding provider down'),
        )
        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='A summary')

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_emb),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=5.0),
            pytest.raises(ToolError, match='Generation failed after exhausting configured retries'),
        ):
            await store_context(
                thread_id='gf-test-1', source='agent', text='x' * 500,
            )

        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                ('gf-test-1',),
            )
            assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_summary_fails_embedding_succeeds_no_data_saved(
        self, setup_backend: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Summary fails but embedding succeeds -- no data saved."""
        backend, repos = setup_backend

        mock_emb = MagicMock()
        mock_emb.embed_query = AsyncMock(return_value=[0.1] * 1024)
        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(
            side_effect=RuntimeError('Summary provider crashed'),
        )

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_emb),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=5.0),
            pytest.raises(ToolError, match='Generation failed after exhausting configured retries'),
        ):
            await store_context(
                thread_id='gf-test-2', source='agent', text='y' * 500,
            )

        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                ('gf-test-2',),
            )
            assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_both_fail_combined_error_message(
        self, setup_backend: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Both embedding and summary fail -- error message contains both."""
        _backend, repos = setup_backend

        mock_emb = MagicMock()
        mock_emb.embed_query = AsyncMock(
            side_effect=RuntimeError('Embedding down'),
        )
        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(
            side_effect=RuntimeError('Summary down'),
        )

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_emb),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=5.0), pytest.raises(ToolError) as exc_info,
        ):
            await store_context(
                thread_id='gf-test-3', source='agent', text='z' * 500,
            )
        error_msg = str(exc_info.value)
        assert 'embedding' in error_msg
        assert 'summary' in error_msg
        assert 'Generation failed after exhausting configured retries' in error_msg

    @pytest.mark.asyncio
    async def test_both_succeed_data_saved(
        self, setup_backend: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Happy path: both succeed -- data is saved."""
        backend, repos = setup_backend

        mock_emb = MagicMock()
        mock_emb.embed_query = AsyncMock(return_value=[0.1] * 1024)
        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='Generated summary')

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_emb),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=5.0),
            patch.object(repos.embeddings, 'store_chunked', new=AsyncMock()),
        ):
            result = await store_context(
                thread_id='gf-test-4', source='agent', text='w' * 500,
            )
        assert result['success'] is True

        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                ('gf-test-4',),
            )
            assert cursor.fetchone()[0] == 1

    @pytest.mark.asyncio
    async def test_summary_disabled_embedding_fails_no_data_saved(
        self, setup_backend: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Single-task gather: only embedding enabled, it fails -- no data saved."""
        backend, repos = setup_backend

        mock_emb = MagicMock()
        mock_emb.embed_query = AsyncMock(
            side_effect=Exception('Embedding service unavailable'),
        )
        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_emb),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.get_summary_provider', return_value=None),
            pytest.raises(ToolError, match='Generation failed after exhausting configured retries'),
        ):
            await store_context(
                thread_id='gf-test-5', source='agent', text='Test content',
            )

        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                ('gf-test-5',),
            )
            assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_error_message_format(
        self, setup_backend: tuple[SQLiteBackend, RepositoryContainer],
    ) -> None:
        """Validate exact error message format includes type and message."""
        _backend, repos = setup_backend

        mock_emb = MagicMock()
        mock_emb.embed_query = AsyncMock(
            side_effect=ValueError('bad dimension'),
        )
        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_emb),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.get_summary_provider', return_value=None), pytest.raises(ToolError) as exc_info,
        ):
            await store_context(
                thread_id='gf-test-6', source='agent', text='Test',
            )
        error_msg = str(exc_info.value)
        assert 'embedding: ToolError: Embedding generation failed: bad dimension' in error_msg


# ---------------------------------------------------------------------------
# update_context tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures('mock_server_dependencies')
class TestUpdateContextGenerationFirst:
    """Tests for update_context generation-first pattern with return_exceptions=True."""

    @pytest_asyncio.fixture
    async def setup_with_entry(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer, int], None]:
        db_path = tmp_path / 'test_gen_first_update.db'
        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            conn.execute(
                '''INSERT INTO context_entries
                   (thread_id, source, text_content, content_type, metadata)
                   VALUES (?, ?, ?, ?, ?)''',
                ('upd-thread', 'agent', 'Original text', 'text', '{}'),
            )
            conn.commit()
        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()
        repos = RepositoryContainer(backend)
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT id FROM context_entries WHERE thread_id = ?',
                ('upd-thread',),
            )
            entry_id = cursor.fetchone()[0]
        yield backend, repos, entry_id
        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_embedding_fails_summary_succeeds_original_preserved(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, int],
    ) -> None:
        """Update: embedding fails, summary succeeds -- original preserved."""
        backend, repos, entry_id = setup_with_entry

        mock_emb = MagicMock()
        mock_emb.embed_query = AsyncMock(side_effect=Exception('Embedding fail'))
        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='New summary')

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_emb),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=5.0),
            pytest.raises(ToolError, match='Generation failed after exhausting configured retries'),
        ):
            await update_context(context_id=entry_id, text='Updated text ' * 30)

        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT text_content FROM context_entries WHERE id = ?', (entry_id,),
            )
            assert cursor.fetchone()[0] == 'Original text'

    @pytest.mark.asyncio
    async def test_summary_fails_embedding_succeeds_original_preserved(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, int],
    ) -> None:
        """Update: summary fails, embedding succeeds -- original preserved."""
        backend, repos, entry_id = setup_with_entry

        mock_emb = MagicMock()
        mock_emb.embed_query = AsyncMock(return_value=[0.1] * 1024)
        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(
            side_effect=RuntimeError('Summary crash'),
        )

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_emb),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=5.0),
            pytest.raises(ToolError, match='Generation failed after exhausting configured retries'),
        ):
            await update_context(context_id=entry_id, text='Updated text ' * 40)

        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT text_content FROM context_entries WHERE id = ?', (entry_id,),
            )
            assert cursor.fetchone()[0] == 'Original text'

    @pytest.mark.asyncio
    async def test_both_fail_combined_error(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, int],
    ) -> None:
        """Update: both fail -- combined error message."""
        _backend, repos, entry_id = setup_with_entry

        mock_emb = MagicMock()
        mock_emb.embed_query = AsyncMock(side_effect=RuntimeError('Emb fail'))
        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(side_effect=RuntimeError('Sum fail'))

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_emb),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=5.0), pytest.raises(ToolError) as exc_info,
        ):
            await update_context(context_id=entry_id, text='Updated text ' * 40)
        error_msg = str(exc_info.value)
        assert 'embedding' in error_msg
        assert 'summary' in error_msg

    @pytest.mark.asyncio
    async def test_both_succeed_data_updated(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, int],
    ) -> None:
        """Update: both succeed -- data is updated."""
        backend, repos, entry_id = setup_with_entry

        mock_emb = MagicMock()
        mock_emb.embed_query = AsyncMock(return_value=[0.1] * 1024)
        mock_chunking = MagicMock()
        mock_chunking.is_enabled = False

        mock_summary = MagicMock()
        mock_summary.summarize = AsyncMock(return_value='New summary')

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=mock_emb),
            patch('app.tools.context.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.get_summary_provider', return_value=mock_summary),
            patch('app.tools.context.compute_summary_total_timeout', return_value=5.0),
            patch.object(repos.embeddings, 'store_chunked', new=AsyncMock()),
            patch.object(repos.embeddings, 'delete_all_chunks', new=AsyncMock()),
        ):
            result = await update_context(context_id=entry_id, text='Updated text ' * 40)
        assert result['success'] is True

        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT text_content FROM context_entries WHERE id = ?', (entry_id,),
            )
            assert cursor.fetchone()[0] == ('Updated text ' * 40).strip()


# ---------------------------------------------------------------------------
# store_context_batch tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures('mock_server_dependencies')
class TestStoreContextBatchGenerationFirst:
    """Tests for store_context_batch generation-first pattern."""

    @pytest.mark.asyncio
    async def test_atomic_embedding_fails_no_data_saved(self) -> None:
        """Atomic batch: embedding fails on one entry -- entire batch fails."""
        repos = _create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(return_value=(101, False))

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch(
                'app.tools.batch.get_embedding_provider',
                return_value=MagicMock(),  # non-None so embedding task is added
            ),
            patch('app.startup.get_summary_provider', return_value=None),
            patch(
                'app.tools.context.generate_embeddings_with_timeout',
                new=AsyncMock(side_effect=ToolError('Embedding generation timed out')),
            ),
            pytest.raises(ToolError, match='Generation failed at index 0'),
        ):
            await app.server.store_context_batch(
                entries=[
                    {'thread_id': 'bf-1', 'source': 'agent', 'text': 'Entry 1'},
                ],
                atomic=True,
            )

        repos.context.store_with_deduplication.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_atomic_summary_fails_no_data_saved(self) -> None:
        """Atomic batch: summary fails -- entire batch fails, no data saved."""
        repos = _create_mock_repositories()

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.context.generate_summary_with_timeout',
                new=AsyncMock(side_effect=ToolError('Summary generation timed out')),
            ),
            pytest.raises(ToolError, match='Generation failed at index 0'),
        ):
            await app.server.store_context_batch(
                entries=[
                    {'thread_id': 'bf-2', 'source': 'agent', 'text': 'x' * 500},
                ],
                atomic=True,
            )

        repos.context.store_with_deduplication.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_atomic_partial_generation_failure(self) -> None:
        """Non-atomic batch: one entry fails generation, others succeed."""
        repos = _create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(return_value=(200, False))

        call_count = 0

        async def selective_summary(_text: str, _source: str) -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError('Provider overloaded')
            return 'Summary ok'

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.context.generate_summary_with_timeout',
                new=AsyncMock(side_effect=selective_summary),
            ),
        ):
            result = await app.server.store_context_batch(
                entries=[
                    {'thread_id': 'bf-3', 'source': 'agent', 'text': 'x' * 500},
                    {'thread_id': 'bf-3', 'source': 'agent', 'text': 'y' * 500},
                ],
                atomic=False,
            )

        assert result['succeeded'] == 1
        assert result['failed'] == 1
        failed = [r for r in result['results'] if not r['success']]
        assert len(failed) == 1
        assert 'Generation failed' in (failed[0].get('error') or '')

    @pytest.mark.asyncio
    async def test_atomic_both_succeed_data_saved(self) -> None:
        """Atomic batch: both embedding+summary succeed -- data saved."""
        repos = _create_mock_repositories()
        repos.context.store_with_deduplication = AsyncMock(
            side_effect=[(101, False), (102, False)],
        )

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.context.generate_summary_with_timeout',
                new=AsyncMock(return_value='Batch summary'),
            ),
        ):
            result = await app.server.store_context_batch(
                entries=[
                    {'thread_id': 'bf-4', 'source': 'agent', 'text': 'x' * 500},
                    {'thread_id': 'bf-4', 'source': 'agent', 'text': 'y' * 500},
                ],
                atomic=True,
            )

        assert result['success'] is True
        assert result['succeeded'] == 2


# ---------------------------------------------------------------------------
# update_context_batch tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures('mock_server_dependencies')
class TestUpdateContextBatchGenerationFirst:
    """Tests for update_context_batch generation-first pattern."""

    @pytest.mark.asyncio
    async def test_atomic_embedding_fails_no_data_modified(self) -> None:
        """Atomic update batch: embedding fails -- no data modified."""
        repos = _create_mock_repositories()

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch(
                'app.tools.batch.get_embedding_provider',
                return_value=MagicMock(),
            ),
            patch('app.startup.get_summary_provider', return_value=None),
            patch(
                'app.tools.context.generate_embeddings_with_timeout',
                new=AsyncMock(side_effect=ToolError('Embedding generation timed out')),
            ),
            pytest.raises(ToolError, match='Generation failed for context 1'),
        ):
            await app.server.update_context_batch(
                updates=[{'context_id': 1, 'text': 'Updated text'}],
                atomic=True,
            )

        repos.context.update_context_entry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_atomic_summary_fails_no_data_modified(self) -> None:
        """Atomic update batch: summary fails -- no data modified."""
        repos = _create_mock_repositories()

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.context.generate_summary_with_timeout',
                new=AsyncMock(side_effect=ToolError('Summary timed out')),
            ),
            pytest.raises(ToolError, match='Generation failed for context 1'),
        ):
            await app.server.update_context_batch(
                updates=[{'context_id': 1, 'text': 'x' * 500}],
                atomic=True,
            )

        repos.context.update_context_entry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_atomic_partial_generation_failure(self) -> None:
        """Non-atomic update batch: one entry fails, others succeed."""
        repos = _create_mock_repositories()

        call_count = 0

        async def selective_summary(_text: str, _source: str) -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError('Provider overloaded')
            return 'Summary ok'

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.context.generate_summary_with_timeout',
                new=AsyncMock(side_effect=selective_summary),
            ),
        ):
            result = await app.server.update_context_batch(
                updates=[
                    {'context_id': 1, 'text': 'x' * 500},
                    {'context_id': 2, 'text': 'y' * 500},
                ],
                atomic=False,
            )

        assert result['succeeded'] == 1
        assert result['failed'] == 1

    @pytest.mark.asyncio
    async def test_no_text_change_skips_generation(self) -> None:
        """Update batch: metadata-only update skips generation entirely."""
        repos = _create_mock_repositories()

        mock_gen_emb = AsyncMock(return_value=None)
        mock_gen_sum = AsyncMock(return_value=None)

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=MagicMock()),
            patch('app.startup.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.context.generate_embeddings_with_timeout',
                new=mock_gen_emb,
            ),
            patch(
                'app.tools.context.generate_summary_with_timeout',
                new=mock_gen_sum,
            ),
        ):
            result = await app.server.update_context_batch(
                updates=[{'context_id': 1, 'metadata': {'key': 'value'}}],
                atomic=True,
            )

        assert result['success'] is True
        # Generation should NOT have been called for metadata-only update
        mock_gen_emb.assert_not_awaited()
        mock_gen_sum.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clear_summary_semantics_preserved(self) -> None:
        """Update batch: text shorter than min_content_length clears existing summary."""
        repos = _create_mock_repositories()

        mock_gen_sum = AsyncMock(return_value='Should not be called')

        with (
            patch('app.tools.batch.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.startup.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools.context.generate_summary_with_timeout',
                new=mock_gen_sum,
            ),
        ):
            # Text is short (< default min_content_length of 500)
            result = await app.server.update_context_batch(
                updates=[{'context_id': 1, 'text': 'Short text'}],
                atomic=True,
            )

        assert result['success'] is True
        # Summary wrapper should NOT be called (text too short)
        mock_gen_sum.assert_not_awaited()
        # Verify update was called with summary=None and clear_summary flag
        repos.context.update_context_entry.assert_awaited_once()
        call_kwargs = repos.context.update_context_entry.call_args.kwargs
        assert call_kwargs.get('summary') is None
