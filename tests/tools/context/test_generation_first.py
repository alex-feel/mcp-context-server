"""Generation-first transactional integrity for store_context and update_context.

The embedding and flat-summary legs run concurrently with their errors collected
(asyncio.gather with return_exceptions=True), so one failure never cancels the other.

Key behavior under test:
- If either generation task fails, no data is saved and an update leaves the original entry unchanged
- Error messages combine the details of every failed task
- Retry budgets are fully managed by tenacity wrappers; no re-invocation at gather level
"""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastmcp.exceptions import ToolError

from app.backends.sqlite_backend import SQLiteBackend
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.schemas import load_schema
from app.tools.context.store import store_context
from app.tools.context.update import update_context

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
        """Embedding fails but summary succeeds -- no data saved."""
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
            patch('app.tools.context.store.ensure_repositories', return_value=repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=5.0),
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
            patch('app.tools.context.store.ensure_repositories', return_value=repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=5.0),
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
            patch('app.tools.context.store.ensure_repositories', return_value=repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=5.0),
            pytest.raises(ToolError) as exc_info,
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
            patch('app.tools.context.store.ensure_repositories', return_value=repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.store.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=5.0),
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
            patch('app.tools.context.store.ensure_repositories', return_value=repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.store.get_summary_provider', return_value=None),
            patch('app.tools._generation.get_summary_provider', return_value=None),
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
            patch('app.tools.context.store.ensure_repositories', return_value=repos),
            patch('app.tools.context.store.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.store.get_summary_provider', return_value=None),
            patch('app.tools._generation.get_summary_provider', return_value=None),
            pytest.raises(ToolError) as exc_info,
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
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer, str], None]:
        db_path = tmp_path / 'test_gen_first_update.db'
        schema_sql = load_schema('sqlite')
        new_id = generate_id()
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, text_content, content_type, metadata, owner_id) '
                "VALUES (?, ?, ?, ?, ?, ?, 'local')",
                (new_id, 'upd-thread', 'agent', 'Original text', 'text', '{}'),
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
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
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
            patch('app.tools.context.update.ensure_repositories', return_value=repos),
            patch('app.tools.context.update.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.update.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=5.0),
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
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
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
            patch('app.tools.context.update.ensure_repositories', return_value=repos),
            patch('app.tools.context.update.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.update.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=5.0),
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
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
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
            patch('app.tools.context.update.ensure_repositories', return_value=repos),
            patch('app.tools.context.update.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.update.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=5.0),
            pytest.raises(ToolError) as exc_info,
        ):
            await update_context(context_id=entry_id, text='Updated text ' * 40)
        error_msg = str(exc_info.value)
        assert 'embedding' in error_msg
        assert 'summary' in error_msg

    @pytest.mark.asyncio
    async def test_both_succeed_data_updated(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
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
            patch('app.tools.context.update.ensure_repositories', return_value=repos),
            patch('app.tools.context.update.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_embedding_provider', return_value=mock_emb),
            patch('app.tools._generation.get_chunking_service', return_value=mock_chunking),
            patch('app.tools.context.update.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.get_summary_provider', return_value=mock_summary),
            patch('app.tools._generation.compute_summary_total_timeout', return_value=5.0),
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
