"""Integration tests for summary generation across the tool lifecycle."""

import asyncio
from collections.abc import Generator
from contextlib import asynccontextmanager
from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.server
import app.startup
import app.tools.context as context_tools
import app.tools.search as search_tools
from app.repositories.embedding_repository import ChunkEmbedding
from app.startup import ensure_repositories
from app.startup import set_backend
from app.startup import set_chunking_service
from app.startup import set_embedding_provider
from app.startup import set_repositories
from app.startup import set_reranking_provider
from app.startup import set_summary_provider

store_context = app.server.store_context
update_context = app.server.update_context
search_context = app.server.search_context


def _create_mock_repositories() -> MagicMock:
    """Create mock repositories with transaction support for tool tests."""
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
    repos.context.check_latest_is_duplicate = AsyncMock(return_value=None)
    repos.context.store_with_deduplication = AsyncMock(return_value=(123, False))
    repos.context.check_entry_exists = AsyncMock(return_value=(True, 'agent'))
    repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content', 'summary']))
    repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
    repos.context.get_content_type = AsyncMock(return_value='text')
    repos.context.update_content_type = AsyncMock(return_value=True)
    repos.context.get_summary = AsyncMock(return_value=None)

    repos.tags = MagicMock()
    repos.tags.store_tags = AsyncMock()
    repos.tags.replace_tags_for_context = AsyncMock()

    repos.images = MagicMock()
    repos.images.store_images = AsyncMock()
    repos.images.replace_images_for_context = AsyncMock()
    repos.images.count_images_for_context = AsyncMock(return_value=0)

    repos.embeddings = MagicMock()
    repos.embeddings.exists = AsyncMock(return_value=False)
    repos.embeddings.store_chunked = AsyncMock()
    repos.embeddings.delete_all_chunks = AsyncMock()

    return repos


@pytest.fixture(autouse=True)
def reset_summary_state() -> Generator[None, None, None]:
    """Reset global summary state between tests."""
    original_summary_provider = app.startup.get_summary_provider()
    original_embedding_provider = app.startup.get_embedding_provider()
    context_tools._summary_semaphore = None

    try:
        yield
    finally:
        set_summary_provider(original_summary_provider)
        set_embedding_provider(original_embedding_provider)
        context_tools._summary_semaphore = None


@pytest.mark.usefixtures('mock_server_dependencies')
class TestGenerateSummaryWithTimeout:
    """Tests for summary generation helper behavior."""

    @pytest.mark.asyncio
    async def test_returns_summary_from_provider(self) -> None:
        """Return the provider result when summary generation succeeds."""
        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(return_value='Generated summary')

        with (
            patch('app.tools.context.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
            patch('app.tools.context.settings') as mock_settings,
        ):
            mock_settings.summary.max_concurrent = 2

            result = await context_tools.generate_summary_with_timeout('Long text', 'agent')

        assert result == 'Generated summary'
        mock_provider.summarize.assert_awaited_once_with('Long text', 'agent')

    @pytest.mark.asyncio
    async def test_timeout_raises_tool_error(self) -> None:
        """Raise ToolError when total timeout is exceeded."""

        async def slow_summary(_text: str, _source: str) -> str:
            await asyncio.sleep(0.2)
            return 'Too late'

        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(side_effect=slow_summary)

        with (
            patch('app.tools.context.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.compute_summary_total_timeout', return_value=0.05),
            patch('app.tools.context.settings') as mock_settings,
        ):
            mock_settings.summary.max_concurrent = 2

            with pytest.raises(ToolError, match='Summary generation exceeded total timeout'):
                await context_tools.generate_summary_with_timeout('Long text', 'agent')

    @pytest.mark.asyncio
    async def test_provider_none_skips_generation(self) -> None:
        """Return None when no summary provider is configured."""
        with patch('app.tools.context.get_summary_provider', return_value=None):
            result = await context_tools.generate_summary_with_timeout('Long text', 'agent')

        assert result is None


@pytest.mark.usefixtures('mock_server_dependencies')
class TestSummaryStoreWithMocks:
    """Tests for summary behavior in store_context with mocked repositories."""

    @pytest.mark.asyncio
    async def test_store_context_generates_summary_in_parallel_with_embeddings(self) -> None:
        """Run summary and embedding generation concurrently before the transaction."""
        repos = _create_mock_repositories()
        embedding_started = asyncio.Event()
        summary_started = asyncio.Event()

        async def fake_embedding(text: str) -> list[ChunkEmbedding]:
            embedding_started.set()
            await asyncio.wait_for(summary_started.wait(), timeout=0.2)
            return [ChunkEmbedding(embedding=[0.1, 0.2], start_index=0, end_index=len(text))]

        async def fake_summary(_text: str, _source: str) -> str:
            summary_started.set()
            await asyncio.wait_for(embedding_started.wait(), timeout=0.2)
            return 'Generated summary'

        with (
            patch('app.tools.context.ensure_repositories', new=AsyncMock(return_value=repos)),
            patch('app.tools.context.get_embedding_provider', return_value=MagicMock()),
            patch('app.tools.context.get_summary_provider', return_value=MagicMock()),
            patch('app.tools.context.generate_embeddings_with_timeout', side_effect=fake_embedding),
            patch('app.tools.context.generate_summary_with_timeout', side_effect=fake_summary),
        ):
            long_text = 'x' * 500
            result = await store_context(
                thread_id='parallel-summary-thread',
                source='agent',
                text=long_text,
            )

        assert result['success'] is True
        assert 'embedding generated' in result['message']
        assert 'summary generated' in result['message']
        repos.context.store_with_deduplication.assert_awaited_once_with(
            thread_id='parallel-summary-thread',
            source='agent',
            content_type='text',
            text_content=long_text,
            metadata=None,
            summary='Generated summary',
            txn=ANY,
        )
        repos.embeddings.store_chunked.assert_awaited_once()


@pytest.mark.usefixtures('initialized_server')
class TestSummaryIntegration:
    """Integration tests for summary behavior with the real SQLite repositories."""

    @pytest.mark.asyncio
    async def test_update_context_regenerates_summary_on_text_change(self) -> None:
        """Regenerate and store a new summary when text changes."""
        repos = await ensure_repositories()
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='update-summary-thread',
            source='agent',
            content_type='text',
            text_content='Original text',
            metadata=None,
            summary='Original summary',
        )

        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(return_value='Updated summary')

        updated_text = 'x' * 500

        with (
            patch('app.tools.context.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.get_embedding_provider', return_value=None),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
        ):
            result = await update_context(
                context_id=context_id,
                text=updated_text,
            )

        rows = await repos.context.get_by_ids([context_id])
        assert rows[0]['text_content'] == updated_text
        assert rows[0]['summary'] == 'Updated summary'
        assert '(summary regenerated)' in result['message']
        mock_provider.summarize.assert_awaited_once_with(updated_text, 'agent')

    @pytest.mark.asyncio
    async def test_update_context_preserves_summary_on_metadata_only_change(self) -> None:
        """Leave an existing summary unchanged when only metadata is updated."""
        repos = await ensure_repositories()
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='metadata-preserve-thread',
            source='agent',
            content_type='text',
            text_content='Text with summary',
            metadata='{"status": "old"}',
            summary='Existing summary',
        )

        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(return_value='Should not be used')

        with patch('app.tools.context.get_summary_provider', return_value=mock_provider):
            result = await update_context(
                context_id=context_id,
                metadata={'status': 'new'},
            )

        rows = await repos.context.get_by_ids([context_id])
        assert rows[0]['summary'] == 'Existing summary'
        assert '(summary regenerated)' not in result['message']
        mock_provider.summarize.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_context_shows_truncated_text_and_summary(self) -> None:
        """Search results show truncated text_content alongside summary field."""
        repos = await ensure_repositories()
        long_text = 'A' * 400
        await repos.context.store_with_deduplication(
            thread_id='search-summary-thread',
            source='agent',
            content_type='text',
            text_content=long_text,
            metadata=None,
            summary='Short summary for search results',
        )

        result = await search_context(thread_id='search-summary-thread', limit=10)

        entry = result['results'][0]
        # text_content is truncated original text, NOT summary
        assert entry['text_content'] != long_text
        assert len(entry['text_content']) <= 303  # 300 + '...'
        assert entry['is_text_content_truncated'] is True
        # summary is a separate field
        assert entry['summary'] == 'Short summary for search results'

    @pytest.mark.asyncio
    async def test_search_context_truncates_without_summary(self) -> None:
        """Truncate long text and show empty summary when no summary stored."""
        repos = await ensure_repositories()
        long_text = 'B' * 400
        await repos.context.store_with_deduplication(
            thread_id='search-fallback-thread',
            source='agent',
            content_type='text',
            text_content=long_text,
            metadata=None,
            summary=None,
        )

        result = await search_context(thread_id='search-fallback-thread', limit=10)

        entry = result['results'][0]
        assert entry['text_content'] != long_text
        assert len(entry['text_content']) < len(long_text)
        assert entry['is_text_content_truncated'] is True
        assert entry['summary'] == ''

    @pytest.mark.asyncio
    async def test_dedup_preserves_existing_summary(self) -> None:
        """Reuse the existing summary for duplicate content instead of regenerating it."""
        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(side_effect=['Original summary', 'Unexpected second summary'])

        with (
            patch('app.tools.context.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.get_embedding_provider', return_value=None),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
        ):
            dedup_text = 'x' * 500
            first_result = await store_context(
                thread_id='dedup-summary-thread',
                source='agent',
                text=dedup_text,
            )
            second_result = await store_context(
                thread_id='dedup-summary-thread',
                source='agent',
                text=dedup_text,
            )

        repos = await ensure_repositories()
        rows = await repos.context.get_by_ids([first_result['context_id']])
        assert first_result['context_id'] == second_result['context_id']
        assert rows[0]['summary'] == 'Original summary'
        assert '(summary preserved)' in second_result['message']
        assert mock_provider.summarize.await_count == 1

    @pytest.mark.asyncio
    async def test_dedup_generates_summary_when_missing(self) -> None:
        """Generate a summary for a duplicate entry that lacks an existing summary."""
        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(return_value='Newly generated summary')

        dedup_text = 'y' * 500

        with (
            patch('app.tools.context.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.get_embedding_provider', return_value=None),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
        ):
            # First store creates the entry (summary generated)
            first_result = await store_context(
                thread_id='dedup-no-summary-thread',
                source='agent',
                text=dedup_text,
            )

        # Manually clear the summary from the database to simulate missing summary
        repos = await ensure_repositories()
        context_id = first_result['context_id']

        if repos.context.backend.backend_type == 'sqlite':
            import sqlite3

            def _clear_summary(conn: sqlite3.Connection) -> None:
                conn.execute(
                    'UPDATE context_entries SET summary = NULL WHERE id = ?',
                    (context_id,),
                )

            await repos.context.backend.execute_write(_clear_summary)

        # Now store duplicate - should detect duplicate but find no summary, so generate one
        mock_provider2 = MagicMock()
        mock_provider2.summarize = AsyncMock(return_value='Summary for missing case')

        with (
            patch('app.tools.context.get_summary_provider', return_value=mock_provider2),
            patch('app.tools.context.get_embedding_provider', return_value=None),
            patch('app.tools.context.compute_summary_total_timeout', return_value=1.0),
        ):
            second_result = await store_context(
                thread_id='dedup-no-summary-thread',
                source='agent',
                text=dedup_text,
            )

        assert second_result['context_id'] == context_id
        assert '(summary generated)' in second_result['message']
        mock_provider2.summarize.assert_awaited_once_with(dedup_text, 'agent')

        rows = await repos.context.get_by_ids([context_id])
        assert rows[0]['summary'] == 'Summary for missing case'


class TestSummarySearchDisplay:
    """Tests for unified search display formatting in semantic, FTS, and hybrid search tools."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('mock_server_dependencies')
    async def test_semantic_search_shows_truncated_text_and_summary(self) -> None:
        """Show truncated text_content and summary as separate fields in semantic search."""
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed_query = AsyncMock(return_value=[0.1] * 768)

        mock_repos = MagicMock()
        mock_repos.embeddings.search = AsyncMock(return_value=([
            {
                'id': 1,
                'thread_id': 'sem-summary-thread',
                'source': 'agent',
                'content_type': 'text',
                'text_content': 'Full text content that is very long',
                'metadata': None,
                'summary': 'Concise semantic summary',
                'created_at': '2026-01-01T00:00:00',
                'updated_at': '2026-01-01T00:00:00',
                'distance': 0.3,
                'matched_chunk_start': None,
                'matched_chunk_end': None,
            },
        ], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 1}))
        mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
        mock_repos.images.get_images_for_context = AsyncMock(return_value=[])

        with (
            patch('app.tools.search.ensure_repositories', new=AsyncMock(return_value=mock_repos)),
            patch('app.startup._embedding_provider', mock_embedding_provider),
            patch('app.tools.search.settings') as mock_settings,
        ):
            mock_settings.semantic_search.enabled = True
            mock_settings.embedding.model = 'test-model'
            mock_settings.search.truncation_length = 150

            result = await search_tools.semantic_search_context(
                query='test query',
                limit=10,
            )

        assert len(result['results']) == 1
        entry = result['results'][0]
        # text_content is truncated original, not summary
        assert entry['text_content'] != 'Concise semantic summary'
        assert entry['is_text_content_truncated'] is False  # short enough, not truncated
        assert entry['summary'] == 'Concise semantic summary'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('mock_server_dependencies')
    async def test_fts_search_shows_truncated_text_and_summary(self) -> None:
        """Show truncated text_content and summary as separate fields in FTS search."""
        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=True)
        mock_repos.fts.search = AsyncMock(return_value=([
            {
                'id': 2,
                'thread_id': 'fts-summary-thread',
                'source': 'user',
                'content_type': 'text',
                'text_content': 'Full text content for FTS indexing',
                'metadata': None,
                'summary': 'Concise FTS summary',
                'created_at': '2026-01-01T00:00:00',
                'updated_at': '2026-01-01T00:00:00',
                'score': 5.0,
                'highlighted': None,
            },
        ], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 1}))
        mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
        mock_repos.images.get_images_for_context = AsyncMock(return_value=[])

        mock_fts_status = MagicMock()
        mock_fts_status.in_progress = False

        with (
            patch('app.tools.search.ensure_repositories', new=AsyncMock(return_value=mock_repos)),
            patch('app.tools.search.settings') as mock_settings,
            patch('app.tools.search.get_fts_migration_status', return_value=mock_fts_status),
        ):
            mock_settings.fts.enabled = True
            mock_settings.fts.language = 'english'
            mock_settings.reranking.enabled = False
            mock_settings.search.truncation_length = 150

            result = await search_tools.fts_search_context(
                query='test query',
                limit=10,
            )

        assert len(result['results']) == 1
        entry = result['results'][0]
        # text_content is truncated original, not summary
        assert entry['text_content'] != 'Concise FTS summary'
        assert entry['is_text_content_truncated'] is False  # short enough, not truncated
        assert entry['summary'] == 'Concise FTS summary'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('mock_server_dependencies')
    async def test_hybrid_search_shows_truncated_text_and_summary(self) -> None:
        """Show truncated text_content and summary as separate fields in hybrid search."""
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed_query = AsyncMock(return_value=[0.1] * 768)

        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=True)
        # FTS returns results with summary
        mock_repos.fts.search = AsyncMock(return_value=([
            {
                'id': 3,
                'thread_id': 'hybrid-summary-thread',
                'source': 'agent',
                'content_type': 'text',
                'text_content': 'Full text content for hybrid search',
                'metadata': None,
                'summary': 'Concise hybrid summary',
                'created_at': '2026-01-01T00:00:00',
                'updated_at': '2026-01-01T00:00:00',
                'score': 5.0,
                'highlighted': None,
            },
        ], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 1}))
        # Semantic returns same entry
        mock_repos.embeddings.search = AsyncMock(return_value=([
            {
                'id': 3,
                'thread_id': 'hybrid-summary-thread',
                'source': 'agent',
                'content_type': 'text',
                'text_content': 'Full text content for hybrid search',
                'metadata': None,
                'summary': 'Concise hybrid summary',
                'created_at': '2026-01-01T00:00:00',
                'updated_at': '2026-01-01T00:00:00',
                'distance': 0.3,
                'matched_chunk_start': None,
                'matched_chunk_end': None,
            },
        ], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 1}))
        mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])
        mock_repos.images.get_images_for_context = AsyncMock(return_value=[])

        mock_fts_status = MagicMock()
        mock_fts_status.in_progress = False

        with (
            patch('app.tools.search.ensure_repositories', new=AsyncMock(return_value=mock_repos)),
            patch('app.startup._embedding_provider', mock_embedding_provider),
            patch('app.tools.search.settings') as mock_settings,
            patch('app.tools.search.get_fts_migration_status', return_value=mock_fts_status),
        ):
            mock_settings.hybrid_search.enabled = True
            mock_settings.hybrid_search.rrf_k = 60
            mock_settings.hybrid_search.rrf_overfetch = 2
            mock_settings.hybrid_search.fts_or_threshold = 4
            mock_settings.fts.enabled = True
            mock_settings.fts.language = 'english'
            mock_settings.semantic_search.enabled = True
            mock_settings.embedding.model = 'test-model'
            mock_settings.embedding.provider = 'ollama'
            mock_settings.reranking.enabled = False
            mock_settings.storage.backend_type = 'sqlite'
            mock_settings.search.truncation_length = 150

            result = await search_tools.hybrid_search_context(
                query='test query',
                limit=10,
            )

        assert len(result['results']) == 1
        entry = result['results'][0]
        # text_content is truncated original, not summary
        assert entry['text_content'] != 'Concise hybrid summary'
        assert entry['is_text_content_truncated'] is False  # short enough, not truncated
        assert entry['summary'] == 'Concise hybrid summary'


class TestSummaryLifespan:
    """Tests for summary provider initialization and shutdown in server lifespan."""

    @pytest.mark.asyncio
    async def test_lifespan_initializes_and_shuts_down_summary_provider(self) -> None:
        """Initialize the summary provider on startup and shut it down on exit."""
        from app.server import lifespan

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.shutdown = AsyncMock()
        mock_backend.backend_type = 'sqlite'

        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)

        mock_summary_provider = MagicMock()
        mock_summary_provider.initialize = AsyncMock()
        mock_summary_provider.shutdown = AsyncMock()
        mock_summary_provider.is_available = AsyncMock(return_value=True)
        mock_summary_provider.provider_name = 'test-summary-provider'

        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = False
        mock_settings.reranking.enabled = False
        mock_settings.chunking.enabled = False
        mock_settings.semantic_search.enabled = False
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        mock_settings.summary.generation_enabled = True
        mock_settings.summary.provider = 'ollama'

        original_backend = app.startup.get_backend()
        original_repos = app.startup.get_repositories()
        original_embedding_provider = app.startup.get_embedding_provider()
        original_reranking_provider = app.startup.get_reranking_provider()
        original_chunking_service = app.startup.get_chunking_service()
        original_summary_provider = app.startup.get_summary_provider()

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch('app.server.init_database', new=AsyncMock()),
                patch('app.server.handle_metadata_indexes', new=AsyncMock()),
                patch('app.server.apply_semantic_search_migration', new=AsyncMock()),
                patch('app.server.apply_jsonb_merge_patch_migration', new=AsyncMock()),
                patch('app.server.apply_function_search_path_migration', new=AsyncMock()),
                patch('app.server.apply_fts_migration', new=AsyncMock()),
                patch('app.server.apply_chunking_migration', new=AsyncMock()),
                patch('app.server.apply_summary_migration', new=AsyncMock()),
                patch('app.server.apply_content_hash_migration', new=AsyncMock()),
                patch('app.server.register_tool', return_value=True),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
                patch(
                    'app.migrations.check_summary_provider_dependencies',
                    new=AsyncMock(return_value={'available': True, 'reason': None}),
                ),
                patch('app.summary.create_summary_provider', return_value=mock_summary_provider),
            ):
                mock_mcp = MagicMock()
                mock_mcp.list_tools = AsyncMock(return_value=[])

                async with lifespan(mock_mcp):
                    assert app.startup.get_summary_provider() is mock_summary_provider

                mock_summary_provider.initialize.assert_awaited_once()
                mock_summary_provider.shutdown.assert_awaited_once()
                assert app.startup.get_summary_provider() is None
        finally:
            set_backend(original_backend)
            set_repositories(original_repos)
            set_embedding_provider(original_embedding_provider)
            set_reranking_provider(original_reranking_provider)
            set_chunking_service(original_chunking_service)
            set_summary_provider(original_summary_provider)

    @pytest.mark.asyncio
    async def test_lifespan_summary_disabled(self) -> None:
        """Set summary provider to None when summary generation is disabled."""
        from app.server import lifespan

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.shutdown = AsyncMock()
        mock_backend.backend_type = 'sqlite'

        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)

        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = False
        mock_settings.reranking.enabled = False
        mock_settings.chunking.enabled = False
        mock_settings.semantic_search.enabled = False
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        mock_settings.summary.generation_enabled = False

        original_backend = app.startup.get_backend()
        original_repos = app.startup.get_repositories()
        original_embedding_provider = app.startup.get_embedding_provider()
        original_reranking_provider = app.startup.get_reranking_provider()
        original_chunking_service = app.startup.get_chunking_service()
        original_summary_provider = app.startup.get_summary_provider()

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch('app.server.init_database', new=AsyncMock()),
                patch('app.server.handle_metadata_indexes', new=AsyncMock()),
                patch('app.server.apply_semantic_search_migration', new=AsyncMock()),
                patch('app.server.apply_jsonb_merge_patch_migration', new=AsyncMock()),
                patch('app.server.apply_function_search_path_migration', new=AsyncMock()),
                patch('app.server.apply_fts_migration', new=AsyncMock()),
                patch('app.server.apply_chunking_migration', new=AsyncMock()),
                patch('app.server.apply_summary_migration', new=AsyncMock()),
                patch('app.server.apply_content_hash_migration', new=AsyncMock()),
                patch('app.server.register_tool', return_value=True),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
            ):
                mock_mcp = MagicMock()
                mock_mcp.list_tools = AsyncMock(return_value=[])

                async with lifespan(mock_mcp):
                    assert app.startup.get_summary_provider() is None

                assert app.startup.get_summary_provider() is None
        finally:
            set_backend(original_backend)
            set_repositories(original_repos)
            set_embedding_provider(original_embedding_provider)
            set_reranking_provider(original_reranking_provider)
            set_chunking_service(original_chunking_service)
            set_summary_provider(original_summary_provider)
