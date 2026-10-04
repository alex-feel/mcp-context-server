"""Tests for ContextRepository.search_contexts filters, pagination and query stats."""

import json
import re
from collections.abc import Awaitable
from collections.abc import Callable
from typing import cast
from unittest.mock import Mock

import pytest

from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessScope
from app.backends.base import StorageBackend
from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from tests.helpers import LOCAL_SCOPE


class TestContextRepositorySearch:
    """Test search functionality of ContextRepository."""

    @pytest.mark.asyncio
    async def test_search_empty_database(self, context_repo: ContextRepository) -> None:
        """Test searching empty database returns empty results."""
        rows, stats = await context_repo.search_contexts(scope=LOCAL_SCOPE)

        assert rows == []
        assert 'execution_time_ms' in stats

    @pytest.mark.asyncio
    async def test_search_by_thread_id(
        self,
        context_repo: ContextRepository,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching by thread_id."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='thread_a',
            source='user',
            content_type='text',
            text_content='Message A',
        )
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='thread_b',
            source='user',
            content_type='text',
            text_content='Message B',
        )

        rows, stats = await context_repo.search_contexts(thread_id='thread_a', scope=LOCAL_SCOPE)

        assert len(rows) == 1
        assert rows[0]['thread_id'] == 'thread_a'

    @pytest.mark.asyncio
    async def test_search_by_source(
        self,
        context_repo: ContextRepository,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching by source."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='source_thread',
            source='user',
            content_type='text',
            text_content='User message',
        )
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='source_thread',
            source='agent',
            content_type='text',
            text_content='Agent message',
        )

        rows, stats = await context_repo.search_contexts(source='agent', scope=LOCAL_SCOPE)

        assert len(rows) == 1
        assert rows[0]['source'] == 'agent'

    @pytest.mark.asyncio
    async def test_search_by_content_type(
        self,
        context_repo: ContextRepository,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching by content_type."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='type_thread',
            source='user',
            content_type='text',
            text_content='Text only',
        )
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='type_thread',
            source='user',
            content_type='multimodal',
            text_content='With image',
        )

        rows, stats = await context_repo.search_contexts(content_type='multimodal', scope=LOCAL_SCOPE)

        assert len(rows) == 1
        assert rows[0]['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    async def test_search_by_tags(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching by tags."""
        ctx_id1, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='tag_thread',
            source='user',
            content_type='text',
            text_content='Tagged 1',
        )
        ctx_id2, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='tag_thread',
            source='user',
            content_type='text',
            text_content='Tagged 2',
        )

        await repos.tags.store_tags(ctx_id1, ['important', 'review'])
        await repos.tags.store_tags(ctx_id2, ['other'])

        rows, stats = await repos.context.search_contexts(tags=['important'], scope=LOCAL_SCOPE)

        assert len(rows) == 1
        assert rows[0]['id'] == ctx_id1

    @pytest.mark.asyncio
    async def test_search_with_limit(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with limit parameter."""
        for i in range(10):
            await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='limit_thread',
                source='user',
                content_type='text',
                text_content=f'Message {i}',
            )

        rows, stats = await repos.context.search_contexts(
            thread_id='limit_thread',
            limit=5,
            scope=LOCAL_SCOPE,
        )

        assert len(rows) == 5

    @pytest.mark.asyncio
    async def test_search_with_offset(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with offset parameter."""
        for i in range(10):
            await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='offset_thread',
                source='user',
                content_type='text',
                text_content=f'Message {i}',
            )

        rows, stats = await repos.context.search_contexts(
            thread_id='offset_thread',
            limit=3,
            offset=5,
            scope=LOCAL_SCOPE,
        )

        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_search_with_metadata_simple(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with simple metadata filter."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='meta_thread',
            source='user',
            content_type='text',
            text_content='Priority 1',
            metadata=json.dumps({'priority': 1}),
        )
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='meta_thread',
            source='user',
            content_type='text',
            text_content='Priority 2',
            metadata=json.dumps({'priority': 2}),
        )

        rows, stats = await repos.context.search_contexts(
            thread_id='meta_thread',
            metadata={'priority': 1},
            scope=LOCAL_SCOPE,
        )

        assert len(rows) == 1
        # Parse metadata JSON and verify
        metadata = json.loads(rows[0]['metadata'])
        assert metadata['priority'] == 1

    @pytest.mark.asyncio
    async def test_search_with_explain_query(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with explain_query=True."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='explain_thread',
            source='user',
            content_type='text',
            text_content='Test entry',
        )

        rows, stats = await repos.context.search_contexts(
            thread_id='explain_thread',
            explain_query=True,
            scope=LOCAL_SCOPE,
        )

        assert len(rows) == 1
        assert 'execution_time_ms' in stats
        assert 'filters_applied' in stats

    @pytest.mark.asyncio
    async def test_search_combined_filters(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test searching with multiple filters combined."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='combo_thread',
            source='user',
            content_type='text',
            text_content='User text',
        )
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='combo_thread',
            source='agent',
            content_type='text',
            text_content='Agent text',
        )
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='other_thread',
            source='user',
            content_type='text',
            text_content='Other user',
        )

        rows, stats = await repos.context.search_contexts(
            thread_id='combo_thread',
            source='user',
            content_type='text',
            scope=LOCAL_SCOPE,
        )

        assert len(rows) == 1
        assert rows[0]['thread_id'] == 'combo_thread'
        assert rows[0]['source'] == 'user'

    @pytest.mark.asyncio
    async def test_filters_applied_counts_every_applied_condition(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """stats.filters_applied counts ALL applied filters, not only the metadata ones.

        The shared clause builder emits the thread/source/content_type, date-range and tag
        conditions itself, so reporting only the metadata count told a client using
        explain_query that its other filters had been dropped -- and disagreed with
        fts_search_context and semantic_search_context, which report the full tally for the
        identical arguments.
        """
        context_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='count_thread',
            source='agent',
            content_type='text',
            text_content='counted entry',
            metadata=json.dumps({'project': 'p'}),
        )
        await repos.tags.store_tags(context_id, ['x'])

        _rows, stats = await repos.context.search_contexts(
            thread_id='count_thread',
            source='agent',
            content_type='text',
            tags=['x'],
            metadata={'project': 'p'},
            start_date='2020-01-01',
            end_date='2999-01-01',
            scope=LOCAL_SCOPE,
        )

        # thread_id + source + content_type + tags + start_date + end_date + one metadata key.
        assert stats['filters_applied'] == 7

    @pytest.mark.asyncio
    async def test_filters_applied_is_zero_without_filters(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """An unfiltered browse still reports zero applied filters."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='no_filter_thread',
            source='user',
            content_type='text',
            text_content='entry',
        )

        _rows, stats = await repos.context.search_contexts(scope=LOCAL_SCOPE)

        assert stats['filters_applied'] == 0

    @pytest.mark.asyncio
    async def test_search_validation_error_stats_include_backend_sqlite(
        self,
        context_repo: ContextRepository,
    ) -> None:
        """The validation-error stats dict exposes the same 'backend' key as success stats.

        An invalid metadata filter short-circuits search_contexts into a structured
        validation-error stats dict; that dict must mirror the success-path stats
        shape (which always includes 'backend') so clients can rely on a uniform
        key set regardless of outcome.
        """
        rows, stats = await context_repo.search_contexts(
            metadata_filters=[{'key': 'status', 'operator': 'not_a_real_operator', 'value': 'x'}],
            scope=LOCAL_SCOPE,
        )

        assert rows == []
        assert stats['error'] == 'Metadata filter validation failed'
        assert stats['validation_errors']
        assert stats['execution_time_ms'] == 0.0
        assert stats['filters_applied'] == 0
        assert stats['rows_returned'] == 0
        assert stats['backend'] == 'sqlite'

    @pytest.mark.asyncio
    async def test_search_validation_error_stats_include_backend_postgresql(self) -> None:
        """The PostgreSQL validation-error stats dict also carries 'backend'.

        The PostgreSQL closure short-circuits on validation errors before touching
        the connection, so a recording stand-in connection suffices -- no live
        PostgreSQL needed.
        """
        pg_backend = Mock()
        pg_backend.backend_type = 'postgresql'

        async def _execute_read(
            closure: Callable[[object], Awaitable[tuple[list[object], dict[str, object]]]],
        ) -> tuple[list[object], dict[str, object]]:
            return await closure(object())

        pg_backend.execute_read = _execute_read
        repo_pg = ContextRepository(cast(StorageBackend, pg_backend))

        rows, stats = await repo_pg.search_contexts(
            metadata_filters=[{'key': 'status', 'operator': 'not_a_real_operator', 'value': 'x'}],
            scope=LOCAL_SCOPE,
        )

        assert rows == []
        assert stats['error'] == 'Metadata filter validation failed'
        assert stats['backend'] == 'postgresql'


def _recording_postgresql_repo(rows: list[object]) -> tuple[ContextRepository, list[tuple[str, tuple[object, ...]]]]:
    """Build a PostgreSQL repository whose connection records every fetched statement with its arguments.

    Args:
        rows: The rows every ``fetch`` returns.

    Returns:
        The repository and the list the recorded ``(statement, arguments)`` pairs are appended to.
    """
    statements: list[tuple[str, tuple[object, ...]]] = []

    class _RecordingConnection:
        async def fetch(self, query: str, *args: object) -> list[object]:
            statements.append((query, args))
            return rows

    pg_backend = Mock()
    pg_backend.backend_type = 'postgresql'

    async def _execute_read(closure: Callable[[object], Awaitable[object]]) -> object:
        return await closure(_RecordingConnection())

    pg_backend.execute_read = _execute_read
    return ContextRepository(cast(StorageBackend, pg_backend)), statements


def _bound(query: str, args: tuple[object, ...], pattern: str) -> object:
    """Return the argument bound to the single ``$n`` placeholder that ``pattern`` captures."""
    match = re.search(pattern, query)
    assert match is not None, f'{pattern!r} not in {query!r}'
    return args[int(match.group(1)) - 1]


class TestSearchContextsScope:
    """search_contexts applies the caller's read scope after every client filter."""

    @pytest.mark.asyncio
    async def test_filters_applied_does_not_count_the_scope(self, repos: RepositoryContainer) -> None:
        """The same client filters report the same count whatever scope runs them."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE, visibility='private', thread_id='scope_count', source='agent',
            content_type='text', text_content='scoped count entry', metadata=json.dumps({'project': 'p'}),
        )
        counts: list[int] = []
        for scope in (LOCAL_SCOPE, AccessScope('bob', frozenset({'team-x'})), SYSTEM_SCOPE):
            _rows, stats = await repos.context.search_contexts(
                thread_id='scope_count', source='agent', metadata={'project': 'p'}, scope=scope,
            )
            counts.append(stats['filters_applied'])

        assert counts == [3, 3, 3]

    @pytest.mark.asyncio
    async def test_postgresql_placeholders_continue_after_every_filter(self) -> None:
        """On PostgreSQL the predicate binds after the filters and before LIMIT and OFFSET, numbered contiguously."""
        repo, statements = _recording_postgresql_repo([])

        await repo.search_contexts(
            thread_id='t', source='agent', tags=['x', 'y'], metadata={'author': 'alice'},
            metadata_filters=[{'key': 'priority', 'operator': 'gt', 'value': 3}],
            scope=AccessScope('bob', frozenset({'team-x'})), limit=5, offset=2,
        )

        [(query, args)] = statements
        assert sorted({int(number) for number in re.findall(r'\$(\d+)', query)}) == list(range(1, len(args) + 1))
        assert query.index('FROM tags') < query.index('context_entries.owner_id')
        assert _bound(query, args, r'context_entries\.owner_id = \$(\d+)') == 'bob'
        assert _bound(query, args, r"principal_type = 'user' AND g\.principal_id = \$(\d+)") == 'bob'
        assert _bound(query, args, r'= ANY\(\$(\d+)::text\[\]\)') == ['team-x']
        assert _bound(query, args, r'LIMIT \$(\d+)') == 5
        assert _bound(query, args, r'OFFSET \$(\d+)') == 2

    @pytest.mark.asyncio
    async def test_system_scope_adds_no_condition(self) -> None:
        """The system scope's empty predicate leaves the statement and its binds as the filters build them."""
        repo, statements = _recording_postgresql_repo([])

        await repo.search_contexts(thread_id='t', scope=SYSTEM_SCOPE, limit=5, offset=0)

        [(query, args)] = statements
        assert 'owner_id' not in query
        assert args == ('t', 5, 0)

    @pytest.mark.asyncio
    async def test_invalid_filter_short_circuits_before_the_scope(self) -> None:
        """A validation error returns before any statement runs, whatever the scope."""
        repo, statements = _recording_postgresql_repo([])

        rows, stats = await repo.search_contexts(
            metadata_filters=[{'key': 'status', 'operator': 'not_a_real_operator', 'value': 'x'}],
            scope=AccessScope('bob', frozenset()),
        )

        assert rows == []
        assert stats['error'] == 'Metadata filter validation failed'
        assert statements == []

    @pytest.mark.asyncio
    async def test_sqlite_explain_runs_with_the_scope(self, repos: RepositoryContainer) -> None:
        """explain_query returns a plan for the scoped statement on SQLite."""
        await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE, visibility='private', thread_id='scope_explain', source='agent',
            content_type='text', text_content='scoped explain entry',
        )

        rows, stats = await repos.context.search_contexts(
            thread_id='scope_explain', explain_query=True, scope=AccessScope('bob', frozenset({'team-x'})),
        )

        assert rows == []
        assert 'context_entries' in stats['query_plan']
