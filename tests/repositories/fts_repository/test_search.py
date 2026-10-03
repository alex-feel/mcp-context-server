"""Unit tests for the FTS search mixin in app.repositories.fts_repository.search.

Covers the SQLite language warning, the PostgreSQL ts_headline subquery structure, and a
pathological boolean query that must degrade without charging the circuit breaker.
"""

import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from app.backends import create_backend
from app.backends.sqlite_backend import SQLiteBackend
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.fts_repository import FtsRepository
from tests.helpers import LOCAL_SCOPE


class TestFtsSQLiteLanguageWarning:
    """Test that SQLite backend logs warning for non-English language parameter."""

    @pytest.fixture
    def mock_sqlite_backend(self) -> MagicMock:
        """Create a mock SQLite backend for testing."""
        from unittest.mock import AsyncMock

        backend = MagicMock()
        backend.backend_type = 'sqlite'
        # Mock execute_read as AsyncMock returning empty list
        backend.execute_read = AsyncMock(return_value=[])
        return backend

    @pytest.fixture
    def repo_sqlite(self, mock_sqlite_backend: MagicMock) -> FtsRepository:
        """Create a repository with mock SQLite backend."""
        return FtsRepository(mock_sqlite_backend)

    @pytest.mark.asyncio
    async def test_no_warning_for_english_language(
        self,
        repo_sqlite: FtsRepository,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that no warning is logged when language is 'english' (default)."""
        import logging

        with caplog.at_level(logging.WARNING):
            await repo_sqlite.search('test', language='english', scope=LOCAL_SCOPE)

        # No warning should be logged for English
        assert 'SQLite FTS5 does not support language-specific stemming' not in caplog.text

    @pytest.mark.asyncio
    async def test_warning_logged_for_non_english_language(
        self,
        repo_sqlite: FtsRepository,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that warning is logged when non-English language is requested with SQLite."""
        import logging

        with caplog.at_level(logging.WARNING):
            await repo_sqlite.search('test', language='german', scope=LOCAL_SCOPE)

        # Warning should be logged for non-English language
        assert 'SQLite FTS5 does not support language-specific stemming' in caplog.text
        assert 'german' in caplog.text
        assert 'unicode61' in caplog.text

    @pytest.mark.asyncio
    async def test_warning_logged_for_french_language(
        self,
        repo_sqlite: FtsRepository,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning for French language parameter."""
        import logging

        with caplog.at_level(logging.WARNING):
            await repo_sqlite.search('recherche', language='french', scope=LOCAL_SCOPE)

        assert 'SQLite FTS5 does not support language-specific stemming' in caplog.text
        assert 'french' in caplog.text


class TestPostgresqlSubqueryStructure:
    """Tests for the PostgreSQL ts_headline subquery optimization.

    These tests verify that _search_postgresql generates a subquery-structured
    SQL query where ts_headline is applied only to LIMIT'd results, not to
    all matching rows.
    """

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock PostgreSQL backend."""
        backend = MagicMock()
        backend.backend_type = 'postgresql'
        return backend

    @pytest.fixture
    def repo(self, mock_backend: MagicMock) -> FtsRepository:
        """Create FtsRepository with mock PostgreSQL backend."""
        return FtsRepository(mock_backend)

    async def _capture_sql(
        self,
        repo: FtsRepository,
        mock_backend: MagicMock,
        *,
        highlight: bool = True,
        mode: Literal['match', 'prefix', 'phrase', 'boolean'] = 'match',
        query: str = 'test query',
        limit: int = 10,
    ) -> str:
        """Execute _search_postgresql and capture the generated SQL.

        Returns:
            The SQL query string passed to conn.fetch.
        """
        from unittest.mock import AsyncMock

        captured_sql: list[str] = []

        async def mock_execute_read(func: Callable[[AsyncMock], Awaitable[object]]) -> object:
            mock_conn = AsyncMock()

            async def capture_fetch(sql: str, *_args: object) -> list[object]:
                captured_sql.append(sql)
                return []

            mock_conn.fetch = capture_fetch
            return await func(mock_conn)

        mock_backend.execute_read = mock_execute_read

        await repo._search_postgresql(
            query=query,
            mode=mode,
            limit=limit,
            offset=0,
            thread_id=None,
            source=None,
            content_type=None,
            tags=None,
            start_date=None,
            end_date=None,
            metadata=None,
            metadata_filters=None,
            highlight=highlight,
            language='english',
            explain_query=False,
            scope=LOCAL_SCOPE,
        )

        assert len(captured_sql) == 1
        return captured_sql[0]

    @pytest.mark.asyncio
    async def test_highlight_true_uses_subquery(
        self, repo: FtsRepository, mock_backend: MagicMock,
    ) -> None:
        """Verify ts_headline is in outer query, not inner subquery."""
        sql = await self._capture_sql(repo, mock_backend, highlight=True)

        # Verify subquery structure
        assert 'FROM (' in sql, 'SQL must contain inline subquery'
        assert ') sub' in sql, 'Subquery must be aliased as sub'

        # Verify ts_headline references sub.text_content (outer query)
        assert 'sub.text_content' in sql, 'ts_headline must reference sub.text_content'

        # Verify inner subquery contains ranking and filtering
        assert 'ts_rank_cd(ce.text_search_vector' in sql
        assert 'ce.text_search_vector @@' in sql

        # Extract the inner subquery (between 'FROM (' and ') sub')
        from_paren_idx = sql.index('FROM (') + len('FROM (')
        sub_end_idx = sql.index(') sub')
        inner_sql = sql[from_paren_idx:sub_end_idx]

        # Verify LIMIT/OFFSET are in the inner subquery
        assert 'LIMIT' in inner_sql, 'LIMIT must be in inner subquery'
        assert 'OFFSET' in inner_sql, 'OFFSET must be in inner subquery'

        # Verify ts_headline is NOT in the inner subquery
        assert 'ts_headline' not in inner_sql, 'ts_headline must NOT be in inner subquery'

        # Verify ts_headline is in the outer SELECT (above FROM ()
        outer_select = sql[:sql.index('FROM (')]
        assert 'ts_headline' in outer_select, 'ts_headline must be in outer SELECT'

    @pytest.mark.asyncio
    async def test_highlight_false_uses_subquery_with_null(
        self, repo: FtsRepository, mock_backend: MagicMock,
    ) -> None:
        """Verify NULL as highlighted in outer query when highlight=False."""
        sql = await self._capture_sql(repo, mock_backend, highlight=False)

        # Verify subquery structure still used
        assert 'FROM (' in sql, 'SQL must contain inline subquery'
        assert ') sub' in sql, 'Subquery must be aliased as sub'

        # Verify NULL as highlighted (no ts_headline)
        assert 'NULL as highlighted' in sql
        assert 'ts_headline' not in sql, 'ts_headline must NOT appear when highlight=False'

        # Verify LIMIT/OFFSET are in the inner subquery
        from_paren_idx = sql.index('FROM (') + len('FROM (')
        sub_end_idx = sql.index(') sub')
        inner_sql = sql[from_paren_idx:sub_end_idx]
        assert 'LIMIT' in inner_sql
        assert 'OFFSET' in inner_sql

    @pytest.mark.asyncio
    async def test_outer_query_orders_by_score_desc(
        self, repo: FtsRepository, mock_backend: MagicMock,
    ) -> None:
        """The PG outer query has an explicit top-level ORDER BY sub.score DESC.

        Regression: an inner-subquery ORDER BY does NOT constrain the enclosing
        query's output order (SQL standard), so without an outer ORDER BY the PG
        FTS results could come back out of best-first order -- diverging from SQLite
        (guaranteed score-DESC) and corrupting hybrid RRF, which ranks by list
        position.
        """
        for highlight in (True, False):
            sql = await self._capture_sql(repo, mock_backend, highlight=highlight)
            outer_tail = sql[sql.index(') sub'):]
            assert 'ORDER BY sub.score DESC' in outer_tail, (
                f'outer query must order by score DESC (highlight={highlight}): {outer_tail}'
            )

    @pytest.mark.asyncio
    async def test_subquery_preserves_column_order(
        self, repo: FtsRepository, mock_backend: MagicMock,
    ) -> None:
        """Verify outer SELECT maintains the expected column order."""
        sql = await self._capture_sql(repo, mock_backend, highlight=True, limit=5)

        # Extract outer SELECT columns (before FROM ()
        outer_select = sql[sql.index('SELECT'):sql.index('FROM (')]
        expected_columns = [
            'sub.id', 'sub.thread_id', 'sub.source', 'sub.content_type',
            'sub.text_content', 'sub.metadata', 'sub.created_at', 'sub.updated_at',
            'sub.score',
        ]
        for col in expected_columns:
            assert col in outer_select, f'Outer SELECT must contain {col}'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mode', ['match', 'prefix', 'phrase', 'boolean'])
    async def test_all_modes_use_subquery(
        self,
        repo: FtsRepository,
        mock_backend: MagicMock,
        mode: Literal['match', 'prefix', 'phrase', 'boolean'],
    ) -> None:
        """Verify all FTS modes produce subquery-structured SQL."""
        sql = await self._capture_sql(repo, mock_backend, mode=mode)

        assert 'FROM (' in sql, f'Mode {mode} must use subquery structure'
        assert ') sub' in sql, f'Mode {mode} must alias subquery as sub'


class TestFtsBooleanPathologicalQueryKeepsBreakerClosed:
    """A malformed boolean query never charges the process-global circuit breaker.

    The SQLite breaker counts failures across every caller and only decrements one per
    success, so a query the server mistakes for a database fault can be repeated until the
    breaker opens and rejects EVERY client's reads and writes for the recovery timeout.
    Deeply nested parentheses in boolean mode -- which PostgreSQL's websearch_to_tsquery
    accepts without error, making this a cross-backend divergence too -- were such a query.
    """

    _DOC_TEXT = 'Structured error handling guide for python services'

    @pytest_asyncio.fixture
    async def fts_repos(
        self,
        tmp_path: Path,
    ) -> AsyncGenerator[tuple[RepositoryContainer, SQLiteBackend], None]:
        """SQLite backend with an FTS5 index and one seeded document.

        Args:
            tmp_path: Per-test temporary directory holding the database file.

        Yields:
            Tuple of (RepositoryContainer, backend) so a test can inspect breaker state.
        """
        from app.schemas import load_schema

        db_path = tmp_path / 'fts_classification.db'
        migration_path = Path(__file__).parents[3] / 'app' / 'migrations' / 'add_fts_sqlite.sql'
        fts_sql = migration_path.read_text().replace('{TOKENIZER}', 'unicode61')

        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(load_schema('sqlite'))
            conn.executescript(fts_sql)
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
                "VALUES (?, ?, ?, ?, ?, 'local')",
                (generate_id(), 'fts-classification', 'agent', 'text', self._DOC_TEXT),
            )
            conn.commit()
        finally:
            conn.close()

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        assert isinstance(backend, SQLiteBackend)
        await backend.initialize()
        try:
            yield RepositoryContainer(backend), backend
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_deeply_nested_boolean_query_degrades_without_charging_breaker(
        self,
        fts_repos: tuple[RepositoryContainer, SQLiteBackend],
    ) -> None:
        """Repeating the pathological query returns results and leaves the breaker closed."""
        repos, backend = fts_repos
        pathological = '(' * 100 + 'error' + ')' * 100

        for _ in range(12):
            results, stats = await repos.fts.search(query=pathological, mode='boolean', limit=10, scope=LOCAL_SCOPE)
            assert stats['backend'] == 'sqlite'
            assert len(results) == 1

        assert backend.circuit_breaker.failures == 0
        assert backend.circuit_breaker.is_open() is False
