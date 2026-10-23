"""Unit tests for the FTS search mixin in app.repositories.fts_repository.search.

Covers the SQLite language warning, the PostgreSQL ts_headline subquery structure, a
pathological boolean query that must degrade without charging the circuit breaker, and the
visibility and ownership every match reports on both backends.
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
from app.backends.base import BackendAccess
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

    async def _capture_statement(
        self,
        repo: FtsRepository,
        mock_backend: MagicMock,
        *,
        highlight: bool = True,
        mode: Literal['match', 'prefix', 'phrase', 'boolean'] = 'match',
        query: str = 'test query',
        limit: int = 10,
        offset: int = 0,
    ) -> tuple[str, tuple[object, ...]]:
        """Execute _search_postgresql and capture the generated SQL with its arguments.

        Returns:
            The SQL query string passed to conn.fetch and the arguments bound to it.
        """
        from unittest.mock import AsyncMock

        captured: list[tuple[str, tuple[object, ...]]] = []

        async def mock_execute_read(func: Callable[[AsyncMock], Awaitable[object]], *, access_scope: BackendAccess) -> object:
            _ = access_scope
            mock_conn = AsyncMock()

            async def capture_fetch(sql: str, *args: object) -> list[object]:
                captured.append((sql, args))
                return []

            mock_conn.fetch = capture_fetch
            return await func(mock_conn)

        mock_backend.execute_read = mock_execute_read

        await repo._search_postgresql(
            query=query,
            mode=mode,
            limit=limit,
            offset=offset,
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

        assert len(captured) == 1
        return captured[0]

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
        sql, _args = await self._capture_statement(
            repo, mock_backend, highlight=highlight, mode=mode, query=query, limit=limit,
        )
        return sql

    @pytest.mark.asyncio
    async def test_rows_report_visibility_and_ownership(
        self, repo: FtsRepository, mock_backend: MagicMock,
    ) -> None:
        """Both selects carry the visibility and the ownership flag, whose bind follows the query's.

        The READ predicate binds $1-$3 and the query $4, so the ownership flag binds the
        caller at $5 and LIMIT and OFFSET follow at $6 and $7.
        """
        sql, args = await self._capture_statement(repo, mock_backend, highlight=True, limit=5, offset=2)

        outer_select = sql[sql.index('SELECT'):sql.index('FROM (')]
        inner_sql = sql[sql.index('FROM (') + len('FROM ('):sql.index(') sub')]
        assert 'sub.visibility' in outer_select
        assert 'sub.is_owner' in outer_select
        assert 'ce.visibility' in inner_sql
        assert 'CASE WHEN ce.owner_id = $5 THEN 1 ELSE 0 END AS is_owner' in inner_sql
        assert 'LIMIT $6 OFFSET $7' in inner_sql
        assert args[4:] == ('local', 5, 2)

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


class TestFtsSQLiteReadBack:
    """Each SQLite match reports its visibility and whether the caller owns it, on every page and both query builds.

    The ownership flag binds the caller ahead of every filter in statement text order, so a
    misplaced bind would shift the thread, source, query, LIMIT or OFFSET values.
    """

    _THREAD = 'fts-read-back'
    # (label, owner, visibility, keyword repeats, thread): more repeats rank higher.
    _ROWS: tuple[tuple[str, str, str, int, str], ...] = (
        ('bob_private', 'bob', 'private', 4, _THREAD),
        ('own_private', 'local', 'private', 3, _THREAD),
        ('bob_public', 'bob', 'public', 2, _THREAD),
        ('own_public', 'local', 'public', 1, _THREAD),
        ('elsewhere', 'bob', 'public', 5, 'fts-read-back-elsewhere'),
    )

    @pytest_asyncio.fixture
    async def seeded(self, tmp_path: Path) -> AsyncGenerator[tuple[RepositoryContainer, dict[str, str]], None]:
        """SQLite backend with an FTS5 index and the rows of :attr:`_ROWS`.

        Args:
            tmp_path: Per-test temporary directory holding the database file.

        Yields:
            The repositories and the id of every row keyed by its label.
        """
        from app.schemas import load_schema

        db_path = tmp_path / 'fts_read_back.db'
        migration_path = Path(__file__).parents[3] / 'app' / 'migrations' / 'add_fts_sqlite.sql'
        fts_sql = migration_path.read_text().replace('{TOKENIZER}', 'unicode61')
        ids = {label: generate_id() for label, *_rest in self._ROWS}

        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(load_schema('sqlite'))
            conn.executescript(fts_sql)
            for label, owner, visibility, repeats, thread_id in self._ROWS:
                conn.execute(
                    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (ids[label], thread_id, 'agent', 'text', ' '.join(['lattice'] * repeats), owner, visibility),
                )
            conn.commit()
        finally:
            conn.close()

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            yield RepositoryContainer(backend), ids
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('mode', 'query'),
        [('match', 'lattice'), ('boolean', '(' * 100 + 'lattice' + ')' * 100)],
        ids=['match', 'degraded-boolean'],
    )
    async def test_rows_report_visibility_and_ownership(
        self,
        seeded: tuple[RepositoryContainer, dict[str, str]],
        mode: Literal['match', 'boolean'],
        query: str,
    ) -> None:
        """A filtered page reads back its rows' visibility and the caller's ownership, the degraded boolean query too."""
        repos, ids = seeded

        rows, _stats = await repos.fts.search(
            query=query, mode=mode, thread_id=self._THREAD, source='agent', limit=2, offset=1, scope=LOCAL_SCOPE,
        )

        assert [(row['id'], row['visibility'], row['is_owner']) for row in rows] == [
            (ids['bob_public'], 'public', 0),
            (ids['own_public'], 'public', 1),
        ]
