"""Tests for SQLite FTS5 boolean-mode malformed-query graceful degradation.

Boolean mode forwards the user's raw query to FTS5 ``MATCH`` so native FTS5 boolean
syntax (AND/OR/NOT, parentheses, quoted phrases) works. It is the one SQLite mode that is
NOT pre-sanitized, so a malformed boolean query (unbalanced parentheses, a dangling
operator, a stray ':') used to raise ``sqlite3.OperationalError`` ("fts5: syntax error" /
"no such column") and leak the raw engine message to the client, while PostgreSQL's
``websearch_to_tsquery`` tolerates the same input. ``_search_sqlite`` now catches that
grammar error and degrades to the crash-safe sanitized term match, so a VALID boolean query
is unaffected while a malformed one returns best-effort results instead of erroring.
"""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import create_backend
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.fts_repository import _is_fts5_grammar_error

# A single document containing both 'error' and 'handling' so a malformed boolean query whose
# surviving terms still match degrades to a non-empty best-effort result set.
_DOC_TEXT = 'Structured error handling guide for python services'


@pytest_asyncio.fixture
async def fts_repos(tmp_path: Path) -> AsyncGenerator[RepositoryContainer, None]:
    """SQLite backend + RepositoryContainer with an FTS5 table and one seeded document.

    Yields:
        RepositoryContainer whose FtsRepository searches a populated FTS5 index.
    """
    from app.schemas import load_schema

    db_path = tmp_path / 'fts_boolean.db'
    migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_fts_sqlite.sql'
    fts_sql = migration_path.read_text().replace('{TOKENIZER}', 'unicode61')

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(load_schema('sqlite'))
        conn.executescript(fts_sql)
        conn.execute(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
            'VALUES (?, ?, ?, ?, ?)',
            (generate_id(), 'fts-bool', 'agent', 'text', _DOC_TEXT),
        )
        conn.commit()
    finally:
        conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    repos = RepositoryContainer(backend)
    try:
        yield repos
    finally:
        await backend.shutdown()


# Malformed boolean queries: each raises an FTS5 grammar error when sent raw to MATCH.
_MALFORMED_BOOLEAN = [
    'error AND (handling',  # unbalanced parenthesis
    '(error',  # leading unbalanced parenthesis
    'error)',  # trailing unbalanced parenthesis
    'error AND',  # dangling trailing operator
    'AND error',  # leading operator
    'error : handling',  # stray colon -> "no such column"
    'c++ error',  # bare special characters
    'error OR (',  # dangling operator + unbalanced parenthesis
]

# The subset whose surviving terms still match the seeded document after degradation.
_MALFORMED_BOOLEAN_WITH_RECALL = [
    'error AND (handling',
    '(error',
    'error AND',
    'AND error',
]


class TestFtsGrammarErrorClassification:
    """The grammar-error classifier distinguishes query errors from operational faults."""

    def test_grammar_errors_are_recognized(self) -> None:
        """FTS5 query-grammar messages classify as grammar errors (eligible for degradation)."""
        assert _is_fts5_grammar_error(sqlite3.OperationalError('fts5: syntax error near "("')) is True
        assert _is_fts5_grammar_error(sqlite3.OperationalError('no such column: error')) is True
        assert _is_fts5_grammar_error(sqlite3.OperationalError('unterminated string')) is True
        assert _is_fts5_grammar_error(sqlite3.OperationalError('malformed MATCH expression')) is True

    def test_operational_faults_are_not_grammar_errors(self) -> None:
        """A locked database or disk fault is NOT a grammar error and must propagate."""
        assert _is_fts5_grammar_error(sqlite3.OperationalError('database is locked')) is False
        assert _is_fts5_grammar_error(sqlite3.OperationalError('disk I/O error')) is False


class TestFtsBooleanMalformedDegradation:
    """Malformed boolean queries degrade gracefully on SQLite instead of raising."""

    @pytest.mark.asyncio
    async def test_valid_boolean_query_unaffected(self, fts_repos: RepositoryContainer) -> None:
        """A well-formed boolean query keeps its native FTS5 semantics (no degradation)."""
        results, stats = await fts_repos.fts.search(query='error AND handling', mode='boolean', limit=10)
        assert stats['backend'] == 'sqlite'
        assert len(results) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize('malformed', _MALFORMED_BOOLEAN)
    async def test_malformed_boolean_does_not_raise(
        self,
        fts_repos: RepositoryContainer,
        malformed: str,
    ) -> None:
        """A malformed boolean query returns a result set instead of raising a grammar error."""
        results, stats = await fts_repos.fts.search(query=malformed, mode='boolean', limit=10)
        assert isinstance(results, list)
        assert stats['backend'] == 'sqlite'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('malformed', _MALFORMED_BOOLEAN_WITH_RECALL)
    async def test_malformed_boolean_degrades_to_term_recall(
        self,
        fts_repos: RepositoryContainer,
        malformed: str,
    ) -> None:
        """A malformed boolean whose surviving terms still match returns best-effort results."""
        results, _ = await fts_repos.fts.search(query=malformed, mode='boolean', limit=10)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_malformed_boolean_explain_query_uses_degraded_query(
        self,
        fts_repos: RepositoryContainer,
    ) -> None:
        """Under explain_query the degraded path still returns a coherent stats/plan shape."""
        results, stats = await fts_repos.fts.search(
            query='error AND (handling',
            mode='boolean',
            limit=10,
            explain_query=True,
        )
        assert len(results) >= 1
        assert 'query_plan' in stats
