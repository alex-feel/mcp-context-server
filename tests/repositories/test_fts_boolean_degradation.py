"""Tests for FTS malformed-query handling: degradation, NUL rejection, and breaker safety.

Boolean mode forwards the user's raw query to FTS5 ``MATCH`` so native FTS5 boolean
syntax (AND/OR/NOT, parentheses, quoted phrases) works. It is the one SQLite mode that is
NOT pre-sanitized, so a malformed boolean query (unbalanced parentheses, a dangling
operator, a stray ':') used to raise ``sqlite3.OperationalError`` ("fts5: syntax error" /
"no such column") and leak the raw engine message to the client, while PostgreSQL's
``websearch_to_tsquery`` tolerates the same input. ``_search_sqlite`` now catches that
grammar error and degrades to the crash-safe sanitized term match (with the configured
language, so the operator-bareword drop keeps cross-backend parity), so a VALID boolean
query is unaffected while a malformed one returns best-effort results instead of erroring.

An embedded NUL is the one input token-quoting provably cannot neutralize (FTS5 reads the
bound query as a NUL-terminated C string, and asyncpg rejects NUL-carrying text bind
parameters), so the shared ``search()`` boundary rejects it with ``FtsValidationError`` on
both backends. Because ``FtsValidationError`` subclasses ``ControlFlowError``, none of
these client-input failures participates in circuit-breaker failure accounting -- a client
repeatedly sending an unparseable query or an invalid metadata filter can no longer open
the breaker into a process-wide outage.
"""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal

import pytest
import pytest_asyncio

from app.backends import create_backend
from app.backends.sqlite_backend import SQLiteBackend
from app.errors import ControlFlowError
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.fts_repository import FtsValidationError
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
    '*error',  # leading '*' -> FTS5 "unknown special query"
    '* error handling',  # leading bare '*' special-query token
    'NEAR(error handling, x)',  # NEAR distance argument not an integer -> "expected integer"
    'NEAR(python services, abc)',  # another non-integer NEAR distance argument
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
        assert _is_fts5_grammar_error(sqlite3.OperationalError('unknown special query: error')) is True
        assert _is_fts5_grammar_error(sqlite3.OperationalError('expected integer, got "x"')) is True

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


def _sqlite_breaker_failures(repos: RepositoryContainer) -> int:
    """Return the SQLite backend's current circuit-breaker failure count.

    Args:
        repos: Repository container whose backend is the SQLite backend under test.

    Returns:
        The circuit breaker's consecutive-failure count.
    """
    backend = repos.fts.backend
    assert isinstance(backend, SQLiteBackend)
    return backend.circuit_breaker.failures


class TestBooleanDegradationLanguageParity:
    """The boolean-degradation fallback honors the configured FTS language.

    PostgreSQL keeps and/or/not as required lexemes for every text-search config except
    english/hindi/russian, so the degraded SQLite term match must keep them as literal
    terms for those languages too, or the identical malformed boolean query returns
    different result sets on the two backends.
    """

    @pytest.mark.asyncio
    async def test_degraded_fallback_drops_operator_barewords_for_english(
        self,
        fts_repos: RepositoryContainer,
    ) -> None:
        """Under English the degraded match transform drops the bare 'and' (stopword parity)."""
        results, _ = await fts_repos.fts.search(
            query='error and (handling',
            mode='boolean',
            limit=10,
            language='english',
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_degraded_fallback_keeps_operator_barewords_for_german(
        self,
        fts_repos: RepositoryContainer,
    ) -> None:
        """Under German the degraded match transform keeps 'and' as a required literal term.

        The seeded document does not contain the word 'and', so keeping it as a term
        yields zero rows -- matching PostgreSQL, whose german config keeps 'and' as a
        required lexeme for the same degraded query.
        """
        results, _ = await fts_repos.fts.search(
            query='error and (handling',
            mode='boolean',
            limit=10,
            language='german',
        )
        assert len(results) == 0


class TestFtsNulQueryRejection:
    """An embedded NUL is rejected at the shared search() boundary on every mode."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mode', ['match', 'prefix', 'phrase', 'boolean'])
    async def test_nul_query_raises_structured_validation_error(
        self,
        fts_repos: RepositoryContainer,
        mode: Literal['match', 'prefix', 'phrase', 'boolean'],
    ) -> None:
        """A NUL-carrying query raises FtsValidationError instead of an engine error."""
        with pytest.raises(FtsValidationError) as excinfo:
            await fts_repos.fts.search(query='a\x00b', mode=mode, limit=10)
        assert any('NUL' in err for err in excinfo.value.validation_errors)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mode', ['match', 'prefix', 'phrase', 'boolean'])
    async def test_unpaired_surrogate_query_raises_structured_validation_error(
        self,
        fts_repos: RepositoryContainer,
        mode: Literal['match', 'prefix', 'phrase', 'boolean'],
    ) -> None:
        """A lone-surrogate query is rejected at the boundary, not left to abort the bind.

        A bare '\\x00' scan misses an unpaired UTF-16 surrogate, which is not
        UTF-8-encodable, so it would pass the old check and abort the PostgreSQL wire
        bind (a non-ControlFlowError charged to the circuit breaker) while SQLite ran it.
        The shared pg_bind_reject_reason probe catches it on both backends.
        """
        with pytest.raises(FtsValidationError) as excinfo:
            await fts_repos.fts.search(query='a\ud800b', mode=mode, limit=10)
        assert any('surrogate' in err for err in excinfo.value.validation_errors)

    def test_fts_validation_error_is_control_flow(self) -> None:
        """FtsValidationError is a ControlFlowError so it never trips the circuit breaker."""
        assert issubclass(FtsValidationError, ControlFlowError)

    @pytest.mark.asyncio
    async def test_repeated_nul_queries_do_not_open_the_breaker(
        self,
        fts_repos: RepositoryContainer,
    ) -> None:
        """Sending NUL queries past the failure threshold leaves the breaker closed.

        Before the boundary rejection, ten such calls opened the shared circuit breaker
        and every subsequent tool call on the backend failed for the breaker window.
        """
        for _ in range(12):
            with pytest.raises(FtsValidationError):
                await fts_repos.fts.search(query='\x00', mode='match', limit=10)
        assert _sqlite_breaker_failures(fts_repos) == 0
        results, _ = await fts_repos.fts.search(query='error', mode='match', limit=10)
        assert len(results) == 1


class TestClientInputErrorsBypassBreaker:
    """Client-input failures raised inside the read path never count as breaker failures."""

    @pytest.mark.asyncio
    async def test_invalid_metadata_filter_does_not_trip_breaker(
        self,
        fts_repos: RepositoryContainer,
    ) -> None:
        """An invalid metadata filter raises FtsValidationError without breaker accounting.

        This error is raised INSIDE the read callable (within get_connection's scope), so
        it exercises the readonly-path ControlFlowError exemption end-to-end: before the
        exemption every such call incremented the breaker failure count.
        """
        for _ in range(12):
            with pytest.raises(FtsValidationError):
                await fts_repos.fts.search(
                    query='error',
                    mode='match',
                    limit=10,
                    metadata_filters=[{'key': 'x', 'operator': 'bogus-operator', 'value': 1}],
                )
        assert _sqlite_breaker_failures(fts_repos) == 0

    @pytest.mark.asyncio
    async def test_non_boolean_grammar_error_becomes_validation_error(
        self,
        fts_repos: RepositoryContainer,
    ) -> None:
        """A grammar error in a sanitized mode classifies as a client validation error.

        Calls the SQLite search implementation directly (below the boundary NUL check) so
        the NUL reaches FTS5 and raises its grammar error; the handler must convert it to
        FtsValidationError (no raw engine message leak, no breaker failure) instead of
        propagating sqlite3.OperationalError.
        """
        with pytest.raises(FtsValidationError):
            await fts_repos.fts._search_sqlite(
                query='a\x00b',
                mode='match',
                limit=10,
                offset=0,
                thread_id=None,
                source=None,
                content_type=None,
                tags=None,
                start_date=None,
                end_date=None,
                metadata=None,
                metadata_filters=None,
                highlight=False,
            )
        assert _sqlite_breaker_failures(fts_repos) == 0

    def test_metadata_filter_validation_error_is_control_flow(self) -> None:
        """MetadataFilterValidationError subclasses ControlFlowError, like its FTS sibling.

        It is raised inside the embedding repository's read callables (both backends,
        both compression modes), so without this parentage an invalid metadata_filters
        on semantic/hybrid search would be charged to the circuit breaker and ten such
        calls would open it into a process-wide outage. This pins the exemption parity.
        """
        from app.repositories.embedding_repository import MetadataFilterValidationError

        assert issubclass(MetadataFilterValidationError, ControlFlowError)
