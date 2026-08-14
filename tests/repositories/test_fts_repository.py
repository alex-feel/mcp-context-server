"""Unit tests for FtsRepository.

Tests the query transformation logic, the PostgreSQL tsquery function selection, and the
classification of a failed MATCH execution. Most cases need no database connection; the
classification cases run against a throwaway SQLite database so they observe the real
result codes and messages the engine produces.
"""

import sqlite3
from collections.abc import AsyncGenerator
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
from app.repositories.fts_repository import FtsValidationError
from app.repositories.fts_repository import _fts_relations_present
from app.repositories.fts_repository import _is_fts5_grammar_error
from app.repositories.fts_repository import _is_postgresql_query_failure


class TestFtsRepositoryQueryTransform:
    """Test query transformation for different modes."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock backend for testing."""
        backend = MagicMock()
        backend.backend_type = 'sqlite'
        return backend

    @pytest.fixture
    def repo(self, mock_backend: MagicMock) -> FtsRepository:
        """Create a repository with mock backend."""
        return FtsRepository(mock_backend)

    def test_transform_query_match_mode(self, repo: FtsRepository) -> None:
        """Test match mode query transformation - words joined with implicit AND."""
        result = repo._transform_query_sqlite('hello world', 'match')
        # Each term is wrapped as an FTS5 string literal (AND logic preserved, crash-safe).
        assert result == '"hello" "world"'

    def test_transform_query_match_mode_single_word(self, repo: FtsRepository) -> None:
        """Test match mode with a single word."""
        result = repo._transform_query_sqlite('python', 'match')
        assert result == '"python"'

    def test_transform_query_phrase_mode(self, repo: FtsRepository) -> None:
        """Test phrase mode query transformation - wrapped in double quotes."""
        result = repo._transform_query_sqlite('hello world', 'phrase')
        assert result == '"hello world"'

    def test_transform_query_phrase_mode_single_word(self, repo: FtsRepository) -> None:
        """Test phrase mode with a single word."""
        result = repo._transform_query_sqlite('python', 'phrase')
        assert result == '"python"'

    def test_transform_query_prefix_mode(self, repo: FtsRepository) -> None:
        """Test prefix mode query transformation - adds * to each word."""
        result = repo._transform_query_sqlite('hello world', 'prefix')
        assert result == '"hello"* "world"*'

    def test_transform_query_prefix_mode_single_word(self, repo: FtsRepository) -> None:
        """Test prefix mode with a single word."""
        result = repo._transform_query_sqlite('python', 'prefix')
        assert result == '"python"*'

    def test_transform_query_boolean_mode(self, repo: FtsRepository) -> None:
        """Test boolean mode query transformation - passthrough as-is."""
        result = repo._transform_query_sqlite('hello AND world', 'boolean')
        assert result == 'hello AND world'

    def test_transform_query_boolean_mode_complex(self, repo: FtsRepository) -> None:
        """Test boolean mode with complex boolean expression."""
        query = 'python AND (async OR await) NOT blocking'
        result = repo._transform_query_sqlite(query, 'boolean')
        assert result == query

    def test_transform_query_strips_whitespace(self, repo: FtsRepository) -> None:
        """Test that queries are stripped of leading/trailing whitespace."""
        result = repo._transform_query_sqlite('  hello world  ', 'match')
        assert result == '"hello" "world"'

    def test_transform_query_prefix_with_existing_wildcard(self, repo: FtsRepository) -> None:
        """Test prefix mode with existing wildcard does not double it."""
        result = repo._transform_query_sqlite('implement*', 'prefix')
        assert result == '"implement"*'

    def test_transform_query_prefix_with_double_wildcard(self, repo: FtsRepository) -> None:
        """Test prefix mode with double wildcard normalizes to single."""
        result = repo._transform_query_sqlite('test**', 'prefix')
        assert result == '"test"*'

    def test_transform_query_prefix_mixed_wildcards(self, repo: FtsRepository) -> None:
        """Test prefix mode with mixed wildcards in multiple words."""
        result = repo._transform_query_sqlite('hello* world', 'prefix')
        assert result == '"hello"* "world"*'

    def test_transform_query_prefix_all_wildcards(self, repo: FtsRepository) -> None:
        """Test prefix mode with all words already having wildcards."""
        result = repo._transform_query_sqlite('hello* world*', 'prefix')
        assert result == '"hello"* "world"*'

    def test_match_and_prefix_operator_input_runs_on_real_fts5(self, repo: FtsRepository) -> None:
        """The transformed match/prefix query for operator/special-char input must EXECUTE on a
        real SQLite FTS5 table without 'fts5: syntax error' (the standalone-tool crash class the
        shared sanitizer closes), while a normal query still matches."""

        db = sqlite3.connect(':memory:')
        db.execute("CREATE VIRTUAL TABLE d USING fts5(b, tokenize='porter unicode61')")
        db.execute("INSERT INTO d(b) VALUES('python async running cat')")
        for mode in ('match', 'prefix'):
            # Includes STANDALONE double-quote tokens ('"', 'cat " dog', 'python "'): a lone '"'
            # satisfies both startswith+endswith (same char), so an un-length-checked phrase guard
            # would emit an unterminated FTS5 string literal. It must run, not raise.
            for query in ['NOT cat', 'python (async)', 'foo:bar', 'cat OR', 'OR cat', 'a "x',
                          'AND OR NOT', '"', 'cat " dog', 'python "']:
                fts = repo._transform_query_sqlite(query, mode)
                # An all-operator/all-special input transforms to the '' match-nothing
                # sentinel; _search_sqlite short-circuits it (FTS5 rejects MATCH ''), so mirror
                # that guard. A non-empty transform must still EXECUTE without a syntax error.
                if fts:
                    db.execute('SELECT rowid FROM d WHERE d MATCH ?', (fts,)).fetchall()
        # A normal match query still finds the row (AND recall preserved).
        normal = repo._transform_query_sqlite('python async', 'match')
        assert db.execute('SELECT rowid FROM d WHERE d MATCH ?', (normal,)).fetchall() == [(1,)]

    def test_sanitize_drops_operator_barewords_only_for_stopword_languages(self) -> None:
        """and/or/not are dropped ONLY for languages PostgreSQL treats them as stopwords.

        PostgreSQL's plainto_tsquery drops and/or/not as stopwords for english, hindi, and russian
        (their ASCII words route through english_stem) but keeps them as required lexemes for every
        other language, so the SQLite sanitizer must mirror that per-language or the two backends
        return different rows for the same non-English query.
        """
        from app.repositories.fts_repository import sanitize_sqlite_fts_terms

        tokens = ['system', 'and', 'or', 'not', 'config']
        # english (default) and russian: operator barewords dropped, mirroring plainto_tsquery.
        assert sanitize_sqlite_fts_terms(tokens) == ['"system"', '"config"']
        assert sanitize_sqlite_fts_terms(tokens, 'russian') == ['"system"', '"config"']
        # german/simple: kept as literal terms, mirroring plainto_tsquery('german'/'simple', ...).
        kept = ['"system"', '"and"', '"or"', '"not"', '"config"']
        assert sanitize_sqlite_fts_terms(tokens, 'german') == kept
        assert sanitize_sqlite_fts_terms(tokens, 'simple') == kept

    def test_transform_match_keeps_operator_barewords_for_non_english_language(self, repo: FtsRepository) -> None:
        """match mode keeps and/or/not as literal terms for a non-stopword language.

        The default (english) drops them; a german deployment keeps them because
        plainto_tsquery('german', 'system and configuration') compiles all three lexemes, so SQLite
        must require them too for cross-backend parity.
        """
        assert repo._transform_query_sqlite('system and configuration', 'match') == '"system" "configuration"'
        assert (
            repo._transform_query_sqlite('system and configuration', 'match', 'german')
            == '"system" "and" "configuration"'
        )

    def test_transform_prefix_keeps_operator_barewords_for_non_english_language(self, repo: FtsRepository) -> None:
        """prefix mode applies the same per-language operator-bareword gate as match mode."""
        assert repo._transform_query_sqlite('and config', 'prefix') == '"config"*'
        assert repo._transform_query_sqlite('and config', 'prefix', 'german') == '"and"* "config"*'

    def test_transform_non_english_operator_terms_run_on_real_fts5(self, repo: FtsRepository) -> None:
        """The kept operator-bareword literal terms EXECUTE on a real FTS5 table without a syntax
        error and match a row containing the word (parity with PostgreSQL requiring the lexeme)."""

        db = sqlite3.connect(':memory:')
        db.execute("CREATE VIRTUAL TABLE d USING fts5(b, tokenize='unicode61')")
        db.execute("INSERT INTO d(b) VALUES('system and configuration')")
        fts = repo._transform_query_sqlite('system and configuration', 'match', 'german')
        assert fts == '"system" "and" "configuration"'
        assert db.execute('SELECT rowid FROM d WHERE d MATCH ?', (fts,)).fetchall() == [(1,)]


class TestFtsSQLiteEmbeddedQuoteSplitsIntoAndedTerms:
    """An embedded double quote must AND two terms, never impose phrase adjacency.

    FTS5 RE-TOKENIZES the content of a quoted string literal, so escaping an embedded
    double quote by doubling it (``"cat""dog"``) decodes back to a literal ``"`` that the
    tokenizer reads as a word boundary: the supposedly neutralized term becomes the strict
    adjacency phrase ``"cat dog"``. PostgreSQL's plainto_tsquery splits the same input into
    independently-ANDed lexemes, so the doubling form silently returned fewer rows on SQLite
    than on PostgreSQL for ordinary input such as ``12"TV``, ``don"t`` or a pasted
    ``KeyError: "foo"``. Splitting the token into separate literals restores parity.
    """

    @pytest.fixture
    def repo(self) -> FtsRepository:
        """Create a repository with a mock SQLite backend.

        Returns:
            FtsRepository bound to a backend that only reports its type.
        """
        backend = MagicMock()
        backend.backend_type = 'sqlite'
        return FtsRepository(backend)

    def test_sanitize_splits_embedded_quote_into_separate_literals(self) -> None:
        """The shared sanitizer emits one literal per quote-delimited fragment."""
        from app.repositories.fts_repository import sanitize_sqlite_fts_terms

        assert sanitize_sqlite_fts_terms(['cat"dog']) == ['"cat"', '"dog"']
        assert sanitize_sqlite_fts_terms(['don"t']) == ['"don"', '"t"']
        # A fragment that is an operator bareword is dropped for a stopword language exactly
        # like a standalone one, matching plainto_tsquery on the same input.
        assert sanitize_sqlite_fts_terms(['and"config']) == ['"config"']
        assert sanitize_sqlite_fts_terms(['and"config'], 'german') == ['"and"', '"config"']

    def test_transform_match_and_prefix_split_embedded_quote(self, repo: FtsRepository) -> None:
        """match and prefix modes apply the identical split, so the two never diverge."""
        assert repo._transform_query_sqlite('alpha"zulu', 'match') == '"alpha" "zulu"'
        assert repo._transform_query_sqlite('alpha"zulu', 'prefix') == '"alpha"* "zulu"*'

    def test_embedded_quote_matches_non_adjacent_document_on_real_fts5(self, repo: FtsRepository) -> None:
        """Both an adjacent and a non-adjacent document match, as on PostgreSQL."""

        db = sqlite3.connect(':memory:')
        db.execute("CREATE VIRTUAL TABLE d USING fts5(b, tokenize='porter unicode61')")
        db.execute("INSERT INTO d(b) VALUES('alpha zulu beta')")
        db.execute("INSERT INTO d(b) VALUES('alpha somewhere else entirely zulu')")

        fts = repo._transform_query_sqlite('alpha"zulu', 'match')
        rows = db.execute('SELECT rowid FROM d WHERE d MATCH ? ORDER BY rowid', (fts,)).fetchall()
        # Two AND-ed terms with no adjacency requirement: both documents qualify. The old
        # doubled-quote escape matched only the adjacent document.
        assert rows == [(1,), (2,)]

        # Control: the plain space-separated query behaves identically.
        plain = repo._transform_query_sqlite('alpha zulu', 'match')
        assert db.execute('SELECT rowid FROM d WHERE d MATCH ? ORDER BY rowid', (plain,)).fetchall() == [(1,), (2,)]

    def test_hyphen_stays_an_adjacency_phrase(self, repo: FtsRepository) -> None:
        """A hyphen keeps its adjacency phrase, the closest FTS5 form to the PG compound lexeme.

        PostgreSQL's plainto_tsquery emits the compound lexeme ``alpha-zulu`` plus its parts,
        which requires the two words adjacent in the document. FTS5's tokenizer drops hyphens
        entirely, so the document side cannot represent the compound at all; the adjacency
        phrase is the tightest expressible approximation and ANDing the parts separately would
        widen recall further away from PostgreSQL.
        """

        assert repo._transform_query_sqlite('alpha-zulu', 'match') == '"alpha zulu"'

        db = sqlite3.connect(':memory:')
        db.execute("CREATE VIRTUAL TABLE d USING fts5(b, tokenize='porter unicode61')")
        db.execute("INSERT INTO d(b) VALUES('alpha zulu beta')")
        db.execute("INSERT INTO d(b) VALUES('alpha somewhere else entirely zulu')")
        fts = repo._transform_query_sqlite('alpha-zulu', 'match')
        assert db.execute('SELECT rowid FROM d WHERE d MATCH ?', (fts,)).fetchall() == [(1,)]

    def test_quote_only_query_matches_nothing_without_syntax_error(self, repo: FtsRepository) -> None:
        """A query of nothing but quotes reduces to the match-nothing sentinel."""
        assert repo._transform_query_sqlite('"', 'match') == ''
        assert repo._transform_query_sqlite('"', 'prefix') == ''


class TestFtsRepositoryPostgreSQLQueryTransform:
    """Test PostgreSQL query transformation for different modes."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock backend for testing."""
        backend = MagicMock()
        backend.backend_type = 'postgresql'
        return backend

    @pytest.fixture
    def repo(self, mock_backend: MagicMock) -> FtsRepository:
        """Create a repository with mock backend."""
        return FtsRepository(mock_backend)

    def test_transform_query_prefix_mode(self, repo: FtsRepository) -> None:
        """Test prefix mode transforms to tsquery format with :* and & operator."""
        result = repo._transform_query_postgresql('hello world', 'prefix')
        assert result == 'hello:* & world:*'

    def test_transform_query_prefix_single_word(self, repo: FtsRepository) -> None:
        """Test prefix mode with single word."""
        result = repo._transform_query_postgresql('python', 'prefix')
        assert result == 'python:*'

    def test_transform_query_prefix_with_existing_star(self, repo: FtsRepository) -> None:
        """Test prefix mode with existing * wildcard."""
        result = repo._transform_query_postgresql('implement*', 'prefix')
        assert result == 'implement:*'

    def test_transform_query_prefix_with_existing_colon_star(self, repo: FtsRepository) -> None:
        """Test prefix mode with existing :* suffix."""
        result = repo._transform_query_postgresql('implement:*', 'prefix')
        assert result == 'implement:*'

    def test_transform_query_prefix_with_double_star(self, repo: FtsRepository) -> None:
        """Test prefix mode with double wildcard normalizes correctly."""
        result = repo._transform_query_postgresql('test**', 'prefix')
        assert result == 'test:*'

    def test_transform_query_prefix_mixed_wildcards(self, repo: FtsRepository) -> None:
        """Test prefix mode with mixed wildcards in multiple words."""
        result = repo._transform_query_postgresql('hello* world:* test', 'prefix')
        assert result == 'hello:* & world:* & test:*'

    def test_transform_query_match_mode_passthrough(self, repo: FtsRepository) -> None:
        """Test match mode returns query as-is."""
        result = repo._transform_query_postgresql('hello world', 'match')
        assert result == 'hello world'

    def test_transform_query_phrase_mode_passthrough(self, repo: FtsRepository) -> None:
        """Test phrase mode returns query as-is."""
        result = repo._transform_query_postgresql('hello world', 'phrase')
        assert result == 'hello world'

    def test_transform_query_boolean_mode_passthrough(self, repo: FtsRepository) -> None:
        """Test boolean mode returns query as-is."""
        result = repo._transform_query_postgresql('hello OR world', 'boolean')
        assert result == 'hello OR world'

    def test_transform_query_strips_whitespace(self, repo: FtsRepository) -> None:
        """Test that queries are stripped of leading/trailing whitespace."""
        result = repo._transform_query_postgresql('  hello  ', 'prefix')
        assert result == 'hello:*'

    def test_transform_query_match_empty_string(self, repo: FtsRepository) -> None:
        """Test that empty/whitespace query returns empty string in match mode."""
        result = repo._transform_query_postgresql('   ', 'match')
        assert result == ''

    def test_transform_query_boolean_with_special_characters(self, repo: FtsRepository) -> None:
        """Test that boolean mode passes through special characters unchanged."""
        result = repo._transform_query_postgresql(
            'error OR "stack trace" -timeout', 'boolean',
        )
        assert result == 'error OR "stack trace" -timeout'

    def test_transform_query_phrase_with_internal_quotes(self, repo: FtsRepository) -> None:
        """Test that phrase mode preserves queries with internal quotes."""
        result = repo._transform_query_postgresql(
            'error "handling"', 'phrase',
        )
        assert result == 'error "handling"'


class TestFtsRepositoryPostgreSQLFunctions:
    """Test PostgreSQL tsquery function selection."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock backend for testing."""
        backend = MagicMock()
        backend.backend_type = 'postgresql'
        return backend

    @pytest.fixture
    def repo(self, mock_backend: MagicMock) -> FtsRepository:
        """Create a repository with mock backend."""
        return FtsRepository(mock_backend)

    def test_get_tsquery_function_match(self, repo: FtsRepository) -> None:
        """Test match mode uses plainto_tsquery."""
        result = repo._get_tsquery_function('match', 'english')
        assert 'plainto_tsquery' in result
        assert 'english' in result

    def test_is_available_pg_probe_is_schema_aware(self) -> None:
        """The PG availability probe resolves context_entries via search_path.

        A schema-blind information_schema.columns query (no table_schema filter)
        reported the FTS column present when it existed in ANY visible schema
        (e.g. a colliding public.context_entries), defeating the schema-aware
        FTS backstop on a non-default POSTGRESQL_SCHEMA. The probe must resolve
        the relation via to_regclass (search_path) like the FTS reads/writes.
        """
        import inspect

        src = inspect.getsource(FtsRepository.is_available)
        assert 'to_regclass' in src
        assert 'pg_attribute' in src
        # The old schema-blind lookup query (FROM information_schema.columns with
        # no table_schema filter) must be gone from the SQL itself.
        assert 'FROM information_schema.columns' not in src
        assert 'FROM\n                        information_schema.columns' not in src

    def test_get_tsquery_function_phrase(self, repo: FtsRepository) -> None:
        """Test phrase mode uses phraseto_tsquery."""
        result = repo._get_tsquery_function('phrase', 'english')
        assert 'phraseto_tsquery' in result
        assert 'english' in result

    def test_get_tsquery_function_prefix(self, repo: FtsRepository) -> None:
        """Test prefix mode uses to_tsquery."""
        result = repo._get_tsquery_function('prefix', 'english')
        assert 'to_tsquery' in result
        assert 'english' in result

    def test_get_tsquery_function_boolean(self, repo: FtsRepository) -> None:
        """Test boolean mode uses websearch_to_tsquery."""
        result = repo._get_tsquery_function('boolean', 'english')
        assert 'websearch_to_tsquery' in result
        assert 'english' in result

    def test_get_tsquery_function_german(self, repo: FtsRepository) -> None:
        """Test function generation with German language."""
        result = repo._get_tsquery_function('match', 'german')
        assert 'plainto_tsquery' in result
        assert 'german' in result

    @pytest.mark.parametrize(
        ('mode', 'expected_func'),
        [
            ('match', 'plainto_tsquery'),
            ('phrase', 'phraseto_tsquery'),
            ('prefix', 'to_tsquery'),
            ('boolean', 'websearch_to_tsquery'),
        ],
    )
    def test_get_tsquery_function_parametrized(
        self,
        repo: FtsRepository,
        mode: Literal['match', 'prefix', 'phrase', 'boolean'],
        expected_func: str,
    ) -> None:
        """Parametrized test for all search modes."""
        result = repo._get_tsquery_function(mode, 'english')
        assert expected_func in result


class TestFtsIsAvailablePropagatesOperationalFaults:
    """is_available() reports feature ABSENCE, never an operational fault.

    Every search request runs this probe (it backs fts_search_context and the hybrid FTS
    leg), so swallowing exceptions turned a transient lock held by an external VACUUM or
    backup -- and a corrupt database image -- into the misleading "FTS index not found,
    restart the server to apply migrations" error, while hybrid search silently returned
    semantic-only results. Swallowing inside the callable also consumed the fault before the
    backend's bounded locked-retry loop and circuit-breaker accounting could see it. Neither
    probe needs a handler for its stated purpose: sqlite_master yields zero rows for a
    missing table and to_regclass yields NULL for a missing relation.
    """

    @pytest.mark.asyncio
    async def test_sqlite_probe_error_propagates(self) -> None:
        """A locked database surfaces as an error, not as 'FTS unavailable'."""
        from collections.abc import Callable
        from typing import cast

        from app.backends.base import StorageBackend

        class _LockedConnection:
            def execute(self, _sql: str, _params: object = None) -> object:
                raise sqlite3.OperationalError('database is locked')

        class _Backend:
            backend_type = 'sqlite'

            async def execute_read(self, operation: Callable[[object], bool]) -> bool:
                return operation(_LockedConnection())

        repo = FtsRepository(cast(StorageBackend, _Backend()))

        with pytest.raises(sqlite3.OperationalError, match='database is locked'):
            await repo.is_available()

    @pytest.mark.asyncio
    async def test_sqlite_probe_reports_absent_table_as_false(self) -> None:
        """A missing FTS table is still reported as unavailable (zero catalog rows)."""
        from collections.abc import Callable
        from typing import cast

        from app.backends.base import StorageBackend

        class _EmptyCursor:
            def fetchone(self) -> object | None:
                return None

        class _Connection:
            def execute(self, _sql: str, _params: object = None) -> _EmptyCursor:
                return _EmptyCursor()

        class _Backend:
            backend_type = 'sqlite'

            async def execute_read(self, operation: Callable[[object], bool]) -> bool:
                return operation(_Connection())

        repo = FtsRepository(cast(StorageBackend, _Backend()))

        assert await repo.is_available() is False

    @pytest.mark.asyncio
    async def test_postgresql_probe_error_propagates(self) -> None:
        """A dropped connection surfaces as an error, not as 'FTS unavailable'."""
        from collections.abc import Awaitable
        from collections.abc import Callable
        from typing import cast

        from app.backends.base import StorageBackend

        class _BrokenConnection:
            async def fetchval(self, _sql: str) -> object:
                raise OSError('connection reset by peer')

        class _Backend:
            backend_type = 'postgresql'

            async def execute_read(self, operation: Callable[[object], Awaitable[bool]]) -> bool:
                return await operation(_BrokenConnection())

        repo = FtsRepository(cast(StorageBackend, _Backend()))

        with pytest.raises(OSError, match='connection reset by peer'):
            await repo.is_available()

    @pytest.mark.asyncio
    async def test_postgresql_probe_reports_absent_column_as_false(self) -> None:
        """A missing relation/column is still reported as unavailable (EXISTS false)."""
        from collections.abc import Awaitable
        from collections.abc import Callable
        from typing import cast

        from app.backends.base import StorageBackend

        class _Connection:
            async def fetchval(self, _sql: str) -> bool:
                return False

        class _Backend:
            backend_type = 'postgresql'

            async def execute_read(self, operation: Callable[[object], Awaitable[bool]]) -> bool:
                return await operation(_Connection())

        repo = FtsRepository(cast(StorageBackend, _Backend()))

        assert await repo.is_available() is False


class TestFtsValidationError:
    """Test FtsValidationError exception."""

    def test_exception_creation(self) -> None:
        """Test exception can be created with message and errors."""
        errors = ['Error 1', 'Error 2']
        exc = FtsValidationError('Validation failed', errors)
        assert exc.message == 'Validation failed'
        assert exc.validation_errors == errors

    def test_exception_string_representation(self) -> None:
        """Test exception string representation."""
        errors = ['Error 1']
        exc = FtsValidationError('Validation failed', errors)
        assert str(exc) == 'Validation failed'

    def test_exception_empty_errors(self) -> None:
        """Test exception with empty errors list."""
        exc = FtsValidationError('No specific errors', [])
        assert exc.message == 'No specific errors'
        assert exc.validation_errors == []


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
            await repo_sqlite.search('test', language='english')

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
            await repo_sqlite.search('test', language='german')

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
            await repo_sqlite.search('recherche', language='french')

        assert 'SQLite FTS5 does not support language-specific stemming' in caplog.text
        assert 'french' in caplog.text


class TestFtsTokenizerSelection:
    """Test language-aware tokenizer selection for SQLite FTS5."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock backend for testing."""
        backend = MagicMock()
        backend.backend_type = 'sqlite'
        return backend

    @pytest.fixture
    def repo(self, mock_backend: MagicMock) -> FtsRepository:
        """Create a repository with mock backend."""
        return FtsRepository(mock_backend)

    @pytest.mark.asyncio
    async def test_desired_tokenizer_for_english(self, repo: FtsRepository) -> None:
        """Test that English language uses Porter stemmer."""
        tokenizer = await repo.get_desired_tokenizer('english')
        assert tokenizer == 'porter unicode61'

    @pytest.mark.asyncio
    async def test_desired_tokenizer_for_english_uppercase(self, repo: FtsRepository) -> None:
        """Test that English (uppercase) uses Porter stemmer."""
        tokenizer = await repo.get_desired_tokenizer('ENGLISH')
        assert tokenizer == 'porter unicode61'

    @pytest.mark.asyncio
    async def test_desired_tokenizer_for_german(self, repo: FtsRepository) -> None:
        """Test that German language uses unicode61 only (no stemming)."""
        tokenizer = await repo.get_desired_tokenizer('german')
        assert tokenizer == 'unicode61'

    @pytest.mark.asyncio
    async def test_desired_tokenizer_for_french(self, repo: FtsRepository) -> None:
        """Test that French language uses unicode61 only (no stemming)."""
        tokenizer = await repo.get_desired_tokenizer('french')
        assert tokenizer == 'unicode61'

    @pytest.mark.asyncio
    async def test_desired_tokenizer_for_spanish(self, repo: FtsRepository) -> None:
        """Test that Spanish language uses unicode61 only (no stemming)."""
        tokenizer = await repo.get_desired_tokenizer('spanish')
        assert tokenizer == 'unicode61'

    @pytest.mark.asyncio
    async def test_get_current_tokenizer_no_fts_table(self) -> None:
        """Test get_current_tokenizer returns None when FTS table doesn't exist."""
        from unittest.mock import AsyncMock

        # Create backend with execute_read configured BEFORE creating repo
        backend = MagicMock()
        backend.backend_type = 'sqlite'
        backend.execute_read = AsyncMock(return_value=None)
        repo = FtsRepository(backend)

        tokenizer = await repo.get_current_tokenizer()
        assert tokenizer is None

    @pytest.mark.asyncio
    async def test_get_current_tokenizer_postgresql_returns_none(self) -> None:
        """Test get_current_tokenizer returns None for PostgreSQL backend."""
        backend = MagicMock()
        backend.backend_type = 'postgresql'
        repo = FtsRepository(backend)

        tokenizer = await repo.get_current_tokenizer()
        assert tokenizer is None


class TestFtsLanguageDetection:
    """Test PostgreSQL FTS language detection."""

    @pytest.fixture
    def mock_pg_backend(self) -> MagicMock:
        """Create a mock PostgreSQL backend for testing."""
        backend = MagicMock()
        backend.backend_type = 'postgresql'
        return backend

    @pytest.fixture
    def repo(self, mock_pg_backend: MagicMock) -> FtsRepository:
        """Create a repository with mock PostgreSQL backend."""
        return FtsRepository(mock_pg_backend)

    @pytest.mark.asyncio
    async def test_get_current_language_sqlite_returns_none(self) -> None:
        """Test get_current_language returns None for SQLite backend."""
        backend = MagicMock()
        backend.backend_type = 'sqlite'
        repo = FtsRepository(backend)

        language = await repo.get_current_language()
        assert language is None

    @pytest.mark.asyncio
    async def test_get_current_language_no_tsvector_column(self) -> None:
        """Test get_current_language returns None when tsvector column doesn't exist."""
        from unittest.mock import AsyncMock

        # Create backend with execute_read configured BEFORE creating repo
        backend = MagicMock()
        backend.backend_type = 'postgresql'
        backend.execute_read = AsyncMock(return_value=None)
        repo = FtsRepository(backend)

        language = await repo.get_current_language()
        assert language is None


class TestFtsHyphenHandlingSQLite:
    """Test hyphen handling in SQLite FTS5 queries.

    These tests verify the fix for the bug where hyphens in queries like
    "full-text" were interpreted as the NOT operator instead of being
    treated as part of the word.
    """

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock SQLite backend for testing."""
        backend = MagicMock()
        backend.backend_type = 'sqlite'
        return backend

    @pytest.fixture
    def repo(self, mock_backend: MagicMock) -> FtsRepository:
        """Create a repository with mock SQLite backend."""
        return FtsRepository(mock_backend)

    # Helper method tests
    def test_escape_double_quotes_no_quotes(self, repo: FtsRepository) -> None:
        """Test double quote escaping with no quotes."""
        assert repo._escape_double_quotes('hello') == 'hello'

    def test_escape_double_quotes_with_quotes(self, repo: FtsRepository) -> None:
        """Test double quote escaping with quotes."""
        assert repo._escape_double_quotes('say "hello"') == 'say ""hello""'

    def test_escape_double_quotes_only_quotes(self, repo: FtsRepository) -> None:
        """Test double quote escaping with only quotes."""
        assert repo._escape_double_quotes('"test"') == '""test""'

    def test_quote_hyphenated_words_single(self, repo: FtsRepository) -> None:
        """Test hyphenated word quoting - single hyphenated word."""
        assert repo._quote_hyphenated_words_sqlite('full-text') == '"full-text"'

    def test_quote_hyphenated_words_with_regular(self, repo: FtsRepository) -> None:
        """Test hyphenated word quoting - mixed with regular words."""
        assert repo._quote_hyphenated_words_sqlite('full-text search') == '"full-text" search'

    def test_quote_hyphenated_words_prefix(self, repo: FtsRepository) -> None:
        """Test hyphenated word quoting - at start of query."""
        assert repo._quote_hyphenated_words_sqlite('pre-commit hook') == '"pre-commit" hook'

    def test_quote_hyphenated_words_no_hyphens(self, repo: FtsRepository) -> None:
        """Test hyphenated word quoting - no hyphens in query."""
        assert repo._quote_hyphenated_words_sqlite('hello world') == 'hello world'

    def test_quote_hyphenated_words_multiple(self, repo: FtsRepository) -> None:
        """Test hyphenated word quoting - multiple hyphenated words."""
        result = repo._quote_hyphenated_words_sqlite('full-text real-time')
        assert result == '"full-text" "real-time"'

    def test_quote_hyphenated_words_multi_hyphen(self, repo: FtsRepository) -> None:
        """Test hyphenated word quoting - word with multiple hyphens."""
        result = repo._quote_hyphenated_words_sqlite('pre-commit-hook')
        assert result == '"pre-commit-hook"'

    def test_quote_hyphenated_with_quotes_not_matched(self, repo: FtsRepository) -> None:
        """Test that words with quotes are not matched as hyphenated.

        The regex pattern requires word characters after the hyphen,
        so 'test-"quoted"' is not recognized as a hyphenated word.
        This is expected behavior - such patterns are rare in practice.
        """
        result = repo._quote_hyphenated_words_sqlite('test-"quoted"')
        # Not matched as hyphenated because " is not a word character
        assert result == 'test-"quoted"'

    def test_quote_hyphenated_word_with_number(self, repo: FtsRepository) -> None:
        """Test hyphenated word with number."""
        result = repo._quote_hyphenated_words_sqlite('utf-8 encoding')
        assert result == '"utf-8" encoding'

    # Transform query tests - match mode
    def test_transform_match_simple(self, repo: FtsRepository) -> None:
        """Test match mode with simple query."""
        result = repo._transform_query_sqlite('hello world', 'match')
        # Match mode now quotes each term as an FTS5 literal (AND logic, crash-safe).
        assert result == '"hello" "world"'

    def test_transform_match_hyphenated(self, repo: FtsRepository) -> None:
        """Test match mode with hyphenated word."""
        result = repo._transform_query_sqlite('full-text search', 'match')
        assert result == '"full text" "search"'

    def test_transform_match_multiple_hyphens(self, repo: FtsRepository) -> None:
        """Test match mode with multi-hyphen word."""
        result = repo._transform_query_sqlite('pre-commit-hook', 'match')
        assert result == '"pre commit hook"'

    def test_transform_match_multiple_hyphenated_words(self, repo: FtsRepository) -> None:
        """Test match mode with multiple hyphenated words."""
        result = repo._transform_query_sqlite('full-text real-time search', 'match')
        assert result == '"full text" "real time" "search"'

    # Transform query tests - prefix mode
    def test_transform_prefix_simple(self, repo: FtsRepository) -> None:
        """Test prefix mode with simple query."""
        result = repo._transform_query_sqlite('hello world', 'prefix')
        assert result == '"hello"* "world"*'

    def test_transform_prefix_hyphenated(self, repo: FtsRepository) -> None:
        """Prefix mode splits a hyphenated word into AND-ed wildcarded literals.

        PostgreSQL's prefix transform emits 'full:* & text:*' with no adjacency
        requirement, so keeping the parts in one literal would make SQLite demand
        adjacency for a query the other backend answers without it.
        """
        result = repo._transform_query_sqlite('full-text', 'prefix')
        assert result == '"full"* "text"*'

    def test_transform_prefix_mixed(self, repo: FtsRepository) -> None:
        """Test prefix mode with mixed words."""
        result = repo._transform_query_sqlite('real-time data', 'prefix')
        assert result == '"real"* "time"* "data"*'

    def test_transform_prefix_multiple_hyphenated(self, repo: FtsRepository) -> None:
        """Test prefix mode with multiple hyphenated words."""
        result = repo._transform_query_sqlite('full-text real-time', 'prefix')
        assert result == '"full"* "text"* "real"* "time"*'

    # Transform query tests - phrase mode (should remain unchanged)
    def test_transform_phrase_hyphenated(self, repo: FtsRepository) -> None:
        """Test phrase mode with hyphenated word - entire phrase is quoted."""
        result = repo._transform_query_sqlite('full-text search', 'phrase')
        assert result == '"full-text search"'

    def test_transform_phrase_with_quotes(self, repo: FtsRepository) -> None:
        """Test phrase mode escapes existing quotes."""
        result = repo._transform_query_sqlite('say "hello"', 'phrase')
        assert result == '"say ""hello"""'

    # Transform query tests - boolean mode (pass-through)
    def test_transform_boolean_hyphenated(self, repo: FtsRepository) -> None:
        """Test boolean mode passes through as-is."""
        result = repo._transform_query_sqlite('"full-text" AND search', 'boolean')
        assert result == '"full-text" AND search'

    def test_transform_boolean_not_operator(self, repo: FtsRepository) -> None:
        """Test boolean mode preserves NOT operator usage."""
        result = repo._transform_query_sqlite('search NOT deprecated', 'boolean')
        assert result == 'search NOT deprecated'


class TestFtsHyphenHandlingPostgreSQL:
    """Test hyphen handling in PostgreSQL tsquery queries.

    These tests verify the fix for the bug where hyphens in prefix mode
    queries caused syntax errors with to_tsquery().
    """

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock PostgreSQL backend for testing."""
        backend = MagicMock()
        backend.backend_type = 'postgresql'
        return backend

    @pytest.fixture
    def repo(self, mock_backend: MagicMock) -> FtsRepository:
        """Create a repository with mock PostgreSQL backend."""
        return FtsRepository(mock_backend)

    # Helper method tests
    def test_handle_hyphenated_prefix_simple(self, repo: FtsRepository) -> None:
        """Test simple word prefix handling."""
        result = repo._handle_hyphenated_prefix_postgresql('hello')
        assert result == 'hello:*'

    def test_handle_hyphenated_prefix_hyphen(self, repo: FtsRepository) -> None:
        """Test hyphenated word prefix handling."""
        result = repo._handle_hyphenated_prefix_postgresql('full-text')
        assert result == 'full:* & text:*'

    def test_handle_hyphenated_prefix_multi_hyphen(self, repo: FtsRepository) -> None:
        """Test multi-hyphen word prefix handling."""
        result = repo._handle_hyphenated_prefix_postgresql('pre-commit-hook')
        assert result == 'pre:* & commit:* & hook:*'

    def test_handle_hyphenated_prefix_with_wildcard(self, repo: FtsRepository) -> None:
        """Test word with existing wildcard."""
        result = repo._handle_hyphenated_prefix_postgresql('full-text*')
        assert result == 'full:* & text:*'

    def test_handle_hyphenated_prefix_with_colon_star(self, repo: FtsRepository) -> None:
        """Test word with existing :* suffix."""
        result = repo._handle_hyphenated_prefix_postgresql('hello:*')
        assert result == 'hello:*'

    # Transform query tests - prefix mode
    def test_transform_prefix_simple(self, repo: FtsRepository) -> None:
        """Test prefix mode with simple words."""
        result = repo._transform_query_postgresql('hello world', 'prefix')
        assert result == 'hello:* & world:*'

    def test_transform_prefix_hyphenated(self, repo: FtsRepository) -> None:
        """Test prefix mode with hyphenated word."""
        result = repo._transform_query_postgresql('full-text', 'prefix')
        assert result == 'full:* & text:*'

    def test_transform_prefix_mixed(self, repo: FtsRepository) -> None:
        """Test prefix mode with mixed words."""
        result = repo._transform_query_postgresql('real-time data', 'prefix')
        assert result == 'real:* & time:* & data:*'

    def test_transform_prefix_multiple_hyphenated(self, repo: FtsRepository) -> None:
        """Test prefix mode with multiple hyphenated words."""
        result = repo._transform_query_postgresql('full-text real-time', 'prefix')
        assert result == 'full:* & text:* & real:* & time:*'

    # Other modes - verify pass-through
    def test_transform_match_passthrough(self, repo: FtsRepository) -> None:
        """Test match mode passes through (plainto_tsquery handles)."""
        result = repo._transform_query_postgresql('full-text search', 'match')
        assert result == 'full-text search'

    def test_transform_phrase_passthrough(self, repo: FtsRepository) -> None:
        """Test phrase mode passes through (phraseto_tsquery handles)."""
        result = repo._transform_query_postgresql('full-text search', 'phrase')
        assert result == 'full-text search'

    def test_transform_boolean_passthrough(self, repo: FtsRepository) -> None:
        """Test boolean mode passes through (websearch_to_tsquery)."""
        result = repo._transform_query_postgresql('full-text -exclude', 'boolean')
        assert result == 'full-text -exclude'


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

        async def mock_execute_read(func):  # noqa: ANN001, ANN202
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


class TestFtsMatchFailureClassification:
    """A failed MATCH is classified by database-fault family, not by known grammar wordings.

    The FTS statement mixes server-authored SQL with exactly one client-controlled fragment --
    the MATCH expression -- and the set of messages FTS5 can produce for a bad expression is
    open-ended: deeply nested parentheses raise 'fts5: parser stack overflow', a wording no
    enumeration of known grammar messages contained. Treating an unenumerated wording as a
    server fault re-raised it and charged the process-global SQLite circuit breaker, so a
    client repeating one malformed query could open the breaker and reject every other
    caller's reads and writes for the recovery timeout. Classification therefore enumerates
    the fault families and attributes everything else to the query.
    """

    @staticmethod
    def _fts5_error(query: str) -> sqlite3.OperationalError:
        """Capture the real error SQLite raises for a MATCH expression.

        Args:
            query: The MATCH expression to bind.

        Returns:
            The OperationalError SQLite raised, carrying its real sqlite_errorcode.
        """
        conn = sqlite3.connect(':memory:')
        try:
            conn.execute('CREATE VIRTUAL TABLE d USING fts5(b)')
            with pytest.raises(sqlite3.OperationalError) as exc_info:
                conn.execute('SELECT rowid FROM d WHERE d MATCH ?', (query,)).fetchall()
        finally:
            conn.close()
        return exc_info.value

    def test_parser_stack_overflow_is_attributed_to_the_query(self) -> None:
        """Deeply nested parentheses are malformed client input, not a database fault."""
        exc = self._fts5_error('(' * 100 + 'term' + ')' * 100)

        assert 'parser stack overflow' in str(exc)
        assert _is_fts5_grammar_error(exc, relations_present=True) is True

    @pytest.mark.parametrize(
        'query',
        [
            'term AND',  # dangling trailing operator
            '(term',  # unbalanced parenthesis
            '*term',  # leading '*' read as an unknown special query
            'term : other',  # stray column filter
            'NEAR(a b, x)',  # non-integer NEAR distance
            'term "unterminated',  # unterminated string literal
        ],
    )
    def test_malformed_expressions_are_attributed_to_the_query(self, query: str) -> None:
        """Every shape of malformed MATCH expression classifies as client input."""
        assert _is_fts5_grammar_error(self._fts5_error(query), relations_present=True) is True

    def test_fault_wording_echoed_from_the_query_stays_a_query_error(self) -> None:
        """A client cannot dress its malformed query up as a database fault.

        FTS5 echoes the offending token back in its message, so the client controls part of
        the text. Classification reads the result code first, which keeps an echoed word that
        happens to appear in a fault message from steering the failure onto the breaker.
        """
        exc = self._fts5_error('*interrupted')

        assert 'interrupted' in str(exc)
        assert _is_fts5_grammar_error(exc, relations_present=True) is True

    def test_missing_fts_table_stays_a_database_fault(self) -> None:
        """A missing FTS index must still propagate, not be reported as a bad query.

        A missing relation reports SQLITE_ERROR -- the same result code FTS5 uses for a
        grammar error -- so the code alone cannot separate the two; an un-provisioned
        index is a server-side schema state that has to reach the operator. The two are
        told apart by the CATALOG rather than by the message, because FTS5 echoes client
        tokens into its messages and a message match would let a crafted query decide
        its own classification.
        """
        conn = sqlite3.connect(':memory:')
        try:
            with pytest.raises(sqlite3.OperationalError) as exc_info:
                conn.execute('SELECT rowid FROM absent_fts WHERE absent_fts MATCH ?', ('term',)).fetchall()
            assert _fts_relations_present(conn) is False
        finally:
            conn.close()

        assert _is_fts5_grammar_error(exc_info.value, relations_present=False) is False

    def test_relation_probe_sees_a_provisioned_index(self) -> None:
        """With both relations present the probe reports so, and a bad query stays a query error."""
        conn = sqlite3.connect(':memory:')
        try:
            conn.execute('CREATE TABLE context_entries (id TEXT)')
            conn.execute('CREATE VIRTUAL TABLE context_entries_fts USING fts5(text_content)')

            assert _fts_relations_present(conn) is True
        finally:
            conn.close()

    def test_classification_follows_the_catalog_not_the_message(self) -> None:
        """The SAME error classifies differently only because the CATALOG differs.

        A missing relation and a bad MATCH expression share a result code, so something
        else has to separate them. Reading the message would hand that decision to the
        client, which controls part of the text FTS5 echoes back; reading whether the
        relations exist is provenance no query can influence.
        """
        exc = self._fts5_error('*"no such table":foo')

        assert _is_fts5_grammar_error(exc, relations_present=True) is True
        assert _is_fts5_grammar_error(exc, relations_present=False) is False


class TestPostgresqlFtsFailureClassification:
    """A failed PostgreSQL FTS statement is classified by SQLSTATE, not by wording.

    The tsquery argument is client-controlled and PostgreSQL rejects an oversized one at
    EXECUTION ('invalid memory alloc request size' while the tsquery is assembled), which
    unclassified charges the process-global circuit breaker -- ten such calls open it for
    every caller, the same denial of service the SQLite path closes. Only SQLSTATEs that
    describe the SERVER's condition count as faults.
    """

    class _PgError(Exception):
        """An exception shaped like the ones asyncpg raises, carrying a SQLSTATE."""

        def __init__(self, sqlstate: str) -> None:
            """Initialize the exception.

            Args:
                sqlstate: The five-character SQLSTATE the server reported.
            """
            super().__init__('postgres said no')
            self.sqlstate = sqlstate

    @pytest.mark.parametrize(
        'sqlstate',
        [
            'XX000',  # internal_error: 'invalid memory alloc request size' on a huge tsquery
            '42601',  # syntax_error in tsquery
            '54000',  # program_limit_exceeded: tsquery/word too large
            '22P02',  # invalid_text_representation
        ],
    )
    def test_statement_level_rejections_are_attributed_to_the_query(self, sqlstate: str) -> None:
        """A rejection of the statement says nothing about database health.

        Args:
            sqlstate: The SQLSTATE PostgreSQL reports for one such rejection.
        """
        assert _is_postgresql_query_failure(self._PgError(sqlstate)) is True

    @pytest.mark.parametrize(
        'sqlstate',
        [
            '08006',  # connection_failure
            '08003',  # connection_does_not_exist
            '53100',  # disk_full
            '53200',  # out_of_memory
            '57P01',  # admin_shutdown
            '58030',  # io_error
            'XX001',  # data_corrupted
            'XX002',  # index_corrupted
            '42P01',  # undefined_table: the FTS index is not provisioned
            '42704',  # undefined_object: the configured text search configuration is missing
            '42501',  # insufficient_privilege
            '3F000',  # invalid_schema_name
        ],
    )
    def test_database_faults_still_propagate(self, sqlstate: str) -> None:
        """A fault of the server or its environment keeps charging the breaker.

        Args:
            sqlstate: The SQLSTATE of one such fault.
        """
        assert _is_postgresql_query_failure(self._PgError(sqlstate)) is False

    def test_an_error_without_a_sqlstate_is_a_fault(self) -> None:
        """A driver or transport failure never reached the server, so it is not the query's."""
        assert _is_postgresql_query_failure(Exception('connection reset')) is False

    @pytest.mark.parametrize(
        'message',
        [
            'database is locked',
            'database table is locked',
            'disk I/O error',
            'database disk image is malformed',
            'database or disk is full',
            'unable to open database file',
            'attempt to write a readonly database',
        ],
    )
    def test_database_fault_families_propagate(self, message: str) -> None:
        """Contention and disk/permission faults stay database faults and keep charging.

        These instances carry no result code (they are constructed in Python, as a wrapper or
        a test does), which exercises the canonical-message fallback.

        Args:
            message: The canonical sqlite3 message for one fault family.
        """
        assert _is_fts5_grammar_error(sqlite3.OperationalError(message), relations_present=True) is False


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
        migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_fts_sqlite.sql'
        fts_sql = migration_path.read_text().replace('{TOKENIZER}', 'unicode61')

        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(load_schema('sqlite'))
            conn.executescript(fts_sql)
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) '
                'VALUES (?, ?, ?, ?, ?)',
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
            results, stats = await repos.fts.search(query=pathological, mode='boolean', limit=10)
            assert stats['backend'] == 'sqlite'
            assert len(results) == 1

        assert backend.circuit_breaker.failures == 0
        assert backend.circuit_breaker.is_open() is False
