"""Unit tests for the pure FTS query handling in app.repositories.fts_repository.query.

Covers the SQLite FTS5 and PostgreSQL tsquery transforms for every search mode, the shared
token sanitizer, the tsquery function selection, and the hyphen and embedded-quote handling.
Most cases need no database; the FTS5 execution cases run against an in-memory SQLite table
so they observe the real MATCH grammar.
"""

import sqlite3
from typing import Literal

import pytest

from app.repositories.fts_repository.query import _escape_double_quotes
from app.repositories.fts_repository.query import _handle_hyphenated_prefix_postgresql
from app.repositories.fts_repository.query import get_tsquery_function
from app.repositories.fts_repository.query import transform_query_postgresql
from app.repositories.fts_repository.query import transform_query_sqlite


class TestFtsRepositoryQueryTransform:
    """Test query transformation for different modes."""

    def test_transform_query_match_mode(self) -> None:
        """Test match mode query transformation - words joined with implicit AND."""
        result = transform_query_sqlite('hello world', 'match')
        # Each term is wrapped as an FTS5 string literal (AND logic preserved, crash-safe).
        assert result == '"hello" "world"'

    def test_transform_query_match_mode_single_word(self) -> None:
        """Test match mode with a single word."""
        result = transform_query_sqlite('python', 'match')
        assert result == '"python"'

    def test_transform_query_phrase_mode(self) -> None:
        """Test phrase mode query transformation - wrapped in double quotes."""
        result = transform_query_sqlite('hello world', 'phrase')
        assert result == '"hello world"'

    def test_transform_query_phrase_mode_single_word(self) -> None:
        """Test phrase mode with a single word."""
        result = transform_query_sqlite('python', 'phrase')
        assert result == '"python"'

    def test_transform_query_prefix_mode(self) -> None:
        """Test prefix mode query transformation - adds * to each word."""
        result = transform_query_sqlite('hello world', 'prefix')
        assert result == '"hello"* "world"*'

    def test_transform_query_prefix_mode_single_word(self) -> None:
        """Test prefix mode with a single word."""
        result = transform_query_sqlite('python', 'prefix')
        assert result == '"python"*'

    def test_transform_query_boolean_mode(self) -> None:
        """Test boolean mode query transformation - passthrough as-is."""
        result = transform_query_sqlite('hello AND world', 'boolean')
        assert result == 'hello AND world'

    def test_transform_query_boolean_mode_complex(self) -> None:
        """Test boolean mode with complex boolean expression."""
        query = 'python AND (async OR await) NOT blocking'
        result = transform_query_sqlite(query, 'boolean')
        assert result == query

    def test_transform_query_strips_whitespace(self) -> None:
        """Test that queries are stripped of leading/trailing whitespace."""
        result = transform_query_sqlite('  hello world  ', 'match')
        assert result == '"hello" "world"'

    def test_transform_query_prefix_with_existing_wildcard(self) -> None:
        """Test prefix mode with existing wildcard does not double it."""
        result = transform_query_sqlite('implement*', 'prefix')
        assert result == '"implement"*'

    def test_transform_query_prefix_with_double_wildcard(self) -> None:
        """Test prefix mode with double wildcard normalizes to single."""
        result = transform_query_sqlite('test**', 'prefix')
        assert result == '"test"*'

    def test_transform_query_prefix_mixed_wildcards(self) -> None:
        """Test prefix mode with mixed wildcards in multiple words."""
        result = transform_query_sqlite('hello* world', 'prefix')
        assert result == '"hello"* "world"*'

    def test_transform_query_prefix_all_wildcards(self) -> None:
        """Test prefix mode with all words already having wildcards."""
        result = transform_query_sqlite('hello* world*', 'prefix')
        assert result == '"hello"* "world"*'

    def test_match_and_prefix_operator_input_runs_on_real_fts5(self) -> None:
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
                fts = transform_query_sqlite(query, mode)
                # An all-operator/all-special input transforms to the '' match-nothing
                # sentinel; _search_sqlite short-circuits it (FTS5 rejects MATCH ''), so mirror
                # that guard. A non-empty transform must still EXECUTE without a syntax error.
                if fts:
                    db.execute('SELECT rowid FROM d WHERE d MATCH ?', (fts,)).fetchall()
        # A normal match query still finds the row (AND recall preserved).
        normal = transform_query_sqlite('python async', 'match')
        assert db.execute('SELECT rowid FROM d WHERE d MATCH ?', (normal,)).fetchall() == [(1,)]

    def test_sanitize_drops_operator_barewords_only_for_stopword_languages(self) -> None:
        """and/or/not are dropped ONLY for languages PostgreSQL treats them as stopwords.

        PostgreSQL's plainto_tsquery drops and/or/not as stopwords for english, hindi, and russian
        (their ASCII words route through english_stem) but keeps them as required lexemes for every
        other language, so the SQLite sanitizer must mirror that per-language or the two backends
        return different rows for the same non-English query.
        """
        from app.repositories.fts_repository.query import sanitize_sqlite_fts_terms

        tokens = ['system', 'and', 'or', 'not', 'config']
        # english (default) and russian: operator barewords dropped, mirroring plainto_tsquery.
        assert sanitize_sqlite_fts_terms(tokens) == ['"system"', '"config"']
        assert sanitize_sqlite_fts_terms(tokens, 'russian') == ['"system"', '"config"']
        # german/simple: kept as literal terms, mirroring plainto_tsquery('german'/'simple', ...).
        kept = ['"system"', '"and"', '"or"', '"not"', '"config"']
        assert sanitize_sqlite_fts_terms(tokens, 'german') == kept
        assert sanitize_sqlite_fts_terms(tokens, 'simple') == kept

    def test_transform_match_keeps_operator_barewords_for_non_english_language(self) -> None:
        """match mode keeps and/or/not as literal terms for a non-stopword language.

        The default (english) drops them; a german deployment keeps them because
        plainto_tsquery('german', 'system and configuration') compiles all three lexemes, so SQLite
        must require them too for cross-backend parity.
        """
        assert transform_query_sqlite('system and configuration', 'match') == '"system" "configuration"'
        assert (
            transform_query_sqlite('system and configuration', 'match', 'german')
            == '"system" "and" "configuration"'
        )

    def test_transform_prefix_keeps_operator_barewords_for_non_english_language(self) -> None:
        """prefix mode applies the same per-language operator-bareword gate as match mode."""
        assert transform_query_sqlite('and config', 'prefix') == '"config"*'
        assert transform_query_sqlite('and config', 'prefix', 'german') == '"and"* "config"*'

    def test_transform_non_english_operator_terms_run_on_real_fts5(self) -> None:
        """The kept operator-bareword literal terms EXECUTE on a real FTS5 table without a syntax
        error and match a row containing the word (parity with PostgreSQL requiring the lexeme)."""

        db = sqlite3.connect(':memory:')
        db.execute("CREATE VIRTUAL TABLE d USING fts5(b, tokenize='unicode61')")
        db.execute("INSERT INTO d(b) VALUES('system and configuration')")
        fts = transform_query_sqlite('system and configuration', 'match', 'german')
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

    def test_sanitize_splits_embedded_quote_into_separate_literals(self) -> None:
        """The shared sanitizer emits one literal per quote-delimited fragment."""
        from app.repositories.fts_repository.query import sanitize_sqlite_fts_terms

        assert sanitize_sqlite_fts_terms(['cat"dog']) == ['"cat"', '"dog"']
        assert sanitize_sqlite_fts_terms(['don"t']) == ['"don"', '"t"']
        # A fragment that is an operator bareword is dropped for a stopword language exactly
        # like a standalone one, matching plainto_tsquery on the same input.
        assert sanitize_sqlite_fts_terms(['and"config']) == ['"config"']
        assert sanitize_sqlite_fts_terms(['and"config'], 'german') == ['"and"', '"config"']

    def test_transform_match_and_prefix_split_embedded_quote(self) -> None:
        """match and prefix modes apply the identical split, so the two never diverge."""
        assert transform_query_sqlite('alpha"zulu', 'match') == '"alpha" "zulu"'
        assert transform_query_sqlite('alpha"zulu', 'prefix') == '"alpha"* "zulu"*'

    def test_embedded_quote_matches_non_adjacent_document_on_real_fts5(self) -> None:
        """Both an adjacent and a non-adjacent document match, as on PostgreSQL."""

        db = sqlite3.connect(':memory:')
        db.execute("CREATE VIRTUAL TABLE d USING fts5(b, tokenize='porter unicode61')")
        db.execute("INSERT INTO d(b) VALUES('alpha zulu beta')")
        db.execute("INSERT INTO d(b) VALUES('alpha somewhere else entirely zulu')")

        fts = transform_query_sqlite('alpha"zulu', 'match')
        rows = db.execute('SELECT rowid FROM d WHERE d MATCH ? ORDER BY rowid', (fts,)).fetchall()
        # Two AND-ed terms with no adjacency requirement: both documents qualify. The old
        # doubled-quote escape matched only the adjacent document.
        assert rows == [(1,), (2,)]

        # Control: the plain space-separated query behaves identically.
        plain = transform_query_sqlite('alpha zulu', 'match')
        assert db.execute('SELECT rowid FROM d WHERE d MATCH ? ORDER BY rowid', (plain,)).fetchall() == [(1,), (2,)]

    def test_hyphen_stays_an_adjacency_phrase(self) -> None:
        """A hyphen keeps its adjacency phrase, the closest FTS5 form to the PG compound lexeme.

        PostgreSQL's plainto_tsquery emits the compound lexeme ``alpha-zulu`` plus its parts,
        which requires the two words adjacent in the document. FTS5's tokenizer drops hyphens
        entirely, so the document side cannot represent the compound at all; the adjacency
        phrase is the tightest expressible approximation and ANDing the parts separately would
        widen recall further away from PostgreSQL.
        """

        assert transform_query_sqlite('alpha-zulu', 'match') == '"alpha zulu"'

        db = sqlite3.connect(':memory:')
        db.execute("CREATE VIRTUAL TABLE d USING fts5(b, tokenize='porter unicode61')")
        db.execute("INSERT INTO d(b) VALUES('alpha zulu beta')")
        db.execute("INSERT INTO d(b) VALUES('alpha somewhere else entirely zulu')")
        fts = transform_query_sqlite('alpha-zulu', 'match')
        assert db.execute('SELECT rowid FROM d WHERE d MATCH ?', (fts,)).fetchall() == [(1,)]

    def test_quote_only_query_matches_nothing_without_syntax_error(self) -> None:
        """A query of nothing but quotes reduces to the match-nothing sentinel."""
        assert transform_query_sqlite('"', 'match') == ''
        assert transform_query_sqlite('"', 'prefix') == ''


class TestFtsRepositoryPostgreSQLQueryTransform:
    """Test PostgreSQL query transformation for different modes."""

    def test_transform_query_prefix_mode(self) -> None:
        """Test prefix mode transforms to tsquery format with :* and & operator."""
        result = transform_query_postgresql('hello world', 'prefix')
        assert result == 'hello:* & world:*'

    def test_transform_query_prefix_single_word(self) -> None:
        """Test prefix mode with single word."""
        result = transform_query_postgresql('python', 'prefix')
        assert result == 'python:*'

    def test_transform_query_prefix_with_existing_star(self) -> None:
        """Test prefix mode with existing * wildcard."""
        result = transform_query_postgresql('implement*', 'prefix')
        assert result == 'implement:*'

    def test_transform_query_prefix_with_existing_colon_star(self) -> None:
        """Test prefix mode with existing :* suffix."""
        result = transform_query_postgresql('implement:*', 'prefix')
        assert result == 'implement:*'

    def test_transform_query_prefix_with_double_star(self) -> None:
        """Test prefix mode with double wildcard normalizes correctly."""
        result = transform_query_postgresql('test**', 'prefix')
        assert result == 'test:*'

    def test_transform_query_prefix_mixed_wildcards(self) -> None:
        """Test prefix mode with mixed wildcards in multiple words."""
        result = transform_query_postgresql('hello* world:* test', 'prefix')
        assert result == 'hello:* & world:* & test:*'

    def test_transform_query_match_mode_passthrough(self) -> None:
        """Test match mode returns query as-is."""
        result = transform_query_postgresql('hello world', 'match')
        assert result == 'hello world'

    def test_transform_query_phrase_mode_passthrough(self) -> None:
        """Test phrase mode returns query as-is."""
        result = transform_query_postgresql('hello world', 'phrase')
        assert result == 'hello world'

    def test_transform_query_boolean_mode_passthrough(self) -> None:
        """Test boolean mode returns query as-is."""
        result = transform_query_postgresql('hello OR world', 'boolean')
        assert result == 'hello OR world'

    def test_transform_query_strips_whitespace(self) -> None:
        """Test that queries are stripped of leading/trailing whitespace."""
        result = transform_query_postgresql('  hello  ', 'prefix')
        assert result == 'hello:*'

    def test_transform_query_match_empty_string(self) -> None:
        """Test that empty/whitespace query returns empty string in match mode."""
        result = transform_query_postgresql('   ', 'match')
        assert result == ''

    def test_transform_query_boolean_with_special_characters(self) -> None:
        """Test that boolean mode passes through special characters unchanged."""
        result = transform_query_postgresql(
            'error OR "stack trace" -timeout', 'boolean',
        )
        assert result == 'error OR "stack trace" -timeout'

    def test_transform_query_phrase_with_internal_quotes(self) -> None:
        """Test that phrase mode preserves queries with internal quotes."""
        result = transform_query_postgresql(
            'error "handling"', 'phrase',
        )
        assert result == 'error "handling"'


class TestFtsRepositoryPostgreSQLFunctions:
    """Test PostgreSQL tsquery function selection."""

    def test_get_tsquery_function_match(self) -> None:
        """Test match mode uses plainto_tsquery."""
        result = get_tsquery_function('match', 'english')
        assert 'plainto_tsquery' in result
        assert 'english' in result

    def test_get_tsquery_function_phrase(self) -> None:
        """Test phrase mode uses phraseto_tsquery."""
        result = get_tsquery_function('phrase', 'english')
        assert 'phraseto_tsquery' in result
        assert 'english' in result

    def test_get_tsquery_function_prefix(self) -> None:
        """Test prefix mode uses to_tsquery."""
        result = get_tsquery_function('prefix', 'english')
        assert 'to_tsquery' in result
        assert 'english' in result

    def test_get_tsquery_function_boolean(self) -> None:
        """Test boolean mode uses websearch_to_tsquery."""
        result = get_tsquery_function('boolean', 'english')
        assert 'websearch_to_tsquery' in result
        assert 'english' in result

    def test_get_tsquery_function_german(self) -> None:
        """Test function generation with German language."""
        result = get_tsquery_function('match', 'german')
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
        mode: Literal['match', 'prefix', 'phrase', 'boolean'],
        expected_func: str,
    ) -> None:
        """Parametrized test for all search modes."""
        result = get_tsquery_function(mode, 'english')
        assert expected_func in result


class TestFtsHyphenHandlingSQLite:
    """Test hyphen handling in SQLite FTS5 queries.

    These tests verify the fix for the bug where hyphens in queries like
    "full-text" were interpreted as the NOT operator instead of being
    treated as part of the word.
    """

    # Helper method tests
    def test_escape_double_quotes_no_quotes(self) -> None:
        """Test double quote escaping with no quotes."""
        assert _escape_double_quotes('hello') == 'hello'

    def test_escape_double_quotes_with_quotes(self) -> None:
        """Test double quote escaping with quotes."""
        assert _escape_double_quotes('say "hello"') == 'say ""hello""'

    def test_escape_double_quotes_only_quotes(self) -> None:
        """Test double quote escaping with only quotes."""
        assert _escape_double_quotes('"test"') == '""test""'

    # Transform query tests - match mode
    def test_transform_match_simple(self) -> None:
        """Test match mode with simple query."""
        result = transform_query_sqlite('hello world', 'match')
        # Match mode now quotes each term as an FTS5 literal (AND logic, crash-safe).
        assert result == '"hello" "world"'

    def test_transform_match_hyphenated(self) -> None:
        """Test match mode with hyphenated word."""
        result = transform_query_sqlite('full-text search', 'match')
        assert result == '"full text" "search"'

    def test_transform_match_multiple_hyphens(self) -> None:
        """Test match mode with multi-hyphen word."""
        result = transform_query_sqlite('pre-commit-hook', 'match')
        assert result == '"pre commit hook"'

    def test_transform_match_multiple_hyphenated_words(self) -> None:
        """Test match mode with multiple hyphenated words."""
        result = transform_query_sqlite('full-text real-time search', 'match')
        assert result == '"full text" "real time" "search"'

    # Transform query tests - prefix mode
    def test_transform_prefix_simple(self) -> None:
        """Test prefix mode with simple query."""
        result = transform_query_sqlite('hello world', 'prefix')
        assert result == '"hello"* "world"*'

    def test_transform_prefix_hyphenated(self) -> None:
        """Prefix mode splits a hyphenated word into AND-ed wildcarded literals.

        PostgreSQL's prefix transform emits 'full:* & text:*' with no adjacency
        requirement, so keeping the parts in one literal would make SQLite demand
        adjacency for a query the other backend answers without it.
        """
        result = transform_query_sqlite('full-text', 'prefix')
        assert result == '"full"* "text"*'

    def test_transform_prefix_mixed(self) -> None:
        """Test prefix mode with mixed words."""
        result = transform_query_sqlite('real-time data', 'prefix')
        assert result == '"real"* "time"* "data"*'

    def test_transform_prefix_multiple_hyphenated(self) -> None:
        """Test prefix mode with multiple hyphenated words."""
        result = transform_query_sqlite('full-text real-time', 'prefix')
        assert result == '"full"* "text"* "real"* "time"*'

    # Transform query tests - phrase mode (should remain unchanged)
    def test_transform_phrase_hyphenated(self) -> None:
        """Test phrase mode with hyphenated word - entire phrase is quoted."""
        result = transform_query_sqlite('full-text search', 'phrase')
        assert result == '"full-text search"'

    def test_transform_phrase_with_quotes(self) -> None:
        """Test phrase mode escapes existing quotes."""
        result = transform_query_sqlite('say "hello"', 'phrase')
        assert result == '"say ""hello"""'

    # Transform query tests - boolean mode (pass-through)
    def test_transform_boolean_hyphenated(self) -> None:
        """Test boolean mode passes through as-is."""
        result = transform_query_sqlite('"full-text" AND search', 'boolean')
        assert result == '"full-text" AND search'

    def test_transform_boolean_not_operator(self) -> None:
        """Test boolean mode preserves NOT operator usage."""
        result = transform_query_sqlite('search NOT deprecated', 'boolean')
        assert result == 'search NOT deprecated'


class TestFtsHyphenHandlingPostgreSQL:
    """Test hyphen handling in PostgreSQL tsquery queries.

    These tests verify the fix for the bug where hyphens in prefix mode
    queries caused syntax errors with to_tsquery().
    """

    # Helper method tests
    def test_handle_hyphenated_prefix_simple(self) -> None:
        """Test simple word prefix handling."""
        result = _handle_hyphenated_prefix_postgresql('hello')
        assert result == 'hello:*'

    def test_handle_hyphenated_prefix_hyphen(self) -> None:
        """Test hyphenated word prefix handling."""
        result = _handle_hyphenated_prefix_postgresql('full-text')
        assert result == 'full:* & text:*'

    def test_handle_hyphenated_prefix_multi_hyphen(self) -> None:
        """Test multi-hyphen word prefix handling."""
        result = _handle_hyphenated_prefix_postgresql('pre-commit-hook')
        assert result == 'pre:* & commit:* & hook:*'

    def test_handle_hyphenated_prefix_with_wildcard(self) -> None:
        """Test word with existing wildcard."""
        result = _handle_hyphenated_prefix_postgresql('full-text*')
        assert result == 'full:* & text:*'

    def test_handle_hyphenated_prefix_with_colon_star(self) -> None:
        """Test word with existing :* suffix."""
        result = _handle_hyphenated_prefix_postgresql('hello:*')
        assert result == 'hello:*'

    # Transform query tests - prefix mode
    def test_transform_prefix_simple(self) -> None:
        """Test prefix mode with simple words."""
        result = transform_query_postgresql('hello world', 'prefix')
        assert result == 'hello:* & world:*'

    def test_transform_prefix_hyphenated(self) -> None:
        """Test prefix mode with hyphenated word."""
        result = transform_query_postgresql('full-text', 'prefix')
        assert result == 'full:* & text:*'

    def test_transform_prefix_mixed(self) -> None:
        """Test prefix mode with mixed words."""
        result = transform_query_postgresql('real-time data', 'prefix')
        assert result == 'real:* & time:* & data:*'

    def test_transform_prefix_multiple_hyphenated(self) -> None:
        """Test prefix mode with multiple hyphenated words."""
        result = transform_query_postgresql('full-text real-time', 'prefix')
        assert result == 'full:* & text:* & real:* & time:*'

    # Other modes - verify pass-through
    def test_transform_match_passthrough(self) -> None:
        """Test match mode passes through (plainto_tsquery handles)."""
        result = transform_query_postgresql('full-text search', 'match')
        assert result == 'full-text search'

    def test_transform_phrase_passthrough(self) -> None:
        """Test phrase mode passes through (phraseto_tsquery handles)."""
        result = transform_query_postgresql('full-text search', 'phrase')
        assert result == 'full-text search'

    def test_transform_boolean_passthrough(self) -> None:
        """Test boolean mode passes through (websearch_to_tsquery)."""
        result = transform_query_postgresql('full-text -exclude', 'boolean')
        assert result == 'full-text -exclude'
