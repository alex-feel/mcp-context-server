"""Tests for the adaptive FTS query preparation in hybrid search.

Covers _prepare_hybrid_fts_query: match versus boolean mode by significant-term count, operator
neutralization, and hyphen and quote handling on both backends.
"""


class TestAdaptiveFtsMode:
    """Test adaptive FTS mode switching for hybrid search.

    Verifies that _prepare_hybrid_fts_query() correctly switches
    between AND (match) and OR (boolean) modes based on query length.
    """

    def test_short_query_uses_match_mode(self) -> None:
        """Queries below threshold use match mode (AND logic)."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='python async',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'match'
        assert query == 'python async'

    def test_exact_threshold_uses_boolean_mode(self) -> None:
        """Queries at exactly the threshold switch to boolean mode."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='python async await patterns',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert 'or' in query.lower()

    def test_sqlite_boolean_mode_neutralizes_fts5_operators(self) -> None:
        """SQLite boolean-mode terms are quoted so operator barewords / special chars
        cannot form malformed FTS5 (which raised a syntax error while PostgreSQL's
        websearch_to_tsquery tolerated the same input -- a recall-parity divergence)."""
        import sqlite3

        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        db = sqlite3.connect(':memory:')
        db.execute('CREATE VIRTUAL TABLE docs USING fts5(body)')
        db.execute("INSERT INTO docs(body) VALUES('foo bar near baz qux')")
        # Each query has >= 4 significant terms and embeds FTS5 operators / specials that
        # would be a syntax error unquoted; all must run cleanly and stay in boolean mode.
        for raw in [
            'foo bar near baz qux',
            'alpha AND beta OR gamma delta',
            'find foo:bar (baz) term here',
            'one two NOT three four',
        ]:
            transformed, mode = _prepare_hybrid_fts_query(raw, or_threshold=4, backend_type='sqlite')
            assert mode == 'boolean'
            assert ' OR ' in transformed
            # No raw operator bareword survives outside quotes (it is wrapped as a string).
            db.execute('SELECT rowid FROM docs WHERE docs MATCH ?', (transformed,)).fetchall()

    def test_sqlite_all_operator_query_returns_empty_sentinel(self) -> None:
        """A query whose significant terms are ALL operator stopwords (and/or/not) empties
        the term list; the SQLite transform must return the '' match-nothing sentinel -- NOT
        a literal phrase (FTS5 porter/unicode61 keeps the dropped word as a token, so the
        phrase would MATCH every document containing it, diverging from PostgreSQL's empty
        tsquery) and NOT a raw uppercase operator (which would raise an FTS5 MATCH syntax
        error). _search_sqlite short-circuits '' to an empty result set, so MATCH is never
        executed with the sentinel."""
        import sqlite3

        from app.repositories.fts_repository.query import transform_query_sqlite
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        db = sqlite3.connect(':memory:')
        db.execute("CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='porter unicode61')")
        db.execute("INSERT INTO docs(body) VALUES('hello world')")
        for raw in ['AND OR NOT AND', 'NOT NOT NOT NOT', 'OR OR OR OR']:
            adaptive, mode = _prepare_hybrid_fts_query(raw, or_threshold=4, backend_type='sqlite')
            fts = transform_query_sqlite(adaptive, mode)
            # All tokens were operator barewords -> the empty match-nothing sentinel.
            assert fts == ''
            # _search_sqlite skips MATCH on the empty sentinel; mirror that guard here so
            # MATCH is never run with '' (which FTS5 rejects), yielding zero results.
            rows = [] if not fts else db.execute('SELECT rowid FROM docs WHERE docs MATCH ?', (fts,)).fetchall()
            assert rows == []

    def test_sqlite_short_query_operators_do_not_crash(self) -> None:
        """A SHORT query (below the OR threshold) containing bare FTS5 operators -- including
        leading/trailing/all-operator forms -- must not crash SQLite MATCH: the short 'match'
        path runs through the same term sanitizer as the OR path. A normal short query keeps
        AND-of-terms recall."""
        import sqlite3

        from app.repositories.fts_repository.query import transform_query_sqlite
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        db = sqlite3.connect(':memory:')
        db.execute("CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='porter unicode61')")
        db.execute("INSERT INTO docs(body) VALUES('python async world')")
        # All have < or_threshold(4) significant terms, so they take the short 'match' path.
        for raw, expect_match in [('AND OR', False), ('OR cat', False), ('cat OR', False),
                                  ('OR NOT AND', False), ('python async', True)]:
            adaptive, mode = _prepare_hybrid_fts_query(raw, or_threshold=4, backend_type='sqlite')
            assert mode == 'match'
            fts = transform_query_sqlite(adaptive, mode)
            # An all-operator query transforms to the '' match-nothing sentinel, which
            # _search_sqlite short-circuits (FTS5 rejects MATCH ''); mirror that guard here.
            rows = [] if not fts else db.execute('SELECT rowid FROM docs WHERE docs MATCH ?', (fts,)).fetchall()
            assert (len(rows) > 0) is expect_match, f'{raw!r} -> {fts!r}'

    def test_sqlite_boolean_mode_drops_operator_stopwords(self) -> None:
        """The FTS5 operator barewords and/or/not are DROPPED on the SQLite branch (not
        quoted) so they are not literal searchable terms -- matching PostgreSQL's
        websearch_to_tsquery, which removes them as stopwords (cross-backend recall parity).
        'near' is KEPT (websearch_to_tsquery keeps it)."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        transformed, mode = _prepare_hybrid_fts_query(
            'alpha AND beta OR gamma NOT delta near epsilon',
            or_threshold=4,
            backend_type='sqlite',
        )
        assert mode == 'boolean'
        for dropped in ('"and"', '"or"', '"not"', '"AND"', '"OR"', '"NOT"'):
            assert dropped not in transformed
        # Real terms (including 'near') survive as quoted literals.
        for kept in ('"alpha"', '"beta"', '"gamma"', '"delta"', '"near"', '"epsilon"'):
            assert kept in transformed

    def test_long_query_uses_boolean_mode(self) -> None:
        """Long queries above threshold use boolean mode (OR logic)."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='DRY extraction embedding helper timeout semaphore pattern',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert ' or ' in query

    def test_postgresql_uses_lowercase_or(self) -> None:
        """PostgreSQL backend uses lowercase 'or' keyword."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='alpha beta gamma delta',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert ' or ' in query
        assert ' OR ' not in query

    def test_sqlite_uses_uppercase_or(self) -> None:
        """SQLite backend uses uppercase 'OR' keyword."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='alpha beta gamma delta',
            or_threshold=4,
            backend_type='sqlite',
        )
        assert mode == 'boolean'
        assert ' OR ' in query

    def test_single_char_words_excluded_from_count(self) -> None:
        """Single-character words are not counted as significant."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        # "a b c python async" has only 2 significant words (python, async)
        query, mode = _prepare_hybrid_fts_query(
            query='a b c python async',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'match'

    def test_hyphen_sanitization(self) -> None:
        """Hyphens are replaced with spaces to prevent NOT interpretation."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='context-server async-await error-handling patterns',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert '-' not in query

    def test_hyphenated_tokens_keep_phrase_semantics_on_both_backends(self) -> None:
        """A hyphenated token is a quoted phrase on BOTH backends in OR mode.

        SQLite's sanitize_sqlite_fts_terms wraps 'a-b' as the FTS5 phrase
        literal "a b" (ordered adjacency); the PostgreSQL branch must wrap the
        hyphen-joined token the same way so websearch_to_tsquery parses it as
        a <-> b instead of ANDing the parts unordered -- otherwise the two
        backends return different recall for the identical hybrid query.
        """
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        pg_query, pg_mode = _prepare_hybrid_fts_query(
            query='a-b c-d e-f g-h',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert pg_mode == 'boolean'
        assert pg_query == '"a b" or "c d" or "e f" or "g h"'

        sqlite_query, sqlite_mode = _prepare_hybrid_fts_query(
            query='a-b c-d e-f g-h',
            or_threshold=4,
            backend_type='sqlite',
        )
        assert sqlite_mode == 'boolean'
        assert sqlite_query == '"a b" OR "c d" OR "e f" OR "g h"'

        # The two transforms differ only in the OR keyword casing.
        assert pg_query.replace(' or ', ' OR ') == sqlite_query

    def test_hyphenated_token_with_embedded_quote_stays_wrappable(self) -> None:
        """An embedded double quote splits the token instead of leaking into the OR join.

        Left raw, the quote opens a websearch_to_tsquery phrase that swallows every
        following ' or ' term into an adjacency phrase, collapsing recall to near zero.
        Splitting on it yields the same fragments the SQLite sanitizer produces.
        """
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='al"pha-beta gamma delta epsilon',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        # Every quote belongs to a phrase wrapper this builder emitted, so they balance.
        assert query.count('"') % 2 == 0
        assert query == 'al or "pha beta" or gamma or delta or epsilon'

    def test_unbalanced_quote_does_not_swallow_following_or_terms(self) -> None:
        """A stray quote in one term must not phrase-capture the rest of the OR join."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        pg_query, pg_mode = _prepare_hybrid_fts_query(
            query='quo"kkazz wombatzz alpha beta',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert pg_mode == 'boolean'
        assert pg_query == 'quo or kkazz or wombatzz or alpha or beta'

        sqlite_query, sqlite_mode = _prepare_hybrid_fts_query(
            query='quo"kkazz wombatzz alpha beta',
            or_threshold=4,
            backend_type='sqlite',
        )
        assert sqlite_mode == 'boolean'
        assert sqlite_query == '"quo" OR "kkazz" OR "wombatzz" OR "alpha" OR "beta"'
        # Both backends split into the same five independent terms.
        assert sqlite_query.replace('"', '').replace(' OR ', ' or ') == pg_query

    def test_threshold_boundary_below(self) -> None:
        """Query with exactly threshold-1 significant words stays in match mode."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='alpha beta gamma',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'match'
        assert query == 'alpha beta gamma'

    def test_empty_after_sanitization_fallback(self) -> None:
        """If all words sanitize to empty, falls back to match mode."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        # Single-char words with hyphens that would sanitize to empty
        query, mode = _prepare_hybrid_fts_query(
            query='- - - -',
            or_threshold=2,
            backend_type='postgresql',
        )
        assert mode == 'match'

    def test_custom_threshold(self) -> None:
        """Custom threshold value is respected."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        # With threshold=2, even 2-word queries switch to OR
        query, mode = _prepare_hybrid_fts_query(
            query='python async',
            or_threshold=2,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert ' or ' in query

    def test_whitespace_handling(self) -> None:
        """Extra whitespace in query is handled gracefully."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='  alpha   beta   gamma   delta  ',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert ' or ' in query

    def test_quoted_phrase_preserved_in_boolean_mode(self) -> None:
        """Quoted phrases are preserved as single tokens in OR mode."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='"error handling" timeout async patterns',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert '"error handling"' in query

    def test_multiple_quoted_phrases_preserved(self) -> None:
        """Multiple quoted phrases are each preserved intact."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='"error handling" "async await" timeout patterns',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert '"error handling"' in query
        assert '"async await"' in query

    def test_quoted_phrase_not_hyphen_sanitized(self) -> None:
        """Quoted phrases containing hyphens are NOT sanitized."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query(
            query='"error-handling" timeout async patterns',
            or_threshold=4,
            backend_type='postgresql',
        )
        assert mode == 'boolean'
        assert '"error-handling"' in query

    def test_sqlite_short_match_returns_raw_query_for_single_transform(self) -> None:
        """A short SQLite query is returned RAW (not pre-sanitized) so transform_query_sqlite
        sanitizes it exactly once -- identically to standalone fts_search_context."""
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        query, mode = _prepare_hybrid_fts_query('python async', or_threshold=4, backend_type='sqlite')
        assert mode == 'match'
        assert query == 'python async'  # raw; the single downstream transform does the escaping

    def test_sqlite_embedded_quote_token_not_double_mangled(self) -> None:
        """An embedded-double-quote token yields the SAME FTS5 form via the hybrid path as via
        standalone fts_search_context -- exactly one sanitize pass, never two.

        The hybrid builder must hand the RAW query to the single downstream transform. Running
        its own sanitize first would send an already-quoted term list through a second wrapping
        pass, so hybrid recall would merely resemble standalone recall for the same input
        instead of matching it. The expected form splits the token on the embedded quote into
        independently AND-ed literals, mirroring PostgreSQL's independently AND-ed lexemes.
        """
        from app.repositories.fts_repository.query import transform_query_sqlite
        from app.tools.search.hybrid import _prepare_hybrid_fts_query

        raw = 'ab"cd ef'
        # Hybrid short path returns the raw query; the single downstream transform escapes it.
        adaptive, mode = _prepare_hybrid_fts_query(raw, or_threshold=4, backend_type='sqlite')
        assert mode == 'match'
        hybrid_fts = transform_query_sqlite(adaptive, mode)
        # Standalone fts_search_context passes the raw query straight to the same transform.
        standalone_fts = transform_query_sqlite(raw, 'match')
        assert hybrid_fts == standalone_fts
        # Split on the embedded quote into separate AND-ed literals: doubling the quote
        # instead would leave it a word boundary inside one literal, which FTS5 reads as a
        # strict two-word adjacency phrase that PostgreSQL never requires.
        assert hybrid_fts == '"ab" "cd" "ef"'
