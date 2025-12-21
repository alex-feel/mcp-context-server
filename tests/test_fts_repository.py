"""Unit tests for FtsRepository.

Tests the query transformation logic and PostgreSQL tsquery function selection
without requiring a database connection.
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock

import pytest

from app.repositories.fts_repository import FtsRepository
from app.repositories.fts_repository import FtsValidationError


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
        assert result == 'hello world'

    def test_transform_query_match_mode_single_word(self, repo: FtsRepository) -> None:
        """Test match mode with a single word."""
        result = repo._transform_query_sqlite('python', 'match')
        assert result == 'python'

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
        assert result == 'hello* world*'

    def test_transform_query_prefix_mode_single_word(self, repo: FtsRepository) -> None:
        """Test prefix mode with a single word."""
        result = repo._transform_query_sqlite('python', 'prefix')
        assert result == 'python*'

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
        assert result == 'hello world'

    def test_transform_query_prefix_with_existing_wildcard(self, repo: FtsRepository) -> None:
        """Test prefix mode with existing wildcard does not double it."""
        result = repo._transform_query_sqlite('implement*', 'prefix')
        assert result == 'implement*'

    def test_transform_query_prefix_with_double_wildcard(self, repo: FtsRepository) -> None:
        """Test prefix mode with double wildcard normalizes to single."""
        result = repo._transform_query_sqlite('test**', 'prefix')
        assert result == 'test*'

    def test_transform_query_prefix_mixed_wildcards(self, repo: FtsRepository) -> None:
        """Test prefix mode with mixed wildcards in multiple words."""
        result = repo._transform_query_sqlite('hello* world', 'prefix')
        assert result == 'hello* world*'

    def test_transform_query_prefix_all_wildcards(self, repo: FtsRepository) -> None:
        """Test prefix mode with all words already having wildcards."""
        result = repo._transform_query_sqlite('hello* world*', 'prefix')
        assert result == 'hello* world*'


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
        assert result == 'hello world'

    def test_transform_match_hyphenated(self, repo: FtsRepository) -> None:
        """Test match mode with hyphenated word."""
        result = repo._transform_query_sqlite('full-text search', 'match')
        assert result == '"full-text" search'

    def test_transform_match_multiple_hyphens(self, repo: FtsRepository) -> None:
        """Test match mode with multi-hyphen word."""
        result = repo._transform_query_sqlite('pre-commit-hook', 'match')
        assert result == '"pre-commit-hook"'

    def test_transform_match_multiple_hyphenated_words(self, repo: FtsRepository) -> None:
        """Test match mode with multiple hyphenated words."""
        result = repo._transform_query_sqlite('full-text real-time search', 'match')
        assert result == '"full-text" "real-time" search'

    # Transform query tests - prefix mode
    def test_transform_prefix_simple(self, repo: FtsRepository) -> None:
        """Test prefix mode with simple query."""
        result = repo._transform_query_sqlite('hello world', 'prefix')
        assert result == 'hello* world*'

    def test_transform_prefix_hyphenated(self, repo: FtsRepository) -> None:
        """Test prefix mode with hyphenated word."""
        result = repo._transform_query_sqlite('full-text', 'prefix')
        assert result == '"full-text"*'

    def test_transform_prefix_mixed(self, repo: FtsRepository) -> None:
        """Test prefix mode with mixed words."""
        result = repo._transform_query_sqlite('real-time data', 'prefix')
        assert result == '"real-time"* data*'

    def test_transform_prefix_multiple_hyphenated(self, repo: FtsRepository) -> None:
        """Test prefix mode with multiple hyphenated words."""
        result = repo._transform_query_sqlite('full-text real-time', 'prefix')
        assert result == '"full-text"* "real-time"*'

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
