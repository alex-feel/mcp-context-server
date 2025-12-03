"""
Tests for application settings validation.

Ensures settings validators fail fast with clear error messages
for invalid configuration values.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from app.settings import AppSettings


@contextmanager
def env_var(key: str, value: str | None) -> Generator[None, None, None]:
    """Context manager for temporarily setting an environment variable."""
    original = os.environ.get(key)
    try:
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]
        yield
    finally:
        if original is not None:
            os.environ[key] = original
        elif key in os.environ:
            del os.environ[key]


class TestFtsLanguageValidation:
    """Test FTS_LANGUAGE setting validation."""

    def test_valid_languages_accepted(self) -> None:
        """Test that all valid PostgreSQL text search configurations are accepted."""
        valid_languages = [
            'simple',
            'arabic',
            'armenian',
            'basque',
            'catalan',
            'danish',
            'dutch',
            'english',
            'finnish',
            'french',
            'german',
            'greek',
            'hindi',
            'hungarian',
            'indonesian',
            'irish',
            'italian',
            'lithuanian',
            'nepali',
            'norwegian',
            'portuguese',
            'romanian',
            'russian',
            'serbian',
            'spanish',
            'swedish',
            'tamil',
            'turkish',
            'yiddish',
        ]

        for lang in valid_languages:
            with env_var('FTS_LANGUAGE', lang):
                settings = AppSettings()
                assert settings.fts_language == lang.lower(), f'Language {lang} should be accepted'

    def test_valid_languages_case_insensitive(self) -> None:
        """Test that language validation is case-insensitive."""
        case_variations = [
            ('english', 'english'),
            ('English', 'english'),
            ('ENGLISH', 'english'),
            ('EnGlIsH', 'english'),
            ('German', 'german'),
            ('FRENCH', 'french'),
            ('Russian', 'russian'),
        ]

        for input_lang, expected_output in case_variations:
            with env_var('FTS_LANGUAGE', input_lang):
                settings = AppSettings()
                assert settings.fts_language == expected_output, (
                    f'Language {input_lang} should be normalized to {expected_output}'
                )

    def test_invalid_language_raises_error(self) -> None:
        """Test that invalid languages raise ValueError with clear message."""
        invalid_languages = [
            'invalid',
            'nonsense',
            'foo',
            'bar',
            'unknown',
            'eng',
            'en',
            'de',
            'fr',
        ]

        for lang in invalid_languages:
            with env_var('FTS_LANGUAGE', lang):
                with pytest.raises(ValidationError) as exc_info:
                    AppSettings()

                # Check error message contains useful information
                error_str = str(exc_info.value)
                assert 'FTS_LANGUAGE' in error_str, f'Error should mention FTS_LANGUAGE for {lang}'
                assert 'valid options' in error_str.lower(), f'Error should mention valid options for {lang}'

    def test_invalid_language_error_shows_valid_options(self) -> None:
        """Test that the error message shows the list of valid options."""
        with env_var('FTS_LANGUAGE', 'invalid_language'):
            with pytest.raises(ValidationError) as exc_info:
                AppSettings()

            error_str = str(exc_info.value)
            # Check that at least some valid languages are mentioned in the error
            assert 'english' in error_str.lower(), 'Error should list english as a valid option'
            assert 'german' in error_str.lower(), 'Error should list german as a valid option'
            assert 'french' in error_str.lower(), 'Error should list french as a valid option'

    def test_default_language_is_english(self) -> None:
        """Test that the default FTS language is english."""
        # Ensure FTS_LANGUAGE is not set
        with env_var('FTS_LANGUAGE', None):
            settings = AppSettings()
            assert settings.fts_language == 'english'

    def test_fts_language_via_environment_variable(self) -> None:
        """Test that FTS_LANGUAGE can be set via environment variable."""
        # Test valid language via env var
        with env_var('FTS_LANGUAGE', 'german'):
            settings = AppSettings()
            assert settings.fts_language == 'german'

        # Test case normalization via env var
        with env_var('FTS_LANGUAGE', 'FRENCH'):
            settings = AppSettings()
            assert settings.fts_language == 'french'

    def test_invalid_language_via_environment_variable(self) -> None:
        """Test that invalid FTS_LANGUAGE via env var raises error."""
        with env_var('FTS_LANGUAGE', 'completely_invalid_language'):
            with pytest.raises(ValidationError) as exc_info:
                AppSettings()

            error_str = str(exc_info.value)
            assert 'FTS_LANGUAGE' in error_str

    def test_whitespace_language_raises_error(self) -> None:
        """Test that whitespace-only language raises error."""
        whitespace_variants = ['   ', '\t', '\n', ' \t\n ']

        for ws in whitespace_variants:
            with env_var('FTS_LANGUAGE', ws), pytest.raises(ValidationError):
                AppSettings()

    def test_all_29_valid_languages_count(self) -> None:
        """Test that exactly 29 valid languages are supported."""
        # This ensures we don't accidentally add or remove languages
        valid_languages = {
            'simple',
            'arabic',
            'armenian',
            'basque',
            'catalan',
            'danish',
            'dutch',
            'english',
            'finnish',
            'french',
            'german',
            'greek',
            'hindi',
            'hungarian',
            'indonesian',
            'irish',
            'italian',
            'lithuanian',
            'nepali',
            'norwegian',
            'portuguese',
            'romanian',
            'russian',
            'serbian',
            'spanish',
            'swedish',
            'tamil',
            'turkish',
            'yiddish',
        }
        assert len(valid_languages) == 29, 'Should have exactly 29 valid PostgreSQL text search configurations'

        # Verify all are accepted
        for lang in valid_languages:
            with env_var('FTS_LANGUAGE', lang):
                settings = AppSettings()
                assert settings.fts_language == lang
