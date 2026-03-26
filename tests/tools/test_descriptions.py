"""Tests for backend-specific tool description generators.

Tests verify that generate_fts_description produces correct output
for both SQLite and PostgreSQL backends across various FTS language
configurations, and that structural invariants of the description
template are maintained.
"""

from __future__ import annotations

from app.tools.descriptions import _FTS_DESCRIPTION_TEMPLATE
from app.tools.descriptions import _POSTGRESQL_MODE_DESCRIPTIONS
from app.tools.descriptions import _SQLITE_MODE_DESCRIPTIONS
from app.tools.descriptions import _generate_postgresql_fts_description
from app.tools.descriptions import _generate_sqlite_fts_description
from app.tools.descriptions import generate_fts_description


class TestGenerateFtsDescription:
    """Test the public generate_fts_description dispatcher."""

    def test_sqlite_english(self) -> None:
        result = generate_fts_description('sqlite', 'english')
        assert 'SQLite' in result
        assert 'FTS5' in result
        assert 'english' in result
        assert 'Stemming: ENABLED' in result
        assert 'Stop words: DISABLED' in result

    def test_sqlite_german(self) -> None:
        result = generate_fts_description('sqlite', 'german')
        assert 'SQLite' in result
        assert 'FTS5' in result
        assert 'Stemming: DISABLED' in result
        assert 'Stop words: DISABLED' in result

    def test_postgresql_english(self) -> None:
        result = generate_fts_description('postgresql', 'english')
        assert 'PostgreSQL' in result
        assert 'tsvector' in result
        assert 'english' in result
        assert 'Stemming: ENABLED' in result
        assert 'Stop words: ENABLED' in result

    def test_postgresql_german(self) -> None:
        result = generate_fts_description('postgresql', 'german')
        assert 'PostgreSQL' in result
        assert 'tsvector' in result
        assert 'Stemming: ENABLED' in result
        assert 'Stop words: ENABLED' in result
        assert 'Snowball' in result
        assert 'for german text' in result

    def test_postgresql_simple(self) -> None:
        result = generate_fts_description('postgresql', 'simple')
        assert 'Stemming: DISABLED' in result
        assert 'Stop words: DISABLED' in result
        assert 'simple configuration' in result


class TestSqliteFtsDescription:
    """Test SQLite-specific FTS description generation."""

    def test_english_stemming_enabled(self) -> None:
        result = _generate_sqlite_fts_description('english')
        assert 'ENABLED' in result
        assert 'Porter algorithm' in result
        assert '"running" WILL match "run"' in result

    def test_non_english_stemming_disabled(self) -> None:
        result = _generate_sqlite_fts_description('french')
        assert 'Stemming: DISABLED' in result
        assert 'Porter stemmer is English-only' in result
        assert 'does NOT match' in result


class TestPostgresqlFtsDescription:
    """Test PostgreSQL-specific FTS description generation."""

    def test_non_simple_stemming_enabled(self) -> None:
        result = _generate_postgresql_fts_description('spanish')
        assert 'Stemming: ENABLED' in result
        assert 'Snowball' in result
        assert 'for spanish text' in result
        assert 'Stop words: ENABLED' in result


class TestDescriptionStructure:
    """Test structural invariants of description output."""

    def test_sqlite_description_contains_all_four_modes(self) -> None:
        result = generate_fts_description('sqlite', 'english')
        for mode in ('match', 'prefix', 'phrase', 'boolean'):
            assert mode in result, f'{mode} missing in sqlite description'

    def test_postgresql_description_contains_all_four_modes(self) -> None:
        result = generate_fts_description('postgresql', 'english')
        for mode in ('match', 'prefix', 'phrase', 'boolean'):
            assert mode in result, f'{mode} missing in postgresql description'

    def test_mode_description_strings_contain_all_modes(self) -> None:
        for desc_name, desc in [
            ('_SQLITE_MODE_DESCRIPTIONS', _SQLITE_MODE_DESCRIPTIONS),
            ('_POSTGRESQL_MODE_DESCRIPTIONS', _POSTGRESQL_MODE_DESCRIPTIONS),
        ]:
            assert isinstance(desc, str)
            assert len(desc) > 0
            for mode in ('match', 'prefix', 'phrase', 'boolean'):
                assert mode in desc, f'{mode} missing in {desc_name}'

    def test_template_contains_all_format_fields(self) -> None:
        expected_fields = [
            '{backend}', '{engine}', '{language}',
            '{stemming_status}', '{stemming_detail}', '{stemming_example}',
            '{stopwords_status}', '{stopwords_detail}', '{stopwords_example}',
            '{non_language_behavior}', '{mode_descriptions}',
        ]
        for field in expected_fields:
            assert field in _FTS_DESCRIPTION_TEMPLATE, f'{field} missing in template'
