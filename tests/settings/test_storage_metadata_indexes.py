"""Tests for the StorageSettings metadata-index settings in app/settings/storage.py: METADATA_INDEXED_FIELDS
parsing and field-name validation, and the METADATA_INDEX_SYNC_MODE values.
"""

import logging

import pytest
from pydantic import ValidationError

from app.settings.storage import StorageSettings
from tests.helpers import env_var

# ============================================================================
# Settings Parsing Tests
# ============================================================================


class TestMetadataIndexedFieldsParsing:
    """Tests for METADATA_INDEXED_FIELDS parsing via metadata_indexed_fields property."""

    def test_parse_simple_fields(self) -> None:
        """Test parsing comma-separated fields without type hints."""
        with env_var('METADATA_INDEXED_FIELDS', 'status,agent_name,task_name'):
            settings = StorageSettings()
            result = settings.metadata_indexed_fields
            assert result == {
                'status': 'string',
                'agent_name': 'string',
                'task_name': 'string',
            }

    def test_parse_fields_with_type_hints(self) -> None:
        """Test parsing fields with explicit type hints."""
        with env_var('METADATA_INDEXED_FIELDS', 'status,priority:integer,completed:boolean,score:float'):
            settings = StorageSettings()
            result = settings.metadata_indexed_fields
            assert result == {
                'status': 'string',
                'priority': 'integer',
                'completed': 'boolean',
                'score': 'float',
            }

    def test_parse_array_and_object_types(self) -> None:
        """Test parsing array and object type hints."""
        with env_var('METADATA_INDEXED_FIELDS', 'technologies:array,references:object,tags:array'):
            settings = StorageSettings()
            result = settings.metadata_indexed_fields
            assert result == {
                'technologies': 'array',
                'references': 'object',
                'tags': 'array',
            }

    def test_parse_empty_string(self) -> None:
        """Test parsing empty string returns empty dict."""
        with env_var('METADATA_INDEXED_FIELDS', ''):
            settings = StorageSettings()
            assert settings.metadata_indexed_fields == {}

    def test_parse_whitespace_only(self) -> None:
        """Test parsing whitespace-only string returns empty dict."""
        with env_var('METADATA_INDEXED_FIELDS', '   \t\n  '):
            settings = StorageSettings()
            assert settings.metadata_indexed_fields == {}

    def test_parse_whitespace_handling(self) -> None:
        """Test whitespace is properly stripped from fields and type hints."""
        with env_var('METADATA_INDEXED_FIELDS', '  status  ,  priority : integer  ,  completed:boolean  '):
            settings = StorageSettings()
            result = settings.metadata_indexed_fields
            assert result == {
                'status': 'string',
                'priority': 'integer',
                'completed': 'boolean',
            }

    def test_invalid_type_hint_defaults_to_string(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test invalid type hints default to string with warning."""
        with (
            env_var('METADATA_INDEXED_FIELDS', 'status,priority:invalid,score:unknown'),
            caplog.at_level(logging.WARNING),
        ):
            settings = StorageSettings()
            result = settings.metadata_indexed_fields
            assert result == {
                'status': 'string',
                'priority': 'string',
                'score': 'string',
            }
            # Check that warnings were logged
            assert 'Invalid type hint "invalid" for field "priority"' in caplog.text
            assert 'Invalid type hint "unknown" for field "score"' in caplog.text

    def test_type_hint_case_insensitive(self) -> None:
        """Test type hints are normalized to lowercase."""
        with env_var('METADATA_INDEXED_FIELDS', 'status:STRING,priority:INTEGER,completed:Boolean'):
            settings = StorageSettings()
            result = settings.metadata_indexed_fields
            assert result == {
                'status': 'string',
                'priority': 'integer',
                'completed': 'boolean',
            }

    def test_default_metadata_indexed_fields(self) -> None:
        """Test the default METADATA_INDEXED_FIELDS value."""
        # Use the default value (no env var override)
        with env_var('METADATA_INDEXED_FIELDS', None):
            settings = StorageSettings()
            result = settings.metadata_indexed_fields
            # Verify default fields from context-preservation-protocol
            assert 'status' in result
            assert 'agent_name' in result
            assert 'task_name' in result
            assert 'project' in result
            assert 'report_type' in result
            assert result.get('references') == 'object'
            assert result.get('technologies') == 'array'

    def test_empty_fields_skipped(self) -> None:
        """Test that empty fields (from double commas) are skipped."""
        with env_var('METADATA_INDEXED_FIELDS', 'status,,priority:integer,,,completed:boolean'):
            settings = StorageSettings()
            result = settings.metadata_indexed_fields
            assert result == {
                'status': 'string',
                'priority': 'integer',
                'completed': 'boolean',
            }

    def test_field_with_multiple_colons(self) -> None:
        """Test field with multiple colons uses only first colon as separator."""
        with env_var('METADATA_INDEXED_FIELDS', 'field_name:string:extra'):
            settings = StorageSettings()
            result = settings.metadata_indexed_fields
            # 'string:extra' is treated as the type hint, which is invalid
            # so it should default to 'string'
            assert 'field_name' in result


class TestMetadataIndexedFieldNames:
    """METADATA_INDEXED_FIELDS field-name validation at the configuration boundary.

    Each configured field name is interpolated verbatim into a generated
    idx_metadata_<field> index name and JSON path/key literal on both backends, so
    the validator refuses names that would either crash schema startup or leave
    metadata-index reconciliation permanently unable to converge. Three constraints
    are enforced: the plain-SQL-identifier grammar, a 50-character length cap (so the
    13-character idx_metadata_ prefix leaves the generated name within PostgreSQL's
    63-byte identifier limit, which otherwise truncates the catalog name and diverges
    the reconciliation diff from SQLite), and case uniqueness under case folding (two
    case-differing names collide on SQLite's case-insensitive CREATE INDEX IF NOT
    EXISTS while PostgreSQL keeps them distinct, a cross-backend divergence).
    """

    def test_default_fields_accepted(self) -> None:
        from app.settings.storage import StorageSettings

        settings = StorageSettings()
        assert 'status' in settings.metadata_indexed_fields
        assert settings.metadata_indexed_fields['technologies'] == 'array'

    def test_invalid_grammar_field_rejected(self) -> None:
        from app.settings.storage import StorageSettings

        with env_var('METADATA_INDEXED_FIELDS', 'bad-field'), pytest.raises(ValidationError, match='invalid field name'):
            StorageSettings()

    def test_field_over_fifty_characters_rejected(self) -> None:
        """A 51-character field truncates in PostgreSQL's catalog once prefixed, so it is refused."""
        from app.settings.storage import StorageSettings

        field = 'a' * 51
        with env_var('METADATA_INDEXED_FIELDS', field), pytest.raises(ValidationError, match='at most 50'):
            StorageSettings()

    def test_field_at_fifty_characters_accepted(self) -> None:
        """The 50-character boundary keeps the generated idx_metadata_ name within 63 bytes."""
        from app.settings.storage import StorageSettings

        field = 'a' * 50
        with env_var('METADATA_INDEXED_FIELDS', field):
            assert field in StorageSettings().metadata_indexed_fields

    def test_casefold_colliding_fields_rejected(self) -> None:
        """Two names differing only in case collide on SQLite, so the config is refused."""
        from app.settings.storage import StorageSettings

        with (
            env_var('METADATA_INDEXED_FIELDS', 'Status,status'),
            pytest.raises(ValidationError, match='differ only in case'),
        ):
            StorageSettings()

    def test_exact_duplicate_field_rejected_with_duplicate_diagnostic(self) -> None:
        """An identical repeated name is refused with an accurate duplicate diagnostic.

        The rejection must NOT claim the two equal names 'differ only in case' --
        that message describes a nonexistent casing problem and misdirects the
        operator away from the actual duplicate entry.
        """
        from app.settings.storage import StorageSettings

        with (
            env_var('METADATA_INDEXED_FIELDS', 'status,status'),
            pytest.raises(ValidationError, match='more than once') as exc_info,
        ):
            StorageSettings()
        assert 'differ only in case' not in str(exc_info.value)

    def test_duplicate_field_with_conflicting_type_hints_rejected(self) -> None:
        """A repeated name carrying conflicting type hints gets the duplicate diagnostic."""
        from app.settings.storage import StorageSettings

        with (
            env_var('METADATA_INDEXED_FIELDS', 'status:string,status:integer'),
            pytest.raises(ValidationError, match='more than once'),
        ):
            StorageSettings()

    def test_distinct_fields_accepted(self) -> None:
        """Case-distinct-but-not-colliding names remain valid."""
        from app.settings.storage import StorageSettings

        with env_var('METADATA_INDEXED_FIELDS', 'status,agent_name'):
            fields = StorageSettings().metadata_indexed_fields
        assert 'status' in fields
        assert 'agent_name' in fields


# ============================================================================
# Sync Mode Validation Tests
# ============================================================================


class TestMetadataIndexSyncMode:
    """Tests for METADATA_INDEX_SYNC_MODE setting."""

    def test_valid_sync_modes(self) -> None:
        """Test all valid sync modes are accepted."""
        valid_modes = ['strict', 'auto', 'warn', 'additive']

        for mode in valid_modes:
            with env_var('METADATA_INDEX_SYNC_MODE', mode):
                settings = StorageSettings()
                assert settings.metadata_index_sync_mode == mode, f'Mode {mode} should be accepted'

    def test_default_sync_mode_is_additive(self) -> None:
        """Test the default sync mode is additive."""
        with env_var('METADATA_INDEX_SYNC_MODE', None):
            settings = StorageSettings()
            assert settings.metadata_index_sync_mode == 'additive'

    def test_invalid_sync_mode_raises_error(self) -> None:
        """Test that invalid sync modes raise validation error."""
        from pydantic import ValidationError

        invalid_modes = ['invalid', 'strict-mode', 'AUTO', 'Additive']

        for mode in invalid_modes:
            with env_var('METADATA_INDEX_SYNC_MODE', mode), pytest.raises(ValidationError):
                StorageSettings()
