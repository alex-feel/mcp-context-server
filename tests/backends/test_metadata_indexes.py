"""
Tests for configurable metadata field indexing.

Tests cover:
- Settings parsing (METADATA_INDEXED_FIELDS)
- Sync mode validation (METADATA_INDEX_SYNC_MODE)
- SQL generation for index creation
- Index detection and management
- Sync mode behaviors (strict, auto, warn, additive)
"""

import logging
import os
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.settings import StorageSettings


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


# ============================================================================
# SQL Generation Tests
# ============================================================================


class TestSQLGeneration:
    """Tests for SQL index creation statement generation."""

    def test_generate_create_index_sqlite(self) -> None:
        """Test SQLite CREATE INDEX SQL generation."""
        from app.migrations.metadata import _generate_create_index_sqlite

        sql = _generate_create_index_sqlite('status')
        assert 'CREATE INDEX IF NOT EXISTS idx_metadata_status' in sql
        assert "json_extract(metadata, '$.status')" in sql
        assert 'WHERE' in sql
        assert 'IS NOT NULL' in sql

    def test_generate_create_index_sqlite_field_names(self) -> None:
        """Test SQLite SQL generation for various field names."""
        from app.migrations.metadata import _generate_create_index_sqlite

        fields = ['agent_name', 'task_name', 'priority', 'some_field_123']
        for field in fields:
            sql = _generate_create_index_sqlite(field)
            assert f'idx_metadata_{field}' in sql
            assert f"'$.{field}'" in sql

    def test_generate_create_index_postgresql_string(self) -> None:
        """Test PostgreSQL CREATE INDEX SQL generation for string type."""
        from app.migrations.metadata import _generate_create_index_postgresql

        sql = _generate_create_index_postgresql('status', 'string')
        # The generated index identifier is quoted so PostgreSQL stores it case-preserved.
        assert 'CREATE INDEX IF NOT EXISTS "idx_metadata_status"' in sql
        assert "metadata->>'status'" in sql
        assert '::' not in sql  # No type casting for strings

    def test_generate_create_index_postgresql_integer(self) -> None:
        """Test PostgreSQL CREATE INDEX SQL generation for integer type."""
        from app.migrations.metadata import _generate_create_index_postgresql

        sql = _generate_create_index_postgresql('priority', 'integer')
        assert 'CREATE INDEX IF NOT EXISTS "idx_metadata_priority"' in sql
        assert "metadata->>'priority'" in sql
        assert '::INTEGER' in sql

    def test_generate_create_index_postgresql_boolean(self) -> None:
        """Test PostgreSQL CREATE INDEX SQL generation for boolean type."""
        from app.migrations.metadata import _generate_create_index_postgresql

        sql = _generate_create_index_postgresql('completed', 'boolean')
        assert 'CREATE INDEX IF NOT EXISTS "idx_metadata_completed"' in sql
        assert '::BOOLEAN' in sql

    def test_generate_create_index_postgresql_float(self) -> None:
        """Test PostgreSQL CREATE INDEX SQL generation for float type."""
        from app.migrations.metadata import _generate_create_index_postgresql

        sql = _generate_create_index_postgresql('score', 'float')
        assert 'CREATE INDEX IF NOT EXISTS "idx_metadata_score"' in sql
        assert '::NUMERIC' in sql

    def test_generate_create_index_postgresql_quotes_mixed_case_identifier(self) -> None:
        """A mixed-case field yields a quoted, case-preserved index identifier on PostgreSQL.

        PostgreSQL folds an UNQUOTED identifier to lowercase in the catalog. If the generated
        idx_metadata_{field} were emitted unquoted, pg_indexes would report the lowercased name
        while the reconciliation diff compares against the case-preserved configured field, so the
        same index would look simultaneously missing and extra forever (strict-mode boot failure,
        auto-mode drop/recreate churn). Quoting the identifier makes it round-trip case-preserved.
        """
        from app.migrations.metadata import _generate_create_index_postgresql

        sql = _generate_create_index_postgresql('camelField', 'string')

        # The generated identifier is double-quoted so PostgreSQL stores it case-preserved.
        assert '"idx_metadata_camelField"' in sql
        # The bare (fold-susceptible) form must NOT appear as the CREATE target.
        assert 'CREATE INDEX IF NOT EXISTS idx_metadata_camelField' not in sql
        # The JSON key accessor stays case-sensitive (JSON keys are case-sensitive), so the
        # mixed-case field name is preserved verbatim in the metadata->> expression.
        assert "metadata->>'camelField'" in sql

    def test_generate_create_index_postgresql_quotes_typed_mixed_case_identifier(self) -> None:
        """The quoted identifier is also emitted on the typed-cast CREATE branch."""
        from app.migrations.metadata import _generate_create_index_postgresql

        sql = _generate_create_index_postgresql('Priority', 'integer')

        assert '"idx_metadata_Priority"' in sql
        assert 'CREATE INDEX IF NOT EXISTS idx_metadata_Priority' not in sql
        assert '::INTEGER' in sql


# ============================================================================
# Index Detection Tests
# ============================================================================


class TestIndexDetection:
    """Tests for detecting existing metadata indexes in database."""

    @pytest_asyncio.fixture
    async def sqlite_backend_with_schema(self, tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
        """Create SQLite backend with schema and the default metadata indexes.

        The base schema no longer declares any idx_metadata_* index (that is the
        single responsibility of handle_metadata_indexes). To keep the detection
        assertions meaningful, this fixture provisions the five default scalar
        indexes the way server startup does, via _create_metadata_index, right
        after applying the base schema.

        Yields:
            An initialized SQLite backend with the default metadata indexes.
        """
        from app.backends import create_backend
        from app.migrations.metadata import _create_metadata_index

        db_path = tmp_path / 'test_indexes.db'
        backend = create_backend(backend_type='sqlite', db_path=db_path)
        await backend.initialize()

        # Apply base schema
        schema_path = Path(__file__).parent.parent.parent / 'app' / 'schemas' / 'sqlite_schema.sql'
        schema_sql = schema_path.read_text()

        def apply_schema(conn: sqlite3.Connection) -> None:
            conn.executescript(schema_sql)

        await backend.execute_write(apply_schema)

        # Provision the default scalar metadata indexes (formerly baked into the
        # base schema) so detection tests observe the same set as a running
        # server whose startup called handle_metadata_indexes.
        for field in ('status', 'agent_name', 'task_name', 'project', 'report_type'):
            await _create_metadata_index(backend, field, 'string')

        yield backend

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_get_existing_metadata_indexes_sqlite(
        self, sqlite_backend_with_schema: StorageBackend,
    ) -> None:
        """Test detection of existing metadata indexes in SQLite."""
        from app.migrations.metadata import _get_existing_metadata_indexes

        existing, orphan_compound = await _get_existing_metadata_indexes(sqlite_backend_with_schema)

        # Should detect the indexes from sqlite_schema.sql
        assert 'status' in existing
        assert 'agent_name' in existing
        assert 'task_name' in existing
        assert 'project' in existing
        assert 'report_type' in existing
        # No orphan compound indexes in fresh schema
        assert orphan_compound == set()

    @pytest.mark.asyncio
    async def test_get_existing_metadata_indexes_detects_compound_as_orphan(
        self, sqlite_backend_with_schema: StorageBackend,
    ) -> None:
        """Test that compound indexes (idx_thread_metadata_*) are detected as orphans."""
        from app.migrations.metadata import _get_existing_metadata_indexes

        existing, orphan_compound = await _get_existing_metadata_indexes(sqlite_backend_with_schema)

        # Simple indexes should be detected
        assert 'status' in existing
        assert 'agent_name' in existing

        # No compound indexes in fresh schema
        # Any idx_thread_metadata_* would be detected as orphan
        assert orphan_compound == set()

    @pytest.mark.asyncio
    async def test_get_existing_metadata_indexes_empty_database(self, tmp_path: Path) -> None:
        """Test detection returns empty sets for database without metadata indexes."""
        from app.backends import create_backend
        from app.migrations.metadata import _get_existing_metadata_indexes

        db_path = tmp_path / 'test_empty.db'
        backend = create_backend(backend_type='sqlite', db_path=db_path)
        await backend.initialize()

        # Create table but no metadata indexes
        def create_table(conn: sqlite3.Connection) -> None:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS context_entries (
                    id INTEGER PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    text_content TEXT,
                    metadata JSON
                )
            ''')

        await backend.execute_write(create_table)

        existing, orphan_compound = await _get_existing_metadata_indexes(backend)
        assert existing == set()
        assert orphan_compound == set()

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_compound_indexes_detected_as_orphans(self, tmp_path: Path) -> None:
        """Test that idx_thread_metadata_* indexes are detected as orphans."""
        from app.backends import create_backend
        from app.migrations.metadata import _get_existing_metadata_indexes

        db_path = tmp_path / 'test_compound_orphan.db'
        backend = create_backend(backend_type='sqlite', db_path=db_path)
        await backend.initialize()

        # Create table and a compound index manually (simulating old schema)
        def setup_with_compound_index(conn: sqlite3.Connection) -> None:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS context_entries (
                    id INTEGER PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    text_content TEXT,
                    metadata JSON
                )
            ''')
            # Create simple index
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metadata_status
                ON context_entries(json_extract(metadata, '$.status'))
            ''')
            # Create compound index (should be detected as orphan)
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_thread_metadata_status
                ON context_entries(thread_id, json_extract(metadata, '$.status'))
            ''')

        await backend.execute_write(setup_with_compound_index)

        existing, orphan_compound = await _get_existing_metadata_indexes(backend)

        # Simple index detected
        assert 'status' in existing
        # Compound index detected as orphan
        assert 'status' in orphan_compound

        await backend.shutdown()


# ============================================================================
# Index Creation/Drop Tests
# ============================================================================


class TestIndexCreation:
    """Tests for creating and dropping metadata indexes."""

    @pytest_asyncio.fixture
    async def sqlite_backend_with_table(self, tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
        """Create SQLite backend with just the context_entries table."""
        from app.backends import create_backend

        db_path = tmp_path / 'test_creation.db'
        backend = create_backend(backend_type='sqlite', db_path=db_path)
        await backend.initialize()

        # Create minimal table for testing
        def create_table(conn: sqlite3.Connection) -> None:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS context_entries (
                    id INTEGER PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    text_content TEXT,
                    metadata JSON
                )
            ''')

        await backend.execute_write(create_table)

        yield backend

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_create_metadata_index_sqlite(self, sqlite_backend_with_table: StorageBackend) -> None:
        """Test creating a metadata index in SQLite."""
        from app.migrations.metadata import _create_metadata_index
        from app.migrations.metadata import _get_existing_metadata_indexes

        # Initially no indexes
        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        assert 'test_field' not in existing

        # Create index
        await _create_metadata_index(sqlite_backend_with_table, 'test_field', 'string')

        # Verify index exists
        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        assert 'test_field' in existing

    @pytest.mark.asyncio
    async def test_create_metadata_index_skips_array_for_sqlite(
        self, sqlite_backend_with_table: StorageBackend, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that array type indexes are skipped in SQLite."""
        from app.migrations.metadata import _create_metadata_index
        from app.migrations.metadata import _get_existing_metadata_indexes

        with caplog.at_level(logging.INFO):
            await _create_metadata_index(sqlite_backend_with_table, 'technologies', 'array')

        # Index should NOT be created
        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        assert 'technologies' not in existing

        # Log message should explain why
        assert 'Skipping index for array field' in caplog.text

    @pytest.mark.asyncio
    async def test_create_metadata_index_skips_object_for_sqlite(
        self, sqlite_backend_with_table: StorageBackend, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that object type indexes are skipped in SQLite."""
        from app.migrations.metadata import _create_metadata_index
        from app.migrations.metadata import _get_existing_metadata_indexes

        with caplog.at_level(logging.INFO):
            await _create_metadata_index(sqlite_backend_with_table, 'references', 'object')

        # Index should NOT be created
        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        assert 'references' not in existing

        # Log message should explain why
        assert 'Skipping index for object field' in caplog.text

    @pytest.mark.asyncio
    async def test_drop_metadata_index_sqlite(self, sqlite_backend_with_table: StorageBackend) -> None:
        """Test dropping a metadata index in SQLite."""
        from app.migrations.metadata import _create_metadata_index
        from app.migrations.metadata import _drop_metadata_index
        from app.migrations.metadata import _get_existing_metadata_indexes

        # Create index first
        await _create_metadata_index(sqlite_backend_with_table, 'test_field', 'string')
        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        assert 'test_field' in existing

        # Drop index
        await _drop_metadata_index(sqlite_backend_with_table, 'test_field')

        # Verify index no longer exists
        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        assert 'test_field' not in existing

    @pytest.mark.asyncio
    async def test_drop_nonexistent_index_succeeds(self, sqlite_backend_with_table: StorageBackend) -> None:
        """Test that dropping a non-existent index doesn't raise error."""
        from app.migrations.metadata import _drop_metadata_index

        # Should not raise error
        await _drop_metadata_index(sqlite_backend_with_table, 'nonexistent_field')


# ============================================================================
# Sync Mode Behavior Tests
# ============================================================================


class TestSyncModes:
    """Tests for metadata index sync mode behaviors."""

    @pytest_asyncio.fixture
    async def sqlite_backend_with_table(self, tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
        """Create SQLite backend with just the context_entries table."""
        from app.backends import create_backend

        db_path = tmp_path / 'test_sync.db'
        backend = create_backend(backend_type='sqlite', db_path=db_path)
        await backend.initialize()

        # Create minimal table for testing
        def create_table(conn: sqlite3.Connection) -> None:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS context_entries (
                    id INTEGER PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    text_content TEXT,
                    metadata JSON
                )
            ''')

        await backend.execute_write(create_table)

        yield backend

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_additive_mode_creates_missing_indexes(
        self, sqlite_backend_with_table: StorageBackend,
    ) -> None:
        """Test additive mode creates missing indexes."""
        from app.migrations import handle_metadata_indexes
        from app.migrations.metadata import _get_existing_metadata_indexes

        # Initially no indexes
        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        assert len(existing) == 0

        with patch('app.migrations.metadata.settings') as mock_settings:
            mock_settings.storage.metadata_indexed_fields = {'status': 'string', 'agent_name': 'string'}
            mock_settings.storage.metadata_index_sync_mode = 'additive'

            await handle_metadata_indexes(sqlite_backend_with_table)

        # Indexes should be created
        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        assert 'status' in existing
        assert 'agent_name' in existing

    @pytest.mark.asyncio
    async def test_additive_mode_preserves_extra_indexes(
        self, sqlite_backend_with_table: StorageBackend,
    ) -> None:
        """Test additive mode does not drop extra indexes."""
        from app.migrations import handle_metadata_indexes
        from app.migrations.metadata import _create_metadata_index
        from app.migrations.metadata import _get_existing_metadata_indexes

        # Create an extra index that's not in config
        await _create_metadata_index(sqlite_backend_with_table, 'extra_field', 'string')
        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        assert 'extra_field' in existing

        with patch('app.migrations.metadata.settings') as mock_settings:
            mock_settings.storage.metadata_indexed_fields = {'status': 'string'}
            mock_settings.storage.metadata_index_sync_mode = 'additive'

            await handle_metadata_indexes(sqlite_backend_with_table)

        # Extra index should still exist
        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        assert 'extra_field' in existing
        assert 'status' in existing

    @pytest.mark.asyncio
    async def test_auto_mode_adds_missing_and_drops_extra(
        self, sqlite_backend_with_table: StorageBackend,
    ) -> None:
        """Test auto mode both adds missing and drops extra indexes."""
        from app.migrations import handle_metadata_indexes
        from app.migrations.metadata import _create_metadata_index
        from app.migrations.metadata import _get_existing_metadata_indexes

        # Create an extra index
        await _create_metadata_index(sqlite_backend_with_table, 'extra_field', 'string')

        with patch('app.migrations.metadata.settings') as mock_settings:
            mock_settings.storage.metadata_indexed_fields = {'status': 'string', 'agent_name': 'string'}
            mock_settings.storage.metadata_index_sync_mode = 'auto'

            await handle_metadata_indexes(sqlite_backend_with_table)

        # Extra index should be dropped, configured indexes should exist
        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        assert 'extra_field' not in existing
        assert 'status' in existing
        assert 'agent_name' in existing

    @pytest.mark.asyncio
    async def test_strict_mode_raises_on_missing_index(
        self, sqlite_backend_with_table: StorageBackend,
    ) -> None:
        """Test strict mode raises ConfigurationError on missing indexes.

        A strict-mode mismatch is a permanent, human-intervention-required
        misconfiguration, so it is classified as a configuration error (exit 78) that
        supervisors must not auto-retry, not a generic failure they crash-loop on.
        """
        from app.errors import ConfigurationError
        from app.migrations import handle_metadata_indexes

        with patch('app.migrations.metadata.settings') as mock_settings:
            mock_settings.storage.metadata_indexed_fields = {'status': 'string'}
            mock_settings.storage.metadata_index_sync_mode = 'strict'

            with pytest.raises(ConfigurationError) as exc_info:
                await handle_metadata_indexes(sqlite_backend_with_table)

            assert 'Metadata index mismatch' in str(exc_info.value)
            assert 'METADATA_INDEX_SYNC_MODE=strict' in str(exc_info.value)
            assert exc_info.value.EXIT_CODE == 78

    @pytest.mark.asyncio
    async def test_strict_mode_raises_on_extra_index(
        self, sqlite_backend_with_table: StorageBackend,
    ) -> None:
        """Test strict mode raises ConfigurationError (exit 78) on extra indexes."""
        from app.errors import ConfigurationError
        from app.migrations import handle_metadata_indexes
        from app.migrations.metadata import _create_metadata_index

        # Create the expected index AND an extra one
        await _create_metadata_index(sqlite_backend_with_table, 'status', 'string')
        await _create_metadata_index(sqlite_backend_with_table, 'extra_field', 'string')

        with patch('app.migrations.metadata.settings') as mock_settings:
            mock_settings.storage.metadata_indexed_fields = {'status': 'string'}
            mock_settings.storage.metadata_index_sync_mode = 'strict'

            with pytest.raises(ConfigurationError) as exc_info:
                await handle_metadata_indexes(sqlite_backend_with_table)

            assert 'Metadata index mismatch' in str(exc_info.value)
            assert 'Extra' in str(exc_info.value)
            assert exc_info.value.EXIT_CODE == 78

    @pytest.mark.asyncio
    async def test_strict_mode_succeeds_when_indexes_match(
        self, sqlite_backend_with_table: StorageBackend,
    ) -> None:
        """Test strict mode succeeds when indexes match configuration exactly."""
        from app.migrations import handle_metadata_indexes
        from app.migrations.metadata import _create_metadata_index

        # Create exactly the expected indexes
        await _create_metadata_index(sqlite_backend_with_table, 'status', 'string')
        await _create_metadata_index(sqlite_backend_with_table, 'agent_name', 'string')

        with patch('app.migrations.metadata.settings') as mock_settings:
            mock_settings.storage.metadata_indexed_fields = {'status': 'string', 'agent_name': 'string'}
            mock_settings.storage.metadata_index_sync_mode = 'strict'

            # Should not raise
            await handle_metadata_indexes(sqlite_backend_with_table)

    @pytest.mark.asyncio
    async def test_warn_mode_logs_but_continues(
        self, sqlite_backend_with_table: StorageBackend, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warn mode logs warnings but continues startup."""
        from app.migrations import handle_metadata_indexes

        with caplog.at_level(logging.WARNING), patch('app.migrations.metadata.settings') as mock_settings:
            mock_settings.storage.metadata_indexed_fields = {'status': 'string'}
            mock_settings.storage.metadata_index_sync_mode = 'warn'

            # Should not raise, should log warning
            await handle_metadata_indexes(sqlite_backend_with_table)

        assert 'Missing metadata indexes' in caplog.text

    @pytest.mark.asyncio
    async def test_array_object_fields_excluded_from_sync(
        self, sqlite_backend_with_table: StorageBackend,
    ) -> None:
        """Test array and object fields are excluded from index sync comparison."""
        from app.migrations import handle_metadata_indexes
        from app.migrations.metadata import _get_existing_metadata_indexes

        with patch('app.migrations.metadata.settings') as mock_settings:
            mock_settings.storage.metadata_indexed_fields = {
                'status': 'string',
                'technologies': 'array',  # Should be excluded
                'references': 'object',  # Should be excluded
            }
            mock_settings.storage.metadata_index_sync_mode = 'additive'

            await handle_metadata_indexes(sqlite_backend_with_table)

        # Only 'status' should be created (array/object skipped)
        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        assert 'status' in existing
        assert 'technologies' not in existing
        assert 'references' not in existing


# ============================================================================
# Integration Tests
# ============================================================================


class TestMetadataIndexingIntegration:
    """Integration tests for metadata indexing with full server lifecycle."""

    @pytest.mark.asyncio
    async def test_handle_metadata_indexes_idempotent(self, tmp_path: Path) -> None:
        """Test that handle_metadata_indexes can be called multiple times."""
        from app.backends import create_backend
        from app.migrations import handle_metadata_indexes
        from app.migrations.metadata import _get_existing_metadata_indexes

        db_path = tmp_path / 'test_idempotent.db'
        backend = create_backend(backend_type='sqlite', db_path=db_path)
        await backend.initialize()

        # Create table
        def create_table(conn: sqlite3.Connection) -> None:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS context_entries (
                    id INTEGER PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    text_content TEXT,
                    metadata JSON
                )
            ''')

        await backend.execute_write(create_table)

        with patch('app.migrations.metadata.settings') as mock_settings:
            mock_settings.storage.metadata_indexed_fields = {'status': 'string', 'agent_name': 'string'}
            mock_settings.storage.metadata_index_sync_mode = 'additive'

            # Call multiple times
            await handle_metadata_indexes(backend)
            await handle_metadata_indexes(backend)
            await handle_metadata_indexes(backend)

        # Should still have correct indexes
        existing, _ = await _get_existing_metadata_indexes(backend)
        assert 'status' in existing
        assert 'agent_name' in existing

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_default_fields_includes_all_required(self) -> None:
        """Test that DEFAULT_METADATA_INDEXED_FIELDS includes all required fields."""
        settings = StorageSettings()
        fields = settings.metadata_indexed_fields

        # Check all context-preservation-protocol required fields
        required_fields = ['status', 'agent_name', 'task_name', 'project', 'report_type']
        for field in required_fields:
            assert field in fields, f'Required field {field} missing from defaults'

        # Check array/object fields have correct types
        assert fields.get('technologies') == 'array'
        assert fields.get('references') == 'object'


# ============================================================================
# Mixed-Case Field Reconciliation Round-Trip
# ============================================================================


class TestMixedCaseFieldReconciliation:
    """A mixed-case metadata field must round-trip cleanly through reconciliation.

    The created index identifier and the name read back from the catalog must
    resolve to the SAME configured field, so a single startup run leaves the
    diff empty (no perpetual drop/recreate in auto mode, no strict-mode false
    positive). On PostgreSQL this depends on the quoted identifier landing
    case-preserved in the catalog; on SQLite the identifier is preserved without
    quoting. The behavior is exercised directly on SQLite here.
    """

    @pytest_asyncio.fixture
    async def sqlite_backend_with_table(self, tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
        """Create a SQLite backend with just the context_entries table.

        Yields:
            An initialized SQLite backend with an empty context_entries table.
        """
        from app.backends import create_backend

        db_path = tmp_path / 'test_mixed_case.db'
        backend = create_backend(backend_type='sqlite', db_path=db_path)
        await backend.initialize()

        def create_table(conn: sqlite3.Connection) -> None:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS context_entries (
                    id INTEGER PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    text_content TEXT,
                    metadata JSON
                )
            ''')

        await backend.execute_write(create_table)

        yield backend

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_created_index_round_trips_to_configured_field(
        self, sqlite_backend_with_table: StorageBackend,
    ) -> None:
        """A mixed-case field creates an index whose detected name equals the field."""
        from app.migrations.metadata import _create_metadata_index
        from app.migrations.metadata import _get_existing_metadata_indexes

        await _create_metadata_index(sqlite_backend_with_table, 'camelField', 'string')

        existing, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)
        # The detected field name matches the configured (case-preserved) field, so the
        # reconciliation diff can pair them and never report the same index as both
        # missing and extra.
        assert 'camelField' in existing

    @pytest.mark.asyncio
    async def test_additive_mode_is_stable_across_reboots_for_mixed_case(
        self, sqlite_backend_with_table: StorageBackend,
    ) -> None:
        """Repeated additive runs on a mixed-case field cause no drop/recreate churn."""
        from app.migrations import handle_metadata_indexes
        from app.migrations.metadata import _get_existing_metadata_indexes

        with patch('app.migrations.metadata.settings') as mock_settings:
            mock_settings.storage.metadata_indexed_fields = {'camelField': 'string'}
            mock_settings.storage.metadata_index_sync_mode = 'additive'

            await handle_metadata_indexes(sqlite_backend_with_table)
            first, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)

            # A second run must observe the SAME index already present, meaning the
            # first run's created name round-tripped and the diff is empty.
            await handle_metadata_indexes(sqlite_backend_with_table)
            second, _ = await _get_existing_metadata_indexes(sqlite_backend_with_table)

        assert first == {'camelField'}
        assert second == {'camelField'}

    @pytest.mark.asyncio
    async def test_strict_mode_passes_when_mixed_case_field_indexed(
        self, sqlite_backend_with_table: StorageBackend,
    ) -> None:
        """Strict mode does not raise a false positive for an already-indexed mixed-case field."""
        from app.migrations import handle_metadata_indexes
        from app.migrations.metadata import _create_metadata_index

        # Provision the mixed-case index exactly as the additive/startup path would.
        await _create_metadata_index(sqlite_backend_with_table, 'camelField', 'string')

        with patch('app.migrations.metadata.settings') as mock_settings:
            mock_settings.storage.metadata_indexed_fields = {'camelField': 'string'}
            mock_settings.storage.metadata_index_sync_mode = 'strict'

            # The created identifier round-trips to the configured field, so strict
            # reconciliation sees neither a missing nor an extra index and must not raise.
            await handle_metadata_indexes(sqlite_backend_with_table)


# ============================================================================
# Base Schema Metadata-Index Absence
# ============================================================================


class TestBaseSchemaHasNoMetadataIndexes:
    """The base schema must not declare any metadata expression index.

    handle_metadata_indexes is the single source of truth for the configurable
    idx_metadata_* set. If the base schema also created them, a fresh database
    configured with a custom METADATA_INDEXED_FIELDS would boot with the default
    indexes that strict-mode reconciliation then rejects as "extra" (and auto
    mode would create-then-drop on every fresh init).
    """

    @pytest.mark.asyncio
    async def test_fresh_sqlite_schema_has_no_metadata_indexes(self, tmp_path: Path) -> None:
        """Applying only the base SQLite schema leaves no idx_metadata_* index behind."""
        from app.backends import create_backend
        from app.migrations.metadata import _get_existing_metadata_indexes

        db_path = tmp_path / 'test_base_schema.db'
        backend = create_backend(backend_type='sqlite', db_path=db_path)
        await backend.initialize()

        schema_path = Path(__file__).parent.parent.parent / 'app' / 'schemas' / 'sqlite_schema.sql'
        schema_sql = schema_path.read_text()

        def apply_schema(conn: sqlite3.Connection) -> None:
            conn.executescript(schema_sql)

        await backend.execute_write(apply_schema)

        # The base schema declares no metadata expression index; the sync layer has
        # not run yet, so detection must report an empty set.
        existing, orphan_compound = await _get_existing_metadata_indexes(backend)
        assert existing == set()
        assert orphan_compound == set()

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_startup_provisioning_creates_default_indexes_on_fresh_schema(
        self, tmp_path: Path,
    ) -> None:
        """After the base schema, handle_metadata_indexes provisions the default set."""
        from app.backends import create_backend
        from app.migrations import handle_metadata_indexes
        from app.migrations.metadata import _get_existing_metadata_indexes

        db_path = tmp_path / 'test_base_then_sync.db'
        backend = create_backend(backend_type='sqlite', db_path=db_path)
        await backend.initialize()

        schema_path = Path(__file__).parent.parent.parent / 'app' / 'schemas' / 'sqlite_schema.sql'
        schema_sql = schema_path.read_text()

        def apply_schema(conn: sqlite3.Connection) -> None:
            conn.executescript(schema_sql)

        await backend.execute_write(apply_schema)

        # Fresh schema starts with no metadata indexes.
        existing, _ = await _get_existing_metadata_indexes(backend)
        assert existing == set()

        # The sync layer (mirroring server startup) provisions the default scalar set.
        with patch('app.migrations.metadata.settings') as mock_settings:
            mock_settings.storage.metadata_indexed_fields = {
                'status': 'string',
                'agent_name': 'string',
                'task_name': 'string',
                'project': 'string',
                'report_type': 'string',
            }
            mock_settings.storage.metadata_index_sync_mode = 'additive'
            await handle_metadata_indexes(backend)

        existing, _ = await _get_existing_metadata_indexes(backend)
        assert existing == {'status', 'agent_name', 'task_name', 'project', 'report_type'}

        await backend.shutdown()

    def test_base_schema_files_declare_no_scalar_metadata_index(self) -> None:
        """Neither base schema file contains a hardcoded idx_metadata_{scalar} CREATE INDEX.

        This static check guards the single-source-of-truth invariant directly against the
        schema files, so a future edit that reintroduces a baked-in scalar metadata index is
        caught even if no fresh-init test happens to exercise that field.
        """
        schemas_dir = Path(__file__).parent.parent.parent / 'app' / 'schemas'
        for filename in ('sqlite_schema.sql', 'postgresql_schema.sql'):
            sql = (schemas_dir / filename).read_text()
            for scalar in ('status', 'agent_name', 'task_name', 'project', 'report_type'):
                needle = f'idx_metadata_{scalar}'
                assert needle not in sql, (
                    f'{filename} still declares {needle!r}; scalar metadata indexes must be '
                    f'provisioned exclusively by handle_metadata_indexes.'
                )
            # The PostgreSQL GIN index is NOT sync-managed and legitimately stays in the schema.
            if filename == 'postgresql_schema.sql':
                assert 'idx_metadata_gin' in sql
