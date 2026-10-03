"""Tests for the metadata-index primitives in app/migrations/metadata.py: CREATE INDEX generation for
both backends, detection of existing metadata indexes (including compound orphans), and index creation and
removal on SQLite.
"""

import logging
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import StorageBackend

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

        The base schema declares no idx_metadata_* index (handle_metadata_indexes
        alone provisions them). To keep the detection assertions meaningful, this
        fixture provisions the five default scalar indexes the way server startup
        does, via _create_metadata_index, right after applying the base schema.

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

        # Provision the default scalar metadata indexes so detection tests
        # observe the same set as a running server whose startup called
        # handle_metadata_indexes.
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

        # Should detect the default indexes the fixture provisioned
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

        # Create the table and a compound idx_thread_metadata_* index manually
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
