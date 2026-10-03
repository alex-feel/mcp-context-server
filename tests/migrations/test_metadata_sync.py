"""Tests for handle_metadata_indexes in app/migrations/metadata.py: METADATA_INDEX_SYNC_MODE behaviors
(additive, auto, strict, warn), idempotent reconciliation, mixed-case field round-trips, and default index
provisioning on a fresh base schema.
"""

import logging
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.settings.storage import StorageSettings

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
