"""ContextRepository reads return the public CONTEXT_ENTRY_COLUMNS and no internal database columns."""

import sqlite3
from pathlib import Path

import pytest

from tests.helpers import LOCAL_SCOPE


class TestInternalColumnsNotExposed:
    """Test that internal database columns are not exposed in API responses.

    PostgreSQL uses text_search_vector (tsvector) for FTS, which is an internal
    implementation detail. These tests verify that explicit column listing in
    search_contexts() and get_by_ids() prevents internal columns from leaking.
    """

    @pytest.fixture
    def test_db_path(self, tmp_path: Path) -> Path:
        """Create a database for testing internal column exposure.

        Returns:
            Path to the test database.
        """
        db_path = tmp_path / 'test_internal_columns.db'

        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(schema_sql)

            # Insert test data
            conn.execute(
                '''
                INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
                VALUES (
                    '0190abcdef1234567890abcd0000000a',
                    'test-thread',
                    'agent',
                    'text',
                    'Test content for column exposure test'
                , 'local')
            ''',
            )
            conn.commit()

        return db_path

    @pytest.mark.asyncio
    async def test_search_contexts_does_not_expose_internal_columns(self, test_db_path: Path) -> None:
        """Test that search_contexts() does not return internal columns like text_search_vector.

        Explicit column listing via CONTEXT_ENTRY_COLUMNS keeps internal PostgreSQL
        columns (the text_search_vector tsvector) out of API responses.
        """
        from app.backends.sqlite_backend import SQLiteBackend
        from app.repositories.context_repository import ContextRepository

        # Initialize backend and repository
        backend = SQLiteBackend(db_path=str(test_db_path))
        await backend.initialize()

        try:
            repo = ContextRepository(backend)

            # Call search_contexts
            rows, _stats = await repo.search_contexts(thread_id='test-thread', scope=LOCAL_SCOPE)

            # Verify we got results
            assert len(rows) == 1

            # Convert Row to dict to check keys
            row_dict = dict(rows[0])

            # Verify expected columns ARE present
            expected_columns = {
                'id',
                'thread_id',
                'source',
                'content_type',
                'text_content',
                'metadata',
                'created_at',
                'updated_at',
            }
            for col in expected_columns:
                assert col in row_dict, f'Expected column {col} missing from result'

            # Verify internal columns are NOT present
            internal_columns = {'text_search_vector', 'fts_vector', 'tsvector'}
            for col in internal_columns:
                assert col not in row_dict, f'Internal column {col} should not be exposed in API response'

        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_get_by_ids_does_not_expose_internal_columns(self, test_db_path: Path) -> None:
        """Test that get_by_ids() does not return internal columns like text_search_vector.

        get_by_ids() uses the same explicit column listing, so internal database
        columns stay out of its rows.
        """
        from app.backends.sqlite_backend import SQLiteBackend
        from app.repositories.context_repository import ContextRepository

        # Initialize backend and repository
        backend = SQLiteBackend(db_path=str(test_db_path))
        await backend.initialize()

        try:
            repo = ContextRepository(backend)

            # First get the ID of the test entry
            rows, _stats = await repo.search_contexts(thread_id='test-thread', scope=LOCAL_SCOPE)
            assert len(rows) == 1
            context_id = rows[0]['id']

            # Call get_by_ids
            result_rows = await repo.get_by_ids([context_id], scope=LOCAL_SCOPE)

            # Verify we got results
            assert len(result_rows) == 1

            # Convert Row to dict to check keys
            row_dict = dict(result_rows[0])

            # Verify expected columns ARE present
            expected_columns = {
                'id',
                'thread_id',
                'source',
                'content_type',
                'text_content',
                'metadata',
                'created_at',
                'updated_at',
            }
            for col in expected_columns:
                assert col in row_dict, f'Expected column {col} missing from result'

            # Verify internal columns are NOT present
            internal_columns = {'text_search_vector', 'fts_vector', 'tsvector'}
            for col in internal_columns:
                assert col not in row_dict, f'Internal column {col} should not be exposed in API response'

        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_context_entry_columns_constant_matches_expected(self) -> None:
        """Test that CONTEXT_ENTRY_COLUMNS constant includes all expected columns.

        This test ensures the column constant is maintained correctly and includes
        all columns defined in the ContextEntryDict TypedDict.
        """
        from app.repositories.context_repository.records import CONTEXT_ENTRY_COLUMNS

        # Parse the column string into a set
        columns = {col.strip() for col in CONTEXT_ENTRY_COLUMNS.split(',')}

        # Verify all expected columns are present
        expected_columns = {
            'id',
            'thread_id',
            'source',
            'content_type',
            'text_content',
            'metadata',
            'summary',
            'created_at',
            'updated_at',
        }

        assert columns == expected_columns, f'Column mismatch: expected {expected_columns}, got {columns}'

        # Verify internal columns are NOT in the constant
        internal_columns = {'text_search_vector', 'fts_vector', 'tsvector'}
        for col in internal_columns:
            assert col not in columns, f'Internal column {col} should not be in CONTEXT_ENTRY_COLUMNS'
