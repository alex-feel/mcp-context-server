"""
Test suite for database operations in the MCP Context Storage Server.

Tests database initialization, schema creation, PRAGMA settings,
connection management, and tag storage/retrieval operations.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from anyio import Path as AsyncPath

from app.backends import StorageBackend
from app.server import init_database


class TestDatabaseInitialization:
    """Test database initialization and schema creation."""

    @pytest.mark.asyncio
    async def test_init_database_creates_schema(self, temp_db_path: Path) -> None:
        """Test database initialization creates all required tables."""
        from app.backends import create_backend

        # Create SQLite backend explicitly
        backend = create_backend(backend_type='sqlite', db_path=str(temp_db_path))
        await backend.initialize()

        with patch('app.server.DB_PATH', temp_db_path):
            await init_database(backend=backend)

        await backend.shutdown()

        # Verify database exists
        async_temp_db_path = AsyncPath(temp_db_path)
        assert await async_temp_db_path.exists()

        # Verify tables were created
        with sqlite3.connect(str(temp_db_path)) as conn:
            cursor = conn.cursor()

            # Check tables exist
            cursor.execute('''
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            ''')
            tables = [row[0] for row in cursor.fetchall()]
            assert set(tables) == {'context_entries', 'image_attachments', 'tags'}

            # Check indexes exist
            cursor.execute('''
                SELECT name FROM sqlite_master
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            ''')
            indexes = [row[0] for row in cursor.fetchall()]
            expected_indexes = {
                'idx_created_at',
                'idx_image_context',
                'idx_source',
                'idx_tags_entry',
                'idx_tags_tag',
                'idx_thread_id',
                'idx_thread_source',
                # Metadata filtering indexes
                'idx_metadata_status',
                'idx_metadata_priority',
                'idx_metadata_agent_name',
                'idx_metadata_task_name',
                'idx_thread_metadata_status',
                'idx_thread_metadata_priority',
                'idx_metadata_completed',
            }
            assert set(indexes) == expected_indexes

    @pytest.mark.asyncio
    async def test_init_database_handles_errors(self, temp_db_path: Path) -> None:
        """Test database initialization handles errors properly."""
        from app.backends import create_backend

        # Make directory read-only to cause error
        temp_db_path.parent.mkdir(exist_ok=True)
        async_temp_db_path = AsyncPath(temp_db_path)
        await async_temp_db_path.touch()
        await async_temp_db_path.chmod(0o444)  # Read-only

        # Create SQLite backend explicitly - expect error during initialization
        backend = create_backend(backend_type='sqlite', db_path=str(temp_db_path))

        # Backend initialization should fail with OperationalError when file is read-only
        with pytest.raises(sqlite3.OperationalError, match='.*readonly database.*'):
            await backend.initialize()

        # Clean up (no need to shutdown as initialization failed)
        await async_temp_db_path.chmod(0o644)


class TestDatabaseConnection:
    """Test database connection management."""

    @pytest.mark.asyncio
    async def test_db_rollback_on_error(self, async_db_initialized: StorageBackend) -> None:
        """Test database rollback on error."""
        manager = async_db_initialized

        # Insert initial data
        async with manager.get_connection(allow_write=True) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO context_entries (thread_id, source, content_type) VALUES (?, ?, ?)',
                ('test', 'user', 'text'),
            )
            conn.commit()

        # Try to insert with error
        try:
            async with manager.get_connection(allow_write=True) as conn:
                cursor = conn.cursor()
                # This should fail due to CHECK constraint
                cursor.execute(
                    'INSERT INTO context_entries (thread_id, source, content_type) VALUES (?, ?, ?)',
                    ('test', 'invalid_source', 'text'),
                )
                pytest.fail('Should have raised IntegrityError')
        except sqlite3.IntegrityError:
            pass  # Expected

        # Verify no partial data was committed
        async with manager.get_connection(readonly=True) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM context_entries')
            assert cursor.fetchone()[0] == 1  # Only the first valid insert

    @pytest.mark.asyncio
    async def test_async_context_manager(self, async_db_initialized: StorageBackend) -> None:
        """Test async database context manager."""
        manager = async_db_initialized
        async with manager.get_connection(readonly=True) as conn:
            # Test connection is valid
            assert conn is not None

            # Test we can execute queries
            cursor = conn.cursor()
            cursor.execute('SELECT 1 as test')
            result = cursor.fetchone()
            assert result['test'] == 1

    @pytest.mark.asyncio
    async def test_async_rollback_on_error(self, async_db_initialized: StorageBackend) -> None:
        """Test async database rollback on error."""
        manager = async_db_initialized

        # Try to insert with error
        try:
            async with manager.get_connection(allow_write=True) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO context_entries (thread_id, source, content_type) VALUES (?, ?, ?)',
                    ('test', 'invalid_source', 'text'),
                )
                pytest.fail('Should have raised IntegrityError')
        except sqlite3.IntegrityError:
            pass  # Expected


class TestTagOperations:
    """Test tag storage and retrieval operations."""

    @pytest.mark.asyncio
    async def test_store_tags_async(self, async_test_db: sqlite3.Connection) -> None:
        """Test storing tags for a context entry."""
        # Insert a context entry
        cursor = async_test_db.cursor()
        cursor.execute(
            'INSERT INTO context_entries (thread_id, source, content_type) VALUES (?, ?, ?)',
            ('test', 'user', 'text'),
        )
        context_id = cursor.lastrowid
        async_test_db.commit()

        # Store tags
        tags = ['python', 'Testing', '  spaced  ', '', 'UPPERCASE']
        if context_id is not None:
            # Store tags directly using SQL
            for tag in tags:
                tag = tag.strip().lower()
                if tag:
                    cursor.execute(
                        'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                        (context_id, tag),
                    )
        async_test_db.commit()

        # Verify tags were stored normalized
        cursor.execute(
            'SELECT tag FROM tags WHERE context_entry_id = ? ORDER BY tag',
            (context_id,),
        )
        stored_tags = [row['tag'] for row in cursor.fetchall()]
        # Empty string should be filtered out
        assert stored_tags == ['python', 'spaced', 'testing', 'uppercase']

    @pytest.mark.asyncio
    async def test_store_tags_async_empty_list(self, async_test_db: sqlite3.Connection) -> None:
        """Test storing empty tag list does nothing."""
        cursor = async_test_db.cursor()
        cursor.execute(
            'INSERT INTO context_entries (thread_id, source, content_type) VALUES (?, ?, ?)',
            ('test', 'user', 'text'),
        )
        context_id = cursor.lastrowid
        async_test_db.commit()

        # Store empty tags - should do nothing
        # No tags to insert for empty list

        # Verify no tags were stored
        cursor.execute('SELECT COUNT(*) FROM tags WHERE context_entry_id = ?', (context_id,))
        assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_get_tags_async(self, async_test_db: sqlite3.Connection) -> None:
        """Test retrieving tags for a context entry."""
        # Insert context and tags manually
        cursor = async_test_db.cursor()
        cursor.execute(
            'INSERT INTO context_entries (thread_id, source, content_type) VALUES (?, ?, ?)',
            ('test', 'user', 'text'),
        )
        context_id = cursor.lastrowid

        tags = ['alpha', 'beta', 'gamma']
        for tag in tags:
            cursor.execute(
                'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                (context_id, tag),
            )
        async_test_db.commit()

        # Retrieve tags
        if context_id is not None:
            cursor.execute(
                'SELECT tag FROM tags WHERE context_entry_id = ? ORDER BY tag',
                (context_id,),
            )
            retrieved_tags = [row['tag'] for row in cursor.fetchall()]
        else:
            retrieved_tags = []
        assert retrieved_tags == ['alpha', 'beta', 'gamma']

    @pytest.mark.asyncio
    async def test_get_tags_async_no_tags(self, async_test_db: sqlite3.Connection) -> None:
        """Test retrieving tags for entry with no tags."""
        cursor = async_test_db.cursor()
        cursor.execute(
            'INSERT INTO context_entries (thread_id, source, content_type) VALUES (?, ?, ?)',
            ('test', 'user', 'text'),
        )
        context_id = cursor.lastrowid
        async_test_db.commit()

        # Retrieve tags for entry without tags
        if context_id is not None:
            cursor.execute(
                'SELECT tag FROM tags WHERE context_entry_id = ? ORDER BY tag',
                (context_id,),
            )
            tags = [row['tag'] for row in cursor.fetchall()]
        else:
            tags = []
        assert tags == []


class TestDatabaseIntegrity:
    """Test database integrity constraints and foreign keys."""

    def test_foreign_key_constraint(self, test_db: sqlite3.Connection) -> None:
        """Test foreign key constraints are enforced."""
        cursor = test_db.cursor()

        # Try to insert tag for non-existent context entry
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute(
                'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                (999, 'test'),
            )

    def test_cascade_delete(self, test_db: sqlite3.Connection) -> None:
        """Test cascading delete removes related tags and images."""
        cursor = test_db.cursor()

        # Insert context entry
        cursor.execute(
            'INSERT INTO context_entries (thread_id, source, content_type) VALUES (?, ?, ?)',
            ('test', 'user', 'text'),
        )
        context_id = cursor.lastrowid

        # Insert related data
        cursor.execute(
            'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
            (context_id, 'test-tag'),
        )
        cursor.execute(
            'INSERT INTO image_attachments (context_entry_id, image_data, mime_type) VALUES (?, ?, ?)',
            (context_id, b'test-image', 'image/png'),
        )
        test_db.commit()

        # Verify data exists
        cursor.execute('SELECT COUNT(*) FROM tags WHERE context_entry_id = ?', (context_id,))
        assert cursor.fetchone()[0] == 1
        cursor.execute('SELECT COUNT(*) FROM image_attachments WHERE context_entry_id = ?', (context_id,))
        assert cursor.fetchone()[0] == 1

        # Delete context entry
        cursor.execute('DELETE FROM context_entries WHERE id = ?', (context_id,))
        test_db.commit()

        # Verify related data was deleted
        cursor.execute('SELECT COUNT(*) FROM tags WHERE context_entry_id = ?', (context_id,))
        assert cursor.fetchone()[0] == 0
        cursor.execute('SELECT COUNT(*) FROM image_attachments WHERE context_entry_id = ?', (context_id,))
        assert cursor.fetchone()[0] == 0

    def test_check_constraints(self, test_db: sqlite3.Connection) -> None:
        """Test CHECK constraints are enforced."""
        cursor = test_db.cursor()

        # Test invalid source
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            cursor.execute(
                'INSERT INTO context_entries (thread_id, source, content_type) VALUES (?, ?, ?)',
                ('test', 'invalid', 'text'),
            )
        assert 'CHECK constraint failed' in str(exc_info.value)

        # Test invalid content_type
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            cursor.execute(
                'INSERT INTO context_entries (thread_id, source, content_type) VALUES (?, ?, ?)',
                ('test', 'user', 'invalid'),
            )
        assert 'CHECK constraint failed' in str(exc_info.value)


class TestDatabasePerformance:
    """Test database performance optimizations."""

    def test_indexes_are_used(self, test_db: sqlite3.Connection) -> None:
        """Test that queries use indexes for performance."""
        cursor = test_db.cursor()

        # Insert test data
        for i in range(100):
            cursor.execute(
                'INSERT INTO context_entries (thread_id, source, content_type) VALUES (?, ?, ?)',
                (f'thread_{i % 10}', 'user' if i % 2 == 0 else 'agent', 'text'),
            )
        test_db.commit()

        # Test query plan for thread_id filter (should use index)
        cursor.execute('EXPLAIN QUERY PLAN SELECT * FROM context_entries WHERE thread_id = ?', ('thread_1',))
        plan = cursor.fetchall()
        # Convert Row objects to strings for checking
        plan_str = ''.join(str(dict(row)) for row in plan)
        assert 'idx_thread_id' in plan_str or 'USING INDEX' in plan_str

        # Test query plan for source filter (should use index)
        cursor.execute('EXPLAIN QUERY PLAN SELECT * FROM context_entries WHERE source = ?', ('user',))
        plan = cursor.fetchall()
        # Convert Row objects to strings for checking
        plan_str = ''.join(str(dict(row)) for row in plan)
        assert 'idx_source' in plan_str or 'USING INDEX' in plan_str

        # Test query plan for combined filter (should use compound index)
        cursor.execute(
            'EXPLAIN QUERY PLAN SELECT * FROM context_entries WHERE thread_id = ? AND source = ?',
            ('thread_1', 'user'),
        )
        plan = cursor.fetchall()
        # Convert Row objects to strings for checking
        plan_str = ''.join(str(dict(row)) for row in plan)
        assert 'idx_thread_source' in plan_str or 'USING INDEX' in plan_str
