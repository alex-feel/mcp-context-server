"""Tests for transaction context functionality.

This module tests the begin_transaction() method on storage backends,
which provides atomic multi-operation transaction support.

Phase 1 of the Transactional Integrity Fix:
- Backend Transaction Infrastructure
- Ensures transactions commit on success and rollback on exception
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.backends.base import TransactionContext
from app.backends.sqlite_backend import SQLiteBackend
from app.backends.sqlite_backend import SQLiteTransactionContext


class TestSQLiteTransactionContext:
    """Tests for SQLite transaction context."""

    @pytest.mark.asyncio
    async def test_begin_transaction_commit_on_success(self, tmp_path: Path) -> None:
        """Test that transaction commits when context exits normally."""
        db_path = tmp_path / 'test_commit.db'

        # Create base schema first
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()

        try:
            async with backend.begin_transaction() as txn:
                assert txn.backend_type == 'sqlite'
                conn = txn.connection
                assert isinstance(conn, sqlite3.Connection)

                # Insert test data
                conn.execute(
                    'INSERT INTO context_entries (thread_id, source, text_content, content_type) VALUES (?, ?, ?, ?)',
                    ('test-thread', 'agent', 'test content', 'text'),
                )

            # Verify data was committed
            async with backend.get_connection(readonly=True) as conn:
                cursor = conn.execute(
                    'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                    ('test-thread',),
                )
                count = cursor.fetchone()[0]
                assert count == 1
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_begin_transaction_rollback_on_exception(self, tmp_path: Path) -> None:
        """Test that transaction rolls back when exception occurs."""
        db_path = tmp_path / 'test_rollback.db'

        # Create base schema first
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()

        async def insert_and_fail(backend: SQLiteBackend) -> None:
            """Insert data and then raise an intentional error."""
            async with backend.begin_transaction() as txn:
                txn.connection.execute(
                    'INSERT INTO context_entries (thread_id, source, text_content, content_type) VALUES (?, ?, ?, ?)',
                    ('rollback-test', 'agent', 'should be rolled back', 'text'),
                )
                raise ValueError('Intentional failure')

        try:
            with pytest.raises(ValueError, match='Intentional failure'):
                await insert_and_fail(backend)

            # Verify data was rolled back
            async with backend.get_connection(readonly=True) as conn:
                cursor = conn.execute(
                    'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                    ('rollback-test',),
                )
                count = cursor.fetchone()[0]
                assert count == 0
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_begin_transaction_multiple_operations(self, tmp_path: Path) -> None:
        """Test multiple operations in single transaction."""
        db_path = tmp_path / 'test_multi_op.db'

        # Create base schema first
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()

        try:
            async with backend.begin_transaction() as txn:
                conn = txn.connection

                # Insert context entry
                cursor = conn.execute(
                    'INSERT INTO context_entries (thread_id, source, text_content, content_type) VALUES (?, ?, ?, ?)',
                    ('multi-op-test', 'agent', 'test content', 'text'),
                )
                context_id = cursor.lastrowid

                # Insert tag
                conn.execute(
                    'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                    (context_id, 'test-tag'),
                )

            # Verify both operations committed
            async with backend.get_connection(readonly=True) as conn:
                # Check context entry
                cursor = conn.execute(
                    'SELECT id FROM context_entries WHERE thread_id = ?',
                    ('multi-op-test',),
                )
                row = cursor.fetchone()
                assert row is not None
                context_id = row[0]

                # Check tag
                cursor = conn.execute(
                    'SELECT tag FROM tags WHERE context_entry_id = ?',
                    (context_id,),
                )
                tag_row = cursor.fetchone()
                assert tag_row is not None
                assert tag_row[0] == 'test-tag'
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_begin_transaction_partial_rollback(self, tmp_path: Path) -> None:
        """Test that partial operations are rolled back on failure."""
        db_path = tmp_path / 'test_partial.db'

        # Create base schema first
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()

        async def insert_with_fk_violation(backend: SQLiteBackend) -> None:
            """Insert data with a foreign key violation to trigger rollback."""
            async with backend.begin_transaction() as txn:
                conn = txn.connection

                # Enable foreign key constraints
                conn.execute('PRAGMA foreign_keys = ON')

                # First operation succeeds
                conn.execute(
                    'INSERT INTO context_entries (thread_id, source, text_content, content_type) VALUES (?, ?, ?, ?)',
                    ('partial-test', 'agent', 'first entry', 'text'),
                )

                # Second operation fails (foreign key violation - non-existent context_id)
                conn.execute(
                    'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                    (999999, 'invalid-tag'),
                )

        try:
            with pytest.raises(sqlite3.IntegrityError):
                await insert_with_fk_violation(backend)

            # Verify first operation was also rolled back
            async with backend.get_connection(readonly=True) as conn:
                cursor = conn.execute(
                    'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                    ('partial-test',),
                )
                count = cursor.fetchone()[0]
                assert count == 0
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_begin_transaction_shutdown_check(self, tmp_path: Path) -> None:
        """Test that begin_transaction raises error when backend is shut down."""
        db_path = tmp_path / 'test_shutdown.db'

        # Create base schema first
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()

        # Shutdown the backend
        await backend.shutdown()

        # Attempting to begin transaction should raise
        with pytest.raises(RuntimeError, match='shutting down'):
            async with backend.begin_transaction():
                pass


class TestTransactionContextProtocol:
    """Tests for TransactionContext protocol compliance."""

    def test_sqlite_context_implements_protocol(self) -> None:
        """Test SQLiteTransactionContext implements TransactionContext protocol."""
        # Create a mock connection
        conn = sqlite3.connect(':memory:')
        ctx = SQLiteTransactionContext(_connection=conn)

        # Verify protocol compliance
        assert isinstance(ctx, TransactionContext)
        assert ctx.backend_type == 'sqlite'
        assert ctx.connection is conn

        conn.close()

    def test_transaction_context_properties(self) -> None:
        """Test that TransactionContext provides required properties."""
        conn = sqlite3.connect(':memory:')
        ctx = SQLiteTransactionContext(_connection=conn)

        # Test connection property returns the connection
        assert ctx.connection is conn

        # Test backend_type property returns correct type
        assert ctx.backend_type == 'sqlite'

        conn.close()


class TestTransactionContextIntegration:
    """Integration tests for transaction context with repositories."""

    @pytest.mark.asyncio
    async def test_transaction_preserves_connection_across_operations(
        self, tmp_path: Path,
    ) -> None:
        """Test that the same connection is used throughout the transaction."""
        db_path = tmp_path / 'test_same_conn.db'

        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()

        try:
            connection_ids: list[int] = []

            async with backend.begin_transaction() as txn:
                # Record the connection id
                connection_ids.append(id(txn.connection))

                # Do some operations
                txn.connection.execute(
                    'INSERT INTO context_entries (thread_id, source, text_content, content_type) VALUES (?, ?, ?, ?)',
                    ('conn-test-1', 'agent', 'content 1', 'text'),
                )

                # Record connection id again
                connection_ids.append(id(txn.connection))

                # Another operation
                txn.connection.execute(
                    'INSERT INTO context_entries (thread_id, source, text_content, content_type) VALUES (?, ?, ?, ?)',
                    ('conn-test-2', 'agent', 'content 2', 'text'),
                )

                # Record connection id once more
                connection_ids.append(id(txn.connection))

            # All connection ids should be the same
            assert len(set(connection_ids)) == 1
        finally:
            await backend.shutdown()
