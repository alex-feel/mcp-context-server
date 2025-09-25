"""
Test suite specifically for detecting and verifying ResourceWarning fixes.

This module tests that all database connections are properly closed and
that no resources are leaked during normal operations or error conditions.
"""

import asyncio
import contextlib
import gc
import sqlite3
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from app.db_manager import DatabaseConnectionManager
from app.db_manager import get_connection_manager
from app.db_manager import reset_connection_manager


class TestResourceWarningDetection:
    """Test suite to detect and verify ResourceWarning fixes."""

    @pytest.fixture(autouse=True)
    def enable_resource_warnings(self):
        """Enable ResourceWarning detection for all tests in this class."""
        warnings.simplefilter('error', ResourceWarning)
        yield
        warnings.simplefilter('default', ResourceWarning)

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database path."""
        return tmp_path / 'test_resource.db'

    @pytest.mark.asyncio
    async def test_connection_manager_no_leaks_on_normal_shutdown(self, temp_db: Path) -> None:
        """Test that normal shutdown doesn't leak connections."""
        manager = DatabaseConnectionManager(temp_db)
        await manager.initialize()

        # Perform some operations
        async with manager.get_connection(readonly=True) as conn:
            conn.execute('SELECT 1')

        await manager.execute_write(lambda conn: conn.execute('CREATE TABLE test (id INTEGER)'))

        # Shutdown should close all connections
        await manager.shutdown()

        # Allow time for async cleanup to complete
        await asyncio.sleep(0.1)

        # Clear manager reference to allow proper garbage collection
        del manager

        # Force garbage collection to detect any unclosed connections
        gc.collect()

        # If there are leaks, ResourceWarning will be raised

    @pytest.mark.asyncio
    async def test_connection_manager_no_leaks_on_error(self, temp_db: Path) -> None:
        """Test that error conditions don't leak connections."""
        manager = DatabaseConnectionManager(temp_db)
        await manager.initialize()

        # Simulate error during read
        with contextlib.suppress(sqlite3.OperationalError):
            async with manager.get_connection(readonly=True) as conn:
                conn.execute('INVALID SQL')

        # Simulate error during write
        with contextlib.suppress(sqlite3.OperationalError):
            await manager.execute_write(lambda conn: conn.execute('INVALID SQL'))

        # Shutdown should still close all connections
        await manager.shutdown()

        # Allow time for async cleanup to complete
        await asyncio.sleep(0.1)

        # Clear manager reference to allow proper garbage collection
        del manager

        # Force garbage collection
        gc.collect()

    @pytest.mark.asyncio
    async def test_connection_pool_health_check_closes_unhealthy(self, temp_db: Path) -> None:
        """Test that health checks properly close unhealthy connections."""
        manager = DatabaseConnectionManager(temp_db)
        await manager.initialize()

        # Create multiple reader connections
        for _ in range(3):
            async with manager.get_connection(readonly=True) as conn:
                conn.execute('SELECT 1')

        # Perform health check
        await manager._perform_health_check()

        # Shutdown and verify no leaks
        await manager.shutdown()

        # Allow time for async cleanup to complete
        await asyncio.sleep(0.1)

        # Clear manager reference to allow proper garbage collection
        del manager

        gc.collect()

    @pytest.mark.asyncio
    async def test_write_queue_cancellation_no_leaks(self, temp_db: Path) -> None:
        """Test that cancelled write operations don't leak connections."""
        manager = DatabaseConnectionManager(temp_db)
        await manager.initialize()

        # Start multiple write operations
        write_tasks = []
        for i in range(5):
            task = asyncio.create_task(
                manager.execute_write(lambda conn, idx=i: conn.execute(f'CREATE TABLE test_{idx} (id INTEGER)')),
            )
            write_tasks.append(task)

        # Cancel some tasks
        for task in write_tasks[2:]:
            task.cancel()

        # Wait for non-cancelled tasks
        for task in write_tasks[:2]:
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Shutdown and verify no leaks
        await manager.shutdown()

        # Allow time for async cleanup to complete
        await asyncio.sleep(0.1)

        # Clear manager reference to allow proper garbage collection
        del manager

        gc.collect()

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_no_leaks(self, temp_db: Path) -> None:
        """Test that circuit breaker failures don't leak connections."""
        manager = DatabaseConnectionManager(temp_db)

        # Set circuit breaker to fail quickly
        manager.circuit_breaker.failure_threshold = 2

        await manager.initialize()

        # Force circuit breaker to open by causing failures
        for _ in range(3):
            with contextlib.suppress(Exception):
                await manager.execute_write(lambda conn: conn.execute('INVALID SQL'))

        # Circuit should be open now
        assert manager.circuit_breaker.is_open()

        # Try operations with open circuit
        with contextlib.suppress(RuntimeError):
            async with manager.get_connection(readonly=True):
                pass

        # Shutdown and verify no leaks
        await manager.shutdown()

        # Allow time for async cleanup to complete
        await asyncio.sleep(0.1)

        # Clear manager reference to allow proper garbage collection
        del manager

        gc.collect()

    @pytest.mark.asyncio
    async def test_concurrent_operations_no_leaks(self, temp_db: Path) -> None:
        """Test that high concurrency doesn't leak connections."""
        manager = DatabaseConnectionManager(temp_db)
        await manager.initialize()

        # Create schema
        await manager.execute_write(
            lambda conn: conn.execute('CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)'),
        )

        # Create operation functions with explicit manager parameter
        async def read_op(mgr: DatabaseConnectionManager, idx: int) -> None:
            async with mgr.get_connection(readonly=True) as conn:
                conn.execute('SELECT * FROM test WHERE id = ?', (idx,))

        async def write_op(mgr: DatabaseConnectionManager, idx: int) -> None:
            await mgr.execute_write(
                lambda conn: conn.execute('INSERT OR REPLACE INTO test (id, value) VALUES (?, ?)', (idx, f'value_{idx}')),
            )

        # Run many concurrent operations
        tasks = []
        for i in range(20):
            tasks.extend([
                asyncio.create_task(read_op(manager, i)),
                asyncio.create_task(write_op(manager, i)),
            ])

        await asyncio.gather(*tasks, return_exceptions=True)

        # Shutdown and verify no leaks
        await manager.shutdown()

        # Allow time for async cleanup to complete
        await asyncio.sleep(0.1)

        # Clear manager reference to allow proper garbage collection
        del manager

        gc.collect()

    @pytest.mark.asyncio
    async def test_singleton_manager_cleanup(self, temp_db: Path) -> None:
        """Test that singleton manager properly cleans up."""
        # Reset any existing instance
        reset_connection_manager()

        # Create and use singleton
        manager = get_connection_manager(temp_db, force_new=True)
        await manager.initialize()

        # Use the manager
        async with manager.get_connection(readonly=True) as conn:
            conn.execute('SELECT 1')

        # Proper cleanup
        await manager.shutdown()
        reset_connection_manager()

        # Allow time for async cleanup to complete
        await asyncio.sleep(0.1)

        # Clear manager reference to allow proper garbage collection
        del manager

        gc.collect()

    def test_temp_db_fixture_no_leaks(self, tmp_path: Path) -> None:
        """Test that the temp_db fixture doesn't leak connections."""

        # Create a database using the fixture logic
        db_path = tmp_path / 'test_fixture.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database with schema
        conn = sqlite3.connect(str(db_path))
        try:
            schema_path = Path(__file__).parent.parent / 'app' / 'schema.sql'
            if schema_path.exists():
                conn.executescript(schema_path.read_text())
            conn.commit()
        finally:
            # This is the fix - always close the connection
            conn.close()

        # Force garbage collection to detect leaks
        gc.collect()

    @pytest.mark.asyncio
    async def test_connection_tracking(self, temp_db: Path) -> None:
        """Test that all created connections are properly tracked and closed."""
        manager = DatabaseConnectionManager(temp_db)

        # Add tracking to verify all connections are closed
        created_connections: list[sqlite3.Connection] = []
        original_create = manager._create_connection

        def tracked_create_impl(
            connections: list[sqlite3.Connection],
            original_fn: Callable[[bool], sqlite3.Connection],
            readonly: bool = False,
        ) -> sqlite3.Connection:
            conn = original_fn(readonly)
            connections.append(conn)
            return conn

        tracked_create = partial(tracked_create_impl, created_connections, original_create)

        # Use patch.object to avoid mypy method-assign error
        with patch.object(manager, '_create_connection', tracked_create):
            await manager.initialize()

            # Perform various operations
            async with manager.get_connection(readonly=True) as conn:
                conn.execute('SELECT 1')

            await manager.execute_write(lambda conn: conn.execute('CREATE TABLE test (id INTEGER)'))

            # Shutdown
            await manager.shutdown()

        # Verify all connections are closed
        for conn in created_connections:
            try:
                conn.execute('SELECT 1')
                pytest.fail(f'Connection {conn} was not closed')
            except sqlite3.ProgrammingError:
                # Expected - connection is closed
                pass

        # Allow time for async cleanup to complete
        await asyncio.sleep(0.1)

        # Clear references to allow proper garbage collection
        del manager
        del created_connections

        gc.collect()

    @pytest.mark.asyncio
    async def test_background_task_cleanup(self, temp_db: Path) -> None:
        """Test that background tasks are properly cleaned up."""
        manager = DatabaseConnectionManager(temp_db)
        await manager.initialize()

        # Verify background tasks are running
        assert manager._write_processor_task is not None
        assert manager._health_check_task is not None
        assert len(manager._background_tasks) >= 2

        # Shutdown should cancel and wait for all tasks
        await manager.shutdown()

        # Verify all tasks are cleaned up
        assert manager._write_processor_task is None
        assert manager._health_check_task is None  # type: ignore[unreachable]
        assert len(manager._background_tasks) == 0

        # Verify shutdown is complete
        assert manager._shutdown_complete.is_set()

        # Allow time for async cleanup to complete
        await asyncio.sleep(0.1)

        # Clear manager reference to allow proper garbage collection
        del manager

        gc.collect()

    @pytest.mark.asyncio
    async def test_write_queue_drainage_on_shutdown(self, temp_db: Path) -> None:
        """Test that pending writes in queue are handled on shutdown."""
        manager = DatabaseConnectionManager(temp_db)
        await manager.initialize()

        # Create schema
        await manager.execute_write(
            lambda conn: conn.execute('CREATE TABLE test (id INTEGER)'),
        )

        # Add many writes to queue without waiting
        write_futures = []
        for i in range(10):
            future = asyncio.create_task(
                manager.execute_write(lambda conn, idx=i: conn.execute(f'INSERT INTO test VALUES ({idx})')),
            )
            write_futures.append(future)

        # Immediately shutdown
        await manager.shutdown()

        # Some writes may have completed, others cancelled
        # The important thing is no resource leaks
        for future in write_futures:
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await future

        # Allow time for async cleanup to complete
        await asyncio.sleep(0.1)

        # Clear references to allow proper garbage collection
        del manager
        del write_futures

        gc.collect()
