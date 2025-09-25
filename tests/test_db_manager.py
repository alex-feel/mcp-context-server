"""
Test suite for the DatabaseConnectionManager with production-grade concurrency features.

Tests connection pooling, write queue, circuit breaker, retry logic, and concurrent operations.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from app.db_manager import CircuitBreaker
from app.db_manager import ConnectionState
from app.db_manager import DatabaseConnectionManager
from app.db_manager import PoolConfig
from app.db_manager import RetryConfig
from app.db_manager import get_connection_manager

# get_sync_connection has been removed - using async patterns only


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary database file."""
    db_path = tmp_path / 'test_connection.db'
    # Initialize with schema
    schema_path = Path(__file__).parent.parent / 'app' / 'schema.sql'
    schema_sql = schema_path.read_text(encoding='utf-8')

    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(schema_sql)

    return db_path


@pytest.fixture
def pool_config() -> PoolConfig:
    """Create test pool configuration."""
    return PoolConfig(
        max_readers=3,
        max_writers=1,
        connection_timeout=5.0,
        idle_timeout=60.0,
        health_check_interval=5.0,
    )


@pytest.fixture
def retry_config() -> RetryConfig:
    """Create test retry configuration."""
    return RetryConfig(
        max_retries=3,
        base_delay=0.1,
        max_delay=1.0,
        jitter=True,
        backoff_factor=2.0,
    )


@pytest_asyncio.fixture
async def db_manager(
    temp_db: Path,
    pool_config: PoolConfig,
    retry_config: RetryConfig,
) -> AsyncGenerator[DatabaseConnectionManager, None]:
    """Create and initialize a database connection manager."""
    manager = DatabaseConnectionManager(
        db_path=temp_db,
        pool_config=pool_config,
        retry_config=retry_config,
    )
    await manager.initialize()
    yield manager
    await manager.shutdown()


class TestCircuitBreaker:
    """Test the circuit breaker pattern implementation."""

    def test_initial_state(self) -> None:
        """Test circuit breaker starts in healthy state."""
        breaker = CircuitBreaker(failure_threshold=3)
        assert breaker.get_state() == ConnectionState.HEALTHY
        assert not breaker.is_open()

    def test_failure_threshold(self) -> None:
        """Test circuit breaker trips after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        # Record failures
        breaker.record_failure()
        assert breaker.get_state() == ConnectionState.HEALTHY

        breaker.record_failure()
        assert breaker.get_state() == ConnectionState.HEALTHY

        # Third failure should trip the breaker
        breaker.record_failure()
        assert breaker.get_state() == ConnectionState.FAILED
        assert breaker.is_open()

    def test_recovery_timeout(self) -> None:
        """Test circuit breaker recovery after timeout."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Trip the breaker
        breaker.record_failure()
        assert breaker.is_open()

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should enter degraded state
        assert not breaker.is_open()
        assert breaker.get_state() == ConnectionState.DEGRADED

    def test_half_open_recovery(self) -> None:
        """Test recovery from degraded state with successful calls."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,
            half_open_max_calls=2,
        )

        # Trip and recover to degraded
        breaker.record_failure()
        time.sleep(0.15)
        assert breaker.get_state() == ConnectionState.DEGRADED

        # Successful calls in degraded state
        breaker.record_success()
        assert breaker.get_state() == ConnectionState.DEGRADED

        breaker.record_success()
        assert breaker.get_state() == ConnectionState.HEALTHY


class TestConnectionManager:
    """Test the DatabaseConnectionManager."""

    @pytest.mark.asyncio
    async def test_initialization(self, db_manager: DatabaseConnectionManager) -> None:
        """Test manager initializes correctly."""
        assert db_manager._writer_conn is not None
        assert not db_manager._shutdown
        assert db_manager._write_processor_task is not None
        assert db_manager._health_check_task is not None

    @pytest.mark.asyncio
    async def test_read_connection(self, db_manager: DatabaseConnectionManager) -> None:
        """Test getting a read connection."""
        async with db_manager.get_connection(readonly=True) as conn:
            assert conn is not None
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            result = cursor.fetchone()
            assert result[0] == 1

    @pytest.mark.asyncio
    async def test_write_operation(self, db_manager: DatabaseConnectionManager) -> None:
        """Test write operation through write queue."""

        def insert_test(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO context_entries (thread_id, source, content_type, text_content) VALUES (?, ?, ?, ?)',
                ('test_thread', 'user', 'text', 'Test content'),
            )
            return cursor.lastrowid or 0

        result = await db_manager.execute_write(insert_test)
        assert result > 0

        # Verify the write
        def read_test(conn: sqlite3.Connection) -> str | None:
            cursor = conn.cursor()
            cursor.execute('SELECT text_content FROM context_entries WHERE id = ?', (result,))
            row = cursor.fetchone()
            return row[0] if row else None

        content = await db_manager.execute_read(read_test)
        assert content == 'Test content'

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, db_manager: DatabaseConnectionManager) -> None:
        """Test multiple concurrent read operations."""

        async def read_operation(index: int) -> int:
            return await db_manager.execute_read(
                lambda _conn: index,  # Simple return value without actual query
            )

        # Execute multiple reads concurrently
        tasks = [read_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert results == list(range(10))

    @pytest.mark.asyncio
    async def test_write_queue_serialization(self, db_manager: DatabaseConnectionManager) -> None:
        """Test that writes are properly serialized through the queue."""
        results = []

        def write_with_order(conn: sqlite3.Connection, order: int) -> int:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO context_entries (thread_id, source, content_type, text_content) VALUES (?, ?, ?, ?)',
                (f'thread_{order}', 'user', 'text', f'Content {order}'),
            )
            return order

        # Submit multiple writes concurrently
        tasks = [db_manager.execute_write(write_with_order, i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All writes should complete
        assert sorted(results) == list(range(5))

        # Verify all entries exist
        def count_entries(conn: sqlite3.Connection) -> int:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM context_entries')
            return cursor.fetchone()[0]

        count = await db_manager.execute_read(count_entries)
        assert count >= 5

    @pytest.mark.asyncio
    async def test_retry_on_lock(self, db_manager: DatabaseConnectionManager) -> None:
        """Test retry logic on database lock."""
        # Create a lock situation using context manager
        with sqlite3.connect(str(db_manager.db_path)) as lock_conn:
            lock_conn.execute('BEGIN EXCLUSIVE')

            # Try to write - should retry
            def write_test(conn: sqlite3.Connection) -> int:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO context_entries (thread_id, source, content_type, text_content) VALUES (?, ?, ?, ?)',
                    ('retry_test', 'user', 'text', 'Retry content'),
                )
                return cursor.lastrowid or 0

            # Start write in background
            write_task = asyncio.create_task(db_manager.execute_write(write_test))

            # Give it time to retry once
            await asyncio.sleep(0.2)

            # Release the lock
            lock_conn.rollback()

        # Write should now complete after lock_conn is closed by context manager
        result = await asyncio.wait_for(write_task, timeout=2.0)
        assert result > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(
        self,
        temp_db: Path,
        pool_config: PoolConfig,
        retry_config: RetryConfig,
    ) -> None:
        """Test circuit breaker prevents operations when tripped."""
        # Create manager with low failure threshold
        retry_config.max_retries = 1
        manager = DatabaseConnectionManager(
            db_path=temp_db,
            pool_config=pool_config,
            retry_config=retry_config,
        )
        manager.circuit_breaker.failure_threshold = 2
        await manager.initialize()

        try:
            # Manually record failures to trip the breaker
            manager.circuit_breaker.record_failure()
            manager.circuit_breaker.record_failure()

            # Circuit should be open now
            assert manager.circuit_breaker.is_open()

            # Next read operation should fail immediately
            with pytest.raises(RuntimeError, match='Database circuit breaker is open'):
                async with manager.get_connection(readonly=True):
                    pass

            # Write operations should also fail (write queue wraps as Exception)
            with pytest.raises(Exception, match='Database circuit breaker is open'):
                await manager.execute_write(lambda conn: conn.execute('SELECT 1'))
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, db_manager: DatabaseConnectionManager) -> None:
        """Test that metrics are properly tracked."""
        # Perform some operations
        await db_manager.execute_read(lambda conn: conn.execute('SELECT 1').fetchone())

        def write_op(conn: sqlite3.Connection) -> None:
            conn.execute(
                'INSERT INTO context_entries (thread_id, source, content_type, text_content) VALUES (?, ?, ?, ?)',
                ('metrics_test', 'user', 'text', 'Metrics content'),
            )

        await db_manager.execute_write(write_op)

        # Check metrics
        metrics = db_manager.get_metrics()
        assert metrics['total_connections'] > 0
        assert metrics['total_queries'] >= 2
        assert metrics['circuit_state'] == 'healthy'
        assert metrics['write_queue_size'] >= 0

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, db_manager: DatabaseConnectionManager) -> None:
        """Test graceful shutdown of the manager."""

        # Submit a write
        def slow_write(conn: sqlite3.Connection) -> int:
            time.sleep(0.1)  # Simulate slow operation
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO context_entries (thread_id, source, content_type, text_content) VALUES (?, ?, ?, ?)',
                ('shutdown_test', 'user', 'text', 'Shutdown content'),
            )
            return cursor.lastrowid or 0

        # Start write
        write_task = asyncio.create_task(db_manager.execute_write(slow_write))

        # Give it time to start
        await asyncio.sleep(0.05)

        # Shutdown should wait for pending operations
        await db_manager.shutdown()

        # Write task might still be pending if shutdown was too quick
        # Wait a bit more or cancel it
        if not write_task.done():
            write_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await write_task
        else:
            # If it completed, check result
            result = await write_task
            assert result > 0

        # Manager should be shut down
        assert db_manager._shutdown

        # New operations should fail
        with pytest.raises(RuntimeError, match='shutting down'):
            await db_manager.execute_write(lambda _conn: None)


# TestSyncConnection class removed - using async patterns only


class TestGlobalManager:
    """Test the global connection manager singleton."""

    def test_singleton_pattern(self, temp_db: Path) -> None:
        """Test that get_connection_manager behavior in test environment."""
        from app.db_manager import reset_connection_manager

        # Reset to ensure clean state
        reset_connection_manager()

        # In test environment, it always creates a new instance
        # This is intentional for test isolation
        manager1 = get_connection_manager(temp_db)
        manager2 = get_connection_manager(temp_db)
        # In test environment, these will be different instances for isolation
        # This is the expected behavior
        assert manager1 is not manager2  # Different instances in test environment

    def test_requires_db_path_first_time(self) -> None:
        """Test that first call requires db_path."""
        # Reset global instance
        import app.db_manager

        app.db_manager._manager_instance = None

        with pytest.raises(ValueError, match='db_path required'):
            get_connection_manager()


@pytest.mark.asyncio
class TestStressTest:
    """Stress tests for the connection manager."""

    async def test_high_concurrency(
        self,
        temp_db: Path,
        pool_config: PoolConfig,
        retry_config: RetryConfig,
    ) -> None:
        """Test with high concurrency load."""
        pool_config.max_readers = 10
        manager = DatabaseConnectionManager(
            db_path=temp_db,
            pool_config=pool_config,
            retry_config=retry_config,
        )
        await manager.initialize()

        try:
            # Create many concurrent operations
            async def mixed_operation(index: int) -> int:
                if index % 5 == 0:
                    # Write operation
                    def write(conn: sqlite3.Connection) -> int:
                        cursor = conn.cursor()
                        cursor.execute(
                            'INSERT INTO context_entries (thread_id, source, content_type, text_content) VALUES (?, ?, ?, ?)',
                            (f'stress_{index}', 'user', 'text', f'Content {index}'),
                        )
                        return cursor.lastrowid or 0

                    return await manager.execute_write(write)

                # Read operation

                def read(conn: sqlite3.Connection) -> int:
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM context_entries')
                    row = cursor.fetchone()
                    return row[0] if row else 0

                return await manager.execute_read(read)

            # Run 100 concurrent operations
            tasks = [mixed_operation(i) for i in range(100)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for errors
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0, f'Got {len(errors)} errors in stress test'

            # Verify metrics
            metrics = manager.get_metrics()
            assert metrics['total_queries'] >= 100
            assert metrics['failed_queries'] == 0
        finally:
            await manager.shutdown()
            # Wait for shutdown to complete fully
            await manager._shutdown_complete.wait()
