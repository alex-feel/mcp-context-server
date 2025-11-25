"""
Base protocol definition for storage backends.

This module defines the StorageBackend Protocol that all backend implementations
must follow. The protocol ensures type-safe, database-agnostic interfaces for
repositories and server components.

The protocol uses @runtime_checkable to enable isinstance() checks and TypeVar
for generic operation signatures that preserve return types.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

T = TypeVar('T')


@runtime_checkable
class StorageBackend(Protocol):
    """
    Protocol defining the interface for storage backend implementations.

    All storage backends (SQLite, PostgreSQL, etc.) must implement
    this protocol to ensure compatibility with the repository layer.

    The protocol defines methods for:
    - Lifecycle management (initialize, shutdown)
    - Connection management (get_connection)
    - Operation execution (execute_write, execute_read)
    - Health monitoring (get_metrics, backend_type)

    Example Implementation:
        class MySQLBackend:
            async def initialize(self) -> None:
                # Connect to MySQL, set up connection pool
                pass

            async def shutdown(self) -> None:
                # Close all connections gracefully
                pass

            @asynccontextmanager
            async def get_connection(self, readonly: bool = False) -> AsyncIterator[Any]:
                # Yield a MySQL connection from pool
                conn = await self.pool.acquire()
                try:
                    yield conn
                finally:
                    await self.pool.release(conn)

            async def execute_write(self, operation: Callable[..., T], *args, **kwargs) -> T:
                # Execute write operation with retry logic
                async with self.get_connection(readonly=False) as conn:
                    return operation(conn, *args, **kwargs)

            async def execute_read(self, operation: Callable[..., T], *args, **kwargs) -> T:
                # Execute read operation
                async with self.get_connection(readonly=True) as conn:
                    return operation(conn, *args, **kwargs)

            def get_metrics(self) -> dict[str, Any]:
                return {
                    'backend_type': 'mysql',
                    'pool_size': self.pool.get_size(),
                    'active_connections': self.pool.get_idle_size(),
                }

            @property
            def backend_type(self) -> str:
                return 'mysql'

    Example Usage:
        # Create backend
        backend = SQLiteBackend(db_path='/path/to/db.sqlite')
        await backend.initialize()

        # Use with repository
        repo = ContextRepository(backend)

        # Execute operations
        async with backend.get_connection(readonly=True) as conn:
            results = conn.execute('SELECT * FROM context_entries').fetchall()

        # Get health metrics
        metrics = backend.get_metrics()
        print(f"Backend: {backend.backend_type}, Metrics: {metrics}")

        # Cleanup
        await backend.shutdown()
    """

    async def initialize(self) -> None:
        """
        Initialize the storage backend.

        This method is called once during server startup to establish connections,
        create connection pools, and perform any necessary setup.

        For SQLite:
            - Creates database file if not exists
            - Initializes connection pool (readers + writer)
            - Starts background health check task
            - Loads sqlite-vec extension

        For PostgreSQL:
            - Creates asyncpg connection pool
            - Verifies schema exists
            - Configures statement cache

        Raises:
            RuntimeError: If initialization fails
            ConnectionError: If database is unreachable

        Example:
            backend = SQLiteBackend(db_path='/data/context.db')
            await backend.initialize()
            # Backend is now ready for use
        """
        ...

    async def shutdown(self) -> None:
        """
        Gracefully shut down the storage backend.

        This method is called during server shutdown to close all connections,
        cancel background tasks, and release resources.

        For SQLite:
            - Cancels health check task
            - Closes all reader connections
            - Closes writer connection
            - Ensures no connections are leaked

        For PostgreSQL:
            - Closes asyncpg pool
            - Waits for in-flight queries to complete
            - Releases all connections

        Raises:
            RuntimeError: If shutdown is called before initialization

        Example:
            await backend.shutdown()
            # All connections closed, resources released
        """
        ...

    def get_connection(
        self,
        readonly: bool = False,
        allow_write: bool = False,
    ) -> AbstractAsyncContextManager[Any]:
        """
        Get a database connection from the pool.

        This is an async context manager that yields a connection and ensures
        proper cleanup on exit.

        Args:
            readonly: If True, returns a read-only connection (for SQLite, a reader)
            allow_write: If True, allows write operations on readonly connection
                        (for SQLite, this is used by _execute_write_internal)

        Yields:
            Connection object (sqlite3.Connection for SQLite, asyncpg.Connection for PostgreSQL)

        For SQLite:
            - readonly=True: Returns connection from reader pool
            - readonly=False: Returns the single writer connection
            - Connections are automatically returned to pool on exit

        For PostgreSQL:
            - Both readonly and write connections come from the same pool
            - readonly parameter is advisory (PostgreSQL handles this via transactions)

        Raises:
            RuntimeError: If backend is shut down or not initialized

        Example:
            async with backend.get_connection(readonly=True) as conn:
                cursor = conn.execute('SELECT * FROM context_entries')
                results = cursor.fetchall()
            # Connection automatically returned to pool
        """
        ...

    async def execute_write(
        self,
        operation: Callable[..., T] | Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a write operation with retry logic and proper connection handling.

        This method wraps write operations with:
        - Retry logic with exponential backoff
        - Circuit breaker protection
        - Connection management
        - Error handling

        Args:
            operation: Sync or async callable that performs the write operation.
                      SQLite backend expects sync callable: Callable[..., T]
                      PostgreSQL backend expects async callable: Callable[..., Awaitable[T]]
            *args: Positional arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of the operation (type preserved via TypeVar)

        For SQLite:
            - Queues write operation in write queue (serialized execution)
            - Uses single writer connection
            - Retries on transient errors (SQLITE_BUSY, SQLITE_LOCKED)
            - Circuit breaker prevents cascading failures
            - Expects sync callable: operation(conn, *args, **kwargs) -> T

        For PostgreSQL:
            - Uses connection from pool
            - Wraps in transaction
            - Retries on connection errors
            - Expects async callable: await operation(conn, *args, **kwargs) -> T

        Raises:
            RuntimeError: If backend is shut down or circuit breaker is open
            Exception: Original exception from operation after retries exhausted

        Example (SQLite - sync):
            def insert_context(conn, text, thread_id):
                cursor = conn.execute(
                    'INSERT INTO context_entries (text_content, thread_id) VALUES (?, ?)',
                    (text, thread_id)
                )
                return cursor.lastrowid

            context_id = await backend.execute_write(insert_context, 'Hello', 'thread-123')

        Example (PostgreSQL - async):
            async def insert_context(conn, text, thread_id):
                row = await conn.fetchrow(
                    'INSERT INTO context_entries (text_content, thread_id) VALUES ($1, $2) RETURNING id',
                    text, thread_id
                )
                return row['id']

            context_id = await backend.execute_write(insert_context, 'Hello', 'thread-123')
        """
        ...

    async def execute_read(
        self,
        operation: Callable[..., T] | Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a read operation with proper connection handling.

        This method wraps read operations with:
        - Read-only connection from pool
        - Error handling
        - Connection cleanup

        Args:
            operation: Sync or async callable that performs the read operation.
                      SQLite backend expects sync callable: Callable[..., T]
                      PostgreSQL backend expects async callable: Callable[..., Awaitable[T]]
            *args: Positional arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of the operation (type preserved via TypeVar)

        For SQLite:
            - Uses connection from reader pool
            - No retry logic needed (reads don't cause locks)
            - Load balancing across multiple readers
            - Expects sync callable: operation(conn, *args, **kwargs) -> T

        For PostgreSQL:
            - Uses connection from pool
            - Can use read replicas if configured
            - Expects async callable: await operation(conn, *args, **kwargs) -> T

        Example (SQLite - sync):
            def get_context_by_id(conn, context_id):
                cursor = conn.execute(
                    'SELECT * FROM context_entries WHERE id = ?',
                    (context_id,)
                )
                return cursor.fetchone()

            entry = await backend.execute_read(get_context_by_id, 123)

        Example (PostgreSQL - async):
            async def get_context_by_id(conn, context_id):
                row = await conn.fetchrow(
                    'SELECT * FROM context_entries WHERE id = $1',
                    context_id
                )
                return row

            entry = await backend.execute_read(get_context_by_id, 123)
        """
        ...

    def get_metrics(self) -> dict[str, Any]:
        """
        Get backend health metrics and statistics.

        Returns a dictionary containing backend-specific health metrics,
        connection pool statistics, and performance indicators.

        Returns:
            Dictionary with metrics. Common keys:
                - backend_type: str - Backend identifier (sqlite, postgresql)
                - pool_size: int - Total connections in pool
                - active_connections: int - Connections currently in use
                - circuit_breaker_state: str - Circuit breaker status
                - total_operations: int - Lifetime operation count
                - failed_operations: int - Failed operation count
                - avg_operation_time_ms: float - Average operation latency

        For SQLite:
            {
                'backend_type': 'sqlite',
                'pool_size': 8,
                'active_readers': 2,
                'writer_busy': False,
                'write_queue_size': 0,
                'circuit_breaker_state': 'HEALTHY',
                'total_writes': 1234,
                'total_reads': 5678,
                'failed_writes': 0,
                'failed_reads': 0,
            }

        For PostgreSQL:
            {
                'backend_type': 'postgresql',
                'pool_size': 20,
                'pool_idle': 15,
                'pool_free': 10,
                'total_queries': 9999,
                'failed_queries': 5,
            }

        Example:
            metrics = backend.get_metrics()
            print(f"Backend: {metrics['backend_type']}")
            print(f"Active connections: {metrics.get('active_connections', 0)}")
        """
        ...

    @property
    def backend_type(self) -> str:
        """
        Get the backend type identifier.

        Returns:
            Backend type string: 'sqlite' or 'postgresql'

        This property is used for:
        - Logging and monitoring
        - Conditional SQL generation in repositories
        - Backend-specific optimizations
        - Metrics tagging

        Example:
            if backend.backend_type == 'sqlite':
                # Use SQLite-specific query
                query = 'SELECT json_extract(metadata, "$.status") FROM ...'
            elif backend.backend_type == 'postgresql':
                # Use PostgreSQL-specific query
                query = "SELECT metadata->>'status' FROM ..."
        """
        ...
