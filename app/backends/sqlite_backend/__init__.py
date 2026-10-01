"""SQLite storage backend implementation.

This package provides a production-grade SQLite backend implementing the StorageBackend
protocol with connection pooling, write queue management, circuit breaker pattern,
and health monitoring. ``SQLiteBackend`` is assembled here from the lifecycle, write-queue,
health, connection-scope and transaction mixins over ``SQLiteConnectionsMixin`` and
``SQLiteBackendCore``; every other name is imported from its defining submodule.
"""

from app.backends.sqlite_backend.connection_scopes import SQLiteConnectionScopeMixin
from app.backends.sqlite_backend.health import SQLiteHealthMixin
from app.backends.sqlite_backend.lifecycle import SQLiteLifecycleMixin
from app.backends.sqlite_backend.transactions import SQLiteTransactionMixin
from app.backends.sqlite_backend.write_queue import SQLiteWriteQueueMixin


class SQLiteBackend(
    SQLiteLifecycleMixin,
    SQLiteWriteQueueMixin,
    SQLiteHealthMixin,
    SQLiteConnectionScopeMixin,
    SQLiteTransactionMixin,
):
    """
    Production-grade SQLite storage backend implementing the StorageBackend protocol.

    Features:
    - Per-use reader connections bounded by a semaphore, and one shared writer connection
    - Write queue for serializing write operations
    - Circuit breaker pattern for fault tolerance
    - Exponential backoff with jitter
    - Health checks and metrics
    - Automatic reconnection
    - Enhanced task lifecycle management for clean shutdown

    Implements the StorageBackend protocol to enable database-agnostic repositories.
    """
