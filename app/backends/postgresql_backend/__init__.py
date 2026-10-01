"""PostgreSQL storage backend implementation.

This package provides a production-grade PostgreSQL backend implementing the StorageBackend
protocol with asyncpg connection pooling, circuit breaker pattern, retry logic, and health monitoring.
``PostgreSQLBackend`` is assembled here from the provisioning, lifecycle, operations and transaction
mixins over ``PostgreSQLBackendCore``; every other name is imported from its defining submodule.
"""

from app.backends.postgresql_backend.lifecycle import PostgreSQLLifecycleMixin
from app.backends.postgresql_backend.operations import PostgreSQLOperationsMixin
from app.backends.postgresql_backend.transactions import PostgreSQLTransactionMixin


class PostgreSQLBackend(PostgreSQLLifecycleMixin, PostgreSQLOperationsMixin, PostgreSQLTransactionMixin):
    """Production-grade PostgreSQL storage backend implementing the StorageBackend protocol.

    Features:
    - asyncpg connection pooling with configurable min/max connections
    - Circuit breaker pattern for fault tolerance
    - Exponential backoff with jitter for transient errors
    - Explicit transaction management
    - Health checks and metrics
    - Automatic schema initialization

    Implements the StorageBackend protocol to enable database-agnostic repositories.
    """
