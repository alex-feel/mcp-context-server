"""
Storage backend implementations for mcp-context-server.

This package provides a protocol-based abstraction for different database backends,
enabling support for SQLite, PostgreSQL, and other storage systems.

Key Components:
    - StorageBackend: Protocol defining the interface for all storage backends
    - SQLiteBackend: Implementation for SQLite database
    - create_backend: Factory function for the request backend that serves the tools
    - create_maintenance_backend: Factory function for the backend migrations, startup
      and the command-line tools run on (the table owner on PostgreSQL)

Example Usage:
    from app.backends import create_backend

    # Create SQLite backend
    backend = create_backend(backend_type='sqlite', db_path='/path/to/db.sqlite')
    await backend.initialize()

    # Use with repositories
    repositories = RepositoryContainer(backend)
"""

from app.backends.base import StorageBackend
from app.backends.factory import create_backend
from app.backends.factory import create_maintenance_backend
from app.backends.sqlite_backend import SQLiteBackend

__all__ = ['StorageBackend', 'SQLiteBackend', 'create_backend', 'create_maintenance_backend']
