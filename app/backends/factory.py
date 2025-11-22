"""
Backend factory for creating storage backend instances.

This module provides a factory function for creating appropriate storage backend
instances based on configuration. The factory enables runtime backend selection
through environment variables or explicit parameters.
"""

from pathlib import Path
from typing import Literal

from app.backends.base import StorageBackend
from app.backends.sqlite_backend import SQLiteBackend
from app.settings import get_settings


def create_backend(
    backend_type: Literal['sqlite', 'postgresql', 'supabase'] | None = None,
    db_path: Path | str | None = None,
) -> StorageBackend:
    """
    Create a storage backend instance based on type.

    This factory function creates the appropriate backend implementation based on
    the backend_type parameter or settings. It enables database-agnostic code by
    returning a StorageBackend protocol implementation.

    Args:
        backend_type: Type of backend to create ('sqlite', 'postgresql', 'supabase').
                     If None, reads from settings.storage.backend_type
        db_path: Path to database file (SQLite only). If None, uses settings.storage.db_path

    Returns:
        StorageBackend implementation (SQLiteBackend, PostgreSQLBackend, etc.)

    Raises:
        ValueError: If backend_type is invalid or required parameters are missing
        NotImplementedError: If backend_type is 'postgresql' or 'supabase' (not yet implemented)

    Example:
        # Create SQLite backend from settings
        backend = create_backend()
        await backend.initialize()

        # Create SQLite backend with explicit path
        backend = create_backend(backend_type='sqlite', db_path='/data/context.db')
        await backend.initialize()

        # Create PostgreSQL backend (not yet implemented)
        backend = create_backend(backend_type='postgresql')
        await backend.initialize()
    """
    settings = get_settings()

    # Determine backend type from parameter or settings
    if backend_type is None:
        backend_type = getattr(settings.storage, 'backend_type', 'sqlite')

    # Validate backend type
    if backend_type not in ('sqlite', 'postgresql', 'supabase'):
        raise ValueError(
            f'Invalid backend_type: {backend_type}. '
            f'Must be one of: sqlite, postgresql, supabase',
        )

    # Create appropriate backend
    if backend_type == 'sqlite':
        # Determine database path
        if db_path is None:
            db_path = settings.storage.db_path
            if db_path is None:
                raise ValueError(
                    'db_path is required for SQLite backend. '
                    'Provide via parameter or set DB_PATH environment variable.',
                )

        # Create SQLite backend
        return SQLiteBackend(db_path=db_path)

    if backend_type == 'postgresql':
        # PostgreSQL backend - implementation not yet available
        raise NotImplementedError(
            'PostgreSQL backend is not yet implemented. '
            'Please use SQLite backend (backend_type="sqlite") for now.',
        )

    if backend_type == 'supabase':
        # Supabase backend - implementation not yet available
        raise NotImplementedError(
            'Supabase backend is not yet implemented. '
            'Please use SQLite backend (backend_type="sqlite") for now.',
        )

    # This should never be reached due to validation above
    raise ValueError(f'Unsupported backend_type: {backend_type}')
