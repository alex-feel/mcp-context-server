"""Schema loading utilities for different storage backends.

This module provides utilities to load the appropriate SQL schema
based on the storage backend type (SQLite, PostgreSQL).
"""

from pathlib import Path


def get_schema_path(backend_type: str) -> Path:
    """Get the schema file path for a given backend type.

    Args:
        backend_type: The storage backend type ('sqlite', 'postgresql')

    Returns:
        Path to the schema SQL file

    Raises:
        ValueError: If backend_type is not supported
    """
    schemas_dir = Path(__file__).parent

    if backend_type == 'sqlite':
        return schemas_dir / 'sqlite_schema.sql'
    if backend_type == 'postgresql':
        return schemas_dir / 'postgresql_schema.sql'
    raise ValueError(f'Unsupported backend type: {backend_type}')


def load_schema(backend_type: str) -> str:
    """Load the SQL schema for a given backend type.

    Args:
        backend_type: The storage backend type ('sqlite', 'postgresql')

    Returns:
        SQL schema as a string

    Raises:
        FileNotFoundError: If schema file does not exist
    """
    schema_path = get_schema_path(backend_type)

    if not schema_path.exists():
        raise FileNotFoundError(f'Schema file not found: {schema_path}')

    return schema_path.read_text(encoding='utf-8')


__all__ = ['get_schema_path', 'load_schema']
