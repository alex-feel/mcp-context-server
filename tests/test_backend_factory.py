"""Tests for backend factory and schema utilities.

This module tests:
- app/backends/factory.py - Backend creation factory
- app/schemas/__init__.py - Schema loading utilities
- app/repositories/base.py - Base repository helper methods
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import MagicMock

import pytest

from app.backends.factory import create_backend
from app.backends.sqlite_backend import SQLiteBackend
from app.repositories.base import BaseRepository
from app.schemas import get_schema_path
from app.schemas import load_schema


class TestBackendFactory:
    """Tests for the create_backend factory function."""

    def test_create_sqlite_backend_with_explicit_path(self, tmp_path: Path) -> None:
        """Test creating SQLite backend with explicit path."""
        db_path = tmp_path / 'test.db'
        backend = create_backend(backend_type='sqlite', db_path=db_path)

        assert isinstance(backend, SQLiteBackend)
        assert backend.backend_type == 'sqlite'

    def test_create_sqlite_backend_with_string_path(self, tmp_path: Path) -> None:
        """Test creating SQLite backend with string path."""
        db_path = str(tmp_path / 'test.db')
        backend = create_backend(backend_type='sqlite', db_path=db_path)

        assert isinstance(backend, SQLiteBackend)
        assert backend.backend_type == 'sqlite'

    def test_create_backend_invalid_type(self) -> None:
        """Test that invalid backend type raises ValueError."""
        # Use cast to pass invalid value for testing error handling
        invalid_type = cast(Any, 'invalid_type')
        with pytest.raises(ValueError, match='Invalid backend_type'):
            create_backend(backend_type=invalid_type)

    def test_create_backend_invalid_type_contains_options(self) -> None:
        """Test error message contains valid options."""
        # Use cast to pass invalid value for testing error handling
        invalid_type = cast(Any, 'mysql')
        with pytest.raises(ValueError, match='sqlite, postgresql'):
            create_backend(backend_type=invalid_type)


class TestSchemaUtilities:
    """Tests for schema loading utilities."""

    def test_get_schema_path_sqlite(self) -> None:
        """Test getting SQLite schema path."""
        path = get_schema_path('sqlite')

        assert path.name == 'sqlite_schema.sql'
        assert path.exists()

    def test_get_schema_path_postgresql(self) -> None:
        """Test getting PostgreSQL schema path."""
        path = get_schema_path('postgresql')

        assert path.name == 'postgresql_schema.sql'
        assert path.exists()

    def test_get_schema_path_unsupported(self) -> None:
        """Test that unsupported backend type raises ValueError."""
        with pytest.raises(ValueError, match='Unsupported backend type'):
            get_schema_path('mysql')

    def test_load_schema_sqlite(self) -> None:
        """Test loading SQLite schema."""
        schema = load_schema('sqlite')

        assert isinstance(schema, str)
        assert 'CREATE TABLE' in schema
        assert 'context_entries' in schema

    def test_load_schema_postgresql(self) -> None:
        """Test loading PostgreSQL schema."""
        schema = load_schema('postgresql')

        assert isinstance(schema, str)
        assert 'CREATE TABLE' in schema
        assert 'context_entries' in schema

    def test_load_schema_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing schema file raises FileNotFoundError."""
        # Monkeypatch get_schema_path to return non-existent path
        def mock_get_schema_path(_backend_type: str) -> Path:
            return Path('/nonexistent/path/schema.sql')

        monkeypatch.setattr('app.schemas.get_schema_path', mock_get_schema_path)

        with pytest.raises(FileNotFoundError, match='Schema file not found'):
            load_schema('sqlite')


class TestBaseRepository:
    """Tests for BaseRepository helper methods."""

    def test_placeholder_sqlite(self) -> None:
        """Test SQLite placeholder generation."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'sqlite'

        repo = BaseRepository(mock_backend)

        # SQLite always uses '?' regardless of position
        assert repo._placeholder(1) == '?'
        assert repo._placeholder(5) == '?'
        assert repo._placeholder(100) == '?'

    def test_placeholder_postgresql(self) -> None:
        """Test PostgreSQL placeholder generation."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'postgresql'

        repo = BaseRepository(mock_backend)

        # PostgreSQL uses $N notation
        assert repo._placeholder(1) == '$1'
        assert repo._placeholder(5) == '$5'
        assert repo._placeholder(100) == '$100'

    def test_placeholders_sqlite(self) -> None:
        """Test SQLite multiple placeholders generation."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'sqlite'

        repo = BaseRepository(mock_backend)

        assert repo._placeholders(1) == '?'
        assert repo._placeholders(3) == '?, ?, ?'
        assert repo._placeholders(5, start=10) == '?, ?, ?, ?, ?'

    def test_placeholders_postgresql(self) -> None:
        """Test PostgreSQL multiple placeholders generation."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'postgresql'

        repo = BaseRepository(mock_backend)

        assert repo._placeholders(1) == '$1'
        assert repo._placeholders(3) == '$1, $2, $3'
        assert repo._placeholders(3, start=5) == '$5, $6, $7'

    def test_json_extract_sqlite(self) -> None:
        """Test SQLite JSON extraction expression."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'sqlite'

        repo = BaseRepository(mock_backend)

        result = repo._json_extract('metadata', 'status')
        assert result == "json_extract(metadata, '$.status')"

        result = repo._json_extract('data', 'user.name')
        assert result == "json_extract(data, '$.user.name')"

    def test_json_extract_postgresql(self) -> None:
        """Test PostgreSQL JSON extraction expression."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'postgresql'

        repo = BaseRepository(mock_backend)

        result = repo._json_extract('metadata', 'status')
        assert result == "metadata->>'status'"

        result = repo._json_extract('data', 'user.name')
        assert result == "data->>'user.name'"

    def test_base_repository_initialization(self) -> None:
        """Test that BaseRepository stores backend reference."""
        mock_backend = MagicMock()
        mock_backend.backend_type = 'sqlite'

        repo = BaseRepository(mock_backend)

        assert repo.backend is mock_backend
