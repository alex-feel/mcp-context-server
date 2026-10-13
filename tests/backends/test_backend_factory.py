"""Tests for backend factory and schema utilities.

This module tests:
- app/backends/factory.py - Backend creation factory
- app/schemas/__init__.py - Schema loading utilities
- app/repositories/base.py - Base repository helper methods
"""

import os
from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

import app.backends.postgresql_backend as pg_module
from app.backends.base import StorageBackend
from app.backends.factory import create_backend
from app.backends.factory import create_maintenance_backend
from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.sqlite_backend import SQLiteBackend
from app.repositories.base import BaseRepository
from app.schemas import get_schema_path
from app.schemas import load_schema
from app.settings import get_settings
from tests.helpers import rebind_package_settings

REQUEST_DSN = 'postgresql://mcp_app:app-secret@db.internal:5432/mcp_context?sslmode=require'
CLI_DSN = 'postgresql://cli_owner:cli-secret@db.internal:5432/mcp_context'

type ConfigureEnv = Callable[..., None]


@pytest.fixture
def configure_postgresql(monkeypatch: pytest.MonkeyPatch) -> Iterator[ConfigureEnv]:
    """Route the settings to PostgreSQL with exactly the given variables and nothing inherited.

    Every ``POSTGRESQL_*`` variable and ``ENABLE_POSTGRESQL_RLS`` is removed first, so the
    shell's own PostgreSQL routing cannot reach the settings under test; the PostgreSQL
    backend package's module-level settings are rebound to the result.

    Yields:
        A callable taking the variables to set as keyword arguments.
    """

    def _configure(**variables: str) -> None:
        for key in list(os.environ):
            if key.startswith('POSTGRESQL_') or key == 'ENABLE_POSTGRESQL_RLS':
                monkeypatch.delenv(key)
        monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
        for key, value in variables.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()
        rebind_package_settings(monkeypatch, pg_module, get_settings())

    yield _configure
    get_settings.cache_clear()


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


async def _dial(backend: StorageBackend, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Initialize ``backend`` over a fake pool and return the ``create_pool`` mock it dialed through.

    Args:
        backend: A PostgreSQL backend built with ``provision_vector=False``, so no probe
            connection runs.
        monkeypatch: Fixture that restores the patched names.

    Returns:
        The awaited ``asyncpg.create_pool`` stand-in; its call carries the DSN and the connect kwargs.
    """
    assert isinstance(backend, PostgreSQLBackend)
    create_pool = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr('asyncpg.create_pool', create_pool)
    monkeypatch.setattr(backend, '_verify_connectivity', AsyncMock())
    monkeypatch.setattr(backend, '_detect_pgpool_ii', AsyncMock())
    monkeypatch.setattr(backend, '_detect_session_mode_pooler', MagicMock())
    await backend.initialize()
    create_pool.assert_awaited_once()
    return create_pool


def _dialed(create_pool: AsyncMock) -> tuple[tuple[object, ...], dict[str, object]]:
    """Return the positional arguments and the user and password kwargs the pool was created with."""
    call = create_pool.await_args
    assert call is not None
    return call.args, {key: call.kwargs[key] for key in ('user', 'password') if key in call.kwargs}


class TestRequestBackendProfile:
    """create_backend picks the request profile from ENABLE_POSTGRESQL_RLS and never takes migration credentials."""

    def test_rls_off_reads_the_base_tables(self, configure_postgresql: ConfigureEnv) -> None:
        configure_postgresql(POSTGRESQL_CONNECTION_STRING=REQUEST_DSN)

        backend = create_backend()

        assert isinstance(backend, PostgreSQLBackend)
        assert backend.readable_relation('context_entries', alias='ce') == 'context_entries ce'
        assert backend.readable_relation('tags', alias='t') == 'tags t'

    def test_rls_on_reads_through_the_view(self, configure_postgresql: ConfigureEnv) -> None:
        configure_postgresql(
            POSTGRESQL_CONNECTION_STRING=REQUEST_DSN, ENABLE_POSTGRESQL_RLS='true', POSTGRESQL_MIGRATION_USER='mcp_owner',
        )

        backend = create_backend()

        assert backend.readable_relation('context_entries', alias='ce') == 'context_entries_readable AS ce'
        assert backend.readable_relation('tags', alias='tags') == 'tags_readable AS tags'
        assert backend.readable_relation('image_attachments', alias='i') == 'image_attachments i'

    @pytest.mark.asyncio
    async def test_request_backend_dials_the_request_credentials(
        self, configure_postgresql: ConfigureEnv, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        configure_postgresql(
            POSTGRESQL_CONNECTION_STRING=REQUEST_DSN,
            ENABLE_POSTGRESQL_RLS='true',
            POSTGRESQL_MIGRATION_USER='mcp_owner',
            POSTGRESQL_MIGRATION_PASSWORD='owner-secret',
        )

        create_pool = await _dial(create_backend(provision_vector=False), monkeypatch)

        assert _dialed(create_pool) == ((REQUEST_DSN,), {})


class TestMaintenanceBackend:
    """create_maintenance_backend dials the request settings with only the set migration fields overriding them."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('migration', 'expected'),
        [
            ({}, {}),
            ({'POSTGRESQL_MIGRATION_USER': 'mcp_owner'}, {'user': 'mcp_owner'}),
            ({'POSTGRESQL_MIGRATION_PASSWORD': 'owner-secret'}, {'password': 'owner-secret'}),
            (
                {'POSTGRESQL_MIGRATION_USER': 'mcp_owner', 'POSTGRESQL_MIGRATION_PASSWORD': 'owner-secret'},
                {'user': 'mcp_owner', 'password': 'owner-secret'},
            ),
        ],
        ids=['no-migration-fields', 'user-only', 'password-only', 'user-and-password'],
    )
    async def test_dsn_form_overrides_only_the_set_fields(
        self,
        configure_postgresql: ConfigureEnv,
        monkeypatch: pytest.MonkeyPatch,
        migration: dict[str, str],
        expected: dict[str, str],
    ) -> None:
        configure_postgresql(POSTGRESQL_CONNECTION_STRING=REQUEST_DSN, **migration)

        create_pool = await _dial(create_maintenance_backend(provision_vector=False), monkeypatch)

        assert _dialed(create_pool) == ((REQUEST_DSN,), expected)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('migration', 'expected'),
        [
            ({}, {}),
            ({'POSTGRESQL_MIGRATION_USER': 'mcp_owner'}, {'user': 'mcp_owner'}),
            (
                {'POSTGRESQL_MIGRATION_USER': 'mcp_owner', 'POSTGRESQL_MIGRATION_PASSWORD': 'owner-secret'},
                {'user': 'mcp_owner', 'password': 'owner-secret'},
            ),
        ],
        ids=['no-migration-fields', 'user-only', 'user-and-password'],
    )
    async def test_component_form_overrides_only_the_set_fields(
        self,
        configure_postgresql: ConfigureEnv,
        monkeypatch: pytest.MonkeyPatch,
        migration: dict[str, str],
        expected: dict[str, str],
    ) -> None:
        configure_postgresql(
            POSTGRESQL_HOST='db.internal',
            POSTGRESQL_PORT='5432',
            POSTGRESQL_USER='mcp_app',
            POSTGRESQL_PASSWORD='app-secret',
            POSTGRESQL_DATABASE='mcp_context',
            **migration,
        )

        create_pool = await _dial(create_maintenance_backend(provision_vector=False), monkeypatch)

        assert _dialed(create_pool) == (('postgresql://mcp_app:app-secret@db.internal:5432/mcp_context',), expected)

    @pytest.mark.asyncio
    async def test_explicit_connection_string_takes_no_override(
        self, configure_postgresql: ConfigureEnv, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        configure_postgresql(
            POSTGRESQL_CONNECTION_STRING=REQUEST_DSN,
            POSTGRESQL_MIGRATION_USER='mcp_owner',
            POSTGRESQL_MIGRATION_PASSWORD='owner-secret',
        )

        create_pool = await _dial(
            create_maintenance_backend(connection_string=CLI_DSN, provision_vector=False), monkeypatch,
        )

        assert _dialed(create_pool) == ((CLI_DSN,), {})

    @pytest.mark.asyncio
    async def test_every_pooled_connection_turns_row_security_off(
        self, configure_postgresql: ConfigureEnv, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        configure_postgresql(POSTGRESQL_CONNECTION_STRING=REQUEST_DSN)
        create_pool = await _dial(create_maintenance_backend(provision_vector=False), monkeypatch)
        conn = MagicMock()
        conn.execute = AsyncMock()
        call = create_pool.await_args
        assert call is not None

        await call.kwargs['setup'](conn)

        executed = conn.execute.await_args
        assert executed is not None
        statements = executed.args[0].split('; ')
        assert 'SET row_security = off' in statements
        assert any(statement.startswith('SET search_path') for statement in statements)

    def test_reads_name_the_base_tables(self, configure_postgresql: ConfigureEnv) -> None:
        configure_postgresql(
            POSTGRESQL_CONNECTION_STRING=REQUEST_DSN, ENABLE_POSTGRESQL_RLS='true', POSTGRESQL_MIGRATION_USER='mcp_owner',
        )

        backend = create_maintenance_backend()

        assert isinstance(backend, PostgreSQLBackend)
        assert backend.readable_relation('context_entries', alias='ce') == 'context_entries ce'

    def test_sqlite_settings_build_the_sqlite_backend(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('STORAGE_BACKEND', 'sqlite')
        monkeypatch.setenv('DB_PATH', str(tmp_path / 'maintenance.db'))
        get_settings.cache_clear()
        try:
            backend = create_maintenance_backend()
        finally:
            get_settings.cache_clear()

        assert isinstance(backend, SQLiteBackend)

    def test_an_explicit_db_path_builds_the_sqlite_backend_at_that_path(
        self, configure_postgresql: ConfigureEnv, tmp_path: Path,
    ) -> None:
        configure_postgresql(POSTGRESQL_CONNECTION_STRING=REQUEST_DSN)

        backend = create_maintenance_backend(db_path=str(tmp_path / 'cli.db'))

        assert isinstance(backend, SQLiteBackend)
        assert backend.db_path == tmp_path / 'cli.db'

    def test_a_db_path_and_a_connection_string_together_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='either a SQLite db_path or a PostgreSQL connection_string'):
            create_maintenance_backend(connection_string=CLI_DSN, db_path=tmp_path / 'cli.db')
