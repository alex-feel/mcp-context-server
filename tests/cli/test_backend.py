"""Tests for the backend construction and the owner precondition shared by the maintenance CLI modes.

Every mode builds its backend through ``make_backend``, which hands a PostgreSQL URL, with
its own credentials, to the maintenance factory, and refuses to touch data when
row-level security filters the role it connected as (``assert_maintenance_connection``).
``owner_refusal`` turns that refusal, or a privilege error the database raises anywhere
in a run, into the remediation the dispatcher prints.
"""

import ast
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.postgresql_backend.session import NO_CREDENTIAL_OVERRIDE
from app.backends.postgresql_backend.session_profiles import MaintenanceProfile
from app.backends.sqlite_backend import SQLiteBackend
from app.cli import _backend
from app.cli._backend import FILTERED_TABLES_SQL
from app.cli._backend import MaintenanceConnectionError
from app.cli._backend import assert_maintenance_connection
from app.cli._backend import make_backend
from app.cli._backend import owner_refusal
from app.migrations.rls_ddl import MANAGED_TABLES
from app.settings import get_settings

CLI_ROOT = Path(__file__).resolve().parents[2] / 'app' / 'cli'
OWNER_DSN = 'postgresql://mcp_owner:owner-secret@db.internal:5432/mcp_context'
REFUSAL = (
    "This command must connect with the table owner's credentials: row-level security filters mcp_app on "
    "context_entries, tags. Rerun with the owner's connection string (docs/migration-v2-to-v3.md)."
)


class _StubBackend:
    """Stand-in for the backend the faked factory returns; ``make_backend`` only constructs it."""


class TestMakeBackend:
    """``make_backend`` builds every backend through the maintenance factory."""

    def test_a_postgresql_url_goes_to_the_maintenance_factory_with_its_own_credentials(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def _fake_factory(**kwargs: object) -> _StubBackend:
            captured.update(kwargs)
            return _StubBackend()

        monkeypatch.setattr(_backend, 'create_maintenance_backend', _fake_factory)

        make_backend(OWNER_DSN, provision_vector=False)

        assert captured == {'connection_string': OWNER_DSN, 'provision_vector': False}

    def test_a_sqlite_path_goes_to_the_maintenance_factory_without_provision_vector(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def _fake_factory(**kwargs: object) -> _StubBackend:
            captured.update(kwargs)
            return _StubBackend()

        monkeypatch.setattr(_backend, 'create_maintenance_backend', _fake_factory)

        make_backend('sqlite:///tmp/context.db', provision_vector=True)

        assert captured == {'db_path': '/tmp/context.db'}

    def test_the_postgresql_backend_runs_the_maintenance_profile_with_no_credential_override(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv('POSTGRESQL_MIGRATION_USER', 'someone_else')
        monkeypatch.setenv('POSTGRESQL_MIGRATION_PASSWORD', 'other-secret')
        get_settings.cache_clear()
        try:
            backend = make_backend(OWNER_DSN)
        finally:
            get_settings.cache_clear()

        assert isinstance(backend, PostgreSQLBackend)
        assert isinstance(backend._profile, MaintenanceProfile)
        assert backend._credentials == NO_CREDENTIAL_OVERRIDE

    def test_the_sqlite_backend_opens_the_given_path(self, tmp_path: Path) -> None:
        backend = make_backend(str(tmp_path / 'context.db'))

        assert isinstance(backend, SQLiteBackend)
        assert backend.db_path == tmp_path / 'context.db'


class _CatalogConnection:
    """A raw connection that answers the precondition with ``filtered`` tables for ``current_user``, recording each query."""

    def __init__(self, current_user: str, filtered: list[str]) -> None:
        self.answer = (current_user, filtered)
        self.queries: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, query: str, *args: object) -> tuple[str, list[str]]:
        self.queries.append((query, args))
        return self.answer


def _backend_over(conn: _CatalogConnection, backend_class: type, backend_type: str) -> tuple[MagicMock, list[BackendAccess]]:
    """Return a backend of ``backend_class`` that runs each read on ``conn``, and the access each read was given."""
    scopes: list[BackendAccess] = []

    async def _execute_read(operation: Callable[[object], Awaitable[object]], *, access_scope: BackendAccess) -> object:
        scopes.append(access_scope)
        return await operation(conn)

    backend = MagicMock(spec=backend_class)
    backend.backend_type = backend_type
    backend.execute_read = _execute_read
    return backend, scopes


def _qualified(schema: str) -> list[str]:
    return [f'{schema}.{table}' for table in MANAGED_TABLES]


class TestAssertMaintenanceConnection:
    """The precondition asks the server whether row-level security filters the connected role on any managed table."""

    @pytest.mark.asyncio
    async def test_an_unfiltered_connection_passes(self) -> None:
        conn = _CatalogConnection('mcp_owner', [])

        await assert_maintenance_connection(cast('asyncpg.Connection[Any]', conn), 'public')

        assert conn.queries == [(FILTERED_TABLES_SQL, (_qualified('"public"'),))]

    @pytest.mark.asyncio
    async def test_a_filtered_connection_is_refused_naming_the_role_and_the_tables(self) -> None:
        conn = _CatalogConnection('mcp_app', ['context_entries', 'tags'])

        with pytest.raises(MaintenanceConnectionError) as refused:
            await assert_maintenance_connection(cast('asyncpg.Connection[Any]', conn), 'public')

        assert str(refused.value) == REFUSAL

    @pytest.mark.asyncio
    async def test_the_managed_names_are_qualified_with_the_quoted_schema(self) -> None:
        conn = _CatalogConnection('mcp_owner', [])

        await assert_maintenance_connection(cast('asyncpg.Connection[Any]', conn), 'tenant one')

        assert conn.queries == [(FILTERED_TABLES_SQL, (_qualified('"tenant one"'),))]

    @pytest.mark.asyncio
    async def test_a_postgresql_backend_reads_through_the_maintenance_access(self) -> None:
        conn = _CatalogConnection('mcp_app', ['context_entries', 'tags'])
        backend, scopes = _backend_over(conn, PostgreSQLBackend, 'postgresql')

        with pytest.raises(MaintenanceConnectionError) as refused:
            await assert_maintenance_connection(cast(StorageBackend, backend), 'public')

        assert str(refused.value) == REFUSAL
        assert scopes == [MAINTENANCE]

    @pytest.mark.asyncio
    async def test_a_sqlite_backend_has_nothing_to_check(self) -> None:
        conn = _CatalogConnection('unused', [])
        backend, scopes = _backend_over(conn, SQLiteBackend, 'sqlite')

        await assert_maintenance_connection(cast(StorageBackend, backend), 'public')

        assert scopes == []
        assert conn.queries == []

    def test_the_check_holds_whatever_row_security_says(self) -> None:
        assert 'pg_catalog.row_security_active(c.oid)' in FILTERED_TABLES_SQL
        assert 'pg_catalog.to_regclass' in FILTERED_TABLES_SQL


def _raise_chained(outer: Exception, inner: BaseException, *, explicit: bool) -> None:
    """Raise ``outer`` from ``inner`` (``explicit``) or while handling it."""
    try:
        raise inner
    except type(inner) as exc:
        if explicit:
            raise outer from exc
        raise outer from None


def _chained(outer: Exception, inner: BaseException, *, explicit: bool) -> Exception:
    """Return ``outer`` as raised from ``inner`` (``explicit``) or while handling it."""
    with pytest.raises(type(outer)) as raised:
        _raise_chained(outer, inner, explicit=explicit)
    return raised.value


class TestOwnerRefusal:
    """A precondition refusal or a privilege error anywhere in a run maps to the owner remediation."""

    def test_the_precondition_refusal_keeps_its_message(self) -> None:
        assert owner_refusal(MaintenanceConnectionError(REFUSAL)) == REFUSAL

    def test_a_privilege_error_wrapped_by_a_mode_maps_to_the_remediation(self) -> None:
        wrapped = _chained(
            RuntimeError('Owner reassignment failed: permission denied for table context_entries'),
            asyncpg.exceptions.InsufficientPrivilegeError('permission denied for table context_entries'),
            explicit=True,
        )

        assert owner_refusal(wrapped) == (
            "This command must connect with the table owner's credentials: the database refused a statement "
            '(Owner reassignment failed: permission denied for table context_entries). '
            "Rerun with the owner's connection string (docs/migration-v2-to-v3.md)."
        )

    def test_a_privilege_error_raised_while_handling_another_maps_too(self) -> None:
        error = _chained(
            asyncpg.exceptions.InsufficientPrivilegeError(
                'query would be affected by row-level security policy for table "tags"',
            ),
            OSError('connection reset'),
            explicit=False,
        )

        message = owner_refusal(error)

        assert message is not None
        assert 'query would be affected by row-level security policy' in message

    def test_any_other_failure_is_not_a_refusal(self) -> None:
        wrapped = _chained(RuntimeError('compression failed'), OSError('connection reset'), explicit=True)

        assert owner_refusal(wrapped) is None


def _calls_create_backend(tree: ast.AST) -> list[int]:
    """Return the line of every import or call of ``create_backend`` in ``tree``."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(alias.name == 'create_backend' for alias in node.names):
            lines.append(node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name == 'create_backend':
                lines.append(node.lineno)
    return lines


def test_no_cli_module_builds_a_backend_with_the_request_factory() -> None:
    """Every command-line backend comes from ``make_backend``, which runs the maintenance factory."""
    offenders = {
        str(path.relative_to(CLI_ROOT)): lines
        for path in sorted(CLI_ROOT.rglob('*.py'))
        if (lines := _calls_create_backend(ast.parse(path.read_text(encoding='utf-8'))))
    }

    assert offenders == {}
