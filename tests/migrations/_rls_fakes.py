"""Fakes shared by the unit tests of the row-level security migration.

:class:`CatalogConnection` answers the catalog reads the migration issues on its own
connection: the server's deparse of each object, the columns of ``context_entries``
and the ACL of the read view. :class:`FakeBackend` runs a migration callable on it the
way ``execute_write`` and ``execute_read`` do. :func:`record_ddl` replaces the lock and
the DDL helper of the given modules, so a test sees every statement in order with
the lock as :data:`LOCK`. :func:`installed_state` builds the catalog state of a database
whose objects are exactly what the migration renders.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from typing import cast

import pytest

from app.backends.base import BackendAccess
from app.migrations.rls import read_definition
from app.migrations.rls_ddl import CONTEXT_TABLE
from app.migrations.rls_ddl import Fingerprint
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import comment_statement
from app.migrations.rls_ddl import digest
from app.migrations.rls_ddl import guard_function
from app.migrations.rls_ddl import guard_trigger
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import read_view
from app.migrations.rls_ddl import records_preenabled
from app.migrations.rls_ddl import reject_function
from app.migrations.rls_ddl import reject_trigger
from app.migrations.rls_ddl import table_policies
from app.migrations.rls_state import InstalledObject
from app.migrations.rls_state import InstalledView
from app.migrations.rls_state import RlsState
from app.settings import AppSettings

SCHEMA = '"tenant one"'
SCHEMA_NAME = 'tenant one'
ROLE = '"mcp_context_request"'
ROLE_NAME = 'mcp_context_request'
LOCK = 'begin_migration'
VIEW = f'{SCHEMA}.context_entries_readable'
COLUMNS = ('id', 'thread_id', 'text_content')
QUOTED_COLUMNS = tuple(f'"{column}"' for column in COLUMNS)
TIMEOUT_S = 300.0


def rls_settings(*, enabled: bool = True, policy_role: str = ROLE_NAME) -> AppSettings:
    """Return settings carrying what the row-level security migration reads.

    Args:
        enabled: The value of ENABLE_POSTGRESQL_RLS.
        policy_role: The value of POSTGRESQL_RLS_POLICY_ROLE.

    Returns:
        The settings.
    """
    return cast(
        AppSettings,
        SimpleNamespace(
            access_control=SimpleNamespace(postgresql_rls_enabled=enabled, postgresql_rls_policy_role=policy_role),
            storage=SimpleNamespace(postgresql_schema=SCHEMA_NAME, postgresql_migration_timeout_s=TIMEOUT_S),
        ),
    )


class CatalogConnection:
    """Answers the catalog reads of the migration: every deparse, the view's columns and its ACL.

    The first read of the view's ACL answers ``acl``; every later one answers
    ``recreated_acl`` when it is given, which is what a view dropped and created again
    holds.
    """

    def __init__(
        self,
        *,
        acl: list[tuple[str | None, str]] | None = None,
        recreated_acl: list[tuple[str | None, str]] | None = None,
    ) -> None:
        self.acl = acl or []
        self.recreated_acl = recreated_acl
        self.reads: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, sql: str, *args: object) -> list[tuple[object, ...]]:
        acl_read_before = any('aclexplode' in read for read, _ in self.reads)
        self.reads.append((sql, args))
        if 'aclexplode' in sql:
            assert args == (VIEW,)
            if acl_read_before and self.recreated_acl is not None:
                return list(self.recreated_acl)
            return list(self.acl)
        assert 'pg_catalog.pg_attribute' in sql
        assert args == (f'{SCHEMA}.context_entries',)
        return [(column,) for column in COLUMNS]

    async def fetchval(self, sql: str, *args: object) -> str:
        self.reads.append((sql, args))
        if 'pg_get_functiondef' in sql:
            return f'functiondef of {args[0]}'
        if 'pg_get_viewdef' in sql:
            return f'viewdef of {args[0]}'
        assert 'pg_get_triggerdef' in sql
        return f'triggerdef of {args[1]} on {args[0]}'

    async def fetchrow(self, sql: str, *args: object) -> tuple[str, bool, list[str], str | None, str | None]:
        self.reads.append((sql, args))
        assert 'pg_catalog.pg_policy' in sql
        return ('r', True, [ROLE_NAME], f'qual of {args[1]}', None)


class FakeBackend:
    """A backend of the given type that runs every operation on one connection and records the access it was given."""

    def __init__(self, conn: object, backend_type: str = 'postgresql') -> None:
        self.conn = conn
        self.backend_type = backend_type
        self.scopes: list[BackendAccess] = []

    async def execute_write(self, operation: Callable[[object], Awaitable[object]], *, access_scope: BackendAccess) -> object:
        self.scopes.append(access_scope)
        return await operation(self.conn)

    async def execute_read(self, operation: Callable[[object], Awaitable[object]], *, access_scope: BackendAccess) -> object:
        self.scopes.append(access_scope)
        return await operation(self.conn)


def record_ddl(monkeypatch: pytest.MonkeyPatch, modules: Iterable[ModuleType]) -> list[str]:
    """Record the lock and every statement the given modules issue through the DDL helper, in order.

    Args:
        monkeypatch: Restores the helpers after the test.
        modules: The modules whose ``begin_migration`` and ``execute_migration_ddl`` are replaced.

    Returns:
        The recorded statements, the lock appearing as :data:`LOCK`.
    """
    statements: list[str] = []

    async def _begin(_conn: object, timeout_s: float) -> None:
        assert timeout_s == TIMEOUT_S
        statements.append(LOCK)

    async def _execute(_conn: object, statement: str, timeout_s: float) -> None:
        assert timeout_s == TIMEOUT_S
        statements.append(statement)

    for module in modules:
        monkeypatch.setattr(module, 'begin_migration', _begin)
        monkeypatch.setattr(module, 'execute_migration_ddl', _execute)
    return statements


async def fingerprinted(conn: CatalogConnection, obj: RlsObject, *, preenabled: bool | None = None) -> str:
    """Return the COMMENT the migration attaches to ``obj`` after reading its deparse from ``conn``."""
    return comment_statement(obj, definition=digest(await read_definition(cast(Any, conn), obj)), preenabled=preenabled)


def rendered_view(*, role_name: str = ROLE_NAME) -> RlsObject:
    """Return the read view the migration renders for :data:`COLUMNS`."""
    return read_view(QUOTED_COLUMNS, schema=SCHEMA, role_name=role_name)


def deparse_digest(obj: RlsObject) -> str:
    """Return the digest a fake server deparse of ``obj`` reads as in :func:`installed_state`."""
    return digest(f'deparse of {obj.target}')


def installed(obj: RlsObject, *, preenabled: bool | None = None) -> InstalledObject:
    """Return ``obj`` as the catalog reads it right after the migration created it."""
    if records_preenabled(obj) and preenabled is None:
        preenabled = False
    definition = deparse_digest(obj)
    return InstalledObject(
        fingerprint=Fingerprint(sha=obj.sha, definition=definition, preenabled=preenabled),
        definition=definition,
        enabled=True,
    )


def installed_state(tables: dict[str, bool], *, preenabled: dict[str, bool] | None = None) -> RlsState:
    """Return the state of a database holding exactly the objects the migration renders for ``tables``.

    Args:
        tables: Each managed table that exists, with whether row-level security is on.
        preenabled: The value each table's read policy recorded; false when absent.

    Returns:
        The state; the read view and its ACL are current when the context table exists.
    """
    recorded = preenabled or {}
    objects: list[RlsObject] = [*helper_functions(SCHEMA), guard_function(SCHEMA), reject_function(SCHEMA)]
    if CONTEXT_TABLE in tables:
        objects.extend((guard_trigger(SCHEMA), reject_trigger(SCHEMA), rendered_view()))
    installed_objects = {obj.key: installed(obj) for obj in objects}
    for table in tables:
        for policy in table_policies(table, schema=SCHEMA, role=ROLE):
            installed_objects[policy.key] = installed(
                policy, preenabled=recorded.get(table, False) if records_preenabled(policy) else None,
            )
    view = None
    if CONTEXT_TABLE in tables:
        view = InstalledView(owned_by_table_owner=True, plain=True, acl={ROLE_NAME: ('SELECT',)})
    return RlsState(
        tables=dict(tables),
        objects=installed_objects,
        columns=QUOTED_COLUMNS if CONTEXT_TABLE in tables else (),
        view=view,
        helpers_executable=True,
    )
