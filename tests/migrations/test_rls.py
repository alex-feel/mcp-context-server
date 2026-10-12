"""Tests for applying the PostgreSQL row-level security objects.

Covers app.migrations.rls: ``apply_rls`` runs only on PostgreSQL with
``ENABLE_POSTGRESQL_RLS``; under the schema-init lock it creates the helper
functions and grants them, the guard trigger and the policies of every managed
table that exists, enables row-level security where it is off, re-creates the
read view of ``context_entries`` last, and fingerprints every object with the
``sha`` of its rendered statement and the digest of the server's own deparse
read back in the same transaction. ``create_read_view`` renders the view from
the table's current columns, leaves SELECT for the policy role as the only
privilege anyone but the owner holds on it, logging what it revokes, and makes
it read-only. A failure surfaces as the migration error every other migration
raises.
"""

import logging
from collections.abc import Awaitable
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from typing import cast

import pytest

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.migrations import rls as rls_module
from app.migrations.rls import apply_rls
from app.migrations.rls import create_read_view
from app.migrations.rls import read_definition
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import comment_statement
from app.migrations.rls_ddl import digest
from app.migrations.rls_ddl import enable_statement
from app.migrations.rls_ddl import guard_function
from app.migrations.rls_ddl import guard_trigger
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import helper_grant
from app.migrations.rls_ddl import read_view
from app.migrations.rls_ddl import read_view_drop
from app.migrations.rls_ddl import read_view_grant
from app.migrations.rls_ddl import read_view_revoke
from app.migrations.rls_ddl import reject_function
from app.migrations.rls_ddl import reject_trigger
from app.migrations.rls_ddl import table_policies
from app.settings import AppSettings

SCHEMA = '"tenant one"'
ROLE = '"mcp_context_request"'
ROLE_NAME = 'mcp_context_request'
LOCK = 'begin_migration'
VIEW = f'{SCHEMA}.context_entries_readable'
COLUMNS = ('id', 'thread_id', 'text_content')


def _settings(*, enabled: bool = True) -> AppSettings:
    return cast(
        AppSettings,
        SimpleNamespace(
            access_control=SimpleNamespace(postgresql_rls_enabled=enabled, postgresql_rls_policy_role=ROLE_NAME),
            storage=SimpleNamespace(postgresql_schema='tenant one', postgresql_migration_timeout_s=300.0),
        ),
    )


class _CatalogConnection:
    """Answers the catalog reads of apply_rls: the managed tables, the view's columns and ACL, and every deparse."""

    def __init__(self, tables: list[tuple[str, bool]], *, acl: list[tuple[str | None, str]] | None = None) -> None:
        self.tables = tables
        self.acl = acl or []
        self.reads: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, sql: str, *args: object) -> list[tuple[object, ...]]:
        self.reads.append((sql, args))
        if 'aclexplode' in sql:
            assert args == (VIEW,)
            return list(self.acl)
        if 'pg_catalog.pg_attribute' in sql:
            assert args == (f'{SCHEMA}.context_entries',)
            return [(column,) for column in COLUMNS]
        assert 'pg_catalog.pg_class' in sql
        assert args[0] == 'tenant one'
        return list(self.tables)

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


class _Backend:
    """A PostgreSQL backend that runs the migration callable on the catalog connection."""

    def __init__(self, conn: _CatalogConnection, backend_type: str = 'postgresql') -> None:
        self.conn = conn
        self.backend_type = backend_type
        self.scopes: list[BackendAccess] = []

    async def execute_write(self, operation: Callable[[object], Awaitable[None]], *, access_scope: BackendAccess) -> None:
        self.scopes.append(access_scope)
        await operation(self.conn)


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the lock and every DDL statement apply_rls issues, in order.

    Returns:
        The recorded statements; the lock appears as ``begin_migration``.
    """
    statements: list[str] = []

    async def _begin(_conn: object, timeout_s: float) -> None:
        assert timeout_s == 300.0
        statements.append(LOCK)

    async def _execute(_conn: object, statement: str, timeout_s: float) -> None:
        assert timeout_s == 300.0
        statements.append(statement)

    monkeypatch.setattr(rls_module, 'begin_migration', _begin)
    monkeypatch.setattr(rls_module, 'execute_migration_ddl', _execute)
    return statements


async def _fingerprinted(conn: _CatalogConnection, obj: RlsObject) -> str:
    return comment_statement(obj, definition=digest(await read_definition(cast(Any, conn), obj)))


async def _recreated(conn: _CatalogConnection, obj: RlsObject) -> list[str]:
    return [obj.drop_statement, obj.statement, await _fingerprinted(conn, obj)]


def _rendered_view() -> RlsObject:
    return read_view([f'"{column}"' for column in COLUMNS], schema=SCHEMA, role_name=ROLE_NAME)


async def _created_view(conn: _CatalogConnection, acl_statements: list[str]) -> list[str]:
    """Return what create_read_view issues around the given ACL statements."""
    view = _rendered_view()
    function = reject_function(SCHEMA)
    trigger = reject_trigger(SCHEMA)
    return [
        view.statement,
        *acl_statements,
        function.statement,
        trigger.statement,
        await _fingerprinted(conn, function),
        await _fingerprinted(conn, trigger),
        await _fingerprinted(conn, view),
    ]


async def _create(conn: _CatalogConnection) -> None:
    await create_read_view(cast(Any, conn), schema='tenant one', policy_role=ROLE_NAME, timeout_s=300.0)


class TestGating:
    """Only a PostgreSQL backend with ENABLE_POSTGRESQL_RLS gets the objects."""

    @pytest.mark.asyncio
    async def test_sqlite_backend_is_never_touched(self, recorded: list[str]) -> None:
        backend = _Backend(_CatalogConnection([]), backend_type='sqlite')

        await apply_rls(cast(StorageBackend, backend), _settings())

        assert backend.scopes == []
        assert recorded == []

    @pytest.mark.asyncio
    async def test_flag_off_leaves_postgresql_untouched(self, recorded: list[str]) -> None:
        backend = _Backend(_CatalogConnection([('context_entries', False)]))

        await apply_rls(cast(StorageBackend, backend), _settings(enabled=False))

        assert backend.scopes == []
        assert recorded == []


class TestApply:
    """The objects are created under the lock in dependency order, each with its fingerprint."""

    @pytest.mark.asyncio
    async def test_statements_follow_the_dependency_order_with_a_fingerprint_per_object(self, recorded: list[str]) -> None:
        conn = _CatalogConnection([('tags', False), ('context_entries', False), ('context_entry_grants', True)])
        backend = _Backend(conn)

        await apply_rls(cast(StorageBackend, backend), _settings())

        helpers = helper_functions(SCHEMA)
        function = guard_function(SCHEMA)
        expected = [
            LOCK,
            *(helper.statement for helper in helpers),
            helper_grant(SCHEMA, ROLE),
            *[await _fingerprinted(conn, helper) for helper in helpers],
            function.statement,
            await _fingerprinted(conn, function),
            *await _recreated(conn, guard_trigger(SCHEMA)),
        ]
        for table, enabled in (('context_entries', False), ('context_entry_grants', True), ('tags', False)):
            for policy in table_policies(table, schema=SCHEMA, role=ROLE):
                expected.extend(await _recreated(conn, policy))
            if not enabled:
                expected.append(enable_statement(table, schema=SCHEMA))
        expected.append(read_view_drop(SCHEMA))
        expected.extend(await _created_view(conn, [read_view_grant(SCHEMA, ROLE)]))

        assert recorded == expected
        assert backend.scopes == [MAINTENANCE]

    @pytest.mark.asyncio
    async def test_absent_tables_get_no_policies(self, recorded: list[str]) -> None:
        backend = _Backend(_CatalogConnection([('context_entries', True), ('context_entry_grants', True)]))

        await apply_rls(cast(StorageBackend, backend), _settings())

        ddl = '\n'.join(recorded)
        assert 'context_entries_read' in ddl
        assert 'context_entry_grants_insert' in ddl
        for absent in ('tags', 'image_attachments', 'embedding_metadata', 'vec_context_embeddings', 'context_index_nodes'):
            assert f'{absent}_read' not in ddl
        assert 'ENABLE ROW LEVEL SECURITY' not in ddl

    @pytest.mark.asyncio
    async def test_no_view_without_the_context_table(self, recorded: list[str]) -> None:
        backend = _Backend(_CatalogConnection([('tags', True)]))

        await apply_rls(cast(StorageBackend, backend), _settings())

        ddl = '\n'.join(recorded)
        assert 'tags_read' in ddl
        assert 'context_entries_readable' not in ddl

    @pytest.mark.asyncio
    async def test_a_failure_surfaces_as_the_migration_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _begin(_conn: object, _timeout_s: float) -> None:
            raise OSError('connection reset')

        monkeypatch.setattr(rls_module, 'begin_migration', _begin)
        backend = _Backend(_CatalogConnection([]))

        with pytest.raises(RuntimeError, match='Row-level security migration failed: connection reset'):
            await apply_rls(cast(StorageBackend, backend), _settings())


class TestCreateReadView:
    """The view is rendered from the live columns, keeps only the policy role's SELECT and rejects writes."""

    @pytest.mark.asyncio
    async def test_a_fresh_view_gets_the_policy_roles_select_and_its_fingerprints(
        self, recorded: list[str], caplog: pytest.LogCaptureFixture,
    ) -> None:
        conn = _CatalogConnection([])

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await _create(conn)

        assert recorded == await _created_view(conn, [read_view_grant(SCHEMA, ROLE)])
        assert LOCK not in recorded
        assert caplog.records == []

    @pytest.mark.asyncio
    async def test_columns_are_read_from_the_context_table_in_attribute_order(self, recorded: list[str]) -> None:
        conn = _CatalogConnection([])

        await _create(conn)

        sql, args = next((sql, args) for sql, args in conn.reads if 'pg_catalog.pg_attribute' in sql)
        assert args == (f'{SCHEMA}.context_entries',)
        assert 'a.attnum > 0' in sql
        assert 'NOT a.attisdropped' in sql
        assert sql.endswith('ORDER BY a.attnum')
        assert recorded[0] == _rendered_view().statement

    @pytest.mark.asyncio
    async def test_every_other_grantee_loses_everything_and_the_policy_role_keeps_select_alone(
        self, recorded: list[str], caplog: pytest.LogCaptureFixture,
    ) -> None:
        conn = _CatalogConnection([], acl=[
            (None, 'SELECT'),
            (ROLE_NAME, 'INSERT'),
            (ROLE_NAME, 'SELECT'),
            (ROLE_NAME, 'UPDATE'),
            ('mcp_outsider', 'INSERT'),
            ('mcp_outsider', 'SELECT'),
        ])

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await _create(conn)

        assert recorded == await _created_view(conn, [
            read_view_revoke(SCHEMA, 'PUBLIC', ('ALL',)),
            read_view_revoke(SCHEMA, '"mcp_outsider"', ('ALL',)),
            read_view_revoke(SCHEMA, ROLE, ('INSERT', 'UPDATE')),
        ])
        [warning] = caplog.records
        assert warning.levelno == logging.WARNING
        message = warning.getMessage()
        assert VIEW in message
        assert 'PUBLIC (SELECT)' in message
        assert 'mcp_outsider (INSERT, SELECT)' in message
        assert 'mcp_context_request (INSERT, UPDATE)' in message

    @pytest.mark.asyncio
    async def test_the_policy_role_without_select_is_granted_it(
        self, recorded: list[str], caplog: pytest.LogCaptureFixture,
    ) -> None:
        conn = _CatalogConnection([], acl=[(ROLE_NAME, 'DELETE')])

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await _create(conn)

        assert recorded == await _created_view(conn, [
            read_view_revoke(SCHEMA, ROLE, ('DELETE',)),
            read_view_grant(SCHEMA, ROLE),
        ])
        [warning] = caplog.records
        assert 'mcp_context_request (DELETE)' in warning.getMessage()

    @pytest.mark.asyncio
    async def test_a_clean_acl_needs_no_statement(self, recorded: list[str], caplog: pytest.LogCaptureFixture) -> None:
        conn = _CatalogConnection([], acl=[(ROLE_NAME, 'SELECT')])

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await _create(conn)

        sql, _ = next((sql, args) for sql, args in conn.reads if 'aclexplode' in sql)
        assert 'a.grantee <> c.relowner' in sql
        assert recorded == await _created_view(conn, [])
        assert caplog.records == []


class TestReadDefinition:
    """The deparse digest covers what the server stores for each object."""

    @pytest.mark.asyncio
    async def test_function_and_trigger_read_their_server_definitions(self) -> None:
        conn = _CatalogConnection([])

        assert await read_definition(cast(Any, conn), helper_functions(SCHEMA)[0]) == (
            f'functiondef of {SCHEMA}.mcp_context_principal_id()'
        )
        assert await read_definition(cast(Any, conn), guard_trigger(SCHEMA)) == (
            f'triggerdef of guard_context_entry_access_columns on {SCHEMA}.context_entries'
        )

    @pytest.mark.asyncio
    async def test_view_reads_its_pretty_printed_server_definition(self) -> None:
        conn = _CatalogConnection([])

        assert await read_definition(cast(Any, conn), _rendered_view()) == f'viewdef of {VIEW}'
        sql, args = conn.reads[-1]
        assert args == (VIEW,)
        assert 'pg_get_viewdef(pg_catalog.to_regclass($1), true)' in sql

    @pytest.mark.asyncio
    async def test_policy_definition_joins_command_kind_roles_and_both_expressions(self) -> None:
        conn = _CatalogConnection([])
        policy = table_policies('tags', schema=SCHEMA, role=ROLE)[0]

        assert await read_definition(cast(Any, conn), policy) == 'r\npermissive\nmcp_context_request\nqual of tags_read\n'
        sql, args = conn.reads[-1]
        assert args == (f'{SCHEMA}.tags', 'tags_read')
        assert 'pg_get_expr(p.polqual, p.polrelid)' in sql
        assert 'pg_get_expr(p.polwithcheck, p.polrelid)' in sql

    @pytest.mark.asyncio
    async def test_a_missing_object_is_an_error(self) -> None:
        class _Empty:
            async def fetchval(self, *_args: object) -> None:
                return None

            async def fetchrow(self, *_args: object) -> None:
                return None

        objects = (
            guard_trigger(SCHEMA),
            guard_function(SCHEMA),
            table_policies('tags', schema=SCHEMA, role=ROLE)[0],
            _rendered_view(),
        )
        for obj in objects:
            with pytest.raises(LookupError, match=obj.name):
                await read_definition(cast(Any, _Empty()), obj)
