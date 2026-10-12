"""Tests for applying the PostgreSQL row-level security objects.

Covers app.migrations.rls: ``apply_rls`` runs only on PostgreSQL with
``ENABLE_POSTGRESQL_RLS``; under the schema-init lock it creates the helper
functions and grants them, the guard trigger and the policies of every managed
table that exists, enables row-level security where it is off, and fingerprints
every object with the ``sha`` of its rendered statement and the digest of the
server's own deparse read back in the same transaction. A failure surfaces as
the migration error every other migration raises.
"""

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
from app.migrations.rls import read_definition
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import comment_statement
from app.migrations.rls_ddl import digest
from app.migrations.rls_ddl import enable_statement
from app.migrations.rls_ddl import guard_function
from app.migrations.rls_ddl import guard_trigger
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import helper_grant
from app.migrations.rls_ddl import table_policies
from app.settings import AppSettings

SCHEMA = '"tenant one"'
ROLE = '"mcp_context_request"'
LOCK = 'begin_migration'


def _settings(*, enabled: bool = True) -> AppSettings:
    return cast(
        AppSettings,
        SimpleNamespace(
            access_control=SimpleNamespace(postgresql_rls_enabled=enabled, postgresql_rls_policy_role='mcp_context_request'),
            storage=SimpleNamespace(postgresql_schema='tenant one', postgresql_migration_timeout_s=300.0),
        ),
    )


class _CatalogConnection:
    """Answers the catalog reads of apply_rls: the existing managed tables and every deparse."""

    def __init__(self, tables: list[tuple[str, bool]]) -> None:
        self.tables = tables
        self.reads: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, sql: str, *args: object) -> list[tuple[str, bool]]:
        self.reads.append((sql, args))
        assert 'pg_catalog.pg_class' in sql
        assert args[0] == 'tenant one'
        return self.tables

    async def fetchval(self, sql: str, *args: object) -> str:
        self.reads.append((sql, args))
        if 'pg_get_functiondef' in sql:
            return f'functiondef of {args[0]}'
        assert 'pg_get_triggerdef' in sql
        return f'triggerdef of {args[1]} on {args[0]}'

    async def fetchrow(self, sql: str, *args: object) -> tuple[str, bool, list[str], str | None, str | None]:
        self.reads.append((sql, args))
        assert 'pg_catalog.pg_policy' in sql
        return ('r', True, ['mcp_context_request'], f'qual of {args[1]}', None)


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
    async def test_a_failure_surfaces_as_the_migration_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _begin(_conn: object, _timeout_s: float) -> None:
            raise OSError('connection reset')

        monkeypatch.setattr(rls_module, 'begin_migration', _begin)
        backend = _Backend(_CatalogConnection([]))

        with pytest.raises(RuntimeError, match='Row-level security migration failed: connection reset'):
            await apply_rls(cast(StorageBackend, backend), _settings())


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

        for obj in (guard_trigger(SCHEMA), guard_function(SCHEMA), table_policies('tags', schema=SCHEMA, role=ROLE)[0]):
            with pytest.raises(LookupError, match=obj.name):
                await read_definition(cast(Any, _Empty()), obj)
