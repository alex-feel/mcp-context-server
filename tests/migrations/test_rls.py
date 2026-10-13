"""Tests for applying the PostgreSQL row-level security objects to a database that has none.

Covers app.migrations.rls: ``apply_rls`` runs only on PostgreSQL with
``ENABLE_POSTGRESQL_RLS``; on a database holding none of the objects it takes the
schema-init lock and creates the helper functions under a search path of pg_catalog
alone, restores the transaction's own search path and grants them, creates the guard
trigger and the policies of every managed table that exists, enables row-level security
where it is off, creates the read views last, ``context_entries_readable`` and then
``tags_readable``, which reads through it, and fingerprints every object with the ``sha``
of its rendered statement and the digest of the server's own deparse read back in the
same transaction, each table's read policy also recording whether row-level security was
on before. ``create_read_view`` renders each view from its table's current columns,
leaves SELECT for the policy role as the only privilege anyone but the owner holds on it,
logging what it revokes, and makes it read-only; ``drop_read_view`` drops
``tags_readable`` first. A failure surfaces as the migration error every other migration
raises.
"""

import logging
from typing import Any
from typing import cast

import pytest

from app.access_scope import READ_VIEW
from app.access_scope import TAGS_READ_VIEW
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.migrations import rls as rls_module
from app.migrations.rls import apply_rls
from app.migrations.rls import create_read_view
from app.migrations.rls import drop_read_view
from app.migrations.rls import read_definition
from app.migrations.rls_ddl import PIN_SEARCH_PATH
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import enable_statement
from app.migrations.rls_ddl import guard_function
from app.migrations.rls_ddl import guard_trigger
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import helper_grant
from app.migrations.rls_ddl import records_preenabled
from app.migrations.rls_ddl import restore_search_path
from app.migrations.rls_ddl import table_policies
from app.migrations.rls_state import RlsState
from app.migrations.rls_view_ddl import read_view_drop
from app.migrations.rls_view_ddl import read_view_grant
from app.migrations.rls_view_ddl import read_view_revoke
from app.migrations.rls_view_ddl import reject_function
from app.migrations.rls_view_ddl import reject_trigger
from tests.migrations._rls_fakes import LOCK
from tests.migrations._rls_fakes import ROLE
from tests.migrations._rls_fakes import ROLE_NAME
from tests.migrations._rls_fakes import SCHEMA
from tests.migrations._rls_fakes import SCHEMA_NAME
from tests.migrations._rls_fakes import SEARCH_PATH
from tests.migrations._rls_fakes import TAGS_VIEW
from tests.migrations._rls_fakes import TIMEOUT_S
from tests.migrations._rls_fakes import VIEW
from tests.migrations._rls_fakes import CatalogConnection
from tests.migrations._rls_fakes import FakeBackend
from tests.migrations._rls_fakes import fingerprinted
from tests.migrations._rls_fakes import record_ddl
from tests.migrations._rls_fakes import rendered_tags_view
from tests.migrations._rls_fakes import rendered_view
from tests.migrations._rls_fakes import rls_settings
from tests.migrations._rls_fakes import table_columns

GRANT = read_view_grant(SCHEMA, ROLE, view=READ_VIEW)
TAGS_GRANT = read_view_grant(SCHEMA, ROLE, view=TAGS_READ_VIEW)


def _empty_state(tables: dict[str, bool]) -> RlsState:
    """Return the state of a database whose managed tables exist and hold none of the objects."""
    return RlsState(tables=tables, objects={}, columns=table_columns(tables), views={}, helpers_executable=False)


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the lock and every DDL statement the migration issues, in order.

    Returns:
        The recorded statements; the lock appears as ``begin_migration``.
    """
    return record_ddl(monkeypatch, [rls_module])


def _serve_states(monkeypatch: pytest.MonkeyPatch, state: RlsState) -> list[str]:
    """Answer every catalog read of apply_rls with ``state``; return the schema and role each read named."""
    reads: list[str] = []

    async def _read(_conn: object, *, schema_name: str, role: str | None) -> RlsState:
        reads.append(f'{schema_name}/{role}')
        return state

    monkeypatch.setattr(rls_module, 'read_rls_state', _read)
    return reads


async def _created_view(conn: CatalogConnection, acl_statements: list[str]) -> list[str]:
    """Return what creating the read view of context_entries issues around the given ACL statements."""
    view = rendered_view()
    function = reject_function(SCHEMA)
    trigger = reject_trigger(SCHEMA, view=READ_VIEW)
    return [
        view.statement,
        *acl_statements,
        function.statement,
        trigger.statement,
        await fingerprinted(conn, function),
        await fingerprinted(conn, trigger),
        await fingerprinted(conn, view),
    ]


async def _created_tags_view(conn: CatalogConnection, acl_statements: list[str]) -> list[str]:
    """Return what creating tags_readable issues around the given ACL statements."""
    view = rendered_tags_view()
    trigger = reject_trigger(SCHEMA, view=TAGS_READ_VIEW)
    return [
        view.statement,
        *acl_statements,
        trigger.statement,
        await fingerprinted(conn, trigger),
        await fingerprinted(conn, view),
    ]


async def _created_views(conn: CatalogConnection) -> list[str]:
    """Return what create_read_view issues for both views on a database whose new views hold no grant."""
    return [*await _created_view(conn, [GRANT]), *await _created_tags_view(conn, [TAGS_GRANT])]


async def _create(conn: CatalogConnection) -> None:
    await create_read_view(cast(Any, conn), schema=SCHEMA_NAME, policy_role=ROLE_NAME, timeout_s=TIMEOUT_S)


class TestGating:
    """Only a PostgreSQL backend with ENABLE_POSTGRESQL_RLS gets the objects."""

    @pytest.mark.asyncio
    async def test_sqlite_backend_is_never_touched(self, recorded: list[str]) -> None:
        backend = FakeBackend(CatalogConnection(), backend_type='sqlite')

        await apply_rls(cast(StorageBackend, backend), rls_settings())

        assert backend.scopes == []
        assert recorded == []

    @pytest.mark.asyncio
    async def test_flag_off_leaves_postgresql_untouched(self, recorded: list[str]) -> None:
        backend = FakeBackend(CatalogConnection())

        await apply_rls(cast(StorageBackend, backend), rls_settings(enabled=False))

        assert backend.scopes == []
        assert recorded == []


class TestApplyToAnEmptyDatabase:
    """Every object is created under the lock in dependency order, each with its fingerprint."""

    @pytest.mark.asyncio
    async def test_statements_follow_the_dependency_order_with_a_fingerprint_per_object(
        self, recorded: list[str], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tables = {'context_entries': False, 'context_entry_grants': True, 'tags': False}
        reads = _serve_states(monkeypatch, _empty_state(tables))
        conn = CatalogConnection()
        backend = FakeBackend(conn)

        await apply_rls(cast(StorageBackend, backend), rls_settings())

        helpers = helper_functions(SCHEMA)
        function = guard_function(SCHEMA)
        trigger = guard_trigger(SCHEMA)
        expected = [
            LOCK,
            PIN_SEARCH_PATH,
            *(helper.statement for helper in helpers),
            restore_search_path(SEARCH_PATH),
            helper_grant(SCHEMA, ROLE),
            *[await fingerprinted(conn, helper) for helper in helpers],
            function.statement,
            await fingerprinted(conn, function),
            trigger.drop_statement,
            trigger.statement,
            await fingerprinted(conn, trigger),
        ]
        for table, enabled in tables.items():
            for policy in table_policies(table, schema=SCHEMA, role=ROLE):
                preenabled = enabled if records_preenabled(policy) else None
                expected.extend([
                    policy.drop_statement, policy.statement, await fingerprinted(conn, policy, preenabled=preenabled),
                ])
            if not enabled:
                expected.append(enable_statement(table, schema=SCHEMA))
        expected.extend([read_view_drop(SCHEMA, view=TAGS_READ_VIEW), read_view_drop(SCHEMA, view=READ_VIEW)])
        expected.extend(await _created_views(conn))

        assert recorded == expected
        assert backend.scopes == [MAINTENANCE]
        assert reads == [f'{SCHEMA_NAME}/{ROLE}', f'{SCHEMA_NAME}/{ROLE}']

    @pytest.mark.asyncio
    async def test_absent_tables_get_no_policies_and_no_tags_view(
        self, recorded: list[str], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _serve_states(monkeypatch, _empty_state({'context_entries': True, 'context_entry_grants': True}))
        backend = FakeBackend(CatalogConnection(tag_columns=()))

        await apply_rls(cast(StorageBackend, backend), rls_settings())

        ddl = '\n'.join(recorded)
        assert 'context_entries_read' in ddl
        assert 'context_entry_grants_insert' in ddl
        for absent in ('tags', 'image_attachments', 'embedding_metadata', 'vec_context_embeddings', 'context_index_nodes'):
            assert f'{absent}_read' not in ddl
        assert 'ENABLE ROW LEVEL SECURITY' not in ddl
        assert f'CREATE VIEW {VIEW} ' in ddl
        assert f'CREATE VIEW {TAGS_VIEW} ' not in ddl

    @pytest.mark.asyncio
    async def test_no_view_or_guard_trigger_without_the_context_table(
        self, recorded: list[str], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _serve_states(monkeypatch, _empty_state({'tags': True}))
        backend = FakeBackend(CatalogConnection())

        await apply_rls(cast(StorageBackend, backend), rls_settings())

        ddl = '\n'.join(recorded)
        assert 'tags_read' in ddl
        assert 'context_entries_readable' not in ddl
        assert 'tags_readable' not in ddl
        assert 'guard_context_entry_access_columns' not in ddl

    @pytest.mark.asyncio
    async def test_a_failure_surfaces_as_the_migration_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _read(_conn: object, *, schema_name: str, role: str | None) -> RlsState:
            _ = schema_name, role
            raise OSError('connection reset')

        monkeypatch.setattr(rls_module, 'read_rls_state', _read)
        backend = FakeBackend(CatalogConnection())

        with pytest.raises(RuntimeError, match='Row-level security migration failed: connection reset'):
            await apply_rls(cast(StorageBackend, backend), rls_settings())


class TestCreateReadView:
    """Each view is rendered from its table's live columns, keeps only the policy role's SELECT and rejects writes."""

    @pytest.mark.asyncio
    async def test_fresh_views_get_the_policy_roles_select_and_their_fingerprints_in_dependency_order(
        self, recorded: list[str], caplog: pytest.LogCaptureFixture,
    ) -> None:
        conn = CatalogConnection()

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await _create(conn)

        assert recorded == await _created_views(conn)
        assert LOCK not in recorded
        assert caplog.records == []

    @pytest.mark.asyncio
    async def test_columns_are_read_from_each_table_in_attribute_order(self, recorded: list[str]) -> None:
        conn = CatalogConnection()

        await _create(conn)

        column_reads = [(sql, args) for sql, args in conn.reads if 'pg_catalog.pg_attribute' in sql]
        assert [args for _, args in column_reads] == [(f'{SCHEMA}.context_entries',), (f'{SCHEMA}.tags',)]
        for sql, _ in column_reads:
            assert 'a.attnum > 0' in sql
            assert 'NOT a.attisdropped' in sql
            assert sql.endswith('ORDER BY a.attnum')
        assert recorded[0] == rendered_view().statement
        assert rendered_tags_view().statement in recorded

    @pytest.mark.asyncio
    async def test_without_the_tags_table_only_the_read_view_of_context_entries_is_created(self, recorded: list[str]) -> None:
        conn = CatalogConnection(tag_columns=())

        await _create(conn)

        assert recorded == await _created_view(conn, [GRANT])

    @pytest.mark.asyncio
    async def test_every_other_grantee_loses_everything_and_the_policy_role_keeps_select_alone(
        self, recorded: list[str], caplog: pytest.LogCaptureFixture,
    ) -> None:
        conn = CatalogConnection(acl=[
            (None, 'SELECT'),
            (ROLE_NAME, 'INSERT'),
            (ROLE_NAME, 'SELECT'),
            (ROLE_NAME, 'UPDATE'),
            ('mcp_outsider', 'INSERT'),
            ('mcp_outsider', 'SELECT'),
        ])

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await _create(conn)

        assert recorded == [
            *await _created_view(conn, [
                read_view_revoke(SCHEMA, 'PUBLIC', ('ALL',), view=READ_VIEW),
                read_view_revoke(SCHEMA, '"mcp_outsider"', ('ALL',), view=READ_VIEW),
                read_view_revoke(SCHEMA, ROLE, ('INSERT', 'UPDATE'), view=READ_VIEW),
            ]),
            *await _created_tags_view(conn, [TAGS_GRANT]),
        ]
        [warning] = caplog.records
        assert warning.levelno == logging.WARNING
        message = warning.getMessage()
        assert VIEW in message
        assert 'PUBLIC (SELECT)' in message
        assert 'mcp_outsider (INSERT, SELECT)' in message
        assert 'mcp_context_request (INSERT, UPDATE)' in message

    @pytest.mark.asyncio
    async def test_default_privileges_on_the_new_tags_view_are_revoked_with_a_warning_naming_it(
        self, recorded: list[str], caplog: pytest.LogCaptureFixture,
    ) -> None:
        conn = CatalogConnection(acl=[(ROLE_NAME, 'SELECT')], tags_acl=[(None, 'SELECT'), ('mcp_outsider', 'SELECT')])

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await _create(conn)

        assert recorded == [
            *await _created_view(conn, []),
            *await _created_tags_view(conn, [
                read_view_revoke(SCHEMA, 'PUBLIC', ('ALL',), view=TAGS_READ_VIEW),
                read_view_revoke(SCHEMA, '"mcp_outsider"', ('ALL',), view=TAGS_READ_VIEW),
                TAGS_GRANT,
            ]),
        ]
        [warning] = caplog.records
        message = warning.getMessage()
        assert TAGS_VIEW in message
        assert VIEW not in message
        assert 'PUBLIC (SELECT)' in message
        assert 'mcp_outsider (SELECT)' in message

    @pytest.mark.asyncio
    async def test_the_policy_role_without_select_is_granted_it(
        self, recorded: list[str], caplog: pytest.LogCaptureFixture,
    ) -> None:
        conn = CatalogConnection(acl=[(ROLE_NAME, 'DELETE')], tags_acl=[(ROLE_NAME, 'SELECT')])

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await _create(conn)

        assert recorded == [
            *await _created_view(conn, [read_view_revoke(SCHEMA, ROLE, ('DELETE',), view=READ_VIEW), GRANT]),
            *await _created_tags_view(conn, []),
        ]
        [warning] = caplog.records
        assert 'mcp_context_request (DELETE)' in warning.getMessage()

    @pytest.mark.asyncio
    async def test_a_clean_acl_needs_no_statement(self, recorded: list[str], caplog: pytest.LogCaptureFixture) -> None:
        conn = CatalogConnection(acl=[(ROLE_NAME, 'SELECT')], tags_acl=[(ROLE_NAME, 'SELECT')])

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await _create(conn)

        acl_reads = [(sql, args) for sql, args in conn.reads if 'aclexplode' in sql]
        assert [args for _, args in acl_reads] == [(VIEW,), (TAGS_VIEW,)]
        assert all('a.grantee <> c.relowner' in sql for sql, _ in acl_reads)
        assert recorded == [*await _created_view(conn, []), *await _created_tags_view(conn, [])]
        assert caplog.records == []


class TestDropReadView:
    """Dropping the views takes every grant on them along, and names each one beyond the policy role's SELECT."""

    @staticmethod
    async def _drop(conn: CatalogConnection) -> None:
        await drop_read_view(cast(Any, conn), schema=SCHEMA_NAME, policy_role=ROLE_NAME, timeout_s=TIMEOUT_S)

    @pytest.mark.asyncio
    async def test_tags_readable_is_dropped_first_and_each_drop_names_the_grants_it_takes_away(
        self, recorded: list[str], caplog: pytest.LogCaptureFixture,
    ) -> None:
        conn = CatalogConnection(
            acl=[(None, 'SELECT'), (ROLE_NAME, 'INSERT'), (ROLE_NAME, 'SELECT'), ('mcp_outsider', 'SELECT')],
            tags_acl=[(ROLE_NAME, 'SELECT'), ('mcp_outsider', 'INSERT')],
        )

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await self._drop(conn)

        assert recorded == [read_view_drop(SCHEMA, view=TAGS_READ_VIEW), read_view_drop(SCHEMA, view=READ_VIEW)]
        assert [args for sql, args in conn.reads if 'aclexplode' in sql] == [(TAGS_VIEW,), (VIEW,)]
        tags_warning, warning = caplog.records
        assert TAGS_VIEW in tags_warning.getMessage()
        assert 'mcp_outsider (INSERT)' in tags_warning.getMessage()
        assert warning.levelno == logging.WARNING
        message = warning.getMessage()
        assert VIEW in message
        assert 'PUBLIC (SELECT)' in message
        assert 'mcp_outsider (SELECT)' in message
        assert 'mcp_context_request (INSERT)' in message

    @pytest.mark.asyncio
    async def test_views_holding_only_the_policy_roles_select_are_dropped_without_a_warning(
        self, recorded: list[str], caplog: pytest.LogCaptureFixture,
    ) -> None:
        conn = CatalogConnection(acl=[(ROLE_NAME, 'SELECT')], tags_acl=[(ROLE_NAME, 'SELECT')])

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await self._drop(conn)

        assert recorded == [read_view_drop(SCHEMA, view=TAGS_READ_VIEW), read_view_drop(SCHEMA, view=READ_VIEW)]
        assert caplog.records == []


class TestReadDefinition:
    """The deparse digest covers what the server stores for each object."""

    @pytest.mark.asyncio
    async def test_function_and_trigger_read_their_server_definitions(self) -> None:
        conn = CatalogConnection()

        assert await read_definition(cast(Any, conn), helper_functions(SCHEMA)[0]) == (
            f'functiondef of {SCHEMA}.mcp_context_principal_id()'
        )
        assert await read_definition(cast(Any, conn), guard_trigger(SCHEMA)) == (
            f'triggerdef of guard_context_entry_access_columns on {SCHEMA}.context_entries'
        )
        assert await read_definition(cast(Any, conn), reject_trigger(SCHEMA, view=TAGS_READ_VIEW)) == (
            f'triggerdef of reject_writes on {TAGS_VIEW}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('view', [rendered_view(), rendered_tags_view()], ids=['context-entries', 'tags'])
    async def test_a_view_reads_its_pretty_printed_server_definition(self, view: RlsObject) -> None:
        conn = CatalogConnection()

        assert await read_definition(cast(Any, conn), view) == f'viewdef of {view.anchor}'
        sql, args = conn.reads[-1]
        assert args == (view.anchor,)
        assert 'pg_get_viewdef(pg_catalog.to_regclass($1), true)' in sql

    @pytest.mark.asyncio
    async def test_policy_definition_joins_command_kind_roles_and_both_expressions(self) -> None:
        conn = CatalogConnection()
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
            rendered_view(),
            rendered_tags_view(),
        )
        for obj in objects:
            with pytest.raises(LookupError, match=obj.name):
                await read_definition(cast(Any, _Empty()), obj)
