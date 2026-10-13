"""Tests for keeping the PostgreSQL row-level security objects current across boots.

Covers the diff of app.migrations.rls_state (``plan_rls``) and its use in
``app.migrations.rls.apply_rls``: a database whose objects match what the code renders
needs nothing, so a boot takes no lock and issues no DDL; otherwise the migration takes
the schema-init lock, reads the catalog again and changes only what still differs,
re-creating an object whose rendered statement changed silently and one changed outside
the server with a WARNING naming it. Each table's read policy records whether
row-level security was on before the migration first enabled it, and keeps that value
whenever the policy is re-created.
"""

import dataclasses
import logging
from typing import Any
from typing import cast

import pytest

from app.access_scope import READ_VIEW
from app.access_scope import TAGS_READ_VIEW
from app.backends import StorageBackend
from app.migrations import rls as rls_module
from app.migrations.rls_ddl import PIN_SEARCH_PATH
from app.migrations.rls_ddl import Fingerprint
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import enable_statement
from app.migrations.rls_ddl import guard_function
from app.migrations.rls_ddl import guard_trigger
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import helper_grant
from app.migrations.rls_ddl import restore_search_path
from app.migrations.rls_ddl import table_policies
from app.migrations.rls_state import InstalledObject
from app.migrations.rls_state import InstalledView
from app.migrations.rls_state import RlsState
from app.migrations.rls_state import TableRepair
from app.migrations.rls_state import plan_rls
from app.migrations.rls_view_ddl import read_view_drop
from app.migrations.rls_view_ddl import read_view_grant
from app.migrations.rls_view_ddl import read_view_revoke
from app.migrations.rls_view_ddl import reject_function
from app.migrations.rls_view_ddl import reject_trigger
from tests.migrations._rls_fakes import LOCK
from tests.migrations._rls_fakes import QUOTED_TAG_COLUMNS
from tests.migrations._rls_fakes import ROLE
from tests.migrations._rls_fakes import ROLE_NAME
from tests.migrations._rls_fakes import SCHEMA
from tests.migrations._rls_fakes import SEARCH_PATH
from tests.migrations._rls_fakes import TAGS_VIEW
from tests.migrations._rls_fakes import CatalogConnection
from tests.migrations._rls_fakes import FakeBackend
from tests.migrations._rls_fakes import fingerprinted
from tests.migrations._rls_fakes import installed
from tests.migrations._rls_fakes import installed_state
from tests.migrations._rls_fakes import record_ddl
from tests.migrations._rls_fakes import rendered_tags_view
from tests.migrations._rls_fakes import rendered_view
from tests.migrations._rls_fakes import rls_settings
from tests.migrations._rls_fakes import table_columns

TABLES = {'context_entries': True, 'context_entry_grants': True, 'tags': True}
CHANGED_OUT_OF_BAND = 'f' * 64
# The sha of a statement this module no longer renders, as an object created by an earlier release carries it.
EARLIER_STATEMENT = 'e' * 64


def _policy(table: str, command: str) -> RlsObject:
    return next(policy for policy in table_policies(table, schema=SCHEMA, role=ROLE) if policy.name == f'{table}_{command}')


def _with_object(state: RlsState, obj: RlsObject, installed_object: InstalledObject | None) -> RlsState:
    """Return ``state`` with ``obj`` replaced by ``installed_object``, or dropped when it is None."""
    objects = dict(state.objects)
    if installed_object is None:
        del objects[obj.key]
    else:
        objects[obj.key] = installed_object
    return dataclasses.replace(state, objects=objects)


def _with_view(state: RlsState, view: str, **changes: Any) -> RlsState:
    """Return ``state`` with what the catalog holds about the read view ``view`` changed."""
    return dataclasses.replace(state, views={**state.views, view: dataclasses.replace(state.views[view], **changes)})


def _drifted(obj: RlsObject, *, preenabled: bool | None = None) -> InstalledObject:
    """Return ``obj`` as the catalog reads it after an out-of-band edit kept its comment."""
    return dataclasses.replace(installed(obj, preenabled=preenabled), definition=CHANGED_OUT_OF_BAND)


class TestPlanOfAnEmptyDatabase:
    """A database without the objects needs every one of them, each read policy recording the table's prior state."""

    def test_every_object_is_created_and_each_table_enabled_where_it_is_off(self) -> None:
        tables = {'context_entries': False, 'context_entry_grants': False, 'tags': True}
        state = RlsState(tables=tables, objects={}, columns=table_columns(tables), views={}, helpers_executable=False)

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.helpers == helper_functions(SCHEMA)
        assert plan.grant_helpers
        assert plan.guard_function == guard_function(SCHEMA)
        assert plan.guard_trigger == guard_trigger(SCHEMA)
        assert plan.tables == tuple(
            TableRepair(
                table=table,
                policies=tuple(
                    (policy, enabled if policy.name == f'{table}_read' else None)
                    for policy in table_policies(table, schema=SCHEMA, role=ROLE)
                ),
                enable=not enabled,
            )
            for table, enabled in state.tables.items()
        )
        assert plan.views == (rendered_view(), rendered_tags_view())
        assert plan.view_acls == ()
        assert plan.drifted == ()
        assert not plan.clean

    def test_without_the_context_table_there_is_no_guard_trigger_and_no_view(self) -> None:
        tables = {'tags': False}
        state = RlsState(tables=tables, objects={}, columns=table_columns(tables), views={}, helpers_executable=False)

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.guard_trigger is None
        assert plan.views == ()
        assert [repair.table for repair in plan.tables] == ['tags']

    def test_without_the_tags_table_only_the_read_view_of_context_entries_is_created(self) -> None:
        tables = {'context_entries': False}
        state = RlsState(tables=tables, objects={}, columns=table_columns(tables), views={}, helpers_executable=False)

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.views == (rendered_view(),)


class TestPlanOfACurrentDatabase:
    """A database holding exactly what the code renders needs nothing."""

    def test_nothing_differs(self) -> None:
        plan = plan_rls(installed_state(TABLES), schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.clean
        assert plan.helpers == ()
        assert not plan.grant_helpers
        assert plan.guard_function is None
        assert plan.guard_trigger is None
        assert plan.tables == ()
        assert plan.views == ()
        assert plan.view_acls == ()
        assert plan.drifted == ()

    def test_a_table_created_since_the_last_boot_gets_its_policies_and_nothing_else_changes(self) -> None:
        state = installed_state(TABLES)
        state = dataclasses.replace(state, tables={**state.tables, 'context_index_nodes': False})

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        policies = table_policies('context_index_nodes', schema=SCHEMA, role=ROLE)
        assert plan.tables == (
            TableRepair(
                table='context_index_nodes',
                policies=tuple((policy, False if policy.name.endswith('_read') else None) for policy in policies),
                enable=True,
            ),
        )
        assert plan.helpers == ()
        assert plan.guard_trigger is None
        assert plan.views == ()
        assert plan.drifted == ()

    def test_a_missing_tags_view_is_created_alone(self) -> None:
        tags_view = rendered_tags_view()
        state = installed_state(TABLES)
        objects = {
            key: value for key, value in state.objects.items()
            if key not in (tags_view.key, reject_trigger(SCHEMA, view=TAGS_READ_VIEW).key)
        }
        views = {name: view for name, view in state.views.items() if name != TAGS_READ_VIEW}
        state = dataclasses.replace(state, objects=objects, views=views)

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.views == (tags_view,)
        assert plan.view_acls == ()
        assert plan.tables == ()
        assert plan.drifted == ()

    def test_row_level_security_switched_off_on_a_table_is_enabled_again_without_touching_its_policies(self) -> None:
        state = installed_state(TABLES)
        state = dataclasses.replace(state, tables={**state.tables, 'tags': False})

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.tables == (TableRepair(table='tags', policies=(), enable=True),)
        assert not plan.clean

    def test_a_policy_role_without_execute_on_the_helpers_is_granted_it_alone(self) -> None:
        state = dataclasses.replace(installed_state(TABLES), helpers_executable=False)

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.grant_helpers
        assert plan.helpers == ()
        assert plan.tables == ()
        assert not plan.clean


class TestPlanOfARenderingThatChanged:
    """An object whose comment names another rendered statement is re-created, with no warning."""

    def test_a_new_policy_role_re_creates_every_policy_and_the_view_but_not_the_helpers(self) -> None:
        state = dataclasses.replace(installed_state(TABLES), helpers_executable=False)

        plan = plan_rls(state, schema=SCHEMA, role_name='mcp_other_request')

        other = '"mcp_other_request"'
        assert plan.helpers == ()
        assert plan.grant_helpers
        assert plan.guard_function is None
        assert plan.guard_trigger is None
        assert [policy for repair in plan.tables for policy, _ in repair.policies] == [
            policy for table in TABLES for policy in table_policies(table, schema=SCHEMA, role=other)
        ]
        assert all(not repair.enable for repair in plan.tables)
        assert plan.views == (rendered_view(role_name='mcp_other_request'), rendered_tags_view())
        assert plan.drifted == ()

    def test_a_column_added_to_the_context_table_re_creates_both_views_and_nothing_else(self) -> None:
        state = installed_state(TABLES)
        state = dataclasses.replace(
            state, columns={**state.columns, 'context_entries': (*state.columns['context_entries'], '"extra"')},
        )

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        view, tags_view = plan.views
        assert view.statement.startswith(f'CREATE VIEW {SCHEMA}.context_entries_readable AS SELECT ce."id"')
        assert 'ce."extra" FROM' in view.statement
        assert tags_view == rendered_tags_view()
        assert plan.tables == ()
        assert plan.drifted == ()

    def test_a_column_added_to_tags_re_creates_the_tags_view_alone(self) -> None:
        state = installed_state(TABLES)
        state = dataclasses.replace(state, columns={**state.columns, 'tags': (*QUOTED_TAG_COLUMNS, '"extra"')})

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        [tags_view] = plan.views
        assert tags_view.name == TAGS_READ_VIEW
        assert 't."extra" FROM' in tags_view.statement
        assert plan.tables == ()
        assert plan.drifted == ()

    def test_a_write_rejection_function_rendered_in_an_earlier_form_re_creates_both_views_silently(self) -> None:
        function = reject_function(SCHEMA)
        earlier = installed(function)
        assert earlier.fingerprint is not None
        stale = dataclasses.replace(earlier, fingerprint=dataclasses.replace(earlier.fingerprint, sha=EARLIER_STATEMENT))

        plan = plan_rls(_with_object(installed_state(TABLES), function, stale), schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.views == (rendered_view(), rendered_tags_view())
        assert plan.drifted == ()
        assert plan.tables == ()

    def test_a_re_created_read_policy_keeps_the_recorded_preenabled_value(self) -> None:
        read = _policy('tags', 'read')
        stale = dataclasses.replace(
            installed(read, preenabled=True),
            fingerprint=Fingerprint(sha='0' * 64, definition=installed(read).definition, preenabled=True),
        )
        state = _with_object(installed_state(TABLES, preenabled={'tags': True}), read, stale)

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.tables == (TableRepair(table='tags', policies=((read, True),), enable=False),)
        assert plan.drifted == ()


class TestPlanOfAnOutOfBandChange:
    """An object changed outside the server is re-created and named, whatever made it differ."""

    def test_an_altered_policy_is_re_created_and_named(self) -> None:
        insert = _policy('tags', 'insert')

        plan = plan_rls(_with_object(installed_state(TABLES), insert, _drifted(insert)), schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.tables == (TableRepair(table='tags', policies=((insert, None),), enable=False),)
        assert plan.drifted == (insert.target,)

    def test_a_policy_without_its_comment_is_re_created_and_named(self) -> None:
        delete = _policy('context_entries', 'delete')
        uncommented = dataclasses.replace(installed(delete), fingerprint=None)

        plan = plan_rls(_with_object(installed_state(TABLES), delete, uncommented), schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.tables == (TableRepair(table='context_entries', policies=((delete, None),), enable=False),)
        assert plan.drifted == (delete.target,)

    @pytest.mark.parametrize('enabled', [True, False])
    def test_a_read_policy_whose_comment_lost_preenabled_falls_back_to_the_tables_current_state(self, enabled: bool) -> None:
        read = _policy('tags', 'read')
        bare = dataclasses.replace(
            installed(read), fingerprint=Fingerprint(sha=read.sha, definition=installed(read).definition, preenabled=None),
        )
        state = _with_object(installed_state(TABLES), read, bare)
        state = dataclasses.replace(state, tables={**state.tables, 'tags': enabled})

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.tables == (TableRepair(table='tags', policies=((read, enabled),), enable=not enabled),)
        assert plan.drifted == (read.target,)

    def test_a_dropped_read_policy_on_an_enabled_table_records_preenabled_true(self) -> None:
        read = _policy('context_entry_grants', 'read')

        plan = plan_rls(_with_object(installed_state(TABLES), read, None), schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.tables == (TableRepair(table='context_entry_grants', policies=((read, True),), enable=False),)
        assert plan.drifted == ()

    def test_a_replaced_helper_and_a_disabled_guard_trigger_are_re_created(self) -> None:
        helper = helper_functions(SCHEMA)[1]
        trigger = guard_trigger(SCHEMA)
        state = _with_object(installed_state(TABLES), helper, _drifted(helper))
        state = _with_object(state, trigger, dataclasses.replace(installed(trigger), enabled=False))

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.helpers == (helper,)
        assert not plan.grant_helpers
        assert plan.guard_trigger == trigger
        assert plan.guard_function is None
        assert plan.drifted == (helper.target, trigger.target)

    @pytest.mark.parametrize(
        'change',
        [
            'view definition', 'view owner', 'view options',
            'reject function', 'reject trigger disabled', 'reject trigger missing',
        ],
    )
    def test_any_change_to_the_view_or_its_write_rejection_re_creates_it_with_the_tags_view(self, change: str) -> None:
        state = installed_state(TABLES)
        installed_view = state.views[READ_VIEW]
        view, function, trigger = rendered_view(), reject_function(SCHEMA), reject_trigger(SCHEMA, view=READ_VIEW)
        expected_drift: tuple[str, ...]
        if change == 'view definition':
            state, expected_drift = _with_object(state, view, _drifted(view)), (view.target,)
        elif change == 'view owner':
            state = _with_view(state, READ_VIEW, owned_by_table_owner=False)
            expected_drift = (view.target,)
        elif change == 'view options':
            state = _with_view(state, READ_VIEW, plain=False)
            expected_drift = (view.target,)
        elif change == 'reject function':
            state, expected_drift = _with_object(state, function, _drifted(function)), (function.target,)
        elif change == 'reject trigger disabled':
            state = _with_object(state, trigger, dataclasses.replace(installed(trigger), enabled=False))
            expected_drift = (trigger.target,)
        else:
            state, expected_drift = _with_object(state, trigger, None), ()

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        assert installed_view == InstalledView(owned_by_table_owner=True, plain=True, acl={ROLE_NAME: ('SELECT',)})
        assert plan.views == (view, rendered_tags_view())
        assert plan.view_acls == ()
        assert plan.drifted == expected_drift
        assert plan.tables == ()

    @pytest.mark.parametrize(
        'change', ['view definition', 'view owner', 'view options', 'reject trigger disabled', 'reject trigger missing'],
    )
    def test_any_change_to_the_tags_view_or_its_trigger_re_creates_the_tags_view_alone(self, change: str) -> None:
        state = installed_state(TABLES)
        view, trigger = rendered_tags_view(), reject_trigger(SCHEMA, view=TAGS_READ_VIEW)
        expected_drift: tuple[str, ...]
        if change == 'view definition':
            state, expected_drift = _with_object(state, view, _drifted(view)), (view.target,)
        elif change == 'view owner':
            state, expected_drift = _with_view(state, TAGS_READ_VIEW, owned_by_table_owner=False), (view.target,)
        elif change == 'view options':
            state, expected_drift = _with_view(state, TAGS_READ_VIEW, plain=False), (view.target,)
        elif change == 'reject trigger disabled':
            state = _with_object(state, trigger, dataclasses.replace(installed(trigger), enabled=False))
            expected_drift = (trigger.target,)
        else:
            state, expected_drift = _with_object(state, trigger, None), ()

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.views == (view,)
        assert plan.view_acls == ()
        assert plan.drifted == expected_drift
        assert plan.tables == ()

    @pytest.mark.parametrize(
        'acl',
        [
            {ROLE_NAME: ('SELECT',), 'mcp_outsider': ('SELECT',)},
            {ROLE_NAME: ('SELECT',), None: ('SELECT',)},
            {ROLE_NAME: ('INSERT', 'SELECT')},
            {},
        ],
        ids=['extra grantee', 'public', 'extra privilege', 'no select'],
    )
    @pytest.mark.parametrize('view', [READ_VIEW, TAGS_READ_VIEW])
    def test_any_acl_but_the_policy_roles_select_repairs_the_acl_of_that_unchanged_view(
        self, view: str, acl: dict[str | None, tuple[str, ...]],
    ) -> None:
        state = _with_view(installed_state(TABLES), view, acl=acl)

        plan = plan_rls(state, schema=SCHEMA, role_name=ROLE_NAME)

        assert plan.views == ()
        assert plan.view_acls == (view,)
        assert not plan.clean


def _serve(monkeypatch: pytest.MonkeyPatch, *states: RlsState) -> list[RlsState]:
    """Answer the catalog reads of apply_rls with ``states`` in turn; return what is left unread."""
    remaining = list(states)

    async def _read(_conn: object, *, schema_name: str, role: str | None) -> RlsState:
        assert (schema_name, role) == ('tenant one', ROLE)
        return remaining.pop(0)

    monkeypatch.setattr(rls_module, 'read_rls_state', _read)
    return remaining


class TestDoubleCheckedLock:
    """The lock is taken only when the unlocked read finds a difference, and the locked read decides what changes."""

    @pytest.mark.asyncio
    async def test_a_current_database_is_read_once_without_the_lock_or_any_statement(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        recorded = record_ddl(monkeypatch, [rls_module])
        remaining = _serve(monkeypatch, installed_state(TABLES), installed_state(TABLES))

        with caplog.at_level(logging.INFO, logger='app.migrations.rls'):
            await rls_module.apply_rls(cast(StorageBackend, FakeBackend(CatalogConnection())), rls_settings())

        assert recorded == []
        assert len(remaining) == 1
        assert [record for record in caplog.records if record.levelno >= logging.INFO] == []

    @pytest.mark.asyncio
    async def test_a_difference_already_repaired_by_another_instance_takes_the_lock_and_changes_nothing(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recorded = record_ddl(monkeypatch, [rls_module])
        stale = _with_object(installed_state(TABLES), _policy('tags', 'insert'), None)
        remaining = _serve(monkeypatch, stale, installed_state(TABLES))

        await rls_module.apply_rls(cast(StorageBackend, FakeBackend(CatalogConnection())), rls_settings())

        assert recorded == [LOCK]
        assert remaining == []

    @pytest.mark.asyncio
    async def test_only_what_still_differs_under_the_lock_is_changed_and_drift_is_named(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        recorded = record_ddl(monkeypatch, [rls_module])
        insert, read = _policy('tags', 'insert'), _policy('tags', 'read')
        unlocked = _with_object(installed_state(TABLES), insert, _drifted(insert))
        locked = _with_object(unlocked, read, _drifted(read, preenabled=True))
        _serve(monkeypatch, unlocked, locked)
        conn = CatalogConnection()

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await rls_module.apply_rls(cast(StorageBackend, FakeBackend(conn)), rls_settings())

        assert recorded == [
            LOCK,
            read.drop_statement, read.statement, await fingerprinted(conn, read, preenabled=True),
            insert.drop_statement, insert.statement, await fingerprinted(conn, insert),
        ]
        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 2
        assert any(read.target in message and 'outside the server' in message for message in messages)
        assert any(insert.target in message and 'outside the server' in message for message in messages)

    @pytest.mark.asyncio
    async def test_a_table_created_since_the_last_boot_is_the_only_change(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = record_ddl(monkeypatch, [rls_module])
        state = installed_state(TABLES)
        state = dataclasses.replace(state, tables={**state.tables, 'context_index_nodes': False})
        _serve(monkeypatch, state, state)
        conn = CatalogConnection()

        await rls_module.apply_rls(cast(StorageBackend, FakeBackend(conn)), rls_settings())

        expected = [LOCK]
        for policy in table_policies('context_index_nodes', schema=SCHEMA, role=ROLE):
            preenabled = False if policy.name.endswith('_read') else None
            comment = await fingerprinted(conn, policy, preenabled=preenabled)
            expected.extend([policy.drop_statement, policy.statement, comment])
        expected.append(enable_statement('context_index_nodes', schema=SCHEMA))
        assert recorded == expected

    @pytest.mark.asyncio
    async def test_a_missing_helper_grant_is_issued_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = record_ddl(monkeypatch, [rls_module])
        state = dataclasses.replace(installed_state(TABLES), helpers_executable=False)
        _serve(monkeypatch, state, state)

        await rls_module.apply_rls(cast(StorageBackend, FakeBackend(CatalogConnection())), rls_settings())

        assert recorded == [LOCK, helper_grant(SCHEMA, ROLE)]

    @pytest.mark.asyncio
    async def test_helpers_rendered_in_an_earlier_form_are_replaced_under_the_pinned_search_path_silently(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        recorded = record_ddl(monkeypatch, [rls_module])
        helpers = helper_functions(SCHEMA)
        state = installed_state(TABLES)
        for helper in helpers:
            earlier = installed(helper)
            assert earlier.fingerprint is not None
            fingerprint = dataclasses.replace(earlier.fingerprint, sha=EARLIER_STATEMENT)
            state = _with_object(state, helper, dataclasses.replace(earlier, fingerprint=fingerprint))
        _serve(monkeypatch, state, state)
        conn = CatalogConnection()

        with caplog.at_level(logging.INFO, logger='app.migrations.rls'):
            await rls_module.apply_rls(cast(StorageBackend, FakeBackend(conn)), rls_settings())

        assert recorded == [
            LOCK,
            PIN_SEARCH_PATH,
            *(helper.statement for helper in helpers),
            restore_search_path(SEARCH_PATH),
            *[await fingerprinted(conn, helper) for helper in helpers],
        ]
        assert [record.levelno for record in caplog.records] == [logging.INFO]


class TestViewRepair:
    """A view that differs is re-created naming the grants its drop removes; an unchanged view only has its ACL repaired."""

    @pytest.mark.asyncio
    async def test_a_re_created_view_names_the_grantees_its_drop_removes(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        recorded = record_ddl(monkeypatch, [rls_module])
        state = installed_state(TABLES)
        state = _with_view(state, READ_VIEW, acl={ROLE_NAME: ('SELECT',), 'mcp_outsider': ('INSERT', 'SELECT')})
        state = dataclasses.replace(
            state, columns={**state.columns, 'context_entries': (*state.columns['context_entries'], '"extra"')},
        )
        _serve(monkeypatch, state, state)
        acl: list[tuple[str | None, str]] = [(ROLE_NAME, 'SELECT'), ('mcp_outsider', 'INSERT'), ('mcp_outsider', 'SELECT')]
        conn = CatalogConnection(acl=acl, recreated_acl=[])

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await rls_module.apply_rls(cast(StorageBackend, FakeBackend(conn)), rls_settings())

        assert recorded[:3] == [
            LOCK, read_view_drop(SCHEMA, view=TAGS_READ_VIEW), read_view_drop(SCHEMA, view=READ_VIEW),
        ]
        assert recorded[3].startswith(f'CREATE VIEW {SCHEMA}.context_entries_readable')
        assert rendered_tags_view().statement in recorded
        [warning] = caplog.records
        assert 'mcp_outsider (INSERT, SELECT)' in warning.getMessage()
        assert ROLE_NAME not in warning.getMessage().split(':', 1)[1]

    @pytest.mark.asyncio
    async def test_an_extra_grantee_on_the_unchanged_view_is_revoked_with_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        recorded = record_ddl(monkeypatch, [rls_module])
        acl: dict[str | None, tuple[str, ...]] = {ROLE_NAME: ('SELECT',), 'mcp_outsider': ('SELECT',)}
        state = _with_view(installed_state(TABLES), READ_VIEW, acl=acl)
        _serve(monkeypatch, state, state)
        conn = CatalogConnection(acl=[(ROLE_NAME, 'SELECT'), ('mcp_outsider', 'SELECT')])

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await rls_module.apply_rls(cast(StorageBackend, FakeBackend(conn)), rls_settings())

        assert recorded == [LOCK, read_view_revoke(SCHEMA, '"mcp_outsider"', ('ALL',), view=READ_VIEW)]
        [warning] = caplog.records
        assert 'mcp_outsider (SELECT)' in warning.getMessage()
        assert TAGS_VIEW not in warning.getMessage()

    @pytest.mark.asyncio
    async def test_a_stray_grantee_on_the_unchanged_tags_view_is_revoked_with_a_warning_naming_it(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        recorded = record_ddl(monkeypatch, [rls_module])
        acl: dict[str | None, tuple[str, ...]] = {ROLE_NAME: ('SELECT',), 'mcp_outsider': ('SELECT',)}
        state = _with_view(installed_state(TABLES), TAGS_READ_VIEW, acl=acl)
        _serve(monkeypatch, state, state)
        conn = CatalogConnection(tags_acl=[(ROLE_NAME, 'SELECT'), ('mcp_outsider', 'SELECT')])

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await rls_module.apply_rls(cast(StorageBackend, FakeBackend(conn)), rls_settings())

        assert recorded == [LOCK, read_view_revoke(SCHEMA, '"mcp_outsider"', ('ALL',), view=TAGS_READ_VIEW)]
        [warning] = caplog.records
        assert TAGS_VIEW in warning.getMessage()
        assert 'mcp_outsider (SELECT)' in warning.getMessage()

    @pytest.mark.asyncio
    async def test_a_tags_view_changed_outside_the_server_is_re_created_alone_and_named(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        recorded = record_ddl(monkeypatch, [rls_module])
        view, trigger = rendered_tags_view(), reject_trigger(SCHEMA, view=TAGS_READ_VIEW)
        state = _with_object(installed_state(TABLES), view, _drifted(view))
        _serve(monkeypatch, state, state)
        conn = CatalogConnection()

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await rls_module.apply_rls(cast(StorageBackend, FakeBackend(conn)), rls_settings())

        assert recorded == [
            LOCK,
            read_view_drop(SCHEMA, view=TAGS_READ_VIEW),
            view.statement,
            read_view_grant(SCHEMA, ROLE, view=TAGS_READ_VIEW),
            trigger.statement,
            await fingerprinted(conn, trigger),
            await fingerprinted(conn, view),
        ]
        [warning] = caplog.records
        assert view.target in warning.getMessage()
        assert 'outside the server' in warning.getMessage()
