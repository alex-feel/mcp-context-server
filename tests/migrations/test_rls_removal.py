"""Tests for removing the PostgreSQL row-level security objects and for refusing to start over them with the flag off.

Covers app.migrations.rls_removal: ``remove_rls`` drops, under the schema-init lock and
in one transaction, the read view and its write rejection, every policy of ours, the
guard trigger and the helpers, and disables row-level security only on the tables whose
read policy recorded that it was off before the migration enabled it; every table left
enabled is reported with the reason, and a dry run reports the same without changing
anything. ``guard_rls_disable_over_enabled`` refuses a PostgreSQL start with
``ENABLE_POSTGRESQL_RLS`` off while any policy of ours or the read view exists.
"""

import dataclasses
from typing import cast

import pytest

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.errors import ConfigurationError
from app.migrations import rls_removal as removal_module
from app.migrations.rls_ddl import CONTEXT_TABLE
from app.migrations.rls_ddl import Fingerprint
from app.migrations.rls_ddl import ObjectKey
from app.migrations.rls_ddl import RlsObjectKind
from app.migrations.rls_ddl import disable_statement
from app.migrations.rls_ddl import guard_function
from app.migrations.rls_ddl import guard_trigger
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import policy_names
from app.migrations.rls_ddl import read_view_drop
from app.migrations.rls_ddl import reject_function
from app.migrations.rls_removal import RlsRemovalReport
from app.migrations.rls_removal import guard_rls_disable_over_enabled
from app.migrations.rls_removal import plan_removal
from app.migrations.rls_removal import remove_rls
from app.migrations.rls_state import RlsState
from tests.migrations._rls_fakes import LOCK
from tests.migrations._rls_fakes import SCHEMA
from tests.migrations._rls_fakes import SCHEMA_NAME
from tests.migrations._rls_fakes import FakeBackend
from tests.migrations._rls_fakes import installed_state
from tests.migrations._rls_fakes import record_ddl
from tests.migrations._rls_fakes import rls_settings

TABLES = {'context_entries': True, 'context_entry_grants': True, 'tags': True}


def _policy_drops(table: str) -> list[str]:
    return [ObjectKey(RlsObjectKind.POLICY, name, f'{SCHEMA}.{table}').drop_statement for name in policy_names(table)]


def _installed() -> RlsState:
    """Return a database the migration enabled on every table but tags, which was enabled before it."""
    return installed_state(TABLES, preenabled={'tags': True})


def _full_removal() -> RlsRemovalReport:
    statements = [read_view_drop(SCHEMA), reject_function(SCHEMA).drop_statement]
    statements.extend([*_policy_drops('context_entries'), disable_statement('context_entries', schema=SCHEMA)])
    statements.extend([*_policy_drops('context_entry_grants'), disable_statement('context_entry_grants', schema=SCHEMA)])
    statements.extend(_policy_drops('tags'))
    statements.extend([guard_trigger(SCHEMA).drop_statement, guard_function(SCHEMA).drop_statement])
    statements.extend(helper.drop_statement for helper in helper_functions(SCHEMA))
    return RlsRemovalReport(
        statements=statements,
        disabled=['context_entries', 'context_entry_grants'],
        left_enabled={'tags': 'preenabled=true'},
    )


class TestPlanRemoval:
    """What removal drops follows the catalog, and only a table enabled by the migration is disabled again."""

    def test_everything_installed_is_dropped_in_reverse_dependency_order(self) -> None:
        assert plan_removal(_installed(), schema=SCHEMA) == _full_removal()

    def test_a_database_without_the_objects_needs_nothing(self) -> None:
        tables = {'context_entries': False, 'tags': False}
        state = RlsState(tables=tables, objects={}, columns=(), view=None, helpers_executable=False)

        assert plan_removal(state, schema=SCHEMA) == RlsRemovalReport(statements=[], disabled=[], left_enabled={})

    def test_a_table_whose_read_policy_says_nothing_readable_is_left_enabled(self) -> None:
        state = _installed()
        objects = dict(state.objects)
        grants_read = ObjectKey(RlsObjectKind.POLICY, 'context_entry_grants_read', f'{SCHEMA}.context_entry_grants')
        context_read = ObjectKey(RlsObjectKind.POLICY, 'context_entries_read', f'{SCHEMA}.context_entries')
        objects[grants_read] = dataclasses.replace(objects[grants_read], fingerprint=None)
        objects[context_read] = dataclasses.replace(
            objects[context_read],
            fingerprint=Fingerprint(sha='a' * 64, definition='b' * 64, preenabled=None),
        )
        state = dataclasses.replace(state, objects=objects)

        report = plan_removal(state, schema=SCHEMA)

        assert report.disabled == []
        assert report.left_enabled == {
            'context_entries': 'no readable comment',
            'context_entry_grants': 'no readable comment',
            'tags': 'preenabled=true',
        }
        assert disable_statement('context_entries', schema=SCHEMA) not in report.statements

    def test_an_enabled_table_without_any_policy_of_ours_is_reported_and_a_disabled_one_is_not(self) -> None:
        state = dataclasses.replace(_installed(), tables={**TABLES, 'image_attachments': True, 'context_index_nodes': False})

        report = plan_removal(state, schema=SCHEMA)

        assert report.left_enabled == {'tags': 'preenabled=true', 'image_attachments': 'no readable comment'}
        untouched = ('image_attachments', 'context_index_nodes')
        assert not any(table in statement for table in untouched for statement in report.statements)

    def test_a_table_switched_off_out_of_band_has_its_policies_dropped_and_is_not_reported(self) -> None:
        state = dataclasses.replace(_installed(), tables={**TABLES, 'context_entry_grants': False})

        report = plan_removal(state, schema=SCHEMA)

        assert report.disabled == ['context_entries']
        assert 'context_entry_grants' not in report.left_enabled
        assert all(statement in report.statements for statement in _policy_drops('context_entry_grants'))


def _serve(monkeypatch: pytest.MonkeyPatch, state: RlsState) -> list[str | None]:
    """Answer the catalog read of remove_rls with ``state``; return the role each read named."""
    roles: list[str | None] = []

    async def _read(_conn: object, *, schema_name: str, role: str | None) -> RlsState:
        assert schema_name == SCHEMA_NAME
        roles.append(role)
        return state

    monkeypatch.setattr(removal_module, 'read_rls_state', _read)
    return roles


class TestRemoveRls:
    """Removal reads the catalog under the lock and either reports what it would drop or drops it."""

    @pytest.mark.asyncio
    async def test_a_dry_run_reports_everything_and_changes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = record_ddl(monkeypatch, [removal_module])
        roles = _serve(monkeypatch, _installed())
        backend = FakeBackend(object())

        report = await remove_rls(cast(StorageBackend, backend), rls_settings(), dry_run=True)

        assert report == _full_removal()
        assert recorded == [LOCK]
        assert roles == [None]
        assert backend.scopes == [MAINTENANCE]

    @pytest.mark.asyncio
    async def test_a_real_run_issues_what_the_dry_run_reports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = record_ddl(monkeypatch, [removal_module])
        _serve(monkeypatch, _installed())

        report = await remove_rls(cast(StorageBackend, FakeBackend(object())), rls_settings(), dry_run=False)

        assert report == _full_removal()
        assert recorded == [LOCK, *_full_removal().statements]

    @pytest.mark.asyncio
    async def test_removal_does_not_depend_on_the_configured_policy_role(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record_ddl(monkeypatch, [removal_module])
        _serve(monkeypatch, _installed())

        report = await remove_rls(
            cast(StorageBackend, FakeBackend(object())), rls_settings(policy_role='some_other_role'), dry_run=False,
        )

        assert report == _full_removal()

    @pytest.mark.asyncio
    async def test_sqlite_has_nothing_to_remove(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = record_ddl(monkeypatch, [removal_module])
        backend = FakeBackend(object(), backend_type='sqlite')

        report = await remove_rls(cast(StorageBackend, backend), rls_settings(), dry_run=False)

        assert report == RlsRemovalReport(statements=[], disabled=[], left_enabled={})
        assert recorded == []
        assert backend.scopes == []

    @pytest.mark.asyncio
    async def test_a_failure_surfaces_as_the_removal_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _begin(_conn: object, _timeout_s: float) -> None:
            raise OSError('connection reset')

        monkeypatch.setattr(removal_module, 'begin_migration', _begin)

        with pytest.raises(RuntimeError, match='Row-level security removal failed: connection reset'):
            await remove_rls(cast(StorageBackend, FakeBackend(object())), rls_settings(), dry_run=False)


class _GuardConnection:
    """Answers the guard's catalog reads: the policies of ours that exist and whether the view exists."""

    def __init__(self, policies: list[tuple[str, str]], *, view: bool) -> None:
        self.policies = policies
        self.view = view
        self.reads: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, sql: str, *args: object) -> list[tuple[str, str]]:
        self.reads.append((sql, args))
        assert 'pg_catalog.pg_policy' in sql
        assert args[0] == SCHEMA_NAME
        return list(self.policies)

    async def fetchval(self, sql: str, *args: object) -> bool:
        self.reads.append((sql, args))
        assert args == (SCHEMA_NAME, 'context_entries_readable')
        return self.view


class TestGuard:
    """Starting with the flag off is refused while the database carries what the migration installed."""

    @pytest.mark.asyncio
    async def test_the_flag_on_is_never_refused(self) -> None:
        conn = _GuardConnection([('context_entries', 'context_entries_read')], view=True)
        backend = FakeBackend(conn)

        await guard_rls_disable_over_enabled(cast(StorageBackend, backend), rls_settings(enabled=True))

        assert conn.reads == []
        assert backend.scopes == []

    @pytest.mark.asyncio
    async def test_sqlite_is_never_read(self) -> None:
        backend = FakeBackend(_GuardConnection([], view=False), backend_type='sqlite')

        await guard_rls_disable_over_enabled(cast(StorageBackend, backend), rls_settings(enabled=False))

        assert backend.scopes == []

    @pytest.mark.asyncio
    async def test_a_database_without_the_objects_starts(self) -> None:
        conn = _GuardConnection([('tags', 'tags_owner_only')], view=False)
        backend = FakeBackend(conn)

        await guard_rls_disable_over_enabled(cast(StorageBackend, backend), rls_settings(enabled=False))

        assert backend.scopes == [MAINTENANCE]

    @pytest.mark.asyncio
    async def test_policies_or_the_view_refuse_the_start_naming_where_they_are(self) -> None:
        conn = _GuardConnection(
            [('tags', 'tags_delete'), (CONTEXT_TABLE, 'context_entries_read'), ('tags', 'tags_read'), ('tags', 'mine')],
            view=True,
        )

        with pytest.raises(ConfigurationError) as raised:
            await guard_rls_disable_over_enabled(cast(StorageBackend, FakeBackend(conn)), rls_settings(enabled=False))

        assert str(raised.value) == (
            'ENABLE_POSTGRESQL_RLS=false, but this database carries the row-level security policies and read view '
            'this server installed (tables: context_entries, tags, context_entries_readable). Starting with the flag '
            'off would leave a non-owner request role reading only public rows. Either set ENABLE_POSTGRESQL_RLS=true '
            'again, or remove them first: stop every server instance, run `mcp-context-server-migrate --disable-rls '
            '--source-url <table-owner connection>`, then start with the flag off (procedure per deployment: '
            "docs/database-backends.md, 'Disabling row-level security')."
        )

    @pytest.mark.asyncio
    async def test_the_view_alone_refuses_the_start(self) -> None:
        conn = _GuardConnection([], view=True)

        with pytest.raises(ConfigurationError, match=r'\(tables: context_entries_readable\)'):
            await guard_rls_disable_over_enabled(cast(StorageBackend, FakeBackend(conn)), rls_settings(enabled=False))
