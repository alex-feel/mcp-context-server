"""Tests for the PostgreSQL role checks of the server start in app.startup.rls_posture.

Covers the discovery of the role the request connection settings log in as, the refusals
that run on it before any DDL (a superuser or BYPASSRLS request role, a missing policy
role, a request role outside the policy role), the runtime refusal of request and
migration credentials that resolve to one role, the posture checks on the request
backend (membership and privileges whenever the roles are split, row-level security
enabled outside the server with the flag off, and with the flag on: enabled and active on
every managed table, never forced, no foreign policy that applies to the request role,
every object of ours current, and the read view owned by the table owner, plain, write
protected, readable by the policy role alone) and the warning for the jwt provider on
PostgreSQL without row-level security.
"""

import dataclasses
import logging
from collections.abc import Awaitable
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import asyncpg
import pytest
from pydantic import SecretStr

import app.startup.rls_posture as posture_module
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.errors import ConfigurationError
from app.errors import DependencyError
from app.migrations.request_role_grants import TABLE_PRIVILEGES
from app.migrations.request_role_grants import HeldPrivileges
from app.migrations.rls_ddl import guard_trigger
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import reject_trigger
from app.migrations.rls_ddl import table_policies
from app.migrations.rls_state import InstalledView
from app.migrations.rls_state import RlsState
from app.settings import AppSettings
from app.startup.rls_posture import ApplyingPolicy
from app.startup.rls_posture import Posture
from app.startup.rls_posture import RequestRole
from app.startup.rls_posture import TableSecurity
from app.startup.rls_posture import discover_request_role
from app.startup.rls_posture import posture_problems
from app.startup.rls_posture import read_login_role
from app.startup.rls_posture import refuse_unsafe_request_role
from app.startup.rls_posture import split_request_role
from app.startup.rls_posture import verify_rls_posture
from app.startup.rls_posture import warn_jwt_without_rls
from tests.migrations._rls_fakes import ROLE
from tests.migrations._rls_fakes import ROLE_NAME
from tests.migrations._rls_fakes import SCHEMA
from tests.migrations._rls_fakes import installed_state
from tests.migrations._rls_fakes import rls_settings

DOCS = "docs/database-backends.md, 'Row-level security (PostgreSQL)'"
REQUEST_DSN = 'postgresql://mcp_app:secret@db.example:5432/mcp'
TABLES = ('context_entries', 'context_entry_grants', 'tags')
APP = RequestRole(name='mcp_app', is_superuser=False, bypass_rls=False, policy_role_exists=True, policy_member=True)


def _settings(*, rls: bool = True, migration_user: str | None = 'mcp_owner') -> AppSettings:
    """Return settings carrying what the role checks read."""
    return cast(
        AppSettings,
        SimpleNamespace(
            access_control=SimpleNamespace(postgresql_rls_enabled=rls, postgresql_rls_policy_role=ROLE_NAME),
            storage=SimpleNamespace(
                backend_type='postgresql',
                postgresql_schema='tenant one',
                postgresql_migration_user=migration_user,
                postgresql_connection_string=SecretStr(REQUEST_DSN),
                postgresql_connect_timeout_s=7.0,
                postgresql_statement_cache_size=0,
            ),
        ),
    )


def _all_held(tables: tuple[str, ...] = TABLES) -> HeldPrivileges:
    """Return every privilege the request role needs on ``tables``."""
    return HeldPrivileges(
        schema_usage=True,
        tables={**dict.fromkeys(tables, frozenset(TABLE_PRIVILEGES)), 'compression_metadata': frozenset({'SELECT'})},
        sequences={'"tenant one"."tags_id_seq"': frozenset({'USAGE', 'SELECT'})},
        merge_patch=True,
    )


def _posture(*, enabled: bool = True, **changes: Any) -> Posture:
    """Return the posture of a database the server prepared with ``ENABLE_POSTGRESQL_RLS`` at ``enabled``."""
    security = TableSecurity(enabled=enabled, forced=False, active=enabled, owned=False)
    posture = Posture(
        policy_member=True,
        held=_all_held(),
        tables=dict.fromkeys(TABLES, security),
        applying_policies=(),
        rls=installed_state(dict.fromkeys(TABLES, True)) if enabled else None,
    )
    return dataclasses.replace(posture, **changes)


def _problems(posture: Posture, settings: AppSettings) -> list[str]:
    """Return what the posture checks find wrong for the request role ``mcp_app``."""
    return posture_problems(posture, settings, request_role=APP.name)


def _single(problems: list[str]) -> str:
    assert len(problems) == 1, problems
    return problems[0]


class _Connection:
    """A one-off connection answering the role discovery with ``row``."""

    def __init__(self, row: tuple[object, ...]) -> None:
        self.row = row
        self.queries: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def fetchrow(self, sql: str, *args: object) -> tuple[object, ...]:
        self.queries.append((sql, args))
        return self.row

    async def close(self) -> None:
        self.closed = True


class TestDiscoverRequestRole:
    """The role the request connection settings log in as, read from the server itself."""

    @pytest.mark.asyncio
    async def test_without_a_migration_user_the_roles_are_one_by_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connect = AsyncMock()
        monkeypatch.setattr('app.startup.rls_posture.asyncpg.connect', connect)

        assert await discover_request_role(_settings(migration_user=None)) is None
        connect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reads_current_user_and_its_attributes_on_the_request_connection_settings(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        conn = _Connection(('mcp_app', False, True, True, False))
        connect = AsyncMock(return_value=conn)
        gucs = AsyncMock()
        monkeypatch.setattr('app.startup.rls_posture.asyncpg.connect', connect)
        monkeypatch.setattr(posture_module, 'apply_session_gucs', gucs)
        settings = _settings()

        role = await discover_request_role(settings)

        assert role == RequestRole(
            name='mcp_app', is_superuser=False, bypass_rls=True, policy_role_exists=True, policy_member=False,
        )
        connect.assert_awaited_once_with(REQUEST_DSN, timeout=7.0, statement_cache_size=0)
        gucs.assert_awaited_once_with(conn, settings, profile_statements=())
        [(sql, args)] = conn.queries
        assert args == (ROLE,)
        assert 'current_user' in sql
        assert "pg_catalog.pg_has_role(current_user, pg_catalog.to_regrole($1), 'USAGE')" in sql
        assert conn.closed

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'error',
        [
            asyncpg.exceptions.InvalidPasswordError('password authentication failed for user "mcp_app"'),
            asyncpg.exceptions.InvalidAuthorizationSpecificationError('no pg_hba.conf entry'),
            asyncpg.exceptions.InvalidCatalogNameError('database "mcp" does not exist'),
            asyncpg.exceptions.InsufficientPrivilegeError('permission denied for database "mcp"'),
            ValueError('invalid DSN'),
        ],
    )
    async def test_a_login_the_configuration_must_fix_exits_as_a_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception,
    ) -> None:
        monkeypatch.setattr('app.startup.rls_posture.asyncpg.connect', AsyncMock(side_effect=error))

        with pytest.raises(ConfigurationError, match='request connection settings'):
            await discover_request_role(_settings())

    @pytest.mark.asyncio
    async def test_an_unreachable_server_is_a_dependency_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        refused = AsyncMock(side_effect=ConnectionRefusedError('refused'))
        monkeypatch.setattr('app.startup.rls_posture.asyncpg.connect', refused)

        with pytest.raises(DependencyError, match='refused'):
            await discover_request_role(_settings())


class TestRefuseUnsafeRequestRole:
    """With row-level security on, a request role the policies could not hold is refused before any DDL."""

    def test_a_role_the_policies_hold_passes(self) -> None:
        refuse_unsafe_request_role(APP, _settings())

    def test_nothing_is_refused_with_the_flag_off(self) -> None:
        superuser = dataclasses.replace(APP, is_superuser=True, policy_member=False)

        refuse_unsafe_request_role(superuser, _settings(rls=False))
        refuse_unsafe_request_role(None, _settings(rls=False))

    @pytest.mark.parametrize(
        ('role', 'attribute'),
        [
            (dataclasses.replace(APP, is_superuser=True), 'rolsuper=true'),
            (dataclasses.replace(APP, bypass_rls=True), 'rolbypassrls=true'),
        ],
        ids=['superuser', 'bypassrls'],
    )
    def test_a_superuser_or_bypassrls_request_role_is_refused(self, role: RequestRole, attribute: str) -> None:
        with pytest.raises(ConfigurationError) as refusal:
            refuse_unsafe_request_role(role, _settings())

        message = str(refusal.value)
        assert "'mcp_app'" in message
        assert attribute in message
        assert 'NOSUPERUSER NOBYPASSRLS' in message
        assert DOCS in message

    def test_a_missing_policy_role_prints_the_statements_that_create_it(self) -> None:
        missing = dataclasses.replace(APP, policy_role_exists=False, policy_member=False)

        with pytest.raises(ConfigurationError) as refusal:
            refuse_unsafe_request_role(missing, _settings())

        message = str(refusal.value)
        assert f'CREATE ROLE {ROLE} NOLOGIN;' in message
        assert f'GRANT {ROLE} TO "mcp_app";' in message
        assert DOCS in message

    def test_a_request_role_outside_the_policy_role_prints_the_grant(self) -> None:
        with pytest.raises(ConfigurationError) as refusal:
            refuse_unsafe_request_role(dataclasses.replace(APP, policy_member=False), _settings())

        message = str(refusal.value)
        assert f'GRANT {ROLE} TO "mcp_app";' in message
        assert 'CREATE ROLE' not in message


class TestSplitRequestRole:
    """Whether requests and migrations log in as two roles is decided from the two current_user values."""

    def test_two_roles_are_split(self) -> None:
        assert split_request_role(APP, 'mcp_owner', _settings()) is APP

    @pytest.mark.parametrize('request_role', [None, dataclasses.replace(APP, name='mcp_owner')])
    def test_one_role_is_not_split_with_the_flag_off(self, request_role: RequestRole | None) -> None:
        assert split_request_role(request_role, 'mcp_owner', _settings(rls=False)) is None

    def test_one_role_with_the_flag_on_is_refused(self) -> None:
        same = dataclasses.replace(APP, name='mcp_owner')

        with pytest.raises(ConfigurationError) as refusal:
            split_request_role(same, 'mcp_owner', _settings(migration_user='mcp_alias'))

        assert str(refusal.value) == (
            "ENABLE_POSTGRESQL_RLS=true, but the request and migration credentials resolve to the same PostgreSQL role "
            "'mcp_owner' (request current_user 'mcp_owner', POSTGRESQL_MIGRATION_USER='mcp_alias'). The table owner "
            'bypasses row-level security, so nothing would be enforced. Create a separate login role for requests '
            f"(CREATE ROLE {ROLE} NOLOGIN; CREATE ROLE <login> LOGIN PASSWORD '...' NOSUPERUSER NOBYPASSRLS; "
            f'GRANT {ROLE} TO <login>;) and point the request connection settings at it ({DOCS}).'
        )

    @pytest.mark.asyncio
    async def test_the_maintenance_role_is_current_user_on_the_maintenance_backend(self) -> None:
        conn = MagicMock()
        conn.fetchval = AsyncMock(return_value='mcp_owner')
        backend = MagicMock()

        async def _read(operation: Callable[[object], Awaitable[object]], *, access_scope: object) -> object:
            assert access_scope is MAINTENANCE
            return await operation(conn)

        backend.execute_read = _read

        assert await read_login_role(cast(StorageBackend, backend)) == 'mcp_owner'
        conn.fetchval.assert_awaited_once_with('SELECT current_user')


class TestSplitRolesInEitherFlagState:
    """Membership and privileges are checked whenever the roles are split."""

    @pytest.mark.parametrize('rls', [True, False])
    def test_a_prepared_database_passes(self, rls: bool) -> None:
        assert _problems(_posture(enabled=rls), rls_settings(enabled=rls)) == []

    @pytest.mark.parametrize('rls', [True, False])
    def test_a_request_role_outside_the_policy_role_is_named_with_the_grant(self, rls: bool) -> None:
        problem = _single(_problems(_posture(enabled=rls, policy_member=False), rls_settings(enabled=rls)))

        assert f'not a member of the policy role {ROLE}' in problem
        assert f'GRANT {ROLE} TO "mcp_app";' in problem

    @pytest.mark.parametrize('rls', [True, False])
    def test_a_revoked_privilege_is_named(self, rls: bool) -> None:
        held = _all_held()
        revoked = HeldPrivileges(
            schema_usage=True,
            tables={**held.tables, 'tags': frozenset({'SELECT', 'UPDATE', 'DELETE'})},
            sequences={'"tenant one"."tags_id_seq"': frozenset({'SELECT'})},
            merge_patch=True,
        )

        problem = _single(_problems(_posture(enabled=rls, held=revoked), rls_settings(enabled=rls)))

        assert f'GRANT INSERT ON TABLE {SCHEMA}.tags TO {ROLE}' in problem
        assert f'GRANT USAGE ON SEQUENCE "tenant one"."tags_id_seq" TO {ROLE}' in problem


class TestFlagOff:
    """With the flag off, row-level security enabled outside the server is refused."""

    def test_row_level_security_active_on_a_managed_table_is_refused(self) -> None:
        tables = {
            'context_entries': TableSecurity(enabled=True, forced=False, active=True, owned=False),
            'context_entry_grants': TableSecurity(enabled=False, forced=False, active=False, owned=False),
            'tags': TableSecurity(enabled=False, forced=False, active=False, owned=False),
        }

        problem = _single(_problems(_posture(enabled=False, tables=tables), rls_settings(enabled=False)))

        assert 'ENABLE_POSTGRESQL_RLS=false' in problem
        assert f'ALTER TABLE {SCHEMA}.context_entries DISABLE ROW LEVEL SECURITY' in problem
        assert 'tags' not in problem


class TestFlagOn:
    """With the flag on, the request role is filtered on every managed table by our objects alone."""

    def test_a_table_without_row_level_security_is_refused(self) -> None:
        tables = {**_posture().tables, 'tags': TableSecurity(enabled=False, forced=False, active=False, owned=False)}

        problem = _single(_problems(_posture(tables=tables), rls_settings()))

        assert f'ALTER TABLE {SCHEMA}.tags ENABLE ROW LEVEL SECURITY' in problem

    def test_a_table_whose_policies_do_not_apply_to_the_request_role_is_refused(self) -> None:
        tables = {**_posture().tables, 'tags': TableSecurity(enabled=True, forced=False, active=False, owned=True)}

        problem = _single(_problems(_posture(tables=tables), rls_settings()))

        assert 'does not apply' in problem
        assert 'tags' in problem
        assert 'privileges of the table owner' in problem

    def test_forced_row_level_security_is_refused(self) -> None:
        tables = {**_posture().tables, 'tags': TableSecurity(enabled=True, forced=True, active=True, owned=False)}

        problem = _single(_problems(_posture(tables=tables), rls_settings()))

        assert f'ALTER TABLE {SCHEMA}.tags NO FORCE ROW LEVEL SECURITY' in problem

    @pytest.mark.parametrize('roles', [('PUBLIC',), ('mcp_app',), ('mcp_context_request', 'mcp_reader')])
    def test_a_foreign_policy_that_applies_to_the_request_role_is_refused(self, roles: tuple[str, ...]) -> None:
        foreign = ApplyingPolicy(table='tags', name='tags_everyone', permissive=True, roles=roles)

        problem = _single(_problems(_posture(applying_policies=(foreign,)), rls_settings()))

        assert 'tags_everyone' in problem
        assert 'tags' in problem
        assert ', '.join(roles) in problem
        assert f'DROP POLICY "tags_everyone" ON {SCHEMA}.tags' in problem

    def test_a_restrictive_foreign_policy_is_refused_as_well(self) -> None:
        foreign = ApplyingPolicy(table='context_entries', name='deny_all', permissive=False, roles=('PUBLIC',))

        problem = _single(_problems(_posture(applying_policies=(foreign,)), rls_settings()))

        assert 'RESTRICTIVE' in problem

    def test_our_own_policies_are_not_foreign(self) -> None:
        ours = tuple(
            ApplyingPolicy(table='tags', name=policy.name, permissive=True, roles=(ROLE_NAME,))
            for policy in table_policies('tags', schema=SCHEMA, role=ROLE)
        )

        assert _problems(_posture(applying_policies=ours), rls_settings()) == []

    def test_an_object_changed_outside_the_server_is_named(self) -> None:
        state = installed_state(dict.fromkeys(TABLES, True))
        helper = helper_functions(SCHEMA)[0]
        drifted = dataclasses.replace(state.objects[helper.key], definition='edited by hand')
        rls = dataclasses.replace(state, objects={**state.objects, helper.key: drifted})

        problem = _single(_problems(_posture(rls=rls), rls_settings()))

        assert helper.target in problem

    def test_a_missing_guard_trigger_is_named(self) -> None:
        state = installed_state(dict.fromkeys(TABLES, True))
        trigger = guard_trigger(SCHEMA)
        objects = {key: value for key, value in state.objects.items() if key != trigger.key}

        problem = _single(_problems(_posture(rls=dataclasses.replace(state, objects=objects)), rls_settings()))

        assert trigger.target in problem


class TestReadView:
    """The read view is owned by the table owner, plain, write protected, readable by the policy role alone."""

    @staticmethod
    def _with_view(**changes: Any) -> RlsState:
        state = installed_state(dict.fromkeys(TABLES, True))
        assert state.view is not None
        return dataclasses.replace(state, view=dataclasses.replace(state.view, **changes))

    def test_another_owner_is_refused(self) -> None:
        problems = _problems(_posture(rls=self._with_view(owned_by_table_owner=False)), rls_settings())

        assert any('not owned by the owner of' in problem for problem in problems)

    def test_view_options_are_refused(self) -> None:
        problems = _problems(_posture(rls=self._with_view(plain=False)), rls_settings())

        assert any('security_barrier' in problem for problem in problems)

    def test_a_disabled_write_rejection_trigger_is_refused(self) -> None:
        state = installed_state(dict.fromkeys(TABLES, True))
        trigger = reject_trigger(SCHEMA)
        disabled = dataclasses.replace(state.objects[trigger.key], enabled=False)
        rls = dataclasses.replace(state, objects={**state.objects, trigger.key: disabled})

        problems = _problems(_posture(rls=rls), rls_settings())

        assert any('reject_writes' in problem and 'disabled' in problem for problem in problems)

    def test_a_grant_beyond_the_policy_roles_select_prints_the_revoke_statements(self) -> None:
        rls = self._with_view(acl={ROLE_NAME: ('INSERT', 'SELECT'), 'mcp_outsider': ('SELECT',), None: ('SELECT',)})

        problem = _single(_problems(_posture(rls=rls), rls_settings()))

        assert f'REVOKE ALL ON {SCHEMA}.context_entries_readable FROM "mcp_outsider"' in problem
        assert f'REVOKE ALL ON {SCHEMA}.context_entries_readable FROM PUBLIC' in problem
        assert f'REVOKE INSERT ON {SCHEMA}.context_entries_readable FROM {ROLE}' in problem
        assert 'mcp_outsider (SELECT)' in problem

    def test_a_policy_role_that_cannot_execute_the_helpers_is_refused(self) -> None:
        rls = dataclasses.replace(installed_state(dict.fromkeys(TABLES, True)), helpers_executable=False)

        problems = _problems(_posture(rls=rls), rls_settings())

        assert any('GRANT EXECUTE ON FUNCTION' in problem and ROLE in problem for problem in problems)

    def test_a_missing_view_is_refused(self) -> None:
        state = installed_state(dict.fromkeys(TABLES, True))
        objects = {key: value for key, value in state.objects.items() if key.name != 'context_entries_readable'}

        problems = _problems(_posture(rls=dataclasses.replace(state, objects=objects, view=None)), rls_settings())

        assert any('context_entries_readable' in problem for problem in problems)


class TestVerifyRlsPosture:
    """The checks run on the request backend and refuse with every problem at once."""

    @staticmethod
    def _backend(posture: Posture) -> tuple[StorageBackend, list[object]]:
        scopes: list[object] = []

        async def _read(operation: Callable[[object], Awaitable[object]], *, access_scope: object) -> object:
            _ = operation
            scopes.append(access_scope)
            return posture

        return cast(StorageBackend, SimpleNamespace(execute_read=_read)), scopes

    @pytest.mark.asyncio
    async def test_a_prepared_database_passes(self) -> None:
        backend, scopes = self._backend(_posture())

        await verify_rls_posture(backend, rls_settings(), request_role=APP, maintenance_role='mcp_owner')

        assert scopes == [MAINTENANCE]

    @pytest.mark.asyncio
    async def test_the_refusal_reports_the_request_role_and_every_problem(self) -> None:
        forced = TableSecurity(enabled=True, forced=True, active=True, owned=False)
        backend, _ = self._backend(
            _posture(policy_member=False, tables={**_posture().tables, 'tags': forced}),
        )

        with pytest.raises(ConfigurationError) as refusal:
            await verify_rls_posture(backend, rls_settings(), request_role=APP, maintenance_role='mcp_owner')

        message = str(refusal.value)
        assert "'mcp_app'" in message
        assert 'rolsuper=false' in message
        assert 'rolbypassrls=false' in message
        assert "'mcp_owner'" in message
        assert 'not a member of the policy role' in message
        assert 'NO FORCE ROW LEVEL SECURITY' in message
        assert DOCS in message


class TestJwtWarning:
    """The jwt provider on PostgreSQL without row-level security is announced once."""

    WARNING = (
        'MCP_AUTH_PROVIDER=jwt on PostgreSQL without ENABLE_POSTGRESQL_RLS: '
        'access isolation is enforced in the application layer only.'
    )

    @staticmethod
    def _settings(
        *, provider: str = 'jwt', backend: str = 'postgresql', rls: bool = False, transport: str = 'http',
    ) -> AppSettings:
        return cast(
            AppSettings,
            SimpleNamespace(
                auth=SimpleNamespace(provider=provider),
                storage=SimpleNamespace(backend_type=backend),
                access_control=SimpleNamespace(postgresql_rls_enabled=rls),
                transport=SimpleNamespace(transport=transport),
            ),
        )

    def test_jwt_on_postgresql_without_rls_over_http_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger=posture_module.__name__)

        warn_jwt_without_rls(self._settings())

        assert [record.getMessage() for record in caplog.records] == [self.WARNING]
        assert caplog.records[0].levelno == logging.WARNING

    @pytest.mark.parametrize(
        'changes',
        [{'provider': 'simple_token'}, {'backend': 'sqlite'}, {'rls': True}, {'transport': 'stdio'}],
    )
    def test_any_other_configuration_is_silent(self, changes: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger=posture_module.__name__)

        warn_jwt_without_rls(self._settings(**changes))

        assert caplog.records == []


def test_the_test_view_state_is_the_one_the_migration_leaves() -> None:
    """The posture fixtures start from a view state the migration itself would leave."""
    state = installed_state(dict.fromkeys(TABLES, True))

    assert state.view == InstalledView(owned_by_table_owner=True, plain=True, acl={ROLE_NAME: ('SELECT',)})
