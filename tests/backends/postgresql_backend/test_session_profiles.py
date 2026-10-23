"""Tests for app/backends/postgresql_backend/session_profiles.py and the entry points that bind through it."""

import contextlib
from collections.abc import AsyncGenerator
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessScope
from app.backends.base import MAINTENANCE
from app.backends.base import ROUTABLE_TABLES
from app.backends.base import BackendAccess
from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.postgresql_backend.session import NO_CREDENTIAL_OVERRIDE
from app.backends.postgresql_backend.session_profiles import BIND_IDENTITY_STATEMENT
from app.backends.postgresql_backend.session_profiles import RLS_READ_VIEWS
from app.backends.postgresql_backend.session_profiles import MaintenanceProfile
from app.backends.postgresql_backend.session_profiles import RlsRequestProfile
from app.backends.postgresql_backend.session_profiles import SessionProfile
from app.backends.postgresql_backend.session_profiles import UnboundProfile
from tests.backends.postgresql_backend._builders import fast_retries

ALICE = AccessScope('alice', frozenset({'team-x'}), may_publish=True, may_write=True)
# Group ids carrying every character an array literal must escape; the scope may not publish.
CAROL = AccessScope('carol', frozenset({'team-x', 'a,b', 'q"x', '{brace}', 'back\\slash'}))
ACCESSES = [ALICE, SYSTEM_SCOPE, MAINTENANCE]
ACCESS_IDS = ['request', 'system', 'maintenance']
UNMAPPED_TABLES = sorted(ROUTABLE_TABLES - {'context_entries'})


class RecordingProfile:
    """A session profile that records each bind and relation request into a shared event log."""

    def __init__(self, events: list[tuple[str, object]], *, wraps_reads: bool = False) -> None:
        self._events = events
        self.wraps_reads = wraps_reads

    def session_statements(self) -> list[str]:
        return []

    async def bind(self, conn: 'asyncpg.pool.PoolConnectionProxy[asyncpg.Record]', access: BackendAccess) -> None:
        self._events.append(('bind', access))
        self._events.append(('bind-connection', conn))

    def readable_relation(self, table: str, *, alias: str) -> str:
        self._events.append(('relation', (table, alias)))
        return f'recorded:{table}:{alias}'


def _recording_connection(events: list[tuple[str, object]]) -> MagicMock:
    """Build a connection whose transaction context records its begin and end."""

    @contextlib.asynccontextmanager
    async def _transaction() -> AsyncGenerator[None, None]:
        events.append(('begin', None))
        yield None
        events.append(('commit', None))

    conn = MagicMock()
    conn.transaction = MagicMock(side_effect=_transaction)
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock()
    conn.fetchrow = AsyncMock()
    conn.fetchval = AsyncMock()
    return conn


def _backend(profile: SessionProfile) -> PostgreSQLBackend:
    """Build a non-shut-down backend over ``profile`` without a pool."""
    backend = PostgreSQLBackend(
        connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        profile=profile,
        credentials=NO_CREDENTIAL_OVERRIDE,
    )
    backend._shutdown = False
    return backend


def _serve_connections(backend: PostgreSQLBackend, connections: list[MagicMock]) -> None:
    """Hand the connections out in order, both to ``get_connection`` and to the pool."""
    queue = list(connections)

    @contextlib.asynccontextmanager
    async def _acquire(*_args: object, **_kwargs: object) -> AsyncGenerator[object, None]:
        yield queue.pop(0)

    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=_acquire)
    backend._pool = pool
    cast(Any, backend).get_connection = _acquire


class TestUnboundProfile:
    """The profile of a request backend without row-level security changes nothing."""

    def test_wraps_no_reads_and_sets_no_session_state(self) -> None:
        profile = UnboundProfile()

        assert profile.wraps_reads is False
        assert profile.session_statements() == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize('access', [ALICE, SYSTEM_SCOPE, MAINTENANCE], ids=['request', 'system', 'maintenance'])
    async def test_bind_issues_no_statement(self, access: BackendAccess) -> None:
        conn = _recording_connection([])

        await UnboundProfile().bind(conn, access)

        conn.execute.assert_not_awaited()
        conn.fetch.assert_not_awaited()
        conn.fetchrow.assert_not_awaited()
        conn.fetchval.assert_not_awaited()

    def test_satisfies_the_session_profile_protocol(self) -> None:
        assert isinstance(UnboundProfile(), SessionProfile)

    @pytest.mark.parametrize('table', sorted(ROUTABLE_TABLES))
    def test_relation_named_by_its_own_alias_is_the_bare_table(self, table: str) -> None:
        assert UnboundProfile().readable_relation(table, alias=table) == table

    @pytest.mark.parametrize('table', sorted(ROUTABLE_TABLES))
    def test_relation_with_a_distinct_alias_keeps_the_alias(self, table: str) -> None:
        assert UnboundProfile().readable_relation(table, alias='ce') == f'{table} ce'

    @pytest.mark.parametrize(
        'table', ['context_entry_grants', 'context_entries_fts', 'compression_metadata', 'CONTEXT_ENTRIES'],
    )
    def test_table_outside_the_routable_set_is_refused(self, table: str) -> None:
        with pytest.raises(ValueError, match='routable'):
            UnboundProfile().readable_relation(table, alias='ce')

    @pytest.mark.parametrize('alias', ['', 'CE', '1ce', 'ce x', 'ce;', 'ce\n'])
    def test_malformed_alias_is_refused(self, alias: str) -> None:
        with pytest.raises(ValueError, match='alias'):
            UnboundProfile().readable_relation('context_entries', alias=alias)


class TestExecuteWriteBinding:
    """execute_write binds the caller's identity first inside every attempt's transaction."""

    @pytest.mark.asyncio
    async def test_binds_before_the_operation_inside_the_transaction(self) -> None:
        events: list[tuple[str, object]] = []
        backend = _backend(RecordingProfile(events))
        conn = _recording_connection(events)
        _serve_connections(backend, [conn])

        async def operation(_conn: object, value: str) -> str:
            events.append(('operation', value))
            return value

        result = await backend.execute_write(operation, 'written', access_scope=ALICE)

        assert result == 'written'
        assert events == [
            ('begin', None), ('bind', ALICE), ('bind-connection', conn), ('operation', 'written'), ('commit', None),
        ]

    @pytest.mark.asyncio
    async def test_access_scope_never_reaches_the_operation(self) -> None:
        backend = _backend(UnboundProfile())
        _serve_connections(backend, [_recording_connection([])])
        received: dict[str, object] = {}

        async def operation(_conn: object, **kwargs: object) -> None:
            received.update(kwargs)

        await backend.execute_write(operation, access_scope=MAINTENANCE, marker='kept')

        assert received == {'marker': 'kept'}

    @pytest.mark.asyncio
    async def test_retry_binds_again_on_the_fresh_connection(self) -> None:
        events: list[tuple[str, object]] = []
        backend = _backend(RecordingProfile(events))
        fast_retries(backend)
        first, second = _recording_connection(events), _recording_connection(events)
        _serve_connections(backend, [first, second])
        attempts = {'n': 0}

        async def operation(_conn: object) -> str:
            attempts['n'] += 1
            if attempts['n'] == 1:
                raise asyncpg.exceptions.ConnectionDoesNotExistError('connection dropped')
            return 'done'

        assert await backend.execute_write(operation, access_scope=ALICE) == 'done'
        assert [connection for kind, connection in events if kind == 'bind-connection'] == [first, second]
        assert [access for kind, access in events if kind == 'bind'] == [ALICE, ALICE]


class TestBeginTransactionBinding:
    """begin_transaction binds the caller's identity before the body runs."""

    @pytest.mark.asyncio
    async def test_binds_first_inside_the_transaction(self) -> None:
        events: list[tuple[str, object]] = []
        backend = _backend(RecordingProfile(events))
        conn = _recording_connection(events)
        _serve_connections(backend, [conn])

        async with backend.begin_transaction(access_scope=ALICE) as txn:
            events.append(('body', txn.connection))

        assert events == [('begin', None), ('bind', ALICE), ('bind-connection', conn), ('body', conn), ('commit', None)]

    @pytest.mark.asyncio
    async def test_unbound_profile_adds_no_statement(self) -> None:
        backend = _backend(UnboundProfile())
        conn = _recording_connection([])
        _serve_connections(backend, [conn])

        async with backend.begin_transaction(access_scope=MAINTENANCE):
            pass

        conn.execute.assert_not_awaited()


class TestExecuteReadBinding:
    """execute_read hands the caller's identity to the profile, inside a transaction only when the profile wraps reads."""

    @pytest.mark.asyncio
    async def test_wrapping_profile_binds_first_inside_a_transaction(self) -> None:
        events: list[tuple[str, object]] = []
        backend = _backend(RecordingProfile(events, wraps_reads=True))
        conn = _recording_connection(events)
        _serve_connections(backend, [conn])

        async def operation(_conn: object, value: str) -> str:
            events.append(('operation', value))
            return value

        result = await backend.execute_read(operation, 'read', access_scope=ALICE)

        assert result == 'read'
        assert events == [
            ('begin', None), ('bind', ALICE), ('bind-connection', conn), ('operation', 'read'), ('commit', None),
        ]

    @pytest.mark.asyncio
    async def test_non_wrapping_profile_runs_the_read_as_issued(self) -> None:
        events: list[tuple[str, object]] = []
        backend = _backend(RecordingProfile(events))
        _serve_connections(backend, [_recording_connection(events)])

        async def operation(_conn: object) -> str:
            events.append(('operation', None))
            return 'read'

        assert await backend.execute_read(operation, access_scope=ALICE) == 'read'
        assert events == [('operation', None)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('access', [ALICE, SYSTEM_SCOPE, MAINTENANCE], ids=['request', 'system', 'maintenance'])
    async def test_unbound_profile_issues_no_statement(self, access: BackendAccess) -> None:
        backend = _backend(UnboundProfile())
        conn = _recording_connection([])
        _serve_connections(backend, [conn])

        async def operation(_conn: object) -> str:
            return 'read'

        assert await backend.execute_read(operation, access_scope=access) == 'read'
        conn.transaction.assert_not_called()
        conn.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_access_scope_never_reaches_the_operation(self) -> None:
        backend = _backend(UnboundProfile())
        _serve_connections(backend, [_recording_connection([])])
        received: dict[str, object] = {}

        async def operation(_conn: object, **kwargs: object) -> None:
            received.update(kwargs)

        await backend.execute_read(operation, access_scope=MAINTENANCE, marker='kept')

        assert received == {'marker': 'kept'}


class TestReadableRelation:
    """The backend names a readable relation through its profile."""

    def test_backend_delegates_to_its_profile(self) -> None:
        events: list[tuple[str, object]] = []
        backend = _backend(RecordingProfile(events))

        assert backend.readable_relation('context_entries', alias='ce') == 'recorded:context_entries:ce'
        assert events == [('relation', ('context_entries', 'ce'))]

    def test_unbound_backend_renders_the_base_form(self) -> None:
        backend = _backend(UnboundProfile())

        assert backend.readable_relation('context_entries', alias='context_entries') == 'context_entries'
        assert backend.readable_relation('tags', alias='t') == 'tags t'


def _statement_recording_connection(events: list[tuple[str, object]]) -> MagicMock:
    """Build a recording connection whose ``execute`` also lands in the event log."""
    conn = _recording_connection(events)

    async def _execute(*args: object) -> str:
        events.append(('execute', args))
        return 'SELECT 1'

    conn.execute = AsyncMock(side_effect=_execute)
    return conn


def _bind_args(scope: AccessScope) -> tuple[object, ...]:
    """Return the arguments the bind statement of ``scope`` is executed with."""
    return (
        BIND_IDENTITY_STATEMENT,
        scope.principal_id,
        scope.groups_list(),
        'on' if scope.may_publish else 'off',
        'on' if scope.may_write else 'off',
    )


class TestBindStatement:
    """The identity is bound by one transaction-local set_config statement."""

    def test_statement_sets_the_four_identity_settings_and_jit_transaction_locally(self) -> None:
        expected = (
            "SELECT pg_catalog.set_config('app.principal_id', $1, true), "
            "pg_catalog.set_config('app.groups', $2::text[]::text, true), "
            "pg_catalog.set_config('app.may_publish', $3, true), "
            "pg_catalog.set_config('app.may_write', $4, true), "
            "pg_catalog.set_config('jit', 'off', true)"
        )

        assert ' '.join(BIND_IDENTITY_STATEMENT.split()) == expected


class TestRlsRequestProfile:
    """The profile of a request backend under row-level security binds the caller and reads through the views."""

    @staticmethod
    def _profile() -> RlsRequestProfile:
        return RlsRequestProfile(read_views=RLS_READ_VIEWS)

    def test_wraps_reads_and_sets_no_session_state(self) -> None:
        profile = self._profile()

        assert profile.wraps_reads is True
        assert profile.session_statements() == []

    def test_satisfies_the_session_profile_protocol(self) -> None:
        assert isinstance(self._profile(), SessionProfile)

    def test_maps_the_context_table_to_its_read_view(self) -> None:
        assert RLS_READ_VIEWS == {'context_entries': 'context_entries_readable'}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('scope', 'expected'),
        [
            (ALICE, ('alice', ['team-x'], 'on', 'on')),
            (CAROL, ('carol', ['a,b', 'back\\slash', 'q"x', 'team-x', '{brace}'], 'off', 'off')),
            (AccessScope('dave', frozenset(), may_write=True), ('dave', [], 'off', 'on')),
        ],
        ids=['publisher-and-writer', 'odd-groups-read-only', 'writer-only'],
    )
    async def test_request_scope_binds_principal_sorted_groups_and_both_flags(
        self, scope: AccessScope, expected: tuple[object, ...],
    ) -> None:
        conn = _recording_connection([])

        await self._profile().bind(conn, scope)

        conn.execute.assert_awaited_once_with(BIND_IDENTITY_STATEMENT, *expected)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('access', [SYSTEM_SCOPE, MAINTENANCE], ids=['system', 'maintenance'])
    async def test_scopes_without_a_principal_bind_nothing(self, access: BackendAccess) -> None:
        conn = _recording_connection([])

        await self._profile().bind(conn, access)

        conn.execute.assert_not_awaited()
        conn.fetch.assert_not_awaited()
        conn.fetchrow.assert_not_awaited()
        conn.fetchval.assert_not_awaited()

    @pytest.mark.parametrize('alias', ['ce', 'context_entries'])
    def test_mapped_table_reads_through_its_view_under_the_statement_alias(self, alias: str) -> None:
        assert self._profile().readable_relation('context_entries', alias=alias) == f'context_entries_readable AS {alias}'

    @pytest.mark.parametrize('table', UNMAPPED_TABLES)
    def test_unmapped_table_keeps_the_base_form(self, table: str) -> None:
        profile = self._profile()

        assert profile.readable_relation(table, alias=table) == table
        assert profile.readable_relation(table, alias='x') == f'{table} x'

    @pytest.mark.parametrize('table', ['context_entry_grants', 'context_entries_readable', 'CONTEXT_ENTRIES'])
    def test_table_outside_the_routable_set_is_refused(self, table: str) -> None:
        with pytest.raises(ValueError, match='routable'):
            self._profile().readable_relation(table, alias='ce')

    @pytest.mark.parametrize('table', ['context_entries', 'tags'])
    @pytest.mark.parametrize('alias', ['', 'CE', '1ce', 'ce x', 'ce;'])
    def test_malformed_alias_is_refused_for_mapped_and_unmapped_tables(self, table: str, alias: str) -> None:
        with pytest.raises(ValueError, match='alias'):
            self._profile().readable_relation(table, alias=alias)

    @pytest.mark.parametrize(
        'read_views',
        [{'context_entry_grants': 'grants_readable'}, {'context_entries': 'Readable'}, {'context_entries': 'v; drop'}],
        ids=['unroutable-table', 'uppercase-view', 'injected-view'],
    )
    def test_construction_refuses_a_malformed_view_map(self, read_views: dict[str, str]) -> None:
        with pytest.raises(ValueError, match='routable|view'):
            RlsRequestProfile(read_views=read_views)


class TestMaintenanceProfile:
    """The profile of the maintenance backend turns row security off and binds nothing."""

    def test_wraps_no_reads_and_turns_row_security_off(self) -> None:
        profile = MaintenanceProfile()

        assert profile.wraps_reads is False
        assert 'SET row_security = off' in profile.session_statements()

    def test_satisfies_the_session_profile_protocol(self) -> None:
        assert isinstance(MaintenanceProfile(), SessionProfile)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('access', ACCESSES, ids=ACCESS_IDS)
    async def test_bind_issues_no_statement(self, access: BackendAccess) -> None:
        conn = _recording_connection([])

        await MaintenanceProfile().bind(conn, access)

        conn.execute.assert_not_awaited()
        conn.fetch.assert_not_awaited()

    @pytest.mark.parametrize('table', sorted(ROUTABLE_TABLES))
    def test_every_table_keeps_the_base_form(self, table: str) -> None:
        assert MaintenanceProfile().readable_relation(table, alias=table) == table
        assert MaintenanceProfile().readable_relation(table, alias='x') == f'{table} x'

    def test_arguments_are_validated(self) -> None:
        with pytest.raises(ValueError, match='routable'):
            MaintenanceProfile().readable_relation('context_entry_grants', alias='g')
        with pytest.raises(ValueError, match='alias'):
            MaintenanceProfile().readable_relation('context_entries', alias='ce;')


class TestRlsBindPlacement:
    """Under row-level security the bind statement is the first statement of every transaction."""

    @staticmethod
    def _rls_backend() -> PostgreSQLBackend:
        return _backend(RlsRequestProfile(read_views=RLS_READ_VIEWS))

    @pytest.mark.asyncio
    async def test_execute_write_binds_before_the_operation(self) -> None:
        events: list[tuple[str, object]] = []
        backend = self._rls_backend()
        _serve_connections(backend, [_statement_recording_connection(events)])

        async def operation(_conn: object) -> None:
            events.append(('operation', None))

        await backend.execute_write(operation, access_scope=CAROL)

        assert events == [('begin', None), ('execute', _bind_args(CAROL)), ('operation', None), ('commit', None)]

    @pytest.mark.asyncio
    async def test_begin_transaction_binds_before_the_body(self) -> None:
        events: list[tuple[str, object]] = []
        backend = self._rls_backend()
        _serve_connections(backend, [_statement_recording_connection(events)])

        async with backend.begin_transaction(access_scope=ALICE):
            events.append(('body', None))

        assert events == [('begin', None), ('execute', _bind_args(ALICE)), ('body', None), ('commit', None)]

    @pytest.mark.asyncio
    async def test_wrapped_read_binds_before_the_operation(self) -> None:
        events: list[tuple[str, object]] = []
        backend = self._rls_backend()
        _serve_connections(backend, [_statement_recording_connection(events)])

        async def operation(_conn: object) -> str:
            events.append(('operation', None))
            return 'read'

        assert await backend.execute_read(operation, access_scope=ALICE) == 'read'
        assert events == [('begin', None), ('execute', _bind_args(ALICE)), ('operation', None), ('commit', None)]

    @pytest.mark.asyncio
    async def test_system_scope_read_is_wrapped_but_binds_nothing(self) -> None:
        events: list[tuple[str, object]] = []
        backend = self._rls_backend()
        _serve_connections(backend, [_statement_recording_connection(events)])

        async def operation(_conn: object) -> str:
            events.append(('operation', None))
            return 'read'

        await backend.execute_read(operation, access_scope=SYSTEM_SCOPE)

        assert events == [('begin', None), ('operation', None), ('commit', None)]

    @pytest.mark.asyncio
    async def test_execute_write_binds_again_after_a_connection_fault(self) -> None:
        events: list[tuple[str, object]] = []
        backend = self._rls_backend()
        fast_retries(backend)
        first, second = _statement_recording_connection(events), _statement_recording_connection(events)
        _serve_connections(backend, [first, second])
        attempts = {'n': 0}

        async def operation(_conn: object) -> str:
            attempts['n'] += 1
            if attempts['n'] == 1:
                raise asyncpg.exceptions.ConnectionDoesNotExistError('connection dropped')
            return 'done'

        assert await backend.execute_write(operation, access_scope=ALICE) == 'done'
        first.execute.assert_awaited_once_with(*_bind_args(ALICE))
        second.execute.assert_awaited_once_with(*_bind_args(ALICE))

    @pytest.mark.asyncio
    async def test_begin_transaction_binds_again_when_re_entered_after_a_fault(self) -> None:
        events: list[tuple[str, object]] = []
        backend = self._rls_backend()
        first, second = _statement_recording_connection(events), _statement_recording_connection(events)
        _serve_connections(backend, [first, second])

        with pytest.raises(asyncpg.exceptions.ConnectionDoesNotExistError):
            async with backend.begin_transaction(access_scope=ALICE):
                raise asyncpg.exceptions.ConnectionDoesNotExistError('connection dropped')
        async with backend.begin_transaction(access_scope=ALICE):
            pass

        first.execute.assert_awaited_once_with(*_bind_args(ALICE))
        second.execute.assert_awaited_once_with(*_bind_args(ALICE))

    @pytest.mark.asyncio
    async def test_wrapped_read_binds_again_after_a_fault(self) -> None:
        events: list[tuple[str, object]] = []
        backend = self._rls_backend()
        first, second = _statement_recording_connection(events), _statement_recording_connection(events)
        _serve_connections(backend, [first, second])

        async def failing(_conn: object) -> None:
            raise asyncpg.exceptions.ConnectionDoesNotExistError('connection dropped')

        async def reading(_conn: object) -> str:
            return 'read'

        with pytest.raises(asyncpg.exceptions.ConnectionDoesNotExistError):
            await backend.execute_read(failing, access_scope=ALICE)
        assert await backend.execute_read(reading, access_scope=ALICE) == 'read'

        first.execute.assert_awaited_once_with(*_bind_args(ALICE))
        second.execute.assert_awaited_once_with(*_bind_args(ALICE))

    @pytest.mark.asyncio
    @pytest.mark.parametrize('profile', [UnboundProfile(), MaintenanceProfile()], ids=['unbound', 'maintenance'])
    async def test_profiles_without_identity_emit_no_bind_on_any_entry_point(self, profile: SessionProfile) -> None:
        backend = _backend(profile)
        connections = [_recording_connection([]) for _ in range(3)]
        _serve_connections(backend, connections)

        async def operation(_conn: object) -> None:
            return None

        await backend.execute_write(operation, access_scope=ALICE)
        async with backend.begin_transaction(access_scope=ALICE):
            pass
        await backend.execute_read(operation, access_scope=ALICE)

        for conn in connections:
            conn.execute.assert_not_awaited()
