"""Session profiles against a real PostgreSQL server.

The request profile under row-level security binds the caller's identity and ``jit=off``
transaction-locally, so the values hold for every statement of the wrapped read and are
gone once it commits, on the same pooled connection. The maintenance profile turns row
security off on every pooled connection, and the maintenance backend logs in with the
migration credentials over the request connection string.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Protocol

import asyncpg
import pytest
import pytest_asyncio

import app.backends.postgresql_backend as pg_module
from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessScope
from app.backends.base import MAINTENANCE
from app.backends.base import StorageBackend
from app.backends.factory import create_backend
from app.backends.factory import create_maintenance_backend
from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.postgresql_backend.session import NO_CREDENTIAL_OVERRIDE
from app.backends.postgresql_backend.session_profiles import RLS_READ_VIEWS
from app.backends.postgresql_backend.session_profiles import MaintenanceProfile
from app.backends.postgresql_backend.session_profiles import RlsRequestProfile
from app.backends.postgresql_backend.session_profiles import SessionProfile
from app.settings import get_settings
from tests.helpers import rebind_package_settings

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

ALICE = AccessScope('alice', frozenset({'team-x'}), may_publish=True, may_write=True)
CAROL = AccessScope('carol', frozenset({'team-x', 'a,b', 'q"x', '{brace}', 'back\\slash'}))

IDENTITY_SQL = (
    "SELECT current_setting('app.principal_id', true) AS principal_id, "
    "current_setting('app.groups', true) AS groups, "
    "current_setting('app.may_publish', true) AS may_publish, "
    "current_setting('app.may_write', true) AS may_write, "
    "current_setting('jit') AS jit, "
    'pg_backend_pid() AS pid'
)


@pytest.fixture
def pg_routing(monkeypatch: pytest.MonkeyPatch, pg_test_url: str) -> Callable[..., None]:
    """Return a callable routing the settings to the test container with a one-connection pool.

    Every inherited ``POSTGRESQL_*`` variable and ``ENABLE_POSTGRESQL_RLS`` is removed before
    the given variables are set, so nothing from the shell reaches the backends under test.

    Returns:
        A callable taking further variables to set as keyword arguments.
    """

    def _route(**variables: str) -> None:
        for key in list(os.environ):
            if key.startswith('POSTGRESQL_') or key == 'ENABLE_POSTGRESQL_RLS':
                monkeypatch.delenv(key)
        monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
        monkeypatch.setenv('POSTGRESQL_CONNECTION_STRING', pg_test_url)
        monkeypatch.setenv('POSTGRESQL_POOL_MIN', '1')
        monkeypatch.setenv('POSTGRESQL_POOL_MAX', '1')
        for key, value in variables.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()
        rebind_package_settings(monkeypatch, pg_module, get_settings())

    return _route


class Starter(Protocol):
    """Initializes a backend and returns it."""

    async def __call__[B: StorageBackend](self, backend: B) -> B: ...


@pytest_asyncio.fixture
async def started() -> AsyncIterator[Starter]:
    """Yield a starter that initializes backends and shuts every one of them down afterwards."""
    backends: list[StorageBackend] = []

    async def _start[B: StorageBackend](backend: B) -> B:
        await backend.initialize()
        backends.append(backend)
        return backend

    yield _start
    for backend in backends:
        await backend.shutdown()
    get_settings.cache_clear()


def _backend(profile: SessionProfile) -> PostgreSQLBackend:
    """Build a backend over the routed request settings with ``profile``."""
    return PostgreSQLBackend(provision_vector=False, profile=profile, credentials=NO_CREDENTIAL_OVERRIDE)


type AnyConnection = asyncpg.Connection[asyncpg.Record] | asyncpg.pool.PoolConnectionProxy[asyncpg.Record]


async def _identity(conn: AnyConnection) -> asyncpg.Record:
    row = await conn.fetchrow(IDENTITY_SQL)
    assert row is not None
    return row


async def _unwrapped_identity(backend: PostgreSQLBackend) -> asyncpg.Record:
    """Read the identity settings on the pooled connection outside any transaction."""
    async with backend.get_connection() as conn:
        return await _identity(conn)


@pytest.mark.asyncio
async def test_identity_holds_inside_the_wrapped_read_and_is_gone_after_commit(
    pg_routing: Callable[..., None], started: Starter,
) -> None:
    pg_routing()
    backend = await started(_backend(RlsRequestProfile(read_views=RLS_READ_VIEWS)))
    before = await _unwrapped_identity(backend)

    inside = await backend.execute_read(_identity, access_scope=CAROL)
    after = await _unwrapped_identity(backend)

    assert inside['pid'] == before['pid'] == after['pid']
    assert inside['principal_id'] == 'carol'
    assert inside['may_publish'] == 'off'
    assert inside['may_write'] == 'off'
    assert inside['jit'] == 'off'
    async with backend.get_connection() as conn:
        assert await conn.fetchval('SELECT $1::text::text[]', inside['groups']) == CAROL.groups_list()
    assert after['principal_id'] in ('', None)
    assert after['groups'] in ('', None)
    assert after['may_publish'] in ('', None)
    assert after['may_write'] in ('', None)
    assert after['jit'] == before['jit']


@pytest.mark.asyncio
async def test_writes_and_transactions_bind_the_caller(
    pg_routing: Callable[..., None], started: Starter,
) -> None:
    pg_routing()
    backend = await started(_backend(RlsRequestProfile(read_views=RLS_READ_VIEWS)))

    written = await backend.execute_write(_identity, access_scope=ALICE)
    async with backend.begin_transaction(access_scope=CAROL) as txn:
        transactional = await _identity(txn.connection)
    after = await _unwrapped_identity(backend)

    assert (written['principal_id'], written['may_publish'], written['may_write'], written['jit']) == (
        'alice', 'on', 'on', 'off',
    )
    assert (
        transactional['principal_id'], transactional['may_publish'], transactional['may_write'], transactional['jit'],
    ) == ('carol', 'off', 'off', 'off')
    assert after['principal_id'] in ('', None)
    assert after['may_write'] in ('', None)


@pytest.mark.asyncio
async def test_scopes_without_a_principal_bind_nothing(
    pg_routing: Callable[..., None], started: Starter,
) -> None:
    pg_routing()
    backend = await started(_backend(RlsRequestProfile(read_views=RLS_READ_VIEWS)))
    baseline = await _unwrapped_identity(backend)
    await backend.execute_read(_identity, access_scope=CAROL)

    for access in (SYSTEM_SCOPE, MAINTENANCE):
        inside = await backend.execute_read(_identity, access_scope=access)
        assert inside['pid'] == baseline['pid']
        assert inside['principal_id'] in ('', None)
        assert inside['jit'] == baseline['jit']


@pytest.mark.asyncio
async def test_maintenance_connections_turn_row_security_off_on_every_acquire(
    pg_routing: Callable[..., None], started: Starter,
) -> None:
    pg_routing()
    maintenance = await started(_backend(MaintenanceProfile()))
    request = await started(_backend(RlsRequestProfile(read_views=RLS_READ_VIEWS)))

    async def _row_security(conn: asyncpg.Connection) -> tuple[str, int]:
        return await conn.fetchval('SHOW row_security'), await conn.fetchval('SELECT pg_backend_pid()')

    first = await maintenance.execute_read(_row_security, access_scope=MAINTENANCE)
    # The pool runs RESET ALL on release; the next acquire of the same connection applies the profile again.
    second = await maintenance.execute_read(_row_security, access_scope=MAINTENANCE)

    assert first == second
    assert first[0] == 'off'
    assert (await request.execute_read(_row_security, access_scope=ALICE))[0] == 'on'


@pytest.mark.asyncio
async def test_maintenance_backend_logs_in_with_the_migration_credentials(
    pg_routing: Callable[..., None], started: Starter,
) -> None:
    pg_routing(POSTGRESQL_MIGRATION_USER='mcp_owner', POSTGRESQL_MIGRATION_PASSWORD='mcp_owner')
    maintenance = await started(create_maintenance_backend(provision_vector=False))
    request = await started(create_backend(provision_vector=False))

    async def _current_user(conn: asyncpg.Connection) -> str:
        return await conn.fetchval('SELECT current_user')

    assert await maintenance.execute_read(_current_user, access_scope=MAINTENANCE) == 'mcp_owner'
    assert await request.execute_read(_current_user, access_scope=MAINTENANCE) == 'mcp_test'
