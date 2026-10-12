"""JWT principals reach only the entries their ownership and grants allow (PostgreSQL, over HTTP).

Runs the access scenario of :mod:`tests.integration._http_jwt` against a real HTTP server
verifying minted JWTs, once per protocol era. The server serves an isolated database on
the pgvector container of the ``pg_test_url`` fixture (``@requires_docker_postgres``,
skipped cleanly without Docker); the test opens the same database as the superuser to
insert the user and write grants no tool writes. The scenario runs twice: on a database
the superuser serves with row-level security off, and on the ``pg_rls_db`` database served
as ``mcp_app`` with ``ENABLE_POSTGRESQL_RLS``, where the principal and groups of each token
become the identity its transactions bind and the policies decide what each request
reaches. The same scenario runs against SQLite in
``tests/integration/sqlite/test_http_jwt_access.py``.
"""

import contextlib
from collections.abc import AsyncIterator
from collections.abc import Iterator
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import pytest
import pytest_asyncio
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from tests.integration._harness.core import CLIENT_MODES
from tests.integration._harness.core import ClientMode
from tests.integration._http_jwt import build_access_env
from tests.integration._http_jwt import free_port
from tests.integration._http_jwt import running_http_server
from tests.integration._http_jwt import server_database
from tests.integration._http_jwt import verify_jwt_access_scoping
from tests.integration.postgresql._rls_catalog import assert_served_with_row_level_security
from tests.integration.postgresql.conftest import RLS_CREDENTIALS
from tests.integration.postgresql.conftest import RlsUrls

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

ACCESS_DATABASE = 'mcp_http_jwt_access'


def _with_database(pg_url: str, database: str) -> str:
    """Return ``pg_url`` pointing at ``database``.

    Returns:
        The connection string of ``database``.
    """
    parts = urlsplit(pg_url)
    return urlunsplit((parts.scheme, parts.netloc, f'/{database}', parts.query, parts.fragment))


@pytest_asyncio.fixture
async def access_database_url(pg_test_url: str) -> AsyncIterator[str]:
    """Create an empty database for the access server and drop it afterwards.

    Yields:
        The connection string of the database.
    """
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {ACCESS_DATABASE} WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE {ACCESS_DATABASE}')
    finally:
        await admin.close()

    yield _with_database(pg_test_url, ACCESS_DATABASE)

    admin = await asyncpg.connect(pg_test_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {ACCESS_DATABASE} WITH (FORCE)')
    finally:
        await admin.close()


@pytest.fixture
def access_server(access_database_url: str, jwt_key_pair: RSAKeyPair) -> Iterator[str]:
    """Run the JWT access server on ``access_database_url``.

    Yields:
        The server origin URL.
    """
    env = build_access_env(
        backend='postgresql', database=access_database_url, port=free_port(), public_key=jwt_key_pair.public_key,
    )
    with running_http_server(env) as base_url:
        yield base_url


@pytest.fixture
def rls_access_server(pg_rls_db: RlsUrls, jwt_key_pair: RSAKeyPair) -> Iterator[str]:
    """Run the JWT access server on ``pg_rls_db`` as ``mcp_app`` with row-level security.

    Yields:
        The server origin URL.
    """
    env = {
        **build_access_env(
            backend='postgresql', database=pg_rls_db.app, port=free_port(), public_key=jwt_key_pair.public_key,
        ),
        **RLS_CREDENTIALS.environment(),
    }
    with running_http_server(env) as base_url:
        yield base_url


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_jwt_principals_reach_only_what_their_grants_allow(
    access_server: str,
    access_database_url: str,
    jwt_key_pair: RSAKeyPair,
    client_mode: ClientMode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group, user-read and user-write grants decide what bob and carol read, update and delete."""
    async with server_database('postgresql', access_database_url, monkeypatch) as database:
        await verify_jwt_access_scoping(access_server, jwt_key_pair, client_mode, database)


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_jwt_principals_reach_only_what_their_grants_allow_under_row_level_security(
    rls_access_server: str,
    pg_rls_db: RlsUrls,
    jwt_key_pair: RSAKeyPair,
    client_mode: ClientMode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same grants decide what bob and carol reach when the database enforces them too.

    Every request runs as ``mcp_app`` under the identity its token carries, so alice's stores
    write their ``author_groups`` read grants through the grants policies, bob reaches her
    entries through the read view's group arm, and carol only through her user grants.
    """
    async with server_database('postgresql', pg_rls_db.admin, monkeypatch) as database:
        await verify_jwt_access_scoping(rls_access_server, jwt_key_pair, client_mode, database)
    await assert_served_with_row_level_security(pg_rls_db)
