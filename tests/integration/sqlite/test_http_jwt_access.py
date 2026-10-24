"""JWT principals reach only the entries their ownership, grants and role allow (SQLite, over HTTP).

Runs the JWT scenarios against real HTTP servers verifying minted JWTs, once per
protocol era, each server on a temporary SQLite file: the access scenario of
:mod:`tests.integration._http_jwt_access`, where alice shares her entries with the
other principals until the operator CLI hands them to a new owner, and the anonymous
principal's scenarios of :mod:`tests.integration._http_jwt_anonymous`, for which the
test opens the same file to count, as the table owner, the rows the anonymous principal
owns. The same scenarios run against PostgreSQL in
``tests/integration/postgresql/test_http_jwt_access.py``.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from tests.integration._harness.core import CLIENT_MODES
from tests.integration._harness.core import ClientMode
from tests.integration._http_jwt import build_access_env
from tests.integration._http_jwt import free_port
from tests.integration._http_jwt import running_http_server
from tests.integration._http_jwt import serve_jwks
from tests.integration._http_jwt import server_database
from tests.integration._http_jwt_access import verify_jwt_access_scoping
from tests.integration._http_jwt_access import verify_reassigned_owner_manages_grants
from tests.integration._http_jwt_anonymous import JWKS_KID
from tests.integration._http_jwt_anonymous import build_anonymous_env
from tests.integration._http_jwt_anonymous import build_jwks_env
from tests.integration._http_jwt_anonymous import mint_anonymous_token
from tests.integration._http_jwt_anonymous import verify_anonymous_arm
from tests.integration._http_jwt_anonymous import verify_jwks_composition
from tests.integration._http_jwt_anonymous import verify_read_only_setting_fails_open
from tests.integration._http_jwt_anonymous import without_read_only_principal

pytestmark = pytest.mark.integration


@pytest.fixture
def access_db_path(tmp_path: Path) -> Path:
    """Return the database file the scenario's servers serve.

    Returns:
        A path inside the test's temporary directory.
    """
    return tmp_path / 'http_jwt_access.db'


@pytest.fixture
def access_server(access_db_path: Path, jwt_key_pair: RSAKeyPair) -> Iterator[str]:
    """Run the JWT access server on ``access_db_path``.

    Yields:
        The server origin URL.
    """
    env = build_access_env(
        backend='sqlite', database=str(access_db_path), port=free_port(), public_key=jwt_key_pair.public_key,
    )
    with running_http_server(env) as base_url:
        yield base_url


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_jwt_principals_reach_only_what_their_grants_allow(
    access_server: str,
    access_db_path: Path,
    jwt_key_pair: RSAKeyPair,
    client_mode: ClientMode,
) -> None:
    """Shared user grants and group grants decide what bob and carol read, update and delete.

    Once the operator reassigns alice's entries, their new owner manages their grants.
    """
    await verify_jwt_access_scoping(access_server, jwt_key_pair, client_mode)
    await verify_reassigned_owner_manages_grants(access_server, jwt_key_pair, client_mode, str(access_db_path))


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_anonymous_principal_reads_public_entries_and_writes_nothing(
    access_db_path: Path,
    jwt_key_pair: RSAKeyPair,
    client_mode: ClientMode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The anonymous principal reads only public entries, and its token writes once the setting is unset."""
    env = build_anonymous_env(
        backend='sqlite', database=str(access_db_path), port=free_port(), public_key=jwt_key_pair.public_key,
    )
    token = mint_anonymous_token(jwt_key_pair)
    with running_http_server(env) as base_url:
        async with server_database('sqlite', str(access_db_path), monkeypatch) as database:
            await verify_anonymous_arm(base_url, jwt_key_pair, client_mode, database, token)
    with running_http_server(without_read_only_principal(env)) as base_url:
        await verify_read_only_setting_fails_open(base_url, token, client_mode)


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_jwks_deployment_verifies_the_anonymous_token_with_its_static_key(
    access_db_path: Path,
    jwt_key_pair: RSAKeyPair,
    client_mode: ClientMode,
) -> None:
    """Beside an identity provider's JWKS keys, the operator's static key verifies the anonymous subject only."""
    anonymous_key_pair = RSAKeyPair.generate()
    with serve_jwks(jwt_key_pair, JWKS_KID) as jwks_uri:
        env = build_jwks_env(
            backend='sqlite',
            database=str(access_db_path),
            port=free_port(),
            jwks_uri=jwks_uri,
            anonymous_public_key=anonymous_key_pair.public_key,
        )
        with running_http_server(env) as base_url:
            await verify_jwks_composition(base_url, jwt_key_pair, anonymous_key_pair, client_mode)
