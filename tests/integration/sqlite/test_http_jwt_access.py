"""JWT principals reach only the entries their ownership and grants allow (SQLite, over HTTP).

Runs the access scenario of :mod:`tests.integration._http_jwt` against a real HTTP server
verifying minted JWTs, once per protocol era. The server serves a temporary SQLite file;
the test opens the same file to insert the user and write grants no tool writes. The same
scenario runs against PostgreSQL in ``tests/integration/postgresql/test_http_jwt_access.py``.
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
from tests.integration._http_jwt import server_database
from tests.integration._http_jwt import verify_jwt_access_scoping

pytestmark = pytest.mark.integration


@pytest.fixture
def access_db_path(tmp_path: Path) -> Path:
    """Return the database file the access server serves.

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group, user-read and user-write grants decide what bob and carol read, update and delete."""
    async with server_database('sqlite', str(access_db_path), monkeypatch) as database:
        await verify_jwt_access_scoping(access_server, jwt_key_pair, client_mode, database)
