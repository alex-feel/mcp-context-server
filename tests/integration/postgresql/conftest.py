"""Docker-compose-based PostgreSQL fixture for integration tests.

The fixture brings up a single pgvector container (image matches the
production deployment) on an ephemeral port, waits for ``pg_isready``,
yields the connection string, then tears the container down with
``docker compose down -v``. When the docker CLI or daemon is unavailable
the fixture skips cleanly so the test session keeps running on machines
without Docker.

Credentials (``mcp_test:mcp_test``) are throwaway: the container exposes
no listening interface beyond loopback for the test session lifetime and
its data directory is mounted on tmpfs.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import socket
import subprocess
import time
import warnings
from collections.abc import AsyncIterator
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import pytest
import pytest_asyncio

COMPOSE_FILE = Path(__file__).parent / 'docker-compose.test.yml'
PROJECT_NAME = 'mcp-context-server-test-pg'

_NON_DEFAULT_SCHEMA = 'mcp_test'
NON_DEFAULT_SCHEMA = _NON_DEFAULT_SCHEMA  # public re-export for tests


def _force_cleanup() -> None:
    """Best-effort idempotent removal of the postgres test container and volumes.

    Skips silently if docker CLI is not on PATH (CI environments without
    docker, fresh developer machines, etc.). Catches all subprocess errors
    so the helper can be safely chained from atexit, sessionfinish, and
    fixture pre-flight without raising. Stderr from a failing teardown is
    surfaced through warnings.warn so silent failures do not accumulate.
    """
    if shutil.which('docker') is None:
        return
    try:
        result = subprocess.run(
            [
                'docker', 'compose',
                '-p', PROJECT_NAME,
                '-f', str(COMPOSE_FILE),
                'down', '-v', '--remove-orphans',
            ],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        warnings.warn(
            f'docker compose down for {PROJECT_NAME} did not complete: {exc!r}',
            stacklevel=2,
        )
        return
    if result.returncode != 0:
        warnings.warn(
            f'docker compose down for {PROJECT_NAME} returned '
            f'exit code {result.returncode}; stderr: {result.stderr.strip()}',
            stacklevel=2,
        )


# Belt-and-suspenders cleanup at interpreter shutdown. The primary safety
# nets are the fixture-level try/finally and pytest_sessionfinish; this
# atexit hook adds redundancy for edge cases (interpreter abnormal exit
# paths that bypass pytest's own shutdown). atexit is not invoked on
# Windows for Ctrl-C delivered via the Win32 console control handler, so
# this layer cannot substitute for pytest_sessionfinish on Windows.
atexit.register(_force_cleanup)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Run idempotent docker cleanup after the full pytest session.

    Called by pytest after all tests complete and before pytest returns
    its exit code. This hook runs even when session-scoped fixture
    finalizers are skipped, such as KeyboardInterrupt during fixture
    setup or pytest.fail before the yield. See pytest issue 4517.

    Args:
        session: The pytest Session object (unused but required by hookspec).
        exitstatus: The exit status pytest will return (unused but required).
    """
    del session, exitstatus  # ARG001: required by pytest hookspec
    _force_cleanup()


def _docker_available() -> bool:
    """Return True when the docker CLI is on PATH and the daemon answers."""
    if shutil.which('docker') is None:
        return False
    try:
        result = subprocess.run(
            ['docker', 'info'], capture_output=True, timeout=5, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _free_port() -> int:
    """Return an unused TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def _wait_for_health(timeout_s: float = 60.0) -> None:
    """Poll ``docker compose ps`` until the postgres-test container is healthy.

    Args:
        timeout_s: Maximum wait time in seconds.

    Raises:
        TimeoutError: When the container does not become healthy in time.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = subprocess.run(
            [
                'docker', 'compose',
                '-p', PROJECT_NAME,
                '-f', str(COMPOSE_FILE),
                'ps', '--format', '{{.Health}}',
            ],
            capture_output=True, text=True, check=False,
        )
        if 'healthy' in result.stdout:
            return
        time.sleep(1.0)
    raise TimeoutError(
        f'PostgreSQL container did not become healthy within {timeout_s}s',
    )


@pytest.fixture(scope='session')
def pg_test_url() -> Iterator[str]:
    """Session-scoped fixture providing a pgvector connection string.

    Skips the test when Docker is unavailable.

    Performs a pre-flight cleanup of any stale container left over from a
    prior aborted run before bringing up a fresh one, and runs a hardened
    teardown (timeout + visible stderr + remove-orphans) at session end.

    Yields:
        Connection string pointing at a fresh pgvector container on an
        ephemeral loopback port.
    """
    if not _docker_available():
        pytest.skip(
            'Docker CLI is not available or the Docker daemon is not '
            'running; PostgreSQL integration tests require docker-compose',
        )

    # Pre-flight cleanup: remove any stale container/volume from a prior
    # aborted run so the new bring-up starts from a known empty state.
    _force_cleanup()

    port = _free_port()
    env = {**os.environ, 'MCP_PG_TEST_PORT': str(port)}

    up_result = subprocess.run(
        [
            'docker', 'compose',
            '-p', PROJECT_NAME,
            '-f', str(COMPOSE_FILE),
            'up', '-d', '--wait',
        ],
        env=env, capture_output=True, text=True, check=False,
    )
    if up_result.returncode != 0:
        pytest.fail(
            'docker compose up failed:\n'
            f'stdout: {up_result.stdout}\nstderr: {up_result.stderr}',
        )

    try:
        _wait_for_health()
        yield f'postgresql://mcp_test:mcp_test@localhost:{port}/mcp_test'
    finally:
        _force_cleanup()


def _replace_db_name_non_default(pg_url: str, new_db: str) -> str:
    """Return ``pg_url`` with the database name replaced by ``new_db``."""
    parts = urlsplit(pg_url)
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        f'/{new_db}',
        parts.query,
        parts.fragment,
    ))


@pytest_asyncio.fixture
async def pg_non_default_schema_db(pg_test_url: str) -> AsyncIterator[str]:
    """Provision an isolated PG database with POSTGRESQL_SCHEMA=mcp_test.

    The fixture creates a fresh database, installs pgvector inside it,
    creates the ``mcp_test`` schema, and yields the connection string.
    Tests that consume this fixture are expected to set ``search_path``
    on every connection they create (helper available via
    ``apply_non_default_search_path``); the production
    ``PostgreSQLBackend`` does not yet install a per-connection
    ``init=`` hook for ``search_path``.

    The database is dropped at teardown so the fixture is self-cleaning
    and does not interact with the singleton ``compression_metadata``
    invariant from other tests.

    Yields:
        Connection string pointing at the per-test isolated database
        where the ``mcp_test`` schema has been pre-created.
    """
    db_name = 'mcp_non_default_schema_e2e'
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {db_name}')
        await admin.execute(f'CREATE DATABASE {db_name}')
    finally:
        await admin.close()

    target_url = _replace_db_name_non_default(pg_test_url, db_name)

    setup = await asyncpg.connect(target_url)
    try:
        await setup.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await setup.execute(
            f'CREATE SCHEMA IF NOT EXISTS {_NON_DEFAULT_SCHEMA}',
        )
    finally:
        await setup.close()

    yield target_url

    admin = await asyncpg.connect(pg_test_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {db_name}')
    finally:
        await admin.close()


async def apply_non_default_search_path(conn: asyncpg.Connection) -> None:
    """Set ``search_path`` on ``conn`` to the non-default schema first.

    Helper for tests using ``pg_non_default_schema_db``. Production
    code may eventually install this via an ``asyncpg.create_pool``
    ``init=`` hook; this helper exists to make the operator's
    ``search_path`` contract explicit at test boundaries until that
    upgrade lands.
    """
    await conn.execute(
        f'SET search_path = {_NON_DEFAULT_SCHEMA}, public',
    )
