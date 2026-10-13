"""Structural tests for the PostgreSQL row-level security wiring in the Docker Compose files.

Two postures are pinned. The bundled PostgreSQL stacks keep a single role with
row-level security off, and their sidecar bootstraps its superuser from the
migration credentials, falling back to the request credentials, so an operator who
splits the roles keeps the table owner apart from the login role while every
default stays what it is without those variables. The :8002 live test stack runs
with row-level security on: a one-shot ``postgres-roles`` service creates the
policy role and the request login role, and the app serves requests as that login
role while migrations run as the sidecar superuser that owns the tables.

The files are parsed with a real YAML parser. The interpolation behavior of the
nested defaults is checked through ``docker compose config`` on a minimal file
holding only the sidecar environment, so the developer's ``deploy/docker/.env``
and shell credentials never reach it; that check is skipped where Docker Compose
is unavailable.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_DOCKER_DIR = Path(__file__).resolve().parents[2] / 'deploy' / 'docker'
_TEST_STACK = _DOCKER_DIR / 'docker-compose.postgresql.ollama.test.yml'
_SQLITE_TEST_STACK = _DOCKER_DIR / 'docker-compose.sqlite.ollama.test.yml'
_BUNDLED_NAMES = (
    'docker-compose.postgresql.ollama.yml',
    'docker-compose.postgresql.ollama.local.yml',
    'docker-compose.postgresql.ollama-openai.yml',
    'docker-compose.postgresql.ollama-openai.local.yml',
    'docker-compose.postgresql.openai.yml',
)
_BUNDLED_FILES = tuple(_DOCKER_DIR / name for name in _BUNDLED_NAMES)
_EXTERNAL_FILES = tuple(sorted(_DOCKER_DIR.glob('docker-compose.postgresql-external.*.yml')))

_SIDECAR_USER = '${POSTGRESQL_MIGRATION_USER:-${POSTGRESQL_USER:-postgres}}'
_SIDECAR_PASSWORD = '${POSTGRESQL_MIGRATION_PASSWORD:-${POSTGRESQL_PASSWORD:-postgres}}'
_RLS_ONLY_KEYS = (
    'ENABLE_POSTGRESQL_RLS',
    'POSTGRESQL_RLS_POLICY_ROLE',
    'POSTGRESQL_MIGRATION_USER',
    'POSTGRESQL_MIGRATION_PASSWORD',
)


def _load(path: Path) -> dict[str, Any]:
    """Parse one compose file.

    Args:
        path: The compose file.

    Returns:
        The parsed document.
    """
    document = yaml.safe_load(path.read_text(encoding='utf-8'))
    assert isinstance(document, dict), f'{path.name} is not a mapping'
    return document


def _service(path: Path, name: str) -> dict[str, Any]:
    """Return one service definition of a compose file.

    Args:
        path: The compose file.
        name: The service name.

    Returns:
        The service mapping.
    """
    services = _load(path)['services']
    assert name in services, f'{path.name} has no {name!r} service'
    service = services[name]
    assert isinstance(service, dict)
    return service


def _environment(service: dict[str, Any]) -> dict[str, str]:
    """Normalize a service environment, list or mapping form, to a mapping.

    Args:
        service: The service mapping.

    Returns:
        The environment as name to raw (uninterpolated) value.
    """
    raw = service.get('environment') or {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    pairs = (str(item).split('=', 1) for item in raw)
    return {pair[0]: pair[1] if len(pair) == 2 else '' for pair in pairs}


def _header(path: Path) -> str:
    """Return the leading comment block of a compose file as one line of text.

    Args:
        path: The compose file.

    Returns:
        The comment text with the comment markers removed and whitespace collapsed.
    """
    lines: list[str] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.startswith('#'):
            break
        lines.append(line.lstrip('#').strip())
    return ' '.join(' '.join(lines).split())


def _compose_unescape(text: str) -> str:
    """Apply Docker Compose's escape rule for a literal dollar sign.

    Args:
        text: A raw value from the compose file.

    Returns:
        The value as the container receives it.
    """
    return text.replace('$$', '$')


def test_bundled_postgresql_stacks_are_exactly_the_five_sidecar_files() -> None:
    """Every bundled PostgreSQL stack with a sidecar is covered by the binding checks."""
    found = {
        path.name
        for path in _DOCKER_DIR.glob('docker-compose.postgresql.*.yml')
        if not path.name.endswith('.test.yml')
    }
    assert found == set(_BUNDLED_NAMES)
    for path in _EXTERNAL_FILES:
        assert 'postgres' not in _load(path)['services'], f'{path.name} unexpectedly carries a sidecar'


@pytest.mark.parametrize('path', _BUNDLED_FILES, ids=_BUNDLED_NAMES)
def test_bundled_sidecar_bootstraps_from_the_migration_credentials(path: Path) -> None:
    """The sidecar superuser comes from the migration credentials, then the request ones."""
    environment = _environment(_service(path, 'postgres'))
    assert environment['POSTGRES_USER'] == _SIDECAR_USER
    assert environment['POSTGRES_PASSWORD'] == _SIDECAR_PASSWORD
    assert environment['POSTGRES_DB'] == '${POSTGRESQL_DATABASE:-mcp_context}'


@pytest.mark.parametrize('path', _BUNDLED_FILES, ids=_BUNDLED_NAMES)
def test_bundled_app_environment_stays_single_role_with_rls_off(path: Path) -> None:
    """The bundled app keeps its request credentials and sets no row-level security variable."""
    environment = _environment(_service(path, 'mcp-context-server'))
    assert environment['POSTGRESQL_USER'] == '${POSTGRESQL_USER:-postgres}'
    assert environment['POSTGRESQL_PASSWORD'] == '${POSTGRESQL_PASSWORD:-postgres}'
    for key in _RLS_ONLY_KEYS:
        assert key not in environment, f'{path.name} sets {key}'
    assert 'postgres-roles' not in _load(path)['services']


@pytest.mark.parametrize('path', _BUNDLED_FILES, ids=_BUNDLED_NAMES)
def test_bundled_header_points_to_the_enable_procedure(path: Path) -> None:
    """The header comment names the documentation section that enables row-level security."""
    header = _header(path)
    assert 'docs/deployment/docker.md' in header
    assert 'Enabling row-level security' in header


def test_test_stack_roles_service_is_a_one_shot_against_the_sidecar() -> None:
    """postgres-roles runs once, after the sidecar is healthy, with the sidecar credentials."""
    roles = _service(_TEST_STACK, 'postgres-roles')
    sidecar = _environment(_service(_TEST_STACK, 'postgres'))
    assert roles['image'] == _service(_TEST_STACK, 'postgres')['image'] == 'pgvector/pgvector:pg18-trixie'
    assert roles['restart'] == 'no'
    assert roles['depends_on'] == {'postgres': {'condition': 'service_healthy'}}
    assert 'ports' not in roles
    assert _environment(roles) == {
        'PGHOST': 'postgres',
        'PGUSER': sidecar['POSTGRES_USER'],
        'PGPASSWORD': sidecar['POSTGRES_PASSWORD'],
        'PGDATABASE': sidecar['POSTGRES_DB'],
    }


def _roles_command() -> list[str]:
    """Return the postgres-roles command line as the container receives it.

    Returns:
        The entrypoint followed by the command, with Compose escapes applied.
    """
    roles = _service(_TEST_STACK, 'postgres-roles')
    parts: list[str] = []
    for key in ('entrypoint', 'command'):
        value = roles.get(key)
        if value is None:
            continue
        assert isinstance(value, list), f'postgres-roles {key} must be a list'
        parts.extend(str(item) for item in value)
    return [_compose_unescape(part) for part in parts]


def test_test_stack_roles_service_escapes_every_dollar_for_compose() -> None:
    """Every dollar sign in the raw command is doubled, so Compose interpolates none of them."""
    roles = _service(_TEST_STACK, 'postgres-roles')
    raw = ' '.join(str(item) for key in ('entrypoint', 'command') for item in roles.get(key) or [])
    assert '$' in raw
    for run in re.findall(r'\$+', raw):
        assert len(run) % 2 == 0, f'unescaped dollar run {run!r} in postgres-roles'


def test_test_stack_roles_service_creates_the_roles_idempotently() -> None:
    """psql stops on the first error and runs one DO block that creates and grants the roles."""
    command = _roles_command()
    assert command[0] == 'psql'
    assert command[1:4] == ['-v', 'ON_ERROR_STOP=1', '-c']
    assert len(command) == 5
    sql = ' '.join(command[4].split())
    assert sql.startswith('DO $$ BEGIN ')
    assert sql.endswith(' END $$;')
    assert (
        "IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mcp_context_request') "
        'THEN CREATE ROLE mcp_context_request NOLOGIN; END IF;'
    ) in sql
    assert (
        "IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mcp_app') "
        'THEN CREATE ROLE mcp_app LOGIN; END IF;'
    ) in sql
    assert "ALTER ROLE mcp_app LOGIN PASSWORD 'mcp_app' NOSUPERUSER NOBYPASSRLS;" in sql
    assert 'GRANT mcp_context_request TO mcp_app;' in sql


def test_test_stack_app_runs_as_the_login_role_with_rls_on() -> None:
    """The app environment is hardcoded to the login role, the owner and row-level security on."""
    app = _service(_TEST_STACK, 'mcp-context-server')
    environment = _environment(app)
    sidecar = _environment(_service(_TEST_STACK, 'postgres'))
    expected = {
        'POSTGRESQL_USER': 'mcp_app',
        'POSTGRESQL_PASSWORD': 'mcp_app',
        'POSTGRESQL_MIGRATION_USER': 'postgres',
        'POSTGRESQL_MIGRATION_PASSWORD': 'postgres',
        'ENABLE_POSTGRESQL_RLS': 'true',
        'POSTGRESQL_RLS_POLICY_ROLE': 'mcp_context_request',
        'POSTGRESQL_DATABASE': 'mcp_context',
    }
    for key, value in expected.items():
        assert environment[key] == value, key
    assert environment['POSTGRESQL_MIGRATION_USER'] == sidecar['POSTGRES_USER']
    assert environment['POSTGRESQL_MIGRATION_PASSWORD'] == sidecar['POSTGRES_PASSWORD']
    assert environment['POSTGRESQL_DATABASE'] == sidecar['POSTGRES_DB']
    assert environment['POSTGRESQL_USER'] != environment['POSTGRESQL_MIGRATION_USER']
    sql = ' '.join(_roles_command()[4].split())
    assert f"CREATE ROLE {environment['POSTGRESQL_RLS_POLICY_ROLE']} NOLOGIN;" in sql
    assert (
        f"ALTER ROLE {environment['POSTGRESQL_USER']} LOGIN PASSWORD '{environment['POSTGRESQL_PASSWORD']}'"
    ) in sql


def test_test_stack_app_waits_for_the_roles_service() -> None:
    """The app starts only after postgres-roles exited successfully."""
    depends_on = _service(_TEST_STACK, 'mcp-context-server')['depends_on']
    assert depends_on['postgres-roles'] == {'condition': 'service_completed_successfully'}
    assert depends_on['postgres'] == {'condition': 'service_healthy'}


def test_test_stack_stays_self_contained_and_unpublished() -> None:
    """No credential or row-level security value is interpolated, and only the app port is published."""
    services = _load(_TEST_STACK)['services']
    for name in ('mcp-context-server', 'postgres', 'postgres-roles'):
        environment = _environment(services[name])
        for key, value in environment.items():
            if key.startswith(('POSTGRES', 'PG', 'ENABLE_POSTGRESQL_RLS')):
                assert '${' not in value, f'{name} interpolates {key}'
    assert 'ports' not in services['postgres']
    assert services['mcp-context-server']['ports'] == ['8002:8002']


def test_sqlite_test_stack_sets_no_rls_variable() -> None:
    """The SQLite live test stack carries no PostgreSQL row-level security variable."""
    environment = _environment(_service(_SQLITE_TEST_STACK, 'mcp-context-server'))
    for key in _RLS_ONLY_KEYS:
        assert key not in environment


def _docker_compose_available() -> bool:
    """Report whether the Docker Compose plugin can run.

    Returns:
        True when ``docker compose version`` succeeds.
    """
    docker = shutil.which('docker')
    if docker is None:
        return False
    result = subprocess.run(
        [docker, 'compose', 'version'],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.returncode == 0


_NO_VARIABLES: dict[str, str] = {}
_REQUEST_ONLY = {'POSTGRESQL_USER': 'alice', 'POSTGRESQL_PASSWORD': 'alice-secret'}
_SPLIT_ROLES = {
    'POSTGRESQL_USER': 'mcp_app',
    'POSTGRESQL_PASSWORD': 'app-secret',
    'POSTGRESQL_MIGRATION_USER': 'owner',
    'POSTGRESQL_MIGRATION_PASSWORD': 'owner-secret',
}


@pytest.mark.skipif(not _docker_compose_available(), reason='docker compose not available')
@pytest.mark.parametrize('path', _BUNDLED_FILES, ids=_BUNDLED_NAMES)
@pytest.mark.parametrize(
    ('variables', 'expected'),
    [
        (_NO_VARIABLES, ('postgres', 'postgres')),
        (_REQUEST_ONLY, ('alice', 'alice-secret')),
        (_SPLIT_ROLES, ('owner', 'owner-secret')),
    ],
    ids=['defaults', 'request-credentials', 'split-roles'],
)
def test_bundled_sidecar_nested_defaults_resolve(
    path: Path,
    variables: dict[str, str],
    expected: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose resolves the sidecar credentials to migration, then request, then built-in values."""
    sidecar = {'image': 'postgres', 'environment': _service(path, 'postgres')['environment']}
    minimal = tmp_path / 'compose.yml'
    minimal.write_text(yaml.safe_dump({'name': 'rls-bindings', 'services': {'postgres': sidecar}}), encoding='utf-8')
    (tmp_path / 'empty.env').write_text('', encoding='utf-8')
    for key in [key for key in os.environ if key.startswith('POSTGRESQL_')]:
        monkeypatch.delenv(key)
    for key, value in variables.items():
        monkeypatch.setenv(key, value)
    docker = shutil.which('docker')
    assert docker is not None
    result = subprocess.run(
        [docker, 'compose', '--env-file', str(tmp_path / 'empty.env'), '-f', str(minimal), 'config', '--format', 'json'],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)['services']['postgres']['environment']
    assert (resolved['POSTGRES_USER'], resolved['POSTGRES_PASSWORD']) == expected
    assert resolved['POSTGRES_DB'] == 'mcp_context'
