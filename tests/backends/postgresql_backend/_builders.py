"""Backend builders shared by the PostgreSQL backend test modules."""

from dataclasses import dataclass

import pytest
from pydantic import SecretStr

import app.backends.postgresql_backend as pg_module
from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.postgresql_backend.session import NO_CREDENTIAL_OVERRIDE
from app.backends.postgresql_backend.session import CredentialOverride
from app.backends.postgresql_backend.session_profiles import MaintenanceProfile
from app.backends.postgresql_backend.session_profiles import SessionProfile
from app.backends.postgresql_backend.session_profiles import UnboundProfile
from app.settings import AppSettings
from tests.helpers import rebind_package_settings


def build_backend(connection_string: str = 'postgresql://postgres:postgres@localhost:5432/testdb') -> PostgreSQLBackend:
    """Build a non-shut-down backend without a pool.

    Args:
        connection_string: DSN handed to the backend.

    Returns:
        The constructed backend.
    """
    backend = PostgreSQLBackend(
        connection_string=connection_string,
        profile=UnboundProfile(),
        credentials=NO_CREDENTIAL_OVERRIDE,
    )
    backend._shutdown = False
    return backend


def fast_retries(backend: PostgreSQLBackend) -> None:
    """Make the write retry loop deterministic and sleepless.

    Args:
        backend: The backend whose retry configuration is tightened.
    """
    backend.retry_config.max_retries = 3
    backend.retry_config.base_delay = 0.0
    backend.retry_config.max_delay = 0.0
    backend.retry_config.jitter = False


@dataclass(frozen=True, slots=True)
class LoginForm:
    """One way a backend obtains its login, with the remediation a refused login must name.

    Attributes:
        connection_string: The DSN handed to the backend, or None for the request settings.
        profile: The session profile of the backend.
        credentials: The credential override of the backend.
        refused_login: The remediation sentence of a refused login (SQLSTATE class 28).
        refused_connect: The remediation sentence of a refused CONNECT (SQLSTATE 42501).
    """

    connection_string: str | None
    profile: SessionProfile
    credentials: CredentialOverride
    refused_login: str
    refused_connect: str


LOGIN_FORMS = [
    pytest.param(
        LoginForm(
            connection_string=None,
            profile=MaintenanceProfile(),
            credentials=CredentialOverride(user='mcp_owner', password=SecretStr('x')),
            refused_login='Check POSTGRESQL_MIGRATION_USER, POSTGRESQL_MIGRATION_PASSWORD and the server pg_hba.conf rules.',
            refused_connect='Grant the configured POSTGRESQL_MIGRATION_USER access to POSTGRESQL_DATABASE.',
        ),
        id='migration-credentials',
    ),
    pytest.param(
        LoginForm(
            connection_string='postgresql://mcp_owner:x@localhost:5432/testdb',
            profile=MaintenanceProfile(),
            credentials=NO_CREDENTIAL_OVERRIDE,
            refused_login='Check the user and password of the connection string and the server pg_hba.conf rules.',
            refused_connect='Grant the user of the connection string access to its database.',
        ),
        id='explicit-connection-string',
    ),
    pytest.param(
        LoginForm(
            connection_string=None,
            profile=UnboundProfile(),
            credentials=NO_CREDENTIAL_OVERRIDE,
            refused_login='Check POSTGRESQL_USER, POSTGRESQL_PASSWORD and the server pg_hba.conf rules.',
            refused_connect='Grant the configured POSTGRESQL_USER access to POSTGRESQL_DATABASE.',
        ),
        id='request-settings',
    ),
]


def build_login_backend(monkeypatch: pytest.MonkeyPatch, form: LoginForm) -> PostgreSQLBackend:
    """Build a backend that logs in the way ``form`` describes.

    A backend built without a connection string dials the request settings, so the
    package is rebound to settings whose request fields are fixed here rather than read
    from the environment the tests run in.

    Args:
        monkeypatch: Fixture that restores the environment and the settings bindings.
        form: The login form to build.

    Returns:
        The constructed backend.
    """
    monkeypatch.delenv('POSTGRESQL_CONNECTION_STRING', raising=False)
    for name, value in (
        ('POSTGRESQL_HOST', 'localhost'),
        ('POSTGRESQL_USER', 'mcp_app'),
        ('POSTGRESQL_PASSWORD', 'app-secret'),
        ('POSTGRESQL_DATABASE', 'testdb'),
    ):
        monkeypatch.setenv(name, value)
    rebind_package_settings(monkeypatch, pg_module, AppSettings())
    return PostgreSQLBackend(
        connection_string=form.connection_string,
        profile=form.profile,
        credentials=form.credentials,
    )
