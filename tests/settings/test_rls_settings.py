"""Tests for the PostgreSQL row-level security settings.

Covers the two ``AccessControlSettings`` fields (``ENABLE_POSTGRESQL_RLS`` and
``POSTGRESQL_RLS_POLICY_ROLE`` with its role-name grammar), the two
``StorageSettings`` migration credentials, the ``AppSettings`` validator that
refuses an unusable row-level security configuration, and the server.json
entries of the four variables.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from pydantic import ValidationError

from app.settings import AppSettings
from app.settings.auth import AccessControlSettings
from app.settings.storage import StorageSettings

SERVER_JSON_PATH = Path(__file__).resolve().parents[2] / 'server.json'

RLS_ENV_KEYS = (
    'ENABLE_POSTGRESQL_RLS',
    'POSTGRESQL_RLS_POLICY_ROLE',
    'POSTGRESQL_MIGRATION_USER',
    'POSTGRESQL_MIGRATION_PASSWORD',
)

SQLITE_MESSAGE = 'ENABLE_POSTGRESQL_RLS requires STORAGE_BACKEND=postgresql'
MIGRATION_USER_MESSAGE = (
    'Row-level security needs a separate migration role that owns the tables: '
    'set POSTGRESQL_MIGRATION_USER and POSTGRESQL_MIGRATION_PASSWORD'
)
IDENTICAL_ROLES_MESSAGE = 'POSTGRESQL_MIGRATION_USER and POSTGRESQL_USER name the same PostgreSQL role'


@pytest.fixture(autouse=True)
def clear_rls_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the four row-level security variables so each test starts from the defaults."""
    for key in RLS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _storage(**fields: Any) -> StorageSettings:
    """Build storage settings in the component form unless a test passes a connection string.

    Every field a validator reads is passed explicitly, so the shell's own
    ``POSTGRESQL_*`` routing cannot reach the test.

    Returns:
        The storage settings with the given fields over the component-form base.
    """
    base: dict[str, Any] = {
        'STORAGE_BACKEND': 'postgresql',
        'POSTGRESQL_CONNECTION_STRING': None,
        'POSTGRESQL_USER': 'mcp_app',
        'POSTGRESQL_MIGRATION_USER': 'mcp_owner',
        'POSTGRESQL_MIGRATION_PASSWORD': SecretStr('owner-secret'),
    }
    base.update(fields)
    return StorageSettings(**base)


def _app(*, rls: bool, policy_role: str = 'mcp_context_request', **storage_fields: Any) -> AppSettings:
    """Build application settings with an explicit storage and access-control configuration.

    Returns:
        The validated application settings.
    """
    return AppSettings(
        storage=_storage(**storage_fields),
        access_control=AccessControlSettings(
            ENABLE_POSTGRESQL_RLS=rls,
            POSTGRESQL_RLS_POLICY_ROLE=policy_role,
        ),
    )


class TestRlsSettingsDefaults:
    """Defaults and environment aliases of the four variables."""

    def test_access_control_defaults(self) -> None:
        """Row-level security is off and the policy role is mcp_context_request by default."""
        settings = AccessControlSettings()
        assert settings.postgresql_rls_enabled is False
        assert settings.postgresql_rls_policy_role == 'mcp_context_request'

    def test_migration_credentials_default_to_none(self) -> None:
        """Both migration credentials are unset by default, so the request credentials apply."""
        settings = StorageSettings()
        assert settings.postgresql_migration_user is None
        assert settings.postgresql_migration_password is None

    def test_access_control_env_aliases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENABLE_POSTGRESQL_RLS and POSTGRESQL_RLS_POLICY_ROLE bind to the access-control fields."""
        monkeypatch.setenv('ENABLE_POSTGRESQL_RLS', 'true')
        monkeypatch.setenv('POSTGRESQL_RLS_POLICY_ROLE', 'kb_request')
        settings = AccessControlSettings()
        assert settings.postgresql_rls_enabled is True
        assert settings.postgresql_rls_policy_role == 'kb_request'

    def test_storage_env_aliases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The migration credentials bind to the storage fields and the password stays masked."""
        monkeypatch.setenv('POSTGRESQL_MIGRATION_USER', 'mcp_owner')
        monkeypatch.setenv('POSTGRESQL_MIGRATION_PASSWORD', 'owner-secret')
        settings = StorageSettings()
        assert settings.postgresql_migration_user == 'mcp_owner'
        password = settings.postgresql_migration_password
        assert password is not None
        assert password.get_secret_value() == 'owner-secret'
        assert 'owner-secret' not in str(password)
        assert 'owner-secret' not in repr(settings)

    @pytest.mark.parametrize('value', ['1', 'yes', 'on', 'TRUE'])
    def test_enable_flag_accepts_boolean_spellings(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The flag parses the boolean spellings pydantic accepts."""
        monkeypatch.setenv('ENABLE_POSTGRESQL_RLS', value)
        assert AccessControlSettings().postgresql_rls_enabled is True


class TestPolicyRoleGrammar:
    """POSTGRESQL_RLS_POLICY_ROLE accepts only a lowercase unquoted PostgreSQL identifier."""

    @pytest.mark.parametrize('role', ['mcp_context_request', '_request', 'a', 'role_2', 'r' * 63])
    def test_valid_role_names(self, role: str) -> None:
        """Lowercase letters, digits and underscores, starting with a letter or underscore, up to 63 characters."""
        assert AccessControlSettings(POSTGRESQL_RLS_POLICY_ROLE=role).postgresql_rls_policy_role == role

    @pytest.mark.parametrize(
        'role',
        [
            '',
            'Mcp_request',
            '1request',
            'r' * 64,
            'mcp-request',
            'mcp request',
            "mcp'request",
            'mcp"request',
            'schema.role',
            'mcp_request\n',
        ],
        ids=['empty', 'uppercase', 'leading-digit', 'too-long', 'hyphen', 'space', 'single-quote',
             'double-quote', 'dot', 'trailing-newline'],
    )
    def test_invalid_role_names_rejected(self, role: str) -> None:
        """Anything outside the grammar fails validation."""
        with pytest.raises(ValidationError, match='POSTGRESQL_RLS_POLICY_ROLE'):
            AccessControlSettings(POSTGRESQL_RLS_POLICY_ROLE=role)

    def test_invalid_role_from_env_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The grammar applies to the environment variable as well."""
        monkeypatch.setenv('POSTGRESQL_RLS_POLICY_ROLE', 'Request-Role')
        with pytest.raises(ValidationError, match='POSTGRESQL_RLS_POLICY_ROLE'):
            AccessControlSettings()


class TestRlsConfigurationValidator:
    """The AppSettings validator that refuses an unusable row-level security configuration."""

    def test_rls_with_sqlite_rejected(self) -> None:
        """Row-level security is a PostgreSQL feature, so the SQLite backend refuses it."""
        with pytest.raises(ValidationError, match=SQLITE_MESSAGE):
            _app(rls=True, STORAGE_BACKEND='sqlite')

    def test_rls_with_sqlite_rejected_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The environment variables reach the validator through the default settings factories."""
        monkeypatch.setenv('STORAGE_BACKEND', 'sqlite')
        monkeypatch.setenv('ENABLE_POSTGRESQL_RLS', 'true')
        monkeypatch.setenv('POSTGRESQL_MIGRATION_USER', 'mcp_owner')
        with pytest.raises(ValidationError, match=SQLITE_MESSAGE):
            AppSettings()

    @pytest.mark.parametrize('migration_user', [None, ''], ids=['unset', 'empty'])
    def test_rls_without_migration_user_rejected(self, migration_user: str | None) -> None:
        """Row-level security needs a migration role that owns the tables."""
        with pytest.raises(ValidationError, match=MIGRATION_USER_MESSAGE):
            _app(rls=True, POSTGRESQL_MIGRATION_USER=migration_user)

    def test_rls_without_migration_user_rejected_in_dsn_form(self) -> None:
        """The migration user is required with a connection string too."""
        with pytest.raises(ValidationError, match=MIGRATION_USER_MESSAGE):
            _app(
                rls=True,
                POSTGRESQL_CONNECTION_STRING=SecretStr('postgresql://mcp_app:pw@db:5432/mcp_context'),
                POSTGRESQL_MIGRATION_USER=None,
            )

    @pytest.mark.parametrize('connection_string', [None, SecretStr('')], ids=['unset', 'empty'])
    def test_identical_configured_roles_rejected_in_component_form(self, connection_string: SecretStr | None) -> None:
        """The same configured user for requests and migrations would leave the owner as the request role.

        An empty connection string selects the component form, as it does when the backend builds its DSN.
        """
        with pytest.raises(ValidationError, match=IDENTICAL_ROLES_MESSAGE) as excinfo:
            _app(
                rls=True,
                POSTGRESQL_CONNECTION_STRING=connection_string,
                POSTGRESQL_USER='postgres',
                POSTGRESQL_MIGRATION_USER='postgres',
            )
        message = str(excinfo.value)
        assert "'postgres'" in message
        assert 'CREATE ROLE "mcp_context_request" NOLOGIN;' in message
        assert 'GRANT "mcp_context_request" TO <login>;' in message

    def test_identical_roles_message_names_configured_policy_role(self) -> None:
        """The remediation statements name the configured policy role."""
        with pytest.raises(ValidationError, match='CREATE ROLE "kb_request" NOLOGIN;'):
            _app(rls=True, policy_role='kb_request', POSTGRESQL_USER='kb', POSTGRESQL_MIGRATION_USER='kb')

    def test_identical_names_accepted_in_dsn_form(self) -> None:
        """With a connection string the request user lives in the secret, so only the runtime check decides."""
        settings = _app(
            rls=True,
            POSTGRESQL_CONNECTION_STRING=SecretStr('postgresql://mcp_app:pw@db:5432/mcp_context'),
            POSTGRESQL_USER='postgres',
            POSTGRESQL_MIGRATION_USER='postgres',
        )
        assert settings.access_control.postgresql_rls_enabled is True
        assert settings.storage.postgresql_migration_user == 'postgres'

    def test_split_roles_accepted(self) -> None:
        """Distinct request and migration users on PostgreSQL pass."""
        settings = _app(rls=True, POSTGRESQL_USER='mcp_app', POSTGRESQL_MIGRATION_USER='mcp_owner')
        assert settings.access_control.postgresql_rls_enabled is True
        assert settings.storage.postgresql_user == 'mcp_app'
        assert settings.storage.postgresql_migration_user == 'mcp_owner'

    def test_migration_password_is_optional(self) -> None:
        """An unset migration password keeps the request password, so it is not required."""
        settings = _app(rls=True, POSTGRESQL_MIGRATION_PASSWORD=None)
        assert settings.storage.postgresql_migration_password is None

    @pytest.mark.parametrize(
        'storage_fields',
        [
            {'STORAGE_BACKEND': 'sqlite'},
            {'POSTGRESQL_MIGRATION_USER': None},
            {'POSTGRESQL_USER': 'postgres', 'POSTGRESQL_MIGRATION_USER': 'postgres'},
        ],
        ids=['sqlite', 'no-migration-user', 'identical-users'],
    )
    def test_rls_off_validates_nothing(self, storage_fields: dict[str, Any]) -> None:
        """With row-level security off none of the three refusals applies."""
        settings = _app(rls=False, **storage_fields)
        assert settings.access_control.postgresql_rls_enabled is False


class TestRlsServerJson:
    """server.json describes the four variables identically in every package."""

    @staticmethod
    def _entries(name: str) -> list[dict[str, Any]]:
        server_json = json.loads(SERVER_JSON_PATH.read_text(encoding='utf-8'))
        return [
            env
            for package in server_json['packages']
            for env in package['environmentVariables']
            if env['name'] == name
        ]

    @pytest.mark.parametrize(
        ('name', 'is_secret', 'value_format', 'default'),
        [
            ('ENABLE_POSTGRESQL_RLS', False, 'boolean', 'false'),
            ('POSTGRESQL_RLS_POLICY_ROLE', False, 'string', 'mcp_context_request'),
            ('POSTGRESQL_MIGRATION_USER', False, 'string', None),
            ('POSTGRESQL_MIGRATION_PASSWORD', True, 'string', None),
        ],
    )
    def test_entry_in_every_package(self, name: str, is_secret: bool, value_format: str, default: str | None) -> None:
        """Each variable appears once per package with its secrecy, format and default."""
        server_json = json.loads(SERVER_JSON_PATH.read_text(encoding='utf-8'))
        entries = self._entries(name)
        assert len(entries) == len(server_json['packages'])
        for entry in entries:
            assert entry['isSecret'] is is_secret
            assert entry['isRequired'] is False
            assert entry['format'] == value_format
            assert entry.get('default') == default
