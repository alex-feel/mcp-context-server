"""Tests for the shared test helpers in tests/helpers.py.

Covers ``postgresql_subprocess_env``: a server or command-line subprocess inherits the
parent environment for its provider configuration, while every PostgreSQL routing input
of the parent (each ``POSTGRESQL_*`` variable and ``ENABLE_POSTGRESQL_RLS``) is removed
before the helper pins the routing, the row-level security credentials when they are
given, and the caller's overrides.
"""

import pytest

from tests.helpers import RlsCredentials
from tests.helpers import postgresql_subprocess_env

TEST_URL = 'postgresql://mcp_app:mcp_app@localhost:5433/mcp_rls'
RLS = RlsCredentials(migration_user='mcp_owner', migration_password='mcp_owner', policy_role='mcp_context_request')
RLS_KEYS = (
    'ENABLE_POSTGRESQL_RLS', 'POSTGRESQL_RLS_POLICY_ROLE', 'POSTGRESQL_MIGRATION_USER', 'POSTGRESQL_MIGRATION_PASSWORD',
)


@pytest.fixture
def supabase_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parent environment routed to a Supabase database, row-level security credentials included."""
    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('POSTGRESQL_HOST', 'aws-0-eu-central-1.pooler.supabase.com')
    monkeypatch.setenv('POSTGRESQL_USER', 'postgres.projectref')
    monkeypatch.setenv('POSTGRESQL_PASSWORD', 'supabase-secret')
    monkeypatch.setenv('POSTGRESQL_SCHEMA', 'tenant')
    monkeypatch.setenv('POSTGRESQL_MIGRATION_USER', 'supabase_owner')
    monkeypatch.setenv('POSTGRESQL_RLS_POLICY_ROLE', 'supabase_requests')
    monkeypatch.setenv('ENABLE_POSTGRESQL_RLS', 'true')
    monkeypatch.setenv('EMBEDDING_MODEL', 'all-minilm')
    monkeypatch.setenv('DISABLED_TOOLS', 'delete_context')


@pytest.mark.usefixtures('supabase_parent')
class TestPostgresqlSubprocessEnv:
    """The parent's PostgreSQL routing never reaches the subprocess; its provider configuration does."""

    def test_only_the_connection_string_survives_of_the_postgresql_family(self) -> None:
        env = postgresql_subprocess_env(TEST_URL)

        assert {key for key in env if key.upper().startswith('POSTGRESQL_')} == {'POSTGRESQL_CONNECTION_STRING'}
        assert env['POSTGRESQL_CONNECTION_STRING'] == TEST_URL
        assert env['STORAGE_BACKEND'] == 'postgresql'
        assert env['MCP_TEST_MODE'] == '1'
        assert env['DISABLED_TOOLS'] == ''

    def test_provider_configuration_is_inherited(self) -> None:
        assert postgresql_subprocess_env(TEST_URL)['EMBEDDING_MODEL'] == 'all-minilm'

    def test_row_level_security_is_absent_unless_credentials_are_given(self) -> None:
        env = postgresql_subprocess_env(TEST_URL)

        assert not set(RLS_KEYS) & set(env)

    def test_credentials_set_exactly_the_row_level_security_keys(self) -> None:
        env = postgresql_subprocess_env(TEST_URL, rls=RLS)
        without = postgresql_subprocess_env(TEST_URL)

        assert {key: env[key] for key in RLS_KEYS} == {
            'ENABLE_POSTGRESQL_RLS': 'true',
            'POSTGRESQL_RLS_POLICY_ROLE': 'mcp_context_request',
            'POSTGRESQL_MIGRATION_USER': 'mcp_owner',
            'POSTGRESQL_MIGRATION_PASSWORD': 'mcp_owner',
        }
        assert {key: value for key, value in env.items() if key not in RLS_KEYS} == without

    def test_overrides_come_last(self) -> None:
        env = postgresql_subprocess_env(TEST_URL, rls=RLS, ENABLE_POSTGRESQL_RLS='false', LOG_LEVEL='INFO')

        assert env['ENABLE_POSTGRESQL_RLS'] == 'false'
        assert env['LOG_LEVEL'] == 'INFO'
        assert env['POSTGRESQL_MIGRATION_USER'] == 'mcp_owner'
