"""Tests for app/settings/auth.py.

Covers AuthSettings (provider selection, the bearer token and client id, the JWT
fields) and AccessControlSettings (defaults, env-var aliases, the enum constraints
on visibility and group-grant policy, and the DDL-safe character-set validation
on the default principal).
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.settings import get_settings
from app.settings.auth import AccessControlSettings
from app.settings.auth import AuthSettings
from tests.helpers import env_var


class TestAuthProviderSetting:
    """Tests for MCP_AUTH_PROVIDER setting in AuthSettings."""

    def test_auth_provider_default_is_none(self) -> None:
        """AuthSettings provider should default to 'none'."""
        from app.settings.auth import AuthSettings

        settings = AuthSettings()
        assert settings.provider == 'none'

    def test_auth_provider_from_env(self) -> None:
        """AuthSettings should load MCP_AUTH_PROVIDER from environment."""
        from app.settings.auth import AuthSettings

        with env_var('MCP_AUTH_PROVIDER', 'simple_token'):
            settings = AuthSettings()
            assert settings.provider == 'simple_token'

    def test_auth_provider_invalid_value(self) -> None:
        """AuthSettings should reject invalid provider values."""
        from app.settings.auth import AuthSettings

        with env_var('MCP_AUTH_PROVIDER', 'invalid'), pytest.raises(ValidationError):
            AuthSettings()


class TestAccessControlSettings:
    """AccessControlSettings defaults and validation."""

    def test_defaults(self) -> None:
        """The documented defaults apply without any env configuration."""
        settings = AccessControlSettings()
        assert settings.default_principal == 'local'
        assert settings.default_visibility == 'private'
        assert settings.default_group_grants == 'none'
        assert settings.publish_role is None

    def test_env_aliases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each field binds to its ACCESS_CONTROL_* env var."""
        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_PRINCIPAL', 'kb-service')
        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_VISIBILITY', 'public')
        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_GROUP_GRANTS', 'author_groups')
        monkeypatch.setenv('ACCESS_CONTROL_PUBLISH_ROLE', 'publisher')
        settings = AccessControlSettings()
        assert settings.default_principal == 'kb-service'
        assert settings.default_visibility == 'public'
        assert settings.default_group_grants == 'author_groups'
        assert settings.publish_role == 'publisher'

    @pytest.mark.parametrize('visibility', ['everyone', 'shared'])
    def test_invalid_visibility_rejected(self, visibility: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """default_visibility only accepts private or public."""
        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_VISIBILITY', visibility)
        with pytest.raises(ValidationError):
            AccessControlSettings()

    def test_invalid_group_grants_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """default_group_grants only accepts none or author_groups."""
        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_GROUP_GRANTS', 'everyone')
        with pytest.raises(ValidationError):
            AccessControlSettings()

    @pytest.mark.parametrize('principal', [
        'local',
        'kb.service@example.com',
        'user:role+tag-1_x',
        'A' * 128,
    ])
    def test_safe_principals_accepted(self, principal: str) -> None:
        """Principals within the DDL-safe character set validate."""
        assert AccessControlSettings(ACCESS_CONTROL_DEFAULT_PRINCIPAL=principal).default_principal == principal

    @pytest.mark.parametrize('principal', [
        "bad'quote",
        'has space',
        'back\\slash',
        '',
        'A' * 129,
        'semi;colon',
    ])
    def test_unsafe_principals_rejected(self, principal: str) -> None:
        """Principals that could break the DDL literal are refused at settings build."""
        with pytest.raises(ValidationError):
            AccessControlSettings(ACCESS_CONTROL_DEFAULT_PRINCIPAL=principal)

    def test_composed_into_app_settings(self) -> None:
        """AppSettings exposes the nested access_control settings."""
        get_settings.cache_clear()
        try:
            assert get_settings().access_control.default_visibility in ('private', 'public')
        finally:
            get_settings.cache_clear()


class TestAuthSettings:
    """Tests for AuthSettings configuration."""

    def test_settings_loads_token_from_env(self) -> None:
        """Settings should load MCP_AUTH_TOKEN from environment."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': 'test-token-123'}, clear=False):
            settings = AuthSettings()
            assert settings.auth_token is not None
            assert settings.auth_token.get_secret_value() == 'test-token-123'

    def test_settings_default_client_id(self) -> None:
        """Settings should have default auth_client_id of 'mcp-client'."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': 'test-token'}, clear=False):
            settings = AuthSettings()
            assert settings.auth_client_id == 'mcp-client'

    def test_settings_custom_client_id(self) -> None:
        """Settings should allow custom auth_client_id via environment."""
        with patch.dict(
            os.environ,
            {'MCP_AUTH_TOKEN': 'test-token', 'MCP_AUTH_CLIENT_ID': 'custom-client'},
            clear=False,
        ):
            settings = AuthSettings()
            assert settings.auth_client_id == 'custom-client'

    def test_token_default_is_none(self) -> None:
        """AuthSettings should have auth_token defaulting to None in Field definition."""
        # Verify the Field default is None by checking the model fields
        field_info = AuthSettings.model_fields['auth_token']
        assert field_info.default is None


class TestJwtAuthSettings:
    """Tests for the JWT fields on AuthSettings."""

    def test_jwt_field_defaults(self) -> None:
        """JWT key/issuer/audience default to unset; algorithm and claim keys have documented defaults."""
        env = {k: v for k, v in os.environ.items() if not k.startswith('MCP_AUTH')}
        with patch.dict(os.environ, env, clear=True):
            settings = AuthSettings()
            assert settings.jwt_public_key is None
            assert settings.jwt_jwks_uri is None
            assert settings.jwt_issuer is None
            assert settings.jwt_audience is None
            assert settings.jwt_algorithm == 'RS256'
            assert settings.groups_claim == 'groups'
            assert settings.roles_claim == 'roles'

    def test_jwt_fields_load_from_env(self) -> None:
        """Every MCP_AUTH_JWT_* / claim-key env var maps onto its settings field."""
        env_updates = {
            'MCP_AUTH_PROVIDER': 'jwt',
            'MCP_AUTH_JWT_PUBLIC_KEY': 'shared-secret',
            'MCP_AUTH_JWT_ISSUER': 'https://issuer.test',
            'MCP_AUTH_JWT_AUDIENCE': 'ctx-server',
            'MCP_AUTH_JWT_ALGORITHM': 'HS256',
            'MCP_AUTH_GROUPS_CLAIM': 'https://example.com/groups',
            'MCP_AUTH_ROLES_CLAIM': 'realm_access.roles',
        }
        with patch.dict(os.environ, env_updates, clear=False):
            settings = AuthSettings()
            assert settings.provider == 'jwt'
            assert settings.jwt_public_key is not None
            assert settings.jwt_public_key.get_secret_value() == 'shared-secret'
            assert settings.jwt_issuer == 'https://issuer.test'
            assert settings.jwt_audience == 'ctx-server'
            assert settings.jwt_algorithm == 'HS256'
            assert settings.groups_claim == 'https://example.com/groups'
            assert settings.roles_claim == 'realm_access.roles'

    def test_jwt_public_key_is_secret(self) -> None:
        """The key/secret value is masked in string representations."""
        with patch.dict(os.environ, {'MCP_AUTH_JWT_PUBLIC_KEY': 'hs-shared-secret'}, clear=False):
            settings = AuthSettings()
            assert 'hs-shared-secret' not in str(settings.jwt_public_key)
            assert 'hs-shared-secret' not in repr(settings.jwt_public_key)
