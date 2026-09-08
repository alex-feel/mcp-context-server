"""Tests for authentication module.

This module tests the SimpleTokenVerifier and JWT authentication mechanisms
using centralized AuthSettings from app.settings.
"""

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from app.settings import AuthSettings
from app.settings import get_settings


@pytest.fixture(scope='module')
def rsa_key_pair() -> RSAKeyPair:
    """Generate one RSA key pair for all JWT tests in this module."""
    return RSAKeyPair.generate()


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


class TestSimpleTokenVerifier:
    """Tests for SimpleTokenVerifier."""

    @pytest.fixture(autouse=True)
    def clear_settings_cache(self) -> None:
        """Clear the settings cache before each test."""
        get_settings.cache_clear()

    def test_verifier_raises_when_token_not_set(self) -> None:
        """Verifier should raise ValueError when MCP_AUTH_TOKEN is not set."""

        # Create mock settings with auth_token = None
        mock_settings = MagicMock()
        mock_settings.auth.auth_token = None

        with patch('app.auth.simple_token.get_settings', return_value=mock_settings):
            from app.auth.simple_token import SimpleTokenVerifier

            with pytest.raises(ValueError, match='MCP_AUTH_TOKEN is required'):
                SimpleTokenVerifier()

    def test_verifier_raises_when_token_empty(self) -> None:
        """Verifier should raise ValueError when MCP_AUTH_TOKEN is empty."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': ''}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            with pytest.raises(ValueError, match='MCP_AUTH_TOKEN cannot be empty'):
                SimpleTokenVerifier()

    def test_verifier_raises_when_token_whitespace(self) -> None:
        """Verifier should raise ValueError when MCP_AUTH_TOKEN is only whitespace."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': '   '}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            with pytest.raises(ValueError, match='MCP_AUTH_TOKEN cannot be empty'):
                SimpleTokenVerifier()

    def test_verifier_initializes_with_valid_token(self) -> None:
        """Verifier should initialize successfully with valid token."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': 'valid-token-123'}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            verifier = SimpleTokenVerifier()
            assert verifier._token.get_secret_value() == 'valid-token-123'

    @pytest.mark.asyncio
    async def test_verify_token_success(self) -> None:
        """verify_token should return AccessToken for valid token."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': 'my-secret-token'}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            verifier = SimpleTokenVerifier()
            result = await verifier.verify_token('my-secret-token')

            assert result is not None
            assert result.token == 'my-secret-token'
            assert result.client_id == 'mcp-client'
            assert 'tools:read' in result.scopes
            assert 'tools:write' in result.scopes

    @pytest.mark.asyncio
    async def test_verify_token_failure_wrong_token(self) -> None:
        """verify_token should return None for wrong token."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': 'correct-token'}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            verifier = SimpleTokenVerifier()
            result = await verifier.verify_token('wrong-token')

            assert result is None

    @pytest.mark.asyncio
    async def test_verify_token_failure_empty_token(self) -> None:
        """verify_token should return None for empty token."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': 'valid-token'}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            verifier = SimpleTokenVerifier()
            result = await verifier.verify_token('')

            assert result is None

    @pytest.mark.asyncio
    async def test_verify_token_failure_whitespace_token(self) -> None:
        """verify_token should return None for whitespace token."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': 'valid-token'}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            verifier = SimpleTokenVerifier()
            result = await verifier.verify_token('   ')

            assert result is None

    @pytest.mark.asyncio
    async def test_verify_token_non_ascii_returns_none_not_raises(self) -> None:
        """A non-ASCII bearer token rejects cleanly instead of crashing.

        hmac.compare_digest raises TypeError on str operands containing non-ASCII
        characters; verify_token compares UTF-8 bytes so a non-ASCII client token is
        an ordinary mismatch (None -> 401) rather than an uncaught error (500).
        """
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': 'valid-token'}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            verifier = SimpleTokenVerifier()
            result = await verifier.verify_token('töken-with-accent')

            assert result is None

    @pytest.mark.asyncio
    async def test_verify_token_non_ascii_exact_match_succeeds(self) -> None:
        """A configured non-ASCII token still matches its exact value (byte compare)."""
        secret = 'töken-Ünïcode'
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': secret}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            verifier = SimpleTokenVerifier()
            result = await verifier.verify_token(secret)

            assert result is not None
            assert result.token == secret

    @pytest.mark.asyncio
    async def test_verify_token_with_custom_client_id(self) -> None:
        """verify_token should use custom client_id from settings."""
        with patch.dict(
            os.environ,
            {'MCP_AUTH_TOKEN': 'token', 'MCP_AUTH_CLIENT_ID': 'my-custom-client'},
            clear=False,
        ):
            from app.auth.simple_token import SimpleTokenVerifier

            verifier = SimpleTokenVerifier()
            result = await verifier.verify_token('token')

            assert result is not None
            assert result.client_id == 'my-custom-client'

    def test_token_not_exposed_in_string(self) -> None:
        """Token should not be exposed in string representation (SecretStr)."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': 'super-secret-token'}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            verifier = SimpleTokenVerifier()
            token_str = str(verifier._token)

            # SecretStr should mask the value
            assert 'super-secret-token' not in token_str

    def test_token_not_exposed_in_repr(self) -> None:
        """Token should not be exposed in repr (SecretStr)."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': 'another-secret'}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            verifier = SimpleTokenVerifier()
            token_repr = repr(verifier._token)

            # SecretStr should mask the value in repr too
            assert 'another-secret' not in token_repr


class TestSimpleTokenVerifierIntegration:
    """Integration tests for SimpleTokenVerifier with FastMCP."""

    @pytest.fixture(autouse=True)
    def clear_settings_cache(self) -> None:
        """Clear the settings cache before each test."""
        get_settings.cache_clear()

    def test_auth_factory_creates_simple_token_verifier(self) -> None:
        """Auth factory should create SimpleTokenVerifier when provider is simple_token."""
        with patch.dict(
            os.environ,
            {'MCP_AUTH_PROVIDER': 'simple_token', 'MCP_AUTH_TOKEN': 'test-token'},
            clear=False,
        ):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.auth.simple_token import SimpleTokenVerifier

            provider = create_auth_provider()
            assert isinstance(provider, SimpleTokenVerifier)

    def test_verifier_instantiates_with_no_args(self) -> None:
        """Verifier should be instantiable with no arguments (required by auth factory)."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': 'test-token'}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            verifier = SimpleTokenVerifier()
            assert verifier is not None


class TestAuthFactory:
    """Tests for the auth provider factory."""

    @pytest.fixture(autouse=True)
    def clear_settings_cache(self) -> None:
        """Clear the settings cache before each test."""
        get_settings.cache_clear()

    def test_factory_returns_none_when_no_auth(self) -> None:
        """Factory should return None when MCP_AUTH_PROVIDER=none."""
        with patch.dict(os.environ, {'MCP_AUTH_PROVIDER': 'none'}, clear=False):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            result = create_auth_provider()
            assert result is None

    def test_factory_returns_none_by_default(self) -> None:
        """Factory should return None when MCP_AUTH_PROVIDER is not set (default is none)."""
        env = {k: v for k, v in os.environ.items() if k != 'MCP_AUTH_PROVIDER'}
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            result = create_auth_provider()
            assert result is None

    def test_factory_creates_simple_token_verifier(self) -> None:
        """Factory should create SimpleTokenVerifier when provider is simple_token."""
        with patch.dict(
            os.environ,
            {'MCP_AUTH_PROVIDER': 'simple_token', 'MCP_AUTH_TOKEN': 'test-token'},
            clear=False,
        ):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.auth.simple_token import SimpleTokenVerifier

            provider = create_auth_provider()
            assert isinstance(provider, SimpleTokenVerifier)

    def test_factory_raises_configuration_error_when_simple_token_without_token(self) -> None:
        """Factory raises ConfigurationError (exit 78) when token is missing.

        A missing required env var is a startup misconfiguration: the factory
        translates SimpleTokenVerifier's ValueError into a ConfigurationError so
        ``main()`` exits 78 (EX_CONFIG, supervisor does NOT restart) rather than
        the generic exit 1 a bare ValueError would cause.
        """
        env = {k: v for k, v in os.environ.items() if k != 'MCP_AUTH_TOKEN'}
        env['MCP_AUTH_PROVIDER'] = 'simple_token'
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.errors import ConfigurationError

            with pytest.raises(ConfigurationError, match='MCP_AUTH_TOKEN is required'):
                create_auth_provider()

    def test_factory_raises_configuration_error_when_simple_token_empty(self) -> None:
        """Factory raises ConfigurationError when MCP_AUTH_TOKEN is empty/whitespace."""
        env = {k: v for k, v in os.environ.items() if k != 'MCP_AUTH_TOKEN'}
        env['MCP_AUTH_PROVIDER'] = 'simple_token'
        env['MCP_AUTH_TOKEN'] = '   '
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.errors import ConfigurationError

            with pytest.raises(ConfigurationError, match='cannot be empty'):
                create_auth_provider()


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


class TestJwtAuthFactory:
    """Tests for the jwt arm of the auth provider factory."""

    @pytest.fixture(autouse=True)
    def clear_settings_cache(self) -> None:
        """Clear the settings cache before each test."""
        get_settings.cache_clear()

    @staticmethod
    def _jwt_env(**overrides: str) -> dict[str, str]:
        """Build a clean environment for the jwt provider with the given vars."""
        env = {k: v for k, v in os.environ.items() if not k.startswith('MCP_AUTH')}
        env['MCP_AUTH_PROVIDER'] = 'jwt'
        env.update(overrides)
        return env

    def test_factory_raises_when_no_key_configured(self) -> None:
        """Neither MCP_AUTH_JWT_PUBLIC_KEY nor MCP_AUTH_JWT_JWKS_URI is a startup misconfiguration."""
        with patch.dict(os.environ, self._jwt_env(), clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.errors import ConfigurationError

            with pytest.raises(ConfigurationError, match='exactly one of'):
                create_auth_provider()

    def test_factory_raises_when_both_keys_configured(self) -> None:
        """Setting both key sources is rejected as mutually exclusive."""
        env = self._jwt_env(
            MCP_AUTH_JWT_PUBLIC_KEY='some-secret',
            MCP_AUTH_JWT_JWKS_URI='https://idp.example.com/certs',
        )
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.errors import ConfigurationError

            with pytest.raises(ConfigurationError, match='mutually exclusive'):
                create_auth_provider()

    def test_factory_treats_whitespace_key_as_unset(self) -> None:
        """A whitespace-only key value counts as unset, mirroring MCP_AUTH_TOKEN handling."""
        with patch.dict(os.environ, self._jwt_env(MCP_AUTH_JWT_PUBLIC_KEY='   '), clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.errors import ConfigurationError

            with pytest.raises(ConfigurationError, match='exactly one of'):
                create_auth_provider()

    def test_factory_creates_verifier_from_static_key(self, rsa_key_pair: RSAKeyPair) -> None:
        """A PEM public key produces a JWTVerifier with issuer/audience/algorithm applied."""
        env = self._jwt_env(
            MCP_AUTH_JWT_PUBLIC_KEY=rsa_key_pair.public_key,
            MCP_AUTH_JWT_ISSUER='https://issuer.test',
            MCP_AUTH_JWT_AUDIENCE='ctx-server',
        )
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from fastmcp.server.auth.providers.jwt import JWTVerifier

            from app.auth import create_auth_provider

            provider = create_auth_provider()
            assert isinstance(provider, JWTVerifier)
            assert provider.issuer == 'https://issuer.test'
            assert provider.audience == 'ctx-server'
            assert provider.algorithm == 'RS256'
            assert provider.jwks_uri is None

    def test_factory_creates_verifier_from_jwks_uri(self) -> None:
        """A JWKS URI produces a JWTVerifier in JWKS mode."""
        env = self._jwt_env(MCP_AUTH_JWT_JWKS_URI='https://idp.example.com/certs')
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from fastmcp.server.auth.providers.jwt import JWTVerifier

            from app.auth import create_auth_provider

            provider = create_auth_provider()
            assert isinstance(provider, JWTVerifier)
            assert provider.jwks_uri == 'https://idp.example.com/certs'
            assert provider.public_key is None

    def test_factory_raises_on_unsupported_algorithm(self) -> None:
        """An unsupported MCP_AUTH_JWT_ALGORITHM value is a startup misconfiguration."""
        env = self._jwt_env(
            MCP_AUTH_JWT_PUBLIC_KEY='some-secret',
            MCP_AUTH_JWT_ALGORITHM='none',
        )
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.errors import ConfigurationError

            with pytest.raises(ConfigurationError, match='Unsupported algorithm'):
                create_auth_provider()

    def test_factory_raises_on_symmetric_algorithm_with_jwks(self) -> None:
        """HS* algorithms cannot fetch keys from a JWKS endpoint."""
        env = self._jwt_env(
            MCP_AUTH_JWT_JWKS_URI='https://idp.example.com/certs',
            MCP_AUTH_JWT_ALGORITHM='HS256',
        )
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.errors import ConfigurationError

            with pytest.raises(ConfigurationError, match='cannot be used with jwks_uri'):
                create_auth_provider()

    def test_factory_raises_on_symmetric_algorithm_with_pem_key(self, rsa_key_pair: RSAKeyPair) -> None:
        """HS* algorithms require a shared secret, not PEM public key material."""
        env = self._jwt_env(
            MCP_AUTH_JWT_PUBLIC_KEY=rsa_key_pair.public_key,
            MCP_AUTH_JWT_ALGORITHM='HS256',
        )
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.errors import ConfigurationError

            with pytest.raises(ConfigurationError, match='shared secret'):
                create_auth_provider()


class TestJwtVerifierClaimsRoundTrip:
    """Round-trip tests: minted JWTs through the factory-built verifier."""

    @pytest.fixture(autouse=True)
    def clear_settings_cache(self) -> None:
        """Clear the settings cache before each test."""
        get_settings.cache_clear()

    @staticmethod
    def _verifier_env(rsa_key_pair: RSAKeyPair) -> dict[str, str]:
        """Build the environment for a static-key verifier with issuer/audience pinned."""
        env = {k: v for k, v in os.environ.items() if not k.startswith('MCP_AUTH')}
        env.update({
            'MCP_AUTH_PROVIDER': 'jwt',
            'MCP_AUTH_JWT_PUBLIC_KEY': rsa_key_pair.public_key,
            'MCP_AUTH_JWT_ISSUER': 'https://issuer.test',
            'MCP_AUTH_JWT_AUDIENCE': 'ctx-server',
        })
        return env

    @pytest.mark.asyncio
    async def test_valid_token_carries_claims(self, rsa_key_pair: RSAKeyPair) -> None:
        """A valid minted token verifies and its claims reach AccessToken.claims."""
        with patch.dict(os.environ, self._verifier_env(rsa_key_pair), clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            provider = create_auth_provider()
            assert provider is not None
            token = rsa_key_pair.create_token(
                subject='alice',
                issuer='https://issuer.test',
                audience='ctx-server',
                additional_claims={'groups': ['team-a'], 'roles': ['publisher']},
            )
            access_token = await provider.verify_token(token)
            assert access_token is not None
            assert access_token.claims.get('sub') == 'alice'
            assert access_token.claims.get('groups') == ['team-a']
            assert access_token.claims.get('roles') == ['publisher']

    @pytest.mark.asyncio
    async def test_expired_token_is_rejected(self, rsa_key_pair: RSAKeyPair) -> None:
        """An expired minted token is rejected."""
        with patch.dict(os.environ, self._verifier_env(rsa_key_pair), clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            provider = create_auth_provider()
            assert provider is not None
            token = rsa_key_pair.create_token(
                subject='alice',
                issuer='https://issuer.test',
                audience='ctx-server',
                expires_in_seconds=-60,
            )
            assert await provider.verify_token(token) is None

    @pytest.mark.asyncio
    async def test_wrong_audience_is_rejected(self, rsa_key_pair: RSAKeyPair) -> None:
        """A token minted for another audience is rejected."""
        with patch.dict(os.environ, self._verifier_env(rsa_key_pair), clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            provider = create_auth_provider()
            assert provider is not None
            token = rsa_key_pair.create_token(
                subject='alice',
                issuer='https://issuer.test',
                audience='other-service',
            )
            assert await provider.verify_token(token) is None

    @pytest.mark.asyncio
    async def test_wrong_signature_is_rejected(self, rsa_key_pair: RSAKeyPair) -> None:
        """A token signed by a different key pair is rejected."""
        with patch.dict(os.environ, self._verifier_env(rsa_key_pair), clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            provider = create_auth_provider()
            assert provider is not None
            other_pair = RSAKeyPair.generate()
            token = other_pair.create_token(
                subject='mallory',
                issuer='https://issuer.test',
                audience='ctx-server',
            )
            assert await provider.verify_token(token) is None


class TestAuthTransportInteraction:
    """Tests for auth behavior conditional on transport mode.

    Verifies that authentication is skipped on stdio transport and only
    initialized for HTTP transports, as per MCP specification.
    """

    @pytest.fixture(autouse=True)
    def clear_settings_cache(self) -> None:
        """Clear the settings cache before each test."""
        get_settings.cache_clear()

    @staticmethod
    def _make_mock_settings(transport: str = 'stdio', auth_provider: str = 'none') -> MagicMock:
        """Create a mock settings object matching the given transport and auth config."""
        mock = MagicMock()
        mock.transport.transport = transport
        mock.transport.host = '0.0.0.0'
        mock.transport.port = 8000
        mock.transport.stateless_http = True
        mock.auth.provider = auth_provider
        mock.instructions = MagicMock()
        mock.logging.level = 'ERROR'
        return mock

    def test_auth_not_initialized_on_stdio(self) -> None:
        """create_auth_provider should NOT be called when transport is stdio."""
        mock_settings = self._make_mock_settings(transport='stdio', auth_provider='none')

        with (
            patch('app.server.settings', mock_settings),
            patch('app.server.create_auth_provider') as mock_create_auth,
            patch('app.server.FastMCP') as mock_fastmcp,
            patch('app.instructions.resolve_instructions', return_value=None),
        ):
            mock_mcp_instance = mock_fastmcp.return_value
            mock_mcp_instance.run.return_value = None

            from app.server import main

            main()

            mock_create_auth.assert_not_called()

    def test_auth_initialized_on_http(self) -> None:
        """create_auth_provider SHOULD be called when transport is http."""
        mock_settings = self._make_mock_settings(transport='http', auth_provider='none')

        with (
            patch('app.server.settings', mock_settings),
            patch('app.server.create_auth_provider', return_value=None) as mock_create_auth,
            patch('app.server.FastMCP') as mock_fastmcp,
            patch('app.instructions.resolve_instructions', return_value=None),
        ):
            mock_mcp_instance = mock_fastmcp.return_value
            mock_mcp_instance.run.return_value = None
            mock_mcp_instance.custom_route.return_value = lambda _f: _f

            from app.server import main

            main()

            mock_create_auth.assert_called_once()

    def test_warning_logged_when_auth_configured_on_stdio(self) -> None:
        """A warning should be logged when MCP_AUTH_PROVIDER is set on stdio transport."""
        mock_settings = self._make_mock_settings(transport='stdio', auth_provider='simple_token')

        with (
            patch('app.server.settings', mock_settings),
            patch('app.server.logger') as mock_logger,
            patch('app.server.FastMCP') as mock_fastmcp,
            patch('app.instructions.resolve_instructions', return_value=None),
        ):
            mock_mcp_instance = mock_fastmcp.return_value
            mock_mcp_instance.run.return_value = None

            from app.server import main

            main()

            mock_logger.warning.assert_called_once()
            warning_args = mock_logger.warning.call_args
            assert 'no effect on stdio transport' in warning_args[0][0]
            assert warning_args[0][1] == 'simple_token'

    def test_no_crash_on_simple_token_without_token_on_stdio(self) -> None:
        """Server should start on stdio even with MCP_AUTH_PROVIDER=simple_token and no token.

        On stdio, create_auth_provider() is never called, so the missing
        MCP_AUTH_TOKEN does not trigger the ValueError from SimpleTokenVerifier.
        """
        mock_settings = self._make_mock_settings(transport='stdio', auth_provider='simple_token')

        with (
            patch('app.server.settings', mock_settings),
            patch('app.server.FastMCP') as mock_fastmcp,
            patch('app.instructions.resolve_instructions', return_value=None),
        ):
            mock_mcp_instance = mock_fastmcp.return_value
            mock_mcp_instance.run.return_value = None

            from app.server import main

            # This should NOT raise ValueError — auth is skipped on stdio
            main()
