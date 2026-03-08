"""Tests for authentication module.

This module tests the SimpleTokenVerifier authentication mechanism
using centralized AuthSettings from app.settings.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.settings import AuthSettings
from app.settings import get_settings


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

    def test_factory_raises_when_simple_token_without_token(self) -> None:
        """Factory should raise ValueError when simple_token provider but no MCP_AUTH_TOKEN."""
        env = {k: v for k, v in os.environ.items() if k != 'MCP_AUTH_TOKEN'}
        env['MCP_AUTH_PROVIDER'] = 'simple_token'
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            with pytest.raises(ValueError, match='MCP_AUTH_TOKEN is required'):
                create_auth_provider()


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
        mock.transport.stateless_http = False
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
