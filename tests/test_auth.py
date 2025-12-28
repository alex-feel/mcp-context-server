"""Tests for authentication module.

This module tests the SimpleTokenVerifier authentication mechanism
using centralized AuthSettings from app.settings.
"""

from __future__ import annotations

import os
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
        from unittest.mock import MagicMock

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

    def test_verifier_can_be_imported_by_fastmcp_path(self) -> None:
        """Verifier should be importable via the path used by FASTMCP_SERVER_AUTH."""
        # This simulates how FastMCP loads the class via ImportString
        from pydantic import ImportString
        from pydantic import TypeAdapter

        type_adapter = TypeAdapter(ImportString)
        auth_class = type_adapter.validate_python('app.auth.simple_token.SimpleTokenVerifier')

        # Should return the class, not an instance
        from app.auth.simple_token import SimpleTokenVerifier

        assert auth_class is SimpleTokenVerifier

    def test_verifier_instantiates_with_no_args(self) -> None:
        """Verifier should be instantiable with no arguments (required by FastMCP)."""
        with patch.dict(os.environ, {'MCP_AUTH_TOKEN': 'test-token'}, clear=False):
            from app.auth.simple_token import SimpleTokenVerifier

            # FastMCP calls the class with no arguments: auth_class()
            verifier = SimpleTokenVerifier()
            assert verifier is not None
