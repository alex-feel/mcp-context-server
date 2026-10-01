"""Tests for app/settings/server.py: the transport and server-instructions settings."""

from tests.helpers import env_var


class TestTransportStatelessHttp:
    """Tests for FASTMCP_STATELESS_HTTP setting in TransportSettings."""

    def test_stateless_http_default_is_true(self) -> None:
        """FASTMCP_STATELESS_HTTP should default to True."""
        from app.settings.server import TransportSettings

        settings = TransportSettings()
        assert settings.stateless_http is True

    def test_stateless_http_enabled_via_env(self) -> None:
        """FASTMCP_STATELESS_HTTP=true should enable stateless mode."""
        from app.settings.server import TransportSettings

        with env_var('FASTMCP_STATELESS_HTTP', 'true'):
            settings = TransportSettings()
            assert settings.stateless_http is True

    def test_stateless_http_disabled_via_env(self) -> None:
        """FASTMCP_STATELESS_HTTP=false should disable stateless mode."""
        from app.settings.server import TransportSettings

        with env_var('FASTMCP_STATELESS_HTTP', 'false'):
            settings = TransportSettings()
            assert settings.stateless_http is False


class TestInstructionsSettings:
    """Tests for InstructionsSettings in app/settings/server.py."""

    def test_server_instructions_default_is_none(self) -> None:
        """MCP_SERVER_INSTRUCTIONS should default to None (use DEFAULT_INSTRUCTIONS)."""
        from app.settings.server import InstructionsSettings

        settings = InstructionsSettings()
        assert settings.server_instructions is None

    def test_server_instructions_from_env(self) -> None:
        """MCP_SERVER_INSTRUCTIONS env var should override default instructions."""
        from app.settings.server import InstructionsSettings

        custom_text = 'Custom server instructions for deployment.'
        with env_var('MCP_SERVER_INSTRUCTIONS', custom_text):
            settings = InstructionsSettings()
            assert settings.server_instructions == custom_text

    def test_server_instructions_empty_string_from_env(self) -> None:
        """Empty MCP_SERVER_INSTRUCTIONS should be treated as empty string (disables instructions)."""
        from app.settings.server import InstructionsSettings

        with env_var('MCP_SERVER_INSTRUCTIONS', ''):
            settings = InstructionsSettings()
            assert settings.server_instructions == ''

    def test_instructions_accessible_via_app_settings(self) -> None:
        """InstructionsSettings should be accessible via AppSettings.instructions."""
        from app.settings import AppSettings

        settings = AppSettings()
        assert hasattr(settings, 'instructions')
        assert settings.instructions.server_instructions is None
