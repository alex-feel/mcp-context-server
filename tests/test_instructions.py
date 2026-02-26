"""Tests for MCP server instructions support."""

import os
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def env_var(key: str, value: str | None) -> Generator[None, None, None]:
    """Context manager for temporarily setting an environment variable."""
    original = os.environ.get(key)
    try:
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]
        yield
    finally:
        if original is not None:
            os.environ[key] = original
        elif key in os.environ:
            del os.environ[key]


class TestInstructionsSettings:
    """Tests for InstructionsSettings in app/settings.py."""

    def test_server_instructions_default_is_none(self) -> None:
        """MCP_SERVER_INSTRUCTIONS should default to None (use DEFAULT_INSTRUCTIONS)."""
        from app.settings import InstructionsSettings

        settings = InstructionsSettings()
        assert settings.server_instructions is None

    def test_server_instructions_from_env(self) -> None:
        """MCP_SERVER_INSTRUCTIONS env var should override default instructions."""
        from app.settings import InstructionsSettings

        custom_text = 'Custom server instructions for deployment.'
        with env_var('MCP_SERVER_INSTRUCTIONS', custom_text):
            settings = InstructionsSettings()
            assert settings.server_instructions == custom_text

    def test_server_instructions_empty_string_from_env(self) -> None:
        """Empty MCP_SERVER_INSTRUCTIONS should be treated as empty string (disables instructions)."""
        from app.settings import InstructionsSettings

        with env_var('MCP_SERVER_INSTRUCTIONS', ''):
            settings = InstructionsSettings()
            assert settings.server_instructions == ''

    def test_instructions_accessible_via_app_settings(self) -> None:
        """InstructionsSettings should be accessible via AppSettings.instructions."""
        from app.settings import AppSettings

        settings = AppSettings()
        assert hasattr(settings, 'instructions')
        assert settings.instructions.server_instructions is None


class TestDefaultInstructions:
    """Tests for DEFAULT_INSTRUCTIONS constant in app/instructions.py."""

    def test_default_instructions_is_non_empty_string(self) -> None:
        """DEFAULT_INSTRUCTIONS must be a non-empty string."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert isinstance(DEFAULT_INSTRUCTIONS, str)
        assert len(DEFAULT_INSTRUCTIONS) > 0

    def test_default_instructions_contains_server_purpose(self) -> None:
        """DEFAULT_INSTRUCTIONS must describe what the server does."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'context' in DEFAULT_INSTRUCTIONS.lower()
        assert 'agent' in DEFAULT_INSTRUCTIONS.lower()

    def test_default_instructions_contains_search_tools(self) -> None:
        """DEFAULT_INSTRUCTIONS must mention key search tools."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'hybrid_search_context' in DEFAULT_INSTRUCTIONS
        assert 'search_context' in DEFAULT_INSTRUCTIONS
        assert 'get_context_by_ids' in DEFAULT_INSTRUCTIONS

    def test_default_instructions_contains_store_tool(self) -> None:
        """DEFAULT_INSTRUCTIONS must mention store_context."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'store_context' in DEFAULT_INSTRUCTIONS

    def test_default_instructions_mentions_thread_id(self) -> None:
        """DEFAULT_INSTRUCTIONS must explain thread_id concept."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'thread_id' in DEFAULT_INSTRUCTIONS

    def test_default_instructions_mentions_metadata(self) -> None:
        """DEFAULT_INSTRUCTIONS must explain metadata concept."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'metadata' in DEFAULT_INSTRUCTIONS

    def test_default_instructions_mentions_references(self) -> None:
        """DEFAULT_INSTRUCTIONS must explain references for knowledge graph."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 'references' in DEFAULT_INSTRUCTIONS.lower()

    def test_default_instructions_character_budget(self) -> None:
        """DEFAULT_INSTRUCTIONS should be within reasonable character budget (~2000-5000 chars)."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        assert 2000 < len(DEFAULT_INSTRUCTIONS) < 5000

    def test_default_instructions_is_static_constant(self) -> None:
        """DEFAULT_INSTRUCTIONS must be a simple string constant, not dynamically generated."""
        from app.instructions import DEFAULT_INSTRUCTIONS

        # Verify it does not contain dynamic format markers or placeholders
        assert '{field_names}' not in DEFAULT_INSTRUCTIONS
        assert '{0}' not in DEFAULT_INSTRUCTIONS


class TestInstructionsResolution:
    """Tests for instructions resolution logic (env var override vs default)."""

    def test_resolve_instructions_returns_default_when_no_env(self) -> None:
        """When MCP_SERVER_INSTRUCTIONS is not set, should use DEFAULT_INSTRUCTIONS."""
        from app.instructions import DEFAULT_INSTRUCTIONS
        from app.instructions import resolve_instructions
        from app.settings import InstructionsSettings

        settings = InstructionsSettings()
        result = resolve_instructions(settings)
        assert result == DEFAULT_INSTRUCTIONS

    def test_resolve_instructions_returns_env_override(self) -> None:
        """When MCP_SERVER_INSTRUCTIONS is set, should use env var value."""
        from app.instructions import resolve_instructions
        from app.settings import InstructionsSettings

        custom = 'Custom instructions text.'
        with env_var('MCP_SERVER_INSTRUCTIONS', custom):
            settings = InstructionsSettings()
            result = resolve_instructions(settings)
            assert result == custom

    def test_resolve_instructions_empty_string_disables(self) -> None:
        """Empty MCP_SERVER_INSTRUCTIONS should return empty string (effectively disables)."""
        from app.instructions import resolve_instructions
        from app.settings import InstructionsSettings

        with env_var('MCP_SERVER_INSTRUCTIONS', ''):
            settings = InstructionsSettings()
            result = resolve_instructions(settings)
            assert result == ''
