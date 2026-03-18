"""Tests for source-aware summary instructions and dynamic prompt resolution.

Tests verify:
- resolve_summary_prompt('user') returns USER_SUMMARY_PROMPT when no custom prompt
- resolve_summary_prompt('agent') returns AGENT_SUMMARY_PROMPT when no custom prompt
- resolve_summary_prompt returns custom prompt (SUMMARY_PROMPT env var) for both sources
- resolve_summary_prompt returns custom prompt when SUMMARY_PROMPT is whitespace-only (falls back)
- USER_SUMMARY_PROMPT contains user-specific instructions only
- AGENT_SUMMARY_PROMPT contains agent-specific instructions only
- Both prompts contain shared base requirements
- DEFAULT_SUMMARY_PROMPT is alias for AGENT_SUMMARY_PROMPT
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

from app.summary.instructions import AGENT_SUMMARY_PROMPT
from app.summary.instructions import DEFAULT_SUMMARY_PROMPT
from app.summary.instructions import USER_SUMMARY_PROMPT
from app.summary.instructions import resolve_summary_prompt


class TestResolveSummaryPrompt:
    """Tests for resolve_summary_prompt() function with source parameter."""

    def test_returns_user_prompt_for_user_source(self) -> None:
        """resolve_summary_prompt('user') returns USER_SUMMARY_PROMPT when no custom prompt."""
        mock_settings = MagicMock()
        mock_settings.summary.prompt = None

        with patch('app.settings.get_settings', return_value=mock_settings):
            result = resolve_summary_prompt('user')

        assert result == USER_SUMMARY_PROMPT

    def test_returns_agent_prompt_for_agent_source(self) -> None:
        """resolve_summary_prompt('agent') returns AGENT_SUMMARY_PROMPT when no custom prompt."""
        mock_settings = MagicMock()
        mock_settings.summary.prompt = None

        with patch('app.settings.get_settings', return_value=mock_settings):
            result = resolve_summary_prompt('agent')

        assert result == AGENT_SUMMARY_PROMPT

    def test_returns_custom_prompt_for_user_source(self) -> None:
        """resolve_summary_prompt('user') returns custom prompt when SUMMARY_PROMPT is set."""
        mock_settings = MagicMock()
        mock_settings.summary.prompt = 'Custom summarization prompt.'

        with patch('app.settings.get_settings', return_value=mock_settings):
            result = resolve_summary_prompt('user')

        assert result == 'Custom summarization prompt.'

    def test_returns_custom_prompt_for_agent_source(self) -> None:
        """resolve_summary_prompt('agent') returns custom prompt when SUMMARY_PROMPT is set."""
        mock_settings = MagicMock()
        mock_settings.summary.prompt = 'Custom summarization prompt.'

        with patch('app.settings.get_settings', return_value=mock_settings):
            result = resolve_summary_prompt('agent')

        assert result == 'Custom summarization prompt.'

    def test_returns_default_when_custom_prompt_is_empty(self) -> None:
        """resolve_summary_prompt falls back to source-specific prompt when custom is empty."""
        mock_settings = MagicMock()
        mock_settings.summary.prompt = ''

        with patch('app.settings.get_settings', return_value=mock_settings):
            result = resolve_summary_prompt('user')

        assert result == USER_SUMMARY_PROMPT

    def test_returns_default_when_custom_prompt_is_whitespace(self) -> None:
        """resolve_summary_prompt falls back to source-specific prompt for whitespace-only custom."""
        mock_settings = MagicMock()
        mock_settings.summary.prompt = '   \n\t  '

        with patch('app.settings.get_settings', return_value=mock_settings):
            result = resolve_summary_prompt('agent')

        assert result == AGENT_SUMMARY_PROMPT

    def test_returns_default_when_custom_prompt_is_none(self) -> None:
        """resolve_summary_prompt falls back when custom prompt is None."""
        mock_settings = MagicMock()
        mock_settings.summary.prompt = None

        with patch('app.settings.get_settings', return_value=mock_settings):
            result = resolve_summary_prompt('agent')

        assert result == AGENT_SUMMARY_PROMPT

    def test_custom_prompt_with_whitespace_is_not_stripped(self) -> None:
        """resolve_summary_prompt returns custom prompt with leading/trailing whitespace intact."""
        mock_settings = MagicMock()
        mock_settings.summary.prompt = '  Custom prompt with spaces  '

        with patch('app.settings.get_settings', return_value=mock_settings):
            result = resolve_summary_prompt('user')

        assert result == '  Custom prompt with spaces  '


class TestUserSummaryPrompt:
    """Tests for USER_SUMMARY_PROMPT constant."""

    def test_contains_user_context(self) -> None:
        """USER_SUMMARY_PROMPT references human user messages."""
        assert 'message from a human user' in USER_SUMMARY_PROMPT

    def test_does_not_contain_agent_context(self) -> None:
        """USER_SUMMARY_PROMPT does not reference agent reports."""
        assert 'work report generated by an AI agent' not in USER_SUMMARY_PROMPT

    def test_starts_with_no_think(self) -> None:
        """USER_SUMMARY_PROMPT starts with /no_think for Qwen3 models."""
        assert USER_SUMMARY_PROMPT.startswith('/no_think')

    def test_contains_shared_requirements(self) -> None:
        """USER_SUMMARY_PROMPT contains shared base requirements."""
        assert 'one paragraph' in USER_SUMMARY_PROMPT
        assert 'Output ONLY the summary' in USER_SUMMARY_PROMPT
        assert 'key topics' in USER_SUMMARY_PROMPT

    def test_contains_english_language_requirement(self) -> None:
        """USER_SUMMARY_PROMPT requires English output."""
        assert 'Always write the summary in English' in USER_SUMMARY_PROMPT


class TestAgentSummaryPrompt:
    """Tests for AGENT_SUMMARY_PROMPT constant."""

    def test_contains_agent_context(self) -> None:
        """AGENT_SUMMARY_PROMPT references AI agent work reports."""
        assert 'work report generated by an AI agent' in AGENT_SUMMARY_PROMPT

    def test_does_not_contain_user_context(self) -> None:
        """AGENT_SUMMARY_PROMPT does not reference human user messages."""
        assert 'message from a human user' not in AGENT_SUMMARY_PROMPT

    def test_starts_with_no_think(self) -> None:
        """AGENT_SUMMARY_PROMPT starts with /no_think for Qwen3 models."""
        assert AGENT_SUMMARY_PROMPT.startswith('/no_think')

    def test_contains_shared_requirements(self) -> None:
        """AGENT_SUMMARY_PROMPT contains shared base requirements."""
        assert 'one paragraph' in AGENT_SUMMARY_PROMPT
        assert 'Output ONLY the summary' in AGENT_SUMMARY_PROMPT
        assert 'named entities' in AGENT_SUMMARY_PROMPT

    def test_contains_english_language_requirement(self) -> None:
        """AGENT_SUMMARY_PROMPT requires English output."""
        assert 'Always write the summary in English' in AGENT_SUMMARY_PROMPT


class TestDefaultSummaryPromptAlias:
    """Tests for DEFAULT_SUMMARY_PROMPT alias."""

    def test_is_alias_for_agent_prompt(self) -> None:
        """DEFAULT_SUMMARY_PROMPT is the same object as AGENT_SUMMARY_PROMPT."""
        assert DEFAULT_SUMMARY_PROMPT is AGENT_SUMMARY_PROMPT

    def test_is_non_empty_string(self) -> None:
        """DEFAULT_SUMMARY_PROMPT is a non-empty string."""
        assert isinstance(DEFAULT_SUMMARY_PROMPT, str)
        assert len(DEFAULT_SUMMARY_PROMPT) > 0
