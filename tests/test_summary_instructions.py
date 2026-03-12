"""Tests for summary instructions and prompt resolution.

Tests verify:
- resolve_summary_prompt returns DEFAULT_SUMMARY_PROMPT when prompt is None
- resolve_summary_prompt returns DEFAULT_SUMMARY_PROMPT when prompt is empty string
- resolve_summary_prompt returns DEFAULT_SUMMARY_PROMPT when prompt is whitespace-only
- resolve_summary_prompt returns custom prompt when set
- resolve_summary_prompt does not strip custom prompt with leading/trailing whitespace
- resolve_summary_prompt returns DEFAULT_SUMMARY_PROMPT for newlines-only and mixed whitespace
- DEFAULT_SUMMARY_PROMPT starts with /no_think
- DEFAULT_SUMMARY_PROMPT contains key constraint phrases
- DEFAULT_SUMMARY_PROMPT is a non-empty string
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.summary.instructions import DEFAULT_SUMMARY_PROMPT
from app.summary.instructions import resolve_summary_prompt


class TestResolveSummaryPrompt:
    """Tests for resolve_summary_prompt() function."""

    def test_returns_default_when_prompt_is_none(self) -> None:
        """resolve_summary_prompt returns DEFAULT_SUMMARY_PROMPT when settings.prompt is None."""
        mock_settings = MagicMock()
        mock_settings.prompt = None

        result = resolve_summary_prompt(mock_settings)

        assert result == DEFAULT_SUMMARY_PROMPT

    def test_returns_default_when_prompt_is_empty_string(self) -> None:
        """resolve_summary_prompt returns DEFAULT_SUMMARY_PROMPT when settings.prompt is empty."""
        mock_settings = MagicMock()
        mock_settings.prompt = ''

        result = resolve_summary_prompt(mock_settings)

        assert result == DEFAULT_SUMMARY_PROMPT

    def test_returns_default_when_prompt_is_whitespace(self) -> None:
        """resolve_summary_prompt returns DEFAULT_SUMMARY_PROMPT when settings.prompt is whitespace."""
        mock_settings = MagicMock()
        mock_settings.prompt = '   \n\t  '

        result = resolve_summary_prompt(mock_settings)

        assert result == DEFAULT_SUMMARY_PROMPT

    def test_returns_custom_prompt_when_set(self) -> None:
        """resolve_summary_prompt returns the custom prompt when settings.prompt has content."""
        mock_settings = MagicMock()
        mock_settings.prompt = 'Summarize concisely in one sentence.'

        result = resolve_summary_prompt(mock_settings)

        assert result == 'Summarize concisely in one sentence.'

    def test_custom_prompt_with_whitespace_is_not_stripped(self) -> None:
        """resolve_summary_prompt returns custom prompt with leading/trailing whitespace intact."""
        mock_settings = MagicMock()
        mock_settings.prompt = '  Custom prompt with spaces  '

        result = resolve_summary_prompt(mock_settings)

        assert result == '  Custom prompt with spaces  '

    def test_returns_default_for_only_newlines(self) -> None:
        """resolve_summary_prompt returns DEFAULT_SUMMARY_PROMPT when prompt is only newlines."""
        mock_settings = MagicMock()
        mock_settings.prompt = '\n\n\n'

        result = resolve_summary_prompt(mock_settings)

        assert result == DEFAULT_SUMMARY_PROMPT

    def test_returns_default_for_mixed_whitespace(self) -> None:
        """resolve_summary_prompt returns DEFAULT_SUMMARY_PROMPT for tabs, spaces, newlines."""
        mock_settings = MagicMock()
        mock_settings.prompt = ' \t \n \r '

        result = resolve_summary_prompt(mock_settings)

        assert result == DEFAULT_SUMMARY_PROMPT


class TestDefaultSummaryPrompt:
    """Tests for DEFAULT_SUMMARY_PROMPT constant."""

    def test_starts_with_no_think(self) -> None:
        """DEFAULT_SUMMARY_PROMPT starts with /no_think for Qwen3 models."""
        assert DEFAULT_SUMMARY_PROMPT.startswith('/no_think')

    def test_contains_expert_summarizer_role(self) -> None:
        """DEFAULT_SUMMARY_PROMPT contains role assignment."""
        assert 'expert summarizer' in DEFAULT_SUMMARY_PROMPT

    def test_contains_single_paragraph_constraint(self) -> None:
        """DEFAULT_SUMMARY_PROMPT requires single paragraph output."""
        assert 'one paragraph' in DEFAULT_SUMMARY_PROMPT

    def test_contains_no_labels_constraint(self) -> None:
        """DEFAULT_SUMMARY_PROMPT instructs against labels and prefixes."""
        assert 'Do not add any labels' in DEFAULT_SUMMARY_PROMPT

    def test_contains_output_only_constraint(self) -> None:
        """DEFAULT_SUMMARY_PROMPT instructs to output ONLY the summary."""
        assert 'Output ONLY the summary' in DEFAULT_SUMMARY_PROMPT

    def test_contains_key_topics_requirement(self) -> None:
        """DEFAULT_SUMMARY_PROMPT requires key topics and entities."""
        assert 'key topics' in DEFAULT_SUMMARY_PROMPT
        assert 'named entities' in DEFAULT_SUMMARY_PROMPT

    def test_is_non_empty_string(self) -> None:
        """DEFAULT_SUMMARY_PROMPT is a non-empty string."""
        assert isinstance(DEFAULT_SUMMARY_PROMPT, str)
        assert len(DEFAULT_SUMMARY_PROMPT) > 0
