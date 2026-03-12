"""Tests for summary provider dependency checking.

Tests for check_summary_provider_dependencies() and provider-specific
check functions in app/migrations/dependencies.py.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.errors import ConfigurationError
from app.errors import DependencyError
from app.errors import classify_provider_error
from app.migrations.dependencies import check_summary_provider_dependencies
from app.settings import EmbeddingSettings
from app.settings import SummarySettings


def _make_summary_settings() -> SummarySettings:
    """Create SummarySettings with defaults for testing."""
    env = {
        'SUMMARY_PROVIDER': 'ollama',
        'SUMMARY_MODEL': 'qwen3:1.7b',
    }
    with patch.dict(os.environ, env, clear=False):
        return SummarySettings()


def _make_embedding_settings() -> EmbeddingSettings:
    """Create EmbeddingSettings with defaults for testing."""
    env = {
        'EMBEDDING_PROVIDER': 'ollama',
        'EMBEDDING_MODEL': 'qwen3-embedding:0.6b',
        'OLLAMA_HOST': 'http://localhost:11434',
    }
    with patch.dict(os.environ, env, clear=False):
        return EmbeddingSettings()


class TestCheckSummaryOllama:
    """Tests for Ollama summary provider dependency checks."""

    @pytest.mark.asyncio
    async def test_all_available(self) -> None:
        """Test that all checks pass when Ollama is fully available."""
        summary_settings = _make_summary_settings()
        embedding_settings = _make_embedding_settings()

        # Mock httpx success
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx_client = MagicMock()
        mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
        mock_httpx_client.__aexit__ = AsyncMock()
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        # Mock ollama client success
        mock_ollama_client = MagicMock()
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = mock_ollama_client

        with (
            patch('importlib.util.find_spec', return_value=MagicMock()),
            patch('httpx.AsyncClient', return_value=mock_httpx_client),
            patch.dict('sys.modules', {'ollama': mock_ollama}),
        ):
            result = await check_summary_provider_dependencies(
                'ollama', summary_settings, embedding_settings,
            )

        assert result['available'] is True
        assert result['reason'] is None

    @pytest.mark.asyncio
    async def test_package_missing(self) -> None:
        """Test failure when langchain-ollama is not installed."""
        summary_settings = _make_summary_settings()
        embedding_settings = _make_embedding_settings()

        with patch('importlib.util.find_spec', return_value=None):
            result = await check_summary_provider_dependencies(
                'ollama', summary_settings, embedding_settings,
            )

        assert result['available'] is False
        assert 'not installed' in (result['reason'] or '')
        assert result['install_instructions'] == 'uv sync --extra summary-ollama'

    @pytest.mark.asyncio
    async def test_service_unavailable(self) -> None:
        """Test failure when Ollama service is not running."""
        summary_settings = _make_summary_settings()
        embedding_settings = _make_embedding_settings()

        # Create async context manager mock that raises on get()
        mock_httpx_client = MagicMock()
        mock_httpx_client.get = AsyncMock(side_effect=Exception('Connection refused'))
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_httpx_client
        async_cm.__aexit__.return_value = None

        with (
            patch('importlib.util.find_spec', return_value=MagicMock()),
            patch('httpx.AsyncClient', return_value=async_cm),
        ):
            result = await check_summary_provider_dependencies(
                'ollama', summary_settings, embedding_settings,
            )

        assert result['available'] is False
        assert 'not accessible' in (result['reason'] or '')
        assert 'ollama serve' in (result['install_instructions'] or '')

    @pytest.mark.asyncio
    async def test_model_not_found(self) -> None:
        """Test failure when summary model is not pulled."""
        summary_settings = _make_summary_settings()
        embedding_settings = _make_embedding_settings()

        # Mock httpx success
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx_client = MagicMock()
        mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
        mock_httpx_client.__aexit__ = AsyncMock()
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        # Mock ollama client to fail model check
        mock_ollama_client = MagicMock()
        mock_ollama_client.show.side_effect = Exception("model 'qwen3:1.7b' not found")
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = mock_ollama_client

        with (
            patch('importlib.util.find_spec', return_value=MagicMock()),
            patch('httpx.AsyncClient', return_value=mock_httpx_client),
            patch.dict('sys.modules', {'ollama': mock_ollama}),
        ):
            result = await check_summary_provider_dependencies(
                'ollama', summary_settings, embedding_settings,
            )

        assert result['available'] is False
        assert 'not available' in (result['reason'] or '')
        assert 'ollama pull' in (result['install_instructions'] or '')


class TestCheckSummaryOpenAI:
    """Tests for OpenAI summary provider dependency checks."""

    @pytest.mark.asyncio
    async def test_all_available(self) -> None:
        """Test that checks pass when OpenAI deps are available."""
        summary_settings = _make_summary_settings()
        embedding_settings = _make_embedding_settings()

        with (
            patch('importlib.util.find_spec', return_value=MagicMock()),
            patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'}),
        ):
            result = await check_summary_provider_dependencies(
                'openai', summary_settings, embedding_settings,
            )

        assert result['available'] is True
        assert result['reason'] is None

    @pytest.mark.asyncio
    async def test_api_key_missing(self) -> None:
        """Test failure when OPENAI_API_KEY is not set."""
        summary_settings = _make_summary_settings()
        embedding_settings = _make_embedding_settings()

        env = {k: v for k, v in os.environ.items() if k != 'OPENAI_API_KEY'}
        with (
            patch('importlib.util.find_spec', return_value=MagicMock()),
            patch.dict(os.environ, env, clear=True),
        ):
            result = await check_summary_provider_dependencies(
                'openai', summary_settings, embedding_settings,
            )

        assert result['available'] is False
        assert 'OPENAI_API_KEY' in (result['reason'] or '')


class TestCheckSummaryAnthropic:
    """Tests for Anthropic summary provider dependency checks."""

    @pytest.mark.asyncio
    async def test_package_missing(self) -> None:
        """Test failure when langchain-anthropic is not installed."""
        summary_settings = _make_summary_settings()
        embedding_settings = _make_embedding_settings()

        with patch('importlib.util.find_spec', return_value=None):
            result = await check_summary_provider_dependencies(
                'anthropic', summary_settings, embedding_settings,
            )

        assert result['available'] is False
        assert 'not installed' in (result['reason'] or '')
        assert result['install_instructions'] == 'uv sync --extra summary-anthropic'

    @pytest.mark.asyncio
    async def test_api_key_missing(self) -> None:
        """Test failure when ANTHROPIC_API_KEY is not set."""
        summary_settings = _make_summary_settings()
        embedding_settings = _make_embedding_settings()

        env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
        with (
            patch('importlib.util.find_spec', return_value=MagicMock()),
            patch.dict(os.environ, env, clear=True),
        ):
            result = await check_summary_provider_dependencies(
                'anthropic', summary_settings, embedding_settings,
            )

        assert result['available'] is False
        assert 'ANTHROPIC_API_KEY' in (result['reason'] or '')

    @pytest.mark.asyncio
    async def test_all_available(self) -> None:
        """Test that checks pass when Anthropic deps are available."""
        summary_settings = _make_summary_settings()
        embedding_settings = _make_embedding_settings()

        with (
            patch('importlib.util.find_spec', return_value=MagicMock()),
            patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test-key'}),
        ):
            result = await check_summary_provider_dependencies(
                'anthropic', summary_settings, embedding_settings,
            )

        assert result['available'] is True
        assert result['reason'] is None


class TestCheckSummaryUnknownProvider:
    """Tests for unknown summary provider names."""

    @pytest.mark.asyncio
    async def test_unknown_provider(self) -> None:
        """Test that unknown provider name returns not available."""
        summary_settings = _make_summary_settings()
        embedding_settings = _make_embedding_settings()

        result = await check_summary_provider_dependencies(
            'nonexistent', summary_settings, embedding_settings,
        )

        assert result['available'] is False
        assert 'Unknown summary provider' in (result['reason'] or '')


class TestClassifyProviderErrorForSummary:
    """Tests for classify_provider_error with summary-relevant error reasons."""

    def test_model_not_found_returns_configuration_error(self) -> None:
        """Model not found requires human action (ollama pull) -> ConfigurationError."""
        error_class = classify_provider_error(
            "Summary model \"qwen3:1.7b\" not available: model 'qwen3:1.7b' not found",
        )
        assert error_class is ConfigurationError

    def test_package_not_installed_returns_configuration_error(self) -> None:
        """Missing package requires installation -> ConfigurationError."""
        error_class = classify_provider_error('langchain-ollama package not installed')
        assert error_class is ConfigurationError

    def test_api_key_not_set_returns_configuration_error(self) -> None:
        """Missing API key requires configuration -> ConfigurationError."""
        error_class = classify_provider_error(
            'ANTHROPIC_API_KEY environment variable is not set',
        )
        assert error_class is ConfigurationError

    def test_connection_refused_returns_dependency_error(self) -> None:
        """Service not accessible is transient -> DependencyError."""
        error_class = classify_provider_error(
            'Ollama service not accessible at http://localhost:11434: Connection refused',
        )
        assert error_class is DependencyError

    def test_timeout_returns_dependency_error(self) -> None:
        """Timeout is transient -> DependencyError."""
        error_class = classify_provider_error(
            'Ollama service not accessible at http://localhost:11434: timeout',
        )
        assert error_class is DependencyError

    def test_unknown_provider_returns_dependency_error(self) -> None:
        """Unknown summary provider doesn't match config indicators -> DependencyError.

        Note: 'unknown provider' is a config indicator but 'unknown summary provider'
        does not contain that exact substring, so it falls through to DependencyError.
        In practice, the check_summary_provider_dependencies() function catches unknown
        providers before classify_provider_error() is ever called.
        """
        error_class = classify_provider_error("Unknown summary provider: 'nonexistent'")
        assert error_class is DependencyError
