"""Tests for embedding provider factory.

This module tests the create_embedding_provider factory function
and its provider mappings.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.embeddings.factory import PROVIDER_CLASSES
from app.embeddings.factory import PROVIDER_INSTALL_INSTRUCTIONS
from app.embeddings.factory import PROVIDER_MODULES
from app.embeddings.factory import create_embedding_provider


class TestProviderMappings:
    """Tests for provider mapping dictionaries."""

    def test_provider_modules_mapping_complete(self) -> None:
        """Verify all providers have module mappings."""
        expected_providers = {'ollama', 'openai', 'azure', 'huggingface', 'voyage'}
        assert set(PROVIDER_MODULES.keys()) == expected_providers

    def test_provider_classes_mapping_complete(self) -> None:
        """Verify all providers have class mappings."""
        expected_providers = {'ollama', 'openai', 'azure', 'huggingface', 'voyage'}
        assert set(PROVIDER_CLASSES.keys()) == expected_providers

    def test_provider_install_instructions_complete(self) -> None:
        """Verify all providers have install instructions."""
        expected_providers = {'ollama', 'openai', 'azure', 'huggingface', 'voyage'}
        assert set(PROVIDER_INSTALL_INSTRUCTIONS.keys()) == expected_providers

    def test_all_mappings_have_same_keys(self) -> None:
        """Verify all provider mappings have identical key sets."""
        assert set(PROVIDER_MODULES.keys()) == set(PROVIDER_CLASSES.keys())
        assert set(PROVIDER_MODULES.keys()) == set(PROVIDER_INSTALL_INSTRUCTIONS.keys())

    @pytest.mark.parametrize(
        ('provider', 'expected_class'),
        [
            ('ollama', 'OllamaEmbeddingProvider'),
            ('openai', 'OpenAIEmbeddingProvider'),
            ('azure', 'AzureEmbeddingProvider'),
            ('huggingface', 'HuggingFaceEmbeddingProvider'),
            ('voyage', 'VoyageEmbeddingProvider'),
        ],
    )
    def test_provider_class_names(self, provider: str, expected_class: str) -> None:
        """Verify provider class name mappings are correct."""
        assert PROVIDER_CLASSES[provider] == expected_class

    @pytest.mark.parametrize(
        ('provider', 'expected_module'),
        [
            ('ollama', 'app.embeddings.providers.langchain_ollama'),
            ('openai', 'app.embeddings.providers.langchain_openai'),
            ('azure', 'app.embeddings.providers.langchain_azure'),
            ('huggingface', 'app.embeddings.providers.langchain_huggingface'),
            ('voyage', 'app.embeddings.providers.langchain_voyage'),
        ],
    )
    def test_provider_module_paths(self, provider: str, expected_module: str) -> None:
        """Verify provider module path mappings are correct."""
        assert PROVIDER_MODULES[provider] == expected_module


class TestCreateEmbeddingProvider:
    """Tests for create_embedding_provider function."""

    def test_create_provider_invalid_provider_raises_value_error(self) -> None:
        """Test factory raises ValueError for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported embedding provider: 'invalid'"):
            create_embedding_provider('invalid')

    def test_create_provider_invalid_provider_lists_supported(self) -> None:
        """Test error message includes list of supported providers."""
        with pytest.raises(ValueError, match='Supported providers:'):
            create_embedding_provider('nonexistent')

    def test_create_provider_import_error_includes_install_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test factory provides helpful error message on ImportError."""
        import importlib

        def mock_import_module(name: str) -> None:
            raise ImportError(f'No module named {name}')

        monkeypatch.setattr(importlib, 'import_module', mock_import_module)

        with pytest.raises(ImportError, match='uv sync --extra embeddings-ollama'):
            create_embedding_provider('ollama')

    @pytest.mark.parametrize(
        ('provider', 'expected_install'),
        [
            ('ollama', 'uv sync --extra embeddings-ollama'),
            ('openai', 'uv sync --extra embeddings-openai'),
            ('azure', 'uv sync --extra embeddings-azure'),
            ('huggingface', 'uv sync --extra embeddings-huggingface'),
            ('voyage', 'uv sync --extra embeddings-voyage'),
        ],
    )
    def test_import_error_messages_per_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        expected_install: str,
    ) -> None:
        """Test each provider shows correct install command on ImportError."""
        import importlib

        def mock_import_module(name: str) -> None:
            raise ImportError(f'No module named {name}')

        monkeypatch.setattr(importlib, 'import_module', mock_import_module)

        with pytest.raises(ImportError, match=expected_install):
            create_embedding_provider(provider)

    def test_create_provider_uses_settings_when_no_override(self) -> None:
        """Test factory uses settings.embedding.provider when not overridden."""
        mock_settings = MagicMock()
        mock_settings.embedding.provider = 'openai'

        with (
            patch('app.embeddings.factory.get_settings', return_value=mock_settings),
            patch('app.embeddings.factory.importlib.import_module') as mock_import,
        ):
            # Create a mock module and class
            mock_module = MagicMock()
            mock_class = MagicMock()
            mock_module.OpenAIEmbeddingProvider = mock_class
            mock_import.return_value = mock_module

            create_embedding_provider()

            mock_import.assert_called_once_with('app.embeddings.providers.langchain_openai')

    def test_create_provider_override_ignores_settings(self) -> None:
        """Test explicit provider parameter overrides settings."""
        mock_settings = MagicMock()
        mock_settings.embedding.provider = 'ollama'  # Settings say ollama

        with (
            patch('app.embeddings.factory.get_settings', return_value=mock_settings),
            patch('app.embeddings.factory.importlib.import_module') as mock_import,
        ):
            mock_module = MagicMock()
            mock_class = MagicMock()
            mock_module.VoyageEmbeddingProvider = mock_class
            mock_import.return_value = mock_module

            create_embedding_provider('voyage')  # Override to voyage

            mock_import.assert_called_once_with('app.embeddings.providers.langchain_voyage')

    def test_create_provider_returns_correct_instance_type(self) -> None:
        """Test factory returns provider instance from correct class."""
        mock_settings = MagicMock()
        mock_settings.embedding.provider = 'huggingface'

        with (
            patch('app.embeddings.factory.get_settings', return_value=mock_settings),
            patch('app.embeddings.factory.importlib.import_module') as mock_import,
        ):
            mock_module = MagicMock()
            mock_provider = MagicMock()
            mock_provider.provider_name = 'huggingface'
            mock_class = MagicMock(return_value=mock_provider)
            mock_module.HuggingFaceEmbeddingProvider = mock_class
            mock_import.return_value = mock_module

            provider = create_embedding_provider()

            mock_class.assert_called_once()
            assert provider.provider_name == 'huggingface'

    def test_create_provider_backward_compatibility_no_embedding_settings(self) -> None:
        """Test factory defaults to 'ollama' when embedding settings not available."""
        mock_settings = MagicMock(spec=[])  # No embedding attribute

        with (
            patch('app.embeddings.factory.get_settings', return_value=mock_settings),
            patch('app.embeddings.factory.importlib.import_module') as mock_import,
        ):
            mock_module = MagicMock()
            mock_class = MagicMock()
            mock_module.OllamaEmbeddingProvider = mock_class
            mock_import.return_value = mock_module

            create_embedding_provider()

            mock_import.assert_called_once_with('app.embeddings.providers.langchain_ollama')
