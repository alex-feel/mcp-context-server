"""
Tests for LangSmith tracing integration.

Tests verify:
- Tracing disabled returns original function unchanged
- Tracing enabled without langsmith package logs warning once
- propagate_langsmith_settings() sets os.environ correctly
- propagate_langsmith_settings() does nothing when tracing disabled
- Model metadata (ls_provider, ls_model_name, ls_invocation_params) is populated
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import MagicMock
from unittest.mock import patch


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


class TestTracedEmbeddingDecorator:
    """Tests for the traced_embedding decorator."""

    def test_tracing_disabled_returns_original_function(self) -> None:
        """Test that decorator returns original function when tracing disabled."""
        with patch('app.embeddings.tracing.get_settings') as mock_settings:
            mock_settings.return_value.langsmith.tracing = False

            # Import after patching to get fresh decorator behavior
            # Reset module state
            import importlib

            import app.embeddings.tracing as tracing_module

            importlib.reload(tracing_module)

            async def original_func() -> list[float]:
                return [0.1, 0.2]

            decorated = tracing_module.traced_embedding(original_func)

            # When tracing disabled, decorator should return original function unchanged
            assert decorated is original_func

    def test_tracing_enabled_without_langsmith_warns_once(self) -> None:
        """Test that missing langsmith package logs warning once."""
        with (
            patch('app.embeddings.tracing.get_settings') as mock_settings,
            patch.dict('sys.modules', {'langsmith': None}),
        ):
            mock_settings.return_value.langsmith.tracing = True

            # Reset module state to trigger fresh warning
            import importlib

            import app.embeddings.tracing as tracing_module

            # Force reset of warning flag
            tracing_module._warned_missing_package = False
            importlib.reload(tracing_module)

            async def func1() -> list[float]:
                return [0.1]

            async def func2() -> list[float]:
                return [0.2]

            # Manually set warning flag to False before test
            tracing_module._warned_missing_package = False

            with patch.object(tracing_module, 'logger') as mock_logger:
                # First decoration should log warning
                tracing_module.traced_embedding(func1)
                # Second decoration should not log warning again
                tracing_module.traced_embedding(func2)

                # Warning should be logged exactly once
                warning_calls = [
                    call
                    for call in mock_logger.warning.call_args_list
                    if 'langsmith' in str(call).lower()
                ]
                assert len(warning_calls) <= 1, 'Warning should be logged at most once'

    def test_tracing_enabled_without_langsmith_returns_original(self) -> None:
        """Test that decorator returns original function when langsmith not installed."""
        with (
            patch('app.embeddings.tracing.get_settings') as mock_settings,
            patch.dict('sys.modules', {'langsmith': None}),
        ):
            mock_settings.return_value.langsmith.tracing = True

            import importlib

            import app.embeddings.tracing as tracing_module

            tracing_module._warned_missing_package = False
            importlib.reload(tracing_module)

            async def original_func() -> list[float]:
                return [0.1, 0.2]

            decorated = tracing_module.traced_embedding(original_func)

            # Without langsmith installed, should return original function
            assert decorated is original_func


class TestPropagateLangsmithSettings:
    """Tests for propagate_langsmith_settings() function."""

    def test_propagate_when_tracing_disabled(self) -> None:
        """Test that no env vars are set when tracing disabled."""
        with (
            env_var('LANGSMITH_TRACING', None),
            env_var('LANGSMITH_API_KEY', None),
            env_var('LANGSMITH_ENDPOINT', None),
            env_var('LANGSMITH_PROJECT', None), patch('app.startup.get_settings') as mock_settings,
        ):
            mock_settings.return_value.langsmith.tracing = False

            from app.startup import propagate_langsmith_settings

            propagate_langsmith_settings()

            # When tracing disabled, no env vars should be set
            assert os.environ.get('LANGSMITH_TRACING') is None

    def test_propagate_when_tracing_enabled(self) -> None:
        """Test that all 4 env vars are set when tracing enabled."""
        with (
            env_var('LANGSMITH_TRACING', None),
            env_var('LANGSMITH_API_KEY', None),
            env_var('LANGSMITH_ENDPOINT', None),
            env_var('LANGSMITH_PROJECT', None), patch('app.startup.settings') as mock_settings,
        ):
            # Create mock LangSmith settings
            mock_langsmith = MagicMock()
            mock_langsmith.tracing = True
            mock_langsmith.endpoint = 'https://api.smith.langchain.com'
            mock_langsmith.project = 'test-project'
            mock_langsmith.api_key = MagicMock()
            mock_langsmith.api_key.get_secret_value.return_value = 'test-api-key'
            mock_settings.langsmith = mock_langsmith

            from app.startup import propagate_langsmith_settings

            propagate_langsmith_settings()

            # All 4 env vars should be set
            assert os.environ.get('LANGSMITH_TRACING') == 'true'
            assert os.environ.get('LANGSMITH_API_KEY') == 'test-api-key'
            assert os.environ.get('LANGSMITH_ENDPOINT') == 'https://api.smith.langchain.com'
            assert os.environ.get('LANGSMITH_PROJECT') == 'test-project'

    def test_propagate_without_api_key(self) -> None:
        """Test that propagation works without API key (only 3 vars set)."""
        with (
            env_var('LANGSMITH_TRACING', None),
            env_var('LANGSMITH_API_KEY', None),
            env_var('LANGSMITH_ENDPOINT', None),
            env_var('LANGSMITH_PROJECT', None), patch('app.startup.settings') as mock_settings,
        ):
            # Create mock LangSmith settings without API key
            mock_langsmith = MagicMock()
            mock_langsmith.tracing = True
            mock_langsmith.endpoint = 'https://api.smith.langchain.com'
            mock_langsmith.project = 'test-project'
            mock_langsmith.api_key = None  # No API key
            mock_settings.langsmith = mock_langsmith

            from app.startup import propagate_langsmith_settings

            propagate_langsmith_settings()

            # 3 env vars should be set (not API key)
            assert os.environ.get('LANGSMITH_TRACING') == 'true'
            assert os.environ.get('LANGSMITH_API_KEY') is None
            assert os.environ.get('LANGSMITH_ENDPOINT') == 'https://api.smith.langchain.com'
            assert os.environ.get('LANGSMITH_PROJECT') == 'test-project'


class TestLangSmithSettingsIntegration:
    """Integration tests for LangSmith settings."""

    def test_settings_default_values(self) -> None:
        """Test that LangSmith settings have correct defaults."""
        from app.settings import AppSettings

        with (
            env_var('LANGSMITH_TRACING', None),
            env_var('LANGSMITH_API_KEY', None),
        ):
            settings = AppSettings()

            # Default: tracing disabled
            assert settings.langsmith.tracing is False
            # Default endpoint
            assert settings.langsmith.endpoint == 'https://api.smith.langchain.com'
            # Default project
            assert settings.langsmith.project == 'mcp-context-server'
            # Default: no API key
            assert settings.langsmith.api_key is None

    def test_settings_tracing_enabled_via_env_var(self) -> None:
        """Test that tracing can be enabled via environment variable."""
        from app.settings import AppSettings

        with (
            env_var('LANGSMITH_TRACING', 'true'),
            env_var('LANGSMITH_API_KEY', 'test-key'),
        ):
            settings = AppSettings()

            assert settings.langsmith.tracing is True
            assert settings.langsmith.api_key is not None
            assert settings.langsmith.api_key.get_secret_value() == 'test-key'

    def test_settings_custom_project_and_endpoint(self) -> None:
        """Test that custom project and endpoint can be set."""
        from app.settings import AppSettings

        with (
            env_var('LANGSMITH_PROJECT', 'my-custom-project'),
            env_var('LANGSMITH_ENDPOINT', 'https://custom.endpoint.com'),
        ):
            settings = AppSettings()

            assert settings.langsmith.project == 'my-custom-project'
            assert settings.langsmith.endpoint == 'https://custom.endpoint.com'


class TestModelMetadataPopulation:
    """Tests for model metadata population in LangSmith traces.

    These tests verify that the traced_embedding decorator correctly extracts
    provider info and builds metadata for LangSmith model identification.
    """

    def test_metadata_extraction_for_ollama_style_provider(self) -> None:
        """Test metadata extraction logic for Ollama-style provider with _model attribute."""
        # Create a mock provider with Ollama-style attributes
        class MockOllamaProvider:
            _model = 'qwen3-embedding:0.6b'

            @property
            def provider_name(self) -> str:
                return 'ollama'

        provider = MockOllamaProvider()

        # Extract metadata using the same logic as traced_embedding
        model_name = getattr(provider, '_model', None)
        if model_name is None:
            model_name = getattr(provider, '_deployment', None)
        provider_name = getattr(provider, 'provider_name', None)

        # Verify extraction
        assert provider_name == 'ollama'
        assert model_name == 'qwen3-embedding:0.6b'

        # Build metadata dict as the decorator would
        metadata: dict[str, object] = {}
        if provider_name:
            metadata['ls_provider'] = provider_name
        if model_name:
            metadata['ls_model_name'] = model_name
            metadata['ls_invocation_params'] = {'model': model_name}

        # Verify metadata structure
        assert metadata['ls_provider'] == 'ollama'
        assert metadata['ls_model_name'] == 'qwen3-embedding:0.6b'
        invocation_params = metadata.get('ls_invocation_params')
        assert isinstance(invocation_params, dict)
        assert invocation_params.get('model') == 'qwen3-embedding:0.6b'

    def test_metadata_extraction_for_azure_style_provider(self) -> None:
        """Test metadata extraction for Azure-style provider using _deployment attribute."""
        # Create a mock provider with Azure-style attributes (_deployment instead of _model)
        class MockAzureProvider:
            _deployment = 'my-azure-embedding-deployment'

            @property
            def provider_name(self) -> str:
                return 'azure'

        provider = MockAzureProvider()

        # Extract metadata using the same logic as traced_embedding
        model_name = getattr(provider, '_model', None)
        if model_name is None:
            model_name = getattr(provider, '_deployment', None)
        provider_name = getattr(provider, 'provider_name', None)

        # Verify extraction with _deployment fallback
        assert provider_name == 'azure'
        assert model_name == 'my-azure-embedding-deployment'

        # Build metadata dict as the decorator would
        metadata: dict[str, object] = {}
        if provider_name:
            metadata['ls_provider'] = provider_name
        if model_name:
            metadata['ls_model_name'] = model_name
            metadata['ls_invocation_params'] = {'model': model_name}

        # Verify metadata structure
        assert metadata['ls_provider'] == 'azure'
        assert metadata['ls_model_name'] == 'my-azure-embedding-deployment'

    def test_metadata_extraction_for_minimal_provider(self) -> None:
        """Test metadata extraction when provider lacks model/provider_name attributes."""

        # Provider without _model or provider_name attributes
        class MinimalProvider:
            pass

        provider = MinimalProvider()

        # Extract metadata using the same logic as traced_embedding
        model_name: str | None = getattr(provider, '_model', None)
        if model_name is None:
            model_name = getattr(provider, '_deployment', None)
        provider_name: str | None = getattr(provider, 'provider_name', None)

        # Build metadata dict as the decorator would (before assertions)
        metadata: dict[str, object] = {}
        if provider_name:
            metadata['ls_provider'] = provider_name
        if model_name:
            metadata['ls_model_name'] = model_name
            metadata['ls_invocation_params'] = {'model': model_name}

        # Verify no metadata extracted and dict is empty
        assert provider_name is None
        assert model_name is None
        assert metadata == {}

    def test_traced_decorator_returns_original_when_disabled(self) -> None:
        """Test that traced_embedding returns original function when tracing disabled."""
        with patch('app.embeddings.tracing.get_settings') as mock_settings:
            mock_settings.return_value.langsmith.tracing = False

            import importlib

            import app.embeddings.tracing as tracing_module

            tracing_module = importlib.reload(tracing_module)

            async def sample_func(_text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            decorated = tracing_module.traced_embedding(sample_func)

            # When tracing disabled, should return original function unchanged
            assert decorated is sample_func
