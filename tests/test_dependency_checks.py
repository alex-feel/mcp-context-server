"""Tests for provider dependency checking functions.

This module tests dependency check functions in app/server.py that verify
availability of embedding providers and vector storage backends.

P1 Priority: These functions have NO test coverage but are called during startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    pass


class TestCheckVectorStorageDependencies:
    """Tests for check_vector_storage_dependencies()."""

    @pytest.mark.asyncio
    async def test_returns_true_when_all_available_sqlite(self) -> None:
        """Test with numpy and sqlite_vec available for SQLite backend."""
        with patch('importlib.util.find_spec') as mock_find_spec:
            # All specs found
            mock_find_spec.return_value = MagicMock()

            # Mock sqlite_vec extension loading
            mock_sqlite_vec = MagicMock()

            with patch.dict('sys.modules', {'sqlite_vec': mock_sqlite_vec}):
                from app.server import check_vector_storage_dependencies

                result = await check_vector_storage_dependencies(backend_type='sqlite')
                assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_numpy_missing(self) -> None:
        """Test with numpy not installed."""

        def find_spec_side_effect(name: str) -> MagicMock | None:
            if name == 'numpy':
                return None
            return MagicMock()

        with patch('importlib.util.find_spec', side_effect=find_spec_side_effect):
            from app.server import check_vector_storage_dependencies

            result = await check_vector_storage_dependencies(backend_type='sqlite')
            assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_sqlite_vec_missing(self) -> None:
        """Test with sqlite_vec not installed."""

        def find_spec_side_effect(name: str) -> MagicMock | None:
            if name == 'sqlite_vec':
                return None
            return MagicMock()

        with patch('importlib.util.find_spec', side_effect=find_spec_side_effect):
            from app.server import check_vector_storage_dependencies

            result = await check_vector_storage_dependencies(backend_type='sqlite')
            assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_extension_load_fails(self) -> None:
        """Test when sqlite-vec extension fails to load."""
        with patch('importlib.util.find_spec', return_value=MagicMock()):
            # Mock sqlite_vec to raise exception on load
            mock_sqlite_vec = MagicMock()
            mock_sqlite_vec.load.side_effect = Exception('Extension load failed')

            with patch.dict('sys.modules', {'sqlite_vec': mock_sqlite_vec}):
                from app.server import check_vector_storage_dependencies

                result = await check_vector_storage_dependencies(backend_type='sqlite')
                assert result is False

    @pytest.mark.asyncio
    async def test_postgresql_checks_pgvector(self) -> None:
        """Test PostgreSQL backend checks pgvector."""

        def find_spec_side_effect(name: str) -> MagicMock | None:
            if name == 'pgvector':
                return None
            return MagicMock()

        with patch('importlib.util.find_spec', side_effect=find_spec_side_effect):
            from app.server import check_vector_storage_dependencies

            result = await check_vector_storage_dependencies(backend_type='postgresql')
            assert result is False

    @pytest.mark.asyncio
    async def test_postgresql_returns_true_when_available(self) -> None:
        """Test PostgreSQL backend returns True when pgvector available."""
        with patch('importlib.util.find_spec', return_value=MagicMock()):
            from app.server import check_vector_storage_dependencies

            result = await check_vector_storage_dependencies(backend_type='postgresql')
            assert result is True


class TestCheckProviderDependencies:
    """Tests for check_provider_dependencies()."""

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_unavailable(self) -> None:
        """Test unknown provider name returns unavailable."""
        mock_settings = MagicMock()

        from app.server import check_provider_dependencies

        result = await check_provider_dependencies('unknown_provider', mock_settings)

        assert result['available'] is False
        assert 'Unknown provider' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_dispatches_to_ollama_check(self) -> None:
        """Test dispatcher routes to Ollama check function."""
        mock_settings = MagicMock()
        mock_settings.ollama_host = 'http://localhost:11434'
        mock_settings.model = 'test-model'

        with patch('importlib.util.find_spec', return_value=None):
            from app.server import check_provider_dependencies

            result = await check_provider_dependencies('ollama', mock_settings)

            # Should fail because langchain_ollama not installed
            assert result['available'] is False
            assert 'langchain-ollama' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_dispatches_to_openai_check(self) -> None:
        """Test dispatcher routes to OpenAI check function."""
        mock_settings = MagicMock()
        mock_settings.openai_api_key = None

        with patch('importlib.util.find_spec', return_value=MagicMock()):
            from app.server import check_provider_dependencies

            result = await check_provider_dependencies('openai', mock_settings)

            # Should fail because API key not set
            assert result['available'] is False
            assert 'OPENAI_API_KEY' in str(result['reason'])


class TestOllamaDependencies:
    """Tests for _check_ollama_dependencies()."""

    @pytest.mark.asyncio
    async def test_package_not_installed(self) -> None:
        """Test when langchain-ollama not installed."""
        mock_settings = MagicMock()

        with patch('importlib.util.find_spec', return_value=None):
            from app.server import _check_ollama_dependencies

            result = await _check_ollama_dependencies(mock_settings)

            assert result['available'] is False
            assert 'langchain-ollama' in str(result['reason'])
            assert result['install_instructions'] is not None

    @pytest.mark.asyncio
    async def test_service_not_running(self) -> None:
        """Test when Ollama service unreachable."""
        from app.server import _check_ollama_dependencies

        mock_settings = MagicMock()
        mock_settings.ollama_host = 'http://localhost:11434'
        mock_settings.model = 'test-model'

        # Create async context manager mock that raises on get()
        mock_httpx_client = MagicMock()
        mock_httpx_client.get = AsyncMock(side_effect=Exception('Connection refused'))

        # Make it work as async context manager
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_httpx_client
        async_cm.__aexit__.return_value = None

        with (
            patch('importlib.util.find_spec', return_value=MagicMock()),
            patch('httpx.AsyncClient', return_value=async_cm),
        ):
            result = await _check_ollama_dependencies(mock_settings)

            assert result['available'] is False
            assert 'not accessible' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_model_not_available(self) -> None:
        """Test when embedding model not pulled."""
        mock_settings = MagicMock()
        mock_settings.ollama_host = 'http://localhost:11434'
        mock_settings.model = 'missing-model'

        with patch('importlib.util.find_spec', return_value=MagicMock()):
            # Mock httpx success
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_httpx_client = MagicMock()
            mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
            mock_httpx_client.__aexit__ = AsyncMock()
            mock_httpx_client.get = AsyncMock(return_value=mock_response)

            # Mock ollama client to fail model check
            mock_ollama_client = MagicMock()
            mock_ollama_client.show.side_effect = Exception('Model not found')
            mock_ollama = MagicMock()
            mock_ollama.Client.return_value = mock_ollama_client

            with (
                patch('httpx.AsyncClient', return_value=mock_httpx_client),
                patch.dict('sys.modules', {'ollama': mock_ollama}),
            ):
                from app.server import _check_ollama_dependencies

                result = await _check_ollama_dependencies(mock_settings)

                assert result['available'] is False
                assert 'not available' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_all_dependencies_available(self) -> None:
        """Test successful check with mocked Ollama."""
        mock_settings = MagicMock()
        mock_settings.ollama_host = 'http://localhost:11434'
        mock_settings.model = 'test-model'

        with patch('importlib.util.find_spec', return_value=MagicMock()):
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
                patch('httpx.AsyncClient', return_value=mock_httpx_client),
                patch.dict('sys.modules', {'ollama': mock_ollama}),
            ):
                from app.server import _check_ollama_dependencies

                result = await _check_ollama_dependencies(mock_settings)

                assert result['available'] is True
                assert result['reason'] is None


class TestOpenAIDependencies:
    """Tests for _check_openai_dependencies()."""

    @pytest.mark.asyncio
    async def test_package_not_installed(self) -> None:
        """Test when langchain-openai not installed."""
        mock_settings = MagicMock()

        with patch('importlib.util.find_spec', return_value=None):
            from app.server import _check_openai_dependencies

            result = await _check_openai_dependencies(mock_settings)

            assert result['available'] is False
            assert 'langchain-openai' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_api_key_not_set(self) -> None:
        """Test when OPENAI_API_KEY not set."""
        mock_settings = MagicMock()
        mock_settings.openai_api_key = None

        with patch('importlib.util.find_spec', return_value=MagicMock()):
            from app.server import _check_openai_dependencies

            result = await _check_openai_dependencies(mock_settings)

            assert result['available'] is False
            assert 'OPENAI_API_KEY' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_all_dependencies_available(self) -> None:
        """Test successful check with API key set."""
        mock_settings = MagicMock()
        mock_settings.openai_api_key = 'test-api-key'

        with patch('importlib.util.find_spec', return_value=MagicMock()):
            from app.server import _check_openai_dependencies

            result = await _check_openai_dependencies(mock_settings)

            assert result['available'] is True


class TestAzureDependencies:
    """Tests for _check_azure_dependencies()."""

    @pytest.mark.asyncio
    async def test_package_not_installed(self) -> None:
        """Test when langchain-openai not installed."""
        mock_settings = MagicMock()

        with patch('importlib.util.find_spec', return_value=None):
            from app.server import _check_azure_dependencies

            result = await _check_azure_dependencies(mock_settings)

            assert result['available'] is False
            assert 'langchain-openai' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_missing_required_vars(self) -> None:
        """Test missing AZURE_* environment variables."""
        mock_settings = MagicMock()
        mock_settings.azure_openai_api_key = None
        mock_settings.azure_openai_endpoint = None
        mock_settings.azure_openai_deployment_name = None

        with patch('importlib.util.find_spec', return_value=MagicMock()):
            from app.server import _check_azure_dependencies

            result = await _check_azure_dependencies(mock_settings)

            assert result['available'] is False
            assert 'AZURE_OPENAI_API_KEY' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_partial_vars_missing(self) -> None:
        """Test only some Azure vars set."""
        mock_settings = MagicMock()
        mock_settings.azure_openai_api_key = 'test-key'
        mock_settings.azure_openai_endpoint = None
        mock_settings.azure_openai_deployment_name = 'deployment'

        with patch('importlib.util.find_spec', return_value=MagicMock()):
            from app.server import _check_azure_dependencies

            result = await _check_azure_dependencies(mock_settings)

            assert result['available'] is False
            assert 'AZURE_OPENAI_ENDPOINT' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_all_dependencies_available(self) -> None:
        """Test successful check with all Azure vars set."""
        mock_settings = MagicMock()
        mock_settings.azure_openai_api_key = 'test-key'
        mock_settings.azure_openai_endpoint = 'https://test.openai.azure.com'
        mock_settings.azure_openai_deployment_name = 'test-deployment'

        with patch('importlib.util.find_spec', return_value=MagicMock()):
            from app.server import _check_azure_dependencies

            result = await _check_azure_dependencies(mock_settings)

            assert result['available'] is True


class TestHuggingFaceDependencies:
    """Tests for _check_huggingface_dependencies()."""

    @pytest.mark.asyncio
    async def test_package_not_installed(self) -> None:
        """Test when langchain-huggingface not installed."""
        mock_settings = MagicMock()

        with patch('importlib.util.find_spec', return_value=None):
            from app.server import _check_huggingface_dependencies

            result = await _check_huggingface_dependencies(mock_settings)

            assert result['available'] is False
            assert 'langchain-huggingface' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_api_token_not_set(self) -> None:
        """Test when HUGGINGFACEHUB_API_TOKEN not set."""
        mock_settings = MagicMock()
        mock_settings.huggingface_api_key = None

        with patch('importlib.util.find_spec', return_value=MagicMock()):
            from app.server import _check_huggingface_dependencies

            result = await _check_huggingface_dependencies(mock_settings)

            assert result['available'] is False
            assert 'HUGGINGFACEHUB_API_TOKEN' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_all_dependencies_available(self) -> None:
        """Test successful check with token set."""
        mock_settings = MagicMock()
        mock_settings.huggingface_api_key = 'test-token'

        with patch('importlib.util.find_spec', return_value=MagicMock()):
            from app.server import _check_huggingface_dependencies

            result = await _check_huggingface_dependencies(mock_settings)

            assert result['available'] is True


class TestVoyageDependencies:
    """Tests for _check_voyage_dependencies()."""

    @pytest.mark.asyncio
    async def test_package_not_installed(self) -> None:
        """Test when langchain-voyageai not installed."""
        mock_settings = MagicMock()

        with patch('importlib.util.find_spec', return_value=None):
            from app.server import _check_voyage_dependencies

            result = await _check_voyage_dependencies(mock_settings)

            assert result['available'] is False
            assert 'langchain-voyageai' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_api_key_not_set(self) -> None:
        """Test when VOYAGE_API_KEY not set."""
        mock_settings = MagicMock()
        mock_settings.voyage_api_key = None

        with patch('importlib.util.find_spec', return_value=MagicMock()):
            from app.server import _check_voyage_dependencies

            result = await _check_voyage_dependencies(mock_settings)

            assert result['available'] is False
            assert 'VOYAGE_API_KEY' in str(result['reason'])

    @pytest.mark.asyncio
    async def test_all_dependencies_available(self) -> None:
        """Test successful check with API key set."""
        mock_settings = MagicMock()
        mock_settings.voyage_api_key = 'test-api-key'

        with patch('importlib.util.find_spec', return_value=MagicMock()):
            from app.server import _check_voyage_dependencies

            result = await _check_voyage_dependencies(mock_settings)

            assert result['available'] is True
