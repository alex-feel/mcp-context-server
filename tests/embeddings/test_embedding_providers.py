"""Tests for embedding provider implementations.

These tests use mocks to avoid requiring actual provider dependencies.
Integration tests with real providers should be in a separate test file
and marked with appropriate skip markers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


class TestOllamaEmbeddingProvider:
    """Tests for OllamaEmbeddingProvider."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings for Ollama provider."""
        mock = MagicMock()
        mock.embedding.model = 'test-model'
        mock.embedding.ollama_host = 'http://localhost:11434'
        mock.embedding.dim = 768
        return mock

    @pytest.fixture
    def mock_ollama_embeddings(self) -> MagicMock:
        """Create mock OllamaEmbeddings class."""
        mock = MagicMock()
        mock.aembed_query = AsyncMock(return_value=[0.1] * 768)
        mock.aembed_documents = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])
        return mock

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        mock_settings: MagicMock,
        mock_ollama_embeddings: MagicMock,
    ) -> None:
        """Test successful initialization."""
        mock_langchain = MagicMock()
        mock_langchain.OllamaEmbeddings = MagicMock(return_value=mock_ollama_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_ollama': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_ollama.get_settings',
                return_value=mock_settings,
            ),
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            await provider.initialize()

            assert provider._embeddings is not None
            assert provider.provider_name == 'ollama'

    @pytest.mark.asyncio
    async def test_embed_query_success(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test successful embedding query."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.embed_query('test text')

            assert len(result) == 768
            assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_query_dimension_validation(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test dimension validation in embed_query."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 512)  # Wrong dimension

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            with pytest.raises(ValueError, match='Dimension mismatch'):
                await provider.embed_query('test')

    @pytest.mark.asyncio
    async def test_embed_query_not_initialized_raises_error(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test embed_query raises error when not initialized."""
        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()

            with pytest.raises(RuntimeError, match='Provider not initialized'):
                await provider.embed_query('test')

    @pytest.mark.asyncio
    async def test_embed_documents_dimension_validation(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test dimension validation in embed_documents."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(
            return_value=[[0.1] * 768, [0.2] * 512],  # Second has wrong dimension
        )

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            with pytest.raises(ValueError, match='Embedding 1 dimension mismatch'):
                await provider.embed_documents(['text1', 'text2'])

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_not_initialized(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test is_available returns False when provider not initialized."""
        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            result = await provider.is_available()

            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_returns_true_when_working(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test is_available returns True when provider works."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.is_available()

            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_error(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test is_available returns False when API fails."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(side_effect=Exception('Connection failed'))

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.is_available()

            assert result is False

    def test_get_dimension_returns_configured_value(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test get_dimension returns the configured dimension."""
        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()

            assert provider.get_dimension() == 768

    @pytest.mark.asyncio
    async def test_shutdown_clears_embeddings(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test shutdown properly clears the embeddings instance."""
        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()
            provider._embeddings = MagicMock()  # Simulate initialized state

            await provider.shutdown()

            assert provider._embeddings is None

    def test_convert_to_python_floats(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test numpy type conversion utility."""
        # Create mock numpy-like objects
        class MockNumpyFloat:
            def __init__(self, val: float) -> None:
                self._val = val

            def item(self) -> float:
                return self._val

        with patch(
            'app.embeddings.providers.langchain_ollama.get_settings',
            return_value=mock_settings,
        ):
            from app.embeddings.providers.langchain_ollama import OllamaEmbeddingProvider

            provider = OllamaEmbeddingProvider()

            # Test with mix of numpy-like and Python floats
            input_data = [MockNumpyFloat(1.5), 2.5, MockNumpyFloat(3.5)]
            result = provider._convert_to_python_floats(input_data)

            assert result == [1.5, 2.5, 3.5]
            assert all(isinstance(x, float) for x in result)


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAIEmbeddingProvider."""

    @pytest.fixture
    def mock_settings_with_key(self) -> MagicMock:
        """Create mock settings with API key."""
        mock = MagicMock()
        mock.embedding.model = 'text-embedding-3-small'
        mock.embedding.dim = 1536
        mock.embedding.openai_api_key = MagicMock()
        mock.embedding.openai_api_key.get_secret_value.return_value = 'test-api-key'
        mock.embedding.openai_api_base = None
        mock.embedding.openai_organization = None
        return mock

    @pytest.fixture
    def mock_settings_without_key(self) -> MagicMock:
        """Create mock settings without API key."""
        mock = MagicMock()
        mock.embedding.model = 'text-embedding-3-small'
        mock.embedding.dim = 1536
        mock.embedding.openai_api_key = None
        mock.embedding.openai_api_base = None
        mock.embedding.openai_organization = None
        return mock

    @pytest.mark.asyncio
    async def test_initialize_raises_without_api_key(
        self,
        mock_settings_without_key: MagicMock,
    ) -> None:
        """Test initialization fails without API key."""
        mock_langchain = MagicMock()

        with (
            patch.dict('sys.modules', {'langchain_openai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_openai.get_settings',
                return_value=mock_settings_without_key,
            ),
        ):
            from app.embeddings.providers.langchain_openai import OpenAIEmbeddingProvider

            provider = OpenAIEmbeddingProvider()

            with pytest.raises(ValueError, match='OPENAI_API_KEY is required'):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_success_with_api_key(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test successful initialization with API key."""
        mock_langchain = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain.OpenAIEmbeddings = MagicMock(return_value=mock_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_openai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_openai.get_settings',
                return_value=mock_settings_with_key,
            ),
        ):
            from app.embeddings.providers.langchain_openai import OpenAIEmbeddingProvider

            provider = OpenAIEmbeddingProvider()
            await provider.initialize()

            assert provider._embeddings is not None
            assert provider.provider_name == 'openai'

    def test_provider_name(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test provider_name property returns 'openai'."""
        with patch(
            'app.embeddings.providers.langchain_openai.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_openai import OpenAIEmbeddingProvider

            provider = OpenAIEmbeddingProvider()

            assert provider.provider_name == 'openai'


class TestAzureEmbeddingProvider:
    """Tests for AzureEmbeddingProvider."""

    @pytest.fixture
    def mock_settings_incomplete(self) -> MagicMock:
        """Create mock settings with missing Azure settings."""
        mock = MagicMock()
        mock.embedding.dim = 1536
        mock.embedding.azure_openai_api_key = None
        mock.embedding.azure_openai_endpoint = None
        mock.embedding.azure_openai_deployment_name = None
        mock.embedding.azure_openai_api_version = '2024-02-01'
        return mock

    @pytest.fixture
    def mock_settings_complete(self) -> MagicMock:
        """Create mock settings with all Azure settings."""
        mock = MagicMock()
        mock.embedding.dim = 1536
        mock.embedding.azure_openai_api_key = MagicMock()
        mock.embedding.azure_openai_api_key.get_secret_value.return_value = 'test-api-key'
        mock.embedding.azure_openai_endpoint = 'https://test.openai.azure.com'
        mock.embedding.azure_openai_deployment_name = 'test-deployment'
        mock.embedding.azure_openai_api_version = '2024-02-01'
        return mock

    @pytest.mark.asyncio
    async def test_initialize_raises_without_api_key(
        self,
        mock_settings_incomplete: MagicMock,
    ) -> None:
        """Test initialization fails without API key."""
        mock_langchain = MagicMock()

        with (
            patch.dict('sys.modules', {'langchain_openai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_azure.get_settings',
                return_value=mock_settings_incomplete,
            ),
        ):
            from app.embeddings.providers.langchain_azure import AzureEmbeddingProvider

            provider = AzureEmbeddingProvider()

            with pytest.raises(ValueError, match='AZURE_OPENAI_API_KEY is required'):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_raises_without_endpoint(
        self,
        mock_settings_incomplete: MagicMock,
    ) -> None:
        """Test initialization fails without endpoint."""
        mock_settings_incomplete.embedding.azure_openai_api_key = MagicMock()
        mock_settings_incomplete.embedding.azure_openai_api_key.get_secret_value.return_value = 'test-key'
        mock_langchain = MagicMock()

        with (
            patch.dict('sys.modules', {'langchain_openai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_azure.get_settings',
                return_value=mock_settings_incomplete,
            ),
        ):
            from app.embeddings.providers.langchain_azure import AzureEmbeddingProvider

            provider = AzureEmbeddingProvider()

            with pytest.raises(ValueError, match='AZURE_OPENAI_ENDPOINT is required'):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_success_with_complete_settings(
        self,
        mock_settings_complete: MagicMock,
    ) -> None:
        """Test successful initialization with complete settings."""
        mock_langchain = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain.AzureOpenAIEmbeddings = MagicMock(return_value=mock_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_openai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_azure.get_settings',
                return_value=mock_settings_complete,
            ),
        ):
            from app.embeddings.providers.langchain_azure import AzureEmbeddingProvider

            provider = AzureEmbeddingProvider()
            await provider.initialize()

            assert provider._embeddings is not None
            assert provider.provider_name == 'azure'


class TestHuggingFaceEmbeddingProvider:
    """Tests for HuggingFaceEmbeddingProvider."""

    @pytest.fixture
    def mock_settings_with_token(self) -> MagicMock:
        """Create mock settings with API token."""
        mock = MagicMock()
        mock.embedding.model = 'sentence-transformers/all-MiniLM-L6-v2'
        mock.embedding.dim = 384
        mock.embedding.huggingface_api_key = MagicMock()
        mock.embedding.huggingface_api_key.get_secret_value.return_value = 'test-token'
        return mock

    @pytest.fixture
    def mock_settings_without_token(self) -> MagicMock:
        """Create mock settings without API token."""
        mock = MagicMock()
        mock.embedding.model = 'sentence-transformers/all-MiniLM-L6-v2'
        mock.embedding.dim = 384
        mock.embedding.huggingface_api_key = None
        return mock

    @pytest.mark.asyncio
    async def test_initialize_raises_without_token(
        self,
        mock_settings_without_token: MagicMock,
    ) -> None:
        """Test initialization fails without API token."""
        mock_langchain = MagicMock()

        with (
            patch.dict('sys.modules', {'langchain_huggingface': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_huggingface.get_settings',
                return_value=mock_settings_without_token,
            ),
        ):
            from app.embeddings.providers.langchain_huggingface import HuggingFaceEmbeddingProvider

            provider = HuggingFaceEmbeddingProvider()

            with pytest.raises(ValueError, match='HUGGINGFACEHUB_API_TOKEN is required'):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_success_with_token(
        self,
        mock_settings_with_token: MagicMock,
    ) -> None:
        """Test successful initialization with API token."""
        mock_langchain = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain.HuggingFaceEndpointEmbeddings = MagicMock(return_value=mock_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_huggingface': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_huggingface.get_settings',
                return_value=mock_settings_with_token,
            ),
        ):
            from app.embeddings.providers.langchain_huggingface import HuggingFaceEmbeddingProvider

            provider = HuggingFaceEmbeddingProvider()
            await provider.initialize()

            assert provider._embeddings is not None
            assert provider.provider_name == 'huggingface'


class TestVoyageEmbeddingProvider:
    """Tests for VoyageEmbeddingProvider."""

    @pytest.fixture
    def mock_settings_with_key(self) -> MagicMock:
        """Create mock settings with API key."""
        mock = MagicMock()
        mock.embedding.model = 'voyage-3'
        mock.embedding.dim = 1024
        mock.embedding.voyage_api_key = MagicMock()
        mock.embedding.voyage_api_key.get_secret_value.return_value = 'test-voyage-key'
        mock.embedding.voyage_truncation = None
        mock.embedding.voyage_batch_size = 7
        return mock

    @pytest.fixture
    def mock_settings_without_key(self) -> MagicMock:
        """Create mock settings without API key."""
        mock = MagicMock()
        mock.embedding.model = 'voyage-3'
        mock.embedding.dim = 1024
        mock.embedding.voyage_api_key = None
        mock.embedding.voyage_truncation = None
        mock.embedding.voyage_batch_size = 7
        return mock

    @pytest.mark.asyncio
    async def test_initialize_raises_without_api_key(
        self,
        mock_settings_without_key: MagicMock,
    ) -> None:
        """Test initialization fails without API key."""
        mock_langchain = MagicMock()

        with (
            patch.dict('sys.modules', {'langchain_voyageai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_voyage.get_settings',
                return_value=mock_settings_without_key,
            ),
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()

            with pytest.raises(ValueError, match='VOYAGE_API_KEY is required'):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_success_with_api_key(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test successful initialization with API key."""
        mock_langchain = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain.VoyageAIEmbeddings = MagicMock(return_value=mock_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_voyageai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_voyage.get_settings',
                return_value=mock_settings_with_key,
            ),
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            await provider.initialize()

            assert provider._embeddings is not None
            assert provider.provider_name == 'voyage'

    @pytest.mark.asyncio
    async def test_embed_query_dimension_validation(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test dimension validation in embed_query."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 512)  # Wrong dimension

        with patch(
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            provider._embeddings = mock_embeddings

            with pytest.raises(ValueError, match='Dimension mismatch'):
                await provider.embed_query('test')

    @pytest.mark.asyncio
    async def test_embed_query_success(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test successful embedding query."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 1024)

        with patch(
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.embed_query('test text')

            assert len(result) == 1024
            assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_documents_success(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test successful batch embedding."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(
            return_value=[[0.1] * 1024, [0.2] * 1024],
        )

        with patch(
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            provider._embeddings = mock_embeddings

            result = await provider.embed_documents(['text1', 'text2'])

            assert len(result) == 2
            assert len(result[0]) == 1024
            assert len(result[1]) == 1024

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_not_initialized(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test is_available returns False when provider not initialized."""
        with patch(
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            result = await provider.is_available()

            assert result is False

    def test_get_dimension_returns_configured_value(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test get_dimension returns the configured dimension."""
        with patch(
            'app.embeddings.providers.langchain_voyage.get_settings',
            return_value=mock_settings_with_key,
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()

            assert provider.get_dimension() == 1024

    @pytest.mark.asyncio
    async def test_truncation_setting_applied(
        self,
        mock_settings_with_key: MagicMock,
    ) -> None:
        """Test truncation setting is passed to VoyageAIEmbeddings."""
        mock_settings_with_key.embedding.voyage_truncation = True

        mock_langchain = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain.VoyageAIEmbeddings = MagicMock(return_value=mock_embeddings)

        with (
            patch.dict('sys.modules', {'langchain_voyageai': mock_langchain}),
            patch(
                'app.embeddings.providers.langchain_voyage.get_settings',
                return_value=mock_settings_with_key,
            ),
        ):
            from app.embeddings.providers.langchain_voyage import VoyageEmbeddingProvider

            provider = VoyageEmbeddingProvider()
            await provider.initialize()

            # Verify truncation was passed in kwargs
            call_kwargs = mock_langchain.VoyageAIEmbeddings.call_args[1]
            assert 'truncation' in call_kwargs
            assert call_kwargs['truncation'] is True
