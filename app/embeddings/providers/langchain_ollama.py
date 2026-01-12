"""
Ollama embedding provider using LangChain integration.

This provider uses langchain-ollama package for native async support
and consistent interface with other LangChain providers.
"""

from __future__ import annotations

import logging
from typing import Any

from app.embeddings.retry import with_retry_and_timeout
from app.settings import get_settings

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider:
    """Ollama embedding provider using LangChain integration.

    Implements the EmbeddingProvider protocol for Ollama models.
    Uses LangChain's OllamaEmbeddings for native async support.

    Environment Variables:
        EMBEDDING_PROVIDER: Must be 'ollama' (default)
        OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
        EMBEDDING_MODEL: Model name (default: qwen3-embedding:0.6b)
        EMBEDDING_DIM: Vector dimensions (default: 1024)
    """

    def __init__(self) -> None:
        """Initialize provider configuration from settings."""
        settings = get_settings()
        self._model = settings.embedding.model
        self._base_url = settings.embedding.ollama_host
        self._dimension = settings.embedding.dim
        self._embeddings: Any = None

    async def initialize(self) -> None:
        """Initialize LangChain OllamaEmbeddings client.

        Raises:
            ImportError: If langchain-ollama is not installed
        """
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as e:
            raise ImportError(
                'langchain-ollama package required. '
                'Install with: uv sync --extra embeddings-ollama',
            ) from e

        # OllamaEmbeddings has no built-in retry
        # Universal wrapper handles all retry logic
        self._embeddings = OllamaEmbeddings(
            model=self._model,
            base_url=self._base_url,
        )
        logger.info(f'Initialized Ollama embedding provider: {self._model} at {self._base_url}')

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._embeddings = None
        logger.info('Ollama embedding provider shut down')

    async def embed_query(self, text: str) -> list[float]:
        """Generate single embedding using async method.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            RuntimeError: If provider not initialized
            ValueError: If embedding dimension mismatch
        """
        if self._embeddings is None:
            raise RuntimeError('Provider not initialized. Call initialize() first.')

        async def _embed() -> list[Any]:
            result: list[Any] = await self._embeddings.aembed_query(text)
            return result

        embedding = await with_retry_and_timeout(_embed, f'{self.provider_name}_embed_query')

        # Convert numpy types to Python float if needed
        embedding = self._convert_to_python_floats(embedding)

        # Validate dimension
        if len(embedding) != self._dimension:
            raise ValueError(
                f'Dimension mismatch: expected {self._dimension}, '
                f'got {len(embedding)}. Check EMBEDDING_DIM setting.',
            )

        return embedding

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate batch embeddings using async method.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If provider not initialized
            ValueError: If any embedding dimension mismatch
        """
        if self._embeddings is None:
            raise RuntimeError('Provider not initialized. Call initialize() first.')

        async def _embed() -> list[list[Any]]:
            result: list[list[Any]] = await self._embeddings.aembed_documents(texts)
            return result

        embeddings = await with_retry_and_timeout(_embed, f'{self.provider_name}_embed_documents')

        # Convert numpy types and validate dimensions
        result: list[list[float]] = []
        for i, emb in enumerate(embeddings):
            emb = self._convert_to_python_floats(emb)
            if len(emb) != self._dimension:
                raise ValueError(
                    f'Embedding {i} dimension mismatch: '
                    f'expected {self._dimension}, got {len(emb)}',
                )
            result.append(emb)

        return result

    async def is_available(self) -> bool:
        """Check if Ollama model is available.

        Returns:
            True if provider is ready to generate embeddings
        """
        if self._embeddings is None:
            return False

        try:
            # Quick test embedding
            await self._embeddings.aembed_query('test')
            return True
        except Exception as e:
            logger.warning(f'Ollama embedding not available: {e}')
            return False

    def get_dimension(self) -> int:
        """Return configured embedding dimension."""
        return self._dimension

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return 'ollama'

    @staticmethod
    def _convert_to_python_floats(embedding: list[Any]) -> list[float]:
        """Convert numpy.float32 or similar to Python float.

        asyncpg with pgvector requires Python float, not numpy.float32.

        Args:
            embedding: Embedding vector potentially containing numpy types

        Returns:
            Embedding vector with Python float values
        """
        return [
            x.item() if hasattr(x, 'item') else float(x)
            for x in embedding
        ]
