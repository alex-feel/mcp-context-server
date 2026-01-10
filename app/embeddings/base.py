"""
Base protocol definition for embedding providers.

This module defines the EmbeddingProvider Protocol that all provider implementations
must follow. The protocol ensures type-safe, provider-agnostic interfaces for
embedding generation.

The protocol uses @runtime_checkable to enable isinstance() checks.
"""

from __future__ import annotations

from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol defining the interface for embedding provider implementations.

    All embedding providers (Ollama, OpenAI, Azure, HuggingFace, Voyage AI) must implement
    this protocol to ensure compatibility with the embedding repository layer.

    This protocol mirrors the architecture pattern from app/backends/base.py
    for consistency across the codebase.

    Example Implementation:
        class OpenAIEmbeddingProvider:
            async def initialize(self) -> None:
                # Initialize OpenAI client
                pass

            async def shutdown(self) -> None:
                # Cleanup resources
                pass

            async def embed_query(self, text: str) -> list[float]:
                # Generate single embedding
                return await self._embeddings.aembed_query(text)

            async def embed_documents(self, texts: list[str]) -> list[list[float]]:
                # Generate batch embeddings
                return await self._embeddings.aembed_documents(texts)

            async def is_available(self) -> bool:
                # Health check
                return True

            def get_dimension(self) -> int:
                return self._dimension

            @property
            def provider_name(self) -> str:
                return 'openai'
    """

    async def initialize(self) -> None:
        """Initialize the embedding provider.

        Called once during server startup to establish connections,
        validate configuration, and perform any necessary setup.

        Raises:
            RuntimeError: If initialization fails
            ImportError: If required dependencies not installed
        """
        ...

    async def shutdown(self) -> None:
        """Gracefully shut down the embedding provider.

        Called during server shutdown to close connections,
        cancel background tasks, and release resources.
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            RuntimeError: If embedding generation fails
        """
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If embedding generation fails
        """
        ...

    async def is_available(self) -> bool:
        """Check if embedding provider is available.

        Returns:
            True if provider is ready to generate embeddings
        """
        ...

    def get_dimension(self) -> int:
        """Get embedding vector dimension.

        Returns:
            Dimension of embedding vectors (e.g., 768, 1536)
        """
        ...

    @property
    def provider_name(self) -> str:
        """Get provider identifier.

        Returns:
            Provider name string (e.g., 'ollama', 'openai')
        """
        ...
