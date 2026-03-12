"""Protocol defining the interface for summary provider implementations.

All summary providers (Ollama, OpenAI, Anthropic) must implement
this protocol for compatibility with the summary generation layer.

Architecture mirrors app/embeddings/base.py for consistency.
"""

from __future__ import annotations

from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class SummaryProvider(Protocol):
    """Protocol defining the interface for summary provider implementations.

    All summary providers (Ollama, OpenAI, Anthropic) must implement
    this protocol to ensure compatibility with the summary generation layer.

    This protocol mirrors the EmbeddingProvider architecture from
    app/embeddings/base.py for consistency across the codebase.

    Example Implementation:
        class OllamaSummaryProvider:
            async def initialize(self) -> None:
                # Initialize ChatOllama client
                pass

            async def shutdown(self) -> None:
                # Cleanup resources
                pass

            async def summarize(self, text: str) -> str:
                # Generate summary for input text
                return summary_text

            async def is_available(self) -> bool:
                # Health check
                return True

            @property
            def provider_name(self) -> str:
                return 'ollama'
    """

    async def initialize(self) -> None:
        """Initialize the summary provider.

        Called once during server startup to establish connections,
        validate configuration, and perform any necessary setup.

        Raises:
            RuntimeError: If initialization fails
            ImportError: If required dependencies not installed
        """
        ...

    async def shutdown(self) -> None:
        """Gracefully shut down the summary provider.

        Called during server shutdown to close connections,
        cancel background tasks, and release resources.
        """
        ...

    async def summarize(self, text: str) -> str:
        """Generate a summary for the given text.

        Args:
            text: Text content to summarize

        Returns:
            Summary string (single paragraph, no labels/prefixes)

        Raises:
            RuntimeError: If summary generation fails
        """
        ...

    async def is_available(self) -> bool:
        """Check if summary provider is available.

        Returns:
            True if provider is ready to generate summaries
        """
        ...

    @property
    def provider_name(self) -> str:
        """Get provider identifier.

        Returns:
            Provider name string (e.g., 'ollama', 'openai', 'anthropic')
        """
        ...
