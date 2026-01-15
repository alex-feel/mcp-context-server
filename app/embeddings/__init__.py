"""
Embedding provider layer for semantic search.

This module provides a pluggable embedding provider architecture supporting
multiple backends (Ollama, OpenAI, Azure, HuggingFace, Voyage AI) via LangChain.

Architecture mirrors app/backends/ for consistency:
- base.py: EmbeddingProvider protocol definition
- factory.py: Provider factory with auto-import
- providers/: Concrete provider implementations

Example Usage:
    from app.embeddings import create_embedding_provider, EmbeddingProvider

    provider = create_embedding_provider()  # Uses EMBEDDING_PROVIDER setting
    await provider.initialize()

    embedding = await provider.embed_query("Hello world")
    embeddings = await provider.embed_documents(["Doc 1", "Doc 2"])

    await provider.shutdown()
"""

from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import create_embedding_provider
from app.embeddings.retry import EmbeddingRetryExhaustedError
from app.embeddings.retry import EmbeddingTimeoutError
from app.embeddings.retry import with_retry_and_timeout
from app.embeddings.tracing import traced_embedding

__all__ = [
    'EmbeddingProvider',
    'create_embedding_provider',
    'EmbeddingRetryExhaustedError',
    'EmbeddingTimeoutError',
    'with_retry_and_timeout',
    'traced_embedding',
]
