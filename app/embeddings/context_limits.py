"""
Context window limits for embedding models by provider.

This module provides reference data for embedding model context windows.
Used for validation, user warnings, and documentation purposes.
Providers handle their own runtime enforcement.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Specification for an embedding model."""

    provider: str
    model: str
    max_tokens: int
    dimension: int
    truncation_behavior: Literal['error', 'silent', 'configurable']
    notes: str = ''


# Known embedding model specifications
EMBEDDING_MODEL_SPECS: dict[str, EmbeddingModelSpec] = {
    # Ollama models
    'qwen3-embedding:0.6b': EmbeddingModelSpec(
        provider='ollama',
        model='qwen3-embedding:0.6b',
        max_tokens=32000,
        dimension=1024,
        truncation_behavior='configurable',
        notes='Context controlled by OLLAMA_NUM_CTX. Truncation controlled by OLLAMA_TRUNCATE.',
    ),
    'qwen3-embedding:4b': EmbeddingModelSpec(
        provider='ollama',
        model='qwen3-embedding:4b',
        max_tokens=40000,
        dimension=2560,
        truncation_behavior='configurable',
    ),
    'qwen3-embedding:8b': EmbeddingModelSpec(
        provider='ollama',
        model='qwen3-embedding:8b',
        max_tokens=40000,
        dimension=4096,
        truncation_behavior='configurable',
    ),
    'nomic-embed-text': EmbeddingModelSpec(
        provider='ollama',
        model='nomic-embed-text',
        max_tokens=8192,
        dimension=768,
        truncation_behavior='configurable',
    ),
    'mxbai-embed-large': EmbeddingModelSpec(
        provider='ollama',
        model='mxbai-embed-large',
        max_tokens=512,
        dimension=1024,
        truncation_behavior='configurable',
    ),
    # OpenAI models
    'text-embedding-3-small': EmbeddingModelSpec(
        provider='openai',
        model='text-embedding-3-small',
        max_tokens=8191,
        dimension=1536,
        truncation_behavior='error',
        notes='Returns error when context exceeded. No truncation option.',
    ),
    'text-embedding-3-large': EmbeddingModelSpec(
        provider='openai',
        model='text-embedding-3-large',
        max_tokens=8191,
        dimension=3072,
        truncation_behavior='error',
    ),
    'text-embedding-ada-002': EmbeddingModelSpec(
        provider='openai',
        model='text-embedding-ada-002',
        max_tokens=8191,
        dimension=1536,
        truncation_behavior='error',
    ),
    # Voyage AI models
    'voyage-3': EmbeddingModelSpec(
        provider='voyage',
        model='voyage-3',
        max_tokens=32000,
        dimension=1024,
        truncation_behavior='configurable',
        notes='Truncation controlled by VOYAGE_TRUNCATION.',
    ),
    'voyage-3-large': EmbeddingModelSpec(
        provider='voyage',
        model='voyage-3-large',
        max_tokens=32000,
        dimension=1024,
        truncation_behavior='configurable',
    ),
    'voyage-3-lite': EmbeddingModelSpec(
        provider='voyage',
        model='voyage-3-lite',
        max_tokens=32000,
        dimension=512,
        truncation_behavior='configurable',
    ),
    # HuggingFace models
    'sentence-transformers/all-MiniLM-L6-v2': EmbeddingModelSpec(
        provider='huggingface',
        model='sentence-transformers/all-MiniLM-L6-v2',
        max_tokens=256,
        dimension=384,
        truncation_behavior='silent',
        notes='ALWAYS silently truncates. max_seq_length=256. Cannot be disabled.',
    ),
    'sentence-transformers/all-mpnet-base-v2': EmbeddingModelSpec(
        provider='huggingface',
        model='sentence-transformers/all-mpnet-base-v2',
        max_tokens=384,
        dimension=768,
        truncation_behavior='silent',
    ),
    'sentence-transformers/all-MiniLM-L12-v2': EmbeddingModelSpec(
        provider='huggingface',
        model='sentence-transformers/all-MiniLM-L12-v2',
        max_tokens=256,
        dimension=384,
        truncation_behavior='silent',
    ),
}


def get_model_spec(model: str) -> EmbeddingModelSpec | None:
    """Get specification for a model by name.

    Args:
        model: Model name (e.g., 'qwen3-embedding:0.6b', 'text-embedding-3-small')

    Returns:
        EmbeddingModelSpec if found, None otherwise
    """
    return EMBEDDING_MODEL_SPECS.get(model)


def get_provider_default_context(provider: str) -> int:
    """Get conservative default context limit for a provider.

    Used when specific model is unknown.

    Args:
        provider: Provider name ('ollama', 'openai', 'azure', 'huggingface', 'voyage')

    Returns:
        Conservative default context limit in tokens
    """
    defaults = {
        'ollama': 4096,      # Ollama server default (models may support more)
        'openai': 8191,      # OpenAI v3 models
        'azure': 8191,       # Same as OpenAI
        'huggingface': 256,  # Most sentence-transformers
        'voyage': 32000,     # Voyage v3 models
    }
    return defaults.get(provider, 2048)
