"""Embedding compression abstractions and provider implementations.

This package provides optional storage compression for context embeddings
via Algorithm 2 of the TurboQuant paper (arXiv:2504.19874v1). When the
``ENABLE_EMBEDDING_COMPRESSION`` setting is enabled, fp32 embeddings are
replaced with bit-packed compressed representations supporting unbiased
inner-product estimation.

This module ships the NumPy implementation and provider abstraction
only; storage and search integration are wired separately.
"""

from app.compression.base import CompressionProvider
from app.compression.factory import create_compression_provider
from app.compression.factory import get_cached_compression_provider
from app.compression.factory import reset_cached_compression_provider
from app.compression.types import CompressionMetadata

__all__ = [
    'CompressionMetadata',
    'CompressionProvider',
    'create_compression_provider',
    'get_cached_compression_provider',
    'reset_cached_compression_provider',
]
