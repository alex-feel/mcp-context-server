"""
Tests for embedding truncation settings validation.

Tests verify:
1. OLLAMA_TRUNCATE default is false
2. OLLAMA_NUM_CTX default is 4096
3. VOYAGE_TRUNCATION default is false
4. Settings validation warnings for chunk size vs context
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from app.settings import AppSettings


@contextmanager
def env_vars(**kwargs: str | None) -> Generator[None, None, None]:
    """Context manager for temporarily setting multiple environment variables."""
    originals = {key: os.environ.get(key) for key in kwargs}
    try:
        for key, value in kwargs.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        yield
    finally:
        for key, original in originals.items():
            if original is not None:
                os.environ[key] = original
            elif key in os.environ:
                del os.environ[key]


class TestOllamaTruncateSettings:
    """Test OLLAMA_TRUNCATE and OLLAMA_NUM_CTX settings."""

    def test_ollama_truncate_default_is_false(self) -> None:
        """Verify OLLAMA_TRUNCATE defaults to false (prevent silent truncation)."""
        with env_vars(OLLAMA_TRUNCATE=None, OLLAMA_NUM_CTX=None):
            settings = AppSettings()
            assert settings.embedding.ollama_truncate is False

    def test_ollama_truncate_can_be_set_true(self) -> None:
        """Verify OLLAMA_TRUNCATE can be explicitly set to true."""
        with env_vars(OLLAMA_TRUNCATE='true'):
            settings = AppSettings()
            assert settings.embedding.ollama_truncate is True

    def test_ollama_truncate_can_be_set_false(self) -> None:
        """Verify OLLAMA_TRUNCATE can be explicitly set to false."""
        with env_vars(OLLAMA_TRUNCATE='false'):
            settings = AppSettings()
            assert settings.embedding.ollama_truncate is False

    def test_ollama_num_ctx_default_is_4096(self) -> None:
        """Verify OLLAMA_NUM_CTX defaults to 4096."""
        with env_vars(OLLAMA_NUM_CTX=None):
            settings = AppSettings()
            assert settings.embedding.ollama_num_ctx == 4096

    def test_ollama_num_ctx_can_be_customized(self) -> None:
        """Verify OLLAMA_NUM_CTX can be set to custom value."""
        with env_vars(OLLAMA_NUM_CTX='8192'):
            settings = AppSettings()
            assert settings.embedding.ollama_num_ctx == 8192

    def test_ollama_num_ctx_minimum_validation(self) -> None:
        """Verify OLLAMA_NUM_CTX validates minimum value (512)."""
        with env_vars(OLLAMA_NUM_CTX='100'), pytest.raises(ValidationError):
            AppSettings()

    def test_ollama_num_ctx_maximum_validation(self) -> None:
        """Verify OLLAMA_NUM_CTX validates maximum value (131072)."""
        with env_vars(OLLAMA_NUM_CTX='200000'), pytest.raises(ValidationError):
            AppSettings()


class TestVoyageTruncationSettings:
    """Test VOYAGE_TRUNCATION settings."""

    def test_voyage_truncation_default_is_false(self) -> None:
        """Verify VOYAGE_TRUNCATION defaults to false (prevent silent truncation)."""
        with env_vars(VOYAGE_TRUNCATION=None):
            settings = AppSettings()
            assert settings.embedding.voyage_truncation is False

    def test_voyage_truncation_can_be_set_true(self) -> None:
        """Verify VOYAGE_TRUNCATION can be explicitly set to true."""
        with env_vars(VOYAGE_TRUNCATION='true'):
            settings = AppSettings()
            assert settings.embedding.voyage_truncation is True

    def test_voyage_truncation_can_be_set_false(self) -> None:
        """Verify VOYAGE_TRUNCATION can be explicitly set to false."""
        with env_vars(VOYAGE_TRUNCATION='false'):
            settings = AppSettings()
            assert settings.embedding.voyage_truncation is False


class TestChunkSizeVsContextValidation:
    """Test validation warnings for chunk size vs context length (universal validator)."""

    def test_warning_when_chunk_size_exceeds_model_context(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify warning when CHUNK_SIZE exceeds model's context limit."""
        import logging

        caplog.set_level(logging.WARNING)

        # Use HuggingFace all-MiniLM-L6-v2 which has 256 tokens context limit.
        # CHUNK_SIZE=1000 chars / 3 = ~333 tokens, exceeds 256.
        # This triggers the universal validator warning.
        with env_vars(
            EMBEDDING_PROVIDER='huggingface',
            EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2',
            HUGGINGFACEHUB_API_TOKEN='test-token',
            ENABLE_CHUNKING='true',
            CHUNK_SIZE='1000',  # ~333 tokens, exceeds model's 256 limit
        ):
            AppSettings()
            assert 'CHUNK_SIZE' in caplog.text
            assert 'exceeds' in caplog.text

    def test_warning_when_chunking_disabled_with_configurable_truncation(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify warning when chunking is disabled with configurable truncation provider."""
        import logging

        caplog.set_level(logging.WARNING)

        with env_vars(
            EMBEDDING_PROVIDER='ollama',
            EMBEDDING_MODEL='qwen3-embedding:0.6b',
            ENABLE_CHUNKING='false',
            OLLAMA_TRUNCATE='false',  # Truncation disabled
        ):
            AppSettings()
            assert 'ENABLE_CHUNKING=false' in caplog.text
            assert 'truncation disabled' in caplog.text

    def test_warning_when_chunking_disabled_with_silent_truncation(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify warning when chunking disabled with model that always truncates."""
        import logging

        caplog.set_level(logging.WARNING)

        with env_vars(
            EMBEDDING_PROVIDER='huggingface',
            EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2',  # Silent truncation
            ENABLE_CHUNKING='false',
            HUGGINGFACEHUB_API_TOKEN='test-token',
        ):
            AppSettings()
            assert 'ENABLE_CHUNKING=false' in caplog.text
            assert 'silently truncates' in caplog.text

    def test_no_warning_when_chunk_size_within_context(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify no warning when CHUNK_SIZE is within model's context bounds."""
        import logging

        caplog.set_level(logging.WARNING)

        with env_vars(
            EMBEDDING_PROVIDER='ollama',
            EMBEDDING_MODEL='qwen3-embedding:0.6b',  # 32000 tokens max
            ENABLE_CHUNKING='true',
            CHUNK_SIZE='1000',  # ~333 tokens estimate, well within 32000
        ):
            AppSettings()
            # Should not warn about chunk size exceeding context
            assert 'exceeds' not in caplog.text.lower() or 'CHUNK_SIZE' not in caplog.text

    def test_warning_for_unknown_model_uses_provider_default(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify warning when model is not in context_limits.py."""
        import logging

        caplog.set_level(logging.WARNING)

        with env_vars(
            EMBEDDING_PROVIDER='ollama',
            EMBEDDING_MODEL='unknown-model-xyz',  # Not in context_limits.py
            ENABLE_CHUNKING='true',
            CHUNK_SIZE='1000',
        ):
            AppSettings()
            assert 'not found in context_limits.py' in caplog.text
            assert 'provider default' in caplog.text

    def test_validates_against_known_model_spec(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify validation uses model spec from context_limits.py."""
        import logging

        caplog.set_level(logging.WARNING)

        # HuggingFace all-MiniLM-L6-v2 has 256 tokens context limit.
        # CHUNK_SIZE=1000 chars / 3 = ~333 tokens, exceeds 256.
        with env_vars(
            EMBEDDING_PROVIDER='huggingface',
            EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2',
            HUGGINGFACEHUB_API_TOKEN='test-token',
            ENABLE_CHUNKING='true',
            CHUNK_SIZE='1000',  # ~333 tokens, exceeds model's 256 limit
        ):
            AppSettings()
            assert 'CHUNK_SIZE' in caplog.text
            assert '256' in caplog.text  # Model's actual limit from context_limits.py
