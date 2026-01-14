"""Tests for embedding model context limits reference data."""

from __future__ import annotations

from app.embeddings.context_limits import EMBEDDING_MODEL_SPECS
from app.embeddings.context_limits import get_model_spec
from app.embeddings.context_limits import get_provider_default_context


class TestEmbeddingModelSpecs:
    """Tests for EMBEDDING_MODEL_SPECS dictionary."""

    def test_qwen3_embedding_spec_exists(self) -> None:
        """Verify qwen3-embedding:0.6b spec is defined."""
        spec = EMBEDDING_MODEL_SPECS.get('qwen3-embedding:0.6b')
        assert spec is not None
        assert spec.provider == 'ollama'
        assert spec.max_tokens == 32000
        assert spec.dimension == 1024
        assert spec.truncation_behavior == 'configurable'

    def test_qwen3_4b_spec_exists(self) -> None:
        """Verify qwen3-embedding:4b spec is defined."""
        spec = EMBEDDING_MODEL_SPECS.get('qwen3-embedding:4b')
        assert spec is not None
        assert spec.provider == 'ollama'
        assert spec.max_tokens == 40000
        assert spec.dimension == 2560
        assert spec.truncation_behavior == 'configurable'

    def test_qwen3_8b_spec_exists(self) -> None:
        """Verify qwen3-embedding:8b spec is defined."""
        spec = EMBEDDING_MODEL_SPECS.get('qwen3-embedding:8b')
        assert spec is not None
        assert spec.provider == 'ollama'
        assert spec.max_tokens == 40000
        assert spec.dimension == 4096

    def test_nomic_embed_text_spec_exists(self) -> None:
        """Verify nomic-embed-text spec is defined."""
        spec = EMBEDDING_MODEL_SPECS.get('nomic-embed-text')
        assert spec is not None
        assert spec.provider == 'ollama'
        assert spec.max_tokens == 8192
        assert spec.dimension == 768

    def test_openai_models_return_error_on_exceed(self) -> None:
        """Verify OpenAI models have error truncation behavior."""
        for model in ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']:
            spec = EMBEDDING_MODEL_SPECS.get(model)
            assert spec is not None, f'Model {model} not found'
            assert spec.truncation_behavior == 'error', f'Model {model} should have error behavior'
            assert spec.provider == 'openai'

    def test_huggingface_models_silently_truncate(self) -> None:
        """Verify HuggingFace models have silent truncation behavior."""
        spec = EMBEDDING_MODEL_SPECS.get('sentence-transformers/all-MiniLM-L6-v2')
        assert spec is not None
        assert spec.truncation_behavior == 'silent'
        assert spec.provider == 'huggingface'
        assert spec.max_tokens == 256

    def test_huggingface_mpnet_spec_exists(self) -> None:
        """Verify sentence-transformers/all-mpnet-base-v2 spec is defined."""
        spec = EMBEDDING_MODEL_SPECS.get('sentence-transformers/all-mpnet-base-v2')
        assert spec is not None
        assert spec.truncation_behavior == 'silent'
        assert spec.dimension == 768
        assert spec.max_tokens == 384

    def test_voyage_models_are_configurable(self) -> None:
        """Verify Voyage models have configurable truncation behavior."""
        for model in ['voyage-3', 'voyage-3-large', 'voyage-3-lite']:
            spec = EMBEDDING_MODEL_SPECS.get(model)
            assert spec is not None, f'Model {model} not found'
            assert spec.truncation_behavior == 'configurable', f'Model {model} should be configurable'
            assert spec.provider == 'voyage'

    def test_voyage_3_spec_details(self) -> None:
        """Verify voyage-3 specific details."""
        spec = EMBEDDING_MODEL_SPECS.get('voyage-3')
        assert spec is not None
        assert spec.max_tokens == 32000
        assert spec.dimension == 1024
        assert 'VOYAGE_TRUNCATION' in spec.notes

    def test_voyage_3_lite_has_smaller_dimension(self) -> None:
        """Verify voyage-3-lite has 512 dimensions."""
        spec = EMBEDDING_MODEL_SPECS.get('voyage-3-lite')
        assert spec is not None
        assert spec.dimension == 512

    def test_mxbai_embed_large_spec_exists(self) -> None:
        """Verify mxbai-embed-large spec is defined."""
        spec = EMBEDDING_MODEL_SPECS.get('mxbai-embed-large')
        assert spec is not None
        assert spec.provider == 'ollama'
        assert spec.max_tokens == 512
        assert spec.dimension == 1024


class TestGetModelSpec:
    """Tests for get_model_spec function."""

    def test_returns_spec_for_known_model(self) -> None:
        """Verify returns spec for known model."""
        spec = get_model_spec('qwen3-embedding:0.6b')
        assert spec is not None
        assert spec.model == 'qwen3-embedding:0.6b'

    def test_returns_none_for_unknown_model(self) -> None:
        """Verify returns None for unknown model."""
        spec = get_model_spec('unknown-model')
        assert spec is None

    def test_returns_none_for_empty_string(self) -> None:
        """Verify returns None for empty string."""
        spec = get_model_spec('')
        assert spec is None

    def test_returns_correct_spec_for_openai_model(self) -> None:
        """Verify returns correct spec for OpenAI model."""
        spec = get_model_spec('text-embedding-3-small')
        assert spec is not None
        assert spec.provider == 'openai'
        assert spec.max_tokens == 8191

    def test_returns_correct_spec_for_huggingface_model(self) -> None:
        """Verify returns correct spec for HuggingFace model."""
        spec = get_model_spec('sentence-transformers/all-MiniLM-L6-v2')
        assert spec is not None
        assert spec.provider == 'huggingface'
        assert spec.truncation_behavior == 'silent'


class TestGetProviderDefaultContext:
    """Tests for get_provider_default_context function."""

    def test_ollama_default_context(self) -> None:
        """Verify Ollama default context is 4096."""
        assert get_provider_default_context('ollama') == 4096

    def test_openai_default_context(self) -> None:
        """Verify OpenAI default context is 8191."""
        assert get_provider_default_context('openai') == 8191

    def test_azure_default_context(self) -> None:
        """Verify Azure default context is 8191 (same as OpenAI)."""
        assert get_provider_default_context('azure') == 8191

    def test_voyage_default_context(self) -> None:
        """Verify Voyage default context is 32000."""
        assert get_provider_default_context('voyage') == 32000

    def test_huggingface_default_context(self) -> None:
        """Verify HuggingFace default context is 256."""
        assert get_provider_default_context('huggingface') == 256

    def test_unknown_provider_default(self) -> None:
        """Verify unknown provider returns conservative default of 2048."""
        assert get_provider_default_context('unknown') == 2048

    def test_empty_provider_default(self) -> None:
        """Verify empty provider returns conservative default of 2048."""
        assert get_provider_default_context('') == 2048
