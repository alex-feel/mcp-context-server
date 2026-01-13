"""Tests for ChunkingSettings and RerankingSettings configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.settings import AppSettings
from app.settings import ChunkingSettings
from app.settings import RerankingSettings


class TestChunkingSettings:
    """Tests for ChunkingSettings validation."""

    def test_default_values(self) -> None:
        """Default values should be valid."""
        settings = ChunkingSettings()
        assert settings.enabled is True
        assert settings.size == 1000
        assert settings.overlap == 100
        assert settings.aggregation == 'max'
        assert settings.dedup_overfetch == 5

    def test_overlap_must_be_less_than_size(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Overlap must be strictly less than chunk size."""
        monkeypatch.setenv('CHUNK_SIZE', '100')
        monkeypatch.setenv('CHUNK_OVERLAP', '100')
        with pytest.raises(ValidationError) as exc_info:
            ChunkingSettings()
        assert 'CHUNK_OVERLAP' in str(exc_info.value)
        assert 'must be less than' in str(exc_info.value)

    def test_overlap_greater_than_size_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Overlap greater than size should fail."""
        monkeypatch.setenv('CHUNK_SIZE', '100')
        monkeypatch.setenv('CHUNK_OVERLAP', '150')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_valid_overlap_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid overlap should pass validation."""
        monkeypatch.setenv('CHUNK_SIZE', '500')
        monkeypatch.setenv('CHUNK_OVERLAP', '100')
        settings = ChunkingSettings()
        assert settings.size == 500
        assert settings.overlap == 100

    def test_size_minimum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Minimum valid chunk size should pass."""
        monkeypatch.setenv('CHUNK_SIZE', '100')
        monkeypatch.setenv('CHUNK_OVERLAP', '50')
        settings = ChunkingSettings()
        assert settings.size == 100

    def test_size_maximum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Maximum valid chunk size should pass."""
        monkeypatch.setenv('CHUNK_SIZE', '10000')
        monkeypatch.setenv('CHUNK_OVERLAP', '100')
        settings = ChunkingSettings()
        assert settings.size == 10000

    def test_size_below_minimum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Chunk size below minimum should fail."""
        monkeypatch.setenv('CHUNK_SIZE', '99')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_size_above_maximum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Chunk size above maximum should fail."""
        monkeypatch.setenv('CHUNK_SIZE', '10001')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_overlap_minimum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Minimum valid overlap (0) should pass."""
        monkeypatch.setenv('CHUNK_OVERLAP', '0')
        settings = ChunkingSettings()
        assert settings.overlap == 0

    def test_overlap_maximum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Maximum valid overlap should pass."""
        monkeypatch.setenv('CHUNK_SIZE', '1000')
        monkeypatch.setenv('CHUNK_OVERLAP', '500')
        settings = ChunkingSettings()
        assert settings.overlap == 500

    def test_overlap_negative_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Negative overlap should fail."""
        monkeypatch.setenv('CHUNK_OVERLAP', '-1')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_overlap_above_maximum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Overlap above maximum should fail."""
        monkeypatch.setenv('CHUNK_OVERLAP', '501')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_aggregation_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Aggregation 'max' should be valid."""
        monkeypatch.setenv('CHUNK_AGGREGATION', 'max')
        settings = ChunkingSettings()
        assert settings.aggregation == 'max'

    def test_aggregation_invalid_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid aggregation (including 'avg' and 'sum') should fail."""
        monkeypatch.setenv('CHUNK_AGGREGATION', 'invalid')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_dedup_overfetch_minimum_valid(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Minimum valid dedup_overfetch should pass."""
        monkeypatch.setenv('CHUNK_DEDUP_OVERFETCH', '1')
        settings = ChunkingSettings()
        assert settings.dedup_overfetch == 1

    def test_dedup_overfetch_maximum_valid(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Maximum valid dedup_overfetch should pass."""
        monkeypatch.setenv('CHUNK_DEDUP_OVERFETCH', '20')
        settings = ChunkingSettings()
        assert settings.dedup_overfetch == 20

    def test_dedup_overfetch_below_minimum_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """dedup_overfetch below minimum should fail."""
        monkeypatch.setenv('CHUNK_DEDUP_OVERFETCH', '0')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_dedup_overfetch_above_maximum_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """dedup_overfetch above maximum should fail."""
        monkeypatch.setenv('CHUNK_DEDUP_OVERFETCH', '21')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_environment_variable_aliases(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Settings should read from environment variables."""
        monkeypatch.setenv('ENABLE_CHUNKING', 'false')
        monkeypatch.setenv('CHUNK_SIZE', '2000')
        monkeypatch.setenv('CHUNK_OVERLAP', '200')
        monkeypatch.setenv('CHUNK_AGGREGATION', 'max')
        monkeypatch.setenv('CHUNK_DEDUP_OVERFETCH', '10')

        settings = ChunkingSettings()
        assert settings.enabled is False
        assert settings.size == 2000
        assert settings.overlap == 200
        assert settings.aggregation == 'max'
        assert settings.dedup_overfetch == 10


class TestRerankingSettings:
    """Tests for RerankingSettings validation."""

    def test_default_values(self) -> None:
        """Default values should be valid."""
        settings = RerankingSettings()
        assert settings.enabled is True
        assert settings.provider == 'flashrank'
        assert settings.model == 'ms-marco-MiniLM-L-12-v2'
        assert settings.max_length == 512
        assert settings.overfetch == 4
        assert settings.cache_dir is None

    def test_max_length_minimum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Minimum valid max_length should pass."""
        monkeypatch.setenv('RERANKING_MAX_LENGTH', '128')
        settings = RerankingSettings()
        assert settings.max_length == 128

    def test_max_length_maximum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Maximum valid max_length should pass."""
        monkeypatch.setenv('RERANKING_MAX_LENGTH', '2048')
        settings = RerankingSettings()
        assert settings.max_length == 2048

    def test_max_length_below_minimum_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """max_length below minimum should fail."""
        monkeypatch.setenv('RERANKING_MAX_LENGTH', '127')
        with pytest.raises(ValidationError):
            RerankingSettings()

    def test_max_length_above_maximum_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """max_length above maximum should fail."""
        monkeypatch.setenv('RERANKING_MAX_LENGTH', '2049')
        with pytest.raises(ValidationError):
            RerankingSettings()

    def test_overfetch_minimum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Minimum valid overfetch should pass."""
        monkeypatch.setenv('RERANKING_OVERFETCH', '1')
        settings = RerankingSettings()
        assert settings.overfetch == 1

    def test_overfetch_maximum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Maximum valid overfetch should pass."""
        monkeypatch.setenv('RERANKING_OVERFETCH', '20')
        settings = RerankingSettings()
        assert settings.overfetch == 20

    def test_overfetch_below_minimum_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """overfetch below minimum should fail."""
        monkeypatch.setenv('RERANKING_OVERFETCH', '0')
        with pytest.raises(ValidationError):
            RerankingSettings()

    def test_overfetch_above_maximum_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """overfetch above maximum should fail."""
        monkeypatch.setenv('RERANKING_OVERFETCH', '21')
        with pytest.raises(ValidationError):
            RerankingSettings()

    def test_cache_dir_custom_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Custom cache_dir should be set."""
        monkeypatch.setenv('RERANKING_CACHE_DIR', '/custom/path')
        settings = RerankingSettings()
        assert settings.cache_dir == '/custom/path'

    def test_cache_dir_default_is_none(self) -> None:
        """Default cache_dir should be None."""
        settings = RerankingSettings()
        assert settings.cache_dir is None

    def test_environment_variable_aliases(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Settings should read from environment variables."""
        monkeypatch.setenv('ENABLE_RERANKING', 'false')
        monkeypatch.setenv('RERANKING_PROVIDER', 'custom')
        monkeypatch.setenv('RERANKING_MODEL', 'custom-model')
        monkeypatch.setenv('RERANKING_MAX_LENGTH', '1024')
        monkeypatch.setenv('RERANKING_OVERFETCH', '8')
        monkeypatch.setenv('RERANKING_CACHE_DIR', '/cache')

        settings = RerankingSettings()
        assert settings.enabled is False
        assert settings.provider == 'custom'
        assert settings.model == 'custom-model'
        assert settings.max_length == 1024
        assert settings.overfetch == 8
        assert settings.cache_dir == '/cache'


class TestAppSettingsIntegration:
    """Tests for AppSettings with nested chunking/reranking."""

    def test_nested_settings_accessible(self) -> None:
        """Nested settings should be accessible."""
        settings = AppSettings()
        assert settings.chunking.enabled is True
        assert settings.reranking.enabled is True
        assert settings.hybrid_rrf_overfetch == 2
        assert settings.search_default_sort_by == 'relevance'

    def test_hybrid_rrf_overfetch_minimum_valid(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Minimum valid hybrid_rrf_overfetch should pass."""
        monkeypatch.setenv('HYBRID_RRF_OVERFETCH', '1')
        settings = AppSettings()
        assert settings.hybrid_rrf_overfetch == 1

    def test_hybrid_rrf_overfetch_maximum_valid(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Maximum valid hybrid_rrf_overfetch should pass."""
        monkeypatch.setenv('HYBRID_RRF_OVERFETCH', '10')
        settings = AppSettings()
        assert settings.hybrid_rrf_overfetch == 10

    def test_hybrid_rrf_overfetch_below_minimum_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """hybrid_rrf_overfetch below minimum should fail."""
        monkeypatch.setenv('HYBRID_RRF_OVERFETCH', '0')
        with pytest.raises(ValidationError):
            AppSettings()

    def test_hybrid_rrf_overfetch_above_maximum_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """hybrid_rrf_overfetch above maximum should fail."""
        monkeypatch.setenv('HYBRID_RRF_OVERFETCH', '11')
        with pytest.raises(ValidationError):
            AppSettings()

    def test_sort_by_relevance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """search_default_sort_by 'relevance' should be valid."""
        monkeypatch.setenv('SEARCH_DEFAULT_SORT_BY', 'relevance')
        settings = AppSettings()
        assert settings.search_default_sort_by == 'relevance'

    def test_sort_by_invalid_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid search_default_sort_by should fail."""
        monkeypatch.setenv('SEARCH_DEFAULT_SORT_BY', 'invalid')
        with pytest.raises(ValidationError):
            AppSettings()

    def test_chunking_settings_nested(self) -> None:
        """Chunking settings should work as nested config."""
        settings = AppSettings()
        assert settings.chunking.size == 1000
        assert settings.chunking.overlap == 100
        assert settings.chunking.aggregation == 'max'

    def test_reranking_settings_nested(self) -> None:
        """Reranking settings should work as nested config."""
        settings = AppSettings()
        assert settings.reranking.provider == 'flashrank'
        assert settings.reranking.model == 'ms-marco-MiniLM-L-12-v2'
        assert settings.reranking.max_length == 512
