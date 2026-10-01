"""Tests for app/settings/search.py.

Covers FTS_LANGUAGE validation, RerankingSettings, RetrievalSettings, and the
search settings composed on AppSettings (hybrid fusion, reranking, chunking, and
the three search feature toggles).
"""

import pytest
from pydantic import ValidationError

from app.settings import AppSettings
from app.settings.search import RerankingSettings
from tests.helpers import env_var


class TestFtsLanguageValidation:
    """Test FTS_LANGUAGE setting validation."""

    def test_valid_languages_accepted(self) -> None:
        """Test that all valid PostgreSQL text search configurations are accepted."""
        valid_languages = [
            'simple',
            'arabic',
            'armenian',
            'basque',
            'catalan',
            'danish',
            'dutch',
            'english',
            'finnish',
            'french',
            'german',
            'greek',
            'hindi',
            'hungarian',
            'indonesian',
            'irish',
            'italian',
            'lithuanian',
            'nepali',
            'norwegian',
            'portuguese',
            'romanian',
            'russian',
            'serbian',
            'spanish',
            'swedish',
            'tamil',
            'turkish',
            'yiddish',
        ]

        for lang in valid_languages:
            with env_var('FTS_LANGUAGE', lang):
                settings = AppSettings()
                assert settings.fts.language == lang.lower(), f'Language {lang} should be accepted'

    def test_valid_languages_case_insensitive(self) -> None:
        """Test that language validation is case-insensitive."""
        case_variations = [
            ('english', 'english'),
            ('English', 'english'),
            ('ENGLISH', 'english'),
            ('EnGlIsH', 'english'),
            ('German', 'german'),
            ('FRENCH', 'french'),
            ('Russian', 'russian'),
        ]

        for input_lang, expected_output in case_variations:
            with env_var('FTS_LANGUAGE', input_lang):
                settings = AppSettings()
                assert settings.fts.language == expected_output, (
                    f'Language {input_lang} should be normalized to {expected_output}'
                )

    def test_invalid_language_raises_error(self) -> None:
        """Test that invalid languages raise ValueError with clear message."""
        invalid_languages = [
            'invalid',
            'nonsense',
            'foo',
            'bar',
            'unknown',
            'eng',
            'en',
            'de',
            'fr',
        ]

        for lang in invalid_languages:
            with env_var('FTS_LANGUAGE', lang):
                with pytest.raises(ValidationError) as exc_info:
                    AppSettings()

                # Check error message contains useful information
                error_str = str(exc_info.value)
                assert 'FTS_LANGUAGE' in error_str, f'Error should mention FTS_LANGUAGE for {lang}'
                assert 'valid options' in error_str.lower(), f'Error should mention valid options for {lang}'

    def test_invalid_language_error_shows_valid_options(self) -> None:
        """Test that the error message shows the list of valid options."""
        with env_var('FTS_LANGUAGE', 'invalid_language'):
            with pytest.raises(ValidationError) as exc_info:
                AppSettings()

            error_str = str(exc_info.value)
            # Check that at least some valid languages are mentioned in the error
            assert 'english' in error_str.lower(), 'Error should list english as a valid option'
            assert 'german' in error_str.lower(), 'Error should list german as a valid option'
            assert 'french' in error_str.lower(), 'Error should list french as a valid option'

    def test_default_language_is_english(self) -> None:
        """Test that the default FTS language is english."""
        # Ensure FTS_LANGUAGE is not set
        with env_var('FTS_LANGUAGE', None):
            settings = AppSettings()
            assert settings.fts.language == 'english'

    def test_fts_language_via_environment_variable(self) -> None:
        """Test that FTS_LANGUAGE can be set via environment variable."""
        # Test valid language via env var
        with env_var('FTS_LANGUAGE', 'german'):
            settings = AppSettings()
            assert settings.fts.language == 'german'

        # Test case normalization via env var
        with env_var('FTS_LANGUAGE', 'FRENCH'):
            settings = AppSettings()
            assert settings.fts.language == 'french'

    def test_invalid_language_via_environment_variable(self) -> None:
        """Test that invalid FTS_LANGUAGE via env var raises error."""
        with env_var('FTS_LANGUAGE', 'completely_invalid_language'):
            with pytest.raises(ValidationError) as exc_info:
                AppSettings()

            error_str = str(exc_info.value)
            assert 'FTS_LANGUAGE' in error_str

    def test_whitespace_language_raises_error(self) -> None:
        """Test that whitespace-only language raises error."""
        whitespace_variants = ['   ', '\t', '\n', ' \t\n ']

        for ws in whitespace_variants:
            with env_var('FTS_LANGUAGE', ws), pytest.raises(ValidationError):
                AppSettings()

    def test_all_29_valid_languages_count(self) -> None:
        """Test that exactly 29 valid languages are supported."""
        # This ensures we don't accidentally add or remove languages
        valid_languages = {
            'simple',
            'arabic',
            'armenian',
            'basque',
            'catalan',
            'danish',
            'dutch',
            'english',
            'finnish',
            'french',
            'german',
            'greek',
            'hindi',
            'hungarian',
            'indonesian',
            'irish',
            'italian',
            'lithuanian',
            'nepali',
            'norwegian',
            'portuguese',
            'romanian',
            'russian',
            'serbian',
            'spanish',
            'swedish',
            'tamil',
            'turkish',
            'yiddish',
        }
        assert len(valid_languages) == 29, 'Should have exactly 29 valid PostgreSQL text search configurations'

        # Verify all are accepted
        for lang in valid_languages:
            with env_var('FTS_LANGUAGE', lang):
                settings = AppSettings()
                assert settings.fts.language == lang


class TestRerankingSettings:
    """Tests for RerankingSettings validation."""

    def test_default_values(self) -> None:
        """Default values should be valid."""
        settings = RerankingSettings()
        assert settings.enabled is True
        assert settings.provider == 'flashrank'
        assert settings.model == 'ms-marco-MiniLM-L-12-v2'
        assert settings.max_length == 512
        assert settings.cache_dir is None
        assert settings.intra_op_threads == 0
        assert settings.batch_size == 32

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
        monkeypatch.setenv('RERANKING_CACHE_DIR', '/cache')
        monkeypatch.setenv('RERANKING_INTRA_OP_THREADS', '2')
        monkeypatch.setenv('RERANKING_BATCH_SIZE', '16')

        settings = RerankingSettings()
        assert settings.enabled is False
        assert settings.provider == 'custom'
        assert settings.model == 'custom-model'
        assert settings.max_length == 1024
        assert settings.cache_dir == '/cache'
        assert settings.intra_op_threads == 2
        assert settings.batch_size == 16

    def test_intra_op_threads_default_is_zero(self) -> None:
        """Default intra_op_threads should be 0 (auto-detect)."""
        settings = RerankingSettings()
        assert settings.intra_op_threads == 0

    def test_intra_op_threads_custom_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Custom intra_op_threads should be set via env var."""
        monkeypatch.setenv('RERANKING_INTRA_OP_THREADS', '4')
        settings = RerankingSettings()
        assert settings.intra_op_threads == 4

    def test_intra_op_threads_negative_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Negative intra_op_threads should fail validation."""
        monkeypatch.setenv('RERANKING_INTRA_OP_THREADS', '-1')
        with pytest.raises(ValidationError):
            RerankingSettings()

    def test_intra_op_threads_zero_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Zero (auto-detect) should be valid."""
        monkeypatch.setenv('RERANKING_INTRA_OP_THREADS', '0')
        settings = RerankingSettings()
        assert settings.intra_op_threads == 0

    def test_batch_size_default_is_32(self) -> None:
        """Default batch_size should be 32."""
        settings = RerankingSettings()
        assert settings.batch_size == 32

    def test_batch_size_custom_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Custom batch_size should be set via env var."""
        monkeypatch.setenv('RERANKING_BATCH_SIZE', '64')
        settings = RerankingSettings()
        assert settings.batch_size == 64

    def test_batch_size_zero_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """batch_size of zero should fail validation (gt=0)."""
        monkeypatch.setenv('RERANKING_BATCH_SIZE', '0')
        with pytest.raises(ValidationError):
            RerankingSettings()

    def test_batch_size_negative_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Negative batch_size should fail validation."""
        monkeypatch.setenv('RERANKING_BATCH_SIZE', '-1')
        with pytest.raises(ValidationError):
            RerankingSettings()


class TestAppSettingsIntegration:
    """Tests for AppSettings with nested chunking/reranking."""

    def test_nested_settings_accessible(self) -> None:
        """Nested settings should be accessible."""
        settings = AppSettings()
        assert settings.chunking.enabled is True
        assert settings.reranking.enabled is True
        assert settings.hybrid_search.rrf_overfetch == 2

    def test_hybrid_rrf_overfetch_minimum_valid(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Minimum valid hybrid_rrf_overfetch should pass."""
        monkeypatch.setenv('HYBRID_RRF_OVERFETCH', '1')
        settings = AppSettings()
        assert settings.hybrid_search.rrf_overfetch == 1

    def test_hybrid_rrf_overfetch_maximum_valid(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Maximum valid hybrid_rrf_overfetch should pass."""
        monkeypatch.setenv('HYBRID_RRF_OVERFETCH', '10')
        settings = AppSettings()
        assert settings.hybrid_search.rrf_overfetch == 10

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

    def test_truncation_length_default(self) -> None:
        """Default truncation_length should be 300."""
        settings = AppSettings()
        assert settings.search.truncation_length == 300

    def test_truncation_length_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Custom SEARCH_TRUNCATION_LENGTH should be accepted."""
        monkeypatch.setenv('SEARCH_TRUNCATION_LENGTH', '300')
        settings = AppSettings()
        assert settings.search.truncation_length == 300

    def test_truncation_length_minimum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SEARCH_TRUNCATION_LENGTH below 50 should fail."""
        monkeypatch.setenv('SEARCH_TRUNCATION_LENGTH', '49')
        with pytest.raises(ValidationError):
            AppSettings()

    def test_truncation_length_maximum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SEARCH_TRUNCATION_LENGTH above 1000 should fail."""
        monkeypatch.setenv('SEARCH_TRUNCATION_LENGTH', '1001')
        with pytest.raises(ValidationError):
            AppSettings()

    def test_truncation_length_at_boundaries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SEARCH_TRUNCATION_LENGTH at exact boundaries should be valid."""
        monkeypatch.setenv('SEARCH_TRUNCATION_LENGTH', '50')
        settings = AppSettings()
        assert settings.search.truncation_length == 50

        monkeypatch.setenv('SEARCH_TRUNCATION_LENGTH', '1000')
        settings = AppSettings()
        assert settings.search.truncation_length == 1000

    def test_chunking_settings_nested(self) -> None:
        """Chunking settings should work as nested config."""
        settings = AppSettings()
        assert settings.chunking.size == 1500
        assert settings.chunking.overlap == 150
        assert settings.chunking.aggregation == 'max'

    def test_reranking_settings_nested(self) -> None:
        """Reranking settings should work as nested config."""
        settings = AppSettings()
        assert settings.reranking.provider == 'flashrank'
        assert settings.reranking.model == 'ms-marco-MiniLM-L-12-v2'
        assert settings.reranking.max_length == 512


class TestRetrievalSettings:
    """Tests for RetrievalSettings.include_summary configuration."""

    def test_default_value_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default include_summary should be False (env var unset)."""
        monkeypatch.delenv('GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY', raising=False)
        from app.settings import get_settings
        from app.settings.search import RetrievalSettings
        get_settings.cache_clear()
        settings = RetrievalSettings()
        assert settings.include_summary is False

    def test_true_value_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY=true should yield True."""
        monkeypatch.setenv('GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY', 'true')
        from app.settings import get_settings
        from app.settings.search import RetrievalSettings
        get_settings.cache_clear()
        settings = RetrievalSettings()
        assert settings.include_summary is True

    def test_false_value_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY=false should yield False."""
        monkeypatch.setenv('GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY', 'false')
        from app.settings import get_settings
        from app.settings.search import RetrievalSettings
        get_settings.cache_clear()
        settings = RetrievalSettings()
        assert settings.include_summary is False

    def test_invalid_value_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY should fail validation."""
        monkeypatch.setenv('GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY', 'not-a-bool')
        from app.settings import get_settings
        from app.settings.search import RetrievalSettings
        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            RetrievalSettings()


class TestAppSettingsComposition:
    """Tests that the three toggles remain composed on AppSettings."""

    def test_defaults_all_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AppSettings defaults give all three toggles mode='auto', enabled True."""
        monkeypatch.delenv('ENABLE_SEMANTIC_SEARCH', raising=False)
        monkeypatch.delenv('ENABLE_FTS', raising=False)
        monkeypatch.delenv('ENABLE_HYBRID_SEARCH', raising=False)
        settings = AppSettings()
        assert settings.semantic_search.mode == 'auto'
        assert settings.fts.mode == 'auto'
        assert settings.hybrid_search.mode == 'auto'
        assert settings.semantic_search.enabled is True
        assert settings.fts.enabled is True
        assert settings.hybrid_search.enabled is True

    def test_semantic_search_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENABLE_SEMANTIC_SEARCH flows through to the nested toggle."""
        monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', 'false')
        settings = AppSettings()
        assert settings.semantic_search.mode == 'false'
        assert settings.semantic_search.enabled is False

    def test_fts_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENABLE_FTS flows through to the nested toggle."""
        monkeypatch.setenv('ENABLE_FTS', 'true')
        settings = AppSettings()
        assert settings.fts.mode == 'true'
        assert settings.fts.enabled is True

    def test_hybrid_search_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENABLE_HYBRID_SEARCH flows through to the nested toggle."""
        monkeypatch.setenv('ENABLE_HYBRID_SEARCH', 'false')
        settings = AppSettings()
        assert settings.hybrid_search.mode == 'false'
        assert settings.hybrid_search.enabled is False

    def test_invalid_nested_value_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An invalid nested toggle value fails AppSettings construction."""
        monkeypatch.setenv('ENABLE_FTS', 'bogus')
        with pytest.raises(ValidationError):
            AppSettings()
