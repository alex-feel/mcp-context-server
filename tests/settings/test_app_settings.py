"""Tests for AppSettings, the settings model composed in app/settings/__init__.py.

Covers its cross-domain model validators (EMBEDDING_DIM against the pgvector
index cap, and the chunk size against the embedding provider context window)
and the search toggle, logging, search, embedding, and storage values that
AppSettings resolves from the environment.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.settings import AppSettings
from tests.helpers import env_var
from tests.helpers import env_vars


class TestPgvectorDimensionLimit:
    """EMBEDDING_DIM against the pgvector index cap on the fp32 PostgreSQL path.

    pgvector caps HNSW/IVFFlat index dimensionality at 2000 for the vector type.
    The fp32 PostgreSQL path always builds an HNSW index over vector(dim), so a
    dimension above 2000 passes pydantic's le=4096 field bound yet crashes the
    semantic-search migration at CREATE INDEX. Enabling compression (BYTEA
    payloads, no pgvector index) or using SQLite (sqlite-vec has no such cap)
    removes the constraint, so the guard is scoped to PostgreSQL + fp32 + embedding
    generation on and rejects the misconfiguration at the settings boundary.
    """

    def test_fp32_postgresql_dim_above_limit_rejected(self) -> None:
        """PostgreSQL + compression off + dim above 2000 is rejected before boot."""
        with (
            env_var('STORAGE_BACKEND', 'postgresql'),
            env_var('ENABLE_EMBEDDING_COMPRESSION', 'false'),
            env_var('ENABLE_EMBEDDING_GENERATION', 'true'),
            env_var('EMBEDDING_DIM', '2500'),
            pytest.raises(ValidationError, match='pgvector index limit'),
        ):
            AppSettings()

    def test_fp32_postgresql_dim_at_limit_accepted(self) -> None:
        """The exact 2000-dimension boundary is a valid fp32 PostgreSQL configuration."""
        with (
            env_var('STORAGE_BACKEND', 'postgresql'),
            env_var('ENABLE_EMBEDDING_COMPRESSION', 'false'),
            env_var('ENABLE_EMBEDDING_GENERATION', 'true'),
            env_var('EMBEDDING_DIM', '2000'),
        ):
            assert AppSettings().embedding.dim == 2000

    def test_compressed_postgresql_dim_above_limit_accepted(self) -> None:
        """With compression on, the vector is stored as BYTEA and the dim cap does not apply."""
        with (
            env_var('STORAGE_BACKEND', 'postgresql'),
            env_var('ENABLE_EMBEDDING_COMPRESSION', 'true'),
            env_var('ENABLE_EMBEDDING_GENERATION', 'true'),
            env_var('EMBEDDING_DIM', '2500'),
        ):
            assert AppSettings().embedding.dim == 2500

    def test_sqlite_dim_above_limit_accepted(self) -> None:
        """SQLite's sqlite-vec has no per-dimension index cap, so the guard does not fire."""
        with (
            env_var('STORAGE_BACKEND', 'sqlite'),
            env_var('ENABLE_EMBEDDING_COMPRESSION', 'false'),
            env_var('ENABLE_EMBEDDING_GENERATION', 'true'),
            env_var('EMBEDDING_DIM', '2500'),
        ):
            assert AppSettings().embedding.dim == 2500

    def test_generation_off_postgresql_dim_above_limit_accepted(self) -> None:
        """With generation off, no fresh fp32 vector table is provisioned, so the guard defers."""
        with (
            env_var('STORAGE_BACKEND', 'postgresql'),
            env_var('ENABLE_EMBEDDING_COMPRESSION', 'false'),
            env_var('ENABLE_EMBEDDING_GENERATION', 'false'),
            env_var('EMBEDDING_DIM', '2500'),
        ):
            assert AppSettings().embedding.dim == 2500


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
            EMBEDDING_OLLAMA_TRUNCATE='false',  # Truncation disabled
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


class TestServerToolRegistration:
    """Test dynamic tool registration based on configuration."""

    @pytest.mark.asyncio
    async def test_semantic_search_force_disabled(
        self,
        tmp_path: Path,
    ) -> None:
        """Test semantic search toggle resolves to force-disabled (mode='false')."""
        # Set environment to force semantic search off
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'ENABLE_SEMANTIC_SEARCH': 'false',
            'ENABLE_FTS': 'false',
            'ENABLE_HYBRID_SEARCH': 'false',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            # Force reimport to get fresh settings
            from app.settings import AppSettings

            settings = AppSettings()

            # mode='false' is the only state where .enabled is False
            assert settings.semantic_search.mode == 'false'
            assert settings.semantic_search.enabled is False

    @pytest.mark.asyncio
    async def test_search_toggles_default_to_auto(self, tmp_path: Path) -> None:
        """Test the three search toggles default to mode='auto' (enabled=True).

        With the tri-state migration, no ENABLE_* env var means mode='auto',
        which reports enabled=True so the runtime decides whether to expose the
        tool based on prerequisite availability.
        """
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
        }

        # Remove the three ENABLE_* toggles to observe the true defaults
        env_copy = os.environ.copy()
        for key in ('ENABLE_SEMANTIC_SEARCH', 'ENABLE_FTS', 'ENABLE_HYBRID_SEARCH'):
            env_copy.pop(key, None)

        with patch.dict(os.environ, {**env_copy, **env}, clear=True):
            from app.settings import AppSettings

            settings = AppSettings()

            assert settings.semantic_search.mode == 'auto'
            assert settings.semantic_search.enabled is True
            assert settings.fts.mode == 'auto'
            assert settings.fts.enabled is True
            assert settings.hybrid_search.mode == 'auto'
            assert settings.hybrid_search.enabled is True

    @pytest.mark.asyncio
    async def test_fts_tool_registration_condition(self, tmp_path: Path) -> None:
        """Test FTS toggle resolves to force-enabled when ENABLE_FTS=true."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'ENABLE_FTS': 'true',
            'ENABLE_SEMANTIC_SEARCH': 'false',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            from app.settings import AppSettings

            settings = AppSettings()

            assert settings.fts.mode == 'true'
            assert settings.fts.enabled is True

    @pytest.mark.asyncio
    async def test_hybrid_search_requires_at_least_one_mode(self, tmp_path: Path) -> None:
        """Test hybrid search registration requires FTS or semantic."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'ENABLE_HYBRID_SEARCH': 'true',
            'ENABLE_FTS': 'false',
            'ENABLE_SEMANTIC_SEARCH': 'false',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            from app.settings import AppSettings

            settings = AppSettings()

            # Hybrid is force-enabled in settings, but the tool won't register
            # because both FTS and semantic are force-disabled
            assert settings.hybrid_search.mode == 'true'
            assert settings.hybrid_search.enabled is True
            assert settings.fts.mode == 'false'
            assert settings.fts.enabled is False
            assert settings.semantic_search.mode == 'false'
            assert settings.semantic_search.enabled is False

    @pytest.mark.asyncio
    async def test_all_search_modes_enabled(self, tmp_path: Path) -> None:
        """Test when all search modes are force-enabled (mode='true')."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'ENABLE_FTS': 'true',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'ENABLE_HYBRID_SEARCH': 'true',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            from app.settings import AppSettings

            settings = AppSettings()

            assert settings.fts.mode == 'true'
            assert settings.fts.enabled is True
            assert settings.semantic_search.mode == 'true'
            assert settings.semantic_search.enabled is True
            assert settings.hybrid_search.mode == 'true'
            assert settings.hybrid_search.enabled is True


class TestServerConfigurationSettings:
    """Test server configuration settings parsing."""

    def test_log_level_default(self, tmp_path: Path) -> None:
        """Test default log level is ERROR."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
        }

        # Remove LOG_LEVEL if set
        env_copy = os.environ.copy()
        if 'LOG_LEVEL' in env_copy:
            del env_copy['LOG_LEVEL']

        with patch.dict(os.environ, {**env_copy, **env}, clear=True):
            from app.settings import AppSettings

            settings = AppSettings()
            assert settings.logging.level == 'ERROR'

    def test_log_level_override(self, tmp_path: Path) -> None:
        """Test log level can be overridden."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
            'LOG_LEVEL': 'DEBUG',
        }

        with patch.dict(os.environ, env, clear=False):
            from app.settings import AppSettings

            settings = AppSettings()
            assert settings.logging.level == 'DEBUG'

    def test_fts_language_default(self, tmp_path: Path) -> None:
        """Test default FTS language is english."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            from app.settings import AppSettings

            settings = AppSettings()
            assert settings.fts.language == 'english'

    def test_hybrid_rrf_k_default(self, tmp_path: Path) -> None:
        """Test default RRF k parameter is 60."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            from app.settings import AppSettings

            settings = AppSettings()
            assert settings.hybrid_search.rrf_k == 60

    def test_embedding_dim_default(self, tmp_path: Path) -> None:
        """Test default embedding dimension is 1024."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
        }

        # Remove EMBEDDING_DIM and EMBEDDING_MODEL if set (e.g., by CI)
        env_copy = os.environ.copy()
        if 'EMBEDDING_DIM' in env_copy:
            del env_copy['EMBEDDING_DIM']
        if 'EMBEDDING_MODEL' in env_copy:
            del env_copy['EMBEDDING_MODEL']

        with patch.dict(os.environ, {**env_copy, **env}, clear=True):
            from app.settings import AppSettings

            settings = AppSettings()
            assert settings.embedding.dim == 1024


class TestServerStorageSettings:
    """Test server storage configuration."""

    def test_storage_backend_default(self, tmp_path: Path) -> None:
        """Test default storage backend is sqlite."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
        }

        # Remove STORAGE_BACKEND to test default
        env_copy = os.environ.copy()
        if 'STORAGE_BACKEND' in env_copy:
            del env_copy['STORAGE_BACKEND']

        with patch.dict(os.environ, {**env_copy, **env}, clear=True):
            from app.settings import AppSettings

            settings = AppSettings()
            assert settings.storage.backend_type == 'sqlite'

    def test_max_image_size_default(self, tmp_path: Path) -> None:
        """Test default max image size is 10 MB."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            from app.settings import AppSettings

            settings = AppSettings()
            assert settings.storage.max_image_size_mb == 10

    def test_max_total_size_default(self, tmp_path: Path) -> None:
        """Test default max total size is 100 MB."""
        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
        }

        with patch.dict(os.environ, env, clear=False):
            from app.settings import AppSettings

            settings = AppSettings()
            assert settings.storage.max_total_size_mb == 100
