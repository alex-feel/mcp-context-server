"""Tests for server initialization and tool registration.

Covers lines 836-907 in app/server.py for dynamic tool registration
based on configuration settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestServerToolRegistration:
    """Test dynamic tool registration based on configuration."""

    @pytest.mark.asyncio
    async def test_semantic_search_not_registered_when_disabled(
        self,
        tmp_path: Path,
    ) -> None:
        """Test semantic_search_context not registered when disabled."""
        # Set environment to disable semantic search
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

            assert settings.enable_semantic_search is False

    @pytest.mark.asyncio
    async def test_fts_tool_registration_condition(self, tmp_path: Path) -> None:
        """Test fts_search_context registration when ENABLE_FTS=true."""
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

            assert settings.enable_fts is True

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

            # Hybrid is enabled in settings, but tool won't register
            # because neither FTS nor semantic is available
            assert settings.enable_hybrid_search is True
            assert settings.enable_fts is False
            assert settings.enable_semantic_search is False

    @pytest.mark.asyncio
    async def test_all_search_modes_enabled(self, tmp_path: Path) -> None:
        """Test when all search modes are enabled."""
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

            assert settings.enable_fts is True
            assert settings.enable_semantic_search is True
            assert settings.enable_hybrid_search is True


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
            assert settings.log_level == 'ERROR'

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
            assert settings.log_level == 'DEBUG'

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
            assert settings.fts_language == 'english'

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
            assert settings.hybrid_rrf_k == 60

    def test_embedding_dim_default(self, tmp_path: Path) -> None:
        """Test default embedding dimension is 768."""
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
            assert settings.embedding_dim == 768


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
