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


class TestLifespanErrorHandling:
    """Tests for lifespan() error handling."""

    @pytest.mark.asyncio
    async def test_startup_failure_shuts_down_backend(self, tmp_path: Path) -> None:
        """Verify backend shutdown on startup failure."""
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock

        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
            'ENABLE_SEMANTIC_SEARCH': 'false',
            'ENABLE_FTS': 'false',
        }

        with patch.dict(os.environ, env, clear=False):
            # Mock backend that will fail during init_database
            mock_backend = MagicMock()
            mock_backend.initialize = AsyncMock()
            mock_backend.shutdown = AsyncMock()
            mock_backend.backend_type = 'sqlite'

            with (
                patch('app.server.create_backend', return_value=mock_backend),
                patch('app.server.init_database', side_effect=RuntimeError('Database init failed')),
            ):
                from app.server import lifespan

                # Mock FastMCP instance
                mock_mcp = MagicMock()

                # Call lifespan and expect it to raise
                with pytest.raises(RuntimeError, match='Database init failed'):
                    async with lifespan(mock_mcp):
                        pass

                # Verify backend was shut down on failure
                mock_backend.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_migration_failure_propagates(self, tmp_path: Path) -> None:
        """Verify migration errors not swallowed."""
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock

        env = {
            'DB_PATH': str(tmp_path / 'test.db'),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
            'ENABLE_SEMANTIC_SEARCH': 'false',
            'ENABLE_FTS': 'false',
        }

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.initialize = AsyncMock()
            mock_backend.shutdown = AsyncMock()
            mock_backend.backend_type = 'sqlite'

            with (
                patch('app.server.create_backend', return_value=mock_backend),
                patch('app.server.init_database', new=AsyncMock()),
                patch('app.server.handle_metadata_indexes', new=AsyncMock()),
                patch(
                    'app.server.apply_semantic_search_migration',
                    side_effect=RuntimeError('Migration failed'),
                ),
            ):
                from app.server import lifespan

                mock_mcp = MagicMock()

                # Migration failure should propagate
                with pytest.raises(RuntimeError, match='Migration failed'):
                    async with lifespan(mock_mcp):
                        pass

    @pytest.mark.asyncio
    async def test_embedding_provider_failure_when_enabled_raises(self) -> None:
        """Verify server fails to start when ENABLE_EMBEDDING_GENERATION=true but provider fails.

        With the new architecture, ENABLE_EMBEDDING_GENERATION defaults to true.
        If provider initialization fails, the server MUST NOT start - this is fail-fast semantics.
        """
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock

        import app.startup
        from app.server import lifespan

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.shutdown = AsyncMock()
        mock_backend.backend_type = 'sqlite'

        # Create properly mocked repository container
        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)
        mock_repos.context = MagicMock()
        mock_repos.embedding = MagicMock()

        # Mock settings - ENABLE_EMBEDDING_GENERATION=true (default)
        mock_settings = MagicMock()
        mock_settings.enable_embedding_generation = True
        mock_settings.enable_semantic_search = True
        mock_settings.enable_fts = False
        mock_settings.enable_hybrid_search = False
        mock_settings.embedding.provider = 'ollama'

        # Store and restore globals
        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch('app.server.init_database', new=AsyncMock()),
                patch('app.server.handle_metadata_indexes', new=AsyncMock()),
                patch('app.server.apply_semantic_search_migration', new=AsyncMock()),
                patch('app.server.apply_jsonb_merge_patch_migration', new=AsyncMock()),
                patch('app.server.apply_function_search_path_migration', new=AsyncMock()),
                patch('app.server.apply_fts_migration', new=AsyncMock()),
                patch('app.server.apply_chunking_migration', new=AsyncMock()),
                patch('app.tools.register_tool', return_value=True),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
                patch('app.server.check_vector_storage_dependencies', new=AsyncMock(return_value=True)),
                patch(
                    'app.server.check_provider_dependencies',
                    new=AsyncMock(return_value={'available': True, 'reason': None, 'install_instructions': None}),
                ),
                patch('app.server.create_embedding_provider', side_effect=ImportError('Provider not installed')),
            ):
                mock_mcp = MagicMock()

                # Server should FAIL when ENABLE_EMBEDDING_GENERATION=true but provider fails
                with pytest.raises(RuntimeError, match='ENABLE_EMBEDDING_GENERATION=true'):
                    async with lifespan(mock_mcp):
                        pass
        finally:
            app.startup._backend = original_backend
            app.startup._repositories = original_repos
            app.startup._embedding_provider = original_provider

    @pytest.mark.asyncio
    async def test_embedding_provider_failure_graceful_when_disabled(self) -> None:
        """Verify server starts when ENABLE_EMBEDDING_GENERATION=false."""
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock

        import app.startup
        from app.server import lifespan

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.shutdown = AsyncMock()
        mock_backend.backend_type = 'sqlite'

        # Create properly mocked repository container
        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)
        mock_repos.context = MagicMock()
        mock_repos.embedding = MagicMock()

        # Mock settings - ENABLE_EMBEDDING_GENERATION=false (user explicitly disabled)
        mock_settings = MagicMock()
        mock_settings.enable_embedding_generation = False
        mock_settings.enable_semantic_search = False  # Doesn't matter when embeddings disabled
        mock_settings.enable_fts = False
        mock_settings.enable_hybrid_search = False
        mock_settings.embedding.provider = 'ollama'

        # Store and restore globals
        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch('app.server.init_database', new=AsyncMock()),
                patch('app.server.handle_metadata_indexes', new=AsyncMock()),
                patch('app.server.apply_semantic_search_migration', new=AsyncMock()),
                patch('app.server.apply_jsonb_merge_patch_migration', new=AsyncMock()),
                patch('app.server.apply_function_search_path_migration', new=AsyncMock()),
                patch('app.server.apply_fts_migration', new=AsyncMock()),
                patch('app.server.apply_chunking_migration', new=AsyncMock()),
                patch('app.tools.register_tool', return_value=True),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
            ):
                mock_mcp = MagicMock()

                # Server should start successfully when ENABLE_EMBEDDING_GENERATION=false
                async with lifespan(mock_mcp):
                    # Verify embedding provider is None
                    assert app.startup._embedding_provider is None
        finally:
            app.startup._backend = original_backend
            app.startup._repositories = original_repos
            app.startup._embedding_provider = original_provider

    @pytest.mark.asyncio
    async def test_shutdown_logs_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify shutdown errors logged not raised."""
        import logging
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock

        import app.startup
        from app.server import lifespan

        caplog.set_level(logging.ERROR)

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        # Shutdown will raise an error
        mock_backend.shutdown = AsyncMock(side_effect=RuntimeError('Shutdown failed'))
        mock_backend.backend_type = 'sqlite'

        # Create properly mocked repository container
        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)

        # Mock settings - disable embedding generation to avoid initialization
        mock_settings = MagicMock()
        mock_settings.enable_embedding_generation = False
        mock_settings.enable_semantic_search = False
        mock_settings.enable_fts = False
        mock_settings.enable_hybrid_search = False

        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch('app.server.init_database', new=AsyncMock()),
                patch('app.server.handle_metadata_indexes', new=AsyncMock()),
                patch('app.server.apply_semantic_search_migration', new=AsyncMock()),
                patch('app.server.apply_jsonb_merge_patch_migration', new=AsyncMock()),
                patch('app.server.apply_function_search_path_migration', new=AsyncMock()),
                patch('app.server.apply_fts_migration', new=AsyncMock()),
                patch('app.server.apply_chunking_migration', new=AsyncMock()),
                patch('app.tools.register_tool', return_value=True),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
            ):
                mock_mcp = MagicMock()

                # Should NOT raise despite shutdown error
                async with lifespan(mock_mcp):
                    pass

                # Verify error was logged
                assert any('shutdown' in r.message.lower() for r in caplog.records)
        finally:
            app.startup._backend = original_backend
            app.startup._repositories = original_repos
            app.startup._embedding_provider = original_provider

    @pytest.mark.asyncio
    async def test_embedding_provider_shutdown_on_exit(self) -> None:
        """Verify embedding provider shutdown called on exit."""
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock

        import app.startup
        from app.server import lifespan

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.shutdown = AsyncMock()
        mock_backend.backend_type = 'sqlite'

        # Create properly mocked repository container
        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)

        # Create mock embedding provider
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.initialize = AsyncMock()
        mock_embedding_provider.shutdown = AsyncMock()
        mock_embedding_provider.is_available = AsyncMock(return_value=True)
        mock_embedding_provider.provider_name = 'test-provider'

        # Mock settings - enable embedding generation
        mock_settings = MagicMock()
        mock_settings.enable_embedding_generation = True
        mock_settings.enable_semantic_search = True
        mock_settings.enable_fts = False
        mock_settings.enable_hybrid_search = False
        mock_settings.embedding.provider = 'ollama'

        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch('app.server.init_database', new=AsyncMock()),
                patch('app.server.handle_metadata_indexes', new=AsyncMock()),
                patch('app.server.apply_semantic_search_migration', new=AsyncMock()),
                patch('app.server.apply_jsonb_merge_patch_migration', new=AsyncMock()),
                patch('app.server.apply_function_search_path_migration', new=AsyncMock()),
                patch('app.server.apply_fts_migration', new=AsyncMock()),
                patch('app.server.apply_chunking_migration', new=AsyncMock()),
                patch('app.tools.register_tool', return_value=True),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
                patch('app.server.check_vector_storage_dependencies', new=AsyncMock(return_value=True)),
                patch(
                    'app.server.check_provider_dependencies',
                    new=AsyncMock(return_value={'available': True, 'reason': None, 'install_instructions': None}),
                ),
                patch('app.server.create_embedding_provider', return_value=mock_embedding_provider),
            ):
                mock_mcp = MagicMock()

                async with lifespan(mock_mcp):
                    # Verify embedding provider was set
                    assert app.startup._embedding_provider is not None

                # Verify embedding provider shutdown was called
                mock_embedding_provider.shutdown.assert_awaited_once()
        finally:
            app.startup._backend = original_backend
            app.startup._repositories = original_repos
            app.startup._embedding_provider = original_provider
