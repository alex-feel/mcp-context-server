"""Tests for server initialization and tool registration.

Covers lines 836-907 in app/server.py for dynamic tool registration
based on configuration settings.
"""

import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest


def _patch_server_migrations() -> AbstractContextManager[Any]:
    """One context manager that neutralizes ALL lifespan migration/init steps.

    Consolidates the per-step ``patch('app.server.<step>', new=AsyncMock())`` block
    into a single ``patch.multiple`` so the enclosing ``with (...)`` statement stays
    well under CPython's static nested-block limit (each parenthesized context
    manager is a nested block; adding the new ``apply_version_migration`` step tipped
    the deepest block over the limit). Reproduces the EXACT set every full-migration
    lifespan test mocked, plus ``apply_version_migration`` -- the optimistic-concurrency
    ``version`` column migration wired into the lifespan, which otherwise runs against
    the MagicMock backend and raises ``object MagicMock can't be used in 'await'``.

    Returns:
        The ``patch.multiple`` context manager neutralizing every lifespan step.
    """
    return patch.multiple(
        'app.server',
        init_database=AsyncMock(),
        handle_metadata_indexes=AsyncMock(),
        guard_compression_disable_over_populated=AsyncMock(),
        apply_semantic_search_migration=AsyncMock(),
        apply_jsonb_merge_patch_migration=AsyncMock(),
        apply_function_search_path_migration=AsyncMock(),
        apply_fts_migration=AsyncMock(),
        apply_chunking_migration=AsyncMock(),
        apply_index_tree_migration=AsyncMock(),
        apply_summary_migration=AsyncMock(),
        apply_content_hash_migration=AsyncMock(),
        apply_version_migration=AsyncMock(),
    )


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
                patch('app.server.guard_compression_disable_over_populated', new=AsyncMock()),
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
        from app.errors import ConfigurationError
        from app.server import lifespan

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.shutdown = AsyncMock()
        mock_backend.backend_type = 'sqlite'
        # The compression validator probes provenance even when disabled; an
        # awaitable execute_read returning None models a never-compressed DB.
        mock_backend.execute_read = AsyncMock(return_value=None)

        # Create properly mocked repository container
        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)
        mock_repos.context = MagicMock()
        mock_repos.embedding = MagicMock()

        # Mock settings - ENABLE_EMBEDDING_GENERATION=true (default)
        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = True
        mock_settings.reranking.enabled = False
        mock_settings.chunking.enabled = False
        mock_settings.summary.generation_enabled = False
        # Step 22 registration reads semantic_search.mode (tri-state string);
        # 'true' forces the tool on (warns when no provider).
        mock_settings.semantic_search.mode = 'true'
        mock_settings.semantic_search.enabled = True
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        mock_settings.embedding.provider = 'ollama'
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the inline INFO log reports
        # disabled without a read.
        mock_settings.compression.enabled = False

        # Store and restore globals
        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                _patch_server_migrations(),
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
                # ConfigurationError is raised for import failures (exit code 78)
                with pytest.raises(ConfigurationError, match='ENABLE_EMBEDDING_GENERATION=true'):
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
        # The compression validator probes provenance even when disabled; an
        # awaitable execute_read returning None models a never-compressed DB.
        mock_backend.execute_read = AsyncMock(return_value=None)

        # Create properly mocked repository container
        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)
        mock_repos.context = MagicMock()
        mock_repos.embedding = MagicMock()

        # Mock settings - ENABLE_EMBEDDING_GENERATION=false (user explicitly disabled)
        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = False
        mock_settings.reranking.enabled = False
        mock_settings.chunking.enabled = False
        mock_settings.summary.generation_enabled = False
        # Step 22 reads semantic_search.mode; 'false' forces the tool off,
        # matching the intent that no provider is available here.
        mock_settings.semantic_search.mode = 'false'
        mock_settings.semantic_search.enabled = False
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        mock_settings.embedding.provider = 'ollama'
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the inline INFO log reports
        # disabled without a read.
        mock_settings.compression.enabled = False

        # Store and restore globals
        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                _patch_server_migrations(),
                patch('app.tools.register_tool', return_value=True),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
            ):
                mock_mcp = MagicMock()
                mock_mcp.list_tools = AsyncMock(return_value=[])

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
        # The compression validator probes provenance even when disabled; an
        # awaitable execute_read returning None models a never-compressed DB.
        mock_backend.execute_read = AsyncMock(return_value=None)

        # Create properly mocked repository container
        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=False)

        # Mock settings - disable embedding generation to avoid initialization
        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = False
        mock_settings.reranking.enabled = False
        mock_settings.chunking.enabled = False
        mock_settings.summary.generation_enabled = False
        # Step 22 reads semantic_search.mode; 'false' forces the tool off.
        mock_settings.semantic_search.mode = 'false'
        mock_settings.semantic_search.enabled = False
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the inline INFO log reports
        # disabled without a read.
        mock_settings.compression.enabled = False

        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                _patch_server_migrations(),
                patch('app.tools.register_tool', return_value=True),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
            ):
                mock_mcp = MagicMock()
                mock_mcp.list_tools = AsyncMock(return_value=[])

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
        # The compression validator probes provenance even when disabled; an
        # awaitable execute_read returning None models a never-compressed DB.
        mock_backend.execute_read = AsyncMock(return_value=None)

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
        mock_settings.embedding.generation_enabled = True
        mock_settings.reranking.enabled = False
        mock_settings.chunking.enabled = False
        mock_settings.summary.generation_enabled = False
        # Step 22 reads semantic_search.mode; 'true' with a provider present
        # registers semantic_search_context.
        mock_settings.semantic_search.mode = 'true'
        mock_settings.semantic_search.enabled = True
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        mock_settings.embedding.provider = 'ollama'
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the inline INFO log reports
        # disabled without a read.
        mock_settings.compression.enabled = False

        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                _patch_server_migrations(),
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
                mock_mcp.list_tools = AsyncMock(return_value=[])

                async with lifespan(mock_mcp):
                    # Verify embedding provider was set
                    assert app.startup._embedding_provider is not None

                # Verify embedding provider shutdown was called
                mock_embedding_provider.shutdown.assert_awaited_once()
        finally:
            app.startup._backend = original_backend
            app.startup._repositories = original_repos
            app.startup._embedding_provider = original_provider


class TestSearchToolRegistrationMatrix:
    """Tri-state registration matrix for the three search tools.

    Exercises the step-22/23/24 registration logic in app/server.py lifespan()
    at the unit level: mocked settings plus a mocked (or absent) embedding
    provider drive whether semantic_search_context, fts_search_context, and
    hybrid_search_context are registered. The registration decision reads
    settings.semantic_search.mode (string) for semantic search and the derived
    settings.fts.enabled / settings.hybrid_search.enabled properties for the
    other two tools.
    """

    @staticmethod
    async def _run_lifespan(
        *,
        semantic_mode: str,
        fts_enabled: bool,
        hybrid_enabled: bool,
        provider_present: bool,
    ) -> set[str]:
        """Run lifespan() with mocked dependencies and capture registered tools.

        Args:
            semantic_mode: Value bound to settings.semantic_search.mode.
            fts_enabled: Value bound to settings.fts.enabled property.
            hybrid_enabled: Value bound to settings.hybrid_search.enabled property.
            provider_present: When True, an embedding provider is created and set
                so get_embedding_provider() returns it; when False, embedding
                generation is disabled and no provider is initialized.

        Returns:
            The set of registered tool function names captured from the
            app.server.register_tool calls made during lifespan startup.
        """
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock

        import app.startup
        from app.server import lifespan

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.shutdown = AsyncMock()
        mock_backend.backend_type = 'sqlite'
        # The compression validator probes provenance even when disabled; an
        # awaitable execute_read returning None models a never-compressed DB.
        mock_backend.execute_read = AsyncMock(return_value=None)

        mock_repos = MagicMock()
        mock_repos.fts.is_available = AsyncMock(return_value=True)
        mock_repos.context = MagicMock()
        mock_repos.embedding = MagicMock()

        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = provider_present
        mock_settings.reranking.enabled = False
        mock_settings.chunking.enabled = False
        mock_settings.summary.generation_enabled = False
        mock_settings.semantic_search.mode = semantic_mode
        # The derived .enabled property mirrors mode != 'false'; set it
        # explicitly because step 24 (hybrid) reads the property directly.
        mock_settings.semantic_search.enabled = semantic_mode != 'false'
        mock_settings.fts.enabled = fts_enabled
        mock_settings.fts.language = 'english'
        mock_settings.hybrid_search.enabled = hybrid_enabled
        mock_settings.embedding.provider = 'ollama'
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the inline INFO log reports
        # disabled without a read.
        mock_settings.compression.enabled = False

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.initialize = AsyncMock()
        mock_embedding_provider.shutdown = AsyncMock()
        mock_embedding_provider.is_available = AsyncMock(return_value=True)
        mock_embedding_provider.provider_name = 'test-provider'

        registered: set[str] = set()

        def _capture_register_tool(_mcp: object, func: object, *_args: object, **_kwargs: object) -> bool:
            registered.add(getattr(func, '__name__', repr(func)))
            return True

        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                _patch_server_migrations(),
                patch('app.server.register_tool', side_effect=_capture_register_tool),
                patch('app.server.generate_fts_description', return_value='fts description'),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
                patch('app.server.check_vector_storage_dependencies', new=AsyncMock(return_value=True)),
                patch(
                    'app.server.check_provider_dependencies',
                    new=AsyncMock(return_value={'available': True, 'reason': None, 'install_instructions': None}),
                ),
                patch('app.server.create_embedding_provider', return_value=mock_embedding_provider),
            ):
                mock_mcp = MagicMock()
                mock_mcp.list_tools = AsyncMock(return_value=[])

                async with lifespan(mock_mcp):
                    pass
        finally:
            app.startup._backend = original_backend
            app.startup._repositories = original_repos
            app.startup._embedding_provider = original_provider

        return registered

    @pytest.mark.asyncio
    async def test_semantic_auto_with_provider_registers(self) -> None:
        """mode='auto' + provider present -> semantic_search_context registered."""
        registered = await self._run_lifespan(
            semantic_mode='auto',
            fts_enabled=False,
            hybrid_enabled=False,
            provider_present=True,
        )
        assert 'semantic_search_context' in registered

    @pytest.mark.asyncio
    async def test_semantic_auto_without_provider_not_registered(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """mode='auto' + no provider -> NOT registered, logged at INFO (not WARNING)."""
        import logging

        caplog.set_level(logging.INFO, logger='app.server')

        registered = await self._run_lifespan(
            semantic_mode='auto',
            fts_enabled=False,
            hybrid_enabled=False,
            provider_present=False,
        )
        assert 'semantic_search_context' not in registered

        semantic_records = [
            r for r in caplog.records
            if 'semantic_search_context not registered' in r.message
        ]
        assert len(semantic_records) >= 1
        # auto + no provider is an informational skip, never a warning
        assert all(r.levelno == logging.INFO for r in semantic_records)

    @pytest.mark.asyncio
    async def test_semantic_true_without_provider_warns_not_registered(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """mode='true' + no provider -> NOT registered + a WARNING logged."""
        import logging

        caplog.set_level(logging.INFO, logger='app.server')

        registered = await self._run_lifespan(
            semantic_mode='true',
            fts_enabled=False,
            hybrid_enabled=False,
            provider_present=False,
        )
        assert 'semantic_search_context' not in registered

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'ENABLE_SEMANTIC_SEARCH=true' in r.message
        ]
        assert len(warnings) >= 1

    @pytest.mark.asyncio
    async def test_semantic_false_with_provider_not_registered(self) -> None:
        """mode='false' -> NOT registered even when an embedding provider exists."""
        registered = await self._run_lifespan(
            semantic_mode='false',
            fts_enabled=False,
            hybrid_enabled=False,
            provider_present=True,
        )
        assert 'semantic_search_context' not in registered

    @pytest.mark.asyncio
    async def test_fts_and_hybrid_auto_register_by_default(self) -> None:
        """FTS and hybrid auto-register by default (enabled property True)."""
        registered = await self._run_lifespan(
            semantic_mode='auto',
            fts_enabled=True,
            hybrid_enabled=True,
            provider_present=False,
        )
        # FTS uses built-in database capabilities, so it registers with no provider.
        assert 'fts_search_context' in registered
        # Hybrid registers because at least one mode (FTS) is available.
        assert 'hybrid_search_context' in registered
