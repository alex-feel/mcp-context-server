"""Tests for the server lifespan in app/server.py.

Covers startup failure cleanup, migration failure propagation, and the shutdown of the
backend and the embedding provider.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.helpers import patch_database_setup_steps


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
                patch('app.startup.database_setup.init_database', side_effect=RuntimeError('Database init failed')),
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
                patch('app.startup.database_setup.init_database', new=AsyncMock()),
                patch('app.startup.database_setup.handle_metadata_indexes', new=AsyncMock()),
                patch('app.startup.database_setup.guard_compression_disable_over_populated', new=AsyncMock()),
                patch(
                    'app.startup.database_setup.apply_semantic_search_migration',
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
        # register_search_tools reads semantic_search.mode; 'false' forces the tool off.
        mock_settings.semantic_search.mode = 'false'
        mock_settings.semantic_search.enabled = False
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the compression announcement reports
        # disabled without a read.
        mock_settings.compression.enabled = False

        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch_database_setup_steps(),
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
        # register_search_tools reads semantic_search.mode; 'true' with a provider present
        # registers semantic_search_context.
        mock_settings.semantic_search.mode = 'true'
        mock_settings.semantic_search.enabled = True
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        mock_settings.embedding.provider = 'ollama'
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the compression announcement reports
        # disabled without a read.
        mock_settings.compression.enabled = False

        original_backend = app.startup._backend
        original_repos = app.startup._repositories
        original_provider = app.startup._embedding_provider

        try:
            with (
                patch('app.server.settings', mock_settings),
                patch('app.server.create_backend', return_value=mock_backend),
                patch_database_setup_steps(),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
                patch('app.startup.providers.check_vector_storage_dependencies', new=AsyncMock(return_value=True)),
                patch(
                    'app.startup.providers.check_provider_dependencies',
                    new=AsyncMock(return_value={'available': True, 'reason': None, 'install_instructions': None}),
                ),
                patch('app.startup.providers.create_embedding_provider', return_value=mock_embedding_provider),
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
