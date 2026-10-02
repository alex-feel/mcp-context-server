"""Tests for the provider initialization phase of the server lifespan (app/startup/providers.py).

The embedding provider outcomes are driven through ``lifespan()`` with the provider
dependency checks and factory patched where ``app.startup.providers`` looks them up.
"""

from unittest.mock import patch

import pytest

from tests.helpers import patch_database_setup_steps


class TestLifespanErrorHandling:
    """Tests for lifespan() error handling of the embedding provider initialization."""

    @pytest.mark.asyncio
    async def test_embedding_provider_failure_when_enabled_raises(self) -> None:
        """Verify server fails to start when ENABLE_EMBEDDING_GENERATION=true but provider fails.

        ENABLE_EMBEDDING_GENERATION defaults to true.
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
        # register_search_tools reads semantic_search.mode (tri-state string);
        # 'true' registers the tool only when a provider is available (this test
        # supplies one); a missing provider logs a warning and skips registration.
        mock_settings.semantic_search.mode = 'true'
        mock_settings.semantic_search.enabled = True
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        mock_settings.embedding.provider = 'ollama'
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the compression announcement reports
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
                patch_database_setup_steps(),
                patch('app.server.RepositoryContainer', return_value=mock_repos),
                patch('app.startup.providers.check_vector_storage_dependencies', new=AsyncMock(return_value=True)),
                patch(
                    'app.startup.providers.check_provider_dependencies',
                    new=AsyncMock(return_value={'available': True, 'reason': None, 'install_instructions': None}),
                ),
                patch('app.startup.providers.create_embedding_provider', side_effect=ImportError('Provider not installed')),
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
        # register_search_tools reads semantic_search.mode; 'false' forces the tool off,
        # matching the intent that no provider is available here.
        mock_settings.semantic_search.mode = 'false'
        mock_settings.semantic_search.enabled = False
        mock_settings.fts.enabled = False
        mock_settings.hybrid_search.enabled = False
        mock_settings.embedding.provider = 'ollama'
        # Compression off: the validator's disabled-branch provenance probe
        # finds no row (execute_read -> None) and the compression announcement reports
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
                patch_database_setup_steps(),
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
