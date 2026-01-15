"""
MCP Context Server implementation using FastMCP.

This server provides persistent multimodal context storage capabilities for LLM agents,
enabling shared memory across different conversation threads with support for text and images.
"""

import logging
import tomllib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Literal
from typing import cast

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# ============================================================================
# CRITICAL: Logger Configuration MUST happen BEFORE importing app modules
# that trigger backend loading (app.backends, app.repositories)
# ============================================================================
from app.logger_config import config_logger
from app.settings import get_settings

settings = get_settings()
config_logger(settings.logging.level)
logger = logging.getLogger(__name__)

# Import startup module for global state access
from app.backends import create_backend
from app.embeddings import create_embedding_provider
from app.migrations import FtsMigrationStatus
from app.migrations import ProviderCheckResult

# Import migration functions from the migrations package
from app.migrations import apply_chunking_migration
from app.migrations import apply_fts_migration
from app.migrations import apply_function_search_path_migration
from app.migrations import apply_jsonb_merge_patch_migration
from app.migrations import apply_semantic_search_migration
from app.migrations import check_provider_dependencies
from app.migrations import check_vector_storage_dependencies
from app.migrations import estimate_migration_time
from app.migrations import handle_metadata_indexes
from app.migrations import reset_fts_migration_status
from app.repositories import RepositoryContainer

# Import from startup module - global state and initialization
from app.startup import DB_PATH
from app.startup import get_backend
from app.startup import get_embedding_provider
from app.startup import get_reranking_provider
from app.startup import init_database
from app.startup import propagate_langsmith_settings
from app.startup import set_backend
from app.startup import set_chunking_service
from app.startup import set_embedding_provider
from app.startup import set_repositories
from app.startup import set_reranking_provider

# Backward compatibility re-exports for validation utilities
# Tests and external code may import these from app.server
from app.startup.validation import deserialize_json_param
from app.startup.validation import truncate_text
from app.startup.validation import validate_date_param
from app.startup.validation import validate_date_range

# Import validation utilities from startup (for internal use)
from app.startup.validation import validate_pool_timeout_for_embedding

# Import tool functions and registration helpers from app.tools
from app.tools import TOOL_ANNOTATIONS
from app.tools import is_tool_disabled
from app.tools import register_tool

# Backward compatibility re-exports
# These are intentionally re-exported for existing code that imports from app.server
__all__ = [
    # From app.migrations
    'FtsMigrationStatus',
    'ProviderCheckResult',
    'estimate_migration_time',
    # From app.startup (for tests)
    'DB_PATH',
    'MAX_IMAGE_SIZE_MB',
    'MAX_TOTAL_SIZE_MB',
    'init_database',
    # From app.tools (for tests)
    'TOOL_ANNOTATIONS',
    'is_tool_disabled',
    # From app.startup.validation (for tests)
    'deserialize_json_param',
    'truncate_text',
    'validate_date_param',
    'validate_date_range',
    # Tool functions (for tests)
    'delete_context',
    'delete_context_batch',
    'fts_search_context',
    'get_context_by_ids',
    'get_statistics',
    'hybrid_search_context',
    'list_threads',
    'search_context',
    'semantic_search_context',
    'store_context',
    'store_context_batch',
    'update_context',
    'update_context_batch',
]
# Import additional startup constants for backward compatibility (used by tests)
from app.startup import MAX_IMAGE_SIZE_MB
from app.startup import MAX_TOTAL_SIZE_MB
from app.tools import delete_context
from app.tools import delete_context_batch
from app.tools import fts_search_context
from app.tools import get_context_by_ids
from app.tools import get_statistics
from app.tools import hybrid_search_context
from app.tools import list_threads
from app.tools import search_context
from app.tools import semantic_search_context
from app.tools import store_context
from app.tools import store_context_batch
from app.tools import update_context
from app.tools import update_context_batch


def _get_server_version() -> str:
    """Get server version from package metadata or pyproject.toml fallback.

    Returns:
        Version string (e.g., '0.14.0') or 'unknown' if unavailable.
    """
    # Primary: installed package metadata (works for pip, uv, editable installs)
    try:
        return pkg_version('mcp-context-server')
    except PackageNotFoundError:
        pass

    # Fallback: read directly from pyproject.toml (for running from source)
    try:
        pyproject_path = Path(__file__).resolve().parents[1] / 'pyproject.toml'
        if pyproject_path.exists():
            with pyproject_path.open('rb') as f:
                data = tomllib.load(f)
            version = data.get('project', {}).get('version')
            if isinstance(version, str):
                return version
    except Exception:
        pass

    return 'unknown'


# Cache version at module load time
SERVER_VERSION = _get_server_version()


# Backward compatibility: alias for the old private function name
_reset_fts_migration_status = reset_fts_migration_status


# Lifespan context manager for FastMCP
@asynccontextmanager
async def lifespan(mcp: FastMCP[None]) -> AsyncGenerator[None, None]:
    """Manage server lifecycle - initialize on startup, cleanup on shutdown.

    This ensures that the database manager's background tasks run in the
    same event loop as FastMCP, preventing the hanging issue.

    Args:
        mcp: The FastMCP server instance for tool registration

    Yields:
        None: Control is yielded back to FastMCP during server operation
    """
    # Startup
    try:
        # Create backend ONCE at the start - used throughout initialization and runtime
        backend = create_backend(backend_type=None, db_path=DB_PATH)
        await backend.initialize()
        set_backend(backend)
        # 1) Ensure schema exists using the shared backend
        await init_database(backend=backend)
        # 2) Handle metadata field indexing (configurable via METADATA_INDEXED_FIELDS)
        await handle_metadata_indexes(backend=backend)
        # 3) Apply semantic search migration if enabled using the shared backend
        await apply_semantic_search_migration(backend=backend)
        # 4) Apply jsonb_merge_patch migration for PostgreSQL (required for metadata_patch)
        await apply_jsonb_merge_patch_migration(backend=backend)
        # 5) Apply function search_path security fix for PostgreSQL
        await apply_function_search_path_migration(backend=backend)
        # 6) Apply FTS migration if enabled
        await apply_fts_migration(backend=backend)
        # 7) Apply chunking migration (1:N embedding relationship)
        await apply_chunking_migration(backend=backend)
        # 8) Validate pool timeout for embedding operations (PostgreSQL only)
        if backend.backend_type == 'postgresql':
            validate_pool_timeout_for_embedding()
        # 9) Initialize repositories with the backend
        repos = RepositoryContainer(backend)
        set_repositories(repos)

        # 10) Register core tools (annotations from TOOL_ANNOTATIONS in app.tools)
        # Additive tools (create new entries)
        register_tool(mcp, store_context)
        register_tool(mcp, store_context_batch)

        # Read-only tools (no modifications)
        register_tool(mcp, search_context)
        register_tool(mcp, get_context_by_ids)
        register_tool(mcp, list_threads)
        register_tool(mcp, get_statistics)

        # Update tools (destructive, not idempotent)
        register_tool(mcp, update_context)
        register_tool(mcp, update_context_batch)

        # Delete tools (destructive, idempotent)
        register_tool(mcp, delete_context)
        register_tool(mcp, delete_context_batch)

        # 11) Propagate LangSmith settings to os.environ BEFORE embedding provider init
        # This enables LangSmith SDK auto-detection when users configure via .env file
        propagate_langsmith_settings()

        # 12) Initialize embedding generation if enabled (BEFORE semantic search)
        # ENABLE_EMBEDDING_GENERATION controls: provider initialization, embedding generation in store/update
        # ENABLE_SEMANTIC_SEARCH controls: semantic_search_context tool registration ONLY
        if settings.embedding.generation_enabled:
            # Step 1: Check vector storage dependencies (provider-agnostic)
            vector_deps_available = await check_vector_storage_dependencies(backend.backend_type)

            if not vector_deps_available:
                raise RuntimeError(
                    'ENABLE_EMBEDDING_GENERATION=true but vector storage dependencies not available. '
                    'Fix: Install provider dependencies (e.g., uv sync --extra embeddings-ollama) '
                    'OR set ENABLE_EMBEDDING_GENERATION=false to disable embeddings.',
                )

            # Step 2: Check provider-specific dependencies based on EMBEDDING_PROVIDER
            provider = settings.embedding.provider
            provider_check = await check_provider_dependencies(provider, settings.embedding)

            if not provider_check['available']:
                install_hint = provider_check.get('install_instructions') or 'Check provider configuration'
                raise RuntimeError(
                    f'ENABLE_EMBEDDING_GENERATION=true but {provider} provider dependencies not met. '
                    f'Reason: {provider_check["reason"]}. '
                    f'Fix: {install_hint} '
                    f'OR set ENABLE_EMBEDDING_GENERATION=false to disable embeddings.',
                )

            # Step 3: Create and initialize provider
            try:
                embedding_provider = create_embedding_provider()
                await embedding_provider.initialize()

                # Verify provider is available
                if not await embedding_provider.is_available():
                    await embedding_provider.shutdown()
                    raise RuntimeError(
                        f'ENABLE_EMBEDDING_GENERATION=true but {embedding_provider.provider_name} '
                        'is not available (service may be down). '
                        'Fix: Ensure the embedding service is running and accessible '
                        'OR set ENABLE_EMBEDDING_GENERATION=false to disable embeddings.',
                    )

                set_embedding_provider(embedding_provider)
                logger.info(f'[OK] Embedding generation enabled with provider: {embedding_provider.provider_name}')

            except ImportError as e:
                raise RuntimeError(
                    f'ENABLE_EMBEDDING_GENERATION=true but provider import failed: {e}. '
                    f'Fix: Install provider dependencies (e.g., uv sync --extra embeddings-{provider}) '
                    f'OR set ENABLE_EMBEDDING_GENERATION=false to disable embeddings.',
                ) from e
            except RuntimeError:
                raise  # Re-raise RuntimeError from above checks
            except Exception as e:
                raise RuntimeError(
                    f'ENABLE_EMBEDDING_GENERATION=true but initialization failed: {e}. '
                    f'Fix: Check provider configuration and service availability '
                    f'OR set ENABLE_EMBEDDING_GENERATION=false to disable embeddings.',
                ) from e
        else:
            set_embedding_provider(None)
            logger.info('Embedding generation disabled (ENABLE_EMBEDDING_GENERATION=false)')

        # 13) Initialize reranking provider if enabled
        # Reranking improves search precision by re-scoring results with a cross-encoder
        if settings.reranking.enabled:
            try:
                from app.reranking import create_reranking_provider

                reranking_provider = create_reranking_provider()
                await reranking_provider.initialize()

                # Verify provider is available
                if not await reranking_provider.is_available():
                    await reranking_provider.shutdown()
                    logger.warning(
                        f'[!] Reranking provider {reranking_provider.provider_name} not available. '
                        'Search results will not be reranked.',
                    )
                    set_reranking_provider(None)
                else:
                    set_reranking_provider(reranking_provider)
                    logger.info(
                        f'[OK] Reranking enabled with provider: {reranking_provider.provider_name} '
                        f'(model: {reranking_provider.model_name})',
                    )
            except ImportError as e:
                logger.warning(
                    f'[!] Reranking dependencies not installed: {e}. '
                    f'Install with: uv sync --extra reranking. '
                    'Search results will not be reranked.',
                )
                set_reranking_provider(None)
            except Exception as e:
                logger.warning(
                    f'[!] Failed to initialize reranking provider: {e}. '
                    'Search results will not be reranked.',
                )
                set_reranking_provider(None)
        else:
            set_reranking_provider(None)
            logger.info('Reranking disabled (ENABLE_RERANKING=false)')

        # 14) Initialize chunking service if enabled
        # Chunking splits long documents into smaller pieces for better semantic search quality
        if settings.chunking.enabled:
            try:
                from app.services import ChunkingService

                chunking_service = ChunkingService(
                    enabled=settings.chunking.enabled,
                    chunk_size=settings.chunking.size,
                    chunk_overlap=settings.chunking.overlap,
                )
                set_chunking_service(chunking_service)
                logger.info(
                    f'[OK] Chunking enabled (size={settings.chunking.size}, '
                    f'overlap={settings.chunking.overlap})',
                )
            except ImportError as e:
                logger.warning(
                    f'[!] Chunking dependencies not installed: {e}. '
                    f'Install with: uv sync --extra embeddings-ollama. '
                    'Text will be embedded as single chunks.',
                )
                set_chunking_service(None)
            except Exception as e:
                logger.warning(
                    f'[!] Failed to initialize chunking service: {e}. '
                    'Text will be embedded as single chunks.',
                )
                set_chunking_service(None)
        else:
            set_chunking_service(None)
            logger.info('Chunking disabled (ENABLE_CHUNKING=false)')

        # 15) Register semantic search tool if enabled AND embedding provider is available
        # This is a separate check because ENABLE_SEMANTIC_SEARCH only controls tool registration
        if settings.semantic_search.enabled:
            if get_embedding_provider() is not None:
                register_tool(mcp, semantic_search_context)
                logger.info('[OK] semantic_search_context registered')
            else:
                # User explicitly set ENABLE_EMBEDDING_GENERATION=false but ENABLE_SEMANTIC_SEARCH=true
                # This is a valid configuration - user wants no embeddings but enabled the flag
                logger.warning(
                    '[!] ENABLE_SEMANTIC_SEARCH=true but ENABLE_EMBEDDING_GENERATION=false - '
                    'semantic_search_context NOT registered (no embedding provider available)',
                )
        else:
            logger.info('Semantic search disabled (ENABLE_SEMANTIC_SEARCH=false)')
            logger.info('[!] semantic_search_context not registered (feature disabled)')

        # 16) Register FTS tool if enabled - ALWAYS register when ENABLE_FTS=true
        # The tool handles graceful degradation during migration
        if settings.fts.enabled:
            # Always register the FTS tool when enabled (DISABLED_TOOLS takes priority)
            # The tool itself checks migration status and returns informative response
            register_tool(mcp, fts_search_context)

            # Check if FTS is available and log status
            fts_available = await repos.fts.is_available()
            if fts_available:
                logger.info('[OK] Full-text search enabled and available')
            else:
                logger.warning('[!] FTS enabled but index may need initialization or migration')
        else:
            logger.info('Full-text search disabled (ENABLE_FTS=false)')
            logger.info('[!] fts_search_context not registered (feature disabled)')

        # 17) Register Hybrid Search tool if enabled AND at least one search mode is available
        if settings.hybrid_search.enabled:
            semantic_available_for_hybrid = (
                settings.semantic_search.enabled and get_embedding_provider() is not None
            )
            fts_available_for_hybrid = settings.fts.enabled

            if semantic_available_for_hybrid or fts_available_for_hybrid:
                # DISABLED_TOOLS takes priority over ENABLE_HYBRID_SEARCH
                register_tool(mcp, hybrid_search_context)
                modes_available = []
                if fts_available_for_hybrid:
                    modes_available.append('fts')
                if semantic_available_for_hybrid:
                    modes_available.append('semantic')
                logger.info(f'[OK] hybrid_search_context modes available: {modes_available}')
            else:
                logger.warning(
                    '[!] Hybrid search enabled but no search modes available - feature disabled. '
                    'Enable ENABLE_FTS=true and/or ENABLE_SEMANTIC_SEARCH=true.',
                )
                logger.info('[!] hybrid_search_context not registered (no search modes available)')
        else:
            logger.info('Hybrid search disabled (ENABLE_HYBRID_SEARCH=false)')
            logger.info('[!] hybrid_search_context not registered (feature disabled)')

        logger.info(f'MCP Context Server initialized (backend: {backend.backend_type})')
    except Exception as e:
        logger.error(f'Failed to initialize server: {e}')
        startup_backend = get_backend()
        if startup_backend:
            await startup_backend.shutdown()
        raise

    # Yield control to FastMCP
    yield

    # Shutdown
    logger.info('Shutting down MCP Context Server')
    # At this point, startup succeeded and _backend must be set
    shutdown_backend = get_backend()
    assert shutdown_backend is not None
    try:
        await shutdown_backend.shutdown()
    except Exception as e:
        logger.error(f'Error during shutdown: {e}')
    finally:
        # Shutdown reranking provider if initialized
        shutdown_reranking_provider = get_reranking_provider()
        if shutdown_reranking_provider is not None:
            try:
                await shutdown_reranking_provider.shutdown()
            except Exception as e:
                logger.error(f'Error shutting down reranking provider: {e}')

        # Shutdown embedding provider if initialized
        shutdown_embedding_provider = get_embedding_provider()
        if shutdown_embedding_provider is not None:
            try:
                await shutdown_embedding_provider.shutdown()
            except Exception as e:
                logger.error(f'Error shutting down embedding provider: {e}')

        set_backend(None)
        set_repositories(None)
        set_embedding_provider(None)
        set_reranking_provider(None)
        set_chunking_service(None)
    logger.info('MCP Context Server shutdown complete')


# Initialize FastMCP server with lifespan management
# mask_error_details=False exposes validation errors for LLM autocorrection
mcp = FastMCP(name='mcp-context-server', lifespan=lifespan, mask_error_details=False)


@mcp.custom_route('/health', methods=['GET'])
async def health(_: Request) -> JSONResponse:
    """Health check endpoint for container orchestration.

    Returns simple status for Docker/Kubernetes liveness probes.
    This endpoint is only available when running in HTTP transport mode.
    """
    return JSONResponse({'status': 'ok'})


# Main entry point
def main() -> None:
    """Main entry point for the MCP Context Server.

    Supports both stdio (default) and HTTP transport modes:
    - stdio: Default for local process spawning (uv run mcp-context-server)
    - http: For Docker/remote deployments (set MCP_TRANSPORT=http)

    Initialization and shutdown are handled by the @mcp.startup and @mcp.shutdown decorators.
    """
    try:
        # Log server version at startup
        logger.info(f'MCP Context Server v{SERVER_VERSION}')

        transport = settings.transport.transport

        if transport == 'stdio':
            logger.info('Transport: STDIO')
            mcp.run()
        else:
            host = settings.transport.host
            port = settings.transport.port
            logger.info('Transport: HTTP')
            logger.info(f'Server URL: http://{host}:{port}/mcp')
            mcp.run(
                transport=cast(Literal['stdio', 'http', 'sse', 'streamable-http'], transport),
                host=host,
                port=port,
            )

    except KeyboardInterrupt:
        logger.info('Server shutdown requested')
    except Exception as e:
        logger.error(f'Server error: {e}')
        raise


if __name__ == '__main__':
    main()
