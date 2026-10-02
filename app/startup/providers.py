"""Provider initialization phase of the server lifespan.

Creates the embedding provider, reranking provider, chunking service, and summary
provider that the tools read through the ``app.startup`` getters, then pre-warms the
Ollama models. An embedding or summary failure aborts startup while its generation is
enabled; a reranking or chunking failure leaves that feature off with a warning.
"""

import logging

from app.embeddings import create_embedding_provider
from app.errors import ConfigurationError
from app.errors import DependencyError
from app.errors import classify_provider_error
from app.migrations import check_provider_dependencies
from app.migrations import check_vector_storage_dependencies
from app.settings import AppSettings
from app.startup import prewarm_ollama_models
from app.startup import propagate_langsmith_settings
from app.startup import set_chunking_service
from app.startup import set_embedding_provider
from app.startup import set_reranking_provider
from app.startup import set_summary_provider

logger = logging.getLogger(__name__)


async def initialize_providers(settings: AppSettings, backend_type: str) -> None:
    """Initialize the embedding, reranking, chunking, and summary components in order.

    Args:
        settings: The application settings the server lifespan runs with.
        backend_type: The storage backend type, which decides the vector storage dependencies.
    """
    # Propagate LangSmith settings to os.environ BEFORE embedding provider init
    # This enables LangSmith SDK auto-detection when users configure via .env file
    propagate_langsmith_settings()
    await _initialize_embedding_provider(settings, backend_type)
    await _initialize_reranking_provider(settings)
    _initialize_chunking_service(settings)
    await _initialize_summary_provider(settings)

    # Pre-warm Ollama models (load into memory for instant first-request response)
    await prewarm_ollama_models()


async def _initialize_embedding_provider(settings: AppSettings, backend_type: str) -> None:
    """Create and register the embedding provider when embedding generation is enabled.

    Args:
        settings: The application settings the server lifespan runs with.
        backend_type: The storage backend type, which decides the vector storage dependencies.

    Raises:
        ConfigurationError: When the vector storage or provider dependencies are missing
            or the provider import fails.
        DependencyError: When the embedding service is unavailable or the provider fails
            to initialize.
    """
    # Initialize embedding generation if enabled (BEFORE semantic search)
    # ENABLE_EMBEDDING_GENERATION controls: provider initialization, embedding generation in store/update
    # ENABLE_SEMANTIC_SEARCH controls: semantic_search_context tool registration ONLY
    if settings.embedding.generation_enabled:
        # Step 1: Check vector storage dependencies (provider-agnostic)
        vector_deps_available = await check_vector_storage_dependencies(backend_type)

        if not vector_deps_available:
            raise ConfigurationError(
                'ENABLE_EMBEDDING_GENERATION=true but vector storage dependencies not available. '
                'Fix: Install provider dependencies (e.g., uv sync --extra embeddings-ollama) '
                'OR set ENABLE_EMBEDDING_GENERATION=false to disable embeddings.',
            )

        # Step 2: Check provider-specific dependencies based on EMBEDDING_PROVIDER
        provider = settings.embedding.provider
        provider_check = await check_provider_dependencies(
            provider, settings.embedding, settings.ollama.host,
            auto_pull=settings.ollama.auto_pull,
            pull_timeout=settings.ollama.pull_timeout,
        )

        if not provider_check['available']:
            install_hint = provider_check.get('install_instructions') or 'Check provider configuration'
            reason = provider_check['reason'] or ''
            # Classify error based on reason (config vs dependency)
            error_class = classify_provider_error(reason)
            raise error_class(
                f'ENABLE_EMBEDDING_GENERATION=true but {provider} provider dependencies not met. '
                f'Reason: {reason}. '
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
                raise DependencyError(
                    f'ENABLE_EMBEDDING_GENERATION=true but {embedding_provider.provider_name} '
                    'is not available (service may be down). '
                    'Fix: Ensure the embedding service is running and accessible '
                    'OR set ENABLE_EMBEDDING_GENERATION=false to disable embeddings.',
                )

            set_embedding_provider(embedding_provider)
            logger.info(
                f'Embedding generation enabled with provider: {embedding_provider.provider_name} '
                f'(model: {settings.embedding.model})',
            )

        except ImportError as e:
            raise ConfigurationError(
                f'ENABLE_EMBEDDING_GENERATION=true but provider import failed: {e}. '
                f'Fix: Install provider dependencies (e.g., uv sync --extra embeddings-{provider}) '
                f'OR set ENABLE_EMBEDDING_GENERATION=false to disable embeddings.',
            ) from e
        except (ConfigurationError, DependencyError):
            raise  # Re-raise our specific error types
        except Exception as e:
            # Unknown initialization errors are treated as dependency issues (may recover)
            raise DependencyError(
                f'ENABLE_EMBEDDING_GENERATION=true but initialization failed: {e}. '
                f'Fix: Check provider configuration and service availability '
                f'OR set ENABLE_EMBEDDING_GENERATION=false to disable embeddings.',
            ) from e
    else:
        set_embedding_provider(None)
        logger.info('Embedding generation disabled (ENABLE_EMBEDDING_GENERATION=false)')


async def _initialize_reranking_provider(settings: AppSettings) -> None:
    """Create and register the reranking provider when reranking is enabled.

    Args:
        settings: The application settings the server lifespan runs with.
    """
    # Initialize reranking provider if enabled
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
                    f'Reranking provider {reranking_provider.provider_name} not available. '
                    'Search results will not be reranked.',
                )
                set_reranking_provider(None)
            else:
                set_reranking_provider(reranking_provider)
                logger.info(
                    f'Reranking enabled with provider: {reranking_provider.provider_name} '
                    f'(model: {reranking_provider.model_name})',
                )
        except ImportError as e:
            logger.warning(
                f'Reranking dependencies not installed: {e}. '
                f'Install with: uv sync --extra reranking. '
                'Search results will not be reranked.',
            )
            set_reranking_provider(None)
        except Exception as e:
            logger.warning(
                f'Failed to initialize reranking provider: {e}. '
                'Search results will not be reranked.',
            )
            set_reranking_provider(None)
    else:
        set_reranking_provider(None)
        logger.info('Reranking disabled (ENABLE_RERANKING=false)')


def _initialize_chunking_service(settings: AppSettings) -> None:
    """Create and register the chunking service when chunking is enabled.

    Args:
        settings: The application settings the server lifespan runs with.
    """
    # Initialize chunking service if enabled
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
                f'Chunking enabled (size={settings.chunking.size}, '
                f'overlap={settings.chunking.overlap})',
            )
        except ImportError as e:
            logger.warning(
                f'Chunking dependencies not installed: {e}. '
                f'Install with: uv sync --extra embeddings-ollama. '
                'Text will be embedded as single chunks.',
            )
            set_chunking_service(None)
        except Exception as e:
            logger.warning(
                f'Failed to initialize chunking service: {e}. '
                'Text will be embedded as single chunks.',
            )
            set_chunking_service(None)
    else:
        set_chunking_service(None)
        logger.info('Chunking disabled (ENABLE_CHUNKING=false)')


async def _initialize_summary_provider(settings: AppSettings) -> None:
    """Create and register the summary provider when summary generation is enabled.

    Args:
        settings: The application settings the server lifespan runs with.

    Raises:
        ConfigurationError: When the provider dependencies are missing or the provider
            import fails.
        DependencyError: When the summary service is unavailable or the provider fails
            to initialize.
    """
    # Initialize summary provider if enabled
    if settings.summary.generation_enabled:
        # Step 1: Check provider-specific dependencies based on SUMMARY_PROVIDER
        from app.migrations import check_summary_provider_dependencies

        summary_provider_name = settings.summary.provider
        summary_check = await check_summary_provider_dependencies(
            summary_provider_name, settings.summary, settings.ollama.host,
            auto_pull=settings.ollama.auto_pull,
            pull_timeout=settings.ollama.pull_timeout,
        )

        if not summary_check['available']:
            install_hint = summary_check.get('install_instructions') or 'Check provider configuration'
            reason = summary_check['reason'] or ''
            error_class = classify_provider_error(reason)
            raise error_class(
                f'ENABLE_SUMMARY_GENERATION=true but {summary_provider_name} provider dependencies not met. '
                f'Reason: {reason}. '
                f'Fix: {install_hint} '
                f'OR set ENABLE_SUMMARY_GENERATION=false to disable summaries.',
            )

        # Step 2: Create and initialize provider
        try:
            from app.summary import create_summary_provider

            summary_provider = create_summary_provider()
            await summary_provider.initialize()

            if not await summary_provider.is_available():
                await summary_provider.shutdown()
                raise DependencyError(
                    f'ENABLE_SUMMARY_GENERATION=true but {summary_provider.provider_name} '
                    'is not available (service may be down). '
                    'Fix: Ensure the summary service is running and accessible '
                    'OR set ENABLE_SUMMARY_GENERATION=false to disable summaries.',
                )

            set_summary_provider(summary_provider)
            logger.info(
                f'Summary generation enabled with provider: {summary_provider.provider_name} '
                f'(model: {settings.summary.model})',
            )

        except ImportError as e:
            raise ConfigurationError(
                f'ENABLE_SUMMARY_GENERATION=true but provider import failed: {e}. '
                f'Fix: Install provider dependencies (e.g., uv sync --extra summary-{settings.summary.provider}) '
                f'OR set ENABLE_SUMMARY_GENERATION=false to disable summaries.',
            ) from e
        except (ConfigurationError, DependencyError):
            raise
        except Exception as e:
            raise DependencyError(
                f'ENABLE_SUMMARY_GENERATION=true but initialization failed: {e}. '
                f'Fix: Check provider configuration and service availability '
                f'OR set ENABLE_SUMMARY_GENERATION=false to disable summaries.',
            ) from e
    else:
        set_summary_provider(None)
        logger.info('Summary generation disabled (ENABLE_SUMMARY_GENERATION=false)')
