"""Embedding-provider runtime setup for the migration CLI.

The ``mcp-context-server-migrate`` CLI runs as a standalone process and does
NOT execute the server lifespan (``app/server.py``), so the global embedding
provider and chunking service -- read by
``app.tools._shared.generate_embeddings_with_timeout`` via
``app.startup.get_embedding_provider`` / ``get_chunking_service`` -- are unset
by default. Any CLI flow that generates embeddings (``--embed-missing``,
``--re-embed``) MUST initialize them exactly as the server lifespan does
before calling the generation pipeline; otherwise ``get_embedding_provider()``
returns ``None`` and every entry is silently skipped.

This module centralizes that setup and teardown so all CLI embedding flows
share one implementation that mirrors the server lifespan.
"""

import contextlib
import logging

from app.embeddings import create_embedding_provider
from app.embeddings.base import EmbeddingProvider
from app.settings import get_settings
from app.startup import set_chunking_service
from app.startup import set_embedding_provider

logger = logging.getLogger(__name__)


class EmbeddingPipelineUnavailableError(RuntimeError):
    """Raised when the configured embedding provider cannot be initialized.

    Signals an operator-actionable condition (provider extra not installed,
    service down, misconfiguration) so the CLI can exit with a clear message
    and a non-zero status instead of a bare traceback.
    """


async def initialize_cli_embedding_pipeline() -> EmbeddingProvider:
    """Create and register the embedding provider and chunking service.

    Mirrors the embedding-generation block of the server lifespan
    (``app/server.py``): create the provider via the configured
    ``EMBEDDING_PROVIDER`` factory, initialize it, verify availability, and
    register it via :func:`app.startup.set_embedding_provider`. When
    ``ENABLE_CHUNKING`` is true the chunking service is registered as well so
    re-embedded documents are split into the same chunks the live server would
    produce. After this call,
    :func:`app.tools._shared.generate_embeddings_with_timeout` returns
    embeddings instead of ``None``.

    Returns:
        The initialized embedding provider. The caller owns its lifetime and
        MUST pass it to :func:`shutdown_cli_embedding_pipeline` when done.

    Raises:
        EmbeddingPipelineUnavailableError: If the provider extra is not
            installed or the provider reports it is not available.
    """
    settings = get_settings()

    try:
        provider = create_embedding_provider()
        await provider.initialize()
    except ImportError as exc:
        raise EmbeddingPipelineUnavailableError(
            f'Embedding provider import failed: {exc}. Install the provider '
            f'extra (e.g. uv sync --extra embeddings-ollama) and re-run.',
        ) from exc

    if not await provider.is_available():
        with contextlib.suppress(Exception):
            await provider.shutdown()
        raise EmbeddingPipelineUnavailableError(
            f'Embedding provider {provider.provider_name!r} is not available. '
            'Ensure the embedding service is running and reachable and that '
            'EMBEDDING_PROVIDER / EMBEDDING_MODEL are configured correctly.',
        )

    set_embedding_provider(provider)
    logger.info(
        'CLI embedding pipeline ready: provider=%s model=%s',
        provider.provider_name,
        settings.embedding.model,
    )

    if settings.chunking.enabled:
        # Degrade gracefully exactly as the server lifespan does
        # (app/server.py): a missing chunking dependency must NOT crash the
        # CLI -- embedding falls back to single chunks. Without this guard the
        # ImportError would escape past callers that only catch
        # EmbeddingPipelineUnavailableError, contradicting this function's
        # contract.
        try:
            from app.services import ChunkingService

            set_chunking_service(
                ChunkingService(
                    enabled=settings.chunking.enabled,
                    chunk_size=settings.chunking.size,
                    chunk_overlap=settings.chunking.overlap,
                ),
            )
        except ImportError as exc:
            logger.warning(
                'Chunking service dependency missing (%s); embedding text as '
                'single chunks.',
                exc,
            )
            set_chunking_service(None)
        except Exception as exc:
            logger.warning(
                'Failed to initialize chunking service (%s); embedding text as '
                'single chunks.',
                exc,
            )
            set_chunking_service(None)
    else:
        set_chunking_service(None)

    return provider


async def shutdown_cli_embedding_pipeline(provider: EmbeddingProvider | None) -> None:
    """Shut down the provider and clear the registered global state.

    Clears the module-level provider/chunking globals so a subsequent
    operation in the same process (notably back-to-back test cases) starts
    from a clean slate, then shuts the provider down. Shutdown failures are
    suppressed so teardown never masks the primary result.

    Args:
        provider: The provider returned by
            :func:`initialize_cli_embedding_pipeline`, or ``None`` when
            initialization never ran.
    """
    set_embedding_provider(None)
    set_chunking_service(None)
    if provider is not None:
        with contextlib.suppress(Exception):
            await provider.shutdown()


__all__ = [
    'EmbeddingPipelineUnavailableError',
    'initialize_cli_embedding_pipeline',
    'shutdown_cli_embedding_pipeline',
]
