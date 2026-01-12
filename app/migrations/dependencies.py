"""
Provider and vector storage dependency checking for semantic search.

This module provides functions to check if all required dependencies
are available before enabling semantic search functionality.
"""

import importlib.util
import logging
from collections.abc import Callable
from typing import Any
from typing import TypedDict
from typing import cast

from app.settings import EmbeddingSettings

logger = logging.getLogger(__name__)


class ProviderCheckResult(TypedDict):
    """Result of provider dependency check."""

    available: bool
    reason: str | None
    install_instructions: str | None


async def check_vector_storage_dependencies(backend_type: str = 'sqlite') -> bool:
    """Check vector storage dependencies for semantic search (provider-AGNOSTIC).

    Performs checks for:
    - Python packages: numpy, sqlite_vec (SQLite) or pgvector (PostgreSQL)
    - sqlite-vec extension loading (SQLite only)

    Provider-specific checks (API keys, service availability, model availability)
    are handled by check_provider_dependencies().

    Args:
        backend_type: Either 'sqlite' or 'postgresql'

    Returns:
        True if vector storage dependencies are available, False otherwise
    """
    logger.info('Checking vector storage dependencies...')

    # Check numpy package (required for vector operations)
    try:
        if importlib.util.find_spec('numpy') is None:
            logger.warning('[X] numpy package not available')
            logger.warning('  Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)')
            return False
        logger.debug('[OK] numpy package available')
    except ImportError as e:
        logger.warning(f'[X] numpy package not available: {e}')
        return False

    # Check sqlite_vec package (SQLite only)
    if backend_type == 'sqlite':
        try:
            if importlib.util.find_spec('sqlite_vec') is None:
                logger.warning('[X] sqlite_vec package not available')
                logger.warning('  Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)')
                return False
            logger.debug('[OK] sqlite_vec package available')
        except ImportError as e:
            logger.warning(f'[X] sqlite_vec package not available: {e}')
            return False

        # Check sqlite-vec extension loading
        try:
            import sqlite3

            import sqlite_vec as sqlite_vec_ext

            test_conn = sqlite3.connect(':memory:')
            test_conn.enable_load_extension(True)
            sqlite_vec_ext.load(test_conn)
            test_conn.enable_load_extension(False)
            test_conn.close()
            logger.debug('[OK] sqlite-vec extension loads successfully')
        except Exception as e:
            logger.warning(f'[X] sqlite-vec extension failed to load: {e}')
            return False

    # Check pgvector package (PostgreSQL only)
    if backend_type == 'postgresql':
        try:
            if importlib.util.find_spec('pgvector') is None:
                logger.warning('[X] pgvector package not available')
                logger.warning('  Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)')
                return False
            logger.debug('[OK] pgvector package available')
        except ImportError as e:
            logger.warning(f'[X] pgvector package not available: {e}')
            return False

    logger.info('[OK] All vector storage dependencies available')
    return True


async def check_provider_dependencies(
    provider: str,
    embedding_settings: EmbeddingSettings,
) -> ProviderCheckResult:
    """Check provider-specific dependencies based on EMBEDDING_PROVIDER setting.

    Dispatches to provider-specific check functions based on the selected provider.
    Each provider has different requirements:
    - ollama: Requires Ollama service running and model available
    - openai: Requires OPENAI_API_KEY
    - azure: Requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, deployment name
    - huggingface: Requires HUGGINGFACEHUB_API_TOKEN
    - voyage: Requires VOYAGE_API_KEY

    Args:
        provider: Provider name from EMBEDDING_PROVIDER setting
        embedding_settings: EmbeddingSettings instance with provider configuration

    Returns:
        ProviderCheckResult with available, reason, and install_instructions
    """
    check_functions: dict[
        str,
        Callable[[EmbeddingSettings], Any],
    ] = {
        'ollama': _check_ollama_dependencies,
        'openai': _check_openai_dependencies,
        'azure': _check_azure_dependencies,
        'huggingface': _check_huggingface_dependencies,
        'voyage': _check_voyage_dependencies,
    }

    if provider not in check_functions:
        return ProviderCheckResult(
            available=False,
            reason=f"Unknown provider: '{provider}'",
            install_instructions=None,
        )

    logger.info(f'Checking {provider} provider dependencies...')
    result = await check_functions[provider](embedding_settings)
    return cast(ProviderCheckResult, result)


async def _check_ollama_dependencies(embedding_settings: EmbeddingSettings) -> ProviderCheckResult:
    """Check Ollama-specific dependencies.

    Checks:
    1. langchain-ollama package is installed
    2. Ollama service is running at OLLAMA_HOST
    3. Embedding model is available

    Args:
        embedding_settings: EmbeddingSettings with ollama_host and model

    Returns:
        ProviderCheckResult
    """
    install_cmd = 'uv sync --extra embeddings-ollama'

    # 1. Check langchain-ollama package
    try:
        if importlib.util.find_spec('langchain_ollama') is None:
            return ProviderCheckResult(
                available=False,
                reason='langchain-ollama package not installed',
                install_instructions=install_cmd,
            )
        logger.debug('[OK] langchain-ollama package available')
    except ImportError as e:
        return ProviderCheckResult(
            available=False,
            reason=f'langchain-ollama package not available: {e}',
            install_instructions=install_cmd,
        )

    # 2. Check Ollama service is running
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(embedding_settings.ollama_host, timeout=2.0)
            if response.status_code != 200:
                return ProviderCheckResult(
                    available=False,
                    reason=f'Ollama service returned status {response.status_code}',
                    install_instructions='Start Ollama service: ollama serve',
                )
        logger.debug(f'[OK] Ollama service running at {embedding_settings.ollama_host}')
    except Exception as e:
        return ProviderCheckResult(
            available=False,
            reason=f'Ollama service not accessible at {embedding_settings.ollama_host}: {e}',
            install_instructions='Start Ollama service: ollama serve',
        )

    # 3. Check embedding model is available
    try:
        import ollama

        ollama_client = ollama.Client(host=embedding_settings.ollama_host, timeout=5.0)
        ollama_client.show(embedding_settings.model)
        logger.debug(f'[OK] Embedding model "{embedding_settings.model}" available')
    except Exception as e:
        return ProviderCheckResult(
            available=False,
            reason=f'Embedding model "{embedding_settings.model}" not available: {e}',
            install_instructions=f'Download model: ollama pull {embedding_settings.model}',
        )

    logger.info('[OK] All Ollama provider dependencies available')
    return ProviderCheckResult(available=True, reason=None, install_instructions=None)


async def _check_openai_dependencies(embedding_settings: EmbeddingSettings) -> ProviderCheckResult:
    """Check OpenAI-specific dependencies.

    Checks:
    1. langchain-openai package is installed
    2. OPENAI_API_KEY is set

    Args:
        embedding_settings: EmbeddingSettings with openai_api_key

    Returns:
        ProviderCheckResult
    """
    install_cmd = 'uv sync --extra embeddings-openai'

    # 1. Check langchain-openai package
    try:
        if importlib.util.find_spec('langchain_openai') is None:
            return ProviderCheckResult(
                available=False,
                reason='langchain-openai package not installed',
                install_instructions=install_cmd,
            )
        logger.debug('[OK] langchain-openai package available')
    except ImportError as e:
        return ProviderCheckResult(
            available=False,
            reason=f'langchain-openai package not available: {e}',
            install_instructions=install_cmd,
        )

    # 2. Check API key is set
    if embedding_settings.openai_api_key is None:
        return ProviderCheckResult(
            available=False,
            reason='OPENAI_API_KEY environment variable is not set',
            install_instructions='Set environment variable: export OPENAI_API_KEY=your-key',
        )
    logger.debug('[OK] OPENAI_API_KEY is set')

    logger.info('[OK] All OpenAI provider dependencies available')
    return ProviderCheckResult(available=True, reason=None, install_instructions=None)


async def _check_azure_dependencies(embedding_settings: EmbeddingSettings) -> ProviderCheckResult:
    """Check Azure OpenAI-specific dependencies.

    Checks:
    1. langchain-openai package is installed (Azure uses same package)
    2. AZURE_OPENAI_API_KEY is set
    3. AZURE_OPENAI_ENDPOINT is set
    4. AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME is set

    Args:
        embedding_settings: EmbeddingSettings with Azure configuration

    Returns:
        ProviderCheckResult
    """
    install_cmd = 'uv sync --extra embeddings-azure'

    # 1. Check langchain-openai package
    try:
        if importlib.util.find_spec('langchain_openai') is None:
            return ProviderCheckResult(
                available=False,
                reason='langchain-openai package not installed',
                install_instructions=install_cmd,
            )
        logger.debug('[OK] langchain-openai package available')
    except ImportError as e:
        return ProviderCheckResult(
            available=False,
            reason=f'langchain-openai package not available: {e}',
            install_instructions=install_cmd,
        )

    # 2. Check required environment variables
    missing_vars: list[str] = []
    if embedding_settings.azure_openai_api_key is None:
        missing_vars.append('AZURE_OPENAI_API_KEY')
    if embedding_settings.azure_openai_endpoint is None:
        missing_vars.append('AZURE_OPENAI_ENDPOINT')
    if embedding_settings.azure_openai_deployment_name is None:
        missing_vars.append('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME')

    if missing_vars:
        return ProviderCheckResult(
            available=False,
            reason=f'Required environment variables not set: {", ".join(missing_vars)}',
            install_instructions=f'Set environment variables: {", ".join(missing_vars)}',
        )
    logger.debug('[OK] All Azure configuration variables are set')

    logger.info('[OK] All Azure OpenAI provider dependencies available')
    return ProviderCheckResult(available=True, reason=None, install_instructions=None)


async def _check_huggingface_dependencies(embedding_settings: EmbeddingSettings) -> ProviderCheckResult:
    """Check HuggingFace-specific dependencies.

    Checks:
    1. langchain-huggingface package is installed
    2. HUGGINGFACEHUB_API_TOKEN is set

    Args:
        embedding_settings: EmbeddingSettings with huggingface_api_key

    Returns:
        ProviderCheckResult
    """
    install_cmd = 'uv sync --extra embeddings-huggingface'

    # 1. Check langchain-huggingface package
    try:
        if importlib.util.find_spec('langchain_huggingface') is None:
            return ProviderCheckResult(
                available=False,
                reason='langchain-huggingface package not installed',
                install_instructions=install_cmd,
            )
        logger.debug('[OK] langchain-huggingface package available')
    except ImportError as e:
        return ProviderCheckResult(
            available=False,
            reason=f'langchain-huggingface package not available: {e}',
            install_instructions=install_cmd,
        )

    # 2. Check API token is set
    if embedding_settings.huggingface_api_key is None:
        return ProviderCheckResult(
            available=False,
            reason='HUGGINGFACEHUB_API_TOKEN environment variable is not set',
            install_instructions='Set environment variable: export HUGGINGFACEHUB_API_TOKEN=your-token',
        )
    logger.debug('[OK] HUGGINGFACEHUB_API_TOKEN is set')

    logger.info('[OK] All HuggingFace provider dependencies available')
    return ProviderCheckResult(available=True, reason=None, install_instructions=None)


async def _check_voyage_dependencies(embedding_settings: EmbeddingSettings) -> ProviderCheckResult:
    """Check Voyage AI-specific dependencies.

    Checks:
    1. langchain-voyageai package is installed
    2. VOYAGE_API_KEY is set

    Args:
        embedding_settings: EmbeddingSettings with voyage_api_key

    Returns:
        ProviderCheckResult
    """
    install_cmd = 'uv sync --extra embeddings-voyage'

    # 1. Check langchain-voyageai package
    try:
        if importlib.util.find_spec('langchain_voyageai') is None:
            return ProviderCheckResult(
                available=False,
                reason='langchain-voyageai package not installed',
                install_instructions=install_cmd,
            )
        logger.debug('[OK] langchain-voyageai package available')
    except ImportError as e:
        return ProviderCheckResult(
            available=False,
            reason=f'langchain-voyageai package not available: {e}',
            install_instructions=install_cmd,
        )

    # 2. Check API key is set
    if embedding_settings.voyage_api_key is None:
        return ProviderCheckResult(
            available=False,
            reason='VOYAGE_API_KEY environment variable is not set',
            install_instructions='Set environment variable: export VOYAGE_API_KEY=your-key',
        )
    logger.debug('[OK] VOYAGE_API_KEY is set')

    logger.info('[OK] All Voyage AI provider dependencies available')
    return ProviderCheckResult(available=True, reason=None, install_instructions=None)
