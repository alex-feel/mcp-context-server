"""
Provider and storage dependency checking for semantic search and summary generation.

This module provides functions to check if all required dependencies
are available before enabling semantic search and summary generation.
"""

import importlib.util
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any
from typing import TypedDict
from typing import cast

from app.settings import EmbeddingSettings

if TYPE_CHECKING:
    from app.settings import SummarySettings

logger = logging.getLogger(__name__)


class ProviderCheckResult(TypedDict):
    """Result of provider dependency check."""

    available: bool
    reason: str | None
    install_instructions: str | None


async def _check_ollama_model(
    *,
    model_name: str,
    ollama_host: str,
    auto_pull: bool,
    pull_timeout: int,
    package_name: str,
    install_cmd: str,
    feature_label: str,
) -> ProviderCheckResult:
    """Check Ollama dependencies and optionally auto-pull missing models.

    Performs three checks:
    1. Required Python package is installed
    2. Ollama service is running and accessible
    3. Requested model is available (auto-pulls if missing and enabled)

    Args:
        model_name: Ollama model identifier (e.g., 'qwen3-embedding:0.6b')
        ollama_host: Ollama server URL
        auto_pull: Whether to auto-pull missing models
        pull_timeout: Timeout in seconds for model pull operations
        package_name: Python package to check (e.g., 'langchain_ollama')
        install_cmd: Install command for the package (e.g., 'uv sync --extra embeddings-ollama')
        feature_label: Human-readable label for log messages (e.g., 'Embedding', 'Summary')

    Returns:
        ProviderCheckResult indicating availability and any failure details
    """
    # 1. Check required Python package
    try:
        if importlib.util.find_spec(package_name) is None:
            return ProviderCheckResult(
                available=False,
                reason=f'{package_name.replace("_", "-")} package not installed',
                install_instructions=install_cmd,
            )
        logger.debug(f'{package_name.replace("_", "-")} package available')
    except ImportError as e:
        return ProviderCheckResult(
            available=False,
            reason=f'{package_name.replace("_", "-")} package not available: {e}',
            install_instructions=install_cmd,
        )

    # 2. Check Ollama service is running
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(ollama_host, timeout=2.0)
            if response.status_code != 200:
                return ProviderCheckResult(
                    available=False,
                    reason=f'Ollama service returned status {response.status_code}',
                    install_instructions='Start Ollama service: ollama serve',
                )
        logger.debug(f'Ollama service running at {ollama_host}')
    except Exception as e:
        return ProviderCheckResult(
            available=False,
            reason=f'Ollama service not accessible at {ollama_host}: {e}',
            install_instructions='Start Ollama service: ollama serve',
        )

    # 3. Check model availability (with optional auto-pull)
    import ollama as ollama_lib

    try:
        show_client = ollama_lib.Client(host=ollama_host, timeout=5.0)
        show_client.show(model_name)
        logger.debug(f'{feature_label} model "{model_name}" available')
    except Exception as show_error:
        if not auto_pull:
            return ProviderCheckResult(
                available=False,
                reason=f'{feature_label} model "{model_name}" not available: {show_error}',
                install_instructions=f'Download model: ollama pull {model_name}',
            )

        # Auto-pull: attempt to download the missing model
        logger.info(f'{feature_label} model "{model_name}" not found, pulling automatically...')
        try:
            import asyncio

            pull_client = ollama_lib.Client(host=ollama_host, timeout=float(pull_timeout))
            await asyncio.to_thread(pull_client.pull, model_name, stream=False)
            logger.info(f'{feature_label} model "{model_name}" pulled successfully')
        except Exception as pull_error:
            return ProviderCheckResult(
                available=False,
                reason=f'{feature_label} model "{model_name}" auto-pull failed: {pull_error}',
                install_instructions=f'Download model manually: ollama pull {model_name}',
            )

        # Post-pull verification
        try:
            verify_client = ollama_lib.Client(host=ollama_host, timeout=5.0)
            verify_client.show(model_name)
            logger.debug(f'{feature_label} model "{model_name}" verified after pull')
        except Exception as verify_error:
            return ProviderCheckResult(
                available=False,
                reason=f'{feature_label} model "{model_name}" not usable after pull: {verify_error}',
                install_instructions=f'Download model manually: ollama pull {model_name}',
            )

    logger.info(f'All Ollama {feature_label.lower()} provider dependencies available')
    return ProviderCheckResult(available=True, reason=None, install_instructions=None)


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
            logger.warning('numpy package not available')
            logger.warning('  Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)')
            return False
        logger.debug('numpy package available')
    except ImportError as e:
        logger.warning(f'numpy package not available: {e}')
        return False

    # Check sqlite_vec package (SQLite only)
    if backend_type == 'sqlite':
        try:
            if importlib.util.find_spec('sqlite_vec') is None:
                logger.warning('sqlite_vec package not available')
                logger.warning('  Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)')
                return False
            logger.debug('sqlite_vec package available')
        except ImportError as e:
            logger.warning(f'sqlite_vec package not available: {e}')
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
            logger.debug('sqlite-vec extension loads successfully')
        except Exception as e:
            logger.warning(f'sqlite-vec extension failed to load: {e}')
            return False

    # Check pgvector package (PostgreSQL only)
    if backend_type == 'postgresql':
        try:
            if importlib.util.find_spec('pgvector') is None:
                logger.warning('pgvector package not available')
                logger.warning('  Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)')
                return False
            logger.debug('pgvector package available')
        except ImportError as e:
            logger.warning(f'pgvector package not available: {e}')
            return False

    logger.info('All vector storage dependencies available')
    return True


async def check_provider_dependencies(
    provider: str,
    embedding_settings: EmbeddingSettings,
    ollama_host: str = 'http://localhost:11434',
    *,
    auto_pull: bool = True,
    pull_timeout: int = 900,
) -> ProviderCheckResult:
    """Check provider-specific dependencies based on EMBEDDING_PROVIDER setting.

    Dispatches uniformly to the provider-specific check function.
    Each provider has different requirements:
    - ollama: Requires Ollama service running and model available
    - openai: Requires OPENAI_API_KEY
    - azure: Requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, deployment name
    - huggingface: Requires HUGGINGFACEHUB_API_TOKEN
    - voyage: Requires VOYAGE_API_KEY

    Args:
        provider: Provider name from EMBEDDING_PROVIDER setting
        embedding_settings: EmbeddingSettings instance with provider configuration
        ollama_host: Ollama server URL (from OllamaSettings.host)
        auto_pull: Whether to auto-pull missing Ollama models
        pull_timeout: Timeout in seconds for Ollama model pull operations

    Returns:
        ProviderCheckResult with available, reason, and install_instructions
    """
    check_functions: dict[
        str,
        Callable[..., Any],
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
    result = await check_functions[provider](
        embedding_settings, ollama_host, auto_pull=auto_pull, pull_timeout=pull_timeout,
    )
    return cast(ProviderCheckResult, result)


async def _check_ollama_dependencies(
    embedding_settings: EmbeddingSettings,
    ollama_host: str,
    *,
    auto_pull: bool = True,
    pull_timeout: int = 900,
) -> ProviderCheckResult:
    """Check Ollama-specific dependencies for embedding generation.

    Args:
        embedding_settings: EmbeddingSettings with model name
        ollama_host: Ollama server URL (from OllamaSettings.host)
        auto_pull: Whether to auto-pull missing models
        pull_timeout: Timeout in seconds for model pull operations

    Returns:
        ProviderCheckResult
    """
    return await _check_ollama_model(
        model_name=embedding_settings.model,
        ollama_host=ollama_host,
        auto_pull=auto_pull,
        pull_timeout=pull_timeout,
        package_name='langchain_ollama',
        install_cmd='uv sync --extra embeddings-ollama',
        feature_label='Embedding',
    )


async def _check_openai_dependencies(
    embedding_settings: EmbeddingSettings,
    _ollama_host: str,
    **_kwargs: Any,
) -> ProviderCheckResult:
    """Check OpenAI-specific dependencies.

    Checks:
    1. langchain-openai package is installed
    2. OPENAI_API_KEY is set

    Args:
        embedding_settings: EmbeddingSettings with openai_api_key
        _ollama_host: Ollama server URL (unused, accepted for uniform dispatch interface)
        **kwargs: Additional keyword arguments (unused, accepted for uniform dispatch interface)

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
        logger.debug('langchain-openai package available')
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
    logger.debug('OPENAI_API_KEY is set')

    logger.info('All OpenAI provider dependencies available')
    return ProviderCheckResult(available=True, reason=None, install_instructions=None)


async def _check_azure_dependencies(
    embedding_settings: EmbeddingSettings,
    _ollama_host: str,
    **_kwargs: Any,
) -> ProviderCheckResult:
    """Check Azure OpenAI-specific dependencies.

    Checks:
    1. langchain-openai package is installed (Azure uses same package)
    2. AZURE_OPENAI_API_KEY is set
    3. AZURE_OPENAI_ENDPOINT is set
    4. AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME is set

    Args:
        embedding_settings: EmbeddingSettings with Azure configuration
        _ollama_host: Ollama server URL (unused, accepted for uniform dispatch interface)
        **kwargs: Additional keyword arguments (unused, accepted for uniform dispatch interface)

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
        logger.debug('langchain-openai package available')
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
    logger.debug('All Azure configuration variables are set')

    logger.info('All Azure OpenAI provider dependencies available')
    return ProviderCheckResult(available=True, reason=None, install_instructions=None)


async def _check_huggingface_dependencies(
    embedding_settings: EmbeddingSettings,
    _ollama_host: str,
    **_kwargs: Any,
) -> ProviderCheckResult:
    """Check HuggingFace-specific dependencies.

    Checks:
    1. langchain-huggingface package is installed
    2. HUGGINGFACEHUB_API_TOKEN is set

    Args:
        embedding_settings: EmbeddingSettings with huggingface_api_key
        _ollama_host: Ollama server URL (unused, accepted for uniform dispatch interface)
        **kwargs: Additional keyword arguments (unused, accepted for uniform dispatch interface)

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
        logger.debug('langchain-huggingface package available')
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
    logger.debug('HUGGINGFACEHUB_API_TOKEN is set')

    logger.info('All HuggingFace provider dependencies available')
    return ProviderCheckResult(available=True, reason=None, install_instructions=None)


async def _check_voyage_dependencies(
    embedding_settings: EmbeddingSettings,
    _ollama_host: str,
    **_kwargs: Any,
) -> ProviderCheckResult:
    """Check Voyage AI-specific dependencies.

    Checks:
    1. langchain-voyageai package is installed
    2. VOYAGE_API_KEY is set

    Args:
        embedding_settings: EmbeddingSettings with voyage_api_key
        _ollama_host: Ollama server URL (unused, accepted for uniform dispatch interface)
        **kwargs: Additional keyword arguments (unused, accepted for uniform dispatch interface)

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
        logger.debug('langchain-voyageai package available')
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
    logger.debug('VOYAGE_API_KEY is set')

    logger.info('All Voyage AI provider dependencies available')
    return ProviderCheckResult(available=True, reason=None, install_instructions=None)


async def check_summary_provider_dependencies(
    provider: str,
    summary_settings: 'SummarySettings',
    ollama_host: str = 'http://localhost:11434',
    *,
    auto_pull: bool = True,
    pull_timeout: int = 900,
) -> ProviderCheckResult:
    """Check provider-specific dependencies for summary generation.

    Dispatches uniformly to the provider-specific check function.
    Each provider has different requirements:
    - ollama: Requires langchain-ollama, Ollama service, summary model available
    - openai: Requires langchain-openai, OPENAI_API_KEY
    - anthropic: Requires langchain-anthropic, ANTHROPIC_API_KEY

    Args:
        provider: Provider name from SUMMARY_PROVIDER setting
        summary_settings: SummarySettings instance with provider configuration
        ollama_host: Ollama server URL (from OllamaSettings.host)
        auto_pull: Whether to auto-pull missing Ollama models
        pull_timeout: Timeout in seconds for Ollama model pull operations

    Returns:
        ProviderCheckResult with available, reason, and install_instructions
    """
    check_functions: dict[str, Callable[..., Any]] = {
        'ollama': _check_ollama_summary_dependencies,
        'openai': _check_openai_summary_dependencies,
        'anthropic': _check_anthropic_summary_dependencies,
    }

    if provider not in check_functions:
        return ProviderCheckResult(
            available=False,
            reason=f"Unknown summary provider: '{provider}'",
            install_instructions=None,
        )

    logger.info(f'Checking {provider} summary provider dependencies...')
    result = await check_functions[provider](
        summary_settings, ollama_host, auto_pull=auto_pull, pull_timeout=pull_timeout,
    )
    return cast(ProviderCheckResult, result)


async def _check_ollama_summary_dependencies(
    summary_settings: 'SummarySettings',
    ollama_host: str,
    *,
    auto_pull: bool = True,
    pull_timeout: int = 900,
) -> ProviderCheckResult:
    """Check Ollama-specific dependencies for summary generation.

    Args:
        summary_settings: SummarySettings with model name
        ollama_host: Ollama server URL (from OllamaSettings.host)
        auto_pull: Whether to auto-pull missing models
        pull_timeout: Timeout in seconds for model pull operations

    Returns:
        ProviderCheckResult
    """
    return await _check_ollama_model(
        model_name=summary_settings.model,
        ollama_host=ollama_host,
        auto_pull=auto_pull,
        pull_timeout=pull_timeout,
        package_name='langchain_ollama',
        install_cmd='uv sync --extra summary-ollama',
        feature_label='Summary',
    )


async def _check_openai_summary_dependencies(
    summary_settings: 'SummarySettings',
    _ollama_host: str,
    **_kwargs: Any,
) -> ProviderCheckResult:
    """Check OpenAI-specific dependencies for summary generation.

    Checks:
    1. langchain-openai package is installed
    2. OPENAI_API_KEY is configured in settings

    Args:
        summary_settings: SummarySettings instance with openai_api_key
        _ollama_host: Ollama server URL (unused, accepted for uniform dispatch interface)
        **_kwargs: Additional keyword arguments (unused, accepted for uniform dispatch interface)

    Returns:
        ProviderCheckResult
    """
    install_cmd = 'uv sync --extra summary-openai'

    # 1. Check langchain-openai package
    try:
        if importlib.util.find_spec('langchain_openai') is None:
            return ProviderCheckResult(
                available=False,
                reason='langchain-openai package not installed',
                install_instructions=install_cmd,
            )
        logger.debug('langchain-openai package available')
    except ImportError as e:
        return ProviderCheckResult(
            available=False,
            reason=f'langchain-openai package not available: {e}',
            install_instructions=install_cmd,
        )

    # 2. Check API key is configured
    if summary_settings.openai_api_key is None:
        return ProviderCheckResult(
            available=False,
            reason='OPENAI_API_KEY environment variable is not set',
            install_instructions='Set environment variable: export OPENAI_API_KEY=your-key',
        )
    logger.debug('OPENAI_API_KEY is set')

    logger.info('All OpenAI summary provider dependencies available')
    return ProviderCheckResult(available=True, reason=None, install_instructions=None)


async def _check_anthropic_summary_dependencies(
    summary_settings: 'SummarySettings',
    _ollama_host: str,
    **_kwargs: Any,
) -> ProviderCheckResult:
    """Check Anthropic-specific dependencies for summary generation.

    Checks:
    1. langchain-anthropic package is installed
    2. ANTHROPIC_API_KEY is configured in settings

    Args:
        summary_settings: SummarySettings instance with anthropic_api_key
        _ollama_host: Ollama server URL (unused, accepted for uniform dispatch interface)
        **_kwargs: Additional keyword arguments (unused, accepted for uniform dispatch interface)

    Returns:
        ProviderCheckResult
    """
    install_cmd = 'uv sync --extra summary-anthropic'

    # 1. Check langchain-anthropic package
    try:
        if importlib.util.find_spec('langchain_anthropic') is None:
            return ProviderCheckResult(
                available=False,
                reason='langchain-anthropic package not installed',
                install_instructions=install_cmd,
            )
        logger.debug('langchain-anthropic package available')
    except ImportError as e:
        return ProviderCheckResult(
            available=False,
            reason=f'langchain-anthropic package not available: {e}',
            install_instructions=install_cmd,
        )

    # 2. Check API key is configured
    if summary_settings.anthropic_api_key is None:
        return ProviderCheckResult(
            available=False,
            reason='ANTHROPIC_API_KEY environment variable is not set',
            install_instructions='Set environment variable: export ANTHROPIC_API_KEY=your-key',
        )
    logger.debug('ANTHROPIC_API_KEY is set')

    logger.info('All Anthropic summary provider dependencies available')
    return ProviderCheckResult(available=True, reason=None, install_instructions=None)
