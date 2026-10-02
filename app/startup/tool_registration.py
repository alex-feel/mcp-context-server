"""Tool registration phases of the server lifespan.

Registers the MCP tools on the FastMCP instance in two phases, the core tools before
provider initialization and the prerequisite-gated search tools after it, then installs
the JSON string deserializer middleware over the complete tool set.
"""

import logging
from typing import Literal
from typing import cast

from fastmcp import FastMCP

from app.middleware import JsonStringDeserializerMiddleware
from app.middleware import build_schema_map
from app.repositories import RepositoryContainer
from app.settings import AppSettings
from app.startup import get_embedding_provider
from app.tools import delete_context
from app.tools import delete_context_batch
from app.tools import fts_search_context
from app.tools import generate_fts_description
from app.tools import get_context_by_ids
from app.tools import get_statistics
from app.tools import grep_context
from app.tools import hybrid_search_context
from app.tools import list_threads
from app.tools import navigate_context
from app.tools import read_context_range
from app.tools import register_tool
from app.tools import search_context
from app.tools import semantic_search_context
from app.tools import store_context
from app.tools import store_context_batch
from app.tools import update_context
from app.tools import update_context_batch

logger = logging.getLogger(__name__)


def register_core_tools(mcp: FastMCP[None], settings: AppSettings) -> None:
    """Register the tools that need no provider: CRUD, browsing, discovery, and navigation.

    Args:
        mcp: The FastMCP server instance to register the tools on.
        settings: The application settings the server lifespan runs with.
    """
    # Register core tools (annotations from TOOL_ANNOTATIONS in app.tools)
    # Additive tools (create new entries)
    register_tool(mcp, store_context)
    register_tool(mcp, store_context_batch)

    # Read-only tools (no modifications)
    register_tool(mcp, search_context)
    register_tool(mcp, get_context_by_ids)
    register_tool(mcp, list_threads)
    register_tool(mcp, get_statistics)

    # Read-only navigation tools (locate / extract). Matching and slicing are
    # pure-Python with no external prerequisites, so registration gates only on
    # the tri-state toggle (auto/true register; false force-disables).
    if settings.grep_context.enabled:
        register_tool(mcp, grep_context)
    else:
        logger.info('grep_context not registered (ENABLE_GREP_CONTEXT=false)')
    if settings.context_navigation.enabled:
        register_tool(mcp, navigate_context)
    else:
        logger.info('navigate_context not registered (ENABLE_CONTEXT_NAVIGATION=false)')
    if settings.context_range.enabled:
        register_tool(mcp, read_context_range)
    else:
        logger.info('read_context_range not registered (ENABLE_CONTEXT_RANGE=false)')

    # Update tools (destructive, not idempotent)
    register_tool(mcp, update_context)
    register_tool(mcp, update_context_batch)

    # Delete tools (destructive, idempotent)
    register_tool(mcp, delete_context)
    register_tool(mcp, delete_context_batch)


async def register_search_tools(
    mcp: FastMCP[None],
    settings: AppSettings,
    backend_type: str,
    repos: RepositoryContainer,
) -> None:
    """Register the semantic, full-text, and hybrid search tools whose prerequisites are present.

    Call after ``app.startup.providers.initialize_providers``: the semantic and hybrid
    registrations read the embedding provider it sets.

    Args:
        mcp: The FastMCP server instance to register the tools on.
        settings: The application settings the server lifespan runs with.
        backend_type: The storage backend type, which selects the full-text search description.
        repos: The repositories, used to report whether the full-text search index is provisioned.
    """
    # Register semantic search tool based on its mode and embedding availability.
    # auto: register when an embedding provider is present (initialized by
    # initialize_providers, so its presence is the authoritative "embeddings are
    # available" signal).
    # true: register when a provider is available; warn and skip when none is (it does
    # not force the tool on without a provider). false: force off.
    semantic_mode = settings.semantic_search.mode
    if semantic_mode == 'false':
        logger.info('Semantic search disabled (ENABLE_SEMANTIC_SEARCH=false)')
        logger.info('semantic_search_context not registered (feature disabled)')
    elif get_embedding_provider() is not None:
        # register_tool logs both outcomes itself ('<tool> registered' or
        # 'not registered (in DISABLED_TOOLS)'); an extra unconditional log
        # here would falsely assert registration when DISABLED_TOOLS skips it.
        register_tool(mcp, semantic_search_context)
    elif semantic_mode == 'true':
        logger.warning(
            'ENABLE_SEMANTIC_SEARCH=true but no embedding provider is available '
            '(ENABLE_EMBEDDING_GENERATION=false or provider initialization failed) - '
            'semantic_search_context NOT registered',
        )
    else:
        logger.info(
            'semantic_search_context not registered '
            '(ENABLE_SEMANTIC_SEARCH=auto and no embedding provider available)',
        )

    # Register FTS tool if enabled - ALWAYS register when ENABLE_FTS=true
    # The tool handles graceful degradation during migration
    if settings.fts.enabled:
        # Generate backend-specific FTS description for AI agents
        fts_description = generate_fts_description(
            cast(Literal['sqlite', 'postgresql'], backend_type),
            settings.fts.language,
        )

        # Always register the FTS tool when enabled (DISABLED_TOOLS takes priority)
        # The tool itself checks migration status and returns informative response
        register_tool(mcp, fts_search_context, description=fts_description)

        # Report whether the FTS index is already provisioned. The probe is purely
        # diagnostic -- the tool is registered above either way and checks migration
        # status itself on every call -- so a transient operational fault here (an
        # external VACUUM or backup holding the SQLite write lock past the read retry
        # budget at the moment the server boots) must not abort startup and take every
        # OTHER tool down with it. is_available() deliberately lets such faults
        # propagate rather than reporting them as "not migrated", so this call site
        # absorbs them and treats availability as unknown, mirroring the FTS migration
        # step's own handling.
        try:
            fts_available = await repos.fts.is_available()
        except Exception as fts_probe_error:
            logger.warning(f'Could not determine full-text search availability: {fts_probe_error}')
        else:
            if fts_available:
                logger.info(f'Full-text search enabled and available (backend: {backend_type})')
            else:
                logger.warning('FTS enabled but index may need initialization or migration')
    else:
        logger.info('Full-text search disabled (ENABLE_FTS=false)')
        logger.info('fts_search_context not registered (feature disabled)')

    # Register Hybrid Search tool if enabled AND at least one search mode is available
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
            logger.info(f'hybrid_search_context modes available: {modes_available}')
        else:
            logger.warning(
                'Hybrid search enabled but no search modes available - feature disabled. '
                'Enable ENABLE_FTS=true and/or ENABLE_SEMANTIC_SEARCH=true.',
            )
            logger.info('hybrid_search_context not registered (no search modes available)')
    else:
        logger.info('Hybrid search disabled (ENABLE_HYBRID_SEARCH=false)')
        logger.info('hybrid_search_context not registered (feature disabled)')


async def install_json_string_middleware(mcp: FastMCP[None]) -> None:
    """Install the JSON string deserializer middleware for the registered tools.

    Call after every tool is registered: the schema map covers only the tools registered
    at call time.

    Args:
        mcp: The FastMCP server instance whose tools the middleware protects.
    """
    # Register schema-aware JSON string deserializer middleware
    # Handles client serialization issues where list/dict params arrive as JSON strings
    # Must run AFTER all tool registrations so schema map includes all tools
    all_tools = await mcp.list_tools(run_middleware=False)
    schema_map = build_schema_map(all_tools)
    if schema_map:
        mcp.add_middleware(JsonStringDeserializerMiddleware(schema_map))
        logger.info(
            'JSON string deserializer middleware registered '
            '(protecting %d tools, %d parameters)',
            len(schema_map),
            sum(len(params) for params in schema_map.values()),
        )
    else:
        logger.info(
            'JSON string deserializer middleware not needed '
            '(no complex params found)',
        )
