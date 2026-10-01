"""
MCP tool implementations for mcp-context-server.

This package contains all MCP tool functions, organized by domain:
- context.py: store_context, get_context_by_ids, update_context, delete_context
- search/: search_context (browse.py), semantic_search_context (semantic.py), fts_search_context
  (fts.py), hybrid_search_context (hybrid.py), the raw semantic and FTS legs they share (legs.py),
  reranking and result display formatting (ranking.py), and argument bounds and filter caps (limits.py)
- discovery.py: list_threads, get_statistics
- batch/: store_context_batch (store.py), update_context_batch (update.py), delete_context_batch
  (delete.py), and the per-entry validation they share (entry_validation.py)
- descriptions.py: Backend-specific dynamic tool descriptions (generate_fts_description)
- _validation.py: Input validation shared by the tools (unstorable text, tag and indexed-value
  limits, image validation and normalization) (not re-exported)
- _generation.py: Embedding, compression, summary, and index_tree node generation with timeout,
  run outside the transaction (not re-exported)
- _transactions.py: Store and update transaction execution, transaction heartbeat, and
  connection error classification (not re-exported)
- _delete_cleanup.py: Entry deletion with the explicit embedding cleanup (not re-exported)
- _responses.py: Store and update response message builders (not re-exported)

The tool registration helpers and TOOL_ANNOTATIONS are defined here for use by server.py.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any

from app.settings import get_settings

# Re-export all tool functions for backward compatibility
from app.tools.batch.delete import delete_context_batch
from app.tools.batch.store import store_context_batch
from app.tools.batch.update import update_context_batch
from app.tools.context import delete_context
from app.tools.context import get_context_by_ids
from app.tools.context import store_context
from app.tools.context import update_context
from app.tools.descriptions import generate_fts_description
from app.tools.discovery import get_statistics
from app.tools.discovery import list_threads
from app.tools.navigation import grep_context
from app.tools.navigation import navigate_context
from app.tools.navigation import read_context_range
from app.tools.search.browse import search_context
from app.tools.search.fts import fts_search_context
from app.tools.search.hybrid import hybrid_search_context
from app.tools.search.semantic import semantic_search_context

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)
settings = get_settings()

# Tool annotations with human-readable titles for MCP protocol hints
# Each tool has a title for display and behavior hints (readOnly, destructive, idempotent)
TOOL_ANNOTATIONS: dict[str, dict[str, Any]] = {
    # Additive tools (create new entries)
    'store_context': {
        'title': 'Store Context',
        'readOnlyHint': False,
        'destructiveHint': False,
    },
    'store_context_batch': {
        'title': 'Store Context (Batch)',
        'readOnlyHint': False,
        'destructiveHint': False,
    },
    # Read-only tools (no modifications)
    'search_context': {
        'title': 'Search Context',
        'readOnlyHint': True,
    },
    'get_context_by_ids': {
        'title': 'Get Context by IDs',
        'readOnlyHint': True,
    },
    'list_threads': {
        'title': 'List Threads',
        'readOnlyHint': True,
    },
    'get_statistics': {
        'title': 'Get Statistics',
        'readOnlyHint': True,
    },
    'semantic_search_context': {
        'title': 'Semantic Search Context',
        'readOnlyHint': True,
    },
    'fts_search_context': {
        'title': 'Full-Text Search Context',
        'readOnlyHint': True,
    },
    'hybrid_search_context': {
        'title': 'Hybrid Search Context',
        'readOnlyHint': True,
    },
    'grep_context': {
        'title': 'Grep Context',
        'readOnlyHint': True,
    },
    'navigate_context': {
        'title': 'Navigate Context',
        'readOnlyHint': True,
    },
    'read_context_range': {
        'title': 'Read Context Range',
        'readOnlyHint': True,
    },
    # Update tools (destructive, not idempotent)
    'update_context': {
        'title': 'Update Context',
        'readOnlyHint': False,
        'destructiveHint': True,
        'idempotentHint': False,
    },
    'update_context_batch': {
        'title': 'Update Context (Batch)',
        'readOnlyHint': False,
        'destructiveHint': True,
        'idempotentHint': False,
    },
    # Delete tools (destructive, idempotent)
    'delete_context': {
        'title': 'Delete Context',
        'readOnlyHint': False,
        'destructiveHint': True,
        'idempotentHint': True,
    },
    'delete_context_batch': {
        'title': 'Delete Context (Batch)',
        'readOnlyHint': False,
        'destructiveHint': True,
        'idempotentHint': True,
    },
}


def is_tool_disabled(tool_name: str) -> bool:
    """Check if a tool is in the disabled list.

    Args:
        tool_name: The name of the tool to check

    Returns:
        True if tool is disabled, False otherwise
    """
    return tool_name.lower() in settings.tools.disabled


def register_tool(
    mcp_instance: 'FastMCP[None]',
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
) -> bool:
    """Register a tool only if it's not in the disabled list.

    Args:
        mcp_instance: The FastMCP server instance to register the tool with
        func: The tool function to register
        name: Optional explicit tool name (defaults to function name)
        description: Optional custom description (overrides function docstring)

    Returns:
        True if tool was registered, False if disabled
    """
    func_name: str = getattr(func, '__name__', '<callable>')
    tool_name: str = name or func_name

    if is_tool_disabled(tool_name):
        logger.info(f'{tool_name} not registered (in DISABLED_TOOLS)')
        return False

    # Get annotations from centralized mapping
    annotations = TOOL_ANNOTATIONS.get(tool_name, {})

    # Pass description if provided (overrides docstring in FastMCP).
    # Forward name=tool_name so an explicit name argument actually renames the
    # registration (FastMCP otherwise derives the name from func.__name__); this
    # keeps the DISABLED_TOOLS check, the TOOL_ANNOTATIONS lookup, and the served
    # tool name in agreement for a caller that passes name != func.__name__.
    if description:
        mcp_instance.tool(name=tool_name, description=description, annotations=annotations)(func)
    else:
        mcp_instance.tool(name=tool_name, annotations=annotations)(func)

    logger.info(f'{tool_name} registered')
    return True


# Public API exports
__all__ = [
    # Tool registration infrastructure
    'TOOL_ANNOTATIONS',
    'generate_fts_description',
    'is_tool_disabled',
    'register_tool',
    # Context CRUD tools
    'delete_context',
    'get_context_by_ids',
    'store_context',
    'update_context',
    # Search tools
    'fts_search_context',
    'hybrid_search_context',
    'search_context',
    'semantic_search_context',
    # Navigation tools (locate / navigate / extract)
    'grep_context',
    'navigate_context',
    'read_context_range',
    # Discovery tools
    'get_statistics',
    'list_threads',
    # Batch tools
    'delete_context_batch',
    'store_context_batch',
    'update_context_batch',
]
