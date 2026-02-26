"""Default server instructions for MCP Context Server.

Provides the DEFAULT_INSTRUCTIONS constant used when MCP_SERVER_INSTRUCTIONS
environment variable is not set. These instructions are sent to MCP clients
during initialization via the MCP protocol's instructions field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.settings import InstructionsSettings

# Default server instructions sent to MCP clients during initialization.
# Override at runtime via MCP_SERVER_INSTRUCTIONS environment variable.
DEFAULT_INSTRUCTIONS: str = ('''
    # MCP Context Server

    Persistent context storage for LLM agents. Store, search, and retrieve context entries across sessions.

    ## Tools
    
    | Tool                      | Purpose                                     |
    |---------------------------|---------------------------------------------|
    | `store_context`           | Store new entry                             |
    | `get_context_by_ids`      | Retrieve full entries by ID                 |
    | `search_context`          | Browse/filter entries (truncated 150 chars) |
    | `hybrid_search_context`   | Combined FTS + semantic (RRF)               |
    | `semantic_search_context` | Vector similarity search                    |
    | `fts_search_context`      | Full-text linguistic search                 |
    | `update_context`          | Update existing entry                       |
    | `delete_context`          | Delete entries by ID                        |
    | `list_threads`            | List all thread IDs                         |
    | `get_statistics`          | Database statistics                         |

    Batch: `store_context_batch`, `update_context_batch`, `delete_context_batch` (up to 100 entries each).
    
    ## Core Concepts
    
    - **thread_id**: Groups related entries (e.g., one per knowledge base, project, session, or task). Scope searches with thread_id when working within a specific thread; omit for cross-thread knowledge discovery.
    - **source**: "user" or "agent" — filters by entry creator.
    - **text**: Prefer storing context in pure Markdown format.
    - **metadata**: JSON object for structured data. Use metadata_filters for advanced operators (gt, lt, contains, exists, etc.). Include descriptive fields (agent_name, task_name, status, project, references) for discoverability. Link entries into a knowledge graph with references. Store cross-references when creating entries: `metadata: {"references": {"context_ids": [<id_1>, <id_2>, ...]}}`. Retrieve referenced entries with `get_context_by_ids` to follow the knowledge chain.
    - **tags**: Lowercase labels for categorization (OR logic).
    
    ## Best Practices
    
    **Searching:**
    - Use `search_context` to browse and list entries within a thread (e.g., fetch recent messages by source filter), then use `get_context_by_ids` to retrieve full content.
    - Use `hybrid_search_context` for knowledge discovery — finding past decisions, patterns, and related work, typically cross-thread (e.g., by project metadata filter or across all projects).
    - These tools complement each other: `search_context` for thread-scoped context loading, `hybrid_search_context` for cross-thread knowledge search.
    - Use `fts_search_context` for precise linguistic queries only (stemming, boolean operators).
    - Use `semantic_search_context` for meaning-based similarity.
    - Prefer `hybrid_search_context` over `fts_search_context` and `semantic_search_context` when available.
    
    **Storing:**
    - Always include descriptive metadata (agent_name, task_name, status, project, references) for discoverability.
    - Add tags for cross-cutting categorization.
    - Add `metadata.references.context_ids` to metadata linking to entries your work builds upon.
    - Use `metadata_patch` (not `metadata`) in `update_context` to preserve fields you do not change.
    
    **Retrieving:**
    - `search_context` returns truncated previews (150 chars) — you can query tens of entries without hurting the context window. Always call `get_context_by_ids` for relevant entries for full content.
    - Check `metadata.references.context_ids` in retrieved entries and follow them with `get_context_by_ids` for deeper context.
''')


def resolve_instructions(instructions_settings: InstructionsSettings) -> str:
    """Resolve the effective instructions text.

    If MCP_SERVER_INSTRUCTIONS env var is set (including empty string),
    uses that value. Otherwise, returns DEFAULT_INSTRUCTIONS.

    Args:
        instructions_settings: The InstructionsSettings instance.

    Returns:
        The resolved instructions text.
    """
    if instructions_settings.server_instructions is not None:
        return instructions_settings.server_instructions
    return DEFAULT_INSTRUCTIONS
