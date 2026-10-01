"""Tool calls through a client session raise no MCP SDK deprecation warning.

The MCP SDK deprecates server-to-client log notifications: every
``ServerSession.send_log_message`` call emits an ``MCPDeprecationWarning``. The
server sends none, so a tool call completes without that warning. The check drives
``list_threads`` and ``store_context`` through an in-memory ``fastmcp.Client``
against a ``FastMCP`` instance running the server's own lifespan, with every
external service disabled so the lifespan completes locally.
"""

import sys
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp import FastMCP
from mcp import MCPDeprecationWarning

import app.server as server_module
from app.settings import AppSettings
from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset the settings singleton before and after each test.

    Yields:
        Control to the test body.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _rebind_module_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every ``app`` module's import-time ``settings`` binding at the fresh singleton.

    Modules bind ``settings = get_settings()`` when first imported, so an environment
    flip made inside a test reaches them only through an explicit rebinding.

    Args:
        monkeypatch: The pytest monkeypatch fixture; each rebinding is undone on teardown.
    """
    fresh_settings = get_settings()
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith('app.') and isinstance(getattr(module, 'settings', None), AppSettings):
            monkeypatch.setattr(module, 'settings', fresh_settings)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FastMCP[None]:
    """Build a FastMCP instance wired to the server lifespan over an isolated SQLite database.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Per-test temporary directory holding the database.

    Returns:
        A FastMCP instance whose lifespan registers the tools when a client connects.
    """
    db_path = tmp_path / 'tool_call_warnings.db'
    monkeypatch.setenv('DB_PATH', str(db_path))
    monkeypatch.setenv('STORAGE_BACKEND', 'sqlite')
    monkeypatch.setenv('DISABLED_TOOLS', '')
    for toggle in (
        'ENABLE_EMBEDDING_GENERATION',
        'ENABLE_SEMANTIC_SEARCH',
        'ENABLE_FTS',
        'ENABLE_HYBRID_SEARCH',
        'ENABLE_CHUNKING',
        'ENABLE_RERANKING',
        'ENABLE_SUMMARY_GENERATION',
        'ENABLE_INDEX_TREE_NODE_SUMMARIES',
        'ENABLE_EMBEDDING_COMPRESSION',
    ):
        monkeypatch.setenv(toggle, 'false')
    get_settings.cache_clear()
    _rebind_module_settings(monkeypatch)
    monkeypatch.setattr(server_module, 'DB_PATH', db_path)

    return FastMCP(
        name='tool-call-warnings',
        lifespan=server_module.lifespan,
        mask_error_details=False,
        strict_input_validation=False,
    )


@pytest.mark.asyncio
async def test_tool_calls_emit_no_mcp_deprecation_warning(server: FastMCP[None]) -> None:
    """``list_threads`` and ``store_context`` complete without an ``MCPDeprecationWarning``."""
    store_arguments: dict[str, Any] = {'thread_id': 'warning-check', 'source': 'user', 'text': 'a stored entry'}

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        async with Client(server) as client:
            listed = await client.call_tool('list_threads', {})
            stored = await client.call_tool('store_context', store_arguments)

    assert listed.is_error is False
    assert stored.is_error is False
    deprecations = [str(warning.message) for warning in recorded if issubclass(warning.category, MCPDeprecationWarning)]
    assert deprecations == []
