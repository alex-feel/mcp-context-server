"""Tests for tool registration and DISABLED_TOOLS configuration.

This module tests the tool registration machinery in app/server.py,
including DISABLED_TOOLS environment variable handling and tool annotations.

P1 Priority: Tool registration is core server functionality with no tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Conditional skip marker for tests requiring fastmcp
requires_fastmcp = pytest.mark.skipif(
    importlib.util.find_spec('fastmcp') is None,
    reason='fastmcp package not installed',
)

# Conditional skip marker for tests requiring sqlite_vec
requires_sqlite_vec = pytest.mark.skipif(
    importlib.util.find_spec('sqlite_vec') is None,
    reason='sqlite-vec package not installed',
)


class TestIsToolDisabled:
    """Tests for is_tool_disabled().

    Note: is_tool_disabled uses settings.tools.disabled which is a property
    computed from the environment at settings load time. We mock the settings
    object directly to test the function behavior.
    """

    def test_returns_false_when_not_in_list(self) -> None:
        """Test tool not in DISABLED_TOOLS returns False."""
        mock_settings = MagicMock()
        mock_settings.tools.disabled = {'delete_context', 'update_context'}

        with patch('app.tools.settings', mock_settings):
            from app.tools import is_tool_disabled

            result = is_tool_disabled('store_context')
            assert result is False

    def test_returns_true_when_in_list(self) -> None:
        """Test tool in DISABLED_TOOLS returns True."""
        mock_settings = MagicMock()
        mock_settings.tools.disabled = {'delete_context', 'update_context'}

        with patch('app.tools.settings', mock_settings):
            from app.tools import is_tool_disabled

            result = is_tool_disabled('delete_context')
            assert result is True

    def test_returns_false_when_list_empty(self) -> None:
        """Test returns False when DISABLED_TOOLS is empty."""
        mock_settings = MagicMock()
        mock_settings.tools.disabled = set()

        with patch('app.tools.settings', mock_settings):
            from app.tools import is_tool_disabled

            result = is_tool_disabled('store_context')
            assert result is False

    def test_case_insensitive_matching(self) -> None:
        """Test tool matching is case-insensitive."""
        mock_settings = MagicMock()
        # Settings stores lowercase, function converts to lowercase
        mock_settings.tools.disabled = {'delete_context'}

        with patch('app.tools.settings', mock_settings):
            from app.tools import is_tool_disabled

            # Test various case combinations
            assert is_tool_disabled('Delete_Context') is True
            assert is_tool_disabled('DELETE_CONTEXT') is True
            assert is_tool_disabled('delete_context') is True


class TestRegisterTool:
    """Tests for register_tool().

    Note: register_tool now takes mcp_instance as first parameter.
    We pass a mock MCP instance and patch settings to test the function behavior.
    """

    def test_registers_when_not_disabled(self) -> None:
        """Test tool registered when not disabled."""
        mock_mcp = MagicMock()
        mock_settings = MagicMock()
        mock_settings.tools.disabled = set()

        async def dummy_tool() -> dict[str, Any]:
            return {'success': True}

        with patch('app.tools.settings', mock_settings):
            from app.tools import register_tool

            result = register_tool(mock_mcp, dummy_tool)

            assert result is True
            mock_mcp.tool.assert_called_once()

    def test_skips_when_disabled(self) -> None:
        """Test tool not registered when disabled."""
        mock_mcp = MagicMock()
        mock_settings = MagicMock()
        mock_settings.tools.disabled = {'dummy_tool'}

        async def dummy_tool() -> dict[str, Any]:
            return {'success': True}

        with patch('app.tools.settings', mock_settings):
            from app.tools import register_tool

            result = register_tool(mock_mcp, dummy_tool)

            assert result is False
            mock_mcp.tool.assert_not_called()

    def test_uses_function_name_as_default(self) -> None:
        """Test tool name defaults to function name."""
        mock_mcp = MagicMock()
        mock_settings = MagicMock()
        mock_settings.tools.disabled = set()

        async def my_custom_tool() -> dict[str, Any]:
            return {'success': True}

        with patch('app.tools.settings', mock_settings):
            from app.tools import register_tool

            register_tool(mock_mcp, my_custom_tool)

            # Check that the decorator was called
            mock_mcp.tool.assert_called_once()

    def test_uses_explicit_name_when_provided(self) -> None:
        """Test explicit name overrides function name for disabled check."""
        mock_mcp = MagicMock()
        mock_settings = MagicMock()
        mock_settings.tools.disabled = {'explicit_name'}  # Disable by explicit name

        async def my_func() -> dict[str, Any]:
            return {'success': True}

        with patch('app.tools.settings', mock_settings):
            from app.tools import register_tool

            # Should be disabled because we pass name='explicit_name'
            result = register_tool(mock_mcp, my_func, name='explicit_name')

            assert result is False

    def test_applies_annotations_from_tool_annotations(self) -> None:
        """Test annotations are fetched from TOOL_ANNOTATIONS."""
        mock_mcp = MagicMock()
        mock_decorator = MagicMock(return_value=lambda f: f)
        mock_mcp.tool.return_value = mock_decorator
        mock_settings = MagicMock()
        mock_settings.tools.disabled = set()

        async def store_context() -> dict[str, Any]:
            return {'success': True}

        with patch('app.tools.settings', mock_settings):
            from app.tools import TOOL_ANNOTATIONS
            from app.tools import register_tool

            register_tool(mock_mcp, store_context)

            # Verify tool decorator was called with annotations from TOOL_ANNOTATIONS
            call_kwargs = mock_mcp.tool.call_args[1]
            expected_annotations = TOOL_ANNOTATIONS.get('store_context', {})
            assert call_kwargs.get('annotations') == expected_annotations

    def test_returns_true_when_registered(self) -> None:
        """Test return value indicates registration."""
        mock_mcp = MagicMock()
        mock_settings = MagicMock()
        mock_settings.tools.disabled = set()

        async def test_tool() -> dict[str, Any]:
            return {'success': True}

        with patch('app.tools.settings', mock_settings):
            from app.tools import register_tool

            result = register_tool(mock_mcp, test_tool)

            assert result is True

    def test_returns_false_when_disabled(self) -> None:
        """Test return value indicates skip."""
        mock_mcp = MagicMock()
        mock_settings = MagicMock()
        mock_settings.tools.disabled = {'test_tool'}

        async def test_tool() -> dict[str, Any]:
            return {'success': True}

        with patch('app.tools.settings', mock_settings):
            from app.tools import register_tool

            result = register_tool(mock_mcp, test_tool)

            assert result is False


class TestDisabledToolsConfiguration:
    """Integration tests for DISABLED_TOOLS configuration."""

    def test_multiple_tools_can_be_disabled(self) -> None:
        """Test multiple tools can be disabled via settings."""
        mock_settings = MagicMock()
        mock_settings.tools.disabled = {
            'delete_context',
            'update_context',
            'delete_context_batch',
        }

        with patch('app.tools.settings', mock_settings):
            from app.tools import is_tool_disabled

            assert is_tool_disabled('delete_context') is True
            assert is_tool_disabled('update_context') is True
            assert is_tool_disabled('delete_context_batch') is True
            assert is_tool_disabled('store_context') is False

    def test_empty_disabled_tools_enables_all(self) -> None:
        """Test all tools enabled when DISABLED_TOOLS empty."""
        mock_settings = MagicMock()
        mock_settings.tools.disabled = set()

        with patch('app.tools.settings', mock_settings):
            from app.tools import is_tool_disabled

            # All standard tools should be enabled
            tools = [
                'store_context',
                'search_context',
                'get_context_by_ids',
                'delete_context',
                'update_context',
                'list_threads',
                'get_statistics',
            ]

            for tool in tools:
                assert is_tool_disabled(tool) is False

    @pytest.mark.integration
    @pytest.mark.asyncio
    @requires_fastmcp
    @requires_sqlite_vec
    async def test_disabled_tool_not_available_via_mcp(self, tmp_path: Path) -> None:
        """Test that a disabled tool is NOT exposed to MCP clients.

        This test verifies that when DISABLED_TOOLS is set:
        1. The disabled tool is not listed in available tools
        2. Calling the disabled tool returns an appropriate error

        Uses FastMCP's subprocess mode via run_server.py wrapper for process isolation.
        This prevents module state corruption that affects other tests.
        """
        import sqlite3

        from fastmcp import Client
        from fastmcp.client.transports import PythonStdioTransport
        from fastmcp.exceptions import ToolError

        from app.schemas import load_schema

        # Create temporary database
        temp_db = tmp_path / 'test_disabled_tools.db'
        temp_db.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database with schema
        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(temp_db)) as conn:
            conn.executescript(schema_sql)
            conn.execute('PRAGMA foreign_keys = ON')
            conn.execute('PRAGMA journal_mode = WAL')
            conn.commit()

        # Use subprocess mode via wrapper script for complete process isolation
        # This prevents module state corruption that breaks other tests
        wrapper_script = Path(__file__).parent / 'run_server.py'

        # Explicitly pass environment to subprocess - don't rely on inheritance
        subprocess_env = {
            'DB_PATH': str(temp_db),
            'MCP_TEST_MODE': '1',
            'DISABLED_TOOLS': 'delete_context',  # Disable delete_context tool
            'ENABLE_SEMANTIC_SEARCH': 'false',  # Disable for speed
            'ENABLE_FTS': 'false',  # Disable for speed
            'ENABLE_HYBRID_SEARCH': 'false',  # Disable for speed
        }

        # Create transport with explicit environment
        transport = PythonStdioTransport(
            script_path=str(wrapper_script),
            env=subprocess_env,
        )

        # Create client with custom transport
        client = Client(transport)

        async with client:
            # Get list of available tools
            tools_result = await client.list_tools()
            tool_names = [tool.name for tool in tools_result]

            # Verify delete_context is NOT in the list of available tools
            assert 'delete_context' not in tool_names, (
                f'delete_context should NOT be in available tools when disabled. '
                f'Available tools: {tool_names}'
            )

            # Verify other tools are still available
            assert 'store_context' in tool_names, 'store_context should be available'
            assert 'search_context' in tool_names, 'search_context should be available'

            # Try to call the disabled tool - should fail with an error
            # We expect ToolError or similar exception when calling a disabled/unknown tool
            call_succeeded = False
            error_message = ''
            try:
                await client.call_tool('delete_context', {'thread_id': 'test'})
                call_succeeded = True
            except ToolError as tool_err:
                # This is expected - disabled tool should not be callable
                error_message = str(tool_err).lower()
            except Exception as exc:
                # Other exceptions may indicate the tool is disabled
                error_message = str(exc).lower()

            # Verify the call did not succeed
            if call_succeeded:
                pytest.fail('Calling disabled tool delete_context should raise an error')

            # Verify error message indicates the tool is unknown/disabled
            valid_error = (
                'delete_context' in error_message
                or 'unknown' in error_message
                or 'not found' in error_message
            )
            if not valid_error:
                pytest.fail(f'Error should mention tool name or indicate unknown tool: {error_message}')


class TestToolAnnotations:
    """Tests for TOOL_ANNOTATIONS dictionary."""

    def test_all_13_tools_have_annotations(self) -> None:
        """Verify all MCP tools have annotation entries."""
        from app.tools import TOOL_ANNOTATIONS

        expected_tools = [
            'store_context',
            'store_context_batch',
            'search_context',
            'get_context_by_ids',
            'list_threads',
            'get_statistics',
            'semantic_search_context',
            'fts_search_context',
            'hybrid_search_context',
            'update_context',
            'update_context_batch',
            'delete_context',
            'delete_context_batch',
        ]

        for tool in expected_tools:
            assert tool in TOOL_ANNOTATIONS, f'Missing annotation for tool: {tool}'

    def test_read_only_tools_have_readonly_hint(self) -> None:
        """Verify search tools have readOnlyHint=True."""
        from app.tools import TOOL_ANNOTATIONS

        read_only_tools = [
            'search_context',
            'get_context_by_ids',
            'list_threads',
            'get_statistics',
            'semantic_search_context',
            'fts_search_context',
            'hybrid_search_context',
        ]

        for tool in read_only_tools:
            assert TOOL_ANNOTATIONS[tool].get('readOnlyHint') is True, (
                f'Tool {tool} should have readOnlyHint=True'
            )

    def test_destructive_tools_have_destructive_hint(self) -> None:
        """Verify delete/update tools have destructiveHint=True."""
        from app.tools import TOOL_ANNOTATIONS

        destructive_tools = [
            'update_context',
            'update_context_batch',
            'delete_context',
            'delete_context_batch',
        ]

        for tool in destructive_tools:
            assert TOOL_ANNOTATIONS[tool].get('destructiveHint') is True, (
                f'Tool {tool} should have destructiveHint=True'
            )

    def test_additive_tools_not_destructive(self) -> None:
        """Verify store tools have destructiveHint=False."""
        from app.tools import TOOL_ANNOTATIONS

        additive_tools = [
            'store_context',
            'store_context_batch',
        ]

        for tool in additive_tools:
            assert TOOL_ANNOTATIONS[tool].get('destructiveHint') is False, (
                f'Tool {tool} should have destructiveHint=False'
            )

    def test_delete_tools_are_idempotent(self) -> None:
        """Verify delete tools have idempotentHint=True."""
        from app.tools import TOOL_ANNOTATIONS

        delete_tools = [
            'delete_context',
            'delete_context_batch',
        ]

        for tool in delete_tools:
            assert TOOL_ANNOTATIONS[tool].get('idempotentHint') is True, (
                f'Tool {tool} should have idempotentHint=True'
            )

    def test_update_tools_not_idempotent(self) -> None:
        """Verify update tools have idempotentHint=False."""
        from app.tools import TOOL_ANNOTATIONS

        update_tools = [
            'update_context',
            'update_context_batch',
        ]

        for tool in update_tools:
            assert TOOL_ANNOTATIONS[tool].get('idempotentHint') is False, (
                f'Tool {tool} should have idempotentHint=False'
            )

    def test_all_tools_have_title(self) -> None:
        """Verify all tools have a title for display."""
        from app.tools import TOOL_ANNOTATIONS

        for tool, tool_annots in TOOL_ANNOTATIONS.items():
            assert 'title' in tool_annots, f'Tool {tool} missing title'
            assert isinstance(tool_annots['title'], str), f'Tool {tool} title should be string'
            assert len(tool_annots['title']) > 0, f'Tool {tool} title should not be empty'
