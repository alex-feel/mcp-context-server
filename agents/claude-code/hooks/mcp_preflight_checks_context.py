#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pyyaml"]
# ///
"""
MCP Pre-Flight Checks Context Hook for Claude Code.

This hook provides pre-flight check instructions to the model at session start,
informing it to verify MCP server availability after the first user message
using the MCPSearch tool.

The list of MCP servers to check is configurable via external YAML configuration.

Trigger: SessionStart with any source
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_config_loader() -> ModuleType:
    """Dynamically load hook_config_loader from the same directory."""
    loader_path = Path(__file__).parent / 'hook_config_loader.py'
    spec = importlib.util.spec_from_file_location('hook_config_loader', loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load hook_config_loader from {loader_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json_output() -> ModuleType:
    """Dynamically load hook_json_output from the same directory."""
    loader_path = Path(__file__).parent / 'hook_json_output.py'
    spec = importlib.util.spec_from_file_location('hook_json_output', loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load hook_json_output from {loader_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Default configuration - used when no config file provided
# Maintains backward compatibility with hardcoded values
DEFAULT_CONFIG: dict[str, Any] = {
    'enabled': True,
    'mcp_servers': [
        {'name': 'context-server', 'description': 'Context storage and retrieval'},
    ],
    'message': {
        'header': 'MCP PRE-FLIGHT CHECKS REQUIRED',
        'session_resume_warning': (
            'This message means the session was JUST STARTED or RESUMED. '
            'You MUST perform the checks below NOW. '
            'Any prior check results in conversation history are STALE and INVALID - '
            'do NOT rely on them. Perform fresh checks regardless of what prior messages show.'
        ),
        'instruction': (
            'After the user sends their first message, you MUST perform pre-flight checks '
            'to verify MCP server availability before proceeding with any work.'
        ),
        'server_criticality_warning': (
            'ALL servers listed below are CRITICALLY important and MUST be checked. '
            'You have NO authority to skip, deprioritize, or make your own importance assessment '
            'for any server. Every server in this list is mandatory - no exceptions.'
        ),
        'mcpsearch_modes_label': 'CRITICAL - How MCPSearch works (READ THIS FIRST):',
        'mcpsearch_modes': (
            'MCPSearch has TWO modes that work DIFFERENTLY: '
            '(1) select:<exact_tool_name> - Returns ONLY that exact tool. Wildcards like * are NOT supported. '
            '(2) Keyword search (no prefix) - Tokenizes query and finds tools matching the keywords. '
            'For server verification, use keyword search mode (no "select:" prefix).'
        ),
        'servers_label': 'MCP servers to verify:',
        'action_label': 'Required action for each server:',
        'action_template': (
            'Use MCPSearch tool with query "{server_name}" (keyword search, NO "select:" prefix) '
            'to find tools from the {server_name} MCP server.'
        ),
        'response_interpretation_label': 'How to interpret MCPSearch responses:',
        'response_success_template': (
            '"Found N tool(s)" or tool references returned = SUCCESS - The MCP server is available. '
            'You may proceed with tasks that depend on this server.'
        ),
        'response_failure_template': (
            '"No matching MCP tools found" = FAILURE - The MCP server is UNAVAILABLE. '
            'You MUST NOT proceed with tasks that depend on this server. Inform the user immediately.'
        ),
        'zero_tools_warning': (
            'CRITICAL: MCPSearch returning 0 tools means the server is UNAVAILABLE - there is NO other interpretation. '
            'If you get 0 results, either try different search queries until you get > 0 results, '
            'or conclude that the server is UNAVAILABLE. '
            '0 tools = UNAVAILABLE = you CANNOT proceed with your work.'
        ),
        'failure_instruction': (
            'If any MCP server check returns "No matching MCP tools found", the server is UNAVAILABLE. '
            'Inform the user immediately and do NOT proceed with tasks that depend on that server.'
        ),
    },
}


def build_context_message(config: dict[str, Any]) -> str:
    """
    Build the context message from config components.

    The message instructs the model to perform MCP server availability checks
    after the first user message.

    Args:
        config: Configuration dictionary with message and mcp_servers

    Returns:
        Complete context message string
    """
    msg_config = config.get('message', DEFAULT_CONFIG['message'])
    servers = config.get('mcp_servers', DEFAULT_CONFIG['mcp_servers'])

    # Build server list section with descriptions
    server_lines: list[str] = []
    for entry in servers:
        name = entry.get('name', '')
        description = entry.get('description', '')
        server_lines.append(f'- {name}: {description}')
    server_list = '\n'.join(server_lines)

    # Build action instructions
    action_template = msg_config.get('action_template', DEFAULT_CONFIG['message']['action_template'])
    action_lines: list[str] = []
    for entry in servers:
        name = entry.get('name', '')
        action_lines.append(action_template.format(server_name=name))
    action_instructions = '\n'.join(f'  {i + 1}. {line}' for i, line in enumerate(action_lines))

    # Build MCPSearch modes section
    mcpsearch_modes_label = msg_config.get(
        'mcpsearch_modes_label', DEFAULT_CONFIG['message'].get('mcpsearch_modes_label', ''),
    )
    mcpsearch_modes = msg_config.get(
        'mcpsearch_modes', DEFAULT_CONFIG['message'].get('mcpsearch_modes', ''),
    )

    # Build response interpretation section
    response_interpretation_label = msg_config.get(
        'response_interpretation_label', DEFAULT_CONFIG['message']['response_interpretation_label'],
    )
    response_success = msg_config.get('response_success_template', DEFAULT_CONFIG['message']['response_success_template'])
    response_failure = msg_config.get('response_failure_template', DEFAULT_CONFIG['message']['response_failure_template'])
    response_interpretation = f'  - {response_success}\n  - {response_failure}'

    # Construct full message
    message_parts = [
        msg_config.get('header', ''),
        '',
        msg_config.get('session_resume_warning', DEFAULT_CONFIG['message']['session_resume_warning']),
        '',
        msg_config.get('instruction', ''),
        '',
        msg_config.get('server_criticality_warning', DEFAULT_CONFIG['message']['server_criticality_warning']),
        '',
        mcpsearch_modes_label,
        mcpsearch_modes,
        '',
        msg_config.get('servers_label', ''),
        server_list,
        '',
        msg_config.get('action_label', ''),
        action_instructions,
        '',
        response_interpretation_label,
        response_interpretation,
        '',
        msg_config.get('zero_tools_warning', DEFAULT_CONFIG['message']['zero_tools_warning']),
        '',
        msg_config.get('failure_instruction', ''),
    ]

    return '\n'.join(message_parts)


def main() -> None:
    """Main hook execution function."""
    try:
        # Load configuration (defaults merged with config file if provided)
        config_loader = _load_config_loader()
        config = config_loader.get_config_from_argv(DEFAULT_CONFIG)

        # Check if hook is enabled
        if not config.get('enabled', True):
            sys.exit(0)

        # Read input from stdin
        input_data = json.load(sys.stdin)

        # Extract key fields
        hook_event_name = input_data.get('hook_event_name', '')

        # Initial validation - only run on SessionStart events
        if hook_event_name != 'SessionStart':
            sys.exit(0)

        # Build and output context message for the model
        context_message = build_context_message(config)
        try:
            json_output = _load_json_output()
            json_output.emit_additional_context('SessionStart', context_message)
        except ImportError:
            print(context_message)

        # Always exit successfully
        sys.exit(0)

    except Exception:
        # Handle all errors silently and exit successfully
        sys.exit(0)


if __name__ == '__main__':
    main()
