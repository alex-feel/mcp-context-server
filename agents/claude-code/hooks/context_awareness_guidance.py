#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pyyaml"]
# ///
"""
Context Awareness Guidance Hook for Claude Code.

This hook provides guidance text to the orchestrator at session start,
informing it about the option to check recent context entries for workflow
continuity after context compaction or session resume.

The guidance content is configurable via external YAML configuration.

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
    'guidance': {
        'header': 'THREAD CONTEXT AWARENESS',
        'purpose': (
            'After context compaction or session resume, you may have lost track of workflow state. '
            'Subagents always retrieve their own context via context-retrieval-protocol skill, '
            'but you as orchestrator might benefit from checking recent context entries.'
        ),
        'when_to_check': [
            'At session start, especially after context compaction',
            'When agent_report_ids tracking is unavailable',
            'When user asks about recent work or workflow state',
        ],
        'how_to_check': {
            'description': 'Use search_context to browse recent entries, then get_context_by_ids for relevant ones',
            'example': "search_context(thread_id='thread-id', source='agent', limit=5)",
        },
        'what_to_look_for': [
            'Report IDs for agent_report_ids tracking',
            'Workflow phase indicators (which agent completed last)',
            'Pending continuation markers (status: pending)',
        ],
        'boundaries': [
            'This is for YOUR workflow awareness, not for agent task construction',
            'Do NOT summarize retrieved content to agents',
            'Agents retrieve context themselves via context-retrieval-protocol',
        ],
        'emphasis': 'This check is OPTIONAL but recommended for workflow continuity.',
        'recovery_after_compaction': {
            'header': 'Recovery After Compaction',
            'steps': [
                'Check for agent_report_ids values - they may have been lost during compaction',
                "Query context-server for most recent reports: search_context(thread_id='...', source='agent', limit=5)",
                'Look for checkpoint entries with status: pending to identify interrupted workflows',
                'Continue workflow from last confirmed checkpoint',
                'Fall back to fresh invocations if report IDs are unavailable',
            ],
        },
    },
}


def build_guidance_message(config: dict[str, Any]) -> str:
    """
    Build the guidance message from config components.

    The message informs the orchestrator about checking recent context
    entries for workflow continuity.

    Args:
        config: Configuration dictionary with guidance content

    Returns:
        Complete guidance message string
    """
    guidance = config.get('guidance', DEFAULT_CONFIG['guidance'])

    # Build when_to_check section
    when_items = guidance.get('when_to_check', DEFAULT_CONFIG['guidance']['when_to_check'])
    when_lines = '\n'.join(f'  - {item}' for item in when_items)

    # Build how_to_check section
    how_to = guidance.get('how_to_check', DEFAULT_CONFIG['guidance']['how_to_check'])
    how_description = how_to.get('description', DEFAULT_CONFIG['guidance']['how_to_check']['description'])
    how_example = how_to.get('example', DEFAULT_CONFIG['guidance']['how_to_check']['example'])
    how_section = f'{how_description}\n  Example: {how_example}'

    # Build what_to_look_for section
    what_items = guidance.get('what_to_look_for', DEFAULT_CONFIG['guidance']['what_to_look_for'])
    what_lines = '\n'.join(f'  - {item}' for item in what_items)

    # Build boundaries section
    boundary_items = guidance.get('boundaries', DEFAULT_CONFIG['guidance']['boundaries'])
    boundary_lines = '\n'.join(f'  - {item}' for item in boundary_items)

    # Build recovery section
    recovery = guidance.get('recovery_after_compaction', DEFAULT_CONFIG['guidance']['recovery_after_compaction'])
    recovery_header = recovery.get('header', DEFAULT_CONFIG['guidance']['recovery_after_compaction']['header'])
    recovery_steps = recovery.get('steps', DEFAULT_CONFIG['guidance']['recovery_after_compaction']['steps'])
    recovery_lines = '\n'.join(f'  {i}. {step}' for i, step in enumerate(recovery_steps, 1))

    # Construct full message
    message_parts = [
        guidance.get('header', DEFAULT_CONFIG['guidance']['header']),
        '',
        guidance.get('purpose', DEFAULT_CONFIG['guidance']['purpose']),
        '',
        'When to check:',
        when_lines,
        '',
        'How to check:',
        f'  {how_section}',
        '',
        'What to look for:',
        what_lines,
        '',
        'Important boundaries:',
        boundary_lines,
        '',
        f'{recovery_header}:',
        recovery_lines,
        '',
        guidance.get('emphasis', DEFAULT_CONFIG['guidance']['emphasis']),
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

        # Only run on SessionStart events
        if hook_event_name != 'SessionStart':
            sys.exit(0)

        # Build and output guidance message for the orchestrator
        guidance_message = build_guidance_message(config)
        try:
            json_output = _load_json_output()
            json_output.emit_additional_context('SessionStart', guidance_message)
        except ImportError:
            print(guidance_message)

        # Always exit successfully
        sys.exit(0)

    except Exception:
        # Handle all errors silently and exit successfully
        sys.exit(0)


if __name__ == '__main__':
    main()
