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
DEFAULT_CONFIG: dict[str, Any] = {
    'enabled': True,
    'guidance': {
        'header': 'THREAD CONTEXT AWARENESS',
        'purpose': (
            'After context compaction or session resume, you may have lost track of session state '
            'AND the original user requirements. ALWAYS retrieve user messages first (source="user") -- '
            'these contain the original requirements and are the PRIMARY SOURCE OF TRUTH. Then check '
            'recent agent reports (source="agent") to restore awareness of prior work.'
        ),
        'when_to_check': [
            'At session start, especially after context compaction (retrieve original user requirements first)',
            'When you are unsure about recent work or session state',
            'When the user asks about prior work or session history',
            "Before any action that depends on interpreting the user's original requirements",
        ],
        'how_to_check': {
            'description': (
                "Retrieve user messages FIRST (source='user'), then agent reports (source='agent'), "
                'then use get_context_by_ids for full content of relevant entries from both searches.'
            ),
            'example': (
                "search_context(thread_id='thread-id', source='user', limit=10)   # USER MESSAGES FIRST\n"
                "search_context(thread_id='thread-id', source='agent', limit=10)  # THEN AGENT REPORTS\n"
                'get_context_by_ids(context_ids=[...relevant IDs, user messages first...])  # FULL CONTENT'
            ),
        },
        'what_to_look_for': [
            "User messages (source='user') containing original requirements - AUTHORITATIVE source of truth",
            'Recent agent work reports and their status',
            'Pending continuation markers (status: pending)',
            'User decisions and preferences from this session',
            'Discrepancies between agent reports and the original user messages',
        ],
        'boundaries': [
            'This check is for restoring YOUR session awareness',
            'Always retrieve user messages first (they are the AUTHORITATIVE source of truth)',
            'User messages override any summary or interpretation (including your own compacted recollection)',
            'Do NOT preemptively load all context -- retrieve user messages, then agent reports as needed',
            'Focus on the most recent entries first',
        ],
        'emphasis': 'This check is OPTIONAL but recommended for session continuity.',
        'recovery_after_compaction': {
            'header': 'Recovery After Compaction',
            'steps': [
                (
                    "FIRST retrieve user messages: search_context(thread_id='...', source='user', limit=10) -- "
                    'original requirements are authoritative'
                ),
                "THEN retrieve agent reports: search_context(thread_id='...', source='agent', limit=10)",
                (
                    'Use get_context_by_ids to read full content of relevant user messages AND agent reports '
                    '(user messages first)'
                ),
                'Look for entries with status: pending to identify interrupted work',
                (
                    'Reconcile any orchestrator task description against retrieved user messages; '
                    'user messages win on conflict'
                ),
                'Continue from last confirmed state, verified against user messages',
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
