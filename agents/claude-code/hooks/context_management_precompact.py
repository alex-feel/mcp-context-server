#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pyyaml"]
# ///
"""
Context Management PreCompact Hook for Claude Code.

This hook triggers before context compaction, prompting the orchestrator to
preserve critical workflow state to the context-server before compaction occurs.

The message content is configurable via external YAML configuration.

Trigger: PreCompact event (fires before Claude Code performs context compaction)
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


# Default configuration - used when no config file provided
# Maintains backward compatibility with hardcoded values
DEFAULT_CONFIG: dict[str, Any] = {
    'enabled': True,
    'subagent_compaction_mode': 'skip',
    'message': {
        'header': 'CONTEXT COMPACTION RULES',
        'instruction': 'Follow these compaction rules for state management:',
        'preservation_rules': [
            {
                'category': 'Current workflow state',
                'action': 'PRESERVE',
                'rationale': 'Required for correct orchestration',
            },
            {
                'category': 'Detailed agent outputs',
                'action': 'DO NOT STORE',
                'rationale': 'Agents retrieve from context-server',
            },
            {
                'category': 'Historical decision reasoning',
                'action': 'SUMMARIZE',
                'rationale': 'Keep decision, compress rationale',
            },
        ],
        'must_responsibilities': [
            'Never lose current state: Active workflow phase, pending gates, agent IDs',
            'Maintain reference IDs: context_ids for all completed work phases',
            'Rely on context-server: Agents retrieve full details themselves',
            'Report, not retain: Present information to user, then release',
        ],
        'must_not_responsibilities': [
            'Attempt to retain full agent outputs in its own context',
            'Build summaries of prior work for agents (they fetch their own)',
            'Store redundant copies of information available in context-server',
        ],
    },
    # Empty reference sections for backward compatibility
    'critical_universal_state': {'variables': []},
    'critical_default_flow_state': {'variables': []},
    'critical_consensus_flow_state': {'variables': []},
    'critical_phased_execution_state': {'variables': []},
    'structured_snapshot_template': {},
}

# Mapping of flow types to their config section names
FLOW_SECTION_MAPPING: dict[str, str] = {
    'universal': 'critical_universal_state',
    'default_flow': 'critical_default_flow_state',
    'consensus_flow': 'critical_consensus_flow_state',
    'phased_execution': 'critical_phased_execution_state',
}

# Human-readable labels for flow types in output
FLOW_LABELS: dict[str, str] = {
    'universal': 'UNIVERSAL (All Flows)',
    'default_flow': 'DEFAULT FLOW',
    'consensus_flow': 'CONSENSUS FLOW',
    'phased_execution': 'PHASED EXECUTION (LARGE Plans)',
}


def is_subagent_context(input_data: dict[str, Any]) -> bool:
    """Detect if PreCompact is firing inside a subagent context.

    Checks the transcript_path field for the /subagents/ directory segment,
    which indicates the transcript belongs to a subagent rather than the
    main session. Path separators are normalized for cross-platform support.

    Args:
        input_data: The parsed JSON input from the PreCompact hook event.

    Returns:
        True if the context indicates a subagent, False otherwise.
    """
    transcript_path = input_data.get('transcript_path', '')
    normalized = transcript_path.replace('\\', '/')
    return '/subagents/' in normalized


def extract_priority_variables(
    config: dict[str, Any],
    priorities: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract variables from reference sections grouped by flow type.

    Filters variables by priority level from all critical_*_state sections
    and returns them organized by flow type.

    Args:
        config: Full configuration dictionary containing reference sections
        priorities: List of priorities to include (default: ['P0', 'P1'])

    Returns:
        Dict mapping flow type to list of filtered variables
    """
    if priorities is None:
        priorities = ['P0', 'P1']

    result: dict[str, list[dict[str, Any]]] = {}

    for flow_type, section_name in FLOW_SECTION_MAPPING.items():
        section = config.get(section_name, {})
        variables = section.get('variables', [])
        filtered = [var for var in variables if var.get('priority', '') in priorities]
        if filtered:
            result[flow_type] = filtered

    return result


def build_flow_checklists(config: dict[str, Any]) -> str:
    """
    Build flow-specific preservation checklists from reference sections.

    Extracts P0/P1 priority variables from all critical_*_state sections
    and formats them as categorized checklists for the orchestrator.

    Args:
        config: Full configuration dictionary containing reference sections

    Returns:
        Formatted checklist string or empty string if no reference sections
    """
    variables_by_flow = extract_priority_variables(config, ['P0', 'P1'])

    if not variables_by_flow:
        return ''

    lines: list[str] = [
        '',
        '=== FLOW-SPECIFIC PRESERVATION CHECKLISTS ===',
        '',
    ]

    for flow_type, label in FLOW_LABELS.items():
        if flow_type not in variables_by_flow:
            continue

        lines.append(f'{label}:')
        for var in variables_by_flow[flow_type]:
            name = var.get('name', 'unknown')
            priority = var.get('priority', 'P?')
            lines.append(f'  [{priority}] {name}')
        lines.append('')

    return '\n'.join(lines)


def get_header_content(config: dict[str, Any]) -> str:
    """
    Get header content, preferring structured_snapshot_template if available.

    Falls back to message.header if structured_snapshot_template is not defined.

    Args:
        config: Full configuration dictionary

    Returns:
        Header string for the precompact message
    """
    message = config.get('message', DEFAULT_CONFIG['message'])
    default_message = DEFAULT_CONFIG['message']

    # Try structured_snapshot_template first
    template_section = config.get('structured_snapshot_template', {})
    template_format = template_section.get('format')

    if template_format:
        # Use description as intro if available
        description = template_section.get('description', '')
        if description:
            return f'{description}\n\n{template_format}'
        return str(template_format)

    # Fall back to message.header
    header = message.get('header', default_message.get('header', ''))
    return str(header)


def build_precompact_message(config: dict[str, Any]) -> str:
    """
    Build the precompact compaction rules message from config components.

    The message instructs the orchestrator on WHAT to preserve/summarize/not store,
    WHY each category matters, and HOW to manage context (MUST/MUST NOT).

    Uses reference sections (critical_*_state) to generate flow-specific checklists
    and structured_snapshot_template for output format guidance.

    Args:
        config: Configuration dictionary with message content and reference sections

    Returns:
        Complete precompact message string with flow-specific checklists
    """
    message = config.get('message', DEFAULT_CONFIG['message'])
    default_message = DEFAULT_CONFIG['message']

    # Build preservation rules section (WHAT + WHY)
    preservation_rules = message.get('preservation_rules', default_message['preservation_rules'])
    rules_lines: list[str] = []
    for rule in preservation_rules:
        category = rule.get('category', '')
        action = rule.get('action', '')
        rationale = rule.get('rationale', '')
        rules_lines.append(f'  - {category}: {action} ({rationale})')

    # Build MUST responsibilities section
    must_items = message.get('must_responsibilities', default_message['must_responsibilities'])
    must_lines = '\n'.join(f'  {i}. {item}' for i, item in enumerate(must_items, 1))

    # Build MUST NOT responsibilities section
    must_not_items = message.get('must_not_responsibilities', default_message['must_not_responsibilities'])
    must_not_lines = '\n'.join(f'  {i}. {item}' for i, item in enumerate(must_not_items, 1))

    # Get header content (prefers structured_snapshot_template if available)
    header = get_header_content(config)

    # Construct full message
    message_parts = [
        header,
        '',
        message.get('instruction', default_message['instruction']),
        '',
        'What to Preserve vs. Summarize:',
        '\n'.join(rules_lines),
        '',
        'The orchestrator MUST:',
        must_lines,
        '',
        'The orchestrator MUST NOT:',
        must_not_lines,
    ]

    # Add flow-specific checklists from reference sections
    flow_checklists = build_flow_checklists(config)
    if flow_checklists:
        message_parts.append(flow_checklists)

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

        # Detect subagent context and handle according to configured mode
        if is_subagent_context(input_data):
            mode = config.get('subagent_compaction_mode', 'skip')
            if mode != 'full':
                # Skip mode (default): exit silently for subagent compaction
                sys.exit(0)
            # Full mode: fall through to orchestrator output for debugging

        # Extract key fields
        hook_event_name = input_data.get('hook_event_name', '')

        # Only run on PreCompact events
        if hook_event_name != 'PreCompact':
            sys.exit(0)

        # Build and output precompact message for the orchestrator
        precompact_message = build_precompact_message(config)
        print(precompact_message)

        # Always exit successfully
        sys.exit(0)

    except Exception:
        # Handle all errors silently and exit successfully
        sys.exit(0)


if __name__ == '__main__':
    main()
