#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pyyaml"]
# ///
"""
Context Management PreCompact Hook for Claude Code.

This hook triggers before context compaction, prompting the principal agent to
preserve critical workflow state to the context-server before compaction occurs.
The principal may be an orchestrator or a worker main agent; the MUST / MUST NOT
sentence subject is rendered from a configurable role label so the same hook
serves both registers.

The message content is configurable via external YAML configuration. The hook
emits the message via the modern JSON ``hookSpecificOutput.additionalContext``
mechanism so Claude Code injects it into the principal's context window as
a system reminder.

Trigger: PreCompact event (fires before Claude Code performs context compaction)

main() relies on its helpers being correct under the platform contract; only
one external-condition handler exists (json.JSONDecodeError for malformed stdin
from the Claude Code wrapper). There is no catch-all except Exception block:
an unexpected exception escapes to Python's default handler, surfacing the
traceback to the operator's TUI so the underlying code-quality defect can be
fixed.
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


def _load_is_subagent() -> ModuleType:
    """Dynamically load hook_is_subagent from the same directory."""
    loader_path = Path(__file__).parent / 'hook_is_subagent.py'
    spec = importlib.util.spec_from_file_location('hook_is_subagent', loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load hook_is_subagent from {loader_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Built-in defaults used when no config file is provided.
DEFAULT_CONFIG: dict[str, Any] = {
    'enabled': True,
    'subagent_compaction_mode': 'skip',
    'message': {
        'header': 'CONTEXT COMPACTION RULES',
        'instruction': 'Follow these compaction rules for state management:',
        # Role label for the MUST / MUST NOT sentence subjects. Rendered with
        # str.capitalize() at sentence-initial position. Defaults to
        # 'the orchestrator' for orchestrator environments; worker environments
        # set this to 'you'.
        'principal_role': 'the orchestrator',
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
    # Lean worker-register recovery directive. Emitted to a subagent whose
    # PreCompact fires while subagent_compaction_mode is not 'full', so a
    # long-running worker that compacts mid-run still restores its working
    # context instead of silently losing it. Kept minimal on purpose: it
    # carries no verbose orchestrator snapshot.
    'subagent_recovery_message': {
        'header': 'CONTEXT COMPACTION RECOVERY (worker)',
        'instruction': (
            'Compaction is about to shrink your context. After it completes, '
            'restore your working state from the context-server before continuing:'
        ),
        'recovery_steps': [
            'Re-read the original user requirements by id (source=user; authoritative) to recover the goal.',
            'Re-read your own dispatched task and work-state by context_id to recover what you were doing.',
            'Store your report to the context-server before you finish so your work is not lost.',
        ],
    },
    # Empty reference sections retained as the no-flow default; populated by
    # config files that define flow-specific critical state.
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


def build_subagent_recovery_message(config: dict[str, Any]) -> str:
    """Build the lean worker-register recovery message for a subagent PreCompact.

    A long-running worker that compacts mid-run needs a minimal directive to
    restore its working state from the context-server afterward. This message
    is deliberately lean: it carries the worker recovery steps only, never the
    verbose orchestrator preservation snapshot.

    Args:
        config: Configuration dictionary; the ``subagent_recovery_message``
            section supplies the header, instruction, and recovery steps.

    Returns:
        A multi-line worker recovery message ready for emission as
        additionalContext.
    """
    recovery = config.get('subagent_recovery_message', DEFAULT_CONFIG['subagent_recovery_message'])
    default_recovery = DEFAULT_CONFIG['subagent_recovery_message']

    header = recovery.get('header', default_recovery['header'])
    instruction = recovery.get('instruction', default_recovery['instruction'])
    steps = recovery.get('recovery_steps', default_recovery['recovery_steps'])
    steps_lines = '\n'.join(f'  {i}. {step}' for i, step in enumerate(steps, 1))

    return f'{header}\n\n{instruction}\n\n{steps_lines}'


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

    The message instructs the principal agent on WHAT to preserve/summarize/not
    store, WHY each category matters, and HOW to manage context (MUST/MUST NOT).
    The MUST / MUST NOT sentence subject is rendered from the configurable
    message.principal_role label (default 'the orchestrator').

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

    # Render the MUST / MUST NOT sentence subjects from the configured role
    # label. The label is capitalized at this sentence-initial position so a
    # default of 'the orchestrator' renders 'The orchestrator MUST:' and a
    # worker label of 'you' renders 'You MUST:'.
    principal_role = message.get('principal_role', default_message['principal_role'])
    role_subject = str(principal_role).capitalize()

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
        f'{role_subject} MUST:',
        must_lines,
        '',
        f'{role_subject} MUST NOT:',
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

        # Only run on PreCompact events; anything else exits silently.
        hook_event_name = input_data.get('hook_event_name', '')
        if hook_event_name != 'PreCompact':
            sys.exit(0)

        json_output = _load_json_output()

        # Branch on principal: a subagent gets the lean worker recovery
        # directive (so a worker that compacts mid-run still restores its
        # state); the main session gets the full orchestrator preservation
        # snapshot. Detection rides the shared is_subagent helper, so the
        # documented agent_id / agent_type fields are the primary signal and
        # transcript_path is the fallback.
        is_subagent_mod = _load_is_subagent()
        if is_subagent_mod.is_subagent(input_data):
            mode = config.get('subagent_compaction_mode', 'skip')
            if mode != 'full':
                # Skip mode (default): emit the lean worker recovery directive
                # instead of exiting silently, so a long-running subagent
                # restores its working context after compaction.
                recovery_message = build_subagent_recovery_message(config)
                json_output.emit_additional_context('PreCompact', recovery_message)
                sys.exit(0)
            # Full mode: fall through to the orchestrator output for debugging.

        # Build and emit the precompact message for the orchestrator via JSON
        # additionalContext (modern context-injection mechanism for PreCompact events)
        precompact_message = build_precompact_message(config)
        json_output.emit_additional_context('PreCompact', precompact_message)

        # Always exit successfully
        sys.exit(0)

    except json.JSONDecodeError:
        # Malformed stdin from the Claude Code wrapper: external contract
        # violation, not a hook-internal defect. Exit 0 because the hook contract
        # requires non-blocking on stdin corruption (the model has no actionable
        # feedback to give).
        sys.exit(0)


if __name__ == '__main__':
    main()
