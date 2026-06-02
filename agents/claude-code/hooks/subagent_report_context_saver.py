#!/usr/bin/env python3
"""
Subagent Report Saver Hook for Claude Code.

This hook intercepts SubagentStop events and instructs the agent to save a
comprehensive work report to the context server before stopping. This ensures
all subagent work is properly documented and preserved for future reference.

The hook blocks the stop action via JSON output (top-level decision: "block")
with the instruction text delivered as the `reason` field, so Claude reads the
instruction as guidance for its next turn. A defense-in-depth ImportError
fallback writes the same instruction to stderr and exits 2 if the shared JSON
helper module cannot be loaded. The hook exits 0 on the modern path so Claude
Code processes the JSON.

Trigger: SubagentStop
Exit Codes:
  - 0: Modern path (JSON decision: "block" emitted) or silent pass-through
  - 2: Defense-in-depth fallback when hook_json_output.py cannot be imported

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


def format_instruction_message() -> str:
    """
    Format the instruction message for the agent.

    Returns:
        Formatted instruction message for stderr
    """
    return '''FRIENDLY REMINDER: Work documentation required before stopping.

If you have already stored your work report, do NOT save it again — just stop.

Otherwise, follow your instructions/skills on context preservation.

If you don't have any instructions/skills on context preservation, do NOT save any report — just stop.'''


def _load_json_output() -> ModuleType:
    """Dynamically load hook_json_output from the same directory."""
    loader_path = Path(__file__).parent / 'hook_json_output.py'
    spec = importlib.util.spec_from_file_location('hook_json_output', loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load hook_json_output from {loader_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Main hook execution function."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)

        # Extract and validate hook event name
        hook_event_name = input_data.get('hook_event_name', '')
        if hook_event_name != 'SubagentStop':
            # Not a SubagentStop event, pass through silently
            sys.exit(0)

        # Check if stop_hook_active to prevent infinite loops
        stop_hook_active = input_data.get('stop_hook_active', False)
        if stop_hook_active:
            # Hook is already active, allow stop to prevent loops
            sys.exit(0)

        # Format the instruction message and emit modern JSON decision block.
        # Fallback to legacy stderr + exit 2 if the JSON helper is missing.
        instruction = format_instruction_message()
        try:
            json_output = _load_json_output()
            json_output.emit_decision_block(instruction)
        except ImportError:
            print(instruction, file=sys.stderr)
            sys.exit(2)
        sys.exit(0)

    except json.JSONDecodeError:
        # Malformed stdin from the Claude Code wrapper: external contract
        # violation, not a hook-internal defect. Exit 0 because the hook contract
        # requires non-blocking on stdin corruption (the model has no actionable
        # feedback to give).
        sys.exit(0)


if __name__ == '__main__':
    main()
