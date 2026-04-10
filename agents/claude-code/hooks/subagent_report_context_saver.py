#!/usr/bin/env python3
"""
Subagent Report Saver Hook for Claude Code.

This hook intercepts SubagentStop events and instructs the agent to save a
comprehensive work report to the context server before stopping. This ensures
all subagent work is properly documented and preserved for future reference.

The hook blocks the stop action (exit code 2) and provides detailed instructions
via stderr for the agent to create and save their work report.

Trigger: SubagentStop
Exit Codes:
  - 0: Silent pass-through (on errors or invalid events)
  - 2: Block stop and provide instructions (on valid SubagentStop)
"""

import json
import sys


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

        # Format and output the instruction message
        instruction = format_instruction_message()
        print(instruction, file=sys.stderr)

        # Exit with code 2 to block the stop and deliver instructions
        sys.exit(2)

    except json.JSONDecodeError:
        # Invalid JSON input, fail silently
        sys.exit(0)
    except Exception:
        # Any other error, fail silently
        sys.exit(0)


if __name__ == '__main__':
    main()
