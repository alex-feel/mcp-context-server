#!/usr/bin/env python3
"""
Context Preservation Stop Hook for Claude Code.

This hook intercepts both Stop (main agent finished responding) and SubagentStop
(spawned subagent finished) events and instructs the principal to store a work
report to the context server before stopping. This ensures work performed by the
main agent and by every subagent is durably preserved for future reference and
survives context compaction.

The store directive is capability-gated and artifact-conditioned: the principal
stores a context-server report only if it has context-server store tools (such
as store_context) AND it produced a work artifact or finding this turn. The
directive also clarifies that the StructuredOutput value returned to the caller
is ephemeral and does NOT substitute for the durable context-server record, and
that a dispatch instruction forbidding report files on disk does NOT relieve a
capable principal of storing a context-server entry. The directive is ASCII-only
so it survives stdout capture intact.

The hook blocks the stop action via JSON output (top-level decision: "block")
with the directive text delivered as the `reason` field, so Claude reads it as
guidance for its next turn. A defense-in-depth ImportError fallback writes the
same directive to stderr and exits 2 if the shared JSON helper module cannot be
loaded. The hook exits 0 on the modern path so Claude Code processes the JSON.

Trigger: Stop and SubagentStop
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
    Format the capability-gated context-preservation directive.

    The directive applies to both the main agent (Stop) and a spawned subagent
    (SubagentStop). It is ASCII-only so it survives stdout capture, and it is
    self-contained so a reader needs no external context to act on it.

    Returns:
        The directive text delivered as the `reason` field of the decision block.
    """
    return (
        'CONTEXT PRESERVATION REQUIRED BEFORE STOPPING.\n'
        '\n'
        'If you have context-server store tools (for example store_context) AND you '
        'produced any work artifact or finding this turn, you MUST store a concise '
        'context-server report before you stop.\n'
        '\n'
        'Rules:\n'
        '- Idempotency: if you have already stored your report this turn, do NOT store '
        'it again -- just stop.\n'
        '- Channel separation: a StructuredOutput or return value to your caller is '
        'ephemeral and is lost on compaction. It does NOT substitute for the durable '
        'context-server record. Store the context-server report in addition to '
        'returning your result.\n'
        '- Dispatch contract: an instruction telling you not to write report, summary, '
        'or findings files to disk does NOT mean "skip the context-server entry". '
        'Storing a context-server entry is not writing a file to disk; do it anyway if '
        'you are capable.\n'
        '- If you genuinely have no context-server store tools, just stop.'
    )


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

        # Extract and validate hook event name. The same directive applies to
        # the main agent (Stop) and a spawned subagent (SubagentStop); the
        # harness auto-converts a Stop-registered hook to SubagentStop in
        # subagent context, so a membership test covers both principals.
        hook_event_name = input_data.get('hook_event_name', '')
        if hook_event_name not in ('Stop', 'SubagentStop'):
            # Not a stop-time event, pass through silently
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
