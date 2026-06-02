#!/usr/bin/env python3
"""
Thread ID Manager Hook for Claude Code

This hook manages thread IDs for the context server:
1. Validates that the hook event is 'SessionStart' or 'SubagentStart'
2. Takes session_id from the Claude Code payload (API contract)
3. Writes it as thread_id to .context_server/.thread_id file ONLY if it differs
4. Outputs unified thread context message for ALL sources

Adapter pattern: Claude Code provides session_id; this hook translates it to
thread_id for the context server, bridging the two naming conventions.

Trigger: SessionStart and SubagentStart with any source

main() relies on its helpers being correct under the platform contract; only
one external-condition handler exists (json.JSONDecodeError for malformed stdin
from the Claude Code wrapper). There is no catch-all except Exception block:
an unexpected exception escapes to Python's default handler, surfacing the
traceback to the operator's TUI so the underlying code-quality defect can be
fixed.
"""

import importlib.util
import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from types import ModuleType


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

        # Extract key fields
        hook_event_name = input_data.get('hook_event_name', '')
        session_id = input_data.get('session_id', '')

        # Validate hook event and session_id
        if hook_event_name not in ('SessionStart', 'SubagentStart'):
            sys.exit(0)

        if not session_id:
            sys.exit(0)

        # Get Claude project directory
        claude_project_dir = os.environ.get('CLAUDE_PROJECT_DIR')
        if not claude_project_dir:
            sys.exit(0)

        # Ensure .context_server directory exists
        context_server_dir = os.path.join(claude_project_dir, '.context_server')
        os.makedirs(context_server_dir, exist_ok=True)

        # File path for thread ID
        thread_id_file = Path(os.path.join(context_server_dir, '.thread_id'))

        # Read current value (if file exists)
        current_value = ''
        if thread_id_file.exists():
            with suppress(OSError):
                current_value = thread_id_file.read_text(encoding='utf-8').strip()

        # Write new thread ID only if it differs (efficiency optimization)
        if session_id != current_value:
            with suppress(OSError):
                thread_id_file.write_text(session_id, encoding='utf-8')

        # Output unified thread context message to the model
        context_message = (
            f'THREAD CONTEXT: Current thread ID is {session_id}.\n'
            'This thread ID identifies the current context-server thread for context storage and retrieval. '
            'Use this value as thread_id when working with the context-server.'
        )
        try:
            json_output = _load_json_output()
            json_output.emit_additional_context(hook_event_name, context_message)
        except ImportError:
            print(context_message)

        sys.exit(0)

    except json.JSONDecodeError:
        # Malformed stdin from the Claude Code wrapper: external contract
        # violation, not a hook-internal defect. Exit 0 because the hook contract
        # requires non-blocking on stdin corruption (the model has no actionable
        # feedback to give).
        sys.exit(0)


if __name__ == '__main__':
    main()
