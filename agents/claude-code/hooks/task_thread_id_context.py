#!/usr/bin/env python3
"""
Claude Code Hook: Task/Agent Thread ID Context Integration

This hook ensures thread ID context is included in task descriptions
when spawning subagents using the Task/Agent tool. It reads the existing thread ID
from the .context_server/.thread_id file and provides guidance to the model to include it
for better context continuity across agent hierarchy.

Triggers on: PreToolUse (Task|Agent)
Target: Task/Agent tool operations
Action: Guide model to include thread ID context in task descriptions

Exit Codes:
- 0: Success (thread ID context present, not found, or not a Task/Agent tool)
- 2: Guidance provided to include thread ID context (blocking)
"""

import importlib.util
import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from types import ModuleType
from typing import Any

# Type alias for JSON-like values that can be recursively processed
JSONValue = str | int | float | bool | None | dict[str, Any] | list[Any]


def _load_json_output() -> ModuleType:
    """Dynamically load hook_json_output from the same directory."""
    loader_path = Path(__file__).parent / 'hook_json_output.py'
    spec = importlib.util.spec_from_file_location('hook_json_output', loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load hook_json_output from {loader_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contains_thread_id(text: str | None, thread_id: str) -> bool:
    """
    Check if text contains the thread ID.

    Args:
        text: Text to check for thread ID
        thread_id: The thread ID to look for

    Returns:
        bool: True if thread ID is found, False otherwise
    """
    if not text or not thread_id:
        return False

    # Check if the thread ID appears in the text
    return thread_id in text


def check_thread_id_in_tool_input(tool_input: dict[str, Any], thread_id: str) -> bool:
    """
    Check if thread ID exists anywhere in the tool input.

    Recursively searches through all string values in the tool_input
    to find the thread ID.

    Args:
        tool_input: The tool input dictionary
        thread_id: The thread ID to search for

    Returns:
        bool: True if thread ID is found, False otherwise
    """
    def search_for_thread_id(obj: JSONValue) -> bool:
        """Recursively search for thread ID in any string value."""
        if isinstance(obj, str):
            return contains_thread_id(obj, thread_id)
        if isinstance(obj, dict):
            for value in obj.values():
                if search_for_thread_id(value):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if search_for_thread_id(item):
                    return True
        return False

    return search_for_thread_id(tool_input)


def read_thread_id(claude_project_dir: str) -> str | None:
    """
    Read the existing thread ID from the .context_server/.thread_id file.

    Args:
        claude_project_dir: The Claude project directory path

    Returns:
        str | None: The thread ID if found, None otherwise
    """
    try:
        # Construct the thread ID file path
        thread_id_file = os.path.join(claude_project_dir, '.context_server', '.thread_id')
        thread_id_path = Path(thread_id_file)

        if thread_id_path.exists():
            with suppress(OSError):
                existing_thread_id = thread_id_path.read_text(encoding='utf-8').strip()
                if existing_thread_id:
                    return existing_thread_id
    except Exception:
        pass

    return None


def generate_thread_context(thread_id: str) -> str:
    """
    Generate thread ID context string.

    Args:
        thread_id: The thread ID to include in context

    Returns:
        str: Formatted thread ID context
    """
    return (
        f'THREAD CONTEXT: Use current thread ID {thread_id} for this task. '
        'This maintains context and continuity across the agent hierarchy.\n'
    )


def main() -> None:
    """Main hook execution."""
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        # Extract and validate event and tool
        hook_event_name = input_data.get('hook_event_name', '')
        tool_name = input_data.get('tool_name', '')

        # Initial validation - exit silently if conditions not met
        if hook_event_name != 'PreToolUse':
            sys.exit(0)

        if tool_name not in ('Task', 'Agent'):
            sys.exit(0)

        # Get Claude project directory
        claude_project_dir = os.environ.get('CLAUDE_PROJECT_DIR')
        if not claude_project_dir:
            sys.exit(0)

        # Read existing thread ID from file
        thread_id = read_thread_id(claude_project_dir)
        if not thread_id:
            # No thread ID found, allow the operation to proceed
            sys.exit(0)

        # Extract tool input
        tool_input = input_data.get('tool_input', {})

        # Check if thread ID is already present anywhere in the tool input
        if check_thread_id_in_tool_input(tool_input, thread_id):
            # Thread ID context is already present, inject reinforcement context
            try:
                json_output = _load_json_output()
                json_output.emit_additional_context('PreToolUse', generate_thread_context(thread_id))
            except ImportError:
                pass
            sys.exit(0)

        # Generate the thread context
        thread_context = generate_thread_context(thread_id)

        # Generate guidance for including thread ID context
        guidance_message = (
            'GUIDANCE: Please include current thread ID context in your task description.\n\n'
            'For better context continuity across the agent hierarchy, the task description must include:\n\n'
            f'{thread_context}\n'
            'This helps subagents maintain context awareness and enables proper context retrieval '
            'and storage operations. The thread ID ensures all agents in the hierarchy share '
            'the same contextual thread.\n\n'
            'Please revise your task description to include this thread ID context.'
        )

        # Provide guidance to the model (exit code 2 for model feedback)
        print(guidance_message, file=sys.stderr)
        sys.exit(2)  # Block tool call and send feedback to Claude Code for processing

    except json.JSONDecodeError:
        sys.exit(0)  # Silent failure for invalid JSON
    except Exception:
        sys.exit(0)  # Silent failure for unexpected errors


if __name__ == '__main__':
    main()
