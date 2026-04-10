#!/usr/bin/env python3
"""Shared utility for emitting JSON hookSpecificOutput.additionalContext from hooks.

Provides a single function to format and write the JSON structure that Claude Code
expects for injecting additional context into the model's context window via hooks.
"""

from __future__ import annotations

import json
import sys


def emit_additional_context(hook_event_name: str, message: str) -> None:
    """Emit a JSON hookSpecificOutput with additionalContext to stdout.

    Writes the structured JSON output that Claude Code interprets as
    hook-injected context. The message appears in the model's context window
    as system-level information.

    Args:
        hook_event_name: The hook event type (e.g., 'SessionStart', 'PreToolUse').
        message: The context message to inject into the model's context.
    """
    hook_output: dict[str, object] = {
        'hookSpecificOutput': {
            'hookEventName': hook_event_name,
            'additionalContext': message,
        },
    }
    sys.stdout.write(json.dumps(hook_output))
    sys.stdout.flush()
