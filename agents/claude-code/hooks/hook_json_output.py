#!/usr/bin/env python3
"""Shared utility for emitting structured JSON output from Claude Code hooks.

Provides three emitter functions for the three documented JSON output schemas:

- emit_additional_context: injects context into the model (UserPromptSubmit,
  SessionStart, SubagentStart, and non-blocking PreToolUse/PostToolUse).
- emit_pre_tool_use_deny: denies a PreToolUse tool call via
  hookSpecificOutput.permissionDecision=deny with permissionDecisionReason.
- emit_decision_block: blocks via top-level decision=block with reason
  (PostToolUse, Stop, SubagentStop).

All emitters write structured JSON to stdout. Callers MUST exit with status 0
after invoking an emitter; Claude Code processes JSON output only when the exit
code is 0.
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


def emit_pre_tool_use_deny(reason: str) -> None:
    """Emit a JSON hookSpecificOutput that DENIES a PreToolUse tool call.

    Writes the structured JSON output that Claude Code interprets as a
    blocking permission decision for PreToolUse. The reason appears in
    the model's context as a permissionDecisionReason, allowing Claude
    to understand why the action was blocked and avoid retrying.

    The caller MUST exit with status 0 after calling this function;
    Claude Code only processes JSON output when the exit code is 0.

    Args:
        reason: The reason for denying the tool call. Shown to the model
            as the permissionDecisionReason.
    """
    hook_output: dict[str, object] = {
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'deny',
            'permissionDecisionReason': reason,
        },
    }
    sys.stdout.write(json.dumps(hook_output))
    sys.stdout.flush()


def emit_decision_block(reason: str) -> None:
    """Emit a JSON top-level decision: block that provides feedback to Claude.

    Writes the structured JSON output that Claude Code interprets as a
    block decision for events that use the top-level decision schema:
    PostToolUse, Stop, SubagentStop. The reason is fed to Claude as a
    user-side message in the next model request.

    The caller MUST exit with status 0 after calling this function;
    Claude Code only processes JSON output when the exit code is 0.

    Args:
        reason: The reason for blocking. Shown to Claude as the feedback
            message that explains what went wrong and how to proceed.
    """
    hook_output: dict[str, object] = {
        'decision': 'block',
        'reason': reason,
    }
    sys.stdout.write(json.dumps(hook_output))
    sys.stdout.flush()
