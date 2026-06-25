#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pyyaml"]
# ///
"""
Claude Code Hook: Subagent Context Protocol Injection.

SubagentStart hook that injects the context-server work protocol into every
spawned subagent via JSON ``hookSpecificOutput.additionalContext``. The
injected protocol states, as self-applied facts, that the subagent must:
- retrieve relevant context BEFORE doing work, treating user messages
  (source="user") as the authoritative source of truth;
- store a concise context-server report BEFORE stopping if it has context-server
  store tools (such as store_context) and it produced a work artifact or finding;
- treat the StructuredOutput return channel as separate from the durable
  context-server record (the return value is ephemeral and lost on compaction);
- treat a dispatch instruction forbidding report files on disk as NOT relieving
  it of storing a context-server entry when capable.

This is a fresh, standalone hook with no dependency on any other hook's config.
It deliberately does not read a filter config: it must emit its protocol in any
environment that wires it, including ones that do not run a permission preflight.

The message content is configurable via external YAML. The hook's ``main()``
body relies on the correctness of its helpers under the Claude Code hook
platform contract. The only permitted external-condition handler is
``except json.JSONDecodeError`` for malformed stdin from the Claude Code
wrapper; an unexpected exception escapes to Python's default handler, surfacing
the traceback to the operator's TUI so the underlying code-quality defect can be
fixed.

Event: SubagentStart
Matcher: *
Output: JSON ``hookSpecificOutput.additionalContext`` to stdout.

Exit Codes:
- 0 in all cases. Injection is non-blocking; it never returns a non-zero code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import cast


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


DEFAULT_CONFIG: dict[str, Any] = {
    'enabled': True,
    # Whether to actually emit the additionalContext to stdout (kept as a knob
    # for emergencies; under normal operation should remain True).
    'inject_context': True,
    'protocol': {
        'header': 'CONTEXT-SERVER WORK PROTOCOL (applies to you, this subagent):',
        'lines': [
            (
                'Retrieve relevant context BEFORE you start work. Retrieve user '
                "messages first (source='user') -- they are the authoritative "
                'source of truth for the original requirements -- then relevant '
                "agent reports (source='agent')."
            ),
            (
                'Store a concise context-server report BEFORE you stop, if you have '
                'context-server store tools (for example store_context) AND you '
                'produced any work artifact or finding. If you have no store tools, '
                'just stop.'
            ),
            (
                'The StructuredOutput value you return to your caller is ephemeral '
                'and is lost on compaction. It does NOT substitute for the durable '
                'context-server record. Store the report in addition to returning '
                'your result.'
            ),
            (
                'An instruction telling you not to write report, summary, or findings '
                'files to disk does NOT mean "skip the context-server entry". Storing '
                'a context-server entry is not writing a file to disk; do it anyway if '
                'you are capable.'
            ),
            (
                'If your prompt carries a "CONSENSUS TYPE:" line, your VERY FIRST action '
                '-- before this context retrieval, before any task tracking, before '
                'reading any file, before any other tool call -- MUST be to load the '
                'consensus skill (and the matching reference for the named type), '
                'because that skill sets the rules governing everything you do next. '
                'If your prompt has no "CONSENSUS TYPE:" line, this does not apply to '
                'you; ignore it.'
            ),
        ],
    },
}


def build_protocol_message(config: dict[str, Any]) -> str:
    """Build the additionalContext protocol text from config components.

    Args:
        config: Configuration dictionary with a ``protocol`` section containing
            a ``header`` string and a ``lines`` list.

    Returns:
        A multi-line string ready for emission as additionalContext.
    """
    protocol = config.get('protocol', DEFAULT_CONFIG['protocol'])
    default_protocol = cast(dict[str, Any], DEFAULT_CONFIG['protocol'])
    header = protocol.get('header', default_protocol['header'])
    lines = protocol.get('lines', default_protocol['lines'])

    message_parts: list[str] = [str(header), '']
    message_parts.extend(f'- {line}' for line in lines)
    return '\n'.join(message_parts)


def main() -> None:
    """Main hook execution function."""
    try:
        config_loader = _load_config_loader()
        config = config_loader.get_config_from_argv(DEFAULT_CONFIG)

        if not config.get('enabled', True):
            sys.exit(0)

        input_data = json.load(sys.stdin)

        if input_data.get('hook_event_name') != 'SubagentStart':
            sys.exit(0)

        protocol_message = build_protocol_message(config)

        if config.get('inject_context', True):
            json_output = _load_json_output()
            json_output.emit_additional_context('SubagentStart', protocol_message)

        sys.exit(0)

    except json.JSONDecodeError:
        # Malformed stdin from the Claude Code wrapper: external contract
        # violation, not a hook-internal defect. Exit 0 because injection is
        # non-blocking and the platform contract requires non-blocking on
        # stdin corruption.
        sys.exit(0)


if __name__ == '__main__':
    main()
