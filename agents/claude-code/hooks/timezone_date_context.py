#!/usr/bin/env python3
"""
Timezone and Date Context Hook for Claude Code

This hook provides timezone and date context to the model at session start,
helping the model understand the user's current timezone and date for better
context when handling date-related queries.

Trigger: SessionStart with any source (no source restrictions)
"""

import importlib.util
import json
import sys
from datetime import UTC
from datetime import datetime
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

        # Initial validation - only run on SessionStart events
        if hook_event_name != 'SessionStart':
            sys.exit(0)

        # Get current timezone and date with timezone awareness
        current_time = datetime.now(tz=UTC).astimezone()
        current_date = current_time.strftime('%Y-%m-%d')

        # Get timezone name with safe ASCII fallback to avoid stdout encoding issues
        try:
            timezone_name = current_time.strftime('%Z')
            if not timezone_name or not timezone_name.isascii():
                timezone_name = 'Local'
        except Exception:
            timezone_name = 'Local'

        # Calculate UTC offset
        utc_offset = current_time.utcoffset()
        if utc_offset:
            total_seconds = int(utc_offset.total_seconds())
            hours, remainder = divmod(abs(total_seconds), 3600)
            minutes = remainder // 60
            sign = '+' if total_seconds >= 0 else '-'
            offset_str = f' (UTC{sign}{hours:02d}:{minutes:02d})'
        else:
            offset_str = ''

        # Combine timezone name with UTC offset
        timezone = f'{timezone_name}{offset_str}'

        # Output timezone and date context message for the model
        context_message = (
            f"TIMEZONE CONTEXT: The user's timezone is {timezone}. "
            f"The current date is {current_date}.\n"
            "Any dates before this are in the past, and any dates after this are in the future. "
            "When the user asks for the 'latest', 'most recent', 'today's', etc. "
            "don't assume your knowledge is up to date.\n\n"
            "CRITICAL: When spawning or resuming subagents via the Task/Agent tool, you MUST include the EXACT format above "
            "(starting with 'TIMEZONE CONTEXT: The user's timezone is...'). "
            "Do NOT condense or reformat - the phrase 'The user's timezone is' is required for validation."
        )

        try:
            json_output = _load_json_output()
            json_output.emit_additional_context('SessionStart', context_message)
        except ImportError:
            print(context_message)

        # Always exit successfully
        sys.exit(0)

    except Exception:
        # Handle all errors silently and exit successfully
        sys.exit(0)


if __name__ == '__main__':
    main()
