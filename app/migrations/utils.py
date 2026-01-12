"""
Utility functions for mcp-context-server migrations.

This module contains shared utilities used across migration modules.
"""


def format_exception_message(e: Exception) -> str:
    """Format exception for error messages, handling empty str(e) cases.

    Some Python exceptions have empty string representations, resulting in
    uninformative error messages. This helper provides meaningful fallbacks.

    Args:
        e: The exception to format

    Returns:
        A non-empty error message string
    """
    msg = str(e)
    if msg:
        return msg
    # Fallback for exceptions with empty __str__
    return repr(e) or type(e).__name__ or 'Unknown error'
