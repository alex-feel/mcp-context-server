"""
Temporary monkey-patches for MCP SDK session crash on client disconnect.

Root cause: BaseSession._send_response() and send_notification() in
mcp/shared/session.py do not handle ClosedResourceError/BrokenResourceError
when _write_stream is closed after client disconnect during long-running tool
execution. The unhandled exception propagates through the anyio TaskGroup,
crashing the session.

Upstream references:
- FastMCP: https://github.com/PrefectHQ/fastmcp/issues/508
- FastMCP: https://github.com/PrefectHQ/fastmcp/issues/823
- MCP SDK: https://github.com/modelcontextprotocol/python-sdk/issues/2064
- MCP SDK PR: https://github.com/modelcontextprotocol/python-sdk/pull/2072
- MCP SDK PR: https://github.com/modelcontextprotocol/python-sdk/pull/2184

REMOVAL: Delete this module and its import in app/server.py once the MCP
Python SDK releases a version with the upstream fix. See CLAUDE.md for details.
"""

import logging
from collections.abc import Callable
from collections.abc import Coroutine

import anyio
from mcp.shared.session import BaseSession

logger = logging.getLogger('mcp.patches.session')

# Attribute names as variables prevent linter auto-fix of setattr/getattr
# (FURB118 only triggers on literal string arguments)
_SEND_RESPONSE_ATTR = '_send_response'
_SEND_NOTIFICATION_ATTR = 'send_notification'

# Store originals for restoration (used by tests and future removal).
# Using getattr avoids type-checker complaints about accessing
# private/generic class attributes directly on BaseSession.
_original_send_response: Callable[..., Coroutine[object, object, None]] = getattr(
    BaseSession, _SEND_RESPONSE_ATTR,
)
_original_send_notification: Callable[..., Coroutine[object, object, None]] = getattr(
    BaseSession, _SEND_NOTIFICATION_ATTR,
)

# Mutable container for patch state tracking
_patch_state: dict[str, bool] = {'applied': False}


async def _patched_send_response(
    self: object,
    request_id: str | int,
    response: object,
) -> None:
    """Wrap _send_response to handle client disconnect gracefully."""
    try:
        await _original_send_response(self, request_id, response)
    except (anyio.ClosedResourceError, anyio.BrokenResourceError):
        logger.info(
            'Cannot deliver response for request %s: client disconnected '
            '(write stream closed during tool execution)',
            request_id,
        )


async def _patched_send_notification(
    self: object,
    notification: object,
    related_request_id: str | int | None = None,
) -> None:
    """Wrap send_notification to handle client disconnect gracefully."""
    try:
        await _original_send_notification(self, notification, related_request_id)
    except (anyio.ClosedResourceError, anyio.BrokenResourceError):
        logger.info(
            'Cannot deliver notification: client disconnected '
            '(write stream closed)',
        )


def apply_session_crash_patches() -> None:
    """Apply monkey-patches to prevent session crashes on client disconnect.

    Patches BaseSession._send_response and BaseSession.send_notification
    to catch ClosedResourceError and BrokenResourceError. Idempotent:
    safe to call multiple times.
    """
    if _patch_state['applied']:
        logger.debug('Session crash patches already applied, skipping')
        return

    setattr(BaseSession, _SEND_RESPONSE_ATTR, _patched_send_response)
    setattr(BaseSession, _SEND_NOTIFICATION_ATTR, _patched_send_notification)
    _patch_state['applied'] = True
    logger.info('Applied session crash patches for client disconnect handling')


def remove_session_crash_patches() -> None:
    """Remove monkey-patches, restoring original BaseSession methods.

    Provided for testing and for future removal when the upstream fix
    is released.
    """
    setattr(BaseSession, _SEND_RESPONSE_ATTR, _original_send_response)
    setattr(BaseSession, _SEND_NOTIFICATION_ATTR, _original_send_notification)
    _patch_state['applied'] = False
    logger.info('Removed session crash patches, restored original BaseSession methods')
