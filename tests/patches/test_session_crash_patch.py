"""
Unit tests for the session crash monkey-patch module.

Tests verify that the monkey-patches for BaseSession._send_response and
BaseSession.send_notification correctly handle ClosedResourceError and
BrokenResourceError when clients disconnect during tool execution.
"""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import anyio
import pytest
from mcp.shared.session import BaseSession

from app.patches.session_crash import _SEND_NOTIFICATION_ATTR
from app.patches.session_crash import _SEND_RESPONSE_ATTR
from app.patches.session_crash import _original_send_notification
from app.patches.session_crash import _original_send_response
from app.patches.session_crash import _patched_send_notification
from app.patches.session_crash import _patched_send_response
from app.patches.session_crash import apply_session_crash_patches
from app.patches.session_crash import remove_session_crash_patches


@pytest.fixture(autouse=True)
def restore_patches():
    """Ensure patches are restored after each test."""
    yield
    remove_session_crash_patches()


class TestPatchedSendResponse:
    """Tests for the _patched_send_response wrapper."""

    @pytest.mark.asyncio
    async def test_normal_call_passes_through(self):
        """Verify normal responses are delivered without interference."""
        mock_self = MagicMock()
        mock_original = AsyncMock()
        with patch(
            'app.patches.session_crash._original_send_response',
            mock_original,
        ):
            await _patched_send_response(mock_self, 'req-1', MagicMock())
            mock_original.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_catches_closed_resource_error(self):
        """Verify ClosedResourceError is caught and logged."""
        mock_self = MagicMock()
        mock_original = AsyncMock(side_effect=anyio.ClosedResourceError())
        with patch(
            'app.patches.session_crash._original_send_response',
            mock_original,
        ):
            # Should not raise
            await _patched_send_response(mock_self, 'req-1', MagicMock())

    @pytest.mark.asyncio
    async def test_catches_broken_resource_error(self):
        """Verify BrokenResourceError is caught and logged."""
        mock_self = MagicMock()
        mock_original = AsyncMock(side_effect=anyio.BrokenResourceError())
        with patch(
            'app.patches.session_crash._original_send_response',
            mock_original,
        ):
            # Should not raise
            await _patched_send_response(mock_self, 'req-1', MagicMock())

    @pytest.mark.asyncio
    async def test_logs_info_on_disconnect(self, caplog):
        """Verify INFO-level log message on client disconnect."""
        mock_self = MagicMock()
        mock_original = AsyncMock(side_effect=anyio.ClosedResourceError())
        with patch(
            'app.patches.session_crash._original_send_response',
            mock_original,
        ):
            with caplog.at_level(logging.INFO, logger='mcp.patches.session'):
                await _patched_send_response(mock_self, 'req-42', MagicMock())
            assert 'req-42' in caplog.text
            assert 'client disconnected' in caplog.text

    @pytest.mark.asyncio
    async def test_other_exceptions_propagate(self):
        """Verify non-disconnect exceptions are NOT caught."""
        mock_self = MagicMock()
        mock_original = AsyncMock(side_effect=RuntimeError('unexpected'))
        with patch(
            'app.patches.session_crash._original_send_response',
            mock_original,
        ), pytest.raises(RuntimeError, match='unexpected'):
            await _patched_send_response(mock_self, 'req-1', MagicMock())


class TestPatchedSendNotification:
    """Tests for the _patched_send_notification wrapper."""

    @pytest.mark.asyncio
    async def test_normal_call_passes_through(self):
        """Verify normal notifications are delivered without interference."""
        mock_self = MagicMock()
        mock_original = AsyncMock()
        with patch(
            'app.patches.session_crash._original_send_notification',
            mock_original,
        ):
            await _patched_send_notification(mock_self, MagicMock())
            mock_original.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_catches_closed_resource_error(self):
        """Verify ClosedResourceError is caught and logged."""
        mock_self = MagicMock()
        mock_original = AsyncMock(side_effect=anyio.ClosedResourceError())
        with patch(
            'app.patches.session_crash._original_send_notification',
            mock_original,
        ):
            await _patched_send_notification(mock_self, MagicMock())

    @pytest.mark.asyncio
    async def test_catches_broken_resource_error(self):
        """Verify BrokenResourceError is caught and logged."""
        mock_self = MagicMock()
        mock_original = AsyncMock(side_effect=anyio.BrokenResourceError())
        with patch(
            'app.patches.session_crash._original_send_notification',
            mock_original,
        ):
            await _patched_send_notification(mock_self, MagicMock())

    @pytest.mark.asyncio
    async def test_logs_info_on_disconnect(self, caplog):
        """Verify INFO-level log message on notification disconnect."""
        mock_self = MagicMock()
        mock_original = AsyncMock(side_effect=anyio.ClosedResourceError())
        with patch(
            'app.patches.session_crash._original_send_notification',
            mock_original,
        ):
            with caplog.at_level(logging.INFO, logger='mcp.patches.session'):
                await _patched_send_notification(mock_self, MagicMock())
            assert 'client disconnected' in caplog.text

    @pytest.mark.asyncio
    async def test_other_exceptions_propagate(self):
        """Verify non-disconnect exceptions are NOT caught."""
        mock_self = MagicMock()
        mock_original = AsyncMock(side_effect=ValueError('bad notification'))
        with patch(
            'app.patches.session_crash._original_send_notification',
            mock_original,
        ), pytest.raises(ValueError, match='bad notification'):
            await _patched_send_notification(mock_self, MagicMock())


class TestApplyRemovePatches:
    """Tests for apply/remove lifecycle."""

    def test_apply_patches_replaces_methods(self):
        """Verify apply_session_crash_patches replaces BaseSession methods."""
        apply_session_crash_patches()
        assert getattr(BaseSession, _SEND_RESPONSE_ATTR) is not _original_send_response
        assert getattr(BaseSession, _SEND_NOTIFICATION_ATTR) is not _original_send_notification

    def test_apply_patches_is_idempotent(self):
        """Verify calling apply twice does not error or double-patch."""
        apply_session_crash_patches()
        apply_session_crash_patches()  # Should not raise
        assert getattr(BaseSession, _SEND_RESPONSE_ATTR) is not _original_send_response

    def test_remove_patches_restores_originals(self):
        """Verify remove_session_crash_patches restores original methods."""
        apply_session_crash_patches()
        remove_session_crash_patches()
        assert getattr(BaseSession, _SEND_RESPONSE_ATTR) is _original_send_response
        assert getattr(BaseSession, _SEND_NOTIFICATION_ATTR) is _original_send_notification

    def test_apply_logs_info(self, caplog):
        """Verify apply logs an INFO message."""
        with caplog.at_level(logging.INFO, logger='mcp.patches.session'):
            apply_session_crash_patches()
        assert 'Applied session crash patches' in caplog.text

    def test_remove_logs_info(self, caplog):
        """Verify remove logs an INFO message."""
        apply_session_crash_patches()
        with caplog.at_level(logging.INFO, logger='mcp.patches.session'):
            remove_session_crash_patches()
        assert 'Removed session crash patches' in caplog.text
