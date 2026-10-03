"""Tests that get_context_by_ids and delete_context normalize context IDs at the tool boundary."""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

import app.tools

# Tools are plain async functions registered at server startup, so tests call them directly.
get_context_by_ids = app.tools.get_context_by_ids
delete_context = app.tools.delete_context


class TestContextIdNormalization:
    """Regression guards: boundary normalization yields canonical lowercase 32-char hex IDs.

    These tests pin the invariant that `normalize_id` runs at the tool boundary, so
    every repository call receives an ID with whitespace stripped and uppercase A-F
    folded to lowercase.
    """

    @pytest.mark.asyncio
    async def test_get_context_by_ids_normalizes_ids_before_lookup(self, mock_repositories):
        """Whitespace and uppercase in context_ids are folded before the repository lookup."""
        mock_repositories.context.get_by_ids = AsyncMock(return_value=[])

        with patch('app.tools.context.retrieve.ensure_repositories', return_value=mock_repositories):
            await get_context_by_ids(
                context_ids=['  0190ABCDEF1234567890ABCD00000D05  '],
                include_images=False,
            )

            mock_repositories.context.get_by_ids.assert_awaited_once_with(['0190abcdef1234567890abcd00000d05'])

    @pytest.mark.asyncio
    async def test_delete_context_normalizes_ids_before_delete(self, mock_repositories):
        """Whitespace and uppercase in context_ids are folded before the repository delete."""
        mock_repositories.context.delete_by_ids = AsyncMock(return_value=1)

        with patch('app.tools.context.delete.ensure_repositories', return_value=mock_repositories):
            await delete_context(
                context_ids=['  0190ABCDEF1234567890ABCD00000D05  '],
                thread_id=None,
            )

            mock_repositories.context.delete_by_ids.assert_awaited_once()
            assert mock_repositories.context.delete_by_ids.await_args.args[0] == ['0190abcdef1234567890abcd00000d05']
