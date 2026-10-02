"""Mock helpers shared by the batch tool tests."""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock


def make_mock_txn() -> tuple[MagicMock, object]:
    """Create a mock transaction and an async context-manager factory that yields it."""
    mock_txn = MagicMock()
    mock_txn.connection = MagicMock()
    mock_txn.backend_type = 'sqlite'

    @asynccontextmanager
    async def mock_begin_transaction():
        yield mock_txn

    return mock_txn, mock_begin_transaction
