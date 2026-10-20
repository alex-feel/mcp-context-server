"""Fixtures shared by the ingest client's tests."""

from collections.abc import AsyncGenerator

import pytest_asyncio

from tests.cli.ingest._stand_in import Exchange
from tests.cli.ingest._stand_in import context_server
from tests.cli.ingest._stand_in import serving


@pytest_asyncio.fixture
async def exchange() -> AsyncGenerator[Exchange]:
    """Serve the stand-in server with every tool a sync needs."""
    async with serving(context_server()) as served:
        yield served
