"""Cache reset shared by the non-default POSTGRESQL_SCHEMA integration tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from app.repositories.embedding_repository.compression_cache import _reset_compression_cache
from app.settings import get_settings


@pytest.fixture(autouse=True)
def reset_caches() -> Generator[None, None, None]:
    """Invalidate settings + compression caches around every test.

    Yields:
        ``None`` (sentinel); caches are cleared again at teardown.
    """
    get_settings.cache_clear()
    _reset_compression_cache()
    yield None
    get_settings.cache_clear()
    _reset_compression_cache()
