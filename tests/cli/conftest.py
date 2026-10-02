"""Fixtures shared by the migration CLI tests."""

from collections.abc import Generator

import pytest

from app.repositories.embedding_repository.compression_cache import _reset_compression_cache
from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset the settings and compression caches around every test."""
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()
