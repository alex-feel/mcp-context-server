"""Pytest configuration for reranking tests."""

from __future__ import annotations

import pytest

from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Clear settings cache before each test.

    This ensures that tests using monkeypatch.setenv to modify
    environment variables get fresh settings without pollution
    from previous tests.
    """
    get_settings.cache_clear()
