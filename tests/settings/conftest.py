"""Pytest configuration for settings tests."""

import pytest

from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Clear the settings singleton before each settings test.

    Tests in this package use monkeypatch.setenv to exercise env-driven
    configuration; clearing the lru_cache ensures each test reads fresh settings
    and does not leak an env-derived singleton to later tests.
    """
    get_settings.cache_clear()
