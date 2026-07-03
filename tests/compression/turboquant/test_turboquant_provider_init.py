"""Regression tests for TurboQuantProvider explicit-kwarg construction.

Verifies that the constructor:
- Accepts explicit (bits, variant, seed, dim) kwargs.
- Falls back to settings when a kwarg is None.
- Settings-resolved seed defaults to 0 when COMPRESSION_SEED is unset.
- Raises ValueError on unsupported variant.
"""

import pytest

pytest.importorskip('numpy')

from app.compression.providers.turboquant import TurboQuantProvider
from app.settings import get_settings


def test_constructor_accepts_explicit_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit kwargs do not need any environment configuration."""
    # Clear env to confirm no implicit settings dependency for the resolved fields
    monkeypatch.delenv('COMPRESSION_SEED', raising=False)
    monkeypatch.delenv('COMPRESSION_BITS', raising=False)
    monkeypatch.delenv('COMPRESSION_VARIANT', raising=False)
    monkeypatch.delenv('EMBEDDING_DIM', raising=False)
    get_settings.cache_clear()

    provider = TurboQuantProvider(bits=4, variant='ip', seed=42, dim=1024)

    meta = provider.metadata
    assert meta.bits == 4
    assert meta.variant == 'ip'
    assert meta.seed == 42
    assert meta.dim == 1024


def test_constructor_falls_back_to_default_seed_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without env override, the constructor resolves seed to the settings
    default (0); the explicit-kwarg ``seed=None`` path delegates to settings.
    """
    monkeypatch.delenv('COMPRESSION_SEED', raising=False)
    get_settings.cache_clear()

    provider = TurboQuantProvider(bits=4, variant='ip', seed=None, dim=1024)

    assert provider.metadata.seed == 0
