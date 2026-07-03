"""Boundary tests for the COMPRESSION_SEED upper bound.

The TurboQuant wire format packs the rotation seed into an unsigned 32-bit
field, so a seed above UINT32_MAX would pass validation, seed-lock into the
database, and then fail every subsequent encode with a raw struct error. Both
the runtime config (``CompressionSettings``) and the persisted provenance model
(``CompressionMetadata``) must reject seeds past the uint32 cap so the
misconfiguration surfaces as a clean load-time error instead of bricking writes.
"""

import pytest
from pydantic import ValidationError

from app.compression.types import CompressionMetadata
from app.settings import CompressionSettings

_UINT32_MAX = 4294967295


def test_metadata_seed_uint32_max_is_accepted() -> None:
    """A seed equal to UINT32_MAX (the wire-format cap) validates successfully."""
    meta = CompressionMetadata(
        provider='turboquant',
        bits=4,
        variant='ip',
        seed=_UINT32_MAX,
        dim=64,
    )
    assert meta.seed == _UINT32_MAX


def test_metadata_seed_above_uint32_is_rejected() -> None:
    """A seed one past UINT32_MAX raises ValidationError."""
    with pytest.raises(ValidationError):
        CompressionMetadata(
            provider='turboquant',
            bits=4,
            variant='ip',
            seed=_UINT32_MAX + 1,
            dim=64,
        )


def test_settings_seed_uint32_max_is_accepted() -> None:
    """The runtime config accepts a seed equal to the wire-format cap."""
    settings = CompressionSettings(COMPRESSION_SEED=_UINT32_MAX)
    assert settings.seed == _UINT32_MAX


def test_settings_seed_above_uint32_is_rejected() -> None:
    """The runtime config rejects a seed past the wire-format cap at load time."""
    with pytest.raises(ValidationError):
        CompressionSettings(COMPRESSION_SEED=_UINT32_MAX + 1)
