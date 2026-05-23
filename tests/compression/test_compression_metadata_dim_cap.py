"""Boundary tests for CompressionMetadata.dim cap.

Verifies that the Pydantic validator accepts dimensions up to the
current upper cap and rejects values one past it. The cap is the only
enforcement layer for the upper bound (SQL migrations enforce only the
positive lower bound via CHECK (dim > 0)).
"""

import pytest
from pydantic import ValidationError

from app.compression.types import CompressionMetadata


def test_dim_8192_is_accepted() -> None:
    """A dimension of 8192 (current upper cap) validates successfully."""
    meta = CompressionMetadata(
        provider='turboquant',
        bits=4,
        variant='ip',
        seed=0,
        dim=8192,
    )
    assert meta.dim == 8192


def test_dim_8193_is_rejected() -> None:
    """A dimension of 8193 (one past cap) raises ValidationError."""
    with pytest.raises(ValidationError):
        CompressionMetadata(
            provider='turboquant',
            bits=4,
            variant='ip',
            seed=0,
            dim=8193,
        )
