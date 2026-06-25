"""Boundary types for the compression subsystem.

These types are exchanged across module boundaries (database I/O,
startup validation) and use Pydantic v2 for validation.

Hot-path types (QuantizedMSE, QuantizedIP, Codebook, MSEPayload,
IPPayload) live with their implementations as frozen/slotted
dataclasses to avoid the per-instance overhead of Pydantic models on
the encode/decode hot path.
"""

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class CompressionMetadata(BaseModel):
    """Provenance row for the singleton ``compression_metadata`` table.

    A single row of this shape is inserted into the
    ``compression_metadata`` table at first startup when embedding
    compression is enabled. On subsequent starts a startup validator
    reads this row and rejects mismatches against the environment-
    derived configuration with ``ConfigurationError`` (exit 78).

    Fields are immutable post-bootstrap; the only mutation path is the
    bootstrap INSERT.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    provider: Literal['turboquant'] = Field(
        ...,
        description='Compression provider identifier.',
    )
    bits: int = Field(
        ...,
        ge=2,
        le=4,
        description='Bits per coordinate (supported range [2, 4]).',
    )
    variant: Literal['mse', 'ip'] = Field(
        ...,
        description='Compression variant. ip uses Algorithm 2 with QJL.',
    )
    seed: int = Field(
        ...,
        ge=0,
        le=4294967295,
        description='Rotation matrix seed (load-bearing invariant). Bounded to the '
                    'unsigned 32-bit range the wire format packs it into.',
    )
    dim: int = Field(
        ...,
        gt=0,
        le=8192,
        description='Vector dimension d.',
    )
