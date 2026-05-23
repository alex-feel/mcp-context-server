"""Tests for byte-alignment guards in ``MSEPayload.concat`` and
``IPPayload.concat``.

The concat helpers stitch per-row uint8 payloads via axis-0 concatenation.
Correctness of that operation requires the per-row segment length to be
a multiple of 8 bits (i.e., ``(dim * bits) % 8 == 0``). The shipped
encoder only emits ``bits in {2, 3, 4}`` at byte-aligned dims, so the
production wire format always satisfies the invariant. Should a future
encoder relax that guarantee, ``assert_byte_aligned`` must reject the
non-aligned payload up front so silently misinterpreted rows cannot
escape into the read path.
"""

import numpy as np
import pytest

pytest.importorskip('numpy')

from app.compression.providers.turboquant._types import IPPayload
from app.compression.providers.turboquant._types import MSEPayload


def test_mse_concat_rejects_non_aligned() -> None:
    """``MSEPayload.concat`` raises ValueError when ``(dim * bits) % 8 != 0``.

    Builds a synthetic MSEPayload via the dataclass constructor (bypassing
    the encoder's bit-width policy) so the alignment guard can be exercised
    independently of the encoder. ``dim=3, bits=5`` yields ``dim*bits=15``,
    which is not divisible by 8.
    """
    payload = MSEPayload(
        bits=5,
        dim=3,
        seed=0,
        n_rows=1,
        norms=np.array([1.0], dtype=np.float32),
        packed_indices=np.array([0x00, 0x00], dtype=np.uint8),
    )
    with pytest.raises(ValueError, match='dim'):
        MSEPayload.concat([payload])


def test_ip_concat_rejects_non_aligned() -> None:
    """``IPPayload.concat`` raises ValueError when
    ``(dim * mse_bits) % 8 != 0``.

    The IP variant's inner packed_indices block uses ``mse_bits`` as the
    per-coordinate width; the alignment invariant is therefore
    ``(dim * mse_bits) % 8 == 0``. ``dim=3, mse_bits=5`` violates it.
    """
    payload = IPPayload(
        bits=6,
        mse_bits=5,
        dim=3,
        seed=0,
        n_rows=1,
        norms=np.array([1.0], dtype=np.float32),
        packed_indices=np.array([0x00, 0x00], dtype=np.uint8),
        qjl_bits=np.array([[0x00]], dtype=np.uint8),
        residual_norms=np.array([0.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match='dim'):
        IPPayload.concat([payload])
