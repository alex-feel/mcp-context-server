"""Regression tests for corrupt-payload detection in ``from_bytes``.

The IP wire format encodes the QJL sign-bit block as a length-prefixed
flat byte array whose size must be a multiple of ``n_rows``. A storage
corruption or wire-format mismatch that violates this invariant would
previously be silently absorbed by a 1-D fallback reshape; the decoder
must instead raise ``ValueError`` so the bug surfaces at parse time
rather than producing misinterpreted data downstream.
"""

import struct

import numpy as np
import pytest

pytest.importorskip('numpy')

from app.compression.providers.turboquant._types import IPPayload


def _build_ip_payload() -> IPPayload:
    return IPPayload(
        bits=4,
        mse_bits=3,
        dim=8,
        seed=42,
        n_rows=3,
        norms=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        packed_indices=np.array([0x12, 0x34, 0x56, 0x78], dtype=np.uint8),
        qjl_bits=np.array([[0x01], [0x02], [0x03]], dtype=np.uint8),
        residual_norms=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )


def _mutate_qjl_block_to_non_divisible(blob: bytes) -> bytes:
    """Return a copy of ``blob`` whose qjl block length is not a multiple
    of ``n_rows``.

    The IP layout: header (20 B) + norms_block + residual_block + qjl_block
    + packed_block. Each block is a 4-byte little-endian length prefix
    followed by ``length`` bytes. This helper truncates the qjl block by
    one byte and also drops one byte from its length prefix so the new
    payload remains well-formed at the block-framing level (so the parse
    reaches the reshape step) but its qjl length becomes a value
    indivisible by ``n_rows == 3`` -- i.e., ``ceil(dim / 8) * n_rows = 3``
    bytes minus 1 leaves 2 bytes, which is not divisible by 3 rows.

    Args:
        blob: Original well-formed serialized IP payload.

    Returns:
        A new bytes object identical to ``blob`` except the qjl block has
        been shortened by one byte (length prefix and data both adjusted)
        so its size is no longer a multiple of ``n_rows``.
    """
    header_size = struct.calcsize('<4sBBHIIB3x')
    offset = header_size
    (norms_len,) = struct.unpack('<I', blob[offset:offset + 4])
    offset += 4 + norms_len
    (residual_len,) = struct.unpack('<I', blob[offset:offset + 4])
    offset += 4 + residual_len

    qjl_len_pos = offset
    (qjl_len,) = struct.unpack('<I', blob[qjl_len_pos:qjl_len_pos + 4])
    qjl_data_start = qjl_len_pos + 4
    qjl_data_end = qjl_data_start + qjl_len

    # Drop one byte from the qjl payload and from its length prefix so the
    # block framing stays consistent but the resulting size is no longer
    # divisible by n_rows.
    new_qjl_len = qjl_len - 1
    new_len_prefix = struct.pack('<I', new_qjl_len)

    return (
        blob[:qjl_len_pos]
        + new_len_prefix
        + blob[qjl_data_start:qjl_data_end - 1]
        + blob[qjl_data_end:]
    )


def test_ip_payload_raises_on_qjl_size_mismatch() -> None:
    """``IPPayload.from_bytes`` raises ValueError when ``qjl_bits`` size
    is not a multiple of ``n_rows``.

    A corrupted blob (storage tampering, partial truncation, or
    wire-format mismatch) must surface as a parse-time error instead of
    silently producing a 1-D array that downstream code misinterprets.
    """
    original = _build_ip_payload()
    blob = original.to_bytes()
    corrupted = _mutate_qjl_block_to_non_divisible(blob)

    with pytest.raises(ValueError, match='qjl_bits size'):
        IPPayload.from_bytes(corrupted)
