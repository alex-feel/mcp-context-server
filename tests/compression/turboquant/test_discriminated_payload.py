"""Regression tests for the discriminated CompressedPayload sum type.

Verifies the sum type ``CompressedPayload = MSEPayload | IPPayload``
discriminates correctly via ``isinstance`` and ``match``, that
``to_bytes`` / ``from_bytes`` round-trip per subtype, that
``payload_from_bytes`` dispatches to the correct subtype based on the
variant_code byte, and that the IP wire format carries an explicit
``mse_bits`` byte (read directly from the blob, not computed from
``bits - 1``).
"""

import struct

import numpy as np
import pytest

pytest.importorskip('numpy')

from app.compression.providers.turboquant._types import CompressedPayload
from app.compression.providers.turboquant._types import IPPayload
from app.compression.providers.turboquant._types import MSEPayload
from app.compression.providers.turboquant._types import payload_from_bytes


def _build_mse_payload() -> MSEPayload:
    return MSEPayload(
        bits=4,
        dim=8,
        seed=42,
        n_rows=3,
        norms=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        packed_indices=np.array([0x12, 0x34, 0x56, 0x78], dtype=np.uint8),
    )


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


def test_mse_payload_roundtrip() -> None:
    """MSEPayload.to_bytes -> MSEPayload.from_bytes preserves all fields."""
    original = _build_mse_payload()
    blob = original.to_bytes()
    decoded = MSEPayload.from_bytes(blob)
    assert decoded.bits == original.bits
    assert decoded.dim == original.dim
    assert decoded.seed == original.seed
    assert decoded.n_rows == original.n_rows
    np.testing.assert_array_equal(decoded.norms, original.norms)
    np.testing.assert_array_equal(decoded.packed_indices, original.packed_indices)


def test_ip_payload_roundtrip() -> None:
    """IPPayload.to_bytes -> IPPayload.from_bytes preserves all fields."""
    original = _build_ip_payload()
    blob = original.to_bytes()
    decoded = IPPayload.from_bytes(blob)
    assert decoded.bits == original.bits
    assert decoded.mse_bits == original.mse_bits
    assert decoded.dim == original.dim
    assert decoded.seed == original.seed
    assert decoded.n_rows == original.n_rows
    np.testing.assert_array_equal(decoded.norms, original.norms)
    np.testing.assert_array_equal(decoded.packed_indices, original.packed_indices)
    np.testing.assert_array_equal(decoded.qjl_bits, original.qjl_bits)
    np.testing.assert_array_equal(decoded.residual_norms, original.residual_norms)


def test_payload_from_bytes_dispatches_to_correct_subtype() -> None:
    """payload_from_bytes returns MSEPayload for variant_code=0 and IPPayload for variant_code=1."""
    mse_blob = _build_mse_payload().to_bytes()
    ip_blob = _build_ip_payload().to_bytes()
    assert isinstance(payload_from_bytes(mse_blob), MSEPayload)
    assert isinstance(payload_from_bytes(ip_blob), IPPayload)


def test_match_on_payload_subtypes_narrows_correctly() -> None:
    """match payload: case MSEPayload | IPPayload covers both arms."""
    payloads: list[CompressedPayload] = [_build_mse_payload(), _build_ip_payload()]
    visited = {'mse': False, 'ip': False}
    for payload in payloads:
        match payload:
            case MSEPayload():
                visited['mse'] = True
            case IPPayload():
                visited['ip'] = True
    assert visited == {'mse': True, 'ip': True}


def test_ip_payload_explicit_mse_bits_read_from_blob() -> None:
    """IPPayload.from_bytes reads mse_bits from the stored byte, not computed from bits."""
    # Construct an IPPayload with a non-default mse_bits value
    # (intentionally different from bits - 1 so a hypothetical computed
    # decoder would produce the wrong value). This proves the field is
    # an explicit byte in the wire format.
    payload = IPPayload(
        bits=4,
        mse_bits=2,  # explicit; differs from bits - 1 = 3
        dim=8,
        seed=42,
        n_rows=2,
        norms=np.array([1.0, 2.0], dtype=np.float32),
        packed_indices=np.array([0x12, 0x34], dtype=np.uint8),
        qjl_bits=np.array([[0x01], [0x02]], dtype=np.uint8),
        residual_norms=np.array([0.1, 0.2], dtype=np.float32),
    )
    blob = payload.to_bytes()
    decoded = IPPayload.from_bytes(blob)
    # mse_bits round-trips verbatim
    assert decoded.mse_bits == 2
    # bits also round-trips verbatim and IS DIFFERENT from mse_bits
    assert decoded.bits == 4
    assert decoded.mse_bits != decoded.bits - 1


def test_ip_payload_header_size_includes_mse_bits_byte() -> None:
    """The IP header is 4 bytes longer than the MSE header (1 mse_bits + 3 pad)."""
    mse_header_size = struct.calcsize('<4sBBHII')
    ip_header_size = struct.calcsize('<4sBBHIIB3x')
    assert ip_header_size == mse_header_size + 4


def test_payload_from_bytes_rejects_unknown_variant_code() -> None:
    """payload_from_bytes raises ValueError on unknown variant codes."""
    # Craft a blob with valid magic and a bogus variant code (2).
    blob = b'TQP1' + bytes([2])  # 5 bytes minimum
    with pytest.raises(ValueError, match='Unknown variant code'):
        payload_from_bytes(blob)


def test_payload_from_bytes_rejects_invalid_magic() -> None:
    """payload_from_bytes raises ValueError when the magic header is wrong."""
    blob = b'XXXX' + bytes([0])
    with pytest.raises(ValueError, match='Invalid TurboQuant payload magic'):
        payload_from_bytes(blob)


def test_mse_payload_from_bytes_rejects_ip_variant() -> None:
    """MSEPayload.from_bytes raises if blob carries the IP variant code."""
    ip_blob = _build_ip_payload().to_bytes()
    with pytest.raises(ValueError, match='Unexpected variant code'):
        MSEPayload.from_bytes(ip_blob)


def test_ip_payload_from_bytes_rejects_mse_variant() -> None:
    """IPPayload.from_bytes raises if blob carries the MSE variant code."""
    mse_blob = _build_mse_payload().to_bytes()
    with pytest.raises(ValueError, match='Unexpected variant code'):
        IPPayload.from_bytes(mse_blob)


def test_decode_ip_reads_stored_mse_bits_for_codebook(monkeypatch: pytest.MonkeyPatch) -> None:
    """decode() selects the inner MSE codebook from the stored mse_bits byte, not bits-1.

    Mutation guard for the R1 decoder fix: since the encoder always sets
    ``mse_bits == bits - 1``, a normal round-trip cannot distinguish the fixed
    decoder from one that derives ``bits - 1``. This constructs a payload whose
    ``mse_bits`` DIVERGES from ``bits - 1`` and asserts the decode path selects the
    inner MSE quantizer from the stored ``mse_bits``. Reverting the fix (deriving
    ``bits - 1`` via ``get_ip_quantizer``) never calls ``get_mse_quantizer`` in the
    IP branch, so ``captured`` would stay empty.
    """
    import contextlib

    from app.compression.providers.turboquant import decoder as decoder_mod

    diverged = IPPayload(
        bits=4,
        mse_bits=2,  # explicit; differs from bits - 1 = 3
        dim=8,
        seed=42,
        n_rows=2,
        norms=np.array([1.0, 2.0], dtype=np.float32),
        packed_indices=np.array([0x12, 0x34], dtype=np.uint8),
        qjl_bits=np.array([[0x01], [0x02]], dtype=np.uint8),
        residual_norms=np.array([0.1, 0.2], dtype=np.float32),
    )

    captured: list[int] = []

    class _StopError(Exception):
        pass

    def _spy(_dim: int, bits: int, _seed: int) -> object:
        captured.append(bits)
        raise _StopError

    monkeypatch.setattr(decoder_mod, 'get_mse_quantizer', _spy)
    with contextlib.suppress(_StopError):
        decoder_mod.decode(diverged)
    # The inner MSE codebook is selected from mse_bits (2), NOT bits - 1 (3).
    assert captured == [2]
