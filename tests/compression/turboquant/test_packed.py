"""Tests for :mod:`app.compression.providers.turboquant._packed`."""

import numpy as np
import pytest

from app.compression.providers.turboquant._packed import pack_bits
from app.compression.providers.turboquant._packed import pack_indices
from app.compression.providers.turboquant._packed import unpack_bits
from app.compression.providers.turboquant._packed import unpack_indices


class TestPackIndices:
    """Round-trip tests for index packing."""

    @pytest.mark.parametrize('bits', [1, 2, 3, 4, 5, 6, 7, 8])
    def test_round_trip(self, bits: int) -> None:
        n = 128
        max_val = (1 << bits) - 1
        rng = np.random.Generator(np.random.PCG64(42))
        indices = rng.integers(0, max_val + 1, size=(n,)).astype(np.int32)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, n)
        np.testing.assert_array_equal(unpacked, indices)

    @pytest.mark.parametrize('bits', [1, 2, 3, 4])
    def test_small_counts(self, bits: int) -> None:
        for count in [1, 2, 7, 15]:
            indices = (np.arange(count, dtype=np.int32) % (1 << bits)).astype(np.int32)
            packed = pack_indices(indices, bits)
            unpacked = unpack_indices(packed, bits, count)
            np.testing.assert_array_equal(unpacked, indices)

    def test_invalid_bits(self) -> None:
        with pytest.raises(ValueError, match='bits must be in'):
            pack_indices(np.zeros(8, dtype=np.int32), bits=0)
        with pytest.raises(ValueError, match='bits must be in'):
            pack_indices(np.zeros(8, dtype=np.int32), bits=9)


class TestPackBits:
    """Round-trip tests for sign-bit packing."""

    def test_round_trip_01(self) -> None:
        n = 100
        rng = np.random.Generator(np.random.PCG64(0))
        raw = rng.integers(0, 2, size=(n,))
        signs: np.ndarray[tuple[int, ...], np.dtype[np.uint8]] = (
            np.asarray(raw, dtype=np.uint8)
        )
        packed = pack_bits(signs)
        unpacked = unpack_bits(packed, n)
        expected = signs.astype(np.float32) * np.float32(2.0) - np.float32(1.0)
        np.testing.assert_allclose(unpacked, expected)

    def test_round_trip_pm1(self) -> None:
        n = 77
        rng = np.random.Generator(np.random.PCG64(1))
        signs01 = rng.integers(0, 2, size=(n,)).astype(np.float32)
        signs = signs01 * np.float32(2.0) - np.float32(1.0)
        # pack_bits accepts uint8 0/1; cast positive entries.
        packed = pack_bits((signs > 0).astype(np.uint8))
        unpacked = unpack_bits(packed, n)
        np.testing.assert_allclose(unpacked, signs)

    def test_all_positive(self) -> None:
        signs = np.ones(16, dtype=np.uint8)
        packed = pack_bits(signs)
        unpacked = unpack_bits(packed, 16)
        assert (unpacked == 1.0).all()

    def test_all_negative(self) -> None:
        signs = np.zeros(16, dtype=np.uint8)
        packed = pack_bits(signs)
        unpacked = unpack_bits(packed, 16)
        assert (unpacked == -1.0).all()
