"""Bit-packing utilities for compact storage of b-bit quantization indices.

Packs b-bit integers into uint8 arrays for memory-efficient storage and
provides the inverse unpacking operations.
"""

from typing import cast

import numpy as np
from numpy.typing import NDArray


def assert_byte_aligned(dim: int, bits: int) -> None:
    """Assert ``(dim * bits) % 8 == 0`` for byte-aligned slicing support.

    Common embedding dims (768, 1024, 1536, 2048, 3072) at bits in
    ``(2, 3, 4)`` all satisfy this constraint. Callers that require
    sliceable payloads must verify this constraint up front.

    Args:
        dim: Vector dimension.
        bits: Bit-width per coordinate.

    Raises:
        ValueError: If ``(dim * bits) % 8 != 0``.
    """
    if (dim * bits) % 8 != 0:
        raise ValueError(
            f'dim * bits must be divisible by 8 for byte-aligned slicing; '
            f'got dim={dim}, bits={bits}, dim*bits={dim * bits}',
        )


def pack_indices(indices: NDArray[np.int32], bits: int) -> NDArray[np.uint8]:
    """Pack b-bit index values into a compact uint8 array.

    Args:
        indices: Integer array with values in ``[0, 2**bits)``. Shape ``(*, d)``.
        bits: Number of bits per index (1-8).

    Returns:
        Packed uint8 array of shape ``(ceil(n * bits / 8),)`` for
        ``n = indices.size``.

    Raises:
        ValueError: If ``bits`` is outside ``[1, 8]``.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f'bits must be in [1, 8], got {bits}')
    if bits == 8:
        return indices.reshape(-1).astype(np.uint8)

    flat = indices.reshape(-1).astype(np.int64)
    n = flat.size
    if n == 0:
        return np.zeros(0, dtype=np.uint8)

    k_arange = np.arange(bits, dtype=np.int64)
    bit_vals = ((flat[:, np.newaxis] >> k_arange) & 1).astype(np.uint8)
    bits_stream = bit_vals.reshape(-1)
    total_bits = bits_stream.size
    n_bytes = (total_bits + 7) // 8
    pad_len = n_bytes * 8 - total_bits
    if pad_len:
        bits_stream = np.concatenate(
            [bits_stream, np.zeros(pad_len, dtype=np.uint8)],
        )
    reshaped = bits_stream.reshape(n_bytes, 8)
    shifts = np.arange(8, dtype=np.int32)
    packed = (reshaped.astype(np.int32) << shifts).sum(axis=1).astype(np.uint8)
    return cast('NDArray[np.uint8]', packed)


def unpack_indices(
    packed: NDArray[np.uint8], bits: int, count: int,
) -> NDArray[np.int32]:
    """Unpack b-bit index values from a compact uint8 array.

    Args:
        packed: Packed uint8 array from :func:`pack_indices`.
        bits: Number of bits per index (1-8).
        count: Number of indices to unpack.

    Returns:
        Integer array of shape ``(count,)`` with values in
        ``[0, 2**bits)``.

    Raises:
        ValueError: If ``bits`` is outside ``[1, 8]``.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f'bits must be in [1, 8], got {bits}')
    if bits == 8:
        return packed[:count].reshape(-1).astype(np.int32)

    if count == 0:
        return np.zeros(0, dtype=np.int32)

    total_bits = count * bits
    shifts_u8 = np.arange(8, dtype=np.uint8)
    bits_from_bytes = (
        (packed[:, np.newaxis] >> shifts_u8) & 1
    ).reshape(-1)
    stream = bits_from_bytes[:total_bits].astype(np.int64)
    mat = stream.reshape(count, bits)
    powers = (1 << np.arange(bits, dtype=np.int64))
    result = (mat * powers).sum(axis=-1).astype(np.int32)
    return cast('NDArray[np.int32]', result)


def pack_bits(signs: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pack a boolean/int8 sign array into a uint8 bitfield.

    Args:
        signs: Array of 0/1 or -1/+1 values. Any shape; values are packed
            in flattened (row-major) order into a single 1-D uint8 array.

    Returns:
        Packed uint8 array of shape ``(ceil(signs.size / 8),)``.
    """
    flat = (signs.reshape(-1) > 0).astype(np.uint8)
    n = flat.size
    n_bytes = (n + 7) // 8
    padded = np.zeros(n_bytes * 8, dtype=np.uint8)
    padded[:n] = flat

    reshaped = padded.reshape(n_bytes, 8)
    shifts = np.arange(8, dtype=np.int32)
    packed = (reshaped.astype(np.int32) << shifts).sum(axis=1).astype(np.uint8)
    return cast('NDArray[np.uint8]', packed)


def pack_bits_batch(signs: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Vectorized sign packing for a batch of shape ``(n, d)``.

    Args:
        signs: 2-D (or higher) array; last axis is the sign vector.

    Returns:
        Packed uint8 array of shape ``(n, ceil(d / 8))``.

    Raises:
        ValueError: If ``signs`` has fewer than 2 dimensions.
    """
    if signs.ndim < 2:
        raise ValueError('pack_bits_batch expects signs of shape (n, d)')
    n, m = signs.shape
    flat = (signs > 0).astype(np.uint8)
    n_bytes = (m + 7) // 8
    padded_cols = n_bytes * 8
    padded = np.zeros((n, padded_cols), dtype=np.uint8)
    padded[:, :m] = flat
    reshaped = padded.reshape(n, n_bytes, 8)
    shifts = np.arange(8, dtype=np.int32)
    packed = (reshaped.astype(np.int32) << shifts).sum(axis=-1).astype(np.uint8)
    return cast('NDArray[np.uint8]', packed)


def unpack_bits(packed: NDArray[np.uint8], count: int) -> NDArray[np.float32]:
    """Unpack a uint8 bitfield into a sign array of ``+1/-1``.

    Args:
        packed: Packed uint8 array from :func:`pack_bits`.
        count: Number of sign values to unpack.

    Returns:
        Float32 array of shape ``(count,)`` with values in
        ``{-1.0, +1.0}``.
    """
    shifts = np.arange(8, dtype=np.uint8)
    unpacked = ((packed[:, np.newaxis] >> shifts) & 1).reshape(-1)[:count]
    return unpacked.astype(np.float32) * np.float32(2.0) - np.float32(1.0)


def unpack_bits_batch(
    packed: NDArray[np.uint8], count: int,
) -> NDArray[np.float32]:
    """Unpack batch-packed rows ``(n, n_bytes)`` to signs ``(n, count)``.

    Args:
        packed: Packed uint8 array of shape ``(n, n_bytes)``.
        count: Number of sign values per row.

    Returns:
        Float32 array of shape ``(n, count)`` with values in
        ``{-1.0, +1.0}``.
    """
    n = packed.shape[0]
    shifts = np.arange(8, dtype=np.uint8)
    bits = ((packed[:, :, np.newaxis] >> shifts) & 1).reshape(n, -1)[:, :count]
    return bits.astype(np.float32) * np.float32(2.0) - np.float32(1.0)
