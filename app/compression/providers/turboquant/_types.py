"""Hot-path frozen dataclasses for TurboQuant compressed representations.

Frozen + slots minimizes per-instance overhead in the hot encode/decode
path.

Wire-format layer types (:class:`MSEPayload`, :class:`IPPayload`)
carry a custom binary serialization protocol (:meth:`to_bytes` per
subtype, :func:`payload_from_bytes` module-level dispatcher) used by
the storage layer. Both subtypes use little-endian byte order and share
a common 8-byte header so the variant code can be read without subtype
context.

The wire-format discriminator is the second byte (offset 4) of the
serialized blob: ``0 = mse``, ``1 = ip``. The :func:`payload_from_bytes`
dispatcher reads this byte and delegates to the corresponding
``from_bytes`` classmethod. At the type-system level, the
:data:`CompressedPayload` alias is ``MSEPayload | IPPayload``; consumers
that handle both variants use ``match payload:`` for exhaustive
narrowing.

Batched scoring API: same-variant concatenation is exposed via
:meth:`MSEPayload.concat` and :meth:`IPPayload.concat`. The compressed
read path uses these to collapse a per-candidate provider call into a
single batched scoring invocation.
"""

import struct
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.compression.providers.turboquant._packed import assert_byte_aligned

_MAGIC = b'TQP1'
_VARIANT_CODE_MSE = 0
_VARIANT_CODE_IP = 1


@dataclass(frozen=True, slots=True)
class Codebook:
    """Precomputed Lloyd-Max codebook scaled for dimension ``d``.

    Attributes:
        centroids: 1-D float32 array of length ``2**bits``, sorted ascending.
        boundaries: 1-D float32 array of length ``2**bits - 1``, midpoints.
        bits: Bit-width.
        mse_cost: Optimal per-coordinate MSE for this codebook on ``N(0, 1)``.
    """

    centroids: NDArray[np.float32]
    boundaries: NDArray[np.float32]
    bits: int
    mse_cost: float


@dataclass(frozen=True, slots=True)
class QuantizedMSE:
    """Algorithm 1 (MSE TurboQuant) quantized representation.

    Attributes:
        packed_indices: uint8 array carrying packed b-bit centroid indices.
        norms: float32 array of shape ``(n, 1)`` -- L2 norms of input rows.
        dim: Vector dimension d.
        bits: Bit-width per coordinate.
        seed: Rotation seed used for encoding.
    """

    packed_indices: NDArray[np.uint8]
    norms: NDArray[np.float32]
    dim: int
    bits: int
    seed: int


@dataclass(frozen=True, slots=True)
class QuantizedIP:
    """Algorithm 2 (IP TurboQuant) quantized representation.

    Attributes:
        mse_data: Inner MSE quantization at ``bits - 1``.
        qjl_bits: Packed uint8 sign bits from QJL projection ``(n, n_bytes)``.
        residual_norms: float32 array of shape ``(n,)``.
        dim: Vector dimension d.
        bits: Total bit budget.
        seed: Rotation seed used for encoding.
    """

    mse_data: QuantizedMSE
    qjl_bits: NDArray[np.uint8]
    residual_norms: NDArray[np.float32]
    dim: int
    bits: int
    seed: int


def _read_block(blob: bytes, start: int) -> tuple[bytes, int]:
    """Read a length-prefixed block from ``blob`` starting at ``start``.

    Args:
        blob: Full payload bytes.
        start: Offset into ``blob`` where the 4-byte length prefix begins.

    Returns:
        Tuple of (block data bytes, offset after the block).

    Raises:
        ValueError: If the length prefix or block data overruns the blob.
    """
    if start + 4 > len(blob):
        raise ValueError('Payload truncated; missing length prefix')
    (length,) = struct.unpack('<I', blob[start:start + 4])
    data_start = start + 4
    data_end = data_start + length
    if data_end > len(blob):
        raise ValueError(
            f'Payload truncated; expected {length} data bytes from offset {data_start}',
        )
    return blob[data_start:data_end], data_end


@dataclass(frozen=True, slots=True)
class MSEPayload:
    """Algorithm 1 (MSE) wire-format payload.

    Binary layout (little-endian):
        * ``magic``: 4 bytes ``b'TQP1'``
        * ``variant_code``: 1 byte (``0`` = mse)
        * ``bits``: 1 byte (full MSE bit-width)
        * ``dim``: 2 bytes (uint16, supports ``d <= 65535``)
        * ``n_rows``: 4 bytes (uint32)
        * ``seed``: 4 bytes (uint32)
        * ``norms_bytes``: 4 bytes (uint32) followed by ``n_rows * 4`` bytes float32
        * ``packed_indices_bytes``: 4 bytes (uint32) followed by the packed indices

    Attributes:
        bits: MSE bit-width per coordinate.
        dim: Vector dimension d.
        seed: Rotation seed.
        n_rows: Number of vectors in the payload.
        norms: float32 array of L2 norms, shape ``(n_rows,)``.
        packed_indices: uint8 array of packed centroid indices.
    """

    bits: int
    dim: int
    seed: int
    n_rows: int
    norms: NDArray[np.float32]
    packed_indices: NDArray[np.uint8]

    def to_bytes(self) -> bytes:
        """Serialize to the binary layout described in the class docstring.

        Returns:
            Bytes blob suitable for storage.
        """
        header = struct.pack(
            '<4sBBHII',
            _MAGIC,
            _VARIANT_CODE_MSE,
            self.bits,
            self.dim,
            self.n_rows,
            self.seed,
        )
        norms_bytes = np.ascontiguousarray(self.norms, dtype=np.float32).tobytes()
        norms_block = struct.pack('<I', len(norms_bytes)) + norms_bytes
        packed_bytes = np.ascontiguousarray(
            self.packed_indices, dtype=np.uint8,
        ).tobytes()
        packed_block = struct.pack('<I', len(packed_bytes)) + packed_bytes
        return header + norms_block + packed_block

    @classmethod
    def from_bytes(cls, blob: bytes) -> 'MSEPayload':
        """Deserialize an MSE payload produced by :meth:`to_bytes`.

        Args:
            blob: Bytes produced by :meth:`to_bytes`.

        Returns:
            Reconstructed :class:`MSEPayload`.

        Raises:
            ValueError: If the blob is malformed, truncated, has an
                unknown magic, or carries a non-mse variant code.
        """
        header_size = struct.calcsize('<4sBBHII')
        if len(blob) < header_size:
            raise ValueError('Payload too short for TurboQuant header')
        magic, variant_code, bits, dim, n_rows, seed = struct.unpack(
            '<4sBBHII', blob[:header_size],
        )
        if magic != _MAGIC:
            raise ValueError(
                f'Invalid TurboQuant payload magic: {magic!r} (expected {_MAGIC!r})',
            )
        if variant_code != _VARIANT_CODE_MSE:
            raise ValueError(
                f'Unexpected variant code {variant_code} in MSEPayload.from_bytes '
                f'(expected {_VARIANT_CODE_MSE} for mse)',
            )

        offset = header_size
        norms_raw, offset = _read_block(blob, offset)
        packed_raw, _ = _read_block(blob, offset)

        norms_arr: NDArray[np.float32] = np.frombuffer(
            norms_raw, dtype=np.float32,
        ).copy()
        packed_arr: NDArray[np.uint8] = np.frombuffer(
            packed_raw, dtype=np.uint8,
        ).copy()

        return cls(
            bits=bits,
            dim=dim,
            seed=seed,
            n_rows=n_rows,
            norms=norms_arr,
            packed_indices=packed_arr,
        )

    @classmethod
    def concat(cls, payloads: 'Sequence[MSEPayload]') -> 'MSEPayload':
        """Concatenate same-variant MSE payloads into ONE synthetic payload.

        Concatenates ``packed_indices``, ``norms``, and sums ``n_rows``.
        All inputs must share ``(bits, dim, seed)``.

        The compressed write path emits one payload per chunk. Each
        per-row ``packed_indices`` block contains ``ceil(dim * bits / 8)``
        bytes. For the bit widths the encoder currently emits (``bits``
        in ``{2, 3, 4}``) at every shipped dimension, the per-row segment
        is byte-aligned, so concatenation along axis 0 of the uint8
        arrays preserves the per-row layout exactly. A future encoder
        relaxing the byte-alignment guarantee would require bit-level
        concatenation instead.

        Args:
            payloads: Sequence of one or more MSE payloads to combine.

        Returns:
            A single :class:`MSEPayload` whose :meth:`to_bytes` decodes
            via :func:`payload_from_bytes` into a payload equivalent to
            having encoded all input rows together.

        Raises:
            ValueError: If ``payloads`` is empty or any pair of inputs
                disagrees on ``(bits, dim, seed)``.
        """
        payloads = list(payloads)
        if not payloads:
            raise ValueError('MSEPayload.concat requires at least one payload')
        head = payloads[0]
        # Guard the per-row byte boundary that the axis-0 uint8 concat below
        # relies on. The helper raises if any future encoder relaxes the
        # alignment guarantee (e.g., adds bits=5..8 at dims where
        # (dim * bits) % 8 != 0); bit-level concat would be required there.
        assert_byte_aligned(head.dim, head.bits)
        for p in payloads[1:]:
            if (p.bits, p.dim, p.seed) != (head.bits, head.dim, head.seed):
                raise ValueError(
                    f'MSEPayload.concat requires matching (bits, dim, seed); '
                    f'got {(p.bits, p.dim, p.seed)} vs '
                    f'{(head.bits, head.dim, head.seed)}',
                )
        combined_norms = np.concatenate([p.norms for p in payloads]).astype(
            np.float32,
        )
        combined_packed = np.concatenate(
            [p.packed_indices for p in payloads],
        ).astype(np.uint8)
        return cls(
            bits=head.bits,
            dim=head.dim,
            seed=head.seed,
            n_rows=sum(p.n_rows for p in payloads),
            norms=combined_norms,
            packed_indices=combined_packed,
        )


@dataclass(frozen=True, slots=True)
class IPPayload:
    """Algorithm 2 (IP with QJL) wire-format payload.

    Binary layout (little-endian):
        * ``magic``: 4 bytes ``b'TQP1'``
        * ``variant_code``: 1 byte (``1`` = ip)
        * ``bits``: 1 byte (total IP bit budget)
        * ``dim``: 2 bytes (uint16)
        * ``n_rows``: 4 bytes (uint32)
        * ``seed``: 4 bytes (uint32)
        * ``mse_bits``: 1 byte (explicit inner-MSE bit-width)
        * ``_pad``: 3 bytes (zero-filled alignment pad)
        * ``norms_bytes``: 4 bytes (uint32) followed by ``n_rows * 4`` bytes float32
        * ``residual_norms_bytes``: 4 bytes (uint32) followed by ``n_rows * 4`` bytes float32
        * ``qjl_bytes``: 4 bytes (uint32) followed by ``n_rows * ceil(dim / 8)`` bytes
        * ``packed_indices_bytes``: 4 bytes (uint32) followed by the packed indices

    The ``mse_bits`` byte is read directly from the wire format by
    :meth:`from_bytes` instead of being computed from ``bits - 1``;
    this lets future encoder variants (for example, MSE at ``bits - 2``
    combined with 2-bit QJL) decode correctly without decoder changes.

    Attributes:
        bits: Total IP bit budget per coordinate (>= 2).
        mse_bits: Inner-MSE bit-width used for the rotated quantization.
            Currently ``bits - 1`` but stored explicitly so the encoder
            can vary the inner width independently in the future.
        dim: Vector dimension d.
        seed: Rotation seed.
        n_rows: Number of vectors in the payload.
        norms: float32 array of L2 norms, shape ``(n_rows,)``.
        packed_indices: uint8 array of packed centroid indices.
        qjl_bits: uint8 array of packed QJL sign bits, shape
            ``(n_rows, ceil(dim / 8))``.
        residual_norms: float32 array of residual norms, shape ``(n_rows,)``.
    """

    bits: int
    mse_bits: int
    dim: int
    seed: int
    n_rows: int
    norms: NDArray[np.float32]
    packed_indices: NDArray[np.uint8]
    qjl_bits: NDArray[np.uint8]
    residual_norms: NDArray[np.float32]

    def to_bytes(self) -> bytes:
        """Serialize to the binary layout described in the class docstring.

        Returns:
            Bytes blob suitable for storage.
        """
        header = struct.pack(
            '<4sBBHIIB3x',
            _MAGIC,
            _VARIANT_CODE_IP,
            self.bits,
            self.dim,
            self.n_rows,
            self.seed,
            self.mse_bits,
        )
        norms_bytes = np.ascontiguousarray(self.norms, dtype=np.float32).tobytes()
        norms_block = struct.pack('<I', len(norms_bytes)) + norms_bytes
        residual_bytes = np.ascontiguousarray(
            self.residual_norms, dtype=np.float32,
        ).tobytes()
        residual_block = struct.pack('<I', len(residual_bytes)) + residual_bytes
        qjl_bytes = np.ascontiguousarray(self.qjl_bits, dtype=np.uint8).tobytes()
        qjl_block = struct.pack('<I', len(qjl_bytes)) + qjl_bytes
        packed_bytes = np.ascontiguousarray(
            self.packed_indices, dtype=np.uint8,
        ).tobytes()
        packed_block = struct.pack('<I', len(packed_bytes)) + packed_bytes
        return header + norms_block + residual_block + qjl_block + packed_block

    @classmethod
    def from_bytes(cls, blob: bytes) -> 'IPPayload':
        """Deserialize an IP payload produced by :meth:`to_bytes`.

        Args:
            blob: Bytes produced by :meth:`to_bytes`.

        Returns:
            Reconstructed :class:`IPPayload`.

        Raises:
            ValueError: If the blob is malformed, truncated, has an
                unknown magic, or carries a non-ip variant code.
        """
        header_size = struct.calcsize('<4sBBHIIB3x')
        if len(blob) < header_size:
            raise ValueError('Payload too short for TurboQuant header')
        magic, variant_code, bits, dim, n_rows, seed, mse_bits = struct.unpack(
            '<4sBBHIIB3x', blob[:header_size],
        )
        if magic != _MAGIC:
            raise ValueError(
                f'Invalid TurboQuant payload magic: {magic!r} (expected {_MAGIC!r})',
            )
        if variant_code != _VARIANT_CODE_IP:
            raise ValueError(
                f'Unexpected variant code {variant_code} in IPPayload.from_bytes '
                f'(expected {_VARIANT_CODE_IP} for ip)',
            )

        offset = header_size
        norms_raw, offset = _read_block(blob, offset)
        residual_raw, offset = _read_block(blob, offset)
        qjl_raw, offset = _read_block(blob, offset)
        packed_raw, _ = _read_block(blob, offset)

        norms_arr: NDArray[np.float32] = np.frombuffer(
            norms_raw, dtype=np.float32,
        ).copy()
        residual_arr: NDArray[np.float32] = np.frombuffer(
            residual_raw, dtype=np.float32,
        ).copy()

        qjl_flat: NDArray[np.uint8] = np.frombuffer(qjl_raw, dtype=np.uint8).copy()
        if n_rows > 0 and qjl_flat.size % n_rows != 0:
            raise ValueError(
                f'payload corrupt: qjl_bits size {qjl_flat.size} not divisible by '
                f'n_rows {n_rows}',
            )
        qjl_arr: NDArray[np.uint8] = (
            qjl_flat.reshape(n_rows, qjl_flat.size // n_rows)
            if n_rows > 0
            else qjl_flat
        )

        packed_arr: NDArray[np.uint8] = np.frombuffer(
            packed_raw, dtype=np.uint8,
        ).copy()

        return cls(
            bits=bits,
            mse_bits=mse_bits,
            dim=dim,
            seed=seed,
            n_rows=n_rows,
            norms=norms_arr,
            packed_indices=packed_arr,
            qjl_bits=qjl_arr,
            residual_norms=residual_arr,
        )

    @classmethod
    def concat(cls, payloads: 'Sequence[IPPayload]') -> 'IPPayload':
        """Concatenate same-variant IP payloads into ONE synthetic payload.

        Concatenates ``packed_indices``, ``norms``, ``residual_norms``,
        and ``qjl_bits`` rows. All inputs must share
        ``(bits, mse_bits, dim, seed)``.

        The compressed write path emits one payload per chunk. Each
        per-row ``packed_indices`` block contains ``ceil(dim * bits / 8)``
        bytes and each per-row ``qjl_bits`` block contains
        ``ceil(dim / 8)`` bytes. For the bit widths the encoder currently
        emits (``bits`` in ``{2, 3, 4}``) at every shipped dimension, the
        per-row segment is byte-aligned, so concatenation along axis 0 of
        the uint8 arrays preserves the per-row layout exactly. A future
        encoder relaxing the byte-alignment guarantee would require
        bit-level concatenation instead.

        Args:
            payloads: Sequence of one or more IP payloads to combine.

        Returns:
            A single :class:`IPPayload` whose :meth:`to_bytes` decodes
            via :func:`payload_from_bytes` into a payload equivalent to
            having encoded all input rows together.

        Raises:
            ValueError: If ``payloads`` is empty or any pair of inputs
                disagrees on ``(bits, mse_bits, dim, seed)``.
        """
        payloads = list(payloads)
        if not payloads:
            raise ValueError('IPPayload.concat requires at least one payload')
        head = payloads[0]
        # Guard the per-row byte boundary that the axis-0 uint8 concat below
        # relies on for the inner-MSE packed_indices block. The helper raises
        # if any future encoder relaxes the alignment guarantee (e.g., adds
        # mse_bits at dims where (dim * mse_bits) % 8 != 0).
        assert_byte_aligned(head.dim, head.mse_bits)
        key_head = (head.bits, head.mse_bits, head.dim, head.seed)
        for p in payloads[1:]:
            key_p = (p.bits, p.mse_bits, p.dim, p.seed)
            if key_p != key_head:
                raise ValueError(
                    f'IPPayload.concat requires matching '
                    f'(bits, mse_bits, dim, seed); got {key_p} vs {key_head}',
                )
        combined_norms = np.concatenate([p.norms for p in payloads]).astype(
            np.float32,
        )
        combined_residual = np.concatenate(
            [p.residual_norms for p in payloads],
        ).astype(np.float32)
        # qjl_bits has shape (n_rows, n_bytes) where n_bytes is constant
        # for a fixed dim because n_bytes = ceil(dim / 8). Concatenate
        # along axis 0 so the combined array has shape
        # (sum(n_rows), n_bytes). A zero-row payload reshapes to an
        # explicit (0, ceil(dim/8)) rather than reshape(0, -1) -- numpy
        # cannot infer the -1 dimension at size 0 -- mirroring
        # MSEPayload.concat's tolerance of the documented "one or more"
        # contract; n_rows >= 1 is unaffected.
        n_bytes = (head.dim + 7) // 8
        combined_qjl = np.concatenate(
            [
                p.qjl_bits.reshape(p.n_rows, -1) if p.n_rows
                else p.qjl_bits.reshape(0, n_bytes)
                for p in payloads
            ],
            axis=0,
        ).astype(np.uint8)
        combined_packed = np.concatenate(
            [p.packed_indices for p in payloads],
        ).astype(np.uint8)
        return cls(
            bits=head.bits,
            mse_bits=head.mse_bits,
            dim=head.dim,
            seed=head.seed,
            n_rows=sum(p.n_rows for p in payloads),
            norms=combined_norms,
            packed_indices=combined_packed,
            qjl_bits=combined_qjl,
            residual_norms=combined_residual,
        )


CompressedPayload = MSEPayload | IPPayload
'''Discriminated wire-format payload alias.

The two subtypes share a common 8-byte header so the variant code
(byte at offset 4) is readable without prior knowledge of the subtype.
Use :func:`payload_from_bytes` to decode a blob whose variant is
unknown at the call site. Consumers that handle both variants should
use ``match payload:`` for exhaustive narrowing.
'''


def payload_from_bytes(blob: bytes) -> CompressedPayload:
    """Decode a wire-format payload, dispatching by variant code.

    Reads the 1-byte variant code at offset 4 and delegates to the
    corresponding subtype's ``from_bytes`` classmethod.

    Args:
        blob: Bytes produced by either :meth:`MSEPayload.to_bytes` or
            :meth:`IPPayload.to_bytes`.

    Returns:
        Either :class:`MSEPayload` or :class:`IPPayload`.

    Raises:
        ValueError: If the blob is too short, has an invalid magic, or
            carries an unknown variant code.
    """
    if len(blob) < 5:
        raise ValueError('Payload too short to read variant code')
    magic = bytes(blob[:4])
    if magic != _MAGIC:
        raise ValueError(
            f'Invalid TurboQuant payload magic: {magic!r} (expected {_MAGIC!r})',
        )
    variant_code = blob[4]
    if variant_code == _VARIANT_CODE_MSE:
        return MSEPayload.from_bytes(blob)
    if variant_code == _VARIANT_CODE_IP:
        return IPPayload.from_bytes(blob)
    raise ValueError(f'Unknown variant code: {variant_code}')
