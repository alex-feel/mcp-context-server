"""Tests for ``MSEPayload.concat`` and ``IPPayload.concat`` batched scoring.

Verifies that same-variant concatenation produces a synthetic payload
whose decoded contents are bit-identical to concatenating the per-row
decode outputs, that mismatched metadata is rejected, that empty input
is rejected, that the byte-alignment invariant holds for the production
bit widths, and that batched scoring through the provider yields
identical results to a per-row scoring loop.
"""

import numpy as np
import pytest

pytest.importorskip('numpy')

from app.compression.providers.turboquant._types import IPPayload
from app.compression.providers.turboquant._types import MSEPayload
from app.compression.providers.turboquant._types import payload_from_bytes
from app.compression.providers.turboquant.encoder import encode
from app.compression.providers.turboquant.provider import TurboQuantProvider


def _encode_one_mse(
    vec: np.ndarray, *, bits: int = 4, seed: int = 17,
) -> MSEPayload:
    """Encode a single fp32 vector into an :class:`MSEPayload`."""
    matrix = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    payload = encode(matrix, bits=bits, variant='mse', seed=seed)
    assert isinstance(payload, MSEPayload)
    return payload


def _encode_one_ip(
    vec: np.ndarray, *, bits: int = 4, seed: int = 17,
) -> IPPayload:
    """Encode a single fp32 vector into an :class:`IPPayload`."""
    matrix = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    payload = encode(matrix, bits=bits, variant='ip', seed=seed)
    assert isinstance(payload, IPPayload)
    return payload


def _random_vectors(
    n: int, dim: int, *, seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# Round-trip and structural integrity
# ---------------------------------------------------------------------------


def test_ip_concat_tolerates_zero_row_payload() -> None:
    """IPPayload.concat handles an n_rows==0 payload (mirrors MSEPayload).

    Regression: ``p.qjl_bits.reshape(p.n_rows, -1)`` raised ValueError for a
    zero-row payload because numpy cannot infer the ``-1`` dimension at size 0.
    The empty payload now reshapes to an explicit ``(0, ceil(dim/8))``; a
    non-empty concat is unaffected. Mirrors MSEPayload.concat's tolerance of the
    documented "one or more" contract.
    """
    dim = 16
    empty = encode(np.zeros((0, dim), dtype=np.float32), bits=4, variant='ip', seed=17)
    full = encode(_random_vectors(3, dim, seed=2), bits=4, variant='ip', seed=17)
    assert isinstance(empty, IPPayload)
    assert isinstance(full, IPPayload)
    assert empty.n_rows == 0
    assert full.n_rows == 3
    # Both orderings must succeed (no ValueError) and preserve the non-empty rows.
    assert IPPayload.concat([empty, full]).n_rows == 3
    assert IPPayload.concat([full, empty]).n_rows == 3


def test_mse_concat_roundtrip() -> None:
    """MSE concat then to_bytes then payload_from_bytes preserves decode parity."""
    dim = 16
    vectors = _random_vectors(5, dim, seed=1)
    per_row = [_encode_one_mse(v) for v in vectors]
    combined = MSEPayload.concat(per_row)
    assert isinstance(combined, MSEPayload)
    assert combined.n_rows == 5
    assert combined.bits == per_row[0].bits
    assert combined.dim == per_row[0].dim
    assert combined.seed == per_row[0].seed

    blob = combined.to_bytes()
    decoded_payload = payload_from_bytes(blob)
    assert isinstance(decoded_payload, MSEPayload)
    assert decoded_payload.n_rows == 5

    provider = TurboQuantProvider(bits=4, variant='mse', seed=17, dim=dim)
    batched_matrix = provider.decode_sync(blob)
    per_row_matrix = np.concatenate(
        [provider.decode_sync(p.to_bytes()) for p in per_row], axis=0,
    )
    # decode() runs a matrix-matrix multiply whose internal BLAS reductions
    # may reorder by a few ulps when the batch size changes from 1 to N.
    # Numerical equivalence within float32 precision is the contract.
    np.testing.assert_allclose(batched_matrix, per_row_matrix, rtol=0, atol=1e-5)


def test_ip_concat_roundtrip() -> None:
    """IP concat then to_bytes then payload_from_bytes preserves all fields and decode parity."""
    dim = 16
    vectors = _random_vectors(5, dim, seed=2)
    per_row = [_encode_one_ip(v) for v in vectors]
    combined = IPPayload.concat(per_row)
    assert isinstance(combined, IPPayload)
    assert combined.n_rows == 5
    assert combined.bits == per_row[0].bits
    assert combined.mse_bits == per_row[0].mse_bits
    assert combined.dim == per_row[0].dim
    assert combined.seed == per_row[0].seed

    blob = combined.to_bytes()
    decoded_payload = payload_from_bytes(blob)
    assert isinstance(decoded_payload, IPPayload)
    assert decoded_payload.n_rows == 5
    assert decoded_payload.mse_bits == per_row[0].mse_bits
    assert decoded_payload.qjl_bits.shape[0] == 5

    provider = TurboQuantProvider(bits=4, variant='ip', seed=17, dim=dim)
    batched_matrix = provider.decode_sync(blob)
    per_row_matrix = np.concatenate(
        [provider.decode_sync(p.to_bytes()) for p in per_row], axis=0,
    )
    np.testing.assert_allclose(batched_matrix, per_row_matrix, rtol=0, atol=1e-5)


def test_ip_concat_byte_alignment() -> None:
    """Concatenated packed_indices stays byte-aligned for production bit widths.

    The IP encoder uses ``mse_bits = bits - 1`` for the inner MSE
    quantization, so each per-row ``packed_indices`` block contains
    ``dim * mse_bits / 8`` bytes (not ``dim * bits / 8``).
    """
    dim = 1024
    bits = 4
    n = 10
    vectors = _random_vectors(n, dim, seed=3)
    per_row = [_encode_one_ip(v, bits=bits) for v in vectors]
    combined = IPPayload.concat(per_row)

    expected_packed_per_row = dim * combined.mse_bits // 8
    assert combined.packed_indices.size % expected_packed_per_row == 0
    assert combined.packed_indices.size == n * expected_packed_per_row

    expected_qjl_per_row = (dim + 7) // 8
    assert combined.qjl_bits.shape == (n, expected_qjl_per_row)


def test_mse_concat_byte_alignment() -> None:
    """MSE concat preserves byte-alignment for production bit widths."""
    dim = 1024
    bits = 4
    n = 7
    vectors = _random_vectors(n, dim, seed=4)
    per_row = [_encode_one_mse(v, bits=bits) for v in vectors]
    combined = MSEPayload.concat(per_row)
    expected_packed_per_row = dim * bits // 8
    assert combined.packed_indices.size % expected_packed_per_row == 0
    assert combined.packed_indices.size == n * expected_packed_per_row


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_mse_concat_rejects_empty_sequence() -> None:
    """Empty sequence is rejected with a descriptive ValueError."""
    with pytest.raises(ValueError, match='at least one payload'):
        MSEPayload.concat([])


def test_ip_concat_rejects_empty_sequence() -> None:
    """Empty sequence is rejected with a descriptive ValueError."""
    with pytest.raises(ValueError, match='at least one payload'):
        IPPayload.concat([])


def test_mse_concat_rejects_mismatched_metadata() -> None:
    """MSE concat rejects payloads with disagreeing (bits, dim, seed)."""
    dim = 16
    p_a = _encode_one_mse(_random_vectors(1, dim, seed=5)[0], bits=4, seed=10)
    p_b = _encode_one_mse(_random_vectors(1, dim, seed=6)[0], bits=4, seed=11)
    with pytest.raises(ValueError, match='matching .bits, dim, seed.'):
        MSEPayload.concat([p_a, p_b])


def test_ip_concat_rejects_mismatched_metadata() -> None:
    """IP concat rejects payloads with disagreeing (bits, mse_bits, dim, seed)."""
    dim = 16
    p_a = _encode_one_ip(_random_vectors(1, dim, seed=7)[0], bits=4, seed=20)
    p_b = _encode_one_ip(_random_vectors(1, dim, seed=8)[0], bits=4, seed=21)
    with pytest.raises(ValueError, match='matching .bits, mse_bits, dim, seed.'):
        IPPayload.concat([p_a, p_b])


# ---------------------------------------------------------------------------
# Numerical equivalence vs the per-row scoring loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('n_candidates', [100, 1000])
def test_concat_batched_scoring_matches_per_row_loop_ip(
    n_candidates: int,
) -> None:
    """Batched IP scoring is numerically identical to a per-row loop."""
    dim = 64
    bits = 4
    seed = 17
    vectors = _random_vectors(n_candidates, dim, seed=100 + n_candidates)
    query = _random_vectors(1, dim, seed=200 + n_candidates).astype(np.float32)

    provider = TurboQuantProvider(bits=bits, variant='ip', seed=seed, dim=dim)
    per_row = [_encode_one_ip(v, bits=bits, seed=seed) for v in vectors]
    per_row_estimates = np.asarray(
        [
            float(provider.estimate_inner_product_sync(p.to_bytes(), query)[0, 0])
            for p in per_row
        ],
        dtype=np.float32,
    )

    combined = IPPayload.concat(per_row)
    batched_estimates = provider.estimate_inner_product_sync(
        combined.to_bytes(), query,
    )[0]

    np.testing.assert_allclose(
        batched_estimates, per_row_estimates, rtol=0, atol=1e-4,
    )
    top_k = min(20, n_candidates)
    per_row_top = np.argsort(-per_row_estimates)[:top_k]
    batched_top = np.argsort(-batched_estimates)[:top_k]
    np.testing.assert_array_equal(per_row_top, batched_top)


@pytest.mark.parametrize('n_candidates', [100, 1000])
def test_concat_batched_scoring_matches_per_row_loop_mse(
    n_candidates: int,
) -> None:
    """Batched MSE scoring (L2 over decode) matches the per-row decode loop."""
    dim = 64
    bits = 4
    seed = 17
    vectors = _random_vectors(n_candidates, dim, seed=300 + n_candidates)
    query = _random_vectors(1, dim, seed=400 + n_candidates).astype(np.float32)

    provider = TurboQuantProvider(bits=bits, variant='mse', seed=seed, dim=dim)
    per_row = [_encode_one_mse(v, bits=bits, seed=seed) for v in vectors]
    per_row_distances = np.asarray(
        [
            float(
                np.linalg.norm(
                    provider.decode_sync(p.to_bytes())[0] - query[0],
                ),
            )
            for p in per_row
        ],
        dtype=np.float32,
    )

    combined = MSEPayload.concat(per_row)
    decoded_matrix = provider.decode_sync(combined.to_bytes())
    diff = decoded_matrix - query[0]
    batched_distances = np.linalg.norm(diff, axis=1).astype(np.float32)

    np.testing.assert_allclose(
        batched_distances, per_row_distances, rtol=0, atol=1e-4,
    )
    top_k = min(20, n_candidates)
    per_row_top = np.argsort(per_row_distances)[:top_k]
    batched_top = np.argsort(batched_distances)[:top_k]
    np.testing.assert_array_equal(per_row_top, batched_top)
