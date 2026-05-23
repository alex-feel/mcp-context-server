"""Recall regression for the TurboQuant compression read path.

Asserts that compressed top-K results overlap the fp32 ground-truth
top-K by at least 0.85 on a synthetic corpus where each query has a
distinguishable nearest-neighbor neighborhood. The test runs four
parametrized cells: ``(bits=2, ip)``, ``(bits=2, mse)``,
``(bits=3, ip)``, ``(bits=4, ip)``.

The cells are tested SEPARATELY because the arXiv 2504.19874v1 Theorem 1
distortion bound differs by variant. The ``bits=2 variant='ip'`` cell is
the tightest (Algorithm 2 with QJL reserves one bit for the QJL sign,
leaving bits-1 for the MSE component); ``bits=2 variant='mse'`` keeps
all two bits for MSE reconstruction.

Synthetic corpus design: for each query, ``TOP_K`` "planted" documents
are built as the query vector plus small Gaussian noise; the remaining
documents are sampled uniformly from the unit sphere. After
L2-normalization the planted documents have inner product close to 1.0
with the query, while random vectors at dimension 1024 have IP
distributed roughly N(0, 1/sqrt(1024)) -- a clear top-K separation that
mirrors real embedding behavior where ground-truth neighbors are
genuinely closer than the rest of the corpus.

Uniform-random vectors on a high-dimensional sphere have near-zero
pairwise inner product with very narrow variance, making top-K
rankings extremely sensitive to noise. The planted design avoids that
degenerate case without introducing implausible structure.
"""

from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from app.compression.providers.turboquant.decoder import decode
from app.compression.providers.turboquant.decoder import estimate_inner_product
from app.compression.providers.turboquant.encoder import encode

# Corpus parameters -- kept constant so ground truth is identical across cells.
N_QUERIES = 30
N_BACKGROUND_DOCS = 1500
DIM = 1024
TOP_K = 5
# Planted-neighbor noise level. Smaller value -> the planted docs sit
# closer to the query, making top-K easier to distinguish from background.
# 0.10 gives expected IP >= 0.99 for planted docs vs ~0 for background.
PLANT_NOISE = 0.10
SEED_CORPUS = 42
SEED_QUANT = 42

# Per-query overlap floor. Failure indicates that the quantizer is
# producing systematically worse approximations than expected.
OVERLAP_FLOOR = 0.85
MEAN_OVERLAP_FLOOR = 0.90


VariantT = Literal['mse', 'ip']


def _make_corpus() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Generate a planted-neighbor synthetic corpus.

    For each of ``N_QUERIES`` queries, ``TOP_K`` documents are built as
    ``query + PLANT_NOISE * noise`` then L2-normalized; these are the
    intended top-K. ``N_BACKGROUND_DOCS`` additional documents are
    sampled uniformly from the unit sphere. The planted documents and
    background documents are concatenated and shuffled so the layout
    does not leak the ground-truth ordering.

    Returns:
        Tuple of ``(docs, queries)``. Both arrays are float32 and
        L2-normalized along the last axis. ``docs`` has shape
        ``(N_QUERIES * TOP_K + N_BACKGROUND_DOCS, DIM)``; ``queries``
        has shape ``(N_QUERIES, DIM)``.
    """
    rng = np.random.default_rng(SEED_CORPUS)

    # Queries: unit-norm random vectors.
    queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Planted neighbors: per-query stack of (TOP_K, DIM) close to query.
    planted_per_query = []
    for q_idx in range(N_QUERIES):
        noise = rng.standard_normal((TOP_K, DIM)).astype(np.float32)
        block = queries[q_idx][None, :] + PLANT_NOISE * noise
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        planted_per_query.append(block)
    planted = np.concatenate(planted_per_query, axis=0)  # (N_QUERIES*TOP_K, DIM)

    # Background documents: unit-norm random vectors with no relation to
    # any query. With DIM=1024, the IP between two random unit vectors is
    # roughly N(0, 1/sqrt(1024)) ~= N(0, 0.031), well below the IP of any
    # planted neighbor (>= 1 - PLANT_NOISE^2 / 2 in expectation).
    background = rng.standard_normal((N_BACKGROUND_DOCS, DIM)).astype(np.float32)
    background /= np.linalg.norm(background, axis=1, keepdims=True)

    return np.concatenate([planted, background], axis=0), queries


def _fp32_top_k(
    docs: NDArray[np.float32], queries: NDArray[np.float32], k: int,
) -> dict[int, set[int]]:
    """Compute fp32 ground-truth top-k indices per query.

    Args:
        docs: Document matrix (n_docs, d).
        queries: Query matrix (n_queries, d).
        k: Number of top results per query.

    Returns:
        Mapping from query index to the set of top-k doc indices.
    """
    scores = queries @ docs.T  # (n_queries, n_docs)
    truth: dict[int, set[int]] = {}
    for q_idx in range(scores.shape[0]):
        # argpartition picks the k largest indices in arbitrary order.
        top = np.argpartition(-scores[q_idx], k - 1)[:k]
        truth[q_idx] = {int(i) for i in top}
    return truth


def _resolve_variant(name: str) -> VariantT:
    """Cast a string parametrize value to the encoder's Literal type."""
    if name == 'ip':
        return 'ip'
    if name == 'mse':
        return 'mse'
    raise ValueError(f'unknown variant: {name!r}')


@pytest.mark.integration
@pytest.mark.parametrize(
    ('bits', 'variant_name'),
    [
        (2, 'ip'),
        (2, 'mse'),
        (3, 'ip'),
        (4, 'ip'),
    ],
)
def test_recall_regression(bits: int, variant_name: str) -> None:
    """Top-K overlap with fp32 ground truth is at least OVERLAP_FLOOR.

    For variant='ip', the unbiased inner-product estimator computes the
    score per (query, doc) pair. For variant='mse', the score is the
    inner product between the query and the decoded vector (the MSE
    quantizer offers no unbiased IP estimator).
    """
    variant = _resolve_variant(variant_name)
    docs, queries = _make_corpus()
    truth = _fp32_top_k(docs, queries, TOP_K)

    # Encode each doc as its own payload (one-row payloads mirror the
    # per-chunk storage layout in vec_context_embeddings_compressed).
    payloads = [
        encode(doc[None, :], bits=bits, variant=variant, seed=SEED_QUANT)
        for doc in docs
    ]

    overlaps: list[float] = []
    failures: list[tuple[int, float]] = []
    for q_idx in range(queries.shape[0]):
        q = queries[q_idx:q_idx + 1]
        # Compute per-doc similarity scores.
        if variant == 'ip':
            scores = np.array(
                [
                    float(estimate_inner_product(payload, q)[0, 0])
                    for payload in payloads
                ],
                dtype=np.float32,
            )
        else:
            scores = np.array(
                [float(decode(payload)[0] @ q[0]) for payload in payloads],
                dtype=np.float32,
            )

        top = np.argpartition(-scores, TOP_K - 1)[:TOP_K]
        compressed_top = {int(i) for i in top}
        overlap = len(compressed_top & truth[q_idx]) / TOP_K
        overlaps.append(overlap)
        if overlap < OVERLAP_FLOOR:
            failures.append((q_idx, overlap))

    min_overlap = float(min(overlaps))
    mean_overlap = float(np.mean(overlaps))

    # Informational summary, useful when triaging regression failures.
    print(
        f'\n[recall] bits={bits} variant={variant} '
        f'min={min_overlap:.3f} mean={mean_overlap:.3f} '
        f'failures={len(failures)}/{len(overlaps)}',
    )

    assert not failures, (
        f'bits={bits} variant={variant}: '
        f'{len(failures)} queries below {OVERLAP_FLOOR} overlap '
        f'(min={min_overlap:.3f}, mean={mean_overlap:.3f}). '
        f'First 3 failures: {failures[:3]}'
    )
    assert mean_overlap >= MEAN_OVERLAP_FLOOR, (
        f'bits={bits} variant={variant}: '
        f'mean overlap {mean_overlap:.3f} below {MEAN_OVERLAP_FLOOR}'
    )
