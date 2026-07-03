"""TurboQuant compression provider (NumPy implementation).

This subpackage is split into multiple private modules (one per
algorithmic concern: types, packing, codebooks, rotation, MSE
quantization, QJL transform, IP quantization, orchestrator) to keep
each module narrowly focused on a single responsibility. Only
:class:`TurboQuantProvider`, :func:`encoder.encode`, and
:func:`decoder.decode` form the public surface; underscore-prefixed
modules are private implementation details.

Algorithm reference: arXiv:2504.19874v1
    * Algorithm 1: MSE-optimal scalar quantization
    * Algorithm 2: Inner-product quantization with QJL

Attribution: see ``THIRD_PARTY_LICENSES.md`` alongside this file.
"""

from app.compression.providers.turboquant.provider import TurboQuantProvider

__all__ = ['TurboQuantProvider']
