"""pgvector fp32 index capability limit shared by every path that builds the fp32 layout.

pgvector caps HNSW (and IVFFlat) index dimensionality at 2000 for the ``vector``
type, so any code path that materializes the fp32 ``vec_context_embeddings``
table plus its ``idx_vec_context_embeddings_hnsw`` index on PostgreSQL from a
dynamic dimension must pre-flight that dimension against the cap or crash
mid-DDL. Three consumers share this module so the constraint cannot drift
per-path: the settings validator (``AppSettings.validate_pgvector_dimension_limit``),
the compression CLI's ``--decompress`` fp32 rebuild
(``app.cli.migrate_compression.run_decompress``), and the migration CLI's
PostgreSQL target auto-init (``app.cli.migrate.initialize_target_postgresql``).

This module is intentionally dependency-free: it MUST NOT import
``app.settings`` (or anything that imports it) because ``app.settings`` is one
of its consumers.
"""

# Maximum dimensionality pgvector can index (HNSW/IVFFlat) for fp32 ``vector`` columns.
PGVECTOR_INDEX_DIM_LIMIT = 2000


def exceeds_pgvector_index_dim_limit(dim: int) -> bool:
    """Return whether ``dim`` cannot be hosted by the PostgreSQL fp32 vector layout.

    A ``vector(dim)`` column itself accepts larger dimensions, but the fp32
    layout always builds an HNSW index over it, and pgvector rejects index
    builds above :data:`PGVECTOR_INDEX_DIM_LIMIT` dimensions -- so any fp32
    provisioning at such a dimension is guaranteed to fail at CREATE INDEX
    time. Compressed payloads (BYTEA) carry no such cap, and SQLite's
    sqlite-vec has no equivalent per-dimension index limit.

    Args:
        dim: Embedding dimensionality to check.

    Returns:
        True when ``dim`` exceeds :data:`PGVECTOR_INDEX_DIM_LIMIT`.
    """
    return dim > PGVECTOR_INDEX_DIM_LIMIT
