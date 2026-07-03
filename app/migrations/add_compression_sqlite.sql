-- Embedding compression migration for SQLite.
-- Replaces the fp32 vec0 virtual table with a bit-packed compressed table when
-- ENABLE_EMBEDDING_COMPRESSION=true.
--
-- Idempotency: DROP IF EXISTS + CREATE IF NOT EXISTS make the script safe to
-- re-run.
-- Atomicity: SQLite executescript runs the body inside a single transaction;
-- partial failures roll back.
--
-- Singleton provenance row: the compression_metadata table uses CHECK (id = 1)
-- to enforce that exactly one configuration is active per database. The seed
-- and bit width are load-bearing -- accidental mutation would corrupt every
-- compressed payload. Constraining the singleton at the SQL layer adds defense
-- in depth beyond the application-level startup validator.

-- Drop the fp32 vec0 virtual table created by the semantic-search migration.
-- When compression is enabled it is replaced wholesale; the chunking 1:N
-- mapping is preserved via embedding_chunks (which references context_id, not
-- the dropped table).
DROP TABLE IF EXISTS vec_context_embeddings;

-- Physical (non-vec0) table holding bit-packed compressed payloads.
-- One row per chunk. context_id is the public UUIDv7 hex FK to
-- context_entries(id). The payload column stores the opaque bytes produced
-- by MSEPayload.to_bytes() / IPPayload.to_bytes() (dispatched via
-- payload_from_bytes); only the matching provider/seed/bits can decode it.
CREATE TABLE IF NOT EXISTS vec_context_embeddings_compressed (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_index INTEGER NOT NULL DEFAULT 0,
    end_index INTEGER NOT NULL DEFAULT 0,
    payload BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

-- B-tree index for context_id lookups (deletes, deduplication, retrieval).
CREATE INDEX IF NOT EXISTS idx_vec_compressed_context
    ON vec_context_embeddings_compressed(context_id);

-- Singleton provenance table.
-- Bootstrap INSERT happens at first startup via the compression validator;
-- subsequent reads validate env-vs-DB consistency for seed/bits/variant/
-- provider/dim.
-- codebook_fingerprint: lowercase hex SHA-256 of the REALIZED numpy.linalg.qr
-- rotation matrix recorded at first compression. The startup validator re-derives
-- and compares it to catch a cross-host BLAS/LAPACK/CPU QR divergence (the same
-- (dim, seed) materializing a DIFFERENT rotation) before it silently corrupts
-- every decode/search. Nullable so a row written before fingerprinting still reads.
CREATE TABLE IF NOT EXISTS compression_metadata (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    provider TEXT NOT NULL,
    bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),
    variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),
    seed INTEGER NOT NULL CHECK (seed >= 0),
    dim INTEGER NOT NULL CHECK (dim > 0),
    codebook_fingerprint TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
