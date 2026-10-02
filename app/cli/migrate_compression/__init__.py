"""Compression CLI for ``mcp-context-server-migrate``.

Implements the ``--compress`` and ``--decompress`` flags, which the
``app.cli.migrate`` dispatcher routes here.

Operations:
    ``compress.run_compress``:
        Read fp32 rows from ``vec_context_embeddings``, encode each with
        the active TurboQuant provider, INSERT into
        ``vec_context_embeddings_compressed``, INSERT the singleton
        provenance row, then DROP TABLE ``vec_context_embeddings``. The
        PostgreSQL HNSW index is dropped first when present.

    ``decompress.run_decompress``:
        Reverse path. Decode each compressed payload with the active
        provider, INSERT into a freshly-recreated fp32 vec table, DROP
        the compressed table, DELETE the singleton provenance row.
        Reconstruction is LOSSY (variant='mse' loses precision; variant=
        'ip' cannot recover the QJL residual).

Both operations:
    * Print a multi-line ASCII ``BACKUP REQUIRED`` warning to stderr.
    * Detect already-compressed (or already-decompressed) state and
      no-op idempotently.
    * Wrap the encode/decode + DDL in a single transaction; on failure
      the source table is left intact.
    * Honor ``--dry-run`` by rolling back the transaction after the
      probe-batch step.

The CLI is single-backend: source and destination are the same database.
For cross-backend migration users first run the integer-to-UUIDv7
migration (``mcp-context-server-migrate --source-url ... --target-url
...``, :mod:`app.cli.migrate_uuid`), then ``--compress`` on the target.

Modules (the package itself binds no names):
    ``compress``, ``decompress``: the entry points, state checks and the
        dry-run plan for each direction.
    ``compress_execution``, ``decompress_execution``: the transactional
        data movement, one branch per backend.
    ``decompress_empty``: the zero-data reverse path, which needs no fp32
        infrastructure.
    ``console``: the operator-facing stderr output and CLI settings loading.
    ``storage``: table probes, fp32 BLOB codecs, the streaming batch size and
        the PostgreSQL migration statement budget.
"""
