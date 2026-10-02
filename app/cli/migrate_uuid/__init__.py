"""Command-line utility for migrating an integer-keyed context database to
the current UUIDv7-keyed schema.

This tool is opt-in: users invoke it manually on a backup of an existing
database. It is NOT auto-applied by the server.

Source database
    Any database that was created with the integer primary-key layout
    (``BIGSERIAL`` on PostgreSQL, ``INTEGER PRIMARY KEY AUTOINCREMENT`` on
    SQLite). The CLI reads from this database read-only.

Target database
    A freshly created database conforming to the current schema (``TEXT``
    primary key on SQLite, ``UUID`` primary key on PostgreSQL).

Migration behavior
    - Generates a deterministic UUIDv7 for every row from the row's
      ``created_at`` timestamp using
      :func:`app.ids.generate_id_with_timestamp`.
    - Builds an in-memory integer-to-UUIDv7 mapping table.
    - Rewrites every JSON ``metadata.references.context_ids`` array by
      mapping each integer entry through the table.
    - Copies ``text_content`` and ``summary`` verbatim. Substrings that
      resemble integer ID references inside free-form text are not
      rewritten; the migration treats free-form text as opaque content.
    - Copies tags, image attachments, embedding metadata, embedding
      chunks, and vector embeddings verbatim (only ``context_id`` is
      remapped). Embeddings are never regenerated.
    - Rebuilds the SQLite FTS5 index after data copy. On a PostgreSQL
      target whose source carries full-text search, provisions the
      generated ``text_search_vector`` column and its GIN index before
      the copy, so the column populates as rows are inserted.

Usage
    mcp-context-server-migrate \\
        --source-url sqlite:///path/to/source.db \\
        --target-url sqlite:///path/to/target.db \\
        [--dry-run] [--report report.json]
"""
