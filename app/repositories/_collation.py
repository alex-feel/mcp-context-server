"""Byte-wise text ordering for cross-backend result parity.

SQLite compares TEXT with its default BINARY collation (byte order). PostgreSQL
compares TEXT with the database's locale collation, which in the shipped
``pgvector/pgvector:pg18-trixie`` image is ``en_US.utf8`` -- a locale that ranks
punctuation, case, and digits differently from raw bytes. An ``ORDER BY`` over a
free-text column therefore produces a DIFFERENT row order on the two backends
for byte-identical data.

Order divergence is not cosmetic. It changes which rows survive a ``LIMIT``
whenever the primary sort key ties (the statistics top-N lists), and it changes
the serialization of public array fields such as an entry's ``tags``, so a
client that diffs or hashes a response sees a change caused purely by the
backend swap -- including across the supported ``mcp-context-server-migrate``
cross-backend path.

``COLLATE "C"`` forces PostgreSQL to compare the same bytes SQLite compares.
The project already relies on this remedy for metadata string comparisons in
``app.query_builder``; this module is the single place the repository layer
renders it, so new ordering sites cannot drift back to locale collation.
"""


def byte_ordered_text(column: str, backend_type: str) -> str:
    """Render an ORDER BY term that sorts TEXT by bytes on either backend.

    Apply this to every free-text ``ORDER BY`` term whose order is observable:
    a public response field, or a tiebreak that decides ``LIMIT`` membership.
    It is unnecessary (and deliberately not applied) for columns whose ordering
    is already byte-wise on both backends, such as the ``UUID``/lowercase-hex
    identifier columns and integer positions.

    Args:
        column: The column or qualified column reference to sort by.
        backend_type: Backend identifier (``'sqlite'`` or ``'postgresql'``).

    Returns:
        The column unchanged on SQLite (already BINARY), or the column with an
        explicit ``COLLATE "C"`` on PostgreSQL.
    """
    if backend_type == 'postgresql':
        return f'{column} COLLATE "C"'
    return column
