"""Read-only access to a SQLite source database and detection of its schema shape."""

import sqlite3
from pathlib import Path
from urllib.parse import quote


def open_source_sqlite(path: str) -> sqlite3.Connection:
    """Open the source SQLite database read-only.

    Uses the URI ``mode=ro`` form so the source DB is not mutated even if
    the migration logic has a bug.

    Args:
        path: Filesystem path to the source SQLite database file.

    Returns:
        ``sqlite3.Connection`` with ``row_factory`` set to
        :class:`sqlite3.Row`.

    Raises:
        sqlite3.OperationalError: If the database cannot be opened.
    """
    abs_path = Path(path).resolve()
    if not abs_path.exists():
        raise sqlite3.OperationalError(f'source database file does not exist: {abs_path}')
    # SQLite percent-decodes URI paths before use, so the path must be
    # percent-encoded ('%', '?', '#' would be misread), and a POSIX
    # double-slash root would be parsed as a URI authority.
    posix_path = abs_path.as_posix()
    if posix_path.startswith('//'):
        posix_path = '/' + posix_path.lstrip('/')
    uri = f"file:{quote(posix_path, safe='/:')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def detect_source_id_kind(conn: sqlite3.Connection) -> str:
    """Inspect the source ``context_entries`` schema and classify the
    primary-key column.

    Args:
        conn: Read-only connection to the source database.

    Returns:
        ``"integer"`` when the source ``id`` column is declared as
        ``INTEGER`` (the integer-keyed layout) or ``"text"`` when the
        source ``id`` column is declared as ``TEXT`` (a UUIDv7-keyed
        layout that does not need migration).

    Raises:
        sqlite3.OperationalError: If ``context_entries`` does not exist
            or lacks an ``id`` column.
    """
    cursor = conn.execute("PRAGMA table_info('context_entries')")
    rows = cursor.fetchall()
    if not rows:
        raise sqlite3.OperationalError("source database has no 'context_entries' table")
    for row in rows:
        column_name = row['name']
        column_type = (row['type'] or '').upper()
        if column_name == 'id':
            if 'INT' in column_type:
                return 'integer'
            return 'text'
    raise sqlite3.OperationalError("source 'context_entries' table has no 'id' column")


def detect_optional_tables(conn: sqlite3.Connection) -> dict[str, bool]:
    """Detect which optional tables exist in the source SQLite database.

    Args:
        conn: Read-only connection to the source database.

    Returns:
        Mapping with keys ``embedding_metadata``, ``embedding_chunks``,
        ``vec_context_embeddings``, ``context_entries_fts``,
        ``image_attachments``, ``tags`` and boolean presence values.
    """
    names = (
        'embedding_metadata',
        'embedding_chunks',
        'vec_context_embeddings',
        'context_entries_fts',
        'image_attachments',
        'tags',
    )
    result: dict[str, bool] = {}
    for name in names:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
            (name,),
        )
        result[name] = cursor.fetchone() is not None
    return result


def table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return True if ``column`` is present on ``table`` in ``conn``.

    Returns:
        True iff the column exists.
    """
    cursor = conn.execute(f"PRAGMA table_info('{table}')")
    return any(row['name'] == column for row in cursor.fetchall())


def detect_source_embedding_dim(source: sqlite3.Connection) -> int | None:
    """Best-effort detection of embedding dimension from the source DB.

    Returns:
        The dimension read from the first ``embedding_metadata`` row, or
        ``None`` when the table is absent or empty.
    """
    if not table_has_column(source, 'embedding_metadata', 'dimensions'):
        return None
    cursor = source.execute('SELECT dimensions FROM embedding_metadata LIMIT 1')
    row = cursor.fetchone()
    if row is None:
        return None
    return int(row['dimensions'])
