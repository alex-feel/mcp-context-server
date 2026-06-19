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

Migration behaviour
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
    - Rebuilds the SQLite FTS5 index after data copy. The PostgreSQL
      schema does not currently maintain a generated ``tsvector`` column;
      callers using the optional PostgreSQL FTS migration must rerun it
      against the target database after this CLI completes.

Usage
    mcp-context-server-migrate \\
        --source-url sqlite:///path/to/source.db \\
        --target-url sqlite:///path/to/target.db \\
        [--dry-run] [--report report.json]
"""

import argparse
import asyncio
import contextlib
import json
import logging
import re
import sqlite3
import sys
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from urllib.parse import urlparse

# UUIDv7 generation for integer-keyed rows uses the timestamp parameter of
# uuid_utils.uuid7() in UNIX seconds (with optional nanos for sub-second
# precision). Upstream tracker on the parameter's units:
# https://github.com/aminalaee/uuid-utils/issues/73
from app.ids import generate_id_with_timestamp

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses for migration state
# ---------------------------------------------------------------------------


@dataclass
class MigrationStats:
    """Counters and warning/error log for a migration run.

    Attributes:
        rows_migrated: Number of ``context_entries`` rows copied to the
            target.
        references_rewritten: Number of integer entries inside
            ``metadata.references.context_ids`` arrays that were
            successfully remapped to UUIDv7 hex strings.
        orphan_references: Number of integer entries inside
            ``metadata.references.context_ids`` arrays that did not match
            any source ``context_entries.id``; these are preserved as
            integers and a warning is logged for each.
        malformed_references: Number of rows whose ``metadata`` contained
            a ``references`` block with an unexpected shape (for example,
            a non-array ``context_ids`` value). The row's metadata is
            preserved unchanged and a warning is logged.
        tags_migrated: Number of tag rows copied.
        images_migrated: Number of ``image_attachments`` rows copied.
        embedding_metadata_migrated: Number of ``embedding_metadata`` rows
            copied.
        embedding_chunks_migrated: Number of ``embedding_chunks`` rows
            copied (SQLite).
        vec_rows_migrated: Number of ``vec_context_embeddings`` rows
            copied.
        fts_rebuilt: Whether the FTS5 index rebuild succeeded on the
            target.
        warnings: Free-form warning messages.
        errors: Free-form error messages.
    """

    rows_migrated: int = 0
    references_rewritten: int = 0
    orphan_references: int = 0
    malformed_references: int = 0
    tags_migrated: int = 0
    images_migrated: int = 0
    embedding_metadata_migrated: int = 0
    embedding_chunks_migrated: int = 0
    vec_rows_migrated: int = 0
    fts_rebuilt: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a plain-dict view suitable for :func:`json.dump`.

        Returns:
            Dictionary with the same key ordering as the dataclass field
            declaration order.
        """
        return cast(dict[str, object], asdict(self))


@dataclass
class MigrationOptions:
    """Parsed CLI arguments.

    Attributes:
        source_url: URL or path identifying the source database. Accepted
            forms: ``sqlite:///abs/path/file.db``, ``/abs/path/file.db``,
            ``postgresql://user:pass@host/db``.
        target_url: URL or path identifying the target database. Same
            forms as ``source_url``.
        dry_run: When True, run the full migration logic in memory but
            issue no INSERT statements against the target.
        report_path: Optional path. When set, write the migration
            statistics as JSON to this file at end of run.
    """

    source_url: str
    target_url: str
    dry_run: bool = False
    report_path: Path | None = None


# ---------------------------------------------------------------------------
# Backend URL parsing
# ---------------------------------------------------------------------------


def parse_backend_url(url: str) -> tuple[str, str]:
    """Classify a database URL and return ``(backend_type, address)``.

    Backend type is one of ``"sqlite"`` or ``"postgresql"``. The address
    form depends on the backend:

    - ``sqlite``: filesystem path (absolute or relative).
    - ``postgresql``: the original URL, suitable for ``asyncpg.connect``.

    Recognition rules:

    - URL starting with ``sqlite://`` or ``sqlite:`` is SQLite.
    - URL starting with ``postgresql://`` or ``postgres://`` is
      PostgreSQL.
    - URL with no scheme and a path-like value is treated as SQLite.

    Args:
        url: The database URL or filesystem path.

    Returns:
        Tuple of ``(backend_type, address)``.

    Raises:
        ValueError: If ``url`` cannot be classified.
    """
    lowered = url.lower().strip()
    if not lowered:
        raise ValueError('database URL must not be empty')
    if lowered.startswith('sqlite://'):
        path = url[len('sqlite://') :]
        if path.startswith('/') and len(path) >= 3 and path[2] == ':':
            # Windows drive form: sqlite:///C:/foo -> C:/foo
            path = path.lstrip('/')
        return ('sqlite', path)
    if lowered.startswith('sqlite:'):
        return ('sqlite', url[len('sqlite:') :])
    if lowered.startswith(('postgresql://', 'postgres://')):
        return ('postgresql', url)
    # Bare Windows absolute path (e.g. ``C:\path\db`` or ``C:/path/db``).
    # ``urlparse`` would misread the single-letter drive as a URL scheme and
    # reject it, so detect it explicitly and treat it as a SQLite filesystem
    # path -- the CLI accepts plain paths without a scheme on every platform.
    if re.match(r'^[A-Za-z]:[\\/]', url):
        return ('sqlite', url)
    parsed = urlparse(url)
    if parsed.scheme in ('', 'file'):
        if parsed.scheme == 'file':
            return ('sqlite', parsed.path)
        return ('sqlite', url)
    raise ValueError(f'Unrecognized database URL scheme: {url!r}')


_POSTGRESQL_CREDENTIAL_RE = re.compile(r'(postgres(?:ql)?://[^:@/]*):[^@/]*@', re.IGNORECASE)


def mask_credentials(url: str) -> str:
    """Mask the password portion of a PostgreSQL URL.

    SQLite paths are returned unchanged. PostgreSQL URLs of the form
    ``postgresql://user:password@host/db`` have ``password`` replaced by
    ``***``.

    Args:
        url: The original URL.

    Returns:
        Same URL with the password segment redacted.
    """
    return _POSTGRESQL_CREDENTIAL_RE.sub(r'\1:***@', url)


# ---------------------------------------------------------------------------
# Source-database connection and schema-shape detection (SQLite)
# ---------------------------------------------------------------------------


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
    uri = f'file:{abs_path.as_posix()}?mode=ro'
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def open_target_sqlite(path: str) -> sqlite3.Connection:
    """Open (creating if necessary) the target SQLite database.

    Args:
        path: Filesystem path to the target SQLite database file.

    Returns:
        Read-write ``sqlite3.Connection`` with ``row_factory`` set to
        :class:`sqlite3.Row`. Foreign-key enforcement is enabled.
    """
    abs_path = Path(path).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(abs_path))
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
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


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return True if ``column`` is present on ``table`` in ``conn``.

    Returns:
        True iff the column exists.
    """
    cursor = conn.execute(f"PRAGMA table_info('{table}')")
    return any(row['name'] == column for row in cursor.fetchall())


# ---------------------------------------------------------------------------
# ID-mapping construction
# ---------------------------------------------------------------------------


def _coerce_datetime(value: object) -> datetime:
    """Coerce SQLite-side timestamp values to :class:`datetime.datetime`.

    SQLite stores timestamps as TEXT or naive Python datetimes. The
    function accepts either form and returns a timezone-aware datetime
    (assuming UTC for naive inputs and ISO-format text).

    Args:
        value: A SQLite timestamp value (str or datetime).

    Returns:
        A timezone-aware :class:`datetime.datetime`.

    Raises:
        ValueError: If ``value`` cannot be parsed.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith('Z'):
            text = text[:-1] + '+00:00'
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            parsed = datetime.strptime(text, '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    raise ValueError(f'unsupported created_at value: {value!r}')


def build_id_mapping(source_rows: Iterable[sqlite3.Row]) -> dict[int, str]:
    """Construct the integer-to-UUIDv7 mapping table.

    For each source row, generates a UUIDv7 from the row's ``created_at``
    timestamp via :func:`app.ids.generate_id_with_timestamp`. The embedded
    48-bit timestamp field is deterministic at millisecond precision; the
    lower 74 random bits are not.

    Args:
        source_rows: Iterable of source ``context_entries`` rows
            containing at minimum the columns ``id`` (integer) and
            ``created_at`` (timestamp).

    Returns:
        Dictionary mapping each source integer ID to a 32-character
        lowercase hex UUIDv7 string.
    """
    mapping: dict[int, str] = {}
    for row in source_rows:
        source_id = int(row['id'])
        created_at = _coerce_datetime(row['created_at'])
        mapping[source_id] = generate_id_with_timestamp(created_at)
    return mapping


# ---------------------------------------------------------------------------
# metadata.references.context_ids rewrite
# ---------------------------------------------------------------------------


def _rewrite_context_ids_list(
    items: list[object],
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    row_pk: int,
) -> list[object]:
    """Rewrite a single ``context_ids`` list.

    Integer entries are remapped to UUIDv7 hex strings via
    ``id_mapping``. Strings are preserved unchanged. Booleans and other
    types are flagged as malformed but preserved.

    Args:
        items: The list pulled from ``references.context_ids``.
        id_mapping: Integer-to-UUIDv7 mapping.
        stats: Mutated to count rewrites, orphans, and malformed entries.
        row_pk: The source row's integer ID, used for log context.

    Returns:
        A new list with integers remapped where possible.
    """
    out: list[object] = []
    for element in items:
        if isinstance(element, bool):
            stats.malformed_references += 1
            stats.errors.append(
                f'row {row_pk}: references.context_ids contains a boolean entry; preserved unchanged',
            )
            out.append(element)
            continue
        if isinstance(element, int):
            mapped = id_mapping.get(element)
            if mapped is not None:
                stats.references_rewritten += 1
                out.append(mapped)
            else:
                stats.orphan_references += 1
                stats.warnings.append(
                    f'row {row_pk}: references.context_ids contains orphan integer {element}; preserved',
                )
                out.append(element)
            continue
        if isinstance(element, str):
            out.append(element)
            continue
        stats.malformed_references += 1
        stats.errors.append(
            f'row {row_pk}: references.context_ids contains non-int/non-str entry '
            f'{type(element).__name__}; preserved',
        )
        out.append(element)
    return out


def _walk_and_rewrite(
    node: object,
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    row_pk: int,
    seen: set[int],
) -> None:
    """Recursively walk ``node`` and rewrite any references.context_ids.

    Dictionaries and lists are mutated in place. A ``seen`` set of object
    ids prevents infinite recursion on self-referential structures.
    """
    obj_id = id(node)
    if obj_id in seen:
        return
    if isinstance(node, dict):
        seen.add(obj_id)
        typed_node = cast(dict[str, object], node)
        references = typed_node.get('references')
        if isinstance(references, dict):
            typed_refs = cast(dict[str, object], references)
            context_ids_value = typed_refs.get('context_ids')
            if isinstance(context_ids_value, list):
                typed_refs['context_ids'] = _rewrite_context_ids_list(
                    cast(list[object], context_ids_value),
                    id_mapping,
                    stats,
                    row_pk,
                )
            elif context_ids_value is not None:
                stats.malformed_references += 1
                stats.errors.append(
                    f'row {row_pk}: metadata.references.context_ids is not a list '
                    f'({type(context_ids_value).__name__}); preserved',
                )
        for value in typed_node.values():
            _walk_and_rewrite(value, id_mapping, stats, row_pk, seen)
    elif isinstance(node, list):
        seen.add(obj_id)
        for element in cast(list[object], node):
            _walk_and_rewrite(element, id_mapping, stats, row_pk, seen)


def rewrite_metadata_references(
    metadata_json: str | None,
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    row_pk: int,
) -> str | None:
    """Rewrite integer ``context_ids`` arrays inside the JSON metadata.

    Walks the parsed metadata structure looking for every
    ``references.context_ids`` list. Each integer entry is replaced with
    its mapped UUIDv7 hex string. Non-integer entries are preserved
    unchanged. Unmapped integers (orphans) are preserved as integers and
    counted in ``stats.orphan_references``; a warning is recorded.
    Malformed structures are preserved unchanged and counted in
    ``stats.malformed_references``.

    Args:
        metadata_json: Raw metadata JSON string from the source row, or
            ``None`` if no metadata was stored.
        id_mapping: Integer-to-UUIDv7 hex mapping.
        stats: Mutated to record rewrite counts and warning/error
            messages.
        row_pk: Source row's integer ID, used for log/error context.

    Returns:
        Re-encoded JSON string with rewritten references, or ``None``
        when the input was ``None``.
    """
    if metadata_json is None:
        return None
    try:
        parsed: object = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        stats.errors.append(f'row {row_pk}: metadata JSON parse failed ({exc}); preserved verbatim')
        return metadata_json
    _walk_and_rewrite(parsed, id_mapping, stats, row_pk, seen=set())
    return json.dumps(parsed, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Per-table copy functions (SQLite)
# ---------------------------------------------------------------------------


def copy_context_entries(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """Copy every row from source ``context_entries`` to target.

    Each source row's integer ID is replaced with the corresponding
    UUIDv7 hex from ``id_mapping``. ``text_content`` and ``summary`` are
    copied verbatim. ``metadata`` is rewritten via
    :func:`rewrite_metadata_references`.

    Args:
        source: Read-only connection to the source database.
        target: Read-write connection to the target database.
        id_mapping: Integer-to-UUIDv7 mapping.
        stats: Mutated to record ``rows_migrated``.
        dry_run: When True, no INSERT is executed.
    """
    has_summary = _table_has_column(source, 'context_entries', 'summary')
    has_content_hash = _table_has_column(source, 'context_entries', 'content_hash')
    columns = [
        'id',
        'thread_id',
        'source',
        'content_type',
        'text_content',
        'metadata',
        'created_at',
        'updated_at',
    ]
    if has_summary:
        columns.insert(6, 'summary')
    if has_content_hash:
        columns.append('content_hash')
    select_sql = f'SELECT {", ".join(columns)} FROM context_entries ORDER BY created_at ASC, id ASC'

    cursor = source.execute(select_sql)
    insert_sql = (
        'INSERT INTO context_entries '
        '(id, thread_id, source, content_type, text_content, metadata, summary, content_hash, '
        'created_at, updated_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
    )
    inserted = 0
    for row in cursor:
        source_id = int(row['id'])
        new_id = id_mapping[source_id]
        rewritten_metadata = rewrite_metadata_references(row['metadata'], id_mapping, stats, source_id)
        summary_value = row['summary'] if has_summary else None
        content_hash_value = row['content_hash'] if has_content_hash else None
        params = (
            new_id,
            row['thread_id'],
            row['source'],
            row['content_type'],
            row['text_content'],
            rewritten_metadata,
            summary_value,
            content_hash_value,
            row['created_at'],
            row['updated_at'],
        )
        if not dry_run:
            target.execute(insert_sql, params)
        inserted += 1
    stats.rows_migrated = inserted


def copy_tags(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """Copy ``tags`` rows from source to target, remapping
    ``context_entry_id``.

    The local ``tags.id`` AUTOINCREMENT counter is regenerated by the
    target schema; the original integer value is not preserved.
    """
    cursor = source.execute('SELECT context_entry_id, tag FROM tags ORDER BY id ASC')
    insert_sql = 'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)'
    inserted = 0
    for row in cursor:
        source_id = int(row['context_entry_id'])
        mapped: str | None = id_mapping.get(source_id)
        if mapped is None:
            stats.warnings.append(
                f'tags row references missing context_entry_id={source_id}; skipped',
            )
            continue
        if not dry_run:
            target.execute(insert_sql, (mapped, row['tag']))
        inserted += 1
    stats.tags_migrated = inserted


def copy_image_attachments(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """Copy ``image_attachments`` rows from source to target.

    The local ``image_attachments.id`` AUTOINCREMENT counter is
    regenerated by the target schema. Image payload columns are copied
    verbatim.
    """
    cursor = source.execute(
        'SELECT context_entry_id, image_data, mime_type, image_metadata, position, created_at '
        'FROM image_attachments ORDER BY id ASC',
    )
    insert_sql = (
        'INSERT INTO image_attachments '
        '(context_entry_id, image_data, mime_type, image_metadata, position, created_at) '
        'VALUES (?, ?, ?, ?, ?, ?)'
    )
    inserted = 0
    for row in cursor:
        source_id = int(row['context_entry_id'])
        mapped: str | None = id_mapping.get(source_id)
        if mapped is None:
            stats.warnings.append(
                f'image_attachments row references missing context_entry_id={source_id}; skipped',
            )
            continue
        if not dry_run:
            target.execute(
                insert_sql,
                (
                    mapped,
                    row['image_data'],
                    row['mime_type'],
                    row['image_metadata'],
                    row['position'],
                    row['created_at'],
                ),
            )
        inserted += 1
    stats.images_migrated = inserted


def copy_embedding_metadata(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """Copy ``embedding_metadata`` rows from source to target."""
    has_chunk_count = _table_has_column(source, 'embedding_metadata', 'chunk_count')
    columns = ['context_id', 'model_name', 'dimensions', 'created_at', 'updated_at']
    if has_chunk_count:
        columns.append('chunk_count')
    cursor = source.execute(f'SELECT {", ".join(columns)} FROM embedding_metadata')

    target_has_chunk_count = _table_has_column(target, 'embedding_metadata', 'chunk_count')
    target_columns = ['context_id', 'model_name', 'dimensions', 'created_at', 'updated_at']
    if target_has_chunk_count:
        target_columns.append('chunk_count')
    placeholders = ', '.join('?' * len(target_columns))
    insert_sql = f'INSERT INTO embedding_metadata ({", ".join(target_columns)}) VALUES ({placeholders})'

    inserted = 0
    for row in cursor:
        source_id = int(row['context_id'])
        mapped: str | None = id_mapping.get(source_id)
        if mapped is None:
            stats.warnings.append(
                f'embedding_metadata row references missing context_id={source_id}; skipped',
            )
            continue
        params: list[object] = [
            mapped,
            row['model_name'],
            row['dimensions'],
            row['created_at'],
            row['updated_at'],
        ]
        if target_has_chunk_count:
            params.append(row['chunk_count'] if has_chunk_count else 1)
        if not dry_run:
            target.execute(insert_sql, params)
        inserted += 1
    stats.embedding_metadata_migrated = inserted


def copy_embedding_chunks(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """Copy ``embedding_chunks`` rows from source to target.

    ``id`` (INTEGER) and ``vec_rowid`` (INTEGER) are preserved verbatim;
    only ``context_id`` is remapped.
    """
    cursor = source.execute(
        'SELECT id, context_id, vec_rowid, start_index, end_index, created_at '
        'FROM embedding_chunks ORDER BY id ASC',
    )
    insert_sql = (
        'INSERT INTO embedding_chunks '
        '(id, context_id, vec_rowid, start_index, end_index, created_at) '
        'VALUES (?, ?, ?, ?, ?, ?)'
    )
    inserted = 0
    for row in cursor:
        source_id = int(row['context_id'])
        mapped: str | None = id_mapping.get(source_id)
        if mapped is None:
            stats.warnings.append(
                f'embedding_chunks row references missing context_id={source_id}; skipped',
            )
            continue
        params = (
            row['id'],
            mapped,
            row['vec_rowid'],
            row['start_index'],
            row['end_index'],
            row['created_at'],
        )
        if not dry_run:
            target.execute(insert_sql, params)
        inserted += 1
    stats.embedding_chunks_migrated = inserted


def copy_vec_embeddings_sqlite(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """Copy ``vec_context_embeddings`` rows from source to target.

    Both ``rowid`` and the ``embedding`` BLOB are copied verbatim. The
    bridge from public TEXT ``context_id`` to INTEGER ``rowid`` lives in
    ``embedding_chunks.vec_rowid``; the vec0 table itself has no
    ``context_id`` column.

    Requires the sqlite-vec extension to be loaded on both connections.
    """
    cursor = source.execute('SELECT rowid, embedding FROM vec_context_embeddings ORDER BY rowid ASC')
    insert_sql = 'INSERT INTO vec_context_embeddings(rowid, embedding) VALUES (?, ?)'
    inserted = 0
    for row in cursor:
        if not dry_run:
            target.execute(insert_sql, (row['rowid'], row['embedding']))
        inserted += 1
    stats.vec_rows_migrated = inserted


def rebuild_fts_sqlite(target: sqlite3.Connection, stats: MigrationStats, dry_run: bool) -> None:
    """Rebuild the SQLite FTS5 external-content index on the target.

    Issues
    ``INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild')``.

    Skipped silently when the FTS5 virtual table does not exist on the
    target. Sets ``stats.fts_rebuilt`` to True on success.
    """
    cursor = target.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='context_entries_fts'",
    )
    if cursor.fetchone() is None:
        return
    if dry_run:
        stats.fts_rebuilt = True
        return
    try:
        target.execute("INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild')")
        stats.fts_rebuilt = True
    except sqlite3.Error as exc:
        stats.errors.append(f'FTS rebuild failed: {exc}')


# ---------------------------------------------------------------------------
# Target schema initialization (SQLite)
# ---------------------------------------------------------------------------


def _load_sqlite_vec_extension(conn: sqlite3.Connection) -> bool:
    """Attempt to load the sqlite-vec extension into ``conn``.

    Returns:
        True when loading succeeded; False when sqlite-vec is not
        available or the platform does not support extension loading.
    """
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.NotSupportedError):
        return False
    try:
        sqlite_vec.load(conn)
    except sqlite3.OperationalError:
        return False
    finally:
        with contextlib.suppress(AttributeError, sqlite3.NotSupportedError):
            conn.enable_load_extension(False)
    return True


def _read_schema_file(filename: str) -> str:
    """Read a packaged schema or migration SQL file by name.

    Returns:
        Contents of the matching file.

    Raises:
        FileNotFoundError: When ``filename`` cannot be located in the
            standard ``schemas`` or ``migrations`` directories.
    """
    candidates = [
        Path(__file__).resolve().parent.parent / 'schemas' / filename,
        Path(__file__).resolve().parent.parent / 'migrations' / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding='utf-8')
    raise FileNotFoundError(f'SQL file not found: {filename}')


def initialize_target_sqlite(
    target: sqlite3.Connection,
    optional_tables: Mapping[str, bool],
    embedding_dim: int | None,
    fts_tokenizer: str,
    stats: MigrationStats,
) -> bool:
    """Initialize the target SQLite schema and applicable migrations.

    Loads the base schema and then applies semantic-search, chunking and
    FTS migrations conditionally based on what the source database
    contained. The vec0 migration is skipped when the sqlite-vec
    extension cannot be loaded.

    Args:
        target: Read-write connection to the target SQLite database.
        optional_tables: Mapping returned by
            :func:`detect_optional_tables` on the source connection.
        embedding_dim: Embedding dimension used to template the
            semantic-search migration. Ignored when sqlite-vec is not
            available.
        fts_tokenizer: Tokenizer specification for the FTS migration
            (for example, ``"porter unicode61"``).
        stats: Mutated to record any FTS or vec0 warnings.

    Returns:
        True when the sqlite-vec extension was loaded on the target.
    """
    base_schema = _read_schema_file('sqlite_schema.sql')
    target.executescript(base_schema)

    vec_loaded = False
    if optional_tables.get('vec_context_embeddings') or optional_tables.get('embedding_metadata'):
        vec_loaded = _load_sqlite_vec_extension(target)
        if optional_tables.get('vec_context_embeddings') and not vec_loaded:
            stats.warnings.append(
                'sqlite-vec extension could not be loaded on target; '
                'vec_context_embeddings will not be copied',
            )

    if optional_tables.get('embedding_metadata') and vec_loaded:
        semantic_sql = _read_schema_file('add_semantic_search_sqlite.sql')
        dim = embedding_dim or 1024
        semantic_sql = semantic_sql.replace('{EMBEDDING_DIM}', str(dim))
        try:
            target.executescript(semantic_sql)
        except sqlite3.OperationalError as exc:
            stats.warnings.append(f'semantic-search target migration partial failure: {exc}')

    if optional_tables.get('embedding_chunks') and vec_loaded:
        chunking_sql = _read_schema_file('add_chunking_sqlite.sql')
        try:
            target.executescript(chunking_sql)
        except sqlite3.OperationalError as exc:
            stats.warnings.append(f'chunking target migration partial failure: {exc}')

    if optional_tables.get('context_entries_fts'):
        fts_sql = _read_schema_file('add_fts_sqlite.sql')
        fts_sql = fts_sql.replace('{TOKENIZER}', fts_tokenizer)
        try:
            target.executescript(fts_sql)
        except sqlite3.OperationalError as exc:
            stats.warnings.append(f'FTS target migration partial failure: {exc}')

    target.commit()
    return vec_loaded


def _detect_source_embedding_dim(source: sqlite3.Connection) -> int | None:
    """Best-effort detection of embedding dimension from the source DB.

    Returns:
        The dimension read from the first ``embedding_metadata`` row, or
        ``None`` when the table is absent or empty.
    """
    if not _table_has_column(source, 'embedding_metadata', 'dimensions'):
        return None
    cursor = source.execute('SELECT dimensions FROM embedding_metadata LIMIT 1')
    row = cursor.fetchone()
    if row is None:
        return None
    return int(row['dimensions'])


# ---------------------------------------------------------------------------
# Target empty-check
# ---------------------------------------------------------------------------


def target_already_has_data_sqlite(path: str) -> bool:
    """Return True if the target SQLite file exists AND contains
    ``context_entries`` rows.

    A target file that does not exist or that exists but has no
    ``context_entries`` table is treated as empty.

    Returns:
        True iff the target already has rows.
    """
    abs_path = Path(path).resolve()
    if not abs_path.exists():
        return False
    if abs_path.stat().st_size == 0:
        return False
    conn = sqlite3.connect(str(abs_path))
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='context_entries'",
        )
        if cursor.fetchone() is None:
            return False
        cursor = conn.execute('SELECT COUNT(*) AS c FROM context_entries')
        row = cursor.fetchone()
        if row is None:
            return False
        return int(row[0]) > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Top-level orchestration (SQLite -> SQLite)
# ---------------------------------------------------------------------------


def run_migration_sqlite_to_sqlite(options: MigrationOptions) -> MigrationStats:
    """Drive a SQLite-to-SQLite migration.

    Opens the source read-only, inspects its schema, initializes the
    target schema, builds the ID mapping, and copies rows table by
    table.

    Args:
        options: Parsed CLI options.

    Returns:
        Populated :class:`MigrationStats` instance.
    """
    stats = MigrationStats()

    _, source_address = parse_backend_url(options.source_url)
    _, target_address = parse_backend_url(options.target_url)

    if not options.dry_run and target_already_has_data_sqlite(target_address):
        stats.errors.append(
            f'target database already contains context_entries rows: {target_address}. '
            f'Recovery: if a prior run was interrupted, delete the target file and rerun; '
            f'the source database is unchanged. See the Recovering From an Interrupted Migration '
            f'section of docs/migration-v2-to-v3.md.',
        )
        return stats

    source = open_source_sqlite(source_address)
    target: sqlite3.Connection | None = None
    try:
        id_kind = detect_source_id_kind(source)
        if id_kind != 'integer':
            stats.warnings.append(
                f'source database id column is {id_kind!r}; nothing to migrate',
            )
            return stats

        optional_tables = detect_optional_tables(source)
        embedding_dim = _detect_source_embedding_dim(source)

        cursor = source.execute(
            'SELECT id, created_at FROM context_entries ORDER BY created_at ASC, id ASC',
        )
        source_rows = cursor.fetchall()

        if source_rows:
            first_created_at = _coerce_datetime(source_rows[0]['created_at'])
            if first_created_at.microsecond == 0:
                logger.info(
                    'source created_at precision appears to be seconds; '
                    'sub-second ordering will use UUIDv7 random tails',
                )

        id_mapping = build_id_mapping(source_rows)

        target = open_target_sqlite(target_address)
        initialize_target_sqlite(
            target,
            optional_tables,
            embedding_dim,
            fts_tokenizer='porter unicode61',
            stats=stats,
        )

        target.execute('BEGIN')
        try:
            copy_context_entries(source, target, id_mapping, stats, options.dry_run)
            if optional_tables.get('tags'):
                copy_tags(source, target, id_mapping, stats, options.dry_run)
            if optional_tables.get('image_attachments'):
                copy_image_attachments(source, target, id_mapping, stats, options.dry_run)
            if optional_tables.get('embedding_metadata'):
                em_cursor = target.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_metadata'",
                )
                if em_cursor.fetchone() is not None:
                    copy_embedding_metadata(source, target, id_mapping, stats, options.dry_run)
            if optional_tables.get('embedding_chunks'):
                ec_cursor = target.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_chunks'",
                )
                if ec_cursor.fetchone() is not None:
                    copy_embedding_chunks(source, target, id_mapping, stats, options.dry_run)
            if optional_tables.get('vec_context_embeddings'):
                vec_cursor = target.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='vec_context_embeddings'",
                )
                if vec_cursor.fetchone() is not None:
                    if _load_sqlite_vec_extension(source):
                        copy_vec_embeddings_sqlite(source, target, stats, options.dry_run)
                    else:
                        stats.warnings.append(
                            'sqlite-vec extension could not be loaded on source; vec rows not copied',
                        )
                else:
                    stats.warnings.append(
                        'target lacks vec_context_embeddings; vec rows not copied',
                    )
            if options.dry_run:
                target.rollback()
            else:
                target.commit()
        except Exception:
            target.rollback()
            raise

        if optional_tables.get('context_entries_fts'):
            rebuild_fts_sqlite(target, stats, options.dry_run)
            if not options.dry_run:
                target.commit()
    finally:
        source.close()
        if target is not None:
            target.close()
    return stats


# ---------------------------------------------------------------------------
# PostgreSQL paths
# ---------------------------------------------------------------------------


async def _target_pg_has_data(conn: 'asyncpg.Connection[asyncpg.Record]') -> bool:
    """Return True if the PostgreSQL target ``context_entries`` table
    has any rows. Returns False when the table does not exist.

    Returns:
        True iff the target already has rows.
    """
    exists = await conn.fetchval(
        "SELECT 1 FROM information_schema.tables WHERE table_name='context_entries' LIMIT 1",
    )
    if not exists:
        return False
    count = await conn.fetchval('SELECT COUNT(*) FROM context_entries')
    return int(count or 0) > 0


def _pg_connect_kwargs() -> dict[str, Any]:
    """Return the shared asyncpg connect kwargs for the migration CLI.

    Imported lazily so the SQLite-only migration paths never import the
    PostgreSQL backend (and therefore never require asyncpg/pgvector to be
    installed). The kwargs apply ``search_path`` (``POSTGRESQL_SCHEMA``) and
    ``statement_cache_size`` (set ``POSTGRESQL_STATEMENT_CACHE_SIZE=0`` for
    transaction-mode poolers such as the Supabase Transaction Pooler). SSL is
    carried by the DSN (``?sslmode=...``), parsed natively by asyncpg.

    Returns:
        Mapping suitable for spreading into ``asyncpg.connect(dsn, **kwargs)``.
    """
    from app.backends.postgresql_backend import build_asyncpg_connect_kwargs
    from app.settings import get_settings

    return build_asyncpg_connect_kwargs(get_settings())


async def _pg_table_exists(conn: 'asyncpg.Connection[asyncpg.Record]', table_name: str) -> bool:
    """Return True if ``table_name`` exists in the connection's current schema.

    Uses ``current_schema()`` so the probe honors the ``search_path`` applied by
    :func:`_pg_connect_kwargs` (correct for non-default ``POSTGRESQL_SCHEMA``).

    Returns:
        True iff the table exists in the resolved schema.
    """
    result = await conn.fetchval(
        'SELECT EXISTS (SELECT 1 FROM information_schema.tables '
        'WHERE table_schema = current_schema() AND table_name = $1)',
        table_name,
    )
    return bool(result)


async def _detect_source_embedding_dim_pg(conn: 'asyncpg.Connection[asyncpg.Record]') -> int | None:
    """Best-effort detection of the embedding dimension from a PostgreSQL source.

    Mirrors :func:`_detect_source_embedding_dim` (the SQLite detector). Guards the
    read behind table existence so a source that never enabled semantic search
    (no ``embedding_metadata`` table) yields ``None`` instead of raising.

    Returns:
        The dimension from the first ``embedding_metadata`` row, or ``None`` when
        the table is absent or empty.
    """
    if not await _pg_table_exists(conn, 'embedding_metadata'):
        return None
    row = await conn.fetchval('SELECT dimensions FROM embedding_metadata LIMIT 1')
    return int(row) if row is not None else None


async def copy_embedding_metadata_pg(
    source: 'asyncpg.Connection[asyncpg.Record]',
    target: 'asyncpg.Connection[asyncpg.Record]',
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """Copy ``embedding_metadata`` rows from a PostgreSQL source to a PostgreSQL target.

    Mirrors :func:`copy_embedding_metadata` (the SQLite path) but uses
    asyncpg placeholders, native UUID binding, and asyncpg's ``fetch``
    cursor. The source ``context_id`` is a BIGINT (integer-keyed v2
    schema); the target ``context_id`` is a UUID (v3 schema). Mapping
    is applied via ``id_mapping``.

    Args:
        source: asyncpg connection to the PostgreSQL source database.
        target: asyncpg connection to the PostgreSQL target database.
        id_mapping: BIGINT-to-UUID mapping built from the source
            ``context_entries.id`` -> ``created_at`` rows.
        stats: Mutated to record ``embedding_metadata_migrated`` and
            warnings.
        dry_run: When True, skip INSERTs (counters still increment).
    """
    has_chunk_count_src = await source.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'embedding_metadata' AND column_name = 'chunk_count'
        )
        ''',
    )
    src_columns = ['context_id', 'model_name', 'dimensions', 'created_at', 'updated_at']
    if has_chunk_count_src:
        src_columns.append('chunk_count')
    select_sql = f'SELECT {", ".join(src_columns)} FROM embedding_metadata ORDER BY context_id ASC'
    rows = await source.fetch(select_sql)

    has_chunk_count_tgt = await target.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'embedding_metadata' AND column_name = 'chunk_count'
        )
        ''',
    )
    tgt_columns = ['context_id', 'model_name', 'dimensions', 'created_at', 'updated_at']
    if has_chunk_count_tgt:
        tgt_columns.append('chunk_count')
    # First column is cast to ::uuid; remaining columns use unadorned $N.
    placeholders = ['$1::uuid'] + [f'${i + 2}' for i in range(len(tgt_columns) - 1)]
    insert_sql = (
        f'INSERT INTO embedding_metadata ({", ".join(tgt_columns)}) '
        f'VALUES ({", ".join(placeholders)})'
    )

    inserted = 0
    for row in rows:
        source_id = int(row['context_id'])
        mapped = id_mapping.get(source_id)
        if mapped is None:
            stats.warnings.append(
                f'embedding_metadata row references missing context_id={source_id}; skipped',
            )
            continue
        params: list[object] = [
            mapped,
            row['model_name'],
            row['dimensions'],
            row['created_at'],
            row['updated_at'],
        ]
        if has_chunk_count_tgt:
            params.append(row['chunk_count'] if has_chunk_count_src else 1)
        if not dry_run:
            await target.execute(insert_sql, *params)
        inserted += 1
    stats.embedding_metadata_migrated = inserted


async def copy_vec_embeddings_pg(
    source: 'asyncpg.Connection[asyncpg.Record]',
    target: 'asyncpg.Connection[asyncpg.Record]',
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """Copy ``vec_context_embeddings`` rows from a PostgreSQL source to a PostgreSQL target.

    Only ``context_id`` is remapped (BIGINT -> UUID). The ``embedding``
    pgvector column is copied verbatim; the source must have pgvector
    installed and the target must have ``vec_context_embeddings``
    initialized. Probes both source and target for the chunking
    migration's ``start_index``/``end_index`` columns (added by
    ``add_chunking_postgresql.sql``); when present on both sides, the
    columns are copied through.

    Args:
        source: asyncpg connection to the PostgreSQL source database.
        target: asyncpg connection to the PostgreSQL target database.
        id_mapping: BIGINT-to-UUID mapping built from the source
            ``context_entries.id`` -> ``created_at`` rows.
        stats: Mutated to record ``vec_rows_migrated`` and warnings.
        dry_run: When True, skip INSERTs (counters still increment).
    """
    target_table_exists = await target.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'vec_context_embeddings'
        )
        ''',
    )
    if not target_table_exists:
        stats.warnings.append(
            'target PostgreSQL database has no vec_context_embeddings table; '
            'fp32 vec rows not copied (initialize the target schema first)',
        )
        return

    has_boundaries_src = await source.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'vec_context_embeddings' AND column_name = 'start_index'
        )
        ''',
    )
    has_boundaries_tgt = await target.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'vec_context_embeddings' AND column_name = 'start_index'
        )
        ''',
    )

    if has_boundaries_src and has_boundaries_tgt:
        select_sql = (
            'SELECT context_id, embedding, start_index, end_index '
            'FROM vec_context_embeddings ORDER BY context_id ASC'
        )
        insert_sql = (
            'INSERT INTO vec_context_embeddings '
            '(context_id, embedding, start_index, end_index) '
            'VALUES ($1::uuid, $2, $3, $4)'
        )
    else:
        select_sql = (
            'SELECT context_id, embedding FROM vec_context_embeddings '
            'ORDER BY context_id ASC'
        )
        insert_sql = (
            'INSERT INTO vec_context_embeddings (context_id, embedding) '
            'VALUES ($1::uuid, $2)'
        )
        if has_boundaries_src and not has_boundaries_tgt:
            stats.warnings.append(
                'source has start_index/end_index columns but target does not; '
                'chunk boundaries not copied (run the chunking migration on the target first)',
            )

    rows = await source.fetch(select_sql)
    inserted = 0
    for row in rows:
        source_id = int(row['context_id'])
        mapped = id_mapping.get(source_id)
        if mapped is None:
            stats.warnings.append(
                f'vec_context_embeddings row references missing context_id={source_id}; skipped',
            )
            continue
        if not dry_run:
            if has_boundaries_src and has_boundaries_tgt:
                await target.execute(
                    insert_sql,
                    mapped, row['embedding'], row['start_index'], row['end_index'],
                )
            else:
                await target.execute(insert_sql, mapped, row['embedding'])
        inserted += 1
    stats.vec_rows_migrated = inserted


async def initialize_target_postgresql(
    target_url: str,
    *,
    embedding_dim: int | None,
    with_semantic: bool,
    stats: MigrationStats,
) -> None:
    """Auto-initialize a PostgreSQL target's v3 schema, mirroring
    :func:`initialize_target_sqlite`.

    Applies the base schema and the migrations the live server applies at
    startup, in the same order, against a backend built from ``target_url``:
    ``init_database`` -> (optional semantic search) -> jsonb_merge_patch ->
    function search_path -> (optional chunking). The embedding-compression
    migration and the seed-locked provenance validator are NEVER run here, so
    the target retains the fp32 layout; compression is a separate, explicit
    ``--compress`` step.

    The semantic and chunking migrations are invoked with ``force=True`` so the
    fp32 vector layout is created regardless of the CLI process's
    ``ENABLE_EMBEDDING_GENERATION`` value (the server-side gate), and
    ``apply_semantic_search_migration``
    receives the SOURCE-detected ``embedding_dim`` so the target vector column
    width matches the data being copied (mirrors the SQLite path's
    ``_detect_source_embedding_dim`` -> ``initialize_target_sqlite`` flow).

    Args:
        target_url: asyncpg DSN for the target database.
        embedding_dim: SOURCE embedding dimension, templated into the semantic
            vector column. Ignored when ``with_semantic`` is False.
        with_semantic: When True, also create the semantic-search and chunking
            layout (PG->PG migrations that copy embeddings). When False (a
            cross-backend migration that drops embeddings), only the base schema
            and the PostgreSQL helper functions are created; the server creates
            the vector layout later, at the operator's configured dimension,
            when re-embedding.
        stats: Mutated to record an informational warning describing the
            auto-init.

    Raises:
        RuntimeError: If the pgvector extension cannot be created (for example,
            insufficient privileges on a managed service such as Supabase).
    """
    import asyncpg

    from app.backends import create_backend
    from app.migrations.chunking import apply_chunking_migration
    from app.migrations.semantic import apply_function_search_path_migration
    from app.migrations.semantic import apply_jsonb_merge_patch_migration
    from app.migrations.semantic import apply_semantic_search_migration
    from app.startup import init_database

    pg_kwargs = _pg_connect_kwargs()

    from app.settings import get_settings

    schema = get_settings().storage.postgresql_schema

    # The pgvector extension must exist before the backend pool is created: the
    # pool's per-connection init registers the vector type and raises when the
    # extension is absent (whenever the pgvector package is installed). The
    # target schema must also exist before the schema-qualified function DDL in
    # the base schema / migrations runs (CREATE FUNCTION "<schema>".update_...);
    # PostgreSQL does not auto-create a non-default schema. Both are created
    # idempotently up front; surface a clear, actionable error on managed
    # services where DDL privileges are restricted.
    ext_conn = await asyncpg.connect(target_url, **pg_kwargs)
    try:
        await ext_conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await ext_conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    except asyncpg.InsufficientPrivilegeError as exc:
        raise RuntimeError(
            'Cannot initialize the target database (insufficient privileges to '
            'CREATE EXTENSION vector or CREATE SCHEMA). Enable pgvector and the '
            f'"{schema}" schema first, then rerun: on Supabase use Dashboard -> '
            'Database -> Extensions -> vector; on self-hosted PostgreSQL run '
            '"CREATE EXTENSION vector;" and "CREATE SCHEMA <schema>;" as a superuser.',
        ) from exc
    finally:
        await ext_conn.close()

    backend = create_backend(backend_type='postgresql', connection_string=target_url)
    await backend.initialize()
    try:
        await init_database(backend=backend)
        if with_semantic:
            await apply_semantic_search_migration(backend, force=True, embedding_dim=embedding_dim)
        await apply_jsonb_merge_patch_migration(backend)
        await apply_function_search_path_migration(backend)
        if with_semantic:
            await apply_chunking_migration(backend, force=True)
    finally:
        await backend.shutdown()

    stats.warnings.append(
        'auto-initialized target PostgreSQL schema '
        f'(semantic_search={"yes" if with_semantic else "no"}, '
        f'embedding_dim={embedding_dim if with_semantic else "n/a"})',
    )


async def run_migration_postgresql(options: MigrationOptions) -> MigrationStats:
    """Drive a PostgreSQL-to-PostgreSQL migration.

    The target PostgreSQL database must already exist (the CLI does not run
    ``CREATE DATABASE``), but its schema is auto-initialized when absent via
    :func:`initialize_target_postgresql` -- the user no longer has to start the
    server once against the target to create the schema. When the source carries
    embeddings, the target is built with the fp32 vector layout (compression is
    never enabled here); enable compression afterward with the separate
    ``--compress`` step. If a pre-existing target lacks the fp32
    ``vec_context_embeddings`` table while the source has embeddings (for example
    a target initialized with compression enabled), the migration aborts with a
    recorded error rather than silently dropping the vectors.

    Args:
        options: Parsed CLI options.

    Returns:
        Populated :class:`MigrationStats` instance.
    """
    import asyncpg

    stats = MigrationStats()
    pg_kwargs = _pg_connect_kwargs()
    source_conn = await asyncpg.connect(options.source_url, **pg_kwargs)
    target_conn = await asyncpg.connect(options.target_url, **pg_kwargs)
    try:
        await source_conn.execute('BEGIN TRANSACTION READ ONLY')

        if not options.dry_run and await _target_pg_has_data(target_conn):
            stats.errors.append(
                'target PostgreSQL database already contains context_entries rows. '
                'Recovery: if a prior run was interrupted, drop and recreate the target database '
                '(or pass a different --target-url) and rerun; the source database is unchanged. '
                'See the Recovering From an Interrupted Migration section of docs/migration-v2-to-v3.md.',
            )
            return stats

        id_column_type = await source_conn.fetchval(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name='context_entries' AND column_name='id'",
        )
        if id_column_type is None:
            stats.errors.append("source PostgreSQL database lacks 'context_entries.id' column")
            return stats
        if str(id_column_type).lower() in ('uuid', 'text', 'character varying'):
            stats.warnings.append(
                f'source PostgreSQL id column is {id_column_type!r}; nothing to migrate',
            )
            return stats

        source_rows = await source_conn.fetch(
            'SELECT id, created_at FROM context_entries ORDER BY created_at ASC, id ASC',
        )
        id_mapping: dict[int, str] = {}
        for row in source_rows:
            id_mapping[int(row['id'])] = generate_id_with_timestamp(row['created_at'])

        # Detect what the SOURCE carries so the target can be shaped to match.
        source_has_embeddings = await _pg_table_exists(source_conn, 'embedding_metadata')
        source_dim = await _detect_source_embedding_dim_pg(source_conn)

        # Auto-initialize the target schema when it has no context_entries table,
        # mirroring the SQLite path (initialize_target_sqlite). This removes the
        # trap where the user had to manually pre-create the fp32 layout: the
        # target is built with ENABLE_EMBEDDING_COMPRESSION effectively off (the
        # compression migration is never run here), so copy_vec_embeddings_pg has
        # an fp32 vec_context_embeddings table to write into. Compression is a
        # separate, explicit --compress step afterward.
        target_initialized = await _pg_table_exists(target_conn, 'context_entries')
        if not target_initialized:
            if options.dry_run:
                stats.warnings.append(
                    'target PostgreSQL database has no context_entries table; '
                    'it would be auto-initialized on a real run',
                )
            else:
                await initialize_target_postgresql(
                    options.target_url,
                    embedding_dim=source_dim,
                    with_semantic=source_has_embeddings,
                    stats=stats,
                )

        # Defensive backstop (never silently drop embeddings): a PRE-EXISTING
        # target that has context_entries but lacks the fp32 vec_context_embeddings
        # table (e.g. initialized with compression enabled or semantic search
        # disabled) cannot receive the source's embeddings -- refuse rather than
        # discard them. Skipped when the target was just auto-initialized
        # (target_initialized is False): a real run already created the fp32 vec
        # table via initialize_target_postgresql, and a dry run reports the
        # auto-init plan instead.
        if target_initialized and source_has_embeddings:
            target_has_vec = await _pg_table_exists(target_conn, 'vec_context_embeddings')
            if not target_has_vec:
                message = (
                    'source has embeddings but the target lacks the fp32 '
                    'vec_context_embeddings table (the target was likely '
                    'initialized with ENABLE_EMBEDDING_COMPRESSION=true or '
                    'ENABLE_SEMANTIC_SEARCH=false). Re-create the target with '
                    'ENABLE_SEMANTIC_SEARCH=true and ENABLE_EMBEDDING_COMPRESSION=false '
                    '(or let this CLI auto-initialize an empty target), run the '
                    'migration, then run --compress to enable compression.'
                )
                if options.dry_run:
                    stats.warnings.append(f'{message} (a real run would abort)')
                else:
                    stats.errors.append(f'{message} Aborting to avoid silently dropping embeddings.')
                    return stats

        if not options.dry_run:
            await target_conn.execute('BEGIN')
        try:
            entry_rows = await source_conn.fetch(
                'SELECT id, thread_id, source, content_type, text_content, '
                'metadata::text AS metadata, summary, content_hash, created_at, updated_at '
                'FROM context_entries ORDER BY created_at ASC, id ASC',
            )
            for entry in entry_rows:
                source_id = int(entry['id'])
                new_id = id_mapping[source_id]
                rewritten_metadata = rewrite_metadata_references(
                    entry['metadata'],
                    id_mapping,
                    stats,
                    source_id,
                )
                if not options.dry_run:
                    await target_conn.execute(
                        'INSERT INTO context_entries '
                        '(id, thread_id, source, content_type, text_content, metadata, summary, '
                        'content_hash, created_at, updated_at) '
                        'VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10)',
                        new_id,
                        entry['thread_id'],
                        entry['source'],
                        entry['content_type'],
                        entry['text_content'],
                        rewritten_metadata,
                        entry['summary'],
                        entry['content_hash'],
                        entry['created_at'],
                        entry['updated_at'],
                    )
                stats.rows_migrated += 1

            tag_rows = (
                await source_conn.fetch('SELECT context_entry_id, tag FROM tags ORDER BY id ASC')
                if await _pg_table_exists(source_conn, 'tags')
                else []
            )
            for tag_row in tag_rows:
                source_id = int(tag_row['context_entry_id'])
                tag_new_id: str | None = id_mapping.get(source_id)
                if tag_new_id is None:
                    stats.warnings.append(
                        f'tags row references missing context_entry_id={source_id}; skipped',
                    )
                    continue
                if not options.dry_run:
                    await target_conn.execute(
                        'INSERT INTO tags (context_entry_id, tag) VALUES ($1::uuid, $2)',
                        tag_new_id,
                        tag_row['tag'],
                    )
                stats.tags_migrated += 1

            image_rows = (
                await source_conn.fetch(
                    'SELECT context_entry_id, image_data, mime_type, image_metadata, position, created_at '
                    'FROM image_attachments ORDER BY id ASC',
                )
                if await _pg_table_exists(source_conn, 'image_attachments')
                else []
            )
            for img in image_rows:
                source_id = int(img['context_entry_id'])
                img_new_id: str | None = id_mapping.get(source_id)
                if img_new_id is None:
                    stats.warnings.append(
                        f'image_attachments row references missing context_entry_id={source_id}; skipped',
                    )
                    continue
                if not options.dry_run:
                    await target_conn.execute(
                        'INSERT INTO image_attachments '
                        '(context_entry_id, image_data, mime_type, image_metadata, position, created_at) '
                        'VALUES ($1::uuid, $2, $3, $4::jsonb, $5, $6)',
                        img_new_id,
                        img['image_data'],
                        img['mime_type'],
                        img['image_metadata'],
                        img['position'],
                        img['created_at'],
                    )
                stats.images_migrated += 1

            # ----- FIX: embeddings copy (was silently dropped before v3) -----
            # Copy embedding_metadata + vec_context_embeddings to restore
            # the embedding state in the target database. PostgreSQL has
            # no embedding_chunks table; the 1:N relationship lives in
            # vec_context_embeddings.id (BIGSERIAL PK) plus context_id
            # (UUID FK). Guarded by source table existence so a v2 source
            # that never enabled semantic search (no embedding_metadata
            # table) does not crash the migration.
            if source_has_embeddings:
                if options.dry_run and not target_initialized:
                    # The target would be auto-initialized on a real run, so its
                    # vec_context_embeddings table does not exist yet. Report
                    # symmetric would-migrate counts straight from the source
                    # instead of letting copy_vec_embeddings_pg emit a
                    # contradictory "initialize the target schema first" warning
                    # with vec_rows_migrated=0 (which would falsely imply the
                    # embeddings are lost). Mirrors the SQLite dry-run, which
                    # previews against an initialized target.
                    stats.embedding_metadata_migrated = int(
                        await source_conn.fetchval('SELECT COUNT(*) FROM embedding_metadata') or 0,
                    )
                    stats.vec_rows_migrated = int(
                        await source_conn.fetchval('SELECT COUNT(*) FROM vec_context_embeddings') or 0,
                    )
                else:
                    await copy_embedding_metadata_pg(
                        source_conn, target_conn, id_mapping, stats, options.dry_run,
                    )
                    await copy_vec_embeddings_pg(
                        source_conn, target_conn, id_mapping, stats, options.dry_run,
                    )

            if not options.dry_run:
                await target_conn.execute('COMMIT')
        except Exception:
            if not options.dry_run:
                await target_conn.execute('ROLLBACK')
            raise
    finally:
        await source_conn.close()
        await target_conn.close()
    return stats


async def run_migration_mixed_sqlite_to_postgresql(options: MigrationOptions) -> MigrationStats:
    """Migrate from a SQLite source to a PostgreSQL target.

    Vector embeddings are dropped (their on-disk binary formats are not portable
    between the two backends; a warning is emitted) -- re-embed the target
    afterward. All other data is copied: context_entries, tags, and image
    attachments. The target schema is auto-initialized when absent (base layout
    without the vector tables, which the server creates at the configured
    EMBEDDING_DIM on re-embed).

    Args:
        options: Parsed CLI options.

    Returns:
        Populated :class:`MigrationStats` instance.
    """
    import asyncpg

    stats = MigrationStats()
    stats.warnings.append(
        'cross-backend migration drops vector embeddings; re-embed the target after migration',
    )

    _, source_address = parse_backend_url(options.source_url)
    source = open_source_sqlite(source_address)
    target_conn = await asyncpg.connect(options.target_url, **_pg_connect_kwargs())
    try:
        id_kind = detect_source_id_kind(source)
        if id_kind != 'integer':
            stats.warnings.append(
                f'source database id column is {id_kind!r}; nothing to migrate',
            )
            return stats

        optional_tables = detect_optional_tables(source)

        if not options.dry_run and await _target_pg_has_data(target_conn):
            stats.errors.append(
                'target PostgreSQL database already contains context_entries rows. '
                'Recovery: if a prior run was interrupted, drop and recreate the target database '
                '(or pass a different --target-url) and rerun; the source database is unchanged. '
                'See the Recovering From an Interrupted Migration section of docs/migration-v2-to-v3.md.',
            )
            return stats

        cursor = source.execute(
            'SELECT id, created_at FROM context_entries ORDER BY created_at ASC, id ASC',
        )
        source_rows = cursor.fetchall()
        id_mapping = build_id_mapping(source_rows)

        # Auto-initialize the target schema when absent (mirrors the SQLite
        # path and the PG->PG path). Cross-backend migration drops vector
        # embeddings, so the target is initialized WITHOUT the semantic/chunking
        # layout (with_semantic=False); the server creates the vector tables at
        # the operator's configured dimension when re-embedding later.
        target_initialized = await _pg_table_exists(target_conn, 'context_entries')
        if not target_initialized:
            if options.dry_run:
                stats.warnings.append(
                    'target PostgreSQL database has no context_entries table; '
                    'it would be auto-initialized on a real run',
                )
            else:
                await initialize_target_postgresql(
                    options.target_url,
                    embedding_dim=None,
                    with_semantic=False,
                    stats=stats,
                )

        if not options.dry_run:
            await target_conn.execute('BEGIN')
        try:
            entry_cursor = source.execute(
                'SELECT id, thread_id, source, content_type, text_content, metadata, '
                'summary, content_hash, created_at, updated_at FROM context_entries '
                'ORDER BY created_at ASC, id ASC',
            )
            for row in entry_cursor:
                source_id = int(row['id'])
                new_id = id_mapping[source_id]
                rewritten_metadata = rewrite_metadata_references(
                    row['metadata'],
                    id_mapping,
                    stats,
                    source_id,
                )
                if not options.dry_run:
                    await target_conn.execute(
                        'INSERT INTO context_entries '
                        '(id, thread_id, source, content_type, text_content, metadata, summary, '
                        'content_hash, created_at, updated_at) '
                        'VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10)',
                        new_id,
                        row['thread_id'],
                        row['source'],
                        row['content_type'],
                        row['text_content'],
                        rewritten_metadata,
                        row['summary'],
                        row['content_hash'],
                        _coerce_datetime(row['created_at']),
                        _coerce_datetime(row['updated_at']),
                    )
                stats.rows_migrated += 1

            # Copy tags and image attachments (portable across backends: tags
            # are TEXT, image payloads are BYTEA<->BLOB). Only the embedding
            # vectors are dropped cross-backend. Reads are guarded by source
            # table presence.
            if optional_tables.get('tags'):
                tag_cursor = source.execute('SELECT context_entry_id, tag FROM tags ORDER BY id ASC')
                for tag_row in tag_cursor:
                    sid = int(tag_row['context_entry_id'])
                    mapped = id_mapping.get(sid)
                    if mapped is None:
                        stats.warnings.append(
                            f'tags row references missing context_entry_id={sid}; skipped',
                        )
                        continue
                    if not options.dry_run:
                        await target_conn.execute(
                            'INSERT INTO tags (context_entry_id, tag) VALUES ($1::uuid, $2)',
                            mapped,
                            tag_row['tag'],
                        )
                    stats.tags_migrated += 1

            if optional_tables.get('image_attachments'):
                image_cursor = source.execute(
                    'SELECT context_entry_id, image_data, mime_type, image_metadata, position, created_at '
                    'FROM image_attachments ORDER BY id ASC',
                )
                for img in image_cursor:
                    sid = int(img['context_entry_id'])
                    mapped = id_mapping.get(sid)
                    if mapped is None:
                        stats.warnings.append(
                            f'image_attachments row references missing context_entry_id={sid}; skipped',
                        )
                        continue
                    # A schema-legal NULL created_at is preserved as NULL rather
                    # than crashing _coerce_datetime (the migration must not
                    # invent data for arbitrary non-app source databases).
                    img_created_at = (
                        _coerce_datetime(img['created_at'])
                        if img['created_at'] is not None
                        else None
                    )
                    if not options.dry_run:
                        await target_conn.execute(
                            'INSERT INTO image_attachments '
                            '(context_entry_id, image_data, mime_type, image_metadata, position, created_at) '
                            'VALUES ($1::uuid, $2, $3, $4::jsonb, $5, $6)',
                            mapped,
                            img['image_data'],
                            img['mime_type'],
                            img['image_metadata'],
                            img['position'],
                            img_created_at,
                        )
                    stats.images_migrated += 1

            if not options.dry_run:
                await target_conn.execute('COMMIT')
        except Exception:
            if not options.dry_run:
                await target_conn.execute('ROLLBACK')
            raise
    finally:
        source.close()
        await target_conn.close()
    return stats


async def run_migration_mixed_postgresql_to_sqlite(options: MigrationOptions) -> MigrationStats:
    """Migrate from a PostgreSQL source to a SQLite target.

    Mirrors :func:`run_migration_mixed_sqlite_to_postgresql` with the backends
    swapped. Vector embeddings are not transferred (re-embed afterward), but
    context_entries, tags, and image attachments are copied, and the SQLite
    target's FTS5 index is rebuilt from the copied rows so full-text search works
    even though FTS is not portable from PostgreSQL.

    Args:
        options: Parsed CLI options.

    Returns:
        Populated :class:`MigrationStats` instance.
    """
    import asyncpg

    stats = MigrationStats()
    stats.warnings.append(
        'cross-backend migration drops vector embeddings; re-embed the target after migration',
    )

    _, target_address = parse_backend_url(options.target_url)
    if not options.dry_run and target_already_has_data_sqlite(target_address):
        stats.errors.append(
            f'target database already contains context_entries rows: {target_address}. '
            f'Recovery: if a prior run was interrupted, delete the target file and rerun; '
            f'the source database is unchanged. See the Recovering From an Interrupted Migration '
            f'section of docs/migration-v2-to-v3.md.',
        )
        return stats

    source_conn = await asyncpg.connect(options.source_url, **_pg_connect_kwargs())
    target: sqlite3.Connection | None = None
    try:
        await source_conn.execute('BEGIN TRANSACTION READ ONLY')

        id_column_type = await source_conn.fetchval(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name='context_entries' AND column_name='id'",
        )
        if id_column_type is None:
            stats.errors.append("source PostgreSQL database lacks 'context_entries.id' column")
            return stats
        if str(id_column_type).lower() in ('uuid', 'text', 'character varying'):
            stats.warnings.append(
                f'source PostgreSQL id column is {id_column_type!r}; nothing to migrate',
            )
            return stats

        source_rows = await source_conn.fetch(
            'SELECT id, created_at FROM context_entries ORDER BY created_at ASC, id ASC',
        )
        id_mapping: dict[int, str] = {}
        for row in source_rows:
            id_mapping[int(row['id'])] = generate_id_with_timestamp(row['created_at'])

        # Detect which optional tables the PostgreSQL source carries so the
        # SQLite target is shaped to match. FTS is offered on the target (it is
        # not portable from PostgreSQL, but the SQLite target supports it and
        # the index is rebuilt locally from the copied rows below).
        source_has_tags = await _pg_table_exists(source_conn, 'tags')
        source_has_images = await _pg_table_exists(source_conn, 'image_attachments')
        target = open_target_sqlite(target_address)
        initialize_target_sqlite(
            target,
            optional_tables={
                'tags': True,
                'image_attachments': True,
                'context_entries_fts': True,
            },
            embedding_dim=None,
            fts_tokenizer='porter unicode61',
            stats=stats,
        )

        entry_rows = await source_conn.fetch(
            'SELECT id, thread_id, source, content_type, text_content, '
            'metadata::text AS metadata, summary, content_hash, created_at, updated_at '
            'FROM context_entries ORDER BY created_at ASC, id ASC',
        )
        target.execute('BEGIN')
        try:
            for row in entry_rows:
                source_id = int(row['id'])
                new_id = id_mapping[source_id]
                rewritten_metadata = rewrite_metadata_references(
                    row['metadata'],
                    id_mapping,
                    stats,
                    source_id,
                )
                if not options.dry_run:
                    target.execute(
                        'INSERT INTO context_entries '
                        '(id, thread_id, source, content_type, text_content, metadata, summary, '
                        'content_hash, created_at, updated_at) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (
                            new_id,
                            row['thread_id'],
                            row['source'],
                            row['content_type'],
                            row['text_content'],
                            rewritten_metadata,
                            row['summary'],
                            row['content_hash'],
                            row['created_at'].isoformat(),
                            row['updated_at'].isoformat(),
                        ),
                    )
                stats.rows_migrated += 1

            # Copy tags and image attachments from the PostgreSQL source into
            # the SQLite target (portable: tags are TEXT, image payloads are
            # BYTEA->BLOB; image_metadata is cast to text for the SQLite TEXT
            # column; timestamps are rendered ISO-8601). Only embeddings are
            # dropped cross-backend. Reads guarded by source table presence.
            if source_has_tags:
                tag_rows = await source_conn.fetch(
                    'SELECT context_entry_id, tag FROM tags ORDER BY id ASC',
                )
                for tag_row in tag_rows:
                    sid = int(tag_row['context_entry_id'])
                    mapped = id_mapping.get(sid)
                    if mapped is None:
                        stats.warnings.append(
                            f'tags row references missing context_entry_id={sid}; skipped',
                        )
                        continue
                    if not options.dry_run:
                        target.execute(
                            'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                            (mapped, tag_row['tag']),
                        )
                    stats.tags_migrated += 1

            if source_has_images:
                image_rows = await source_conn.fetch(
                    'SELECT context_entry_id, image_data, mime_type, '
                    'image_metadata::text AS image_metadata, position, created_at '
                    'FROM image_attachments ORDER BY id ASC',
                )
                for img in image_rows:
                    sid = int(img['context_entry_id'])
                    mapped = id_mapping.get(sid)
                    if mapped is None:
                        stats.warnings.append(
                            f'image_attachments row references missing context_entry_id={sid}; skipped',
                        )
                        continue
                    # Preserve a schema-legal NULL created_at as NULL instead of
                    # crashing on None.isoformat() (the source may be an
                    # arbitrary non-app v2 database).
                    img_created_at = (
                        img['created_at'].isoformat() if img['created_at'] is not None else None
                    )
                    if not options.dry_run:
                        target.execute(
                            'INSERT INTO image_attachments '
                            '(context_entry_id, image_data, mime_type, image_metadata, position, created_at) '
                            'VALUES (?, ?, ?, ?, ?, ?)',
                            (
                                mapped,
                                img['image_data'],
                                img['mime_type'],
                                img['image_metadata'],
                                img['position'],
                                img_created_at,
                            ),
                        )
                    stats.images_migrated += 1

            if options.dry_run:
                target.rollback()
            else:
                target.commit()
        except Exception:
            target.rollback()
            raise

        # Rebuild the SQLite FTS5 index from the copied rows, outside the data
        # transaction (mirrors the SQLite->SQLite path), so the SQLite target
        # has working full-text search even though FTS is not portable from
        # PostgreSQL.
        rebuild_fts_sqlite(target, stats, options.dry_run)
        if not options.dry_run:
            target.commit()
    finally:
        await source_conn.close()
        if target is not None:
            target.close()
    return stats


# ---------------------------------------------------------------------------
# Argparse and main entrypoint
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for ``mcp-context-server-migrate``.

    Returns:
        Configured argparse parser.
    """
    parser = argparse.ArgumentParser(
        prog='mcp-context-server-migrate',
        description=(
            'Migrate an integer-keyed MCP context database to the UUIDv7 '
            'schema, compress/decompress an existing UUIDv7 database with '
            'TurboQuant embedding compression, or re-embed an existing '
            'database under a new model.'
        ),
    )
    parser.add_argument(
        '--source-url',
        required=True,
        help='Source database URL or filesystem path (sqlite:/// or postgresql://).',
    )
    parser.add_argument(
        '--target-url',
        required=False,
        default=None,
        help=(
            'Target database URL or filesystem path. Required for the v2->v3 '
            'migration; ignored when --compress or --decompress is set.'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run the full migration logic in memory but issue no writes against the target.',
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=None,
        metavar='PATH',
        help='Write a JSON migration report to PATH.',
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--compress',
        action='store_true',
        help=(
            'Compress an existing database with fp32 embeddings. Requires '
            'ENABLE_EMBEDDING_COMPRESSION=true in the environment. Reads '
            'from --source-url; --target-url is ignored. Use --dry-run to '
            'preview. May be combined with --embed-missing to also backfill '
            'entries lacking embeddings (compress runs first, then backfill).'
        ),
    )
    mode_group.add_argument(
        '--decompress',
        action='store_true',
        help=(
            'Decompress a database with compressed embeddings back to fp32 '
            '(lossy reconstruction). Requires ENABLE_EMBEDDING_COMPRESSION '
            'to be unset or false. Reads from --source-url; --target-url is '
            'ignored. Use --dry-run to preview. Not combinable with '
            '--embed-missing (a co-passed --embed-missing is ignored; run it '
            'separately after decompressing).'
        ),
    )
    mode_group.add_argument(
        '--re-embed',
        action='store_true',
        help=(
            'Re-embed EVERY context_entries row using the currently '
            'configured EMBEDDING_PROVIDER/EMBEDDING_MODEL, deleting existing '
            'embeddings first. The one-command path for switching the '
            'embedding MODEL on an existing database. Works for fp32 and '
            'compressed layouts. Requires ENABLE_EMBEDDING_GENERATION=true. '
            'Reads from --source-url; --target-url is ignored. Use --dry-run '
            'to preview the entry count without calling the provider. Refuses '
            'a dimension change (a different EMBEDDING_DIM than stored): a '
            'dimension change requires the documented rebuild. A co-passed '
            '--embed-missing is ignored because --re-embed already covers '
            'every entry.'
        ),
    )
    # --embed-missing is intentionally OUTSIDE mode_group: Shape gamma
    # allows composition with --compress (one-shot compress+backfill) AND
    # standalone operation (fp32-only backfill or compressed-only backfill,
    # depending on the env var state).
    parser.add_argument(
        '--embed-missing',
        action='store_true',
        help=(
            'Generate embeddings for context_entries rows that lack an '
            'embedding_metadata row, calling the configured embedding '
            'provider (EMBEDDING_PROVIDER, EMBEDDING_MODEL). Works '
            'standalone (against the existing storage layout) or composed '
            'with --compress (compress first, then backfill into the '
            'compressed table). Reads from --source-url; --target-url is '
            'ignored. Use --dry-run to preview the missing-entry count '
            'without calling the provider.'
        ),
    )
    return parser


def print_summary(stats: MigrationStats, source_url: str, target_url: str) -> None:
    """Print a human-readable summary of the migration to stdout."""
    source_display = mask_credentials(source_url)
    target_display = mask_credentials(target_url)
    print('Migration summary')
    print(f'  source: {source_display}')
    print(f'  target: {target_display}')
    print(f'  rows migrated: {stats.rows_migrated}')
    print(f'  references rewritten: {stats.references_rewritten}')
    print(f'  orphan references: {stats.orphan_references}')
    print(f'  malformed references: {stats.malformed_references}')
    print(f'  tags migrated: {stats.tags_migrated}')
    print(f'  images migrated: {stats.images_migrated}')
    print(f'  embedding_metadata migrated: {stats.embedding_metadata_migrated}')
    print(f'  embedding_chunks migrated: {stats.embedding_chunks_migrated}')
    print(f'  vec rows migrated: {stats.vec_rows_migrated}')
    print(f'  FTS rebuilt: {stats.fts_rebuilt}')
    if stats.warnings:
        print(f'  warnings: {len(stats.warnings)}')
        for message in stats.warnings:
            print(f'    - {message}')
    if stats.errors:
        print(f'  errors: {len(stats.errors)}')
        for message in stats.errors:
            print(f'    - {message}')
    if stats.rows_migrated > 0 and not stats.errors:
        print(
            'Next steps: point the server at the new target database '
            '(DB_PATH=... for SQLite or POSTGRESQL_CONNECTION_STRING=... for PostgreSQL).',
        )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``mcp-context-server-migrate`` script.

    Args:
        argv: Optional override for ``sys.argv[1:]`` (used by tests).

    Returns:
        Process exit code: 0 on success, 1 on user error or recorded
        errors, 2 on unrecoverable migration failure.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Single-backend in-place operations: --compress, --decompress,
    # --re-embed, --embed-missing. All dispatch on --source-url alone;
    # --target-url is ignored. Composition rule: --compress and
    # --embed-missing can be combined (--compress runs first, then
    # --embed-missing against the compressed layout). --compress,
    # --decompress, and --re-embed are mutually exclusive (enforced by
    # argparse mode_group). Both --decompress and --re-embed return before the
    # --embed-missing check below, so a co-passed --embed-missing is silently
    # superseded: --re-embed already re-embeds every entry (gaps included),
    # and --decompress is documented as not combinable with --embed-missing
    # (run it separately afterward). Imported lazily so callers running the
    # v2->v3 migration do not pay the compression/numpy import cost.
    if args.compress:
        from app.cli.migrate_compression import run_compress
        rc = run_compress(args.source_url, dry_run=args.dry_run)
        if rc != 0:
            return rc
        if args.embed_missing:
            from app.cli.migrate_embeddings import run_embed_missing
            return run_embed_missing(args.source_url, dry_run=args.dry_run)
        return 0
    if args.decompress:
        from app.cli.migrate_compression import run_decompress
        return run_decompress(args.source_url, dry_run=args.dry_run)
    if args.re_embed:
        from app.cli.migrate_reembed import run_reembed
        return run_reembed(args.source_url, dry_run=args.dry_run)
    if args.embed_missing:
        from app.cli.migrate_embeddings import run_embed_missing
        return run_embed_missing(args.source_url, dry_run=args.dry_run)

    if not args.target_url:
        logger.error(
            '--target-url is required for the v2->v3 migration. '
            'For an in-place operation against --source-url, pass one of '
            '--compress, --decompress, --re-embed, or --embed-missing '
            '(none of which use --target-url).',
        )
        return 1

    options = MigrationOptions(
        source_url=args.source_url,
        target_url=args.target_url,
        dry_run=args.dry_run,
        report_path=args.report,
    )

    try:
        src_kind, _ = parse_backend_url(options.source_url)
        tgt_kind, _ = parse_backend_url(options.target_url)
    except ValueError as exc:
        logger.error('invalid database URL: %s', exc)
        return 1

    try:
        if src_kind == 'sqlite' and tgt_kind == 'sqlite':
            stats = run_migration_sqlite_to_sqlite(options)
        elif src_kind == 'postgresql' and tgt_kind == 'postgresql':
            stats = asyncio.run(run_migration_postgresql(options))
        elif src_kind == 'sqlite' and tgt_kind == 'postgresql':
            stats = asyncio.run(run_migration_mixed_sqlite_to_postgresql(options))
        elif src_kind == 'postgresql' and tgt_kind == 'sqlite':
            stats = asyncio.run(run_migration_mixed_postgresql_to_sqlite(options))
        else:
            logger.error('unsupported backend combination: %s -> %s', src_kind, tgt_kind)
            return 1
    except Exception as exc:
        logger.exception('migration failed: %s', exc)
        return 2

    print_summary(stats, options.source_url, options.target_url)
    if options.report_path is not None:
        try:
            options.report_path.write_text(
                json.dumps(stats.to_dict(), indent=2),
                encoding='utf-8',
            )
        except OSError as exc:
            logger.error('failed to write report: %s', exc)
            stats.errors.append(f'failed to write report: {exc}')

    return 0 if not stats.errors else 1


if __name__ == '__main__':
    sys.exit(main())
