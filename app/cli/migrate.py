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
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from urllib.parse import quote
from urllib.parse import urlparse

from pydantic import ValidationError

from app.errors import ConfigurationError

# UUIDv7 generation for integer-keyed rows uses the timestamp parameter of
# uuid_utils.uuid7() in UNIX seconds (with optional nanos for sub-second
# precision). Upstream tracker on the parameter's units:
# https://github.com/aminalaee/uuid-utils/issues/73
from app.ids import generate_id_with_timestamp
from app.metadata_types import non_finite_metadata_error
from app.metadata_types import pg_indexed_cast_error
from app.metadata_types import pg_indexed_metadata_text
from app.metadata_types import unstorable_string_error
from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT
from app.pgvector_limits import exceeds_pgvector_index_dim_limit

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
            successfully remapped to UUIDv7 hex strings AND reached the
            target. A remapping the run then discards -- because the
            re-encode was rejected and the original metadata was preserved,
            or because the whole row was skipped -- is not counted.
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
        if path.startswith('//'):
            # SQLAlchemy POSIX absolute form: sqlite:////abs/path keeps an
            # extra leading slash after the scheme strip. Collapse the run to
            # a single slash; a retained double-slash prefix would later be
            # read as a file-URI authority by the SQLite backend and rejected
            # (sqlite3.OperationalError: invalid uri authority).
            path = '/' + path.lstrip('/')
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


def _open_sqlite_target(path: str, dry_run: bool) -> sqlite3.Connection:
    """Open the target SQLite database, or an in-memory database on dry-run.

    A dry run must issue no writes against the target, so it operates against an
    ephemeral in-memory database instead of creating and schema-committing a file
    at the target path (schema init commits before the data-copy rollback and
    would otherwise persist an empty database file on disk).

    Args:
        path: Filesystem path to the target SQLite database file.
        dry_run: When True, open an in-memory database and touch no disk file.

    Returns:
        A read-write ``sqlite3.Connection`` with ``row_factory`` set to
        :class:`sqlite3.Row` and foreign-key enforcement enabled.
    """
    if dry_run:
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        return conn
    return open_target_sqlite(path)


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
    # A bool is a degenerate int and is NOT a valid timestamp -- reject it explicitly.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Some non-app source databases store created_at as Unix epoch SECONDS. Coerce via
        # epoch + timedelta (NOT datetime.fromtimestamp, which raises OSError for negative /
        # out-of-range epochs on Windows) so a numeric -- including pre-1970 -- created_at
        # never aborts the whole migration; _created_at_for_id anchors a resulting pre-1970
        # value for id derivation while the stored created_at value is preserved verbatim.
        try:
            return _NULL_CREATED_AT_ANCHOR + timedelta(seconds=float(value))
        except (OverflowError, ValueError):
            # An out-of-range epoch (extreme future/past) cannot derive a datetime: anchor
            # it rather than abort, matching the NULL / pre-1970 handling.
            return _NULL_CREATED_AT_ANCHOR
    raise ValueError(f'unsupported created_at value: {value!r}')


# A schema-legal NULL created_at in an arbitrary non-app source row cannot derive
# a UUIDv7 id (the id embeds a timestamp), so it is anchored to a fixed epoch
# rather than aborting the whole migration. The stored created_at VALUE is kept
# NULL (not invented); only id derivation uses the anchor.
_NULL_CREATED_AT_ANCHOR = datetime(1970, 1, 1, tzinfo=UTC)


def _sqlite_timestamp(value: datetime | None) -> str | None:
    """Render a source datetime as SQLite's canonical TEXT timestamp.

    SQLite stores created_at/updated_at as "YYYY-MM-DD HH:MM:SS" (UTC, the
    CURRENT_TIMESTAMP form the server writes) and orders/filters them as TEXT.
    Writing ``datetime.isoformat()`` ("YYYY-MM-DDTHH:MM:SS+00:00") instead stores a
    'T'/offset form that mis-sorts under SQLite's TEXT comparison and ``ORDER BY
    created_at`` (and skews date-range filters), so normalize to the space form in
    UTC. ``None`` is preserved (schema-legal NULL).

    Returns:
        The canonical "YYYY-MM-DD HH:MM:SS" UTC string, or ``None`` for ``None``.
    """
    if value is None:
        return None
    dt = value.astimezone(UTC) if value.tzinfo is not None else value
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def _created_at_for_id(value: object) -> datetime:
    """Coerce a source ``created_at`` to a datetime for UUIDv7 id derivation.

    Unlike :func:`_coerce_datetime`, a missing (NULL) ``created_at`` -- and any
    other value that cannot be parsed (a malformed non-ISO / non-epoch string, an
    out-of-range epoch) -- is tolerated: it falls back to
    :data:`_NULL_CREATED_AT_ANCHOR` so one bad row in an arbitrary non-app source
    database cannot abort the entire migration. The stored ``created_at`` value is
    preserved verbatim by the callers that bind it (see
    :func:`_stored_datetime_or_none`); only the derived id timestamp is anchored.

    Args:
        value: A source ``created_at`` value (datetime, ISO/epoch text, numeric
            epoch, None, or an unparseable value).

    Returns:
        A timezone-aware :class:`datetime.datetime`, or the epoch anchor when
        ``value`` is None or cannot be parsed.
    """
    if value is None:
        return _NULL_CREATED_AT_ANCHOR
    try:
        coerced = _coerce_datetime(value)
    except (ValueError, OverflowError):
        # A malformed non-NULL created_at (e.g. a non-ISO string like '2024/01/01'
        # or '15-06-2024', for which both datetime.fromisoformat and the
        # '%Y-%m-%d %H:%M:%S' fallback raise) must NOT abort the whole migration:
        # anchor its derived id like the NULL / pre-1970 / out-of-range-epoch cases
        # while the binding callers preserve the stored value verbatim (or NULL).
        return _NULL_CREATED_AT_ANCHOR
    # A pre-1970 (negative-epoch) timestamp makes uuid_utils.uuid7 raise
    # OverflowError on the negative seconds; anchor it like NULL for id
    # derivation while the stored created_at value is preserved verbatim.
    if coerced < _NULL_CREATED_AT_ANCHOR:
        return _NULL_CREATED_AT_ANCHOR
    return coerced


def _stored_datetime_or_none(value: object) -> datetime | None:
    """Coerce a stored ``created_at`` / ``updated_at`` for a verbatim target bind.

    Mirrors :func:`_created_at_for_id`'s tolerance for the value bound INTO the
    target timestamp column (not the derived id): a NULL or otherwise unparseable
    value yields ``None`` (stored as SQL NULL on the target, exactly as a NULL
    source already does) rather than aborting the whole migration on one bad row
    in an arbitrary non-app source database. A well-formed value is coerced to a
    timezone-aware datetime.

    Args:
        value: A stored timestamp value (datetime, ISO/epoch text, numeric epoch,
            None, or an unparseable value).

    Returns:
        A timezone-aware :class:`datetime.datetime`, or ``None`` when the value is
        NULL or cannot be parsed.
    """
    if value is None:
        return None
    try:
        return _coerce_datetime(value)
    except (ValueError, OverflowError):
        return None


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
    null_created_at = 0
    for row in source_rows:
        source_id = int(row['id'])
        if row['created_at'] is None:
            null_created_at += 1
        created_at = _created_at_for_id(row['created_at'])
        mapping[source_id] = generate_id_with_timestamp(created_at)
    if null_created_at:
        logger.warning(
            '%d source context_entries row(s) had NULL created_at; their ids '
            'were anchored to %s (the stored created_at is preserved as NULL)',
            null_created_at,
            _NULL_CREATED_AT_ANCHOR.isoformat(),
        )
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

    Re-encoding is done with ``allow_nan=False``. This function runs on ALL FOUR
    migration directions, and it is where an invalid token would be MANUFACTURED:
    ``json.loads`` accepts the non-standard ``NaN``/``Infinity``/``-Infinity`` AND
    silently turns a standard-but-overflowing literal such as ``1e400`` into
    ``inf``, and a default ``json.dumps`` then writes those back as tokens no
    RFC 8259 parser accepts. That converts VALID source metadata into INVALID
    target metadata on the SQLite paths (``json_valid`` flips to 0) and aborts the
    whole transaction on the PostgreSQL paths (the jsonb bind rejects it). With
    ``allow_nan=False`` the encoder raises instead, and the row's ORIGINAL metadata
    is preserved verbatim -- which is valid JSON on every target, since only the
    Python float round trip could not represent it. Reference rewriting is skipped
    for that row and the omission is recorded in ``stats.errors`` (a non-empty
    error list makes the CLI exit non-zero), so the operator can repair the value
    rather than discover a stale reference later.

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
    # _walk_and_rewrite counts each remapping as it mutates the parsed structure, but
    # the encoder below can still reject that structure -- in which case the mutations
    # are discarded and the ORIGINAL metadata is returned. Snapshot the counter so the
    # discarded remappings are not reported as remappings that reached the target.
    references_rewritten_before = stats.references_rewritten
    _walk_and_rewrite(parsed, id_mapping, stats, row_pk, seen=set())
    try:
        return json.dumps(parsed, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        stats.references_rewritten = references_rewritten_before
        stats.errors.append(
            f'row {row_pk}: metadata contains a number Python cannot round-trip as JSON '
            f'({exc}); the original metadata is preserved verbatim and any integer '
            f'context_ids references inside it were NOT rewritten',
        )
        return metadata_json


def _pg_unstorable_column_reason(value: str | None, *, is_jsonb: bool) -> str | None:
    """Return why a SQLite value cannot be stored on the PostgreSQL target, else None.

    An embedded NUL (U+0000) or unpaired UTF-16 surrogate is legal in a SQLite
    TEXT value but fatal on PostgreSQL: asyncpg rejects a NUL text bind
    (``CharacterNotInRepertoireError``, SQLSTATE 22021) and an unpaired surrogate
    is not UTF-8-encodable at all. A ``jsonb`` column has a second failure mode --
    the store path serializes a metadata NUL as the JSON escape ``\\u0000`` (six
    ASCII characters, not a literal byte), which passes the raw-string check yet
    is rejected by PostgreSQL's jsonb parser (SQLSTATE 22P05). A ``jsonb`` column
    has a THIRD failure mode: a non-finite JSON number. ``json.loads`` accepts the
    non-standard tokens ``NaN``/``Infinity``/``-Infinity`` AND silently converts a
    standard-but-overflowing literal such as ``1e400`` into ``inf``, while
    ``json.dumps`` -- which :func:`rewrite_metadata_references` applies to every
    parseable ``metadata`` value on the copy path -- re-emits those values as the
    invalid tokens the jsonb parser rejects. Such a value is also unstorable
    through the server's own store boundary, which runs the same
    :func:`app.metadata_types.non_finite_metadata_error` guard, so a v3 target must
    not receive it either way.

    Accordingly this runs the shared
    :func:`app.metadata_types.unstorable_string_error` walker on the raw
    serialized string (catching a literal NUL/surrogate in the bind) and, for a
    ``jsonb`` column, ALSO on the decoded structure (``json.loads`` restores the
    ``\\u0000`` escape to a real U+0000 the walker detects) plus the shared
    non-finite-number guard on that same decoded structure. For a ``jsonb`` column
    a value ``json.loads`` cannot parse is itself unstorable: the target binds it
    through a ``$n::jsonb`` cast
    that PostgreSQL rejects mid-transaction (SQLSTATE 22P02), aborting the whole
    migration with no row identification -- the exact failure this pre-check
    exists to convert into a skip-and-warn -- so malformed JSON returns a reason
    rather than passing as storable. (A raw TEXT column has no such cast, so
    unparseable content is not rejected there.)

    Args:
        value: The raw SQLite column value (already serialized to a JSON string
            for a ``jsonb`` column), or ``None``.
        is_jsonb: True when ``value`` is bound into a PostgreSQL ``jsonb`` column,
            enabling the decoded-escape and non-finite-number checks in addition
            to the raw-string check.

    Returns:
        The shared guard's operator-facing message for the first offending
        sequence or value found, else None.
    """
    if value is None:
        return None
    raw_reason = unstorable_string_error(value)
    if raw_reason is not None:
        return raw_reason
    if not is_jsonb:
        return None
    try:
        decoded: object = json.loads(value)
    except (json.JSONDecodeError, ValueError) as exc:
        return (
            f'value is not valid JSON, so the target rejects the jsonb bind '
            f'mid-transaction (SQLSTATE 22P02): {exc}. SQLite stores the same '
            f'malformed metadata verbatim.'
        )
    string_reason = unstorable_string_error(decoded)
    if string_reason is not None:
        return string_reason
    return non_finite_metadata_error(decoded)


# PostgreSQL refuses to index a btree tuple larger than roughly a third of an 8KB
# page: BTMaxItemSize is 2704 bytes on btree version 4.
#
# For thread_id and tags this matters only for a SQLite SOURCE: SQLite indexes those
# columns with no size limit, while every PostgreSQL database declares idx_thread_id
# and idx_tags_tag in its base schema and therefore cannot already hold a value that
# breaches the ceiling. An INDEXED METADATA value is different, and a PostgreSQL
# source can hold one: the metadata expression indexes are deliberately NOT in the
# base schema (see app/schemas/postgresql_schema.sql -- a database initialized only by
# this CLI has none until its first server startup), so a source whose
# METADATA_INDEXED_FIELDS never covered a field, or which was never started as a
# server, can carry a value the target's index rejects.
#
# Bound at the source, such a value aborts the INSERT mid-transaction, ROLLBACKs the
# entire run, and reports a raw driver error naming no source row -- the exact failure
# the NUL/surrogate pre-check exists to convert into a per-row skip-and-warn.
#
# The budget is deliberately measured against the UNCOMPRESSED value. PostgreSQL
# compresses an index attribute larger than 512 bytes in line, so a highly repetitive
# oversized value can still fit while an incompressible one of the same length cannot.
# Modeling that compression is not possible from here, and the two errors are not
# symmetric: over-accepting costs the WHOLE run, while over-skipping costs one row that
# is named in the errors and migrates on a rerun once the value is shortened.
_PG_BTREE_MAX_ITEM_BYTES = 2704

# What an index tuple costs BESIDES the payload of the value being checked, for any
# index shape:
#   16 bytes  IndexTupleData header (8 bytes), MAXALIGNed to 16 once a nullable
#             trailing column adds the null bitmap
#    4 bytes  the long varlena header carried by a text datum past the 126-byte
#             short-header threshold
#    8 bytes  one MAXALIGN quantum of slack on the assembled tuple, covering the
#             inter-attribute padding a fixed-width trailing column can introduce
_PG_INDEX_TUPLE_FIXED_BYTES = 16 + 4 + 8

# Budget for a value that is indexed ON ITS OWN: idx_tags_tag on ``tags(tag)`` and the
# metadata expression indexes ``idx_metadata_<field>`` on
# ``context_entries((metadata->>'<field>'))`` that handle_metadata_indexes provisions
# for every string-typed METADATA_INDEXED_FIELDS entry.
_PG_MAX_INDEXED_VALUE_BYTES = _PG_BTREE_MAX_ITEM_BYTES - _PG_INDEX_TUPLE_FIXED_BYTES

# thread_id needs a SMALLER budget because it is the leading column of
# idx_context_entries_dedup_hash (thread_id, source, content_hash), which the base
# schema declares on every PostgreSQL target: the tuple must also hold 6 bytes of
# source ('agent' as a short-header varlena) and 65 bytes of content_hash (a 64-character
# SHA-256 hex string, likewise short-header). The other compound indexes thread_id feeds
# (idx_thread_source, idx_thread_created) have narrower trailing columns than that, so
# the dedup index sets the ceiling.
_PG_MAX_INDEXED_THREAD_ID_BYTES = _PG_MAX_INDEXED_VALUE_BYTES - (6 + 65)


def _pg_unindexable_column_reason(value: str | None, max_bytes: int) -> str | None:
    """Return why a value is too large for a PostgreSQL btree index, else None.

    Args:
        value: The candidate value for an INDEXED target column.
        max_bytes: Payload budget for the widest index this value feeds --
            :data:`_PG_MAX_INDEXED_THREAD_ID_BYTES` for thread_id (a compound index
            whose trailing columns share the tuple), :data:`_PG_MAX_INDEXED_VALUE_BYTES`
            for a value indexed on its own.

    Returns:
        A reason string when the encoded value exceeds the index-tuple budget,
        else None.
    """
    if value is None:
        return None
    encoded_bytes = len(value.encode('utf-8'))
    if encoded_bytes <= max_bytes:
        return None
    return (
        f'the value is {encoded_bytes} UTF-8 bytes, which exceeds the PostgreSQL btree '
        f'index-tuple budget of {max_bytes} bytes for this indexed '
        f'column; SQLite indexes it without a size limit, so this row is skipped -- shorten '
        f'the value in the source database and rerun to migrate it'
    )


def _first_pg_unindexable_column(
    columns: Iterable[tuple[str, str | None, int]],
) -> tuple[str, str] | None:
    """Return the first ``(column, reason)`` a PostgreSQL btree index cannot hold, else None.

    Args:
        columns: Ordered ``(column_name, value, max_bytes)`` candidates for one row,
            limited to columns the target schema actually indexes. ``max_bytes`` is the
            payload budget of the widest index that column feeds.

    Returns:
        The ``(column_name, reason)`` of the first oversized column, else None.
    """
    for name, value, max_bytes in columns:
        reason = _pg_unindexable_column_reason(value, max_bytes)
        if reason is not None:
            return name, reason
    return None


def _first_pg_unindexable_metadata_field(metadata_json: str | None) -> tuple[str, str] | None:
    """Return the first indexed metadata field a PostgreSQL target cannot index, else None.

    A ``METADATA_INDEXED_FIELDS`` key gets an expression btree index
    ``idx_metadata_<field>`` on ``context_entries((metadata->>'<field>'))``
    (app.migrations.metadata), evaluated on every INSERT. SQLite's equivalent
    ``json_extract`` index has neither a size limit nor a cast, so a source can hold a
    value the target cannot index at all -- aborting the whole run when the target
    already carries the index, or breaking the target's first server startup (which
    creates the index) when the CLI initialized the target itself. Both ways that
    happens are checked, in the order they would fail:

    * WIDTH, for a ``string``-typed field, whose TEXT btree entry is bounded by the
      index-tuple ceiling that also bounds thread_id and tags. The width is measured on
      the text the expression YIELDS (:func:`~app.metadata_types.pg_indexed_metadata_text`),
      so a list or object is measured as the whole serialized JSON ``->>`` renders it as.
    * CAST COMPATIBILITY, for an ``integer``/``boolean``/``float``-typed field, whose
      index expression carries a hard SQL cast the value must survive. This is the same
      check the write boundary applies
      (:func:`~app.metadata_types.pg_indexed_cast_error`), so a value the running server
      would refuse to store is a value the migration refuses to import -- SQLite happily
      holds ``{"priority": "high"}`` under an integer-typed field, and the cast is where
      that stops being portable.

    ``array``/``object``-typed fields are exempt from both: they build no expression
    index at all, being served by the always-present jsonb_path_ops GIN index, which
    hashes its entries. Only top-level keys are inspected, because
    ``metadata->>'<field>'`` addresses top-level keys only. Unparseable or non-object
    metadata returns None: the jsonb bind itself rejects it, which the unstorable
    pre-check reports with a more specific reason.

    Args:
        metadata_json: The metadata JSON string about to be bound into the target's
            ``jsonb`` column, or None when the row has no metadata.

    Returns:
        The ``('metadata.<field>', reason)`` of the first unindexable value, else None.
    """
    if metadata_json is None:
        return None
    try:
        parsed: object = json.loads(metadata_json)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    from app.settings import get_settings

    indexed_fields = get_settings().storage.metadata_indexed_fields
    for key, value in cast(dict[str, object], parsed).items():
        type_hint = indexed_fields.get(key)
        if type_hint is None:
            continue
        if type_hint == 'string':
            reason = _pg_unindexable_column_reason(pg_indexed_metadata_text(value), _PG_MAX_INDEXED_VALUE_BYTES)
            if reason is not None:
                return f'metadata.{key}', reason
            continue
        cast_error = pg_indexed_cast_error(key, value, type_hint)
        if cast_error is not None:
            return (
                f'metadata.{key}',
                (
                    f'{cast_error}; the PostgreSQL expression index evaluates that cast on every '
                    f'INSERT while SQLite indexes the value uncast, so this row is skipped -- '
                    f'correct the value in the source database and rerun to migrate it'
                ),
            )
    return None


def _first_pg_unstorable_column(
    columns: Iterable[tuple[str, str | None, bool]],
) -> tuple[str, str] | None:
    """Return the first ``(column, reason)`` a PostgreSQL target cannot store, else None.

    Each candidate is a ``(name, value, is_jsonb)`` triple. Columns are checked in
    the given order and the first offending one short-circuits, so a caller can
    identify the exact column that would abort the row's INSERT on PostgreSQL.

    Args:
        columns: Ordered ``(column_name, value, is_jsonb)`` candidates for one row.

    Returns:
        The ``(column_name, reason)`` of the first unstorable column, else None.
    """
    for name, value, is_jsonb in columns:
        reason = _pg_unstorable_column_reason(value, is_jsonb=is_jsonb)
        if reason is not None:
            return name, reason
    return None


# Tags are a SET of labels per entry, and the target schema enforces that with a
# UNIQUE index on (context_entry_id, tag). A legacy source predating the write-path
# deduplication can hold the same label twice for one entry, which would abort the
# whole run on the second INSERT, so the copy collapses the duplicates instead of
# carrying them across. MIN(id) keeps the ordering deterministic and identical on
# both backends.
_SELECT_DISTINCT_TAGS_SQL = (
    'SELECT MIN(id) AS id, context_entry_id, tag FROM tags '
    'GROUP BY context_entry_id, tag ORDER BY id ASC'
)


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
    cursor = source.execute(_SELECT_DISTINCT_TAGS_SQL)
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


def _malformed_image_metadata_reason(value: object) -> str | None:
    """Return why an ``image_metadata`` value is undecodable, else None.

    Args:
        value: The ``image_attachments.image_metadata`` value read from the source.

    Returns:
        A reason string when the value is neither absent nor valid JSON, else None.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return f'value is a {type(value).__name__}, not the JSON text the column stores'
    try:
        json.loads(value)
    except (json.JSONDecodeError, ValueError) as exc:
        return (
            f'value is not valid JSON, so no reader can decode it: {exc}. '
            f'Correct it in the source database and rerun to migrate this attachment.'
        )
    return None


def copy_image_attachments(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """Copy ``image_attachments`` rows from source to target.

    The local ``image_attachments.id`` AUTOINCREMENT counter is regenerated by the
    target schema. Image payload columns are copied verbatim.

    ``image_metadata`` is the exception: it is validated as JSON before the bind. On a
    PostgreSQL target the ``$4::jsonb`` cast validates it for free and a malformed
    value is skipped with a reason; a SQLite target binds through a plain ``?`` and
    would import a payload no reader can decode. Readers degrade gracefully on such a
    row, but the migration has no business creating one, and the two targets should
    reject the same input.
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
        metadata_reason = _malformed_image_metadata_reason(row['image_metadata'])
        if metadata_reason is not None:
            stats.errors.append(
                f'image_attachments row context_entry_id={source_id} column '
                f"'image_metadata' skipped: {metadata_reason}",
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
    # Probe both sides for the start_index/end_index boundary columns, mirroring
    # copy_embedding_metadata's chunk_count guard. A pre-f36266c source schema
    # (embedding_chunks created before the boundary columns and never upgraded
    # in-place by a live server -- the CLI's "run on a backup" workflow bypasses
    # that backfill) lacks them; naming them unconditionally would raise
    # sqlite3.OperationalError and abort the whole migration. When the source
    # lacks them, default to 0 (the chunking migration's own backfill default).
    source_has_boundaries = _table_has_column(source, 'embedding_chunks', 'start_index')
    target_has_boundaries = _table_has_column(target, 'embedding_chunks', 'start_index')

    source_columns = ['id', 'context_id', 'vec_rowid']
    if source_has_boundaries:
        source_columns += ['start_index', 'end_index']
    source_columns.append('created_at')
    cursor = source.execute(
        f'SELECT {", ".join(source_columns)} FROM embedding_chunks ORDER BY id ASC',
    )

    target_columns = ['id', 'context_id', 'vec_rowid']
    if target_has_boundaries:
        target_columns += ['start_index', 'end_index']
    target_columns.append('created_at')
    placeholders = ', '.join('?' * len(target_columns))
    insert_sql = f'INSERT INTO embedding_chunks ({", ".join(target_columns)}) VALUES ({placeholders})'

    inserted = 0
    for row in cursor:
        source_id = int(row['context_id'])
        mapped: str | None = id_mapping.get(source_id)
        if mapped is None:
            stats.warnings.append(
                f'embedding_chunks row references missing context_id={source_id}; skipped',
            )
            continue
        params: list[object] = [row['id'], mapped, row['vec_rowid']]
        if target_has_boundaries:
            params += [
                row['start_index'] if source_has_boundaries else 0,
                row['end_index'] if source_has_boundaries else 0,
            ]
        params.append(row['created_at'])
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
            semantic-search migration. When ``None`` (the source
            ``embedding_metadata`` table exists but is empty), falls back
            to ``get_settings().embedding.dim``, mirroring the PostgreSQL
            counterpart. Ignored when sqlite-vec is not available.
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
        if embedding_dim is not None:
            dim = embedding_dim
        else:
            from app.settings import get_settings

            dim = get_settings().embedding.dim
        semantic_sql = semantic_sql.replace('{EMBEDDING_DIM}', str(dim))
        try:
            target.executescript(semantic_sql)
        except sqlite3.OperationalError as exc:
            stats.warnings.append(f'semantic-search target migration partial failure: {exc}')

    if optional_tables.get('embedding_chunks') and vec_loaded:
        chunking_sql = _read_schema_file('add_chunking_sqlite.sql')
        try:
            target.executescript(chunking_sql)
            # add_chunking_sqlite.sql does NOT add embedding_metadata.chunk_count -- the
            # server's chunking migration adds it in Python (SQLite has no ADD COLUMN IF
            # NOT EXISTS). Mirror that here so a CLI-migrated SQLite target matches a
            # server-initialized DB AND the PostgreSQL CLI target (whose
            # add_chunking_postgresql.sql includes chunk_count); otherwise
            # copy_embedding_metadata silently drops per-context chunk counts because the
            # target lacks the column.
            meta_cols = [r[1] for r in target.execute('PRAGMA table_info(embedding_metadata)').fetchall()]
            if meta_cols and 'chunk_count' not in meta_cols:
                target.execute('ALTER TABLE embedding_metadata ADD COLUMN chunk_count INTEGER NOT NULL DEFAULT 1')
        except sqlite3.OperationalError as exc:
            stats.warnings.append(f'chunking target migration partial failure: {exc}')

    if optional_tables.get('context_entries_fts'):
        fts_sql = _read_schema_file('add_fts_sqlite.sql')
        fts_sql = fts_sql.replace('{TOKENIZER}', fts_tokenizer)
        try:
            target.executescript(fts_sql)
        except sqlite3.OperationalError as exc:
            stats.warnings.append(f'FTS target migration partial failure: {exc}')

    # index_tree node-summary table: provisioned unconditionally so a migrated
    # target matches a server-initialized DB (the server creates it at startup when
    # ENABLE_INDEX_TREE_NODE_SUMMARIES is on, default true). Harmless when the
    # feature is later disabled -- read methods degrade to empty. Shares the server
    # migration's DDL (sqlite_index_tree_ddl) so the two cannot drift.
    from app.migrations.index_tree import sqlite_index_tree_ddl
    create_table_sql, create_index_sql = sqlite_index_tree_ddl()
    try:
        target.execute(create_table_sql)
        target.execute(create_index_sql)
    except sqlite3.OperationalError as exc:
        stats.warnings.append(f'index_tree target migration partial failure: {exc}')

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


def target_sqlite_is_compressed(path: str) -> bool:
    """Return True if the target SQLite file is configured for COMPRESSED embeddings.

    Probes the REAL target file (never the dry-run ``:memory:`` handle) for either
    marker of the compressed embedding layout: a populated ``compression_metadata``
    provenance row, or the ``vec_context_embeddings_compressed`` payload table. A
    target file that does not exist, is empty, or carries neither marker is treated
    as an fp32 (uncompressed) target.

    Args:
        path: Filesystem path to the target SQLite database file.

    Returns:
        True iff the target already carries the compressed embedding layout.
    """
    abs_path = Path(path).resolve()
    if not abs_path.exists():
        return False
    if abs_path.stat().st_size == 0:
        return False
    conn = sqlite3.connect(str(abs_path))
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('compression_metadata', 'vec_context_embeddings_compressed')",
        )
        present = {str(row[0]) for row in cursor.fetchall()}
        if 'vec_context_embeddings_compressed' in present:
            return True
        if 'compression_metadata' not in present:
            return False
        row = conn.execute('SELECT COUNT(*) FROM compression_metadata WHERE id = 1').fetchone()
        return row is not None and int(row[0]) > 0
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

    if target_already_has_data_sqlite(target_address):
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

        # Defensive backstop (never silently drop embeddings), symmetric with the
        # PostgreSQL runner's pre-existing-target check. On PostgreSQL a compressed
        # target manifests as a MISSING vec_context_embeddings table, which that
        # runner detects directly. On SQLite the same condition is MASKED:
        # initialize_target_sqlite re-executes add_semantic_search_sqlite.sql, whose
        # CREATE VIRTUAL TABLE IF NOT EXISTS re-creates the fp32 vec0 table, so the
        # 'target lacks vec_context_embeddings' warning below can never fire and the
        # migrated fp32 vectors land in a database that already carries compression
        # provenance. The next server start then applies the compression migration,
        # whose leading DROP TABLE IF EXISTS vec_context_embeddings destroys every
        # migrated vector -- and --compress in between is a no-op, because it
        # early-returns on the existing provenance row. Probe the REAL target file
        # (the dry-run handle is an in-memory database that would see nothing) and
        # refuse BEFORE initialize_target_sqlite masks the condition.
        source_has_embeddings = bool(
            optional_tables.get('embedding_metadata') or optional_tables.get('vec_context_embeddings'),
        )
        if source_has_embeddings and target_sqlite_is_compressed(target_address):
            message = (
                'source has embeddings but the target database is already configured '
                'for compressed embeddings (it carries a compression_metadata '
                'provenance row and/or a vec_context_embeddings_compressed table). '
                'The fp32 vectors this migration copies would be destroyed the next '
                'time the server applies the compression migration, and --compress '
                'would not encode them (it is a no-op while a provenance row exists). '
                'Use an empty target file (this CLI initializes it), or run '
                'mcp-context-server-migrate --decompress against the target first to '
                'clear its compression provenance, then rerun the migration and '
                'finish with --compress.'
            )
            if options.dry_run:
                stats.warnings.append(f'{message} (a real run would abort)')
            else:
                stats.errors.append(f'{message} Aborting to avoid silently dropping embeddings.')
                return stats

        cursor = source.execute(
            'SELECT id, created_at FROM context_entries ORDER BY created_at ASC, id ASC',
        )
        source_rows = cursor.fetchall()

        if source_rows:
            first_created_at = _created_at_for_id(source_rows[0]['created_at'])
            if first_created_at.microsecond == 0:
                logger.info(
                    'source created_at precision appears to be seconds; '
                    'sub-second ordering will use UUIDv7 random tails',
                )

        id_mapping = build_id_mapping(source_rows)

        from app.repositories.fts_repository import desired_sqlite_fts_tokenizer
        from app.settings import get_settings

        target = _open_sqlite_target(target_address, options.dry_run)
        initialize_target_sqlite(
            target,
            optional_tables,
            embedding_dim,
            # Derive the FTS tokenizer from FTS_LANGUAGE via the shared source of truth so a
            # CLI-migrated target matches what the server would build (a non-English language
            # gets plain unicode61, not the English Porter stemmer).
            fts_tokenizer=desired_sqlite_fts_tokenizer(get_settings().fts.language),
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


async def _target_pg_has_data(
    conn: 'asyncpg.Connection[asyncpg.Record]',
    schema: str | None = None,
) -> bool:
    """Return True if the PostgreSQL target ``context_entries`` table
    has any rows. Returns False when the table does not exist.

    ``schema`` MUST be the configured ``POSTGRESQL_SCHEMA`` for a TARGET probe so both
    the existence check and the ``COUNT(*)`` resolve EXPLICITLY against the schema the
    migration will WRITE to -- not via ``current_schema()``. ``current_schema()`` returns
    the first EXISTING schema in the ``search_path``, so before a non-default target schema
    is created it falls back to ``public``: the empty-target check would then read
    ``public.context_entries`` and either (a) falsely abort a legitimate migration when
    ``public`` already holds rows, or (b) when ``public`` is empty, let the caller's
    ``target_initialized`` probe also resolve to ``public`` and skip schema creation so the
    data copy silently writes to the wrong schema. Binding the configured schema explicitly
    fixes both. ``schema=None`` keeps the ``current_schema()`` form, correct for a SOURCE
    probe whose configured schema already exists.

    Returns:
        True iff the target already has rows.
    """
    if not await _pg_table_exists(conn, 'context_entries', schema=schema):
        return False
    if schema is None:
        count = await conn.fetchval('SELECT COUNT(*) FROM context_entries')
    else:
        # The schema is a SQL identifier (a table qualifier) that cannot be bound as a
        # parameter, so it must be quoted as an identifier. Route it through the shared
        # quote_pg_identifier helper -- the same one CREATE SCHEMA uses -- so an embedded
        # double-quote in POSTGRESQL_SCHEMA is doubled correctly and the two sites cannot
        # disagree on the same name (lazy import mirrors initialize_target_postgresql so
        # SQLite-only migration paths never import the PostgreSQL backend).
        from app.backends.postgresql_backend import quote_pg_identifier

        count = await conn.fetchval(f'SELECT COUNT(*) FROM {quote_pg_identifier(schema)}.context_entries')
    return int(count or 0) > 0


def _pg_connect_kwargs() -> dict[str, Any]:
    """Return the shared asyncpg connect kwargs for the migration CLI.

    Imported lazily so the SQLite-only migration paths never import the
    PostgreSQL backend (and therefore never require asyncpg/pgvector to be
    installed). The kwargs apply ``statement_cache_size`` (set
    ``POSTGRESQL_STATEMENT_CACHE_SIZE=0`` for transaction-mode poolers such as the
    Supabase Transaction Pooler). SSL is carried by the DSN (``?sslmode=...``),
    parsed natively by asyncpg. The SESSION parameters (``search_path``,
    ``extra_float_digits``, the TCP keepalive GUCs) are NOT startup-packet
    parameters -- a pooler would refuse the connection over them -- and are applied
    by :func:`_pg_connect` right after the dial instead.

    ``timeout`` (``POSTGRESQL_CONNECT_TIMEOUT_S``) is added here rather than by
    :func:`app.backends.postgresql_backend.build_asyncpg_connect_kwargs`, whose scope
    is what the pool merges into ``create_pool``; the pool supplies the same
    establishment budget separately. Without it every migration connection would
    silently fall back to asyncpg's built-in 60-second default, so a DSN whose TLS and
    startup handshake needs the longer budget the operator configured would boot the
    server fine yet abort the migration -- and a deliberately SHORT budget would not
    fail fast either.

    Returns:
        Mapping suitable for spreading into ``asyncpg.connect(dsn, **kwargs)``.
    """
    from app.backends.postgresql_backend import build_asyncpg_connect_kwargs
    from app.settings import get_settings

    settings = get_settings()
    kwargs = build_asyncpg_connect_kwargs(settings)
    kwargs['timeout'] = settings.storage.postgresql_connect_timeout_s
    return kwargs


async def _pg_connect(dsn: str) -> 'asyncpg.Connection[asyncpg.Record]':
    """Open a PostgreSQL connection configured exactly like the server's pool ones.

    The single dial point for every PostgreSQL connection the migration CLI opens.
    Dialing and configuring in one place is what keeps the CLI's sessions equivalent
    to the server's: the server's pool applies its session parameters through the
    pool ``setup`` callback, which a one-off connection never runs, so a CLI
    connection that only spread the connect kwargs would resolve bare table names
    through the server's default ``search_path`` rather than ``POSTGRESQL_SCHEMA``
    and would read float8 text at the server's default precision.

    Args:
        dsn: The PostgreSQL connection URL.

    Returns:
        The established connection, with its session parameters already applied.
    """
    import asyncpg

    from app.backends.postgresql_backend import apply_session_gucs

    conn: asyncpg.Connection[asyncpg.Record] = await asyncpg.connect(dsn, **_pg_connect_kwargs())
    try:
        await apply_session_gucs(conn)
    except BaseException:
        await conn.close()
        raise
    return conn


async def _pg_table_exists(
    conn: 'asyncpg.Connection[asyncpg.Record]',
    table_name: str,
    schema: str | None = None,
) -> bool:
    """Return True if ``table_name`` exists in the resolved schema.

    When ``schema`` is None the probe uses ``current_schema()`` (correct for a SOURCE
    connection, whose configured ``POSTGRESQL_SCHEMA`` already exists). When ``schema`` is
    given the probe binds that name EXPLICITLY -- required for a TARGET probe, because
    ``current_schema()`` returns the first EXISTING schema in the ``search_path`` and a
    not-yet-created non-default ``POSTGRESQL_SCHEMA`` would silently fall back to ``public``,
    making the probe inspect the wrong schema (see :func:`_target_pg_has_data`).

    Returns:
        True iff the table exists in the resolved schema.
    """
    if schema is None:
        result = await conn.fetchval(
            'SELECT EXISTS (SELECT 1 FROM information_schema.tables '
            'WHERE table_schema = current_schema() AND table_name = $1)',
            table_name,
        )
    else:
        result = await conn.fetchval(
            'SELECT EXISTS (SELECT 1 FROM information_schema.tables '
            'WHERE table_schema = $1 AND table_name = $2)',
            schema,
            table_name,
        )
    return bool(result)


async def _pg_column_exists(
    conn: 'asyncpg.Connection[asyncpg.Record]',
    table_name: str,
    column_name: str,
    schema: str | None = None,
) -> bool:
    """Return True if ``column_name`` exists on ``table_name`` in the resolved schema.

    Mirrors :func:`_pg_table_exists`'s schema-resolution contract: ``None`` uses
    ``current_schema()`` (correct for a SOURCE connection whose configured
    ``POSTGRESQL_SCHEMA`` already exists); an explicit ``schema`` binds that name
    directly. Used to guard the source SELECT against a v2 PostgreSQL source that
    predates the ``summary`` / ``content_hash`` ALTER-TABLE columns, mirroring the
    SQLite :func:`_table_has_column` guard so every migration direction tolerates
    their absence identically (the source is opened read-only and is never
    auto-migrated, so a missing column would otherwise raise UndefinedColumnError
    and abort the whole migration).

    Returns:
        True iff the column exists on the table in the resolved schema.
    """
    if schema is None:
        result = await conn.fetchval(
            'SELECT EXISTS (SELECT 1 FROM information_schema.columns '
            'WHERE table_schema = current_schema() AND table_name = $1 AND column_name = $2)',
            table_name,
            column_name,
        )
    else:
        result = await conn.fetchval(
            'SELECT EXISTS (SELECT 1 FROM information_schema.columns '
            'WHERE table_schema = $1 AND table_name = $2 AND column_name = $3)',
            schema,
            table_name,
            column_name,
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
            WHERE table_schema = current_schema()
              AND table_name = 'embedding_metadata' AND column_name = 'chunk_count'
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
            WHERE table_schema = current_schema()
              AND table_name = 'embedding_metadata' AND column_name = 'chunk_count'
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
            WHERE table_schema = current_schema()
              AND table_name = 'vec_context_embeddings'
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
            WHERE table_schema = current_schema()
              AND table_name = 'vec_context_embeddings' AND column_name = 'start_index'
        )
        ''',
    )
    has_boundaries_tgt = await target.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'vec_context_embeddings' AND column_name = 'start_index'
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
    source_has_fts: bool,
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
            vector column. Ignored when ``with_semantic`` is False. When None on
            a ``with_semantic`` run (the source ``embedding_metadata`` table
            exists but is empty), this function resolves the same
            ``settings.embedding.dim`` fallback the semantic migration would have
            applied and uses that resolved value everywhere -- for the
            capacity pre-flight, for the migration call, and for the reported
            dimension -- so the report can never disagree with the built DDL.
        with_semantic: When True, also create the semantic-search and chunking
            layout (PG->PG migrations that copy embeddings). When False (a
            cross-backend migration that drops embeddings), only the base schema
            and the PostgreSQL helper functions are created; the server creates
            the vector layout later, at the operator's configured dimension,
            when re-embedding.
        source_has_fts: When True, provision the tsvector FTS column + GIN index
            on the target regardless of the CLI process's ENABLE_FTS setting,
            mirroring the SQLite CLI target's source-presence gate. When False
            (the source had no FTS) it is left unprovisioned.
        stats: Mutated to record an informational warning describing the
            auto-init.

    Raises:
        RuntimeError: If a ``with_semantic`` run's effective embedding dimension
            (``embedding_dim`` when known, else the ``settings.embedding.dim``
            fallback) exceeds the pgvector index cap (raised BEFORE any target
            DDL, so the target schema is never left partially initialized); if
            the target schema cannot be created; or -- on a ``with_semantic``
            run -- if the pgvector extension cannot be created (insufficient
            privileges on a managed service such as Supabase, or pgvector not
            installed on the host at all). A ``with_semantic=False`` run never
            touches pgvector, so it initializes cleanly on a pgvector-less host.
    """
    # fp32 capability pre-flight, BEFORE any connection or target DDL: pgvector
    # cannot build an HNSW index over vector columns wider than
    # PGVECTOR_INDEX_DIM_LIMIT dimensions, and the semantic-search migration
    # this function force-applies templates a dimension into vector(dim) and then
    # builds that index. Without this check the run dies mid-pipeline at CREATE
    # INDEX and leaves the target schema partially initialized (base tables
    # created, vector layout half-built).
    #
    # The dimension the DDL will ACTUALLY use is the source-detected value when
    # known, else apply_semantic_search_migration's own settings.embedding.dim
    # fallback (it resolves embedding_dim=None to settings.embedding.dim). A
    # source whose embedding_metadata table exists but is empty detects as None,
    # so validating only the source-detected value would leave the settings
    # fallback -- the value the DDL templates -- unchecked and crash mid-index.
    # Resolve the same fallback here and validate whichever value the DDL uses.
    #
    # effective_dim is resolved ONCE here and then used for the pre-flight, for the
    # dimension passed to the semantic migration, and for the final report line, so
    # the reported dimension can never disagree with the dimension the DDL built.
    effective_dim: int | None = None
    if with_semantic:
        effective_dim = embedding_dim
        if effective_dim is None:
            from app.settings import get_settings

            effective_dim = get_settings().embedding.dim
        if exceeds_pgvector_index_dim_limit(effective_dim):
            source_clause = (
                f'source embedding dimension ({effective_dim})'
                if embedding_dim is not None
                else (
                    f'configured EMBEDDING_DIM ({effective_dim}, the fallback used because '
                    'the source embedding_metadata table is empty)'
                )
            )
            raise RuntimeError(
                f'Cannot initialize the target database: the {source_clause} exceeds the '
                f'pgvector index limit of {PGVECTOR_INDEX_DIM_LIMIT} dimensions for fp32 '
                f'vectors, so building the fp32 vector layout would fail at the HNSW '
                f'CREATE INDEX and leave the target schema partially initialized. '
                f'Embeddings of this dimension cannot be copied into an fp32 target. '
                f'Recovery: migrate from a source copy whose embedding tables '
                f'(embedding_metadata, vec_context_embeddings) are dropped so the target '
                f'initializes without the fp32 vector layout, then start the target server '
                f'with ENABLE_EMBEDDING_COMPRESSION=true (compressed payloads have no '
                f'pgvector dimension cap) and re-embed with --embed-missing.',
            )

    import asyncpg

    from app.backends import create_backend
    from app.backends.postgresql_backend import quote_pg_identifier
    from app.migrations.chunking import apply_chunking_migration
    from app.migrations.fts import apply_fts_migration
    from app.migrations.index_tree import apply_index_tree_migration
    from app.migrations.semantic import apply_function_search_path_migration
    from app.migrations.semantic import apply_jsonb_merge_patch_migration
    from app.migrations.semantic import apply_semantic_search_migration
    from app.settings import get_settings
    from app.startup import init_database

    schema = get_settings().storage.postgresql_schema

    # The target schema must exist before the schema-qualified function DDL in
    # the base schema / migrations runs (CREATE FUNCTION "<schema>".update_...);
    # PostgreSQL does not auto-create a non-default schema, so it is created
    # first and UNCONDITIONALLY. The pgvector extension is needed ONLY when the
    # fp32 vector layout will be built (with_semantic): a cross-backend
    # migration drops embeddings and must initialize cleanly on a pgvector-less
    # host (the pgvector-free compressed deployment shape), so it never issues
    # CREATE EXTENSION at all. When the extension IS needed it must exist
    # before the semantic migration's vector(dim) DDL and the backend pool's
    # vector-codec registration. Surface a clear, actionable error on managed
    # services where DDL privileges are restricted and on hosts where the
    # pgvector extension is not installed at all (missing control file --
    # IF NOT EXISTS does not suppress that failure).
    ext_conn = await _pg_connect(target_url)
    try:
        try:
            await ext_conn.execute(f'CREATE SCHEMA IF NOT EXISTS {quote_pg_identifier(schema)}')
        except asyncpg.InsufficientPrivilegeError as exc:
            raise RuntimeError(
                'Cannot initialize the target database (insufficient privileges '
                f'to CREATE SCHEMA "{schema}"). Create the schema first, then '
                f'rerun: execute \'CREATE SCHEMA "{schema}";\' as a privileged user.',
            ) from exc
        if with_semantic:
            try:
                await ext_conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            except asyncpg.InsufficientPrivilegeError as exc:
                raise RuntimeError(
                    'Cannot initialize the target database (insufficient '
                    'privileges to CREATE EXTENSION vector). This migration '
                    'copies embeddings, which require pgvector. Enable it first, '
                    'then rerun: on Supabase use Dashboard -> Database -> '
                    'Extensions -> vector; on self-hosted PostgreSQL run '
                    '"CREATE EXTENSION vector;" as a superuser.',
                ) from exc
            except asyncpg.UndefinedFileError as exc:
                raise RuntimeError(
                    'Cannot initialize the target database: the pgvector '
                    'extension is not installed on the target PostgreSQL host. '
                    'This migration copies embeddings, which require pgvector. '
                    'Install it on the host first (for example use a '
                    'pgvector/pgvector image or the PostgreSQL pgvector package), '
                    'then rerun.',
                ) from exc
    finally:
        await ext_conn.close()

    # provision_vector mirrors with_semantic: a vector-carrying target has its
    # extension guaranteed by the block above (created or failed loudly), while
    # a vector-free target must not let the CLI process's env-driven gate force
    # pgvector provisioning it cannot satisfy on a pgvector-less host.
    backend = create_backend(
        backend_type='postgresql',
        connection_string=target_url,
        provision_vector=with_semantic,
    )
    await backend.initialize()
    try:
        await init_database(backend=backend)
        if with_semantic:
            await apply_semantic_search_migration(backend, force=True, embedding_dim=effective_dim)
        await apply_jsonb_merge_patch_migration(backend)
        await apply_function_search_path_migration(backend)
        # FTS: create the tsvector GENERATED column + GIN index ONLY when the
        # SOURCE had full-text search, mirroring the SQLite CLI target's
        # source-presence gate (initialize_target_sqlite keys FTS on
        # optional_tables['context_entries_fts']). force=True bypasses the CLI
        # process's ENABLE_FTS toggle so the migrated target's FTS capability is
        # decided solely by the source -- otherwise a SQLite->PG or PG->PG
        # migration run with ENABLE_FTS=false would silently drop the FTS the
        # SQLite target keeps. It MUST run BEFORE the data copy so the STORED
        # generated column auto-populates as rows are INSERTed.
        if source_has_fts:
            await apply_fts_migration(backend, force=True)
        if with_semantic:
            await apply_chunking_migration(backend, force=True)
        # index_tree node-summary table: provisioned regardless of with_semantic
        # (it concerns node summaries, not vectors) so a migrated target matches a
        # server-initialized DB. force=True mirrors the SQLite path; the table is
        # harmless when the node-summary feature is later disabled.
        await apply_index_tree_migration(backend, force=True)
    finally:
        await backend.shutdown()

    # Report the RESOLVED dimension the vector column was actually built at, not the
    # raw parameter: a source whose embedding_metadata table is empty detects as
    # None and the semantic migration falls back to settings.embedding.dim, so
    # printing the parameter would tell the operator that an irreversible schema
    # decision was made at an unknown width.
    stats.warnings.append(
        'auto-initialized target PostgreSQL schema '
        f'(semantic_search={"yes" if with_semantic else "no"}, '
        f'embedding_dim={effective_dim if with_semantic else "n/a"})',
    )


async def ensure_target_pg_fts(
    target_url: str,
    target_conn: 'asyncpg.Connection[asyncpg.Record]',
    *,
    target_schema: str,
    source_has_fts: bool,
    dry_run: bool,
    stats: MigrationStats,
) -> None:
    """Provision FTS on a PRE-EXISTING PostgreSQL target when the source has it.

    :func:`initialize_target_postgresql` provisions FTS only when it runs --
    that is, only when the target had NO ``context_entries`` table at all. A
    pre-existing target (its schema bootstrapped by a server started with
    ``ENABLE_FTS=false``, or created by any means other than this CLI)
    silently lost the full-text search the source had -- the exact regression
    class the source-presence gate closed for freshly initialized targets.
    Unlike the embeddings backstop, which must ABORT (vectors are not
    derivable from the copied rows), FTS is fully derivable: the migration
    adds a STORED generated ``text_search_vector`` column plus its GIN index,
    so the backstop PROVISIONS it instead. It MUST run before the data copy
    so the generated column populates as rows are INSERTed. Mirrors the
    SQLite target path, which re-applies the IF-NOT-EXISTS FTS DDL keyed only
    on source presence.

    Args:
        target_url: asyncpg DSN for the target database.
        target_conn: Open target connection used for the column probe.
        target_schema: Explicit schema for the probe, matching the
            ``context_entries`` probe that established the target as
            pre-existing.
        source_has_fts: Whether the source carries full-text search.
        dry_run: When True, record the plan instead of provisioning.
        stats: Mutated with the provisioning (or plan) note.
    """
    if not source_has_fts:
        return
    if await _pg_column_exists(
        target_conn, 'context_entries', 'text_search_vector', schema=target_schema,
    ):
        return
    if dry_run:
        stats.warnings.append(
            'source has full-text search but the pre-existing target lacks the '
            'text_search_vector column; it would be provisioned on a real run',
        )
        return

    from app.backends import create_backend
    from app.migrations.fts import apply_fts_migration

    # provision_vector=False: this backend runs FTS DDL only (tsvector column +
    # GIN index), which never touches the vector type, and the pre-existing
    # target may be a pgvector-less host (a compressed deployment) -- the CLI
    # process's env-driven gate must not force pgvector provisioning here.
    backend = create_backend(
        backend_type='postgresql',
        connection_string=target_url,
        provision_vector=False,
    )
    await backend.initialize()
    try:
        await apply_fts_migration(backend, force=True)
    finally:
        await backend.shutdown()
    stats.warnings.append(
        'provisioned full-text search on the pre-existing target '
        '(source has FTS; the target lacked the text_search_vector column)',
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
    source_conn = await _pg_connect(options.source_url)
    # Open the target connection INSIDE the try so a failed target connect
    # (unreachable host, bad credentials, role/connection limit, SSL) closes the
    # already-open source connection via the finally instead of leaking it.
    target_conn: asyncpg.Connection | None = None
    try:
        target_conn = await _pg_connect(options.target_url)
        await source_conn.execute('BEGIN TRANSACTION READ ONLY')

        # Bind the configured POSTGRESQL_SCHEMA EXPLICITLY for every TARGET probe so they
        # inspect the schema the migration will WRITE to even before it is created --
        # current_schema() would fall back to public and mis-resolve a non-default schema
        # (false abort, or silent wrong-schema copy). See _target_pg_has_data.
        from app.settings import get_settings
        target_schema = get_settings().storage.postgresql_schema

        if await _target_pg_has_data(target_conn, schema=target_schema):
            stats.errors.append(
                'target PostgreSQL database already contains context_entries rows. '
                'Recovery: if a prior run was interrupted, drop and recreate the target database '
                '(or pass a different --target-url) and rerun; the source database is unchanged. '
                'See the Recovering From an Interrupted Migration section of docs/migration-v2-to-v3.md.',
            )
            return stats

        id_column_type = await source_conn.fetchval(
            'SELECT data_type FROM information_schema.columns '
            "WHERE table_schema = current_schema() AND table_name = 'context_entries' AND column_name = 'id'",
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
        null_created_at = 0
        for row in source_rows:
            if row['created_at'] is None:
                null_created_at += 1
            id_mapping[int(row['id'])] = generate_id_with_timestamp(_created_at_for_id(row['created_at']))
        if null_created_at:
            logger.warning(
                '%d source context_entries row(s) had NULL created_at; their ids '
                'were anchored to %s (the stored created_at is preserved as NULL)',
                null_created_at,
                _NULL_CREATED_AT_ANCHOR.isoformat(),
            )

        # Detect what the SOURCE carries so the target can be shaped to match.
        source_has_embeddings = await _pg_table_exists(source_conn, 'embedding_metadata')
        # The vector table is detected INDEPENDENTLY of embedding_metadata: a source can
        # carry the metadata table without a vec_context_embeddings table (e.g. semantic
        # search was provisioned but never populated, or the vec table was dropped). The
        # vector copy and its dry-run COUNT must gate on THIS, not on embedding_metadata,
        # or `SELECT ... FROM vec_context_embeddings` crashes the whole migration -- the
        # SQLite path already detects the vec table separately.
        source_has_vec = await _pg_table_exists(source_conn, 'vec_context_embeddings')
        source_dim = await _detect_source_embedding_dim_pg(source_conn)
        # FTS on PostgreSQL is the text_search_vector generated column on
        # context_entries (no separate table); detect it so the auto-init
        # provisions target FTS iff the source had it, mirroring the SQLite path.
        source_has_fts = await _pg_column_exists(source_conn, 'context_entries', 'text_search_vector')

        # Auto-initialize the target schema when it has no context_entries table,
        # mirroring the SQLite path (initialize_target_sqlite). This removes the
        # trap where the user had to manually pre-create the fp32 layout: the
        # target is built with ENABLE_EMBEDDING_COMPRESSION effectively off (the
        # compression migration is never run here), so copy_vec_embeddings_pg has
        # an fp32 vec_context_embeddings table to write into. Compression is a
        # separate, explicit --compress step afterward.
        target_initialized = await _pg_table_exists(target_conn, 'context_entries', schema=target_schema)
        if not target_initialized:
            # fp32 capability pre-flight for the auto-init: an embedding-carrying
            # source whose dimension exceeds the pgvector index cap cannot be
            # copied into the fp32 vector layout the auto-init would force-build
            # (the HNSW CREATE INDEX would fail mid-pipeline and leave a
            # partially initialized target). Refuse HERE, before any target DDL,
            # with a recorded error (clean exit 1) instead of letting
            # initialize_target_postgresql's own guard surface as an unhandled
            # exception (exit 2); a dry run reports the same refusal as a plan
            # warning, mirroring the pre-existing-target embeddings backstop.
            #
            # source_dim is None when the source embedding_metadata table exists
            # but is empty; the auto-init's semantic migration then falls back to
            # settings.embedding.dim, so the value the target DDL actually
            # templates is that fallback. Resolve it here (mirroring
            # initialize_target_postgresql's own pre-flight) so an over-limit
            # fallback is refused before any DDL and surfaced under --dry-run,
            # instead of slipping past the source_dim-only check to crash the
            # real run mid-index.
            effective_source_dim = source_dim
            fallback_dim_used = False
            if source_has_embeddings and effective_source_dim is None:
                from app.settings import get_settings

                effective_source_dim = get_settings().embedding.dim
                fallback_dim_used = True
            if (
                source_has_embeddings
                and effective_source_dim is not None
                and exceeds_pgvector_index_dim_limit(effective_source_dim)
            ):
                dim_clause = (
                    f'configured EMBEDDING_DIM ({effective_source_dim}, the fallback used '
                    'because the source embedding_metadata table is empty)'
                    if fallback_dim_used
                    else f'source embedding dimension ({effective_source_dim})'
                )
                message = (
                    f'{dim_clause} exceeds the pgvector index limit of '
                    f'{PGVECTOR_INDEX_DIM_LIMIT} dimensions for fp32 vectors: '
                    'auto-initializing the target would fail at the HNSW '
                    'CREATE INDEX and leave the target schema partially initialized. '
                    'Recovery: migrate from a source copy whose embedding tables '
                    '(embedding_metadata, vec_context_embeddings) are dropped so the '
                    'target initializes without the fp32 vector layout, then start the '
                    'target server with ENABLE_EMBEDDING_COMPRESSION=true (compressed '
                    'payloads have no pgvector dimension cap) and re-embed with '
                    '--embed-missing.'
                )
                if options.dry_run:
                    stats.warnings.append(f'{message} (a real run would abort)')
                else:
                    stats.errors.append(f'{message} Aborting before any target DDL.')
                    return stats
            elif options.dry_run:
                stats.warnings.append(
                    'target PostgreSQL database has no context_entries table; '
                    'it would be auto-initialized on a real run',
                )
            else:
                await initialize_target_postgresql(
                    options.target_url,
                    embedding_dim=source_dim,
                    with_semantic=source_has_embeddings,
                    source_has_fts=source_has_fts,
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

        # FTS backstop for a PRE-EXISTING target: initialize_target_postgresql
        # runs only when context_entries is absent, so a target bootstrapped by
        # other means would silently lose the source's full-text search.
        if target_initialized:
            await ensure_target_pg_fts(
                options.target_url,
                target_conn,
                target_schema=target_schema,
                source_has_fts=source_has_fts,
                dry_run=options.dry_run,
                stats=stats,
            )

        if not options.dry_run:
            await target_conn.execute('BEGIN')
        try:
            # Guard summary / content_hash: a v2 PostgreSQL source predating those
            # ALTER-TABLE columns (and never re-run against the server, so the
            # auto-migrations never fired) lacks them. The source is read-only and is
            # never auto-migrated here, so naming the columns unconditionally would
            # raise UndefinedColumnError and abort the whole migration -- whereas the
            # SQLite source path (copy_context_entries) already guards via
            # _table_has_column. Substitute NULL when absent to keep the row keys and
            # the INSERT below unchanged, giving every direction identical tolerance.
            summary_col_src = (
                'summary' if await _pg_column_exists(source_conn, 'context_entries', 'summary')
                else 'NULL AS summary'
            )
            content_hash_col_src = (
                'content_hash' if await _pg_column_exists(source_conn, 'context_entries', 'content_hash')
                else 'NULL AS content_hash'
            )
            entry_rows = await source_conn.fetch(
                f'SELECT id, thread_id, source, content_type, text_content, '
                f'metadata::text AS metadata, {summary_col_src}, {content_hash_col_src}, created_at, updated_at '
                f'FROM context_entries ORDER BY created_at ASC, id ASC',
            )
            # Source ids whose context_entries row was skipped. Their children must be
            # skipped too: a tag, attachment or embedding row pointing at an id the
            # target never received would violate the foreign key and abort the run --
            # replacing one skipped row with a total failure.
            pg_skipped_context_ids: set[int] = set()
            for entry in entry_rows:
                source_id = int(entry['id'])
                new_id = id_mapping[source_id]
                references_rewritten_before = stats.references_rewritten
                rewritten_metadata = rewrite_metadata_references(
                    entry['metadata'],
                    id_mapping,
                    stats,
                    source_id,
                )
                # The metadata expression indexes are deliberately absent from the
                # PostgreSQL base schema (a database this CLI initialized has none until
                # its first server startup), so a PostgreSQL SOURCE can legitimately hold
                # a value the TARGET's idx_metadata_<field> cannot index -- an oversized
                # one, or one the index's cast rejects. Unchecked, that value aborts the
                # INSERT mid-transaction and rolls the whole run back with a raw driver
                # error naming no source row. thread_id and tags need no equivalent check
                # here: their indexes ARE in the base schema, so the source could not have
                # stored a value that breaches them. Checked unconditionally so --dry-run
                # surfaces the row too.
                unindexable = _first_pg_unindexable_metadata_field(rewritten_metadata)
                if unindexable is not None:
                    column, reason = unindexable
                    stats.errors.append(
                        f'context_entries row id={source_id} thread_id={entry["thread_id"]!r} '
                        f'column {column!r} skipped: {reason}',
                    )
                    stats.references_rewritten = references_rewritten_before
                    pg_skipped_context_ids.add(source_id)
                    continue
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
                await source_conn.fetch(_SELECT_DISTINCT_TAGS_SQL)
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
                if source_id in pg_skipped_context_ids:
                    stats.warnings.append(
                        f'tags row context_entry_id={source_id} skipped: parent context_entries row was skipped',
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
                if source_id in pg_skipped_context_ids:
                    stats.warnings.append(
                        f'image_attachments row context_entry_id={source_id} skipped: '
                        f'parent context_entries row was skipped',
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
            if source_has_embeddings and not source_has_vec:
                stats.warnings.append(
                    'source PostgreSQL database has an embedding_metadata table but no '
                    'vec_context_embeddings table; vector rows not copied (re-embed the '
                    'target afterward). Metadata rows are still migrated.',
                )
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
                    if source_has_vec:
                        stats.vec_rows_migrated = int(
                            await source_conn.fetchval('SELECT COUNT(*) FROM vec_context_embeddings') or 0,
                        )
                else:
                    # A skipped parent's embedding rows are excluded the same way its
                    # tags and attachments are: the copies resolve their parent through
                    # this mapping, so dropping the skipped ids from it turns an FK
                    # violation that would abort the run into their own skip-and-warn.
                    embedding_id_mapping = {
                        source_key: target_key
                        for source_key, target_key in id_mapping.items()
                        if source_key not in pg_skipped_context_ids
                    }
                    await copy_embedding_metadata_pg(
                        source_conn, target_conn, embedding_id_mapping, stats, options.dry_run,
                    )
                    # Gate the vector copy on the SOURCE vec table (copy_vec_embeddings_pg
                    # reads FROM vec_context_embeddings, which would crash if absent).
                    if source_has_vec:
                        await copy_vec_embeddings_pg(
                            source_conn, target_conn, embedding_id_mapping, stats, options.dry_run,
                        )

            if not options.dry_run:
                await target_conn.execute('COMMIT')
        except Exception:
            if not options.dry_run:
                await target_conn.execute('ROLLBACK')
            raise
    finally:
        await source_conn.close()
        if target_conn is not None:
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
    # Open the target connection INSIDE the try so a failed target connect closes
    # the already-open SQLite source via the finally instead of leaking it.
    target_conn: asyncpg.Connection | None = None
    try:
        target_conn = await _pg_connect(options.target_url)
        id_kind = detect_source_id_kind(source)
        if id_kind != 'integer':
            stats.warnings.append(
                f'source database id column is {id_kind!r}; nothing to migrate',
            )
            return stats

        optional_tables = detect_optional_tables(source)

        # Bind the configured POSTGRESQL_SCHEMA EXPLICITLY for every TARGET probe (see
        # _target_pg_has_data and run_migration_postgresql): current_schema() would fall
        # back to public before a non-default target schema exists.
        from app.settings import get_settings
        target_schema = get_settings().storage.postgresql_schema

        if await _target_pg_has_data(target_conn, schema=target_schema):
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
        target_initialized = await _pg_table_exists(target_conn, 'context_entries', schema=target_schema)
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
                    source_has_fts=optional_tables.get('context_entries_fts', False),
                    stats=stats,
                )

        # FTS backstop for a PRE-EXISTING target (mirrors the PG->PG path):
        # the auto-init above runs only when context_entries is absent.
        if target_initialized:
            await ensure_target_pg_fts(
                options.target_url,
                target_conn,
                target_schema=target_schema,
                source_has_fts=optional_tables.get('context_entries_fts', False),
                dry_run=options.dry_run,
                stats=stats,
            )

        if not options.dry_run:
            await target_conn.execute('BEGIN')
        try:
            # Guard summary / content_hash on the SQLite source (a v2 DB predating
            # those ALTER-TABLE columns lacks them), mirroring copy_context_entries
            # and the PostgreSQL source paths so all four directions tolerate their
            # absence identically. NULL substitution keeps the row keys and INSERT below.
            summary_col_src = (
                'summary' if _table_has_column(source, 'context_entries', 'summary') else 'NULL AS summary'
            )
            content_hash_col_src = (
                'content_hash' if _table_has_column(source, 'context_entries', 'content_hash')
                else 'NULL AS content_hash'
            )
            entry_cursor = source.execute(
                f'SELECT id, thread_id, source, content_type, text_content, metadata, '
                f'{summary_col_src}, {content_hash_col_src}, created_at, updated_at FROM context_entries '
                f'ORDER BY created_at ASC, id ASC',
            )
            # Source ids of context_entries rows skipped for a PostgreSQL-unstorable
            # value. Their tags/image_attachments children must be skipped too: the
            # parent id stays in id_mapping (so the orphan-FK check would not catch
            # them), and inserting a child that references a never-inserted parent
            # would raise an FK violation on PostgreSQL -- relocating the very abort
            # this guard prevents.
            skipped_context_ids: set[int] = set()
            for row in entry_cursor:
                source_id = int(row['id'])
                new_id = id_mapping[source_id]
                # Snapshot the rewrite counter so a row this loop ends up SKIPPING does
                # not leave its remappings counted: none of them reach the target.
                references_rewritten_before = stats.references_rewritten
                rewritten_metadata = rewrite_metadata_references(
                    row['metadata'],
                    id_mapping,
                    stats,
                    source_id,
                )
                # A NUL (U+0000) or unpaired UTF-16 surrogate is legal in a SQLite
                # TEXT value but fatal on the PostgreSQL target: without this check
                # the asyncpg bind raises mid-transaction, ROLLBACKs the whole run,
                # and reports only the raw driver error with no row identification.
                # Checked UNCONDITIONALLY (not behind dry_run) so --dry-run surfaces
                # every affected row before a real run; the offending row is
                # identified and skipped, mirroring the orphan-FK skip-and-warn.
                row_thread_id = row['thread_id']
                unstorable = _first_pg_unstorable_column(
                    (
                        ('thread_id', row_thread_id, False),
                        ('text_content', row['text_content'], False),
                        ('summary', row['summary'], False),
                        ('content_hash', row['content_hash'], False),
                        ('metadata', rewritten_metadata, True),
                    ),
                )
                if unstorable is None:
                    # Same skip-and-warn shape for the OTHER SQLite-accepts /
                    # PostgreSQL-rejects class on this path: a value SQLite indexed
                    # happily but the target's btree cannot hold. Only the columns the
                    # target schema itself indexes are checked; text_content and
                    # summary are unindexed and may be arbitrarily large. The metadata
                    # bound into the target is inspected too: every
                    # METADATA_INDEXED_FIELDS key is indexed by the expression index
                    # idx_metadata_<field>, under the same btree ceiling for a
                    # string-typed field and under a hard SQL cast for a typed one.
                    unstorable = _first_pg_unindexable_column(
                        (('thread_id', row_thread_id, _PG_MAX_INDEXED_THREAD_ID_BYTES),),
                    ) or _first_pg_unindexable_metadata_field(rewritten_metadata)
                if unstorable is not None:
                    column, reason = unstorable
                    stats.errors.append(
                        f'context_entries row id={source_id} thread_id={row_thread_id!r} '
                        f'column {column!r} skipped: {reason}',
                    )
                    stats.references_rewritten = references_rewritten_before
                    skipped_context_ids.add(source_id)
                    continue
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
                        _stored_datetime_or_none(row['created_at']),
                        _stored_datetime_or_none(row['updated_at']),
                    )
                stats.rows_migrated += 1

            # Copy tags and image attachments (portable across backends: tags
            # are TEXT, image payloads are BYTEA<->BLOB). Only the embedding
            # vectors are dropped cross-backend. Reads are guarded by source
            # table presence.
            if optional_tables.get('tags'):
                tag_cursor = source.execute(_SELECT_DISTINCT_TAGS_SQL)
                for tag_row in tag_cursor:
                    sid = int(tag_row['context_entry_id'])
                    mapped = id_mapping.get(sid)
                    if mapped is None:
                        stats.warnings.append(
                            f'tags row references missing context_entry_id={sid}; skipped',
                        )
                        continue
                    if sid in skipped_context_ids:
                        stats.warnings.append(
                            f'tags row context_entry_id={sid} skipped: parent context_entries row was skipped',
                        )
                        continue
                    # Skip a tag carrying a PostgreSQL-unstorable NUL/surrogate
                    # (see the context_entries guard above), unconditionally so
                    # --dry-run surfaces it too.
                    tag_unstorable = _first_pg_unstorable_column(
                        (('tag', tag_row['tag'], False),),
                    ) or _first_pg_unindexable_column(
                        # idx_tags_tag is a btree: a legacy tag longer than its
                        # index-tuple budget aborts the INSERT mid-transaction. The tag
                        # is indexed on its own, so it gets the full single-column budget.
                        (('tag', tag_row['tag'], _PG_MAX_INDEXED_VALUE_BYTES),),
                    )
                    if tag_unstorable is not None:
                        column, reason = tag_unstorable
                        stats.errors.append(
                            f'tags row context_entry_id={sid} column {column!r} skipped: {reason}',
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
                    if sid in skipped_context_ids:
                        stats.warnings.append(
                            f'image_attachments row context_entry_id={sid} skipped: '
                            f'parent context_entries row was skipped',
                        )
                        continue
                    # Skip an attachment whose mime_type or image_metadata carries a
                    # PostgreSQL-unstorable NUL/surrogate (see the context_entries
                    # guard above), unconditionally so --dry-run surfaces it too.
                    img_unstorable = _first_pg_unstorable_column(
                        (
                            ('mime_type', img['mime_type'], False),
                            ('image_metadata', img['image_metadata'], True),
                        ),
                    )
                    if img_unstorable is not None:
                        column, reason = img_unstorable
                        stats.errors.append(
                            f'image_attachments row context_entry_id={sid} column {column!r} skipped: {reason}',
                        )
                        continue
                    # A NULL or malformed created_at is preserved as NULL rather
                    # than crashing _coerce_datetime (the migration must not invent
                    # data for, nor abort on, arbitrary non-app source databases).
                    img_created_at = _stored_datetime_or_none(img['created_at'])
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
        if target_conn is not None:
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

    stats = MigrationStats()
    stats.warnings.append(
        'cross-backend migration drops vector embeddings; re-embed the target after migration',
    )

    _, target_address = parse_backend_url(options.target_url)
    if target_already_has_data_sqlite(target_address):
        stats.errors.append(
            f'target database already contains context_entries rows: {target_address}. '
            f'Recovery: if a prior run was interrupted, delete the target file and rerun; '
            f'the source database is unchanged. See the Recovering From an Interrupted Migration '
            f'section of docs/migration-v2-to-v3.md.',
        )
        return stats

    source_conn = await _pg_connect(options.source_url)
    target: sqlite3.Connection | None = None
    try:
        await source_conn.execute('BEGIN TRANSACTION READ ONLY')

        id_column_type = await source_conn.fetchval(
            'SELECT data_type FROM information_schema.columns '
            "WHERE table_schema = current_schema() AND table_name = 'context_entries' AND column_name = 'id'",
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
        null_created_at = 0
        for row in source_rows:
            if row['created_at'] is None:
                null_created_at += 1
            id_mapping[int(row['id'])] = generate_id_with_timestamp(_created_at_for_id(row['created_at']))
        if null_created_at:
            logger.warning(
                '%d source context_entries row(s) had NULL created_at; their ids '
                'were anchored to %s (the stored created_at is preserved as NULL)',
                null_created_at,
                _NULL_CREATED_AT_ANCHOR.isoformat(),
            )

        # Detect which optional tables the PostgreSQL source carries so the
        # SQLite target is shaped to match. FTS is offered on the target (it is
        # not portable from PostgreSQL, but the SQLite target supports it and
        # the index is rebuilt locally from the copied rows below).
        source_has_tags = await _pg_table_exists(source_conn, 'tags')
        source_has_images = await _pg_table_exists(source_conn, 'image_attachments')
        from app.repositories.fts_repository import desired_sqlite_fts_tokenizer
        from app.settings import get_settings

        target = _open_sqlite_target(target_address, options.dry_run)
        initialize_target_sqlite(
            target,
            optional_tables={
                'tags': True,
                'image_attachments': True,
                'context_entries_fts': True,
            },
            embedding_dim=None,
            # Derive the FTS tokenizer from FTS_LANGUAGE via the shared source of truth so the
            # PostgreSQL->SQLite target's rebuilt FTS index matches what the server would build
            # for the configured language, instead of a hardcoded English tokenizer.
            fts_tokenizer=desired_sqlite_fts_tokenizer(get_settings().fts.language),
            stats=stats,
        )

        # Guard summary / content_hash on the PostgreSQL source (a v2 DB predating
        # those ALTER-TABLE columns lacks them), mirroring copy_context_entries so
        # all four migration directions tolerate their absence identically. NULL
        # substitution keeps the row keys and the INSERT below unchanged.
        summary_col_src = (
            'summary' if await _pg_column_exists(source_conn, 'context_entries', 'summary')
            else 'NULL AS summary'
        )
        content_hash_col_src = (
            'content_hash' if await _pg_column_exists(source_conn, 'context_entries', 'content_hash')
            else 'NULL AS content_hash'
        )
        entry_rows = await source_conn.fetch(
            f'SELECT id, thread_id, source, content_type, text_content, '
            f'metadata::text AS metadata, {summary_col_src}, {content_hash_col_src}, created_at, updated_at '
            f'FROM context_entries ORDER BY created_at ASC, id ASC',
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
                            _sqlite_timestamp(row['created_at']),
                            _sqlite_timestamp(row['updated_at']),
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
                    _SELECT_DISTINCT_TAGS_SQL,
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
                    # Preserve a schema-legal NULL created_at as NULL, and render a
                    # present timestamp in SQLite's canonical space form (NOT isoformat's
                    # 'T'/offset, which mis-sorts under SQLite TEXT date comparison).
                    img_created_at = _sqlite_timestamp(img['created_at'])
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


def print_summary(stats: MigrationStats, source_url: str, target_url: str, dry_run: bool = False) -> None:
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
    if dry_run:
        print('Dry run: no changes were written to the target.')
    elif stats.rows_migrated > 0 and not stats.errors:
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
        errors, 2 on unrecoverable migration failure, 78 (EX_CONFIG) on a
        settings ValidationError -- the same classification the server's
        guarded import applies.
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
    #
    # Settings validation surfaces on these paths at the first get_settings()
    # call (transitively, at backend-module import), never through
    # app.server's guarded import: mcp-context-server-migrate is its own
    # console script and never imports app.server. Classify it here exactly
    # like the server does -- a permanent misconfiguration exits EX_CONFIG
    # (78) with the pydantic detail on stderr, instead of an unhandled
    # traceback and the generic exit 1 supervisors cannot distinguish from a
    # transient failure.
    try:
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
    except ValidationError as e:
        print(f'Configuration invalid: {e}', file=sys.stderr)
        return ConfigurationError.EXIT_CODE

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
    except ValidationError as exc:
        # Same classification as the in-place dispatch above: a settings
        # ValidationError is a permanent misconfiguration (EX_CONFIG), not a
        # migration failure worth the generic exit 2.
        print(f'Configuration invalid: {exc}', file=sys.stderr)
        return ConfigurationError.EXIT_CODE
    except Exception as exc:
        logger.exception('migration failed: %s', exc)
        return 2

    print_summary(stats, options.source_url, options.target_url, options.dry_run)
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
