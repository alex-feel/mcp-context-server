"""SQLite target database: opening, schema initialization, emptiness and layout
probes, and the FTS5 rebuild after the data copy.
"""

import contextlib
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from app.cli.migrate_uuid.records import MigrationStats


def _open_target_file(path: str) -> sqlite3.Connection:
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


def open_target_sqlite(path: str, dry_run: bool) -> sqlite3.Connection:
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
    return _open_target_file(path)


def load_sqlite_vec_extension(conn: sqlite3.Connection) -> bool:
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


def read_schema_file(filename: str) -> str:
    """Read a packaged schema or migration SQL file by name.

    Returns:
        Contents of the matching file.

    Raises:
        FileNotFoundError: When ``filename`` cannot be located in the
            standard ``schemas`` or ``migrations`` directories.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / 'schemas' / filename,
        Path(__file__).resolve().parents[2] / 'migrations' / filename,
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
    base_schema = read_schema_file('sqlite_schema.sql')
    target.executescript(base_schema)

    vec_loaded = False
    if optional_tables.get('vec_context_embeddings') or optional_tables.get('embedding_metadata'):
        vec_loaded = load_sqlite_vec_extension(target)
        if optional_tables.get('vec_context_embeddings') and not vec_loaded:
            stats.warnings.append(
                'sqlite-vec extension could not be loaded on target; '
                'vec_context_embeddings will not be copied',
            )

    if optional_tables.get('embedding_metadata') and vec_loaded:
        semantic_sql = read_schema_file('add_semantic_search_sqlite.sql')
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
        chunking_sql = read_schema_file('add_chunking_sqlite.sql')
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
        fts_sql = read_schema_file('add_fts_sqlite.sql')
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
