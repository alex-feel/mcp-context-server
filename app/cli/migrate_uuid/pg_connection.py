"""PostgreSQL connections and catalog probes used by the migration runners."""

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import asyncpg


async def target_pg_has_data(
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
    if not await pg_table_exists(conn, 'context_entries', schema=schema):
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
        from app.backends.postgresql_backend.session import quote_pg_identifier

        count = await conn.fetchval(f'SELECT COUNT(*) FROM {quote_pg_identifier(schema)}.context_entries')
    return int(count or 0) > 0


def pg_connect_kwargs() -> dict[str, Any]:
    """Return the shared asyncpg connect kwargs for the migration CLI.

    Imported lazily so the SQLite-only migration paths never import the
    PostgreSQL backend (and therefore never require asyncpg/pgvector to be
    installed). The kwargs apply ``statement_cache_size`` (set
    ``POSTGRESQL_STATEMENT_CACHE_SIZE=0`` for transaction-mode poolers such as the
    Supabase Transaction Pooler). SSL is carried by the DSN (``?sslmode=...``),
    parsed natively by asyncpg. The SESSION parameters (``search_path``,
    ``extra_float_digits``, the TCP keepalive GUCs) are NOT startup-packet
    parameters -- a pooler would refuse the connection over them -- and are applied
    by :func:`pg_connect` right after the dial instead.

    ``timeout`` (``POSTGRESQL_CONNECT_TIMEOUT_S``) is added here rather than by
    :func:`app.backends.postgresql_backend.session.build_asyncpg_connect_kwargs`, whose scope
    is what the pool merges into ``create_pool``; the pool supplies the same
    establishment budget separately. Without it every migration connection would
    silently fall back to asyncpg's built-in 60-second default, so a DSN whose TLS and
    startup handshake needs the longer budget the operator configured would boot the
    server fine yet abort the migration -- and a deliberately SHORT budget would not
    fail fast either.

    Returns:
        Mapping suitable for spreading into ``asyncpg.connect(dsn, **kwargs)``.
    """
    from app.backends.postgresql_backend.session import build_asyncpg_connect_kwargs
    from app.settings import get_settings

    settings = get_settings()
    kwargs = build_asyncpg_connect_kwargs(settings)
    kwargs['timeout'] = settings.storage.postgresql_connect_timeout_s
    return kwargs


async def pg_connect(dsn: str) -> 'asyncpg.Connection[asyncpg.Record]':
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

    from app.backends.postgresql_backend.session import apply_session_gucs

    conn: asyncpg.Connection[asyncpg.Record] = await asyncpg.connect(dsn, **pg_connect_kwargs())
    try:
        await apply_session_gucs(conn)
    except BaseException:
        await conn.close()
        raise
    return conn


async def pg_table_exists(
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
    making the probe inspect the wrong schema (see :func:`target_pg_has_data`).

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


async def pg_column_exists(
    conn: 'asyncpg.Connection[asyncpg.Record]',
    table_name: str,
    column_name: str,
    schema: str | None = None,
) -> bool:
    """Return True if ``column_name`` exists on ``table_name`` in the resolved schema.

    Mirrors :func:`pg_table_exists`'s schema-resolution contract: ``None`` uses
    ``current_schema()`` (correct for a SOURCE connection whose configured
    ``POSTGRESQL_SCHEMA`` already exists); an explicit ``schema`` binds that name
    directly. Used to guard the source SELECT against a v2 PostgreSQL source that
    predates the ``summary`` / ``content_hash`` ALTER-TABLE columns, mirroring the
    SQLite :func:`table_has_column` guard so every migration direction tolerates
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


async def detect_source_embedding_dim_pg(conn: 'asyncpg.Connection[asyncpg.Record]') -> int | None:
    """Best-effort detection of the embedding dimension from a PostgreSQL source.

    Mirrors :func:`detect_source_embedding_dim` (the SQLite detector). Guards the
    read behind table existence so a source that never enabled semantic search
    (no ``embedding_metadata`` table) yields ``None`` instead of raising.

    Returns:
        The dimension from the first ``embedding_metadata`` row, or ``None`` when
        the table is absent or empty.
    """
    if not await pg_table_exists(conn, 'embedding_metadata'):
        return None
    row = await conn.fetchval('SELECT dimensions FROM embedding_metadata LIMIT 1')
    return int(row) if row is not None else None
