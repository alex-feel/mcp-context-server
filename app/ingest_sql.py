"""SQL fragments that read the reserved ``ingest`` metadata block of a context entry.

A context entry managed by a corpus sync carries ``metadata.ingest = {scope, source_id,
meta_digest, synced_at}``. The repository reads compare, order and select the block through
these fragments, and the ingest index is rendered from the same fragments, so a query
expression always matches the index expression it relies on: SQLite matches expression
indexes structurally (column qualifiers resolved), PostgreSQL by expression and collation.

The fragments are total: they read stored JSON through text accessors only (``json_extract``
on SQLite; ``#>>`` and ``->`` on PostgreSQL) and never cast stored data, so they cannot raise
on any stored value. Source ids compare by code point: SQLite's default ``BINARY`` collation
compares UTF-8 bytes, and PostgreSQL bounds and orders the key under the ``C`` collation.

This module imports nothing, so the migrations and the repositories can share it.
"""

INGEST_INDEX_NAME = 'idx_context_ingest'


def _qualifier(alias: str | None) -> str:
    """Return the prefix qualifying the ``metadata`` column with ``alias``, empty when there is none."""
    return f'{alias}.' if alias else ''


def source_id_sql(backend_type: str, alias: str | None = None) -> str:
    """Render the block's source id as text, NULL when the entry carries no source id.

    Args:
        backend_type: ``'sqlite'`` or ``'postgresql'``.
        alias: The name qualifying ``context_entries``' columns in the statement, if any.

    Returns:
        The SQL expression.
    """
    q = _qualifier(alias)
    if backend_type == 'sqlite':
        return f"json_extract({q}metadata, '$.ingest.source_id')"
    return f"({q}metadata #>> '{{ingest,source_id}}')"


def source_id_key_sql(backend_type: str, alias: str | None = None) -> str:
    """Render the source id as the key that range bounds and ordering compare in code-point order.

    Args:
        backend_type: ``'sqlite'`` or ``'postgresql'``.
        alias: The name qualifying ``context_entries``' columns in the statement, if any.

    Returns:
        The SQL expression: the source id itself on SQLite, the source id under the ``C``
        collation on PostgreSQL.
    """
    if backend_type == 'sqlite':
        return source_id_sql(backend_type, alias)
    return f'({source_id_sql(backend_type, alias)} COLLATE "C")'


def scope_sql(backend_type: str, alias: str | None = None) -> str:
    """Render the block's scope as text, the thread id an entry must sit in to be a member of its corpus.

    Args:
        backend_type: ``'sqlite'`` or ``'postgresql'``.
        alias: The name qualifying ``context_entries``' columns in the statement, if any.

    Returns:
        The SQL expression.
    """
    q = _qualifier(alias)
    if backend_type == 'sqlite':
        return f"json_extract({q}metadata, '$.ingest.scope')"
    return f"({q}metadata #>> '{{ingest,scope}}')"


def block_sql(backend_type: str, alias: str | None = None) -> str:
    """Render the whole block, which both backends return to the application as JSON text.

    Args:
        backend_type: ``'sqlite'`` or ``'postgresql'``.
        alias: The name qualifying ``context_entries``' columns in the statement, if any.

    Returns:
        The SQL expression: ``json_extract`` of the object on SQLite, the jsonb value on PostgreSQL.
    """
    q = _qualifier(alias)
    if backend_type == 'sqlite':
        return f"json_extract({q}metadata, '$.ingest')"
    return f"{q}metadata -> 'ingest'"


def ingest_index_ddl(backend_type: str) -> str:
    """Render the partial index serving the ingest reads.

    The index holds only entries whose block carries a source id. On SQLite it covers every
    column the reads use; PostgreSQL keys it on the thread, the ``C``-collated source id and
    the id. The statement is idempotent (``IF NOT EXISTS``).

    Args:
        backend_type: ``'sqlite'`` or ``'postgresql'``.

    Returns:
        The ``CREATE INDEX`` statement.
    """
    source_id = source_id_sql(backend_type)
    if backend_type == 'sqlite':
        columns = (
            f'thread_id, {source_id}, id, {scope_sql(backend_type)}, {block_sql(backend_type)}, '
            'content_hash, owner_id, visibility'
        )
        return f'CREATE INDEX IF NOT EXISTS {INGEST_INDEX_NAME} ON context_entries({columns}) WHERE {source_id} IS NOT NULL'
    columns = f'thread_id, {source_id_key_sql(backend_type)}, id'
    return f'CREATE INDEX IF NOT EXISTS {INGEST_INDEX_NAME} ON context_entries ({columns}) WHERE {source_id} IS NOT NULL'
