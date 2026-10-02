"""PostgreSQL connection parameters shared by the pool and every one-off connection.

Identifier quoting, the asyncpg connect kwargs, the session ``SET`` statements every connection
runs, and the server-side statement timeout derived from the client command timeout.
"""

from typing import Any

import asyncpg

from app.settings import AppSettings
from app.settings import get_settings


def quote_pg_identifier(name: str) -> str:
    """Return ``name`` as a PostgreSQL quoted identifier, doubling embedded quotes.

    The single source of truth for quoting a configured PostgreSQL identifier
    (currently the schema name) so the connection ``search_path`` and any
    schema-qualified DDL (e.g. the migration CLI's ``CREATE SCHEMA``) escape it
    IDENTICALLY and cannot drift: a name containing a double quote yields a valid
    quoted identifier rather than a malformed / identifier-injecting string.

    Args:
        name: The raw identifier (e.g. ``POSTGRESQL_SCHEMA``).

    Returns:
        The identifier wrapped in double quotes with embedded quotes doubled.
    """
    return '"' + name.replace('"', '""') + '"'


def build_asyncpg_connect_kwargs(app_settings: AppSettings | None = None) -> dict[str, Any]:
    """Build the asyncpg connection kwargs shared by the pool and the migration CLI.

    Returns a dict suitable for spreading into ``asyncpg.connect(dsn, **kwargs)``
    or merging into ``asyncpg.create_pool(dsn, **kwargs)``. This is the single
    source of truth for the connection parameters that BOTH the long-lived server
    pool (``PostgreSQLBackend.initialize``) and the short-lived migration CLI
    (``app.cli.migrate_uuid.pg_connection``, ``app.cli.migrate_compression.decompress``) must
    apply identically:

    - ``statement_cache_size``: ``POSTGRESQL_STATEMENT_CACHE_SIZE`` (default 100;
      set 0 to disable prepared statements for transaction-mode poolers such as
      PgBouncer transaction mode, Pgpool-II, AWS RDS Proxy, or the Supabase
      Transaction Pooler).

    The STARTUP PACKET is deliberately left empty. asyncpg delivers
    ``server_settings`` as PostgreSQL startup-packet parameters, and an external
    pooler decides for itself what it accepts there: PgBouncer's always-allowed
    set is client_encoding, datestyle, timezone, standard_conforming_strings and
    application_name, and it REFUSES the connection with ``unsupported startup
    parameter: <name>`` for anything else unless the operator lists it in
    ``ignore_startup_parameters`` (which makes PgBouncer drop the value) or, on
    1.22+, ``track_extra_parameters`` (which makes it forward it). Sending
    anything there therefore turns the documented PgBouncer transaction-mode
    deployment into one that cannot connect at all with a stock configuration,
    and against a pooler that strips rather than refuses it turns a dropped
    ``search_path`` into every query resolving to the wrong schema. Every session
    parameter this server needs is applied with ``SET`` instead -- see
    :func:`session_guc_set_statements`, which every connection path runs.

    SSL is intentionally NOT included: asyncpg parses ``sslmode`` natively from
    the DSN query string, so SSL is carried by the connection URL itself.

    Args:
        app_settings: Resolved application settings. Defaults to
            ``get_settings()`` so callers in a different process (the CLI)
            pick up that process's environment.

    Returns:
        Mapping with the ``statement_cache_size`` key.
    """
    resolved = app_settings if app_settings is not None else get_settings()
    return {'statement_cache_size': resolved.storage.postgresql_statement_cache_size}


def session_guc_set_statements(app_settings: AppSettings | None = None) -> list[str]:
    """Build the ``SET`` statements every PostgreSQL connection needs.

    The session parameters this server depends on, delivered as ordinary
    statements rather than startup-packet parameters so an external pooler cannot
    refuse the connection over them (see :func:`build_asyncpg_connect_kwargs`).
    Every connection path runs them: the pool through ``setup_pool_connection``,
    which executes on every acquire and therefore also restores them after the
    pool's ``RESET ALL``, and each one-off connection through
    :func:`apply_session_gucs` immediately after dialing.

    The statements are:

    - ``search_path``, always ``"<POSTGRESQL_SCHEMA>", public`` (the schema
      double-quoted through :func:`quote_pg_identifier` so mixed-case and
      reserved identifiers are safe, and so the connection search_path and the
      migration CLI's schema-qualified DDL escape it identically). With the
      default ``POSTGRESQL_SCHEMA=public`` this is the benign no-op
      ``"public", public``.
    - ``extra_float_digits``, pinned to a shortest-round-trip setting so the
      numeric metadata-filter discriminator (``metadata_sql.pg_numeric_compare``)
      sees the Ryu shortest-repr float8 text it relies on: a cluster or role
      default of 0 or negative reverts float8out to ``%.15g`` and would
      misclassify every high-magnitude float-origin stored value, silently
      resurrecting the eq-against-its-own-value divergence the discriminator
      exists to close.
    - The server-side TCP keepalive GUCs, each omitted when its setting is 0 (the
      documented way to disable that probe parameter). They are the SECONDARY
      half of the keepalive story -- the PRIMARY half is the client-side
      ``setsockopt`` applied to the socket -- and through a pooler they describe
      the pooler-to-server hop rather than this connection.

    Args:
        app_settings: Resolved application settings. Defaults to
            ``get_settings()``.

    Returns:
        The ``SET`` statements, in the order they should be executed.
    """
    storage = (app_settings if app_settings is not None else get_settings()).storage
    statements = [
        f'SET search_path = {quote_pg_identifier(storage.postgresql_schema)}, public',
        'SET extra_float_digits = 1',
    ]
    keepalive_gucs = (
        ('tcp_keepalives_idle', storage.postgresql_tcp_keepalives_idle_s),
        ('tcp_keepalives_interval', storage.postgresql_tcp_keepalives_interval_s),
        ('tcp_keepalives_count', storage.postgresql_tcp_keepalives_count),
    )
    # The values are integers validated at the settings boundary (ge=0), so the
    # interpolation cannot carry anything but digits.
    statements.extend(f'SET {name} = {value}' for name, value in keepalive_gucs if value > 0)
    return statements


async def apply_session_gucs(conn: asyncpg.Connection, app_settings: AppSettings | None = None) -> None:
    """Apply the session parameters to a connection that bypasses the pool.

    The pool applies them through its ``setup`` callback on every acquire; a
    one-off connection (the boot-time provisioning probes and both migration CLIs)
    has no such callback and must run them itself, immediately after dialing and
    before its first statement. Without this the connection would silently use the
    server's default ``search_path`` -- resolving bare table names to ``public``
    instead of ``POSTGRESQL_SCHEMA`` -- and the server's default
    ``extra_float_digits``.

    All statements go out in ONE simple-query round trip.

    Args:
        conn: The freshly established connection.
        app_settings: Resolved application settings. Defaults to ``get_settings()``.
    """
    await conn.execute('; '.join(session_guc_set_statements(app_settings)))


def statement_timeout_ms(command_timeout_s: float) -> int:
    """Derive the server-side statement_timeout from the client command timeout.

    Uses 90 percent of the client-side command timeout so the server-side
    backstop fires just before asyncpg's own cancellation, and floors the
    result at 1 ms: PostgreSQL treats ``SET statement_timeout = 0`` as
    UNLIMITED, so truncating a sub-millisecond command timeout to 0 would
    silently disable the very backstop the setting exists to provide.

    Args:
        command_timeout_s: ``POSTGRESQL_COMMAND_TIMEOUT_S`` in seconds.

    Returns:
        Millisecond value for ``SET statement_timeout``, at least 1.
    """
    return max(1, int(command_timeout_s * 1000 * 0.9))
