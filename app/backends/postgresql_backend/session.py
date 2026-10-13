"""PostgreSQL connection parameters shared by the pool and every one-off connection.

The request connection string, identifier quoting, the asyncpg connect kwargs with an optional
credential override, the remediation a refused login names, the session ``SET`` statements every
connection runs, and the server-side statement timeout derived from the client command timeout.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import asyncpg
from pydantic import SecretStr

from app.settings import AppSettings
from app.settings import get_settings


@dataclass(frozen=True, slots=True)
class CredentialOverride:
    """The user and password a backend logs in with in place of those of its connection string.

    asyncpg gives explicit ``user=`` and ``password=`` connect kwargs precedence over the
    credentials of a URL or key=value DSN, so a backend can dial the request connection
    settings as another role without rewriting the DSN. An unset or empty field keeps the
    connection string's own value.

    Attributes:
        user: The role to log in as, or None to keep the connection string's user.
        password: That role's password, or None to keep the connection string's password.
    """

    user: str | None
    password: SecretStr | None

    def connect_kwargs(self) -> dict[str, str]:
        """Return the asyncpg connect kwargs for the fields that are set.

        Returns:
            ``user`` and ``password`` entries for the non-empty fields only.
        """
        kwargs: dict[str, str] = {}
        if self.user:
            kwargs['user'] = self.user
        if self.password is not None and self.password.get_secret_value():
            kwargs['password'] = self.password.get_secret_value()
        return kwargs


NO_CREDENTIAL_OVERRIDE = CredentialOverride(user=None, password=None)


def login_remediation(credentials: CredentialOverride, *, explicit_connection_string: bool) -> str:
    """Return the advice for a login the server refused (SQLSTATE class 28).

    The advice names where the refused user and password come from, as
    :func:`_login_variables` resolves them.

    Args:
        credentials: The backend's credential override.
        explicit_connection_string: Whether the backend dials a connection string it was
            handed rather than the request connection settings.

    Returns:
        The remediation sentence.
    """
    user, password = _login_variables(credentials, explicit_connection_string=explicit_connection_string)
    if user is None and password is None:
        return 'Check the user and password of the connection string and the server pg_hba.conf rules.'
    return (
        f'Check {user or "the user of the connection string"}, '
        f'{password or "the password of the connection string"} and the server pg_hba.conf rules.'
    )


def connect_privilege_remediation(credentials: CredentialOverride, *, explicit_connection_string: bool) -> str:
    """Return the advice for a login the server refused to connect to the database (SQLSTATE 42501).

    Args:
        credentials: The backend's credential override.
        explicit_connection_string: Whether the backend dials a connection string it was
            handed rather than the request connection settings.

    Returns:
        The remediation sentence naming the role that needs the grant and the database.
    """
    user, _ = _login_variables(credentials, explicit_connection_string=explicit_connection_string)
    if user is None:
        return 'Grant the user of the connection string access to its database.'
    database = 'the database of the connection string' if explicit_connection_string else 'POSTGRESQL_DATABASE'
    return f'Grant the configured {user} access to {database}.'


def _login_variables(credentials: CredentialOverride, *, explicit_connection_string: bool) -> tuple[str | None, str | None]:
    """Return the variables that supply the user and the password of a backend's login.

    A field the override sets wins over the connection string's own, as asyncpg applies
    it, and comes from the migration variables: the server's one override is the
    table-owner login of the maintenance backend. A field the override leaves unset
    comes from the connection string the backend was handed, or else from the request
    variables.

    Args:
        credentials: The backend's credential override.
        explicit_connection_string: Whether the backend dials a connection string it was
            handed rather than the request connection settings.

    Returns:
        The user variable and the password variable, None for a field the connection
        string the backend was handed supplies.
    """
    overridden = credentials.connect_kwargs()
    unset_user, unset_password = (None, None) if explicit_connection_string else ('POSTGRESQL_USER', 'POSTGRESQL_PASSWORD')
    return (
        'POSTGRESQL_MIGRATION_USER' if 'user' in overridden else unset_user,
        'POSTGRESQL_MIGRATION_PASSWORD' if 'password' in overridden else unset_password,
    )


def request_connection_string(app_settings: AppSettings) -> str:
    """Return the request connection settings as one connection string.

    ``POSTGRESQL_CONNECTION_STRING`` when it is set, otherwise a DSN built from the
    component fields. Supports self-hosted PostgreSQL and Supabase alike; for Supabase,
    use the connection string or the component fields with the Supabase host. The user,
    password and database name are URL-encoded, so characters such as ``#``, ``@``,
    ``:``, ``/`` and ``?`` cannot break the URL parsing in asyncpg.

    Args:
        app_settings: Resolved application settings.

    Returns:
        The connection string for asyncpg.
    """
    storage = app_settings.storage
    if storage.postgresql_connection_string:
        return storage.postgresql_connection_string.get_secret_value()

    # URL-encode every user-controlled component of the DSN, not only the password: a
    # colon in the user reads as the start of the password, an @ in the user or database
    # reads as the host boundary, and a / or ? in the database reads as the start of the
    # path or query string -- any of which corrupts the parsed DSN. safe='' encodes ALL
    # special characters (e.g. # becomes %23); asyncpg automatically URL-decodes the
    # connection string.
    encoded_user = quote(storage.postgresql_user, safe='')
    encoded_password = quote(storage.postgresql_password.get_secret_value(), safe='')
    encoded_database = quote(storage.postgresql_database, safe='')

    # An IPv6 host literal contains colons, which the DSN authority parses as the
    # host:port separator; bracket it (RFC 3986 IP-literal) so ::1 or a full IPv6
    # address interpolates as [::1]:port rather than corrupting the parse. Bracketing
    # is idempotent: a host already in bracket form (e.g. '[::1]', a natural
    # copy-paste from URI-style examples) is left as-is, because wrapping it again
    # would produce '[[::1]]', which asyncpg rejects at DSN parse with a plain
    # ValueError that never names the bracket cause. A hostname or IPv4 literal has
    # no colon and is left as-is.
    host = storage.postgresql_host
    if host.startswith('[') and host.endswith(']'):
        host_part = host
    elif ':' in host:
        host_part = f'[{host}]'
    else:
        host_part = host

    conn_str = f'postgresql://{encoded_user}:{encoded_password}@{host_part}:{storage.postgresql_port}/{encoded_database}'
    if storage.postgresql_ssl_mode != 'prefer':
        conn_str += f'?sslmode={storage.postgresql_ssl_mode}'
    return conn_str


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


def build_asyncpg_connect_kwargs(
    app_settings: AppSettings | None = None,
    *,
    credentials: CredentialOverride,
) -> dict[str, Any]:
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

    - ``user`` and ``password``: the fields of ``credentials`` that are set,
      which asyncpg applies over the credentials of the DSN.

    SSL is intentionally NOT included: asyncpg parses ``sslmode`` natively from
    the DSN query string, so SSL is carried by the connection URL itself.

    Args:
        app_settings: Resolved application settings. Defaults to
            ``get_settings()`` so callers in a different process (the CLI)
            pick up that process's environment.
        credentials: The login that replaces the DSN's own, field by field;
            :data:`NO_CREDENTIAL_OVERRIDE` dials the DSN as written.

    Returns:
        Mapping with the ``statement_cache_size`` key, plus ``user`` and
        ``password`` for the overridden fields.
    """
    resolved = app_settings if app_settings is not None else get_settings()
    return {
        'statement_cache_size': resolved.storage.postgresql_statement_cache_size,
        **credentials.connect_kwargs(),
    }


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


async def apply_session_gucs(
    conn: asyncpg.Connection,
    app_settings: AppSettings | None = None,
    *,
    profile_statements: Sequence[str],
) -> None:
    """Apply the session parameters to a connection that bypasses the pool.

    The pool applies them through its ``setup`` callback on every acquire; a
    one-off connection (the boot-time provisioning probes and both migration CLIs)
    has no such callback and must run them itself, immediately after dialing and
    before its first statement. Without this the connection would silently use the
    server's default ``search_path`` -- resolving bare table names to ``public``
    instead of ``POSTGRESQL_SCHEMA`` -- and the server's default
    ``extra_float_digits``.

    All statements go out in ONE simple-query round trip, the session profile's
    statements after the server's own.

    Args:
        conn: The freshly established connection.
        app_settings: Resolved application settings. Defaults to ``get_settings()``.
        profile_statements: The session statements of the profile the connection
            serves (``SessionProfile.session_statements``).
    """
    await conn.execute('; '.join([*session_guc_set_statements(app_settings), *profile_statements]))


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
