"""asyncpg pool callbacks of the PostgreSQL backend.

The typed ``connect`` dial, the per-connection ``init`` (TCP keepalive, pgvector and uuid codecs),
the per-acquire ``setup`` (statement timeout and session parameters), and the ``reset`` that
validates a connection before it returns to the pool.
"""

import asyncio
import logging
import socket
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
from typing import cast

import asyncpg

from app.backends.postgresql_backend.acquire_faults import ConnectionEstablishmentTimeoutError
from app.backends.postgresql_backend.acquire_faults import record_preparation_interrupted
from app.backends.postgresql_backend.session import session_guc_set_statements
from app.backends.postgresql_backend.session import statement_timeout_ms
from app.errors import ConfigurationError
from app.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


async def connect_pool_connection(*args: Any, **kwargs: Any) -> asyncpg.Connection:
    """Establish a new pooled connection, recording establishment faults.

    Passed as ``asyncpg.create_pool(connect=...)`` so every NEW connection the
    pool dials goes through this wrapper. Delegates to ``asyncpg.connect``
    verbatim (the pool forwards the DSN plus its loop/connection_class/
    record_class and the shared connect kwargs, including the ``timeout``
    establishment budget) and re-raises only the establishment TimeoutError as
    ConnectionEstablishmentTimeoutError -- the same typing pattern
    ``setup_pool_connection`` applies to setup timeouts -- so the acquire-phase
    handlers can charge an unreachable database deterministically. Every other
    failure (refused port, DNS resolution, connection reset) propagates
    unchanged.

    A dial CANCELLED by the enclosing acquire deadline cannot be typed the same
    way: swallowing the CancelledError would corrupt the cancellation
    bookkeeping ``asyncio.timeout`` relies on to convert it into the acquire's
    TimeoutError. It is therefore recorded on the acquire's ``_AcquireTracker``
    and re-raised unchanged, which is what lets the bare-TimeoutError arms tell
    an unreachable database from a saturated pool.

    Args:
        *args: Positional arguments the pool forwards (the DSN).
        **kwargs: Keyword arguments the pool forwards to ``asyncpg.connect``.

    Returns:
        The established asyncpg connection.

    Raises:
        ConnectionEstablishmentTimeoutError: When establishing the connection
            exceeded the connect timeout.
        asyncio.CancelledError: Re-raised unchanged when the enclosing acquire
            deadline cancelled the dial, after recording the interruption.
    """
    try:
        conn: asyncpg.Connection = await asyncpg.connect(*args, **kwargs)
    except TimeoutError as e:
        raise ConnectionEstablishmentTimeoutError(
            f'timed out establishing a new database connection: {e}',
        ) from e
    except asyncio.CancelledError:
        record_preparation_interrupted()
        raise
    return conn


async def setup_pool_connection(conn: asyncpg.Connection) -> None:
    """Configure session state before a pooled connection is handed to a caller.

    Sets statement_timeout to prevent queries from hanging indefinitely, plus every
    session parameter this server depends on (see ``session_guc_set_statements``:
    search_path, extra_float_digits and the server-side TCP keepalive GUCs). Runs as
    the pool's ``setup`` callback: AFTER ``init`` but BEFORE the connection is
    returned from ``pool.acquire()``, so it also restores all of them after the
    pool's ``RESET ALL`` on release -- which is what makes delivering them as
    statements, rather than as startup-packet parameters an external pooler can
    refuse, safe. All of them go out in ONE simple-query round trip, the same single
    round trip the statement timeout alone would cost.

    A TimeoutError here means the SET statement ran on a dead connection (e.g. a
    pooled connection whose backend became unreachable) and exceeded the pool
    command_timeout. asyncpg re-raises setup failures from ``pool.acquire()``
    verbatim, where a bare TimeoutError is indistinguishable from the
    saturation TimeoutError of a full pool -- which the backend deliberately
    leaves uncharged on the circuit breaker. Re-raise it as
    ConnectionDoesNotExistError so the fault stays distinguishable: the write
    path's charged retryable arm handles it, the acquire-phase generic arms of
    get_connection and begin_transaction charge it, and the uncharged
    saturation arm cannot swallow it.

    When the ACQUIRE budget wins the race instead, this callback is CANCELLED
    rather than timed out (asyncpg wraps the queue wait, the dial, ``init`` and
    ``setup`` in ONE ``wait_for``), so no typed exception is constructed and the
    acquire surfaces a BARE TimeoutError. That is the phase which dominates the
    most common outage onset -- a warm pool blackholed by a firewall DROP reuses
    an existing connection, skipping the dial entirely and hanging here -- so the
    interruption is recorded on the acquire's ``_AcquireTracker`` before the
    CancelledError is re-raised unchanged.

    Raises:
        asyncpg.exceptions.ConnectionDoesNotExistError: When the setup statement
            timed out (the connection is unusable; asyncpg closes it).
        asyncio.CancelledError: Re-raised unchanged when the enclosing acquire
            deadline cancelled the setup, after recording the interruption.
    """
    try:
        # Set statement timeout to prevent infinite hangs.
        # Use slightly less than command_timeout for graceful handling.
        timeout_ms = statement_timeout_ms(settings.storage.postgresql_command_timeout_s)
        session_statements = [f'SET statement_timeout = {timeout_ms}', *session_guc_set_statements(settings)]
        await conn.execute('; '.join(session_statements))
        logger.debug(f'Connection setup: statement_timeout={timeout_ms}ms, {len(session_statements) - 1} session GUCs')
    except TimeoutError as e:
        logger.warning(f'Connection setup timed out, connection is unusable: {e}')
        raise asyncpg.exceptions.ConnectionDoesNotExistError(
            f'connection setup timed out; the pooled connection is unusable: {e}',
        ) from e
    except asyncio.CancelledError:
        record_preparation_interrupted()
        raise
    except Exception as e:
        logger.warning(f'Connection setup failed: {e}')
        raise  # asyncpg will close connection and create new one


async def init_pool_connection(conn: asyncpg.Connection, *, provision_vector: bool) -> None:
    """Initialize each connection with TCP keepalive and pgvector type registration.

    Configures TCP keepalive on the client socket to prevent network intermediaries
    (NAT, firewalls, proxies, Supavisor) from closing idle connections.

    Also auto-detects the schema where pgvector extension is installed and registers
    the vector type codec for semantic search support.

    Args:
        conn: The newly established pooled connection.
        provision_vector: Whether to register the pgvector codec: the backend's
            resolved ``_provision_vector`` decision, bound by ``initialize()``.

    Raises:
        ConfigurationError: If the pgvector extension is not installed, or
            codec registration fails for a permanent reason (exit 78 at
            boot; the supervisor never retries it).
        TimeoutError: Re-raised unchanged when the codec registration
            round-trips time out, so the caller's retry arms still see a
            transient fault rather than a permanent misconfiguration.
        asyncpg.exceptions.PostgresConnectionError: Re-raised unchanged for
            the whole connection-failure family (backend gone, connection
            rejected, protocol violation) during a failover or pooler recycle.
        asyncpg.exceptions.OperatorInterventionError: Re-raised unchanged for
            an administrative or crash restart, a server still in recovery,
            or a cancelled query.
        asyncpg.exceptions.InterfaceError: Re-raised unchanged for the
            remaining transient driver-transport faults.
        OSError: Re-raised unchanged for transient socket faults.
    """
    # === TCP Keepalive Configuration ===
    # Set keepalive on the client socket via setsockopt. This is the
    # PRIMARY mechanism, and the ONLY one that reaches the app's own
    # socket when an external pooler terminates the connection: the
    # server-side GUCs (setup_pool_connection) then describe the
    # pooler-to-server hop rather than this one.
    tcp_idle = settings.storage.postgresql_tcp_keepalives_idle_s
    tcp_interval = settings.storage.postgresql_tcp_keepalives_interval_s
    tcp_count = settings.storage.postgresql_tcp_keepalives_count

    if tcp_idle > 0 or tcp_interval > 0 or tcp_count > 0:
        try:
            transport = getattr(conn, '_transport', None)
            raw_sock = transport.get_extra_info('socket') if transport is not None else None
            if raw_sock is not None:
                raw_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                if tcp_idle > 0 and hasattr(socket, 'TCP_KEEPIDLE'):
                    raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, tcp_idle)
                if tcp_interval > 0 and hasattr(socket, 'TCP_KEEPINTVL'):
                    raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, tcp_interval)
                if tcp_count > 0 and hasattr(socket, 'TCP_KEEPCNT'):
                    raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, tcp_count)
                logger.debug(
                    'TCP keepalive configured: idle=%ds, interval=%ds, count=%d',
                    tcp_idle, tcp_interval, tcp_count,
                )
            else:
                logger.warning('Could not access socket for TCP keepalive configuration')
        except Exception as e:
            # TCP keepalive failure is non-fatal — log and continue
            logger.warning('Failed to configure TCP keepalive: %s', e)

    # === pgvector Type Registration ===
    # Register the vector codec whenever the fp32 vector layout will be
    # provisioned (provision_vector: a compression-off generation-on
    # server, or any database that already carries the fp32 vec table). A
    # compressed server and a fresh generation-off database have no fp32 vec
    # table, so the codec is never needed and a missing pgvector extension must
    # NOT fail connection setup -- mirroring the same gate on
    # _ensure_pgvector_extension in initialize().
    # (The SQLite _load_sqlite_vec_extension load is NOT an analogue: it is
    # unconditional -- see its docstring -- because the vec0 module must stay
    # reachable for durable stale-embedding cleanup when generation is off.)
    if provision_vector:
        try:
            from pgvector.asyncpg import register_vector

            # AUTO-DETECT: Query where pgvector extension is installed
            # Works for ALL PostgreSQL variants (local, Supabase, AWS RDS, etc.)
            result = await conn.fetchrow('''
                SELECT n.nspname
                FROM pg_extension e
                JOIN pg_namespace n ON e.extnamespace = n.oid
                WHERE e.extname = 'vector'
            ''')

            if not result:
                # Extension not installed - fail fast with clear instructions
                raise ConfigurationError(
                    'pgvector extension is not installed. '
                    'Enable it via: CREATE EXTENSION vector; (PostgreSQL) '
                    'or Dashboard → Extensions → vector (Supabase)',
                )

            schema = result['nspname']

            # Register vector types using detected schema
            # Note: Type stubs for register_vector are incomplete (missing schema parameter)
            # Actual function signature: async def register_vector(conn, schema='public')
            # Using cast to work around incomplete type stubs
            register_func = cast(Callable[..., Awaitable[None]], register_vector)
            await register_func(conn, schema)
            logger.debug(f'Registered pgvector types from schema: {schema}')

        except ImportError:
            # ImportError is OK - semantic search is optional
            logger.debug('pgvector not installed, skipping vector type registration')

        except ConfigurationError:
            # Re-raise ConfigurationError as-is (from "extension not installed" check above)
            raise

        except (
            asyncpg.exceptions.ClientConfigurationError,
            asyncpg.exceptions.UnsupportedClientFeatureError,
            asyncpg.exceptions.UnsupportedServerFeatureError,
            asyncpg.exceptions.DataError,
        ) as e:
            # The PERMANENT InterfaceError subclasses, peeled off before the
            # transient tuple below, which they would otherwise match through
            # their shared InterfaceError base. A client misconfiguration, a
            # feature this driver or this server does not support, and a codec
            # that cannot decode what the server sends are all deterministic:
            # retrying reproduces them exactly, so they belong in the
            # ConfigurationError (exit 78) class like everywhere else in this
            # backend, and treating them as transient would spin the retry arms
            # and the supervisor against a fault that never clears.
            logger.error(f'PostgreSQL client configuration invalid: {e}')
            raise ConfigurationError(
                f'pgvector codec registration failed: {type(e).__name__}: {e}',
            ) from e

        except (
            TimeoutError,
            asyncpg.exceptions.PostgresConnectionError,
            asyncpg.exceptions.OperatorInterventionError,
            asyncpg.exceptions.InterfaceError,
            OSError,
        ) as e:
            # The two awaits above (the pg_extension probe and
            # register_vector's type introspection) are real server
            # round-trips, so this block sees the whole transient
            # transport family: a backend terminated by a failover or
            # a pooler recycle, a reset connection, a command timeout.
            # The families are named by their asyncpg BASE classes rather than
            # by individual members, because a member-by-member list silently
            # excludes the siblings nobody enumerated: PostgresConnectionError
            # covers the whole SQLSTATE class 08 (connection failure, rejection,
            # protocol violation) and not only ConnectionDoesNotExistError, and
            # OperatorInterventionError covers class 57 (admin shutdown, crash
            # restart, "cannot connect now" during recovery, a cancelled query)
            # -- exactly the faults a failover or a rolling restart produces,
            # and every one of them self-clearing.
            # asyncpg re-raises an init failure VERBATIM out of
            # pool.acquire(), so relabeling one of these as
            # ConfigurationError would make a self-clearing blip
            # permanent: at runtime execute_write's typed retry arms
            # would no longer match it (no retry, breaker charged),
            # and at boot initialize()'s ladder would exit 78 -- which
            # the supervisor never retries -- instead of the retryable
            # DependencyError exit 69. Re-raise unchanged and let the
            # existing classification handle it.
            logger.warning(
                f'pgvector codec registration hit a transient connection fault, '
                f'connection will be retried: {type(e).__name__}: {e}',
            )
            raise

        except Exception as e:
            # STRICT: All other errors are FATAL. The type name is part
            # of the message because several candidates here (a bare
            # TimeoutError above all) stringify to '', which would leave
            # the operator with 'pgvector codec registration failed: '
            # and no cause at all.
            logger.error(f'Failed to register pgvector type codec: {type(e).__name__}: {e}')
            logger.error('Ensure pgvector extension is enabled and accessible')
            raise ConfigurationError(
                f'pgvector codec registration failed: {type(e).__name__}: {e}',
            ) from e

    # === UUID Type Codec Registration ===
    # asyncpg's default codec maps the PostgreSQL ``uuid`` type to
    # ``asyncpg.pgproto.pgproto.UUID``, a fast subclass that
    # ``isinstance``-tests as ``uuid.UUID`` but is not a literal
    # ``str``. The repository layer exchanges identifiers as
    # canonical 32-char lowercase hex strings, so this codec
    # round-trips both directions through ``str``.
    #
    # Encoder: pass the string through verbatim. PostgreSQL's
    # native UUID input parser accepts both 32-char hex and
    # 36-char hyphenated forms (case-insensitive).
    # Decoder: normalize the 36-char hyphenated text returned by
    # PostgreSQL into the 32-char lowercase hex canonical form.
    from app.ids import normalize_id

    await conn.set_type_codec(
        'uuid',
        schema='pg_catalog',
        encoder=lambda v: v,
        decoder=normalize_id,
        format='text',
    )
    logger.debug('Registered uuid->str type codec')


async def reset_pool_connection(conn: asyncpg.Connection) -> None:
    """Validate and reset connection before returning to pool.

    Ensures clean state by:
    1. Rolling back any active transaction (safe no-op if none active)
    2. Validating connection health with lightweight query
    3. Resetting session GUC parameters

    If any step fails, connection is terminated and pool creates a new one.
    This catches corrupted connections before they cause protocol errors.

    Called BEFORE connection returns to pool. If it raises, connection
    is terminated (not returned to pool).
    """
    try:
        # Abort any active transaction FIRST
        # ROLLBACK is safe even if no transaction is active (no-op)
        # This handles cases where request cancellation left uncommitted work
        await conn.execute('ROLLBACK')
        # Lightweight validation - catches protocol state mismatches
        await conn.fetchval('SELECT 1')
        # Reset session state (GUC parameters only, NOT transactions)
        await conn.execute('RESET ALL')
        logger.debug('Connection reset successful before pool return')
    except Exception as e:
        logger.warning(f'Connection reset failed, connection will be terminated: {e}')
        raise  # asyncpg will terminate connection and create new one
