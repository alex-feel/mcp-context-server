"""Lifecycle of the PostgreSQL backend.

Pool creation with its startup error classification, the boot-time connectivity and pooler
probes, and shutdown.
"""

import logging
from functools import partial
from typing import Any
from urllib.parse import urlsplit

import asyncpg

from app.backends.postgresql_backend.acquire_faults import charge_cancelled_preparation
from app.backends.postgresql_backend.pool_callbacks import connect_pool_connection
from app.backends.postgresql_backend.pool_callbacks import init_pool_connection
from app.backends.postgresql_backend.pool_callbacks import reset_pool_connection
from app.backends.postgresql_backend.pool_callbacks import setup_pool_connection
from app.backends.postgresql_backend.provisioning import PostgreSQLProvisioningMixin
from app.backends.postgresql_backend.session import build_asyncpg_connect_kwargs
from app.errors import ConfigurationError
from app.errors import DependencyError
from app.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class PostgreSQLLifecycleMixin(PostgreSQLProvisioningMixin):
    """Create and verify the connection pool, probe for poolers, and shut down."""

    async def initialize(self) -> None:
        """Initialize the PostgreSQL backend with connection pool."""
        logger.info(f'Initializing PostgreSQL backend: {self.backend_type}')

        try:
            # Resolve whether the pgvector extension and vector codec are needed. The
            # vector type is used ONLY by the fp32 vec_context_embeddings layout, so a
            # compressed (BYTEA-only) database -- a compression-on server, or a
            # generation-off archive restored on a host without pgvector -- must NOT be
            # forced to create the unused extension, while a compression-off database that
            # will provision or already carries the fp32 layout still must. See
            # _resolve_provision_vector. A caller that knows the answer up front (the
            # migration CLI's target init, keyed on with_semantic) supplies the explicit
            # constructor override instead, so a vector-free target never pays the
            # settings-driven gate of a process whose env does not describe the target.
            if self._provision_vector_override is not None:
                self._provision_vector = self._provision_vector_override
            else:
                self._provision_vector = await self._resolve_provision_vector()

            # Pre-create pgvector extension when the vector layout will be
            # provisioned. This prevents "unknown type: public.vector" warnings
            # during pool initialization and lets the migration's vector DDL run.
            if self._provision_vector:
                await self._ensure_pgvector_extension()

            # Prepare pool configuration with hardening parameters
            pool_kwargs: dict[str, Any] = {
                'min_size': settings.storage.postgresql_pool_min,
                'max_size': settings.storage.postgresql_pool_max,
                'command_timeout': settings.storage.postgresql_command_timeout_s,
                # asyncpg.create_pool has NO acquire-timeout parameter: unknown
                # kwargs fall through **connect_kwargs to asyncpg.connect(), so
                # 'timeout' here is the per-connection ESTABLISHMENT timeout
                # (TCP connect + startup handshake). The acquire-wait bound
                # (POSTGRESQL_POOL_TIMEOUT_S) is passed per-call at every
                # pool.acquire(timeout=...) site instead.
                'timeout': settings.storage.postgresql_connect_timeout_s,
                # Dial NEW connections through the typed-connect wrapper so an
                # establishment timeout surfaces as
                # ConnectionEstablishmentTimeoutError instead of the bare
                # TimeoutError a saturated pool's acquire deadline raises; the
                # acquire-phase circuit-breaker arms rely on that distinction.
                'connect': connect_pool_connection,
                # statement_cache_size is merged below via
                # build_asyncpg_connect_kwargs() so the pool and the migration
                # CLI share one source of truth.
                'max_cached_statement_lifetime': settings.storage.postgresql_max_cached_statement_lifetime_s,
                'max_cacheable_statement_size': settings.storage.postgresql_max_cacheable_statement_size,
                # Connection lifecycle callbacks
                #
                # ``init`` awaits real server round-trips (the pg_extension probe,
                # register_vector, the uuid codec's type introspection) inside the SAME
                # acquire deadline as the dial, so it is wrapped to record a cancellation
                # on the acquire's dial tracker; without that, an acquire whose budget
                # expires during init surfaces a bare TimeoutError that the uncharged
                # saturation arm would swallow.
                'init': charge_cancelled_preparation(  # TCP keepalive + type registration
                    partial(init_pool_connection, provision_vector=self._provision_vector),
                ),
                'setup': setup_pool_connection,  # Session config before acquire
                'reset': reset_pool_connection,  # Health check before pool return
            }

            # Merge the shared connection kwargs. build_asyncpg_connect_kwargs() is the
            # single source of truth shared with the migration CLI, and it forwards
            # ONLY statement_cache_size (0 disables prepared statements for
            # transaction-mode poolers). The PostgreSQL startup packet is left empty on
            # purpose: an external pooler refuses any startup parameter it does not
            # allowlist, so anything sent there is a deployment that cannot connect at
            # all. search_path, extra_float_digits and the server-side TCP keepalive
            # GUCs are applied as statements by setup_pool_connection, which runs on
            # every acquire and therefore also restores them after the pool's RESET ALL
            # (see build_asyncpg_connect_kwargs and docs/database-backends.md).
            pool_kwargs.update(build_asyncpg_connect_kwargs(settings))

            # Add connection recycling settings if configured (0 means disabled)
            if settings.storage.postgresql_max_inactive_lifetime_s > 0:
                pool_kwargs['max_inactive_connection_lifetime'] = (
                    settings.storage.postgresql_max_inactive_lifetime_s
                )
            if settings.storage.postgresql_max_queries > 0:
                pool_kwargs['max_queries'] = settings.storage.postgresql_max_queries

            # Create connection pool with hardening configuration
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                **pool_kwargs,
            )

            # Prove the pool can actually reach the database BEFORE reporting
            # success. asyncpg's create_pool() contacts the server only to
            # pre-connect min_size connections, so with POSTGRESQL_POOL_MIN=0 (an
            # explicitly supported cold-pool choice) it succeeds against an
            # unreachable host, a wrong password or a nonexistent database, and
            # the classification ladder below -- which turns those into exit 78
            # instead of a supervisor restart loop -- never observes a real dial.
            await self._verify_connectivity()

            # Detect Pgpool-II and log result
            await self._detect_pgpool_ii()

            # Detect Supabase Session Pooler endpoint and warn on oversized pool
            self._detect_session_mode_pooler()

            logger.info('PostgreSQL backend initialized successfully')

        except asyncpg.exceptions.ClientConfigurationError as e:
            # Must precede the InterfaceError tuple below:
            # ClientConfigurationError subclasses BOTH InterfaceError and
            # ValueError, and asyncpg raises it for permanent client-side
            # misconfigurations (invalid sslmode/target_session_attrs/
            # gsslib values, unresolvable DSN options) that a broad
            # InterfaceError match would misclassify as a retryable
            # DependencyError and restart-loop on.
            logger.error(f'PostgreSQL client configuration invalid: {e}')
            await self._record_charged_failure(e)
            raise ConfigurationError(
                f'PostgreSQL client configuration invalid: {e}. '
                'Check POSTGRESQL_CONNECTION_STRING and the POSTGRESQL_* '
                'connection options.',
            ) from e

        except (
            OSError,  # Includes ConnectionRefusedError, TimeoutError
            asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.exceptions.InterfaceError,
            asyncpg.exceptions.TooManyConnectionsError,
        ) as e:
            logger.error(f'Failed to initialize PostgreSQL backend: {e}')
            await self._record_charged_failure(e)
            raise DependencyError(
                f'PostgreSQL connection failed: {e}. '
                'Ensure PostgreSQL is running and accessible.',
            ) from e

        except asyncpg.exceptions.InvalidAuthorizationSpecificationError as e:
            # SQLSTATE class 28 as a WHOLE, not only InvalidPasswordError (28P01, a
            # subclass of this): 28000 invalid_authorization_specification is what
            # PostgreSQL returns for 'no pg_hba.conf entry for host ...', an equally
            # permanent credential/authorization misconfiguration. Matching only the
            # password subclass would send it to the terminal handler below as a
            # retryable DependencyError (exit 69), and the supervisor would restart-loop
            # forever.
            logger.error(f'PostgreSQL authentication failed: {e}')
            await self._record_charged_failure(e)
            raise ConfigurationError(
                f'PostgreSQL authentication failed: {e}. '
                'Check POSTGRESQL_USER, POSTGRESQL_PASSWORD and the server pg_hba.conf rules.',
            ) from e

        except asyncpg.exceptions.InsufficientPrivilegeError as e:
            # SQLSTATE 42501 raised by the boot dial means the role may not CONNECT
            # to the database (a revoked grant): permanent, so exit 78 rather than a
            # restart loop. Mirrors the same arm in _ensure_pgvector_extension.
            logger.error(f'PostgreSQL permission denied: {e}')
            await self._record_charged_failure(e)
            raise ConfigurationError(
                f'PostgreSQL permission denied: {e}. '
                'Grant the configured POSTGRESQL_USER access to POSTGRESQL_DATABASE.',
            ) from e

        except asyncpg.exceptions.InvalidCatalogNameError as e:
            logger.error(f'PostgreSQL database does not exist: {e}')
            await self._record_charged_failure(e)
            raise ConfigurationError(
                f'PostgreSQL database does not exist: {e}. '
                'Create the database or check POSTGRESQL_DATABASE.',
            ) from e

        except ConfigurationError:
            raise  # Re-raise already-classified errors (from init_pool_connection)

        except DependencyError:
            raise  # Re-raise already-classified errors (from _ensure_pgvector_extension)

        except ValueError as e:
            # asyncpg raises plain ValueError for invalid construction inputs
            # (pool size combinations, non-positive command_timeout) before
            # any network I/O. DSN option errors instead surface as
            # ClientConfigurationError and are classified above -- its
            # InterfaceError base would shadow this clause. These are
            # permanent misconfigurations: exit 78 so the supervisor
            # does not restart-loop on them.
            logger.error(f'PostgreSQL configuration invalid: {e}')
            await self._record_charged_failure(e)
            raise ConfigurationError(
                f'PostgreSQL configuration invalid: {e}. '
                'Check POSTGRESQL_POOL_* values and the connection string.',
            ) from e

        except Exception as e:
            logger.error(f'Failed to initialize PostgreSQL backend: {e}')
            await self._record_charged_failure(e)
            # Default to DependencyError for unknown errors (safer - allows retry)
            raise DependencyError(
                f'PostgreSQL initialization failed: {e}. '
                'Ensure PostgreSQL is running and accessible.',
            ) from e

    async def _verify_connectivity(self) -> None:
        """Dial the database once through the pool and let faults classify.

        The unconditional boot-time reachability check. It exists because
        ``asyncpg.create_pool()`` is not one: ``Pool._initialize`` guards every
        pre-connect behind ``if self._minsize:``, so with POSTGRESQL_POOL_MIN=0
        the pool is created without ever contacting the server. Boot validation
        must not depend on a diagnostic probe either -- a probe swallows failures
        by design -- so this acquire deliberately lets EVERY failure propagate to
        ``initialize()``'s classification ladder, which turns a wrong password,
        a nonexistent database or an invalid client configuration into
        ConfigurationError (exit 78, no supervisor restart loop) and an
        unreachable host into DependencyError (exit 69, retryable). Without it a
        permanent credential misconfiguration would surface much later as a raw,
        unclassified schema-statement error.
        """
        assert self._pool is not None, 'Pool not initialized'

        async with self._pool.acquire(
            timeout=settings.storage.postgresql_pool_timeout_s,
        ) as conn:
            await conn.fetchval('SELECT 1')
        logger.debug('PostgreSQL connectivity verified')

    async def _detect_pgpool_ii(self) -> None:
        """Detect if connected through Pgpool-II and log result.

        Uses SHOW POOL_VERSION command which is Pgpool-II specific.
        On direct PostgreSQL connections, this command raises UndefinedObjectError
        (error code 42704: unrecognized configuration parameter).

        Only the detection QUERY is diagnostic: its failure is swallowed. The
        ACQUIRE sits outside that handling on purpose, so an establishment,
        authentication or connection-init fault reaches ``initialize()``'s
        classification ladder instead of being logged at WARNING (invisible at
        the default LOG_LEVEL=ERROR, and with an EMPTY message for a bare
        TimeoutError, whose ``str()`` is '') and followed by
        'initialized successfully'.
        """
        assert self._pool is not None, 'Pool not initialized'

        async with self._pool.acquire(
            timeout=settings.storage.postgresql_pool_timeout_s,
        ) as conn:
            try:
                version = await conn.fetchval('SHOW POOL_VERSION')
            except asyncpg.exceptions.UndefinedObjectError:
                # Expected when not behind Pgpool-II - pool_version is not a known parameter
                logger.info('Looks like direct PostgreSQL connection (at least, no Pgpool-II)')
                self._pgpool_version = None
                return
            except Exception as e:
                # Log but do not fail initialization. Name the exception type too:
                # several exceptions relevant here (TimeoutError above all) have an
                # empty str(), which would otherwise log a warning with no cause.
                logger.warning(f'Pgpool-II detection check failed ({type(e).__name__}): {e}')
                self._pgpool_version = None
                return

        if version:
            logger.warning(
                f'Pgpool-II detected: version {version}. '
                f'Recommended to set POSTGRESQL_STATEMENT_CACHE_SIZE=0.',
            )
            self._pgpool_version = str(version)
        else:
            logger.info('Looks like direct PostgreSQL connection (at least, no Pgpool-II)')
            self._pgpool_version = None

    def _detect_session_mode_pooler(self) -> None:
        """Detect a Supabase Session Pooler endpoint and warn if pool too large.

        Inspects the connection string host/port (the authoritative endpoint the
        pool actually dials, including the POSTGRESQL_CONNECTION_STRING form).
        When a Supabase Session Pooler (host contains ``pooler.supabase.com`` on
        port 5432) is detected AND POSTGRESQL_POOL_MAX exceeds the conservative
        default per-session client cap, logs a targeted WARNING naming the
        symptom (MaxClientsInSessionMode) and the fix.

        Defense-in-depth only: does NOT modify pool size. Non-fatal -- any
        parsing failure is logged and treated as "not a session pooler".

        Mirrors _detect_pgpool_ii() in level, message shape, and resilience.
        """
        from app.startup.validation import is_supabase_session_pooler

        try:
            parsed = urlsplit(self.connection_string)
            host = (parsed.hostname or '').lower()
            port = parsed.port if parsed.port is not None else 5432
            if not host and '://' not in self.connection_string:
                # libpq key-value DSN form ("host=... port=..."), which asyncpg
                # also accepts: urlsplit yields no hostname, so parse the
                # whitespace-separated key=value tokens directly so the advisory
                # fires for this spelling too.
                kv = dict(
                    token.split('=', 1)
                    for token in self.connection_string.split()
                    if '=' in token
                )
                host = kv.get('host', '').lower()
                port = int(kv['port']) if kv.get('port') else 5432
        except Exception as e:
            # Malformed connection string is non-fatal for this advisory;
            # the pool creation above already succeeded with this string.
            logger.warning(f'Session-mode pooler detection check failed: {e}')
            self._session_mode_pooler = False
            return

        if not is_supabase_session_pooler(host, port):
            self._session_mode_pooler = False
            return

        self._session_mode_pooler = True

        pool_max = settings.storage.postgresql_pool_max
        cap = settings.storage.postgresql_session_pooler_max_clients
        if pool_max > cap:
            logger.warning(
                f'Supabase Session Pooler detected ({host}:{port}) with '
                f'POSTGRESQL_POOL_MAX={pool_max}, which exceeds '
                f'POSTGRESQL_SESSION_POOLER_MAX_CLIENTS={cap} (the session-mode '
                f'per-session client cap; default 15 on Supabase Free/Pro tiers). '
                f'This can intermittently fail with "MaxClientsInSessionMode: '
                f'max clients reached - in Session mode max clients are limited '
                f'to pool_size". Lower POSTGRESQL_POOL_MAX to fit your pooler '
                f'capacity (or raise POSTGRESQL_SESSION_POOLER_MAX_CLIENTS if your '
                f'tier allows more), or use the Transaction-mode pooler (port 6543) '
                f'or a Direct Connection. See docs/database-backends.md '
                f'"Session Pooler Connection Limits".',
            )

    async def shutdown(self) -> None:
        """Gracefully shut down the PostgreSQL backend."""
        logger.info('Shutting down PostgreSQL backend')

        self._shutdown = True

        try:
            # Close connection pool
            if self._pool:
                await self._pool.close()
                self._pool = None

            logger.info('PostgreSQL backend shutdown complete')

        except Exception as e:
            logger.error(f'Error during PostgreSQL backend shutdown: {e}')
            raise
