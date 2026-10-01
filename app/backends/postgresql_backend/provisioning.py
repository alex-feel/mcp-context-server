"""Boot-time pgvector provisioning of the PostgreSQL backend.

Decides whether the pgvector extension and vector codec are needed, and creates the extension
on a one-off connection before the pool exists.
"""

import logging

import asyncpg

from app.backends.postgresql_backend.core import PostgreSQLBackendCore
from app.backends.postgresql_backend.session import apply_session_gucs
from app.backends.postgresql_backend.session import build_asyncpg_connect_kwargs
from app.errors import ConfigurationError
from app.errors import DependencyError
from app.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class PostgreSQLProvisioningMixin(PostgreSQLBackendCore):
    """Decide and perform the pgvector provisioning the boot needs."""

    async def _ensure_pgvector_extension(self) -> None:
        """Ensure pgvector extension exists before pool creation.

        Creates a temporary connection to enable the pgvector extension,
        then immediately closes it. This allows pool connections to
        successfully register pgvector types during initialization.

        STRICT MODE: Fails fast if extension cannot be created.
        On Supabase: Enable via Dashboard → Extensions → vector (recommended).

        Raises:
            ConfigurationError: For permission/auth errors requiring human intervention
            DependencyError: For connection errors that may resolve with retry
        """
        try:
            conn = await asyncpg.connect(
                self.connection_string,
                timeout=settings.storage.postgresql_connect_timeout_s,
                **build_asyncpg_connect_kwargs(settings),
            )
            try:
                await apply_session_gucs(conn, settings)
                await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                logger.debug('pgvector extension ensured before pool creation')
            finally:
                await conn.close()

        except asyncpg.exceptions.InsufficientPrivilegeError as e:
            # Permission denied - common on managed services
            logger.error(
                'Cannot CREATE EXTENSION (insufficient privileges). '
                'Enable pgvector via database management interface: '
                'Supabase: Dashboard → Extensions → vector, '
                'AWS RDS: rds_superuser privileges required',
            )
            raise ConfigurationError(
                f'pgvector extension required but cannot be created (insufficient privileges): {e}',
            ) from e

        except asyncpg.exceptions.ClientConfigurationError as e:
            # Must precede the InterfaceError tuple below:
            # ClientConfigurationError subclasses InterfaceError, and asyncpg
            # raises it for permanent client-side misconfigurations (invalid
            # sslmode/target_session_attrs/gsslib values, unresolvable DSN
            # options) that a broad InterfaceError match would misclassify as
            # a retryable DependencyError and restart-loop on.
            logger.error(f'PostgreSQL client configuration invalid: {e}')
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
            logger.error(f'Failed to connect to PostgreSQL: {e}')
            raise DependencyError(
                f'PostgreSQL connection failed: {e}. '
                'Ensure PostgreSQL is running and accessible.',
            ) from e

        except asyncpg.exceptions.InvalidAuthorizationSpecificationError as e:
            # SQLSTATE class 28 as a whole (InvalidPasswordError 28P01 is a subclass):
            # 28000 covers 'no pg_hba.conf entry for host ...', equally permanent, so
            # both must reach exit 78 instead of the retryable terminal handler.
            logger.error(f'PostgreSQL authentication failed: {e}')
            raise ConfigurationError(
                f'PostgreSQL authentication failed: {e}. '
                'Check POSTGRESQL_USER, POSTGRESQL_PASSWORD and the server pg_hba.conf rules.',
            ) from e

        except asyncpg.exceptions.InvalidCatalogNameError as e:
            logger.error(f'PostgreSQL database does not exist: {e}')
            raise ConfigurationError(
                f'PostgreSQL database does not exist: {e}. '
                'Create the database or check POSTGRESQL_DATABASE.',
            ) from e

        except ValueError as e:
            # asyncpg raises a plain builtins.ValueError for a malformed DSN
            # authority (e.g. an invalid host literal) before any network I/O --
            # a permanent client-side misconfiguration, NOT ClientConfigurationError
            # (classified above). Without this arm the catch-all below would wrap it
            # as a retryable DependencyError (exit 69) and the supervisor would
            # restart-loop on a config error; initialize() classifies the same
            # ValueError as ConfigurationError (exit 78), but its
            # 'except DependencyError: raise' re-raise would win first when this
            # pre-check runs, so the classification must happen here too.
            logger.error(f'PostgreSQL configuration invalid: {e}')
            raise ConfigurationError(
                f'PostgreSQL configuration invalid: {e}. '
                'Check POSTGRESQL_HOST and POSTGRESQL_CONNECTION_STRING.',
            ) from e

        except Exception as e:
            logger.error(f'Failed to ensure pgvector extension: {e}')
            # Default to DependencyError for unknown errors (safer - allows retry)
            raise DependencyError(f'pgvector extension is required but could not be created: {e}') from e

    async def _resolve_provision_vector(self) -> bool:
        """Decide whether the pgvector extension and vector codec are needed at boot.

        The ``vector`` type is used ONLY by the fp32 ``vec_context_embeddings`` layout, so the
        extension and codec are needed exactly when that layout WILL be provisioned or ALREADY
        exists:

        - ``generation`` on + ``compression`` OFF: always provision, WITHOUT a probe (the common
          fast path). The compression-off server provisions the fp32 layout, so the extension must
          exist before its ``vector(dim)`` DDL runs.
        - ``generation`` on + ``compression`` ON (the v3.0.0 default): the server strips every fp32
          statement (``skip_fp32_vec``), stores payloads as BYTEA, and reads them in pure Python, so
          the ``vector`` type is never created or bound. Provisioning it anyway would force ``CREATE
          EXTENSION vector`` and CRASH boot on a host that lacks pgvector -- exactly the pgvector-free
          deployment the compression docs promote -- so fall through to the fp32-table probe and
          provision ONLY if a stray fp32 table actually exists (it needs the codec). The migration
          CLI does NOT rely on this gate at all: ``initialize_target_postgresql`` passes the
          explicit ``provision_vector`` constructor override keyed on ``with_semantic`` (and
          creates the extension itself, before its ``vector(dim)`` DDL, exactly when the fp32
          layout will be built), while the paths that DO read fp32 vectors (``--compress``, a
          PG->PG copy) operate on a database where the fp32 table already exists, so the probe
          returns True for them.
        - the fp32 ``vec_context_embeddings`` table already exists: it is read via the vector codec
          (an fp32 archive, or the ``--compress`` CLI reading it before it replaces it with the
          compressed table).
        - ``compression`` off + ``generation`` off + ``embedding_metadata`` present: the
          infra-present fallthrough re-provisions the fp32 layout, which needs the type.

        Returning True for every generation-on boot would force a compressed (BYTEA-only) server --
        the default configuration -- to create the unused pgvector extension, crashing boot on a
        pgvector-less PostgreSQL host. Keying on ``embedding_metadata`` presence alone cannot tell
        the two payload formats apart (that table exists under both), which is why the fp32 table
        itself is probed. A connection failure PROPAGATES rather than answering
        False: a probe cannot tell a connect fault from a negative answer, and a wrong False
        silently skips the extension and codec that a later ``vector(dim)`` DDL needs.
        ``initialize()`` calls this inside its classification ladder, so the propagated fault
        becomes ConfigurationError (exit 78) or DependencyError (exit 69) instead of a silently
        wrong provisioning decision.

        Returns:
            True when the pgvector extension and vector codec must be provisioned.
        """
        if settings.embedding.generation_enabled and not settings.compression.enabled:
            return True
        compression_enabled = settings.compression.enabled
        connect_kwargs = build_asyncpg_connect_kwargs()
        # A connect fault propagates DELIBERATELY. Swallowing it and returning False
        # would answer a question the probe never got to ask: on a
        # generation-off/compression-off database that already carries
        # embedding_metadata, False means the boot skips CREATE EXTENSION vector and
        # the vector codec, and the semantic migration's `CREATE TABLE ... vector(dim)`
        # then fails unclassified. initialize() calls this INSIDE its classification
        # ladder, so a wrong password becomes ConfigurationError (exit 78) and a
        # transient outage becomes DependencyError (exit 69, retried by the supervisor
        # with the probe re-run) instead of a silently wrong provisioning decision.
        conn = await asyncpg.connect(
            self.connection_string,
            timeout=settings.storage.postgresql_connect_timeout_s,
            **connect_kwargs,
        )
        try:
            # The to_regclass probes below resolve through search_path, so the session
            # parameters must be in place before the first one runs.
            await apply_session_gucs(conn)
            fp32_present = bool(
                await conn.fetchval("SELECT to_regclass('vec_context_embeddings') IS NOT NULL"),
            )
            if fp32_present:
                return True
            if compression_enabled:
                return False
            # compression off + generation off: the semantic/chunking infra-present
            # fallthrough re-provisions the fp32 vector layout iff embedding_metadata exists.
            return bool(await conn.fetchval("SELECT to_regclass('embedding_metadata') IS NOT NULL"))
        finally:
            await conn.close()
