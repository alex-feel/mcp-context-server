"""
Backend factory for creating storage backend instances.

``create_backend`` builds the request backend that serves the tools, selected through
environment variables or explicit parameters. ``create_maintenance_backend`` builds the
backend that migrations, startup and the command-line tools run on: on PostgreSQL it logs
in as the role that owns the tables.
"""

from pathlib import Path
from typing import Literal

from app.backends.base import StorageBackend
from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.postgresql_backend.session import NO_CREDENTIAL_OVERRIDE
from app.backends.postgresql_backend.session import CredentialOverride
from app.backends.postgresql_backend.session_profiles import RLS_READ_VIEWS
from app.backends.postgresql_backend.session_profiles import MaintenanceProfile
from app.backends.postgresql_backend.session_profiles import RlsRequestProfile
from app.backends.postgresql_backend.session_profiles import SessionProfile
from app.backends.postgresql_backend.session_profiles import UnboundProfile
from app.backends.sqlite_backend import SQLiteBackend
from app.settings import get_settings


def create_backend(
    backend_type: Literal['sqlite', 'postgresql'] | None = None,
    db_path: Path | str | None = None,
    connection_string: str | None = None,
    provision_vector: bool | None = None,
) -> StorageBackend:
    """
    Create a storage backend instance based on type.

    This factory function creates the appropriate backend implementation based on
    the backend_type parameter or settings. It enables database-agnostic code by
    returning a StorageBackend protocol implementation.

    A PostgreSQL backend dials the request credentials. With ENABLE_POSTGRESQL_RLS it
    binds the caller's identity in every transaction and reads the context table through
    its read view; without it the application predicate is the whole access control.

    Args:
        backend_type: Type of backend to create ('sqlite', 'postgresql').
                     If None, reads from settings.storage.backend_type
        db_path: Path to database file (SQLite only). If None, uses settings.storage.db_path
        connection_string: PostgreSQL connection string (PostgreSQL only).
                          If None, builds from settings.storage.postgresql_* settings
        provision_vector: Explicit pgvector-provisioning decision (PostgreSQL only).
                          When set, overrides the backend's settings-and-probe resolution;
                          used by callers that know up front whether the database carries
                          the fp32 vector layout (e.g. the migration CLI's target init).
                          If None, the backend resolves it at initialize() time.

    Returns:
        StorageBackend implementation (SQLiteBackend, PostgreSQLBackend)

    Raises:
        ValueError: If backend_type is invalid or required parameters are missing

    Example:
        # Create SQLite backend from settings
        backend = create_backend()
        await backend.initialize()

        # Create SQLite backend with explicit path
        backend = create_backend(backend_type='sqlite', db_path='/data/context.db')
        await backend.initialize()

        # Create PostgreSQL backend from settings
        backend = create_backend(backend_type='postgresql')
        await backend.initialize()

        # Create PostgreSQL backend with explicit connection string
        backend = create_backend(
            backend_type='postgresql',
            connection_string='postgresql://user:pass@localhost:5432/dbname',
        )
        await backend.initialize()
    """
    settings = get_settings()

    # Determine backend type from parameter or settings
    if backend_type is None:
        backend_type = getattr(settings.storage, 'backend_type', 'sqlite')

    # Validate backend type
    if backend_type not in ('sqlite', 'postgresql'):
        raise ValueError(
            f'Invalid backend_type: {backend_type}. Must be one of: sqlite, postgresql',
        )

    # Create appropriate backend
    if backend_type == 'sqlite':
        # Determine database path
        if db_path is None:
            db_path = settings.storage.db_path
            if db_path is None:
                raise ValueError(
                    'db_path is required for SQLite backend. Provide via parameter or set DB_PATH environment variable.',
                )

        # Create SQLite backend
        return SQLiteBackend(db_path=db_path)

    # backend_type == 'postgresql'
    # With row-level security on, every transaction binds the caller and plain reads of the
    # context table go through its read view; otherwise the application predicate is the
    # whole access control, so transactions bind nothing and reads name the base tables.
    # The request backend always dials the request credentials.
    profile: SessionProfile = (
        RlsRequestProfile(read_views=RLS_READ_VIEWS)
        if settings.access_control.postgresql_rls_enabled
        else UnboundProfile()
    )
    return PostgreSQLBackend(
        connection_string=connection_string,
        provision_vector=provision_vector,
        profile=profile,
        credentials=NO_CREDENTIAL_OVERRIDE,
    )


def create_maintenance_backend(
    *,
    connection_string: str | None = None,
    provision_vector: bool | None = None,
) -> StorageBackend:
    """Create the backend migrations, startup and the command-line tools run on.

    On PostgreSQL it dials the request connection settings (``POSTGRESQL_CONNECTION_STRING``
    when set, otherwise the component fields) with ``POSTGRESQL_MIGRATION_USER`` and
    ``POSTGRESQL_MIGRATION_PASSWORD`` overriding the user and the password, each only when
    it is set, so a deployment without them dials exactly the request settings. The
    connections run as the table owner under the maintenance session profile: no identity
    is bound, reads name the base tables, and every connection turns ``row_security`` off.
    An explicit ``connection_string`` (the command-line tools) is dialed with its own
    credentials and no override. With the SQLite settings it returns the SQLite backend,
    which enforces access in the application layer only.

    Args:
        connection_string: PostgreSQL connection string whose credentials are authoritative.
            If None, the request connection settings with the migration credentials.
        provision_vector: Explicit pgvector-provisioning decision (PostgreSQL only); if
            None, the backend resolves it at initialize() time.

    Returns:
        The maintenance StorageBackend.
    """
    settings = get_settings()

    if connection_string is None and settings.storage.backend_type == 'sqlite':
        return create_backend(backend_type='sqlite')

    credentials = NO_CREDENTIAL_OVERRIDE
    if connection_string is None:
        credentials = CredentialOverride(
            user=settings.storage.postgresql_migration_user,
            password=settings.storage.postgresql_migration_password,
        )
    return PostgreSQLBackend(
        connection_string=connection_string,
        provision_vector=provision_vector,
        profile=MaintenanceProfile(),
        credentials=credentials,
    )
