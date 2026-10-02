"""Backend builders shared by the PostgreSQL backend test modules."""

from app.backends.postgresql_backend import PostgreSQLBackend


def build_backend(connection_string: str = 'postgresql://postgres:postgres@localhost:5432/testdb') -> PostgreSQLBackend:
    """Build a non-shut-down backend without a pool.

    Args:
        connection_string: DSN handed to the backend.

    Returns:
        The constructed backend.
    """
    backend = PostgreSQLBackend(connection_string=connection_string)
    backend._shutdown = False
    return backend


def fast_retries(backend: PostgreSQLBackend) -> None:
    """Make the write retry loop deterministic and sleepless.

    Args:
        backend: The backend whose retry configuration is tightened.
    """
    backend.retry_config.max_retries = 3
    backend.retry_config.base_delay = 0.0
    backend.retry_config.max_delay = 0.0
    backend.retry_config.jitter = False
