"""Test-environment detection and the connection-pool configuration of the SQLite backend."""

import os
import sys
from dataclasses import dataclass


def is_test_environment() -> bool:
    """Detect if running in test environment.

    Returns:
        bool: True if running in test environment, False otherwise
    """
    return any([
        'pytest' in sys.modules,
        os.environ.get('PYTEST_CURRENT_TEST'),
        os.environ.get('CI') == 'true',
    ])


@dataclass
class PoolConfig:
    """Configuration for connection pooling."""

    max_readers: int = 8
    connection_timeout: float = 10.0
    # Recycle the writer connection after this many seconds without a write, so
    # an idle process stops pinning the database file and its -wal/-shm siblings;
    # the next write recreates it. Applied by the health-check loop. Readers need
    # no equivalent: they are per-use connections closed right after each read.
    idle_timeout: float = 300.0
    health_check_interval: float = 30.0

    def __post_init__(self) -> None:
        """Adjust settings based on environment."""
        if is_test_environment():
            # Optimize for test environment
            self.connection_timeout = 1.0  # Fast timeout in tests
            self.health_check_interval = 5.0  # More frequent health checks in tests
