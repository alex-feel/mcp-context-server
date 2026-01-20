"""Tests for Pgpool-II detection functionality."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import asyncpg.exceptions
import pytest


class TestPgpoolDetection:
    """Test Pgpool-II detection in PostgreSQLBackend."""

    @pytest.mark.asyncio
    async def test_pgpool_detected_when_show_pool_version_succeeds(self) -> None:
        """Pgpool-II should be detected when SHOW POOL_VERSION returns a value."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()

        # Mock connection that returns Pgpool-II version
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value='4.5.2 (firebrick)')

        mock_pool_acquire = AsyncMock()
        mock_pool_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool_acquire.__aexit__ = AsyncMock(return_value=None)
        backend._pool.acquire = MagicMock(return_value=mock_pool_acquire)

        await backend._detect_pgpool_ii()

        assert backend._pgpool_version == '4.5.2 (firebrick)'
        mock_conn.fetchval.assert_called_once_with('SHOW POOL_VERSION')

    @pytest.mark.asyncio
    async def test_direct_connection_when_undefined_object_error(self) -> None:
        """Direct PostgreSQL connection detected when SHOW POOL_VERSION raises UndefinedObjectError."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()

        # Mock connection that raises UndefinedObjectError (error code 42704)
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(
            side_effect=asyncpg.exceptions.UndefinedObjectError(
                'unrecognized configuration parameter "pool_version"',
            ),
        )

        mock_pool_acquire = AsyncMock()
        mock_pool_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool_acquire.__aexit__ = AsyncMock(return_value=None)
        backend._pool.acquire = MagicMock(return_value=mock_pool_acquire)

        await backend._detect_pgpool_ii()

        assert backend._pgpool_version is None

    @pytest.mark.asyncio
    async def test_detection_handles_empty_version_response(self) -> None:
        """Detection should handle empty version response gracefully."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()

        # Mock connection that returns empty/None
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)

        mock_pool_acquire = AsyncMock()
        mock_pool_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool_acquire.__aexit__ = AsyncMock(return_value=None)
        backend._pool.acquire = MagicMock(return_value=mock_pool_acquire)

        await backend._detect_pgpool_ii()

        assert backend._pgpool_version is None

    @pytest.mark.asyncio
    async def test_detection_handles_unexpected_error(self) -> None:
        """Detection should not fail initialization on unexpected errors."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()

        # Mock connection that raises unexpected error
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=RuntimeError('Unexpected error'))

        mock_pool_acquire = AsyncMock()
        mock_pool_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool_acquire.__aexit__ = AsyncMock(return_value=None)
        backend._pool.acquire = MagicMock(return_value=mock_pool_acquire)

        # Should not raise, just log and continue
        await backend._detect_pgpool_ii()

        assert backend._pgpool_version is None

    def test_metrics_include_pgpool_info_when_detected(self) -> None:
        """get_metrics() should include Pgpool-II detection results when detected."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()
        backend._pool.get_size = MagicMock(return_value=5)
        backend._pool.get_idle_size = MagicMock(return_value=3)
        backend._pool.get_min_size = MagicMock(return_value=2)
        backend._pool.get_max_size = MagicMock(return_value=10)
        backend._pgpool_version = '4.5.2 (firebrick)'

        metrics = backend.get_metrics()

        assert metrics['pgpool_detected'] is True
        assert metrics['pgpool_version'] == '4.5.2 (firebrick)'

    def test_metrics_include_pgpool_info_when_not_detected(self) -> None:
        """get_metrics() should include pgpool_detected=False when not behind Pgpool-II."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()
        backend._pool.get_size = MagicMock(return_value=5)
        backend._pool.get_idle_size = MagicMock(return_value=3)
        backend._pool.get_min_size = MagicMock(return_value=2)
        backend._pool.get_max_size = MagicMock(return_value=10)
        backend._pgpool_version = None

        metrics = backend.get_metrics()

        assert metrics['pgpool_detected'] is False
        assert metrics['pgpool_version'] is None

    def test_metrics_omit_pgpool_info_before_detection_runs(self) -> None:
        """get_metrics() should not include pgpool fields if detection never ran."""
        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend()
        backend._pool = MagicMock()
        backend._pool.get_size = MagicMock(return_value=5)
        backend._pool.get_idle_size = MagicMock(return_value=3)
        backend._pool.get_min_size = MagicMock(return_value=2)
        backend._pool.get_max_size = MagicMock(return_value=10)
        # _pgpool_version attribute not set (detection never ran)

        metrics = backend.get_metrics()

        assert 'pgpool_detected' not in metrics
        assert 'pgpool_version' not in metrics
