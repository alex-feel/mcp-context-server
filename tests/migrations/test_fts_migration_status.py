"""FTS migration status: FtsMigrationStatus, time estimates, the in-progress fts_search_context response, reset."""

from collections.abc import Generator
from unittest.mock import patch

import pytest


class TestFtsGracefulDegradation:
    """Test FTS graceful degradation during migration.

    These tests verify that when FTS migration is in progress, the fts_search_context
    tool returns informative responses instead of errors.
    """

    @pytest.fixture
    def reset_migration_status(self) -> Generator[None, None, None]:
        """Reset FTS migration status before and after each test.

        Uses the reset function exported by app.migrations to ensure clean state.

        Yields:
            None: Fixture provides no value, only cleanup behavior.
        """
        from app.migrations import reset_fts_migration_status as _reset_fts_migration_status

        _reset_fts_migration_status()
        yield
        _reset_fts_migration_status()

    def test_migration_status_dataclass_creation(self) -> None:
        """Test that FtsMigrationStatus dataclass can be created with all fields."""
        from datetime import UTC
        from datetime import datetime

        from app.migrations import FtsMigrationStatus

        status = FtsMigrationStatus(
            in_progress=True,
            started_at=datetime.now(tz=UTC),
            estimated_seconds=120,
            backend='sqlite',
            old_language='english',
            new_language='german',
            records_count=1000,
        )

        assert status.in_progress is True
        assert status.started_at is not None
        assert status.estimated_seconds == 120
        assert status.backend == 'sqlite'
        assert status.old_language == 'english'
        assert status.new_language == 'german'
        assert status.records_count == 1000

    def test_migration_status_defaults(self) -> None:
        """Test that FtsMigrationStatus dataclass has correct defaults."""
        from app.migrations import FtsMigrationStatus

        status = FtsMigrationStatus()

        assert status.in_progress is False
        assert status.started_at is None
        assert status.estimated_seconds is None
        assert status.backend is None
        assert status.old_language is None
        assert status.new_language is None
        assert status.records_count is None

    def test_estimate_migration_time_small_dataset(self) -> None:
        """Test migration time estimation for small dataset."""
        from app.migrations import estimate_migration_time

        # Small dataset: returns minimum time (around 2 seconds)
        estimated = estimate_migration_time(100)
        assert estimated >= 1  # Minimum bound
        assert estimated <= 10  # Should be quick

    def test_estimate_migration_time_large_dataset(self) -> None:
        """Test migration time estimation for large dataset."""
        from app.migrations import estimate_migration_time

        # Large dataset: should scale appropriately
        estimated = estimate_migration_time(100000)
        assert estimated >= 60  # Should take more time
        # Rough estimate: ~10-15 sec per 1000 records, so 100k = 1000-1500 sec

    def test_estimate_migration_time_zero_records(self) -> None:
        """Test migration time estimation for zero records."""
        from app.migrations import estimate_migration_time

        # Zero records: returns minimum time (around 2 seconds)
        estimated = estimate_migration_time(0)
        assert estimated >= 1  # Minimum bound for setup overhead
        assert estimated <= 10  # Should be quick

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reset_migration_status')
    async def test_graceful_degradation_response_structure(self) -> None:
        """Test that migration in progress response has correct structure.

        This test verifies the FtsMigrationInProgressDict TypedDict structure.
        """
        from datetime import UTC
        from datetime import datetime

        from app.migrations import FtsMigrationStatus

        # Create a migration in progress status
        migration_status = FtsMigrationStatus(
            in_progress=True,
            started_at=datetime.now(tz=UTC),
            estimated_seconds=120,
            backend='sqlite',
            old_language='unicode61',
            new_language='porter unicode61',
            records_count=1000,
        )

        # Mock the global migration status
        with patch('app.migrations.fts._fts_migration_status', migration_status):
            from app.tools import fts_search_context

            # Call the tool function directly
            result = await fts_search_context(query='test query', limit=50)

            # Verify response structure for migration in progress
            assert result['migration_in_progress'] is True
            assert 'message' in result
            assert 'started_at' in result
            assert 'estimated_remaining_seconds' in result
            assert 'old_language' in result
            assert 'new_language' in result
            assert 'suggestion' in result

            # Verify message content
            assert 'being rebuilt' in result['message'].lower()
            assert 'porter unicode61' in result['message']

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reset_migration_status')
    async def test_graceful_degradation_remaining_time_calculation(self) -> None:
        """Test that remaining time is calculated correctly during migration."""
        from datetime import UTC
        from datetime import datetime
        from datetime import timedelta

        from app.migrations import FtsMigrationStatus

        # Create a migration that started 30 seconds ago with 120 second estimate
        start_time = datetime.now(tz=UTC) - timedelta(seconds=30)
        migration_status = FtsMigrationStatus(
            in_progress=True,
            started_at=start_time,
            estimated_seconds=120,
            backend='sqlite',
            old_language='unicode61',
            new_language='porter unicode61',
            records_count=1000,
        )

        with patch('app.migrations.fts._fts_migration_status', migration_status):
            from app.tools import fts_search_context

            result = await fts_search_context(query='test query', limit=50)

            # Should have approximately 90 seconds remaining (120 - 30)
            remaining = result['estimated_remaining_seconds']
            assert 80 <= remaining <= 100  # Allow some tolerance for timing

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reset_migration_status')
    async def test_graceful_degradation_suggestion_format(self) -> None:
        """Test that suggestion message includes retry time."""
        from datetime import UTC
        from datetime import datetime

        from app.migrations import FtsMigrationStatus

        migration_status = FtsMigrationStatus(
            in_progress=True,
            started_at=datetime.now(tz=UTC),
            estimated_seconds=60,
            backend='postgresql',
            old_language='english',
            new_language='german',
            records_count=500,
        )

        with patch('app.migrations.fts._fts_migration_status', migration_status):
            from app.tools import fts_search_context

            result = await fts_search_context(query='test query', limit=50)

            # Verify suggestion includes retry time
            assert 'retry' in result['suggestion'].lower()
            assert 'seconds' in result['suggestion'].lower()


class TestResetFtsMigrationStatus:
    """Tests for reset_fts_migration_status()."""

    def test_resets_to_default(self) -> None:
        """Test global status reset to defaults."""
        from datetime import UTC
        from datetime import datetime

        # First, set the global status to a non-default value
        import app.migrations.fts as fts_module
        from app.migrations import FtsMigrationStatus
        from app.migrations import reset_fts_migration_status as _reset_fts_migration_status

        original_status = fts_module._fts_migration_status

        try:
            # Set migration in progress
            fts_module._fts_migration_status = FtsMigrationStatus(
                in_progress=True,
                started_at=datetime.now(tz=UTC),
                estimated_seconds=120,
                backend='sqlite',
                old_language='english',
                new_language='german',
                records_count=1000,
            )

            # Verify it's set
            assert fts_module._fts_migration_status.in_progress is True
            assert fts_module._fts_migration_status.estimated_seconds == 120

            # Reset to defaults
            _reset_fts_migration_status()

            # Verify it's back to defaults (capture status to avoid mypy narrowing issues)
            reset_status = fts_module._fts_migration_status
            assert reset_status.in_progress is False
            assert reset_status.started_at is None
            assert reset_status.estimated_seconds is None
            assert reset_status.backend is None
            assert reset_status.old_language is None
            assert reset_status.new_language is None
            assert reset_status.records_count is None
        finally:
            # Restore original status
            fts_module._fts_migration_status = original_status

    def test_reset_creates_new_instance(self) -> None:
        """Test that reset creates a fresh FtsMigrationStatus instance."""
        from datetime import UTC
        from datetime import datetime

        import app.migrations.fts as fts_module
        from app.migrations import FtsMigrationStatus
        from app.migrations import reset_fts_migration_status as _reset_fts_migration_status

        original_status = fts_module._fts_migration_status

        try:
            # Set migration in progress
            old_instance = FtsMigrationStatus(
                in_progress=True,
                started_at=datetime.now(tz=UTC),
                estimated_seconds=60,
            )
            fts_module._fts_migration_status = old_instance

            # Get reference to old instance
            pre_reset_id = id(fts_module._fts_migration_status)

            # Reset
            _reset_fts_migration_status()

            # Should be a new instance (capture to avoid mypy narrowing)
            new_status = fts_module._fts_migration_status
            post_reset_id = id(new_status)
            assert pre_reset_id != post_reset_id

            # But it should be equivalent to default
            default = FtsMigrationStatus()
            assert new_status.in_progress == default.in_progress
            assert new_status.started_at == default.started_at
        finally:
            fts_module._fts_migration_status = original_status

    def test_reset_idempotent(self) -> None:
        """Test that calling reset multiple times is safe."""
        import app.migrations.fts as fts_module
        from app.migrations import reset_fts_migration_status as _reset_fts_migration_status

        original_status = fts_module._fts_migration_status

        try:
            # Call reset multiple times
            _reset_fts_migration_status()
            status_1 = fts_module._fts_migration_status

            _reset_fts_migration_status()
            status_2 = fts_module._fts_migration_status

            _reset_fts_migration_status()
            status_3 = fts_module._fts_migration_status

            # All should have default values
            assert status_1.in_progress is False
            assert status_2.in_progress is False
            assert status_3.in_progress is False
        finally:
            fts_module._fts_migration_status = original_status
