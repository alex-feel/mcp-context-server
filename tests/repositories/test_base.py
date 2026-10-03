"""Tests for BaseRepository helpers.

Covers _parse_date_for_postgresql, which converts ISO 8601 strings into the timezone-aware datetime
objects asyncpg needs for TIMESTAMPTZ parameters.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import pytest


class TestParseDateForPostgresql:
    """Test _parse_date_for_postgresql helper for asyncpg datetime conversion.

    asyncpg requires Python datetime objects for TIMESTAMPTZ parameters.
    This helper converts ISO 8601 date strings to datetime objects.
    """

    def test_none_returns_none(self) -> None:
        """Test None input returns None."""
        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql(None)
        assert result is None

    def test_space_separated_datetime_parses_as_datetime(self) -> None:
        """The space-separated form routes to the datetime branch, not date-only.

        A predicate keyed on 'T' alone would route this form into the date-only
        branch, where parsing fails; a None result binds as created_at >= NULL
        and silently returns zero rows on PostgreSQL, while SQLite handles the
        same input.
        """
        from app.repositories.base import BaseRepository

        parsed = BaseRepository._parse_date_for_postgresql('2025-11-29 10:00:00')
        assert parsed == datetime(2025, 11, 29, 10, 0, 0, tzinfo=UTC)

    def test_unparseable_input_raises_instead_of_returning_none(self) -> None:
        """Parser drift surfaces loudly instead of degrading to a NULL bind."""
        from app.repositories.base import BaseRepository

        with pytest.raises(ValueError, match='validate_date_param'):
            BaseRepository._parse_date_for_postgresql('not-a-date')

    def test_date_only_returns_datetime_utc(self) -> None:
        """Test date-only format returns datetime at start of day UTC."""

        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql('2025-11-29')
        assert result is not None
        assert result.year == 2025
        assert result.month == 11
        assert result.day == 29
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0
        assert result.tzinfo == UTC

    def test_datetime_without_timezone_interpreted_as_utc(self) -> None:
        """Test naive datetime (without timezone) is interpreted as UTC.

        Industry standard: Naive datetime is interpreted as UTC to match
        Elasticsearch, MongoDB, DynamoDB, and Firestore behavior.
        This ensures deterministic behavior regardless of server timezone.
        """
        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql('2025-11-29T10:30:45')
        assert result is not None
        assert result.year == 2025
        assert result.month == 11
        assert result.day == 29
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 45
        # A naive datetime is interpreted as UTC (industry standard)
        assert result.tzinfo == UTC

    def test_datetime_with_z_suffix(self) -> None:
        """Test datetime with Z suffix returns UTC datetime."""

        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql('2025-11-29T10:30:45Z')
        assert result is not None
        assert result.year == 2025
        assert result.hour == 10
        assert result.tzinfo == UTC

    def test_datetime_with_positive_offset(self) -> None:
        """Test datetime with positive timezone offset."""
        from datetime import timezone

        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql('2025-11-29T10:30:45+02:00')
        assert result is not None
        assert result.year == 2025
        assert result.hour == 10
        expected_tz = timezone(timedelta(hours=2))
        assert result.tzinfo == expected_tz

    def test_datetime_with_negative_offset(self) -> None:
        """Test datetime with negative timezone offset."""
        from datetime import timezone

        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql('2025-11-29T10:30:45-05:00')
        assert result is not None
        assert result.year == 2025
        assert result.hour == 10
        expected_tz = timezone(timedelta(hours=-5))
        assert result.tzinfo == expected_tz

    def test_datetime_with_microseconds(self) -> None:
        """Test datetime with microseconds."""
        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql('2025-11-29T10:30:45.123456')
        assert result is not None
        assert result.microsecond == 123456

    def test_result_is_datetime_type(self) -> None:
        """Test that result is always a datetime object (not date)."""
        from datetime import datetime as dt

        from app.repositories.base import BaseRepository

        # Date-only input should still return datetime
        result = BaseRepository._parse_date_for_postgresql('2025-11-29')
        assert isinstance(result, dt)
