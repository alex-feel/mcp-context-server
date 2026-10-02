"""Tests for SQLite datetime() normalization of search date bounds.

Stores entries through store_context and filters them through search_context against a real SQLite
database, covering the ISO 8601 forms (T separator, Z suffix, UTC offsets, date-only) that the context
repository's date clause normalizes with datetime().
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import pytest

import app.tools

# Get the actual async functions from app.tools
store_context = app.tools.store_context
search_context = app.tools.search_context


@pytest.mark.usefixtures('mock_server_dependencies', 'initialized_server')
class TestSQLiteDatetimeNormalization:
    """Test SQLite datetime() normalization for ISO 8601 formats.

    SQLite stores timestamps as TEXT in 'YYYY-MM-DD HH:MM:SS' format (space separator).
    ISO 8601 uses 'T' separator (e.g., '2025-11-29T10:00:00').

    Without datetime() normalization, TEXT comparison fails:
    - Space character (ASCII 0x20) < T character (ASCII 0x54)
    - Therefore: '2025-11-29 18:07:40' < '2025-11-29T00:00:00' (incorrect!)

    SQLite's datetime() function normalizes all ISO 8601 formats to space-separated format:
    - datetime('2025-11-29T10:00:00')       -> '2025-11-29 10:00:00'
    - datetime('2025-11-29T10:00:00Z')      -> '2025-11-29 10:00:00'
    - datetime('2025-11-29T10:00:00+02:00') -> '2025-11-29 08:00:00' (UTC converted)
    """

    @pytest.mark.asyncio
    async def test_sqlite_datetime_t_separator(self) -> None:
        """Test ISO 8601 with T separator works in SQLite.

        The T-separator form must match entries stored with the space separator;
        a plain TEXT comparison misses them because 'T' > ' ' in ASCII ordering.
        """
        # Store entry (will use CURRENT_TIMESTAMP which has space separator)
        result = await store_context(
            thread_id='sqlite-iso8601-t-test',
            source='user',
            text='Entry with space separator in timestamp',
        )
        assert result['success']

        # Get today's date in T-separator format (before stored time)
        today_t = datetime.now(UTC).strftime('%Y-%m-%dT00:00:00')

        # Search using T-separator format - should find the entry
        search_result = await search_context(
            thread_id='sqlite-iso8601-t-test',
            start_date=today_t,
        limit=50,
        )

        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry with space separator in timestamp'

    @pytest.mark.asyncio
    async def test_sqlite_datetime_z_suffix(self) -> None:
        """Test ISO 8601 with Z suffix (UTC) works in SQLite.

        SQLite datetime() treats Z suffix as UTC (no-op, already UTC).
        """
        # Store entry
        result = await store_context(
            thread_id='sqlite-iso8601-z-test',
            source='agent',
            text='Entry for Z suffix test',
        )
        assert result['success']

        # Get today in Z-suffix format
        today_z = datetime.now(UTC).strftime('%Y-%m-%dT00:00:00Z')

        # Search using Z-suffix format - should find the entry
        search_result = await search_context(
            thread_id='sqlite-iso8601-z-test',
            start_date=today_z,
        limit=50,
        )

        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry for Z suffix test'

    @pytest.mark.asyncio
    async def test_sqlite_datetime_positive_timezone_offset(self) -> None:
        """Test ISO 8601 with positive timezone offset (+HH:MM) works in SQLite.

        SQLite datetime() converts timezone offsets to UTC.
        For example: '2025-11-29T10:00:00+02:00' -> '2025-11-29 08:00:00' (UTC)
        """
        # Store entry
        result = await store_context(
            thread_id='sqlite-iso8601-tz-positive-test',
            source='user',
            text='Entry for positive timezone test',
        )
        assert result['success']

        # Use positive offset (e.g., Eastern European Time +02:00)
        # Use start of day in a positive timezone to ensure we catch the entry
        today_tz = datetime.now(UTC).strftime('%Y-%m-%dT00:00:00+02:00')

        # Search using timezone offset format - should find the entry
        search_result = await search_context(
            thread_id='sqlite-iso8601-tz-positive-test',
            start_date=today_tz,
        limit=50,
        )

        # Entry should be found (datetime() normalizes +02:00 to UTC, which is 2 hours earlier)
        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry for positive timezone test'

    @pytest.mark.asyncio
    async def test_sqlite_datetime_negative_timezone_offset(self) -> None:
        """Test ISO 8601 with negative timezone offset (-HH:MM) works in SQLite.

        SQLite datetime() converts timezone offsets to UTC.
        For example: '2025-11-29T10:00:00-05:00' -> '2025-11-29 15:00:00' (UTC)
        """
        # Store entry
        result = await store_context(
            thread_id='sqlite-iso8601-tz-negative-test',
            source='agent',
            text='Entry for negative timezone test',
        )
        assert result['success']

        # Use negative offset (e.g., EST -05:00)
        # Use yesterday at midnight in negative timezone to ensure we catch today's entry
        yesterday = datetime.now(UTC) - timedelta(days=1)
        yesterday_tz = yesterday.strftime('%Y-%m-%dT00:00:00-05:00')

        # Search using timezone offset format - should find the entry
        search_result = await search_context(
            thread_id='sqlite-iso8601-tz-negative-test',
            start_date=yesterday_tz,
        limit=50,
        )

        # Entry should be found (datetime() normalizes -05:00 to UTC, which is 5 hours later)
        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry for negative timezone test'

    @pytest.mark.asyncio
    async def test_sqlite_date_only_still_works(self) -> None:
        """Test that date-only format works with datetime() normalization.

        Date-only format (YYYY-MM-DD) normalizes to the start of the day.
        datetime('2025-11-29') -> '2025-11-29 00:00:00'
        """
        # Store entry
        result = await store_context(
            thread_id='sqlite-date-only-test',
            source='user',
            text='Entry for date-only test',
        )
        assert result['success']

        # Use date-only format
        today = datetime.now(UTC).strftime('%Y-%m-%d')

        # Search using date-only format - should find the entry
        search_result = await search_context(
            thread_id='sqlite-date-only-test',
            start_date=today,
        limit=50,
        )

        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry for date-only test'

    @pytest.mark.asyncio
    async def test_sqlite_datetime_end_date_with_t_separator(self) -> None:
        """Test end_date with T separator works correctly.

        Ensures both start_date and end_date handle T-separator properly.
        """
        # Store entry
        result = await store_context(
            thread_id='sqlite-end-date-t-test',
            source='user',
            text='Entry for end_date T separator test',
        )
        assert result['success']

        # Use date range with T-separator
        today_t_start = datetime.now(UTC).strftime('%Y-%m-%dT00:00:00')
        tomorrow_t_end = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%dT23:59:59')

        # Search using T-separator for both dates
        search_result = await search_context(
            thread_id='sqlite-end-date-t-test',
            start_date=today_t_start,
            end_date=tomorrow_t_end,
        limit=50,
        )

        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry for end_date T separator test'

    @pytest.mark.asyncio
    async def test_sqlite_datetime_mixed_formats(self) -> None:
        """Test mixing date-only start with T-separator end date.

        Ensures different formats can be mixed in the same query.
        """
        # Store entry
        result = await store_context(
            thread_id='sqlite-mixed-format-test',
            source='agent',
            text='Entry for mixed format test',
        )
        assert result['success']

        # Use date-only for start, T-separator for end
        today_date_only = datetime.now(UTC).strftime('%Y-%m-%d')
        tomorrow_t = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%dT23:59:59Z')

        # Search using mixed formats
        search_result = await search_context(
            thread_id='sqlite-mixed-format-test',
            start_date=today_date_only,
            end_date=tomorrow_t,
        limit=50,
        )

        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry for mixed format test'
