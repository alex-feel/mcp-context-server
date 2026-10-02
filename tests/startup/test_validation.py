"""Tests for date parameter validation in app.startup.validation.

Covers validate_date_param (accepted formats, end-of-day expansion, canonicalization to extended ISO 8601,
rejected inputs) and validate_date_range.
"""

import pytest
from fastmcp.exceptions import ToolError

from app.startup.validation import validate_date_param
from app.startup.validation import validate_date_range


class TestDateValidation:
    """Test date parameter validation functions."""

    def test_valid_date_only_start_date(self) -> None:
        """Test date-only format (YYYY-MM-DD) for start_date - unchanged."""
        result = validate_date_param('2025-11-29', 'start_date')
        assert result == '2025-11-29'

    def test_valid_date_only_end_date_expands_to_end_of_day(self) -> None:
        """Test date-only format for end_date expands to end-of-day (T23:59:59.999999).

        This follows Elasticsearch precedent where missing time components are replaced
        with max values for 'lte' operations, matching user expectations that
        end_date='2025-11-29' should include ALL entries on November 29th.

        Uses microsecond precision (.999999) for PostgreSQL compatibility where
        CURRENT_TIMESTAMP stores microseconds (e.g., 23:59:59.500000).
        """
        result = validate_date_param('2025-11-29', 'end_date')
        assert result == '2025-11-29T23:59:59.999999'

    def test_end_date_with_datetime_not_expanded(self) -> None:
        """Test end_date with full datetime format is NOT expanded.

        Only date-only end_date values should be expanded to end-of-day.
        Explicit datetime values keep their instant (the Z suffix is
        canonicalized to the equivalent +00:00 offset).
        """
        # With T separator
        result = validate_date_param('2025-11-29T14:00:00', 'end_date')
        assert result == '2025-11-29T14:00:00'

        # With timezone (Z canonicalizes to +00:00)
        result = validate_date_param('2025-11-29T14:00:00Z', 'end_date')
        assert result == '2025-11-29T14:00:00+00:00'

        # With timezone offset
        result = validate_date_param('2025-11-29T14:00:00+02:00', 'end_date')
        assert result == '2025-11-29T14:00:00+02:00'

    def test_valid_datetime(self) -> None:
        """Test full datetime format without timezone."""
        result = validate_date_param('2025-11-29T10:00:00', 'start_date')
        assert result == '2025-11-29T10:00:00'

    def test_valid_datetime_with_timezone_offset(self) -> None:
        """Test datetime with timezone offset."""
        result = validate_date_param('2025-11-29T10:00:00+02:00', 'start_date')
        assert result == '2025-11-29T10:00:00+02:00'

    def test_valid_datetime_with_negative_timezone(self) -> None:
        """Test datetime with negative timezone offset."""
        result = validate_date_param('2025-11-29T10:00:00-05:00', 'start_date')
        assert result == '2025-11-29T10:00:00-05:00'

    def test_valid_datetime_utc_z_suffix(self) -> None:
        """Test datetime with Z suffix canonicalizes to the +00:00 offset form."""
        result = validate_date_param('2025-11-29T10:00:00Z', 'start_date')
        assert result == '2025-11-29T10:00:00+00:00'

    def test_valid_datetime_with_microseconds(self) -> None:
        """Test datetime with microseconds."""
        result = validate_date_param('2025-11-29T10:00:00.123456', 'start_date')
        assert result == '2025-11-29T10:00:00.123456'

    def test_none_passthrough(self) -> None:
        """Test None value passes through unchanged."""
        result = validate_date_param(None, 'start_date')
        assert result is None

    def test_invalid_format_day_month_year(self) -> None:
        """Test invalid DD-MM-YYYY format raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_param('29-11-2025', 'start_date')
        assert 'Invalid start_date format' in str(exc_info.value)
        assert 'ISO 8601' in str(exc_info.value)

    def test_invalid_format_slash_separator(self) -> None:
        """Test invalid YYYY/MM/DD format raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_param('2025/11/29', 'end_date')
        assert 'Invalid end_date format' in str(exc_info.value)

    def test_invalid_date_values(self) -> None:
        """Test invalid date values raise ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_param('2025-13-45', 'start_date')
        assert 'Invalid start_date format' in str(exc_info.value)

    def test_invalid_month_out_of_range(self) -> None:
        """Test month out of range raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_param('2025-00-15', 'start_date')
        assert 'Invalid start_date format' in str(exc_info.value)

    def test_invalid_empty_string(self) -> None:
        """Test empty string raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_param('', 'start_date')
        assert 'Invalid start_date format' in str(exc_info.value)

    def test_invalid_random_text(self) -> None:
        """Test random text raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_param('not-a-date', 'end_date')
        assert 'Invalid end_date format' in str(exc_info.value)


class TestDateCanonicalization:
    """validate_date_param re-serializes every accepted input in extended ISO form.

    Python's fromisoformat accepts a superset (compact '20250601', week dates
    '2025-W23-1', the space-separated datetime) of what the storage layer parses:
    SQLite's datetime() returns NULL for the compact and week-date forms (silently
    filtering out every row) and the PostgreSQL parameter parser keys its branch on
    the 'T' separator. Canonicalization guarantees both backends receive one
    representation they parse.
    """

    def test_space_separated_datetime_canonicalizes_to_t(self) -> None:
        """The documented space-separated datetime form gains the 'T' separator."""
        result = validate_date_param('2025-11-29 10:00:00', 'start_date')
        assert result == '2025-11-29T10:00:00'

    def test_compact_date_canonicalizes_to_extended(self) -> None:
        """The ISO 8601 basic/compact date form is expanded to the extended form."""
        result = validate_date_param('20250601', 'start_date')
        assert result == '2025-06-01'

    def test_week_date_canonicalizes_to_extended(self) -> None:
        """An ISO 8601 week date is resolved to its extended calendar date."""
        result = validate_date_param('2025-W23-1', 'start_date')
        assert result == '2025-06-02'

    def test_compact_end_date_expansion_is_not_mixed_format(self) -> None:
        """A compact end_date expands on the CANONICAL form, not the raw input."""
        result = validate_date_param('20250601', 'end_date')
        assert result == '2025-06-01T23:59:59.999999'

    @pytest.mark.parametrize(
        'raw',
        ['20250601', '2025-W23-1', '2025-11-29 10:00:00', '2025-11-29T10:00:00Z'],
    )
    def test_canonical_output_parses_in_sqlite_datetime(self, raw: str) -> None:
        """Every canonicalized value parses in SQLite's datetime() (non-NULL).

        The raw compact and week-date inputs return NULL from datetime(), which
        turns the WHERE clause into a filter that matches nothing on the default
        backend while PostgreSQL parses the same input.
        """
        import sqlite3

        canonical = validate_date_param(raw, 'start_date')
        with sqlite3.connect(':memory:') as conn:
            parsed = conn.execute('SELECT datetime(?)', (canonical,)).fetchone()[0]
        assert parsed is not None

    def test_sub_minute_offset_rejected(self) -> None:
        """A timezone offset carrying sub-minute precision is rejected loudly.

        Python's fromisoformat accepts '+05:30:15' and isoformat() re-serializes it
        verbatim, but SQLite's datetime() understands only a whole-minute '[+-]HH:MM'
        offset and returns NULL for anything finer, so the filter would silently match
        zero rows on SQLite while PostgreSQL filtered correctly. The validator rejects
        it rather than silently rounding the instant.
        """
        with pytest.raises(ToolError, match='sub-minute'):
            validate_date_param('2025-11-29T10:00:00+05:30:15', 'start_date')

    def test_whole_minute_offset_accepted_and_parses_in_sqlite(self) -> None:
        """A whole-minute offset (e.g. +05:30) is accepted and parses in SQLite datetime().

        Guards against over-rejecting a legitimate half-hour offset alongside the
        sub-minute rejection above.
        """
        import sqlite3

        canonical = validate_date_param('2025-11-29T10:00:00+05:30', 'start_date')
        assert canonical == '2025-11-29T10:00:00+05:30'
        with sqlite3.connect(':memory:') as conn:
            parsed = conn.execute('SELECT datetime(?)', (canonical,)).fetchone()[0]
        assert parsed is not None


class TestDateRangeValidation:
    """Test date range validation (start_date <= end_date)."""

    def test_valid_range_same_date(self) -> None:
        """Test same date for start and end is valid."""
        # Should not raise
        validate_date_range('2025-11-29', '2025-11-29')

    def test_valid_range_start_before_end(self) -> None:
        """Test start_date before end_date is valid."""
        # Should not raise
        validate_date_range('2025-11-01', '2025-11-30')

    def test_valid_range_with_datetimes(self) -> None:
        """Test datetime range is valid."""
        # Should not raise
        validate_date_range('2025-11-29T00:00:00', '2025-11-29T23:59:59')

    def test_invalid_range_start_after_end(self) -> None:
        """Test start_date after end_date raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_range('2025-12-01', '2025-11-01')
        assert 'Invalid date range' in str(exc_info.value)
        assert 'start_date' in str(exc_info.value)
        assert 'after' in str(exc_info.value)

    def test_invalid_range_with_time(self) -> None:
        """Test datetime range where start is after end."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_range('2025-11-29T23:00:00', '2025-11-29T10:00:00')
        assert 'Invalid date range' in str(exc_info.value)

    def test_none_start_date_valid(self) -> None:
        """Test None start_date with valid end_date."""
        # Should not raise
        validate_date_range(None, '2025-11-29')

    def test_none_end_date_valid(self) -> None:
        """Test valid start_date with None end_date."""
        # Should not raise
        validate_date_range('2025-11-29', None)

    def test_both_none_valid(self) -> None:
        """Test both dates None is valid."""
        # Should not raise
        validate_date_range(None, None)

    def test_mixed_offset_valid_range_accepted(self) -> None:
        """A valid range whose bounds carry different UTC offsets is accepted.

        start is 18:00 UTC and end is 20:00 UTC -- a valid two-hour range.
        Comparing the wall-clock values with the offsets stripped (23:00 > 20:00)
        would reject it.
        """
        # Should not raise
        validate_date_range('2025-06-01T23:00:00+05:00', '2025-06-01T20:00:00Z')

    def test_mixed_offset_inverted_range_rejected(self) -> None:
        """A truly inverted mixed-offset range is rejected.

        start is 10:00 UTC and end is 06:00 UTC -- inverted. Comparing the
        wall-clock values with the offsets stripped (10:00 < 11:00) would accept
        it and silently return empty results downstream.
        """
        with pytest.raises(ToolError) as exc_info:
            validate_date_range('2025-06-01T10:00:00Z', '2025-06-01T11:00:00+05:00')
        assert 'Invalid date range' in str(exc_info.value)

    def test_naive_and_aware_bounds_compare_as_utc(self) -> None:
        """A naive bound is interpreted as UTC when compared with an aware bound."""
        # naive 10:00 (=10:00 UTC) <= aware 12:00+00:00 -- valid; should not raise
        validate_date_range('2025-06-01T10:00:00', '2025-06-01T12:00:00+00:00')
        # naive 13:00 (=13:00 UTC) > aware 12:00+00:00 -- inverted
        with pytest.raises(ToolError):
            validate_date_range('2025-06-01T13:00:00', '2025-06-01T12:00:00+00:00')
