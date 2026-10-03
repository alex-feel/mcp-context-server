"""Tests for the overfetch clamp in ``app.tools.search.hybrid``."""


class TestClampOverfetch:
    """The overfetch window is clamped to a safe ceiling before it is bound.

    The clamp is defense in depth: with offset already bounded it never trims a
    valid request, but no future multiplier change can grow the row count reaching
    a LIMIT/OFFSET bind out of range.
    """

    def test_clamps_value_above_ceiling(self) -> None:
        """A window above MAX_OVERFETCH_ROWS is clamped down to the ceiling."""
        from app.tools.search.hybrid import MAX_OVERFETCH_ROWS
        from app.tools.search.hybrid import _clamp_overfetch

        assert _clamp_overfetch(MAX_OVERFETCH_ROWS + 1) == MAX_OVERFETCH_ROWS
        assert _clamp_overfetch(MAX_OVERFETCH_ROWS * 1000) == MAX_OVERFETCH_ROWS

    def test_passes_value_below_ceiling_unchanged(self) -> None:
        """A window at or below the ceiling passes through untouched."""
        from app.tools.search.hybrid import MAX_OVERFETCH_ROWS
        from app.tools.search.hybrid import _clamp_overfetch

        assert _clamp_overfetch(0) == 0
        assert _clamp_overfetch(42) == 42
        assert _clamp_overfetch(MAX_OVERFETCH_ROWS - 1) == MAX_OVERFETCH_ROWS - 1
        assert _clamp_overfetch(MAX_OVERFETCH_ROWS) == MAX_OVERFETCH_ROWS
