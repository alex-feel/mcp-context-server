"""Tests for logger configuration and path shortening."""
from __future__ import annotations

import logging
import os

from app.logger_config import _MAX_PATH_SEGMENTS
from app.logger_config import _PACKAGE_DIR_MARKERS
from app.logger_config import config_logger


def _get_shortpathname(pathname: str) -> str:
    """Create a LogRecord, format it via the configured formatter, and return shortpathname."""
    config_logger('DEBUG')
    root = logging.getLogger()
    assert root.handlers, 'Expected at least one handler'
    formatter = root.handlers[0].formatter
    assert formatter is not None

    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname=pathname,
        lineno=42,
        msg='Test message',
        args=(),
        exc_info=None,
    )
    formatter.format(record)
    result = vars(record)['shortpathname']
    assert isinstance(result, str)
    return result


class TestPathShortening:
    """Test the _ShortPath formatter path shortening logic."""

    # --- site-packages paths ---

    def test_site_packages_prefix_stripped(self) -> None:
        """Path after site-packages with 3 segments is returned directly."""
        sep = os.sep
        path = sep.join([
            '', 'app', '.venv', 'lib', 'python3.12',
            'site-packages', 'app', 'tools', 'context.py',
        ])
        assert _get_shortpathname(path) == sep.join(['app', 'tools', 'context.py'])

    def test_third_party_site_packages(self) -> None:
        """Third-party library path after site-packages is stripped correctly."""
        sep = os.sep
        path = sep.join([
            '', 'app', '.venv', 'lib', 'python3.12',
            'site-packages', 'mcp', 'server', 'streamable_http.py',
        ])
        assert _get_shortpathname(path) == sep.join(
            ['mcp', 'server', 'streamable_http.py'],
        )

    def test_short_after_site_packages(self) -> None:
        """Two-segment path after site-packages returned directly."""
        sep = os.sep
        path = sep.join([
            '', 'app', '.venv', 'lib', 'python3.12',
            'site-packages', 'httpx', '_client.py',
        ])
        assert _get_shortpathname(path) == sep.join(['httpx', '_client.py'])

    def test_single_file_after_site_packages(self) -> None:
        """Single file directly under site-packages."""
        sep = os.sep
        path = sep.join([
            '', 'app', '.venv', 'lib', 'python3.12',
            'site-packages', 'module.py',
        ])
        assert _get_shortpathname(path) == 'module.py'

    def test_exactly_five_segments_after_site_packages(self) -> None:
        """Exactly _MAX_PATH_SEGMENTS segments after site-packages -- no capping needed."""
        sep = os.sep
        path = sep.join([
            '', 'app', '.venv', 'lib', 'python3.12',
            'site-packages', 'a', 'b', 'c', 'd', 'e.py',
        ])
        assert _get_shortpathname(path) == sep.join(['a', 'b', 'c', 'd', 'e.py'])

    def test_deep_path_after_site_packages_capped(self) -> None:
        """Deep path after site-packages is capped at _MAX_PATH_SEGMENTS."""
        sep = os.sep
        path = sep.join([
            '', 'app', '.venv', 'lib', 'python3.12',
            'site-packages', 'a', 'b', 'c', 'd', 'e', 'f', 'module.py',
        ])
        assert _get_shortpathname(path) == sep.join(
            ['c', 'd', 'e', 'f', 'module.py'],
        )

    # --- dist-packages paths ---

    def test_dist_packages_stripped(self) -> None:
        """Debian/Ubuntu dist-packages path is stripped correctly."""
        sep = os.sep
        path = sep.join([
            '', 'usr', 'lib', 'python3', 'dist-packages',
            'some_module', 'core.py',
        ])
        assert _get_shortpathname(path) == sep.join(['some_module', 'core.py'])

    # --- non-site-packages paths (local development) ---

    def test_no_marker_short_path_preserved(self) -> None:
        """Short path without any marker is returned unchanged."""
        sep = os.sep
        path = sep.join(['app', 'tools', 'context.py'])
        assert _get_shortpathname(path) == path

    def test_no_marker_long_path_capped(self) -> None:
        """Long path without marker is capped at last _MAX_PATH_SEGMENTS segments."""
        sep = os.sep
        path = sep.join([
            '', 'home', 'user', 'projects',
            'mcp-context-server', 'app', 'tools', 'context.py',
        ])
        # 8 segments (including empty string from leading sep): last 5
        expected = sep.join(
            ['projects', 'mcp-context-server', 'app', 'tools', 'context.py'],
        )
        assert _get_shortpathname(path) == expected

    # --- edge cases ---

    def test_site_packages_in_name_not_dir(self) -> None:
        """'site-packages' as part of a directory name (without trailing sep) is NOT stripped."""
        sep = os.sep
        path = sep.join(['', 'app', 'my-site-packages-tool', 'main.py'])
        # The marker is 'site-packages' + os.sep, so 'my-site-packages-tool' does NOT match
        assert _get_shortpathname(path) == path

    def test_site_packages_preferred_over_dist(self) -> None:
        """If both markers exist, site-packages (checked first) wins."""
        sep = os.sep
        path = sep.join([
            '', 'usr', 'lib', 'dist-packages', 'venv',
            'site-packages', 'pkg', 'mod.py',
        ])
        assert _get_shortpathname(path) == sep.join(['pkg', 'mod.py'])


class TestModuleConstants:
    """Test module-level constants are correctly defined."""

    def test_package_dir_markers_is_tuple(self) -> None:
        """_PACKAGE_DIR_MARKERS should be a tuple for immutability."""
        assert isinstance(_PACKAGE_DIR_MARKERS, tuple)

    def test_package_dir_markers_contain_separators(self) -> None:
        """Each marker must end with os.sep for correct matching."""
        for marker in _PACKAGE_DIR_MARKERS:
            assert marker.endswith(os.sep), f'{marker!r} must end with os.sep'

    def test_max_path_segments_value(self) -> None:
        """_MAX_PATH_SEGMENTS must be 5."""
        assert _MAX_PATH_SEGMENTS == 5


class TestConfigLogger:
    """Test the config_logger function itself."""

    def test_sets_root_level(self) -> None:
        """config_logger sets the root logger level."""
        config_logger('DEBUG')
        assert logging.getLogger().level == logging.DEBUG

    def test_case_insensitive_level(self) -> None:
        """Level string is case-insensitive."""
        config_logger('warning')
        assert logging.getLogger().level == logging.WARNING

    def test_invalid_level_defaults_to_error(self) -> None:
        """Invalid level string defaults to ERROR."""
        config_logger('INVALID')
        assert logging.getLogger().level == logging.ERROR

    def test_reconfigure_preserves_handler_count(self) -> None:
        """Re-calling config_logger does not add duplicate handlers."""
        config_logger('INFO')
        handler_count = len(logging.getLogger().handlers)
        config_logger('DEBUG')
        assert len(logging.getLogger().handlers) == handler_count
        assert logging.getLogger().level == logging.DEBUG
