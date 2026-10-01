"""Error formatting conformance for every code file under app/tools.

Tool errors are rendered through format_exception_message; a bare str(e) is
allowed only in logger calls, which are internal diagnostics.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_FILES = sorted(path.relative_to(REPO_ROOT).as_posix() for path in (REPO_ROOT / 'app' / 'tools').rglob('*.py'))


class TestErrorFormatting:
    """Verify all tool files use format_exception_message instead of str(e)."""

    @pytest.mark.parametrize('file_path', TOOL_FILES)
    def test_no_str_e_in_tool_errors(self, file_path: str) -> None:
        """Verify no tool file uses str(e) in error contexts."""
        content = (REPO_ROOT / file_path).read_text(encoding='utf-8')
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if 'str(e)' in stripped:
                # Allow in logger calls (internal diagnostics)
                if stripped.startswith('logger.'):
                    continue
                pytest.fail(f'{file_path}:{i}: Found str(e) in non-logger context: {stripped}')
