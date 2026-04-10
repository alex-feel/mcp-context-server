#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pyyaml"]
# ///
"""
Base configuration loader for Claude Code hooks.

This module provides standardized YAML/JSON config file loading
with sensible defaults, error handling, and type safety.

Usage in hooks:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from hook_config_loader import get_config_from_argv

    DEFAULT_CONFIG = {'protected_files': ['LICENSE']}
    config = get_config_from_argv(DEFAULT_CONFIG)
"""

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import cast

yaml: ModuleType | None
try:
    import yaml as _yaml
    yaml = _yaml
except ImportError:
    yaml = None


STANDARD_EXTENSIONS: dict[str, list[str]] = {
    'python': ['.py'],
    'web': ['.ts', '.tsx', '.js', '.jsx'],
}


def load_config(
    config_path: str | None = None,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load configuration from file or return defaults.

    Args:
        config_path: Path to YAML/JSON config file. If None, returns defaults.
        defaults: Default configuration to use when no file or file missing.

    Returns:
        Configuration dictionary with defaults merged.

    Raises:
        ValueError: If config file exists but has invalid format or YAML
            required but pyyaml not installed.
    """
    if defaults is None:
        defaults = {}

    if not config_path:
        return defaults.copy()

    path = Path(config_path)
    if not path.exists():
        return defaults.copy()

    content = path.read_text(encoding='utf-8')

    loaded: Any
    if path.suffix in ('.yaml', '.yml'):
        if yaml is None:
            raise ValueError(
                f'YAML config file {path} requires pyyaml package. '
                'Add inline script dependency or use JSON format.',
            )
        loaded = yaml.safe_load(content)
    elif path.suffix == '.json':
        loaded = json.loads(content)
    else:
        # Try YAML first (most common), fall back to JSON
        if yaml is not None:
            try:
                loaded = yaml.safe_load(content)
            except Exception:
                loaded = json.loads(content)
        else:
            loaded = json.loads(content)

    if loaded is None:
        return defaults.copy()

    if not isinstance(loaded, dict):
        raise ValueError(f'Config file {path} must contain a dictionary at root level')

    # Merge with defaults (config values override defaults)
    # Cast needed because isinstance narrows Any to dict[Unknown, Unknown]
    result = defaults.copy()
    typed_loaded = cast(dict[str, Any], loaded)
    result.update(typed_loaded)
    return result


def get_config_from_argv(defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convenience function to load config from command-line argument.

    Expects config path as sys.argv[1] if provided.

    Args:
        defaults: Default configuration when no config provided.

    Returns:
        Configuration dictionary.
    """
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    return load_config(config_path, defaults)


def check_file_relevance(
    config: dict[str, Any],
    input_data: dict[str, Any],
) -> tuple[bool, str | None]:
    """Check if the edited file matches the hook's target file extensions.

    Reads 'file_extensions' from config and checks against the file path
    from the hook input data.

    Args:
        config: Hook configuration dictionary (should contain 'file_extensions' key).
        input_data: Hook event input data (JSON from stdin).

    Returns:
        Tuple of (is_relevant, file_path).
        is_relevant is True if file matches or no filtering configured.
        file_path is the extracted path, or None if not found.
    """
    file_extensions = config.get('file_extensions')
    if not file_extensions:
        return True, None

    tool_input = input_data.get('tool_input', {})
    file_path = tool_input.get('file_path')
    if not file_path:
        tool_response = input_data.get('tool_response', {})
        file_path = tool_response.get('filePath')

    if not file_path:
        return False, None

    ext = Path(file_path).suffix.lower()
    return ext in file_extensions, file_path
