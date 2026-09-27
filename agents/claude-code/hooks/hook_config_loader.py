#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pyyaml"]
# ///
"""
Base configuration loader for Claude Code hooks.

This module provides standardized YAML/JSON config file loading
with sensible defaults, error handling, and type safety.

Usage in hooks (dynamic loading via importlib.util)::

    import importlib.util
    from pathlib import Path
    from types import ModuleType

    def _load_config_loader() -> ModuleType:
        loader_path = Path(__file__).parent / 'hook_config_loader.py'
        spec = importlib.util.spec_from_file_location('hook_config_loader', loader_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load hook_config_loader from {loader_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    DEFAULT_CONFIG = {'protected_files': ['LICENSE']}
    config = _load_config_loader().get_config_from_argv(DEFAULT_CONFIG)

Hooks that judge an edited file also use it to decide whether the file is one
their configuration skips: check_file_relevance() for the file a tool names,
and path_is_skipped() for a file a shell command writes. Two optional keys
drive that decision. 'exclude_paths' lists path globs. 'exclude_temp_paths:
true' also skips loose scratch files in the system's temporary locations, as
judged by a function the calling hook passes in; this module holds no opinion
on what a scratch file is, so a hook that passes no judge skips none.
"""

import json
import os
import posixpath
import sys
from fnmatch import fnmatch
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Protocol
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

_WINDOWS = sys.platform == 'win32'


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


def _fully_absolute(path: str) -> bool:
    """Report whether a path names one location regardless of any current directory.

    A Windows path rooted without a drive (``/tmp/x``) depends on the current
    drive, so it does not qualify there.

    Args:
        path: The path to classify.

    Returns:
        True when the path is absolute in the platform's own full sense.
    """
    if _WINDOWS:
        drive, rest = os.path.splitdrive(path)
        return bool(drive) and rest.startswith(('\\', '/'))
    return os.path.isabs(path)


def _outside_worktree(normalized: str) -> str:
    """Map a path inside a Claude Code worktree onto the same path in its checkout.

    Claude Code creates a worktree at ``<repo>/.claude/worktrees/<name>/``. A
    worktree is a full checkout of the project rather than harness internals, so
    a file in it is judged as the file it mirrors in the main checkout: a
    ``.claude/**`` exclusion keeps skipping the worktree's own ``.claude`` files
    without swallowing everything else in it. A worktree created inside another
    worktree maps through every level.

    Args:
        normalized: A forward-slash path with ``..`` segments already collapsed.

    Returns:
        The path with every worktree prefix removed, or the path unchanged when it
        does not lie strictly inside a worktree.
    """
    segments = normalized.split('/')
    while True:
        for index in range(len(segments) - 3):
            if segments[index] == '.claude' and segments[index + 1] == 'worktrees' and segments[index + 2]:
                segments = segments[:index] + segments[index + 3:]
                break
        else:
            return '/'.join(segments)


def _comparable_path(file_path: str) -> str:
    """Return a path in the form exclusion patterns are matched against.

    An absolute path is resolved to its real location, which collapses ``..``
    segments and follows links, so neither a spelling such as ``.claude/../src``
    nor a link can move a file into or out of an exclusion. A relative path has
    its ``..`` segments collapsed lexically, because resolving it would depend on
    the directory the hook happens to run in.

    Args:
        file_path: The candidate file path (absolute or relative).

    Returns:
        The normalized forward-slash path, mapped out of any Claude Code worktree.
    """
    if _fully_absolute(file_path):
        try:
            resolved = os.path.realpath(file_path)
        except (OSError, ValueError):
            resolved = os.path.normpath(file_path)
        normalized = resolved.replace('\\', '/')
    else:
        normalized = posixpath.normpath(file_path.replace('\\', '/'))
    return _outside_worktree(normalized)


def _path_is_excluded(file_path: str, patterns: list[str]) -> bool:
    """Return True when file_path falls under any exclusion glob in patterns.

    Matching is platform-neutral: backslashes are normalized to forward slashes
    before comparison, and the path is first normalized and mapped out of any
    Claude Code worktree (see _comparable_path). A pattern ending in ``/**`` (for
    example ``.workflows/**``) matches that directory at any depth by path
    segment; a bare directory name (for example ``.workflows``) matches when it
    appears as a path segment; any other pattern is matched with ``fnmatch``
    against the full normalized path.

    Args:
        file_path: The candidate file path (absolute or relative).
        patterns: Exclusion globs from the hook config's ``exclude_paths``.

    Returns:
        True if the path matches any exclusion pattern, False otherwise.
    """
    normalized = _comparable_path(file_path)
    segments = [segment for segment in normalized.split('/') if segment]
    for pattern in patterns:
        normalized_pattern = pattern.replace('\\', '/')
        if normalized_pattern.endswith('/**'):
            directory = normalized_pattern[:-3].strip('/')
            if directory and directory in segments:
                return True
            continue
        bare = normalized_pattern.strip('/')
        if bare and bare in segments:
            return True
        if fnmatch(normalized, normalized_pattern):
            return True
    return False


def _anchored(file_path: str, input_data: dict[str, Any]) -> str:
    """Join a relative path onto the event's working directory.

    A path rooted in any way (a drive, a leading separator, a leading ``~``) is
    returned unchanged, and so is every path when the event carries no absolute
    working directory.

    Args:
        file_path: The path as the tool call or the shell command spelled it.
        input_data: Hook event input data (JSON from stdin).

    Returns:
        The path the file names from the event's working directory.
    """
    if not file_path or file_path.startswith(('/', '\\', '~')) or os.path.splitdrive(file_path)[0]:
        return file_path
    cwd = input_data.get('cwd')
    if isinstance(cwd, str) and _fully_absolute(cwd):
        return os.path.join(cwd, file_path)
    return file_path


class TemporaryPathJudge(Protocol):
    """Decides whether a path a hook event names is a loose scratch file.

    The judgment needs the event itself, because a relative path is resolved
    against the event's working directory and the session's own scratch
    location is reported in the event.
    """

    def __call__(self, path: str, event: dict[str, Any], *, shell: bool = False) -> bool:
        """Report whether the path is a loose scratch file in a temporary location.

        Args:
            path: The path as the tool call or the shell command spelled it.
            event: The hook event input data.
            shell: True when the path was taken from a shell command.

        Returns:
            True only for a loose scratch file.
        """
        ...


def path_is_skipped(
    config: dict[str, Any],
    file_path: str,
    input_data: dict[str, Any],
    *,
    is_temporary: TemporaryPathJudge | None = None,
    shell: bool = False,
) -> bool:
    """Report whether a hook's configuration tells it to skip a file.

    Two optional keys decide it. 'exclude_paths' lists path globs (see
    _path_is_excluded), matched against the location the path names: a relative
    path, which is how a shell command usually spells its target, is taken from
    the event's working directory, so a file is excluded or not the same way
    whichever tool writes it. 'exclude_temp_paths: true' also skips a loose
    scratch file in the system's temporary locations; what counts as one is
    decided by the is_temporary judge the calling hook supplies, and without a
    judge nothing counts, so the key alone changes nothing. A hook whose
    configuration carries neither key skips nothing.

    Args:
        config: Hook configuration dictionary.
        file_path: The path as the tool call or the shell command spelled it.
        input_data: Hook event input data (JSON from stdin).
        is_temporary: Judge for loose scratch files, or None.
        shell: True when the path was taken from a shell command.

    Returns:
        True when the configuration excludes the file.
    """
    exclude_paths = cast(list[str], config.get('exclude_paths') or [])
    if exclude_paths and _path_is_excluded(_anchored(file_path, input_data), exclude_paths):
        return True
    if config.get('exclude_temp_paths') is not True or is_temporary is None:
        return False
    # Skipping is the relaxation, so a judge that fails renders "not skipped" and
    # the hook applies, rather than raising and leaving the hook with no verdict.
    try:
        return is_temporary(file_path, input_data, shell=shell) is True
    except Exception:
        return False


def check_file_relevance(
    config: dict[str, Any],
    input_data: dict[str, Any],
    *,
    is_temporary: TemporaryPathJudge | None = None,
) -> tuple[bool, str | None]:
    """Check if the edited file matches the hook's target file extensions.

    Reads 'file_extensions' from config and checks against the file path
    from the hook input data. When the config supplies an 'exclude_paths' list,
    a file under any of those globs is treated as not relevant regardless of its
    extension -- this lets hooks skip Claude-internal ephemeral directories (for
    example .workflows/ Workflow-tool scripts or .claude/ internals) that are
    tool-generated rather than project source. A file inside a Claude Code
    worktree (.claude/worktrees/<name>/) is matched as the file it mirrors in the
    main checkout, so a .claude/ exclusion does not swallow the worktree itself.
    'exclude_paths' defaults to empty, so hooks that omit it are unaffected.
    'exclude_temp_paths' skips loose scratch files as path_is_skipped describes.

    Args:
        config: Hook configuration dictionary (should contain 'file_extensions';
            may contain an optional 'exclude_paths' list of path globs and an
            optional 'exclude_temp_paths' flag).
        input_data: Hook event input data (JSON from stdin).
        is_temporary: Judge for loose scratch files, or None.

    Returns:
        Tuple of (is_relevant, file_path).
        is_relevant is True if file matches and is not excluded, False otherwise.
        file_path is the extracted path, or None if not found.
    """
    file_extensions = config.get('file_extensions')
    if not file_extensions:
        return True, None

    # tool_input and tool_response are dicts for built-in tools, but the wire
    # format is not guaranteed (some tools report a plain-string tool_response),
    # so non-dict values are treated as carrying no file path. The path itself is
    # required to be a string for the same reason: a number or a list reaches
    # Path() below and raises TypeError, which would leave the calling hook with
    # no verdict at all.
    tool_input = input_data.get('tool_input')
    raw_path: Any = None
    if isinstance(tool_input, dict):
        raw_path = cast(dict[str, Any], tool_input).get('file_path')
    if not raw_path:
        tool_response = input_data.get('tool_response')
        if isinstance(tool_response, dict):
            raw_path = cast(dict[str, Any], tool_response).get('filePath')

    if not isinstance(raw_path, str) or not raw_path:
        return False, None
    file_path: str = raw_path

    if path_is_skipped(config, file_path, input_data, is_temporary=is_temporary):
        return False, file_path

    ext = Path(file_path).suffix.lower()
    return ext in file_extensions, file_path
