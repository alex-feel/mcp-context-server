#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pyyaml"]
# ///
"""
Git worktree context detection hook for AEGIS.

Provides canonical project name from git remote URL and worktree metadata
to enable proper context isolation across parallel worktree sessions.

Fallback chain for project name:
1. Parse repo name from git remote URL (origin -> upstream -> first available)
2. Basename of git toplevel directory
3. Current directory basename

Event: SessionStart
Type: command
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_config_loader() -> ModuleType:
    """Dynamically load hook_config_loader from the same directory."""
    loader_path = Path(__file__).parent / 'hook_config_loader.py'
    spec = importlib.util.spec_from_file_location('hook_config_loader', loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load hook_config_loader from {loader_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json_output() -> ModuleType:
    """Dynamically load hook_json_output from the same directory."""
    loader_path = Path(__file__).parent / 'hook_json_output.py'
    spec = importlib.util.spec_from_file_location('hook_json_output', loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load hook_json_output from {loader_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Remote URL parsing patterns (priority order)
URL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # GitHub HTTPS: https://github.com/user/repo.git
    ('github_https', re.compile(r'github\.com[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?$')),
    # GitHub SSH: git@github.com:user/repo.git
    ('github_ssh', re.compile(r'github\.com[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?$')),
    # GitLab HTTPS/SSH (supports subgroups)
    ('gitlab', re.compile(r'gitlab\.com[:/]([\w.-]+(?:/[\w.-]+)*)/([\w.-]+?)(?:\.git)?$')),
    # Bitbucket HTTPS/SSH
    ('bitbucket', re.compile(r'bitbucket\.org[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?$')),
    # Azure DevOps HTTPS
    ('azure', re.compile(r'dev\.azure\.com/[\w.-]+/[\w.-]+/_git/([\w.-]+?)(?:\.git)?$')),
    # Generic: any URL ending with /repo.git or /repo
    ('generic', re.compile(r'.*/([^/]+?)(?:\.git)?$')),
]


# Pattern templates for custom hosts (keyed by type name)
# Used to build regex patterns for self-hosted Git platforms
PATTERN_TEMPLATES: dict[str, str] = {
    # GitHub: owner/repo format
    'github': r'{host}[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?$',
    # GitLab: supports nested groups (group/subgroup/.../repo)
    'gitlab': r'{host}[:/]([\w.-]+(?:/[\w.-]+)*)/([\w.-]+?)(?:\.git)?$',
    # Bitbucket: workspace/repo format
    'bitbucket': r'{host}[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?$',
    # Azure DevOps: org/project/_git/repo format
    'azure': r'{host}/[\w.-]+/[\w.-]+/_git/([\w.-]+?)(?:\.git)?$',
    # Generic: just extract last path component
    'generic': r'.*{host}.*/([^/]+?)(?:\.git)?$',
}


# Default configuration - used when no config file provided
DEFAULT_CONFIG: dict[str, Any] = {
    'enabled': True,
    'remote_priority': ['origin', 'upstream'],
    'output_template': (
        'WORKTREE CONTEXT:\n'
        '- Canonical Project: {project}\n'
        '- Worktree: {worktree_id}\n'
        '- Is Linked Worktree: {is_linked}\n'
        '- Worktree Path: {worktree_path}\n\n'
        "Use canonical project name '{project}' for all context metadata.\n"
        "Worktree identifier '{worktree_id}' provides disambiguation."
    ),
    'include_branch': True,
    'branch_template': '- Current Branch: {branch}',
    # Custom repository hosts (empty by default for backward compatibility)
    'custom_hosts': [],
}


def build_custom_patterns(custom_hosts: list[dict[str, str]]) -> list[tuple[str, re.Pattern[str]]]:
    """Build URL patterns from custom host configuration.

    Each custom host entry specifies a hostname and a pattern type.
    The hostname is escaped for regex and substituted into the pattern template.

    Args:
        custom_hosts: List of dicts with 'host' and 'type' keys.

    Returns:
        List of (name, pattern) tuples for use in URL matching.
    """
    patterns: list[tuple[str, re.Pattern[str]]] = []

    for entry in custom_hosts:
        host = entry.get('host', '')
        pattern_type = entry.get('type', 'generic')

        if not host:
            continue

        # Get pattern template (default to generic if unknown type)
        template = PATTERN_TEMPLATES.get(pattern_type, PATTERN_TEMPLATES['generic'])

        # Escape dots in hostname for regex
        escaped_host = re.escape(host)

        # Build pattern by substituting escaped hostname
        pattern_str = template.format(host=escaped_host)

        try:
            compiled = re.compile(pattern_str)
            pattern_name = f'custom_{host.replace(".", "_")}_{pattern_type}'
            patterns.append((pattern_name, compiled))
        except re.error:
            # Skip invalid patterns silently (graceful degradation)
            continue

    return patterns


def parse_repo_name_from_url(
    url: str,
    custom_patterns: list[tuple[str, re.Pattern[str]]] | None = None,
) -> str | None:
    """Parse repository name from git remote URL.

    Supports HTTPS, SSH, and various hosting providers including
    GitHub, GitLab (with subgroups), Bitbucket, and Azure DevOps.
    Custom patterns are checked FIRST (higher priority).

    Args:
        url: Git remote URL to parse.
        custom_patterns: Optional list of custom (name, pattern) tuples.

    Returns:
        Repository name if pattern matched, None otherwise.
    """
    url = url.strip()

    # Build combined pattern list: custom patterns first, then built-in
    all_patterns = (custom_patterns or []) + URL_PATTERNS

    for _provider, pattern in all_patterns:
        match = pattern.search(url)
        if match:
            # Return last capture group (repo name)
            return match.group(match.lastindex) if match.lastindex else None
    return None


def run_git_command(args: list[str], cwd: str) -> str | None:
    """Run git command and return output, or None on failure.

    Args:
        args: Git command arguments (without 'git' prefix).
        cwd: Working directory for command execution.

    Returns:
        Stripped stdout if successful, None otherwise.
    """
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.SubprocessError, OSError):
        return None


def get_canonical_project_name(cwd: str, config: dict[str, Any]) -> str:
    """Get canonical project name using fallback chain.

    Priority:
    1. Parse repo name from git remote URL (origin -> upstream -> first)
    2. Basename of git toplevel directory
    3. Current directory basename

    Args:
        cwd: Current working directory path.
        config: Hook configuration dictionary.

    Returns:
        Canonical project name.
    """
    # Build custom patterns from config (checked before built-in patterns)
    custom_hosts = config.get('custom_hosts', [])
    custom_patterns = build_custom_patterns(custom_hosts) if custom_hosts else None

    # Try configured remotes in order
    remote_names = config.get('remote_priority', DEFAULT_CONFIG['remote_priority'])

    for remote_name in remote_names:
        url = run_git_command(['remote', 'get-url', remote_name], cwd)
        if url:
            repo_name = parse_repo_name_from_url(url, custom_patterns)
            if repo_name:
                return repo_name

    # Fallback: Try first available remote not in priority list
    remotes_output = run_git_command(['remote'], cwd)
    if remotes_output:
        for remote_line in remotes_output.split('\n'):
            first_remote = remote_line.strip()
            if first_remote and first_remote not in remote_names:
                url = run_git_command(['remote', 'get-url', first_remote], cwd)
                if url:
                    repo_name = parse_repo_name_from_url(url, custom_patterns)
                    if repo_name:
                        return repo_name

    # Fallback: Git toplevel basename
    toplevel = run_git_command(['rev-parse', '--show-toplevel'], cwd)
    if toplevel:
        return Path(toplevel).name

    # Final fallback: Current directory basename
    return Path(cwd).name


def detect_linked_worktree(git_common_dir: str | None, toplevel_path: Path) -> bool:
    """Detect if current worktree is a linked worktree.

    For main worktree: git_common_dir is relative '.git' or absolute path to .git
    For linked worktree: git_common_dir points to main repo's .git directory

    Args:
        git_common_dir: Output of git rev-parse --git-common-dir.
        toplevel_path: Resolved path to worktree root.

    Returns:
        True if linked worktree, False otherwise.
    """
    if not git_common_dir:
        return False

    common_path = Path(git_common_dir)
    if common_path.is_absolute():
        # Check if common dir is NOT the toplevel's .git
        expected_git_dir = toplevel_path / '.git'
        return common_path.resolve() != expected_git_dir.resolve()

    # Relative path - check if it's just '.git'
    return git_common_dir not in ['.git', './.git']


def get_worktree_info(cwd: str, config: dict[str, Any]) -> dict[str, Any]:
    """Get complete worktree information for metadata.

    Args:
        cwd: Current working directory path.
        config: Hook configuration dictionary.

    Returns:
        Dictionary with project, worktree_id, worktree_path, is_linked_worktree,
        and is_git_repo fields.
    """
    # Get git common dir to detect linked worktree
    git_common_dir = run_git_command(['rev-parse', '--git-common-dir'], cwd)
    git_toplevel = run_git_command(['rev-parse', '--show-toplevel'], cwd)

    if not git_toplevel:
        # Not a git repository
        return {
            'project': Path(cwd).name,
            'worktree_id': Path(cwd).name,
            'worktree_path': str(Path(cwd).resolve()),
            'is_linked_worktree': False,
            'is_git_repo': False,
        }

    # Normalize paths for comparison
    toplevel_path = Path(git_toplevel).resolve()

    # Detect linked worktree
    is_linked = detect_linked_worktree(git_common_dir, toplevel_path)

    return {
        'project': get_canonical_project_name(cwd, config),
        'worktree_id': toplevel_path.name,
        'worktree_path': str(toplevel_path),
        'is_linked_worktree': is_linked,
        'is_git_repo': True,
    }


def main() -> None:
    """Main hook execution function."""
    try:
        # Load configuration (defaults merged with config file if provided)
        config_loader = _load_config_loader()
        config = config_loader.get_config_from_argv(DEFAULT_CONFIG)

        # Check if hook is enabled
        if not config.get('enabled', True):
            sys.exit(0)

        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        # Extract and validate event type
        hook_event_name = input_data.get('hook_event_name', '')
        if hook_event_name != 'SessionStart':
            sys.exit(0)

        # Get project directory from environment
        cwd = os.environ.get('CLAUDE_PROJECT_DIR', os.getcwd())

        # Get worktree information
        worktree_info = get_worktree_info(cwd, config)

        # Format output message
        template = config.get('output_template', DEFAULT_CONFIG['output_template'])
        message = template.format(
            project=worktree_info['project'],
            worktree_id=worktree_info['worktree_id'],
            worktree_path=worktree_info['worktree_path'],
            is_linked=str(worktree_info['is_linked_worktree']).lower(),
        )

        # Optionally include branch
        if config.get('include_branch', True) and worktree_info.get('is_git_repo'):
            branch = run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'], cwd)
            if branch:
                branch_template = config.get('branch_template', DEFAULT_CONFIG['branch_template'])
                message += '\n' + branch_template.format(branch=branch)

        try:
            json_output = _load_json_output()
            json_output.emit_additional_context('SessionStart', message)
        except ImportError:
            print(message)
        sys.exit(0)

    except json.JSONDecodeError:
        # Graceful degradation - don't block session start
        sys.exit(0)

    except Exception:
        # Graceful degradation - don't block session start
        sys.exit(0)


if __name__ == '__main__':
    main()
