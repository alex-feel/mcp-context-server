"""Tests for the mcp-context-ingest console script and the module form of the command."""

import subprocess
import sys
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from mcp.client.stdio import get_default_environment

from app.cli.ingest.cli import main

SCRIPT = 'mcp-context-ingest'
TARGET = 'app.cli.ingest.cli:main'
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_the_console_script_resolves_to_main() -> None:
    """The installed entry point names the command's main function and loads it."""
    (script,) = entry_points(group='console_scripts', name=SCRIPT)

    assert script.value == TARGET
    assert script.load() is main


def test_the_project_declares_the_console_script() -> None:
    """pyproject.toml declares the same target the installed metadata carries."""
    project = tomllib.loads((REPOSITORY_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))

    assert project['project']['scripts'][SCRIPT] == TARGET


def test_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    """--help prints the usage under the command's name and exits 0."""
    with pytest.raises(SystemExit) as raised:
        main(['--help'])

    assert raised.value.code == 0
    assert capsys.readouterr().out.startswith(f'usage: {SCRIPT}')


def test_the_module_runs_the_command() -> None:
    """python -m app.cli.ingest.cli runs main, which is how a child process starts the command."""
    completed = subprocess.run(
        [sys.executable, '-m', 'app.cli.ingest.cli', '--help'],
        env=get_default_environment(),
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode('utf-8', errors='replace')
    assert completed.stdout.decode('utf-8', errors='replace').startswith(f'usage: {SCRIPT}')
