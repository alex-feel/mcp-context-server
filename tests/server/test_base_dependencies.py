"""The server starts from a base install, with no extra and no dev group.

``uvx mcp-context-server`` installs only the project's base dependencies, so
every third-party module that ``import app.server`` needs must be reachable
from ``[project].dependencies``. A module that arrives only through an optional
extra, a dev group, or a dependency the base set no longer pulls in crashes that
install with ModuleNotFoundError at startup. The check runs the import in an
isolated environment built from ``uv.lock`` with exactly the base set.
"""

import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_server_imports_from_a_base_install() -> None:
    """app.server imports in an environment holding only the base dependencies."""
    uv = shutil.which('uv')
    assert uv is not None, 'uv must be on PATH to build the base-install environment'
    env = {key: value for key, value in os.environ.items() if key != 'VIRTUAL_ENV'}

    completed = subprocess.run(
        [uv, 'run', '--isolated', '--no-dev', '--frozen', 'python', '-c', 'import app.server'],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert completed.returncode == 0, (
        'import app.server failed in a base-only install; declare the missing '
        f'distribution in [project].dependencies:\n{completed.stderr[-2000:]}'
    )
