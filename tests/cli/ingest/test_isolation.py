"""The ingest command is a pure MCP client: it never loads the server's settings, storage or tools.

Loading any of them would read the operator's shell configuration, such as a
storage backend and database credentials exported for a server, so the command's
import graph is checked in a fresh interpreter whose environment carries none of
the test process's variables.
"""

import json
import subprocess
import sys
from pathlib import Path

from mcp.client.stdio import get_default_environment

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN = ('app.settings', 'app.backends', 'app.repositories', 'app.tools', 'app.server')
PROBE = f'''
import json
import sys

import app.cli.ingest.cli

forbidden = {FORBIDDEN!r}
loaded = [name for name in sys.modules if any(name == root or name.startswith(root + '.') for root in forbidden)]
print(json.dumps(sorted(loaded)))
'''


def test_the_command_imports_no_server_module() -> None:
    """Importing the command loads none of the server's settings, backends, repositories, tools or entry point."""
    completed = subprocess.run(
        [sys.executable, '-c', PROBE],
        env=get_default_environment(),
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode('utf-8', errors='replace')
    assert json.loads(completed.stdout) == []
