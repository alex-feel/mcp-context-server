"""Exit-code contract for settings validation failures at server import time.

``app/server.py`` performs the FIRST ``get_settings()`` call in its import
graph, so a pydantic ``ValidationError`` fires at import time -- BEFORE
``main()``'s error-classification handler exists. The module classifies it
inline and exits with EX_CONFIG (78) so supervisors do not restart-loop on a
permanent misconfiguration; the generic exit 1 of an unhandled traceback
keeps a Docker ``restart: on-failure`` policy retrying forever.
"""

import os
import subprocess
import sys
from pathlib import Path


def test_settings_validation_failure_exits_with_ex_config(tmp_path: Path) -> None:
    """POSTGRESQL_POOL_MIN above POSTGRESQL_POOL_MAX exits 78, not 1.

    The cross-field pool-size validator raises inside ``get_settings()``
    during ``import app.server``; the guarded module-level settings load must
    convert that into a clean EX_CONFIG exit with a readable message instead
    of an unhandled-traceback exit 1.
    """
    env = os.environ.copy()
    env['POSTGRESQL_POOL_MIN'] = '5'
    env['POSTGRESQL_POOL_MAX'] = '2'
    env['DB_PATH'] = str(tmp_path / 'unused.db')
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, '-c', 'import app.server'],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 78, (
        f'expected EX_CONFIG (78), got {result.returncode}; stderr:\n{result.stderr}'
    )
    assert 'Configuration invalid' in result.stderr
    assert 'POSTGRESQL_POOL_MIN' in result.stderr
