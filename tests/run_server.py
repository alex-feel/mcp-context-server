#!/usr/bin/env python
"""Wrapper to run the MCP server with proper Python path and environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Force test mode for all test runs
# Check if we're being run from pytest or in a test context
if 'pytest' in sys.modules or any('test' in arg.lower() for arg in sys.argv):
    # We're in a test context - use temporary database
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix='mcp_server_wrapper_')
    test_db = Path(temp_dir) / 'test_wrapper.db'

    os.environ['DB_PATH'] = str(test_db)
    os.environ['MCP_TEST_MODE'] = '1'

    print(f'[TEST SERVER] FORCED test mode with DB_PATH={test_db}', file=sys.stderr)

    # Double-check we're not using the default database
    default_db = Path.home() / '.mcp' / 'context_storage.db'
    if test_db.resolve() == default_db.resolve():
        raise RuntimeError(
            f'CRITICAL: Test server attempting to use default database!\n'
            f'Default: {default_db}\n'
            f'DB_PATH: {test_db}',
        )
else:
    # Normal mode - check environment
    if os.environ.get('MCP_TEST_MODE') == '1':
        db_path = os.environ.get('DB_PATH')
        if db_path:
            print(f'[TEST SERVER] Running in test mode with DB_PATH={db_path}', file=sys.stderr)

            # Double-check we're not using the default database
            default_db = Path.home() / '.mcp' / 'context_storage.db'
            if Path(db_path).resolve() == default_db.resolve():
                raise RuntimeError(
                    f'CRITICAL: Test server attempting to use default database!\n'
                    f'Default: {default_db}\n'
                    f'DB_PATH: {db_path}',
                )
        else:
            print('[TEST SERVER] WARNING: MCP_TEST_MODE=1 but DB_PATH not set!', file=sys.stderr)
    else:
        print('[TEST SERVER] Running in normal mode', file=sys.stderr)

# Now import and run the server
if __name__ == '__main__':
    from app.server import main

    # Run the server's main function
    # The server will use DB_PATH from environment via settings.py
    main()
