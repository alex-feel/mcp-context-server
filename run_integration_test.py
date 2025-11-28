#!/usr/bin/env python
"""Run integration tests for MCP Context Storage Server.

This script runs the full integration test suite against a real MCP server.
The server subprocess (run_server.py) automatically enables test mode and
semantic search when detecting test context.
"""

from __future__ import annotations

import asyncio
import sys

from tests.test_real_server import MCPServerIntegrationTest


async def main() -> None:
    """Run the integration test suite."""
    print('[INFO] Running integration tests')
    print('[INFO] Note: Server wrapper handles temp DB and semantic search enablement')

    test = MCPServerIntegrationTest()
    try:
        success = await test.run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'Integration test failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
