#!/usr/bin/env python
"""Run integration tests for MCP Context Storage Server."""

from __future__ import annotations

import asyncio
import sys

from tests.test_real_server import MCPServerIntegrationTest


async def main():
    """Run the integration test suite."""
    test = MCPServerIntegrationTest()
    try:
        success = await test.run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'Integration test failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
