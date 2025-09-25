"""Integration tests for the real running MCP Context Storage Server.

Tests the actual server running via subprocess with uvx command,
verifying all 6 tools work correctly via FastMCP client.
"""

import asyncio
import base64
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client


class MCPServerIntegrationTest:
    """Integration test for real MCP Context Storage Server."""

    def __init__(self, temp_db_path: Path | None = None) -> None:
        """Initialize the integration test suite.

        Args:
            temp_db_path: Optional path to temporary database.
        """
        self.client: Client[Any] | None = None
        self.test_results: list[tuple[str, bool, str]] = []
        self.test_thread_id = f'integration_test_{int(time.time())}'
        self.temp_db_path = temp_db_path
        self.original_env: dict[str, str | None] = {}

    async def start_server(self) -> bool:
        """Start the MCP server via FastMCP Client.

        Returns:
            bool: True (server starts automatically with Client).
        """
        print('[OK] Server will be started by FastMCP Client')
        return True

    async def connect_client(self) -> bool:
        """Connect FastMCP client to server.

        Returns:
            bool: True if client connected successfully.

        Raises:
            RuntimeError: If attempting to use default database in test mode.
        """
        try:
            # Use the wrapper script that sets up Python path correctly
            wrapper_script = Path(__file__).parent / 'run_server.py'
            print(f'[INFO] Connecting to server via wrapper: {wrapper_script}')

            # Environment variables MUST be set BEFORE creating Client
            # The Client spawns a subprocess that inherits the current environment
            if self.temp_db_path:
                # Store original env to restore later
                self.original_env['DB_PATH'] = os.environ.get('DB_PATH')
                self.original_env['MCP_TEST_MODE'] = os.environ.get('MCP_TEST_MODE')

                # Set BOTH DB_PATH and MCP_TEST_MODE
                # These MUST be set before Client() is called
                os.environ['DB_PATH'] = str(self.temp_db_path)
                os.environ['MCP_TEST_MODE'] = '1'  # THIS IS CRITICAL!

                print('[INFO] Environment set BEFORE Client creation:')
                print(f'[INFO] DB_PATH={os.environ.get("DB_PATH")}')
                print(f'[INFO] MCP_TEST_MODE={os.environ.get("MCP_TEST_MODE")}')
                print(f'[INFO] Using temporary database: {self.temp_db_path}')

                # Verify it's not the default database
                default_db = Path.home() / '.mcp' / 'context_storage.db'
                if self.temp_db_path.resolve() == default_db.resolve():
                    raise RuntimeError(
                        f'CRITICAL: Attempting to use default database in test!\n'
                        f'Default: {default_db}\n'
                        f'Current: {self.temp_db_path}',
                    )

                # Initialize the database schema
                self._initialize_database()

            # Create client with wrapper script
            # The wrapper will detect pytest and force test mode with temp DB
            self.client = Client(str(wrapper_script))
            print(f'[INFO] Client created with wrapper: {wrapper_script}')

            # Connect to server
            await self.client.__aenter__()

            # Verify connection by pinging
            await self.client.ping()

            print('[OK] Client connected successfully')
            return True

        except Exception as e:
            print(f'[ERROR] Failed to connect client: {e}')
            import traceback

            traceback.print_exc()
            return False

    def _initialize_database(self) -> None:
        """Initialize the temporary database with schema."""
        if not self.temp_db_path:
            return

        try:
            # Ensure parent directory exists
            self.temp_db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create database and apply schema
            schema_path = Path(__file__).parent.parent / 'app' / 'schema.sql'
            with sqlite3.connect(str(self.temp_db_path)) as conn:
                # Read and execute schema
                schema_sql = schema_path.read_text(encoding='utf-8')
                conn.executescript(schema_sql)

                # Apply optimizations
                conn.execute('PRAGMA foreign_keys = ON')
                conn.execute('PRAGMA journal_mode = WAL')
                conn.execute('PRAGMA synchronous = NORMAL')
                conn.execute('PRAGMA temp_store = MEMORY')
                conn.execute('PRAGMA busy_timeout = 30000')
                conn.commit()

            print(f'[OK] Database schema initialized at {self.temp_db_path}')
        except Exception as e:
            print(f'[WARNING] Failed to initialize database: {e}')
            # Continue anyway - the server will initialize on startup

    def _create_test_image(self) -> str:
        """Create a small test image as base64.

        Returns:
            str: Base64 encoded test image.
        """
        # Create a simple 1x1 pixel PNG image
        png_header = b'\x89PNG\r\n\x1a\n'
        ihdr = b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
        idat = b'\x00\x00\x00\x0bIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05W\xbf\xaa\xd4'
        iend = b'\x00\x00\x00\x00IEND\xaeB`\x82'
        png_data = png_header + ihdr + idat + iend
        return base64.b64encode(png_data).decode('utf-8')

    def _extract_content(self, result: object) -> dict[str, Any]:
        """Extract content from FastMCP CallToolResult.

        Args:
            result: CallToolResult object from FastMCP.

        Returns:
            dict: The actual result content.
        """
        # FastMCP CallToolResult has structured_content attribute
        content = getattr(result, 'structured_content', None)
        if content is not None:
            if isinstance(content, dict):
                # Handle wrapped results
                if 'result' in content:
                    if isinstance(content['result'], list):
                        return {'success': True, 'results': content['result']}
                    if isinstance(content['result'], dict):
                        return content['result']
                # Special handling for list_threads - it returns threads directly
                if 'threads' in content:
                    return {'success': True, 'threads': content['threads'], 'total_threads': content.get('total_threads', 0)}
                # Special handling for get_statistics - it returns stats directly
                if 'total_entries' in content:
                    return {'success': True, **content}  # Include all statistics fields
                # Direct dict results
                return content
            # List results
            if isinstance(content, list):
                return {'success': True, 'results': content}

        # Should not reach here with current FastMCP, but return error for safety
        return {'success': False, 'error': 'Unable to extract content from result'}

    async def test_store_context(self) -> bool:
        """Test storing text and multimodal context.

        Returns:
            bool: True if test passed.
        """
        test_name = 'store_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Test text storage
            text_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': self.test_thread_id,
                    'source': 'agent',  # Must be 'user' or 'agent'
                    'text': 'This is a test message for integration testing',
                    'metadata': {'test': True, 'timestamp': time.time()},
                    'tags': ['test', 'integration'],
                },
            )

            text_data = self._extract_content(text_result)
            print(f'DEBUG store text_data: {text_data}')  # Debug output

            # store_context returns a dict with success and nested results
            if not text_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store text context: {text_data}'))
                return False

            # Extract context_id directly from response
            text_context_id = text_data.get('context_id')
            print(f'DEBUG text_context_id: {text_context_id}')  # Debug output

            # Test image storage
            image_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': self.test_thread_id,
                    'source': 'user',  # Must be 'user' or 'agent'
                    'text': 'Test message with image',
                    'images': [
                        {
                            'data': self._create_test_image(),
                            'mime_type': 'image/png',
                        },
                    ],
                    'tags': ['test', 'image'],
                },
            )

            image_data = self._extract_content(image_result)
            print(f'DEBUG store image_data: {image_data}')  # Debug output

            # store_context returns a dict with success and nested results
            if not image_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store image context: {image_data}'))
                return False

            # Extract context_id directly from response
            image_context_id = image_data.get('context_id')
            print(f'DEBUG image_context_id: {image_context_id}')  # Debug output

            # Verify both contexts were stored
            if text_context_id and image_context_id:
                self.test_results.append((test_name, True, f'Stored contexts: {text_context_id}, {image_context_id}'))
                return True
            self.test_results.append((test_name, False, 'Missing context IDs'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_context(self) -> bool:
        """Test searching with various filters.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # First store some test data
            await self.client.call_tool(
                'store_context',
                {
                    'thread_id': self.test_thread_id,
                    'source': 'user',  # Must be 'user' or 'agent'
                    'text': 'Message for search testing',
                    'tags': ['searchable', 'test'],
                },
            )

            # Test search by thread
            thread_results = await self.client.call_tool(
                'search_context',
                {'thread_id': self.test_thread_id},
            )

            thread_data = self._extract_content(thread_results)

            # search_context returns success with results
            if not thread_data.get('success'):
                self.test_results.append((test_name, False, f'Thread search failed: {thread_data}'))
                return False

            # Test search by source
            source_results = await self.client.call_tool(
                'search_context',
                {'source': 'user'},
            )

            source_data = self._extract_content(source_results)

            # search_context returns success with results
            if not source_data.get('success'):
                self.test_results.append((test_name, False, f'Source search failed: {source_data}'))
                return False

            # Test search by tags
            tag_results = await self.client.call_tool(
                'search_context',
                {'tags': ['searchable']},
            )

            tag_data = self._extract_content(tag_results)

            # search_context returns success with results
            if not tag_data.get('success'):
                self.test_results.append((test_name, False, f'Tag search failed: {tag_data}'))
                return False

            # Test pagination
            paginated_results = await self.client.call_tool(
                'search_context',
                {
                    'thread_id': self.test_thread_id,
                    'limit': 1,
                    'offset': 0,
                },
            )

            paginated_data = self._extract_content(paginated_results)

            # search_context returns success with results
            if not paginated_data.get('success'):
                self.test_results.append((test_name, False, f'Pagination failed: {paginated_data}'))
                return False

            # Verify all searches returned results
            all_have_results = all([
                len(thread_data.get('results', [])) > 0,
                len(source_data.get('results', [])) > 0,
                len(tag_data.get('results', [])) > 0,
                len(paginated_data.get('results', [])) > 0,
            ])

            if all_have_results:
                self.test_results.append((test_name, True, 'All search filters working'))
                return True
            self.test_results.append((test_name, False, 'Some searches returned no results'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_get_context_by_ids(self) -> bool:
        """Test retrieving specific contexts by IDs.

        Returns:
            bool: True if test passed.
        """
        test_name = 'get_context_by_ids'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Store test data
            result1 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': self.test_thread_id,
                    'source': 'agent',  # Must be 'user' or 'agent'
                    'text': 'First context for retrieval',
                },
            )

            result2 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': self.test_thread_id,
                    'source': 'user',  # Must be 'user' or 'agent'
                    'text': 'Second context with image',
                    'images': [
                        {
                            'data': self._create_test_image(),
                            'mime_type': 'image/png',
                        },
                    ],
                },
            )

            data1 = self._extract_content(result1)
            data2 = self._extract_content(result2)

            if not (data1.get('success') and data2.get('success')):
                self.test_results.append((test_name, False, f'Failed to store test contexts: {data1}, {data2}'))
                return False

            context_ids = [data1['context_id'], data2['context_id']]

            # Test retrieval without images
            without_images = await self.client.call_tool(
                'get_context_by_ids',
                {
                    'context_ids': context_ids,
                    'include_images': False,
                },
            )

            without_data = self._extract_content(without_images)

            # get_context_by_ids returns success with results
            if not without_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to retrieve without images: {without_data}'))
                return False

            # Test retrieval with images
            with_images = await self.client.call_tool(
                'get_context_by_ids',
                {
                    'context_ids': context_ids,
                    'include_images': True,
                },
            )

            with_data = self._extract_content(with_images)

            # get_context_by_ids returns success with results
            if not with_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to retrieve with images: {with_data}'))
                return False

            # Verify both retrievals got the correct number of results
            if len(without_data.get('results', [])) == 2 and len(with_data.get('results', [])) == 2:
                self.test_results.append((test_name, True, f'Retrieved {len(context_ids)} contexts'))
                return True
            self.test_results.append((test_name, False, 'Incorrect number of results'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_delete_context(self) -> bool:
        """Test deletion operations.

        Returns:
            bool: True if test passed.
        """
        test_name = 'delete_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create a separate thread for deletion tests
            delete_thread = f'{self.test_thread_id}_delete'

            # Store multiple contexts
            result1 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': delete_thread,
                    'source': 'user',  # Must be 'user' or 'agent'
                    'text': 'Context to delete by ID',
                },
            )

            result2 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': delete_thread,
                    'source': 'agent',  # Must be 'user' or 'agent'
                    'text': 'Context to delete with thread',
                },
            )

            result3 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': delete_thread,
                    'source': 'user',  # Must be 'user' or 'agent'
                    'text': 'Another context in thread',
                },
            )

            data1 = self._extract_content(result1)
            data2 = self._extract_content(result2)
            data3 = self._extract_content(result3)

            if not all([
                data1.get('success'),
                data2.get('success'),
                data3.get('success'),
            ]):
                self.test_results.append((test_name, False, f'Failed to store test contexts: {data1}, {data2}, {data3}'))
                return False

            # Test delete by ID
            delete_by_id = await self.client.call_tool(
                'delete_context',
                {'context_ids': [data1['context_id']]},
            )

            delete_data = self._extract_content(delete_by_id)

            if not delete_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to delete by ID: {delete_data}'))
                return False

            # Verify deletion by trying to retrieve
            check_deleted = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [data1['context_id']]},
            )

            check_data = self._extract_content(check_deleted)

            # get_context_by_ids returns success with results
            if len(check_data.get('results', [])) > 0:
                self.test_results.append((test_name, False, f'Context not deleted by ID: {check_data}'))
                return False

            # Test delete by thread
            delete_by_thread = await self.client.call_tool(
                'delete_context',
                {'thread_id': delete_thread},
            )

            thread_delete_data = self._extract_content(delete_by_thread)

            if not thread_delete_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to delete by thread: {thread_delete_data}'))
                return False

            # Verify thread deletion
            check_thread = await self.client.call_tool(
                'search_context',
                {'thread_id': delete_thread},
            )

            check_thread_data = self._extract_content(check_thread)

            # search_context returns success with results
            if len(check_thread_data.get('results', [])) > 0:
                self.test_results.append((test_name, False, f'Thread contexts not deleted: {check_thread_data}'))
                return False

            deleted_count = delete_data.get('deleted_count', 0) + thread_delete_data.get('deleted_count', 0)
            self.test_results.append((test_name, True, f'Deleted {deleted_count} contexts'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_list_threads(self) -> bool:
        """Test thread listing resource.

        Returns:
            bool: True if test passed.
        """
        test_name = 'list_threads'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create multiple threads with contexts
            threads = [
                f'{self.test_thread_id}_list_1',
                f'{self.test_thread_id}_list_2',
                f'{self.test_thread_id}_list_3',
            ]

            for thread in threads:
                # Store multiple contexts per thread
                for i in range(3):
                    result = await self.client.call_tool(
                        'store_context',
                        {
                            'thread_id': thread,
                            'source': 'agent' if i % 2 == 0 else 'user',  # Alternate sources
                            'text': f'Message {i} in {thread}',
                        },
                    )
                    data = self._extract_content(result)
                    if not data.get('success'):
                        self.test_results.append((test_name, False, f'Failed to store context for {thread}: {data}'))
                        return False

            # List threads
            thread_list = await self.client.call_tool('list_threads', {})

            list_data = self._extract_content(thread_list)

            # list_threads returns a dict with threads array (no success flag needed)
            if 'threads' not in list_data:
                self.test_results.append((test_name, False, f'Failed to list threads: {list_data}'))
                return False

            # Verify threads are in the list
            listed_threads = list_data['threads']
            thread_ids = [t['thread_id'] for t in listed_threads]

            all_present = all(thread in thread_ids for thread in threads)

            if all_present:
                # Check that threads have correct statistics
                for thread_info in listed_threads:
                    if thread_info['thread_id'] in threads and thread_info.get('entry_count', 0) != 3:
                        error_msg = (
                            f"Thread {thread_info['thread_id']} has wrong count: "
                            f"{thread_info.get('entry_count', 0)}"
                        )
                        self.test_results.append((test_name, False, error_msg))
                        return False

                self.test_results.append((test_name, True, f'Listed {len(threads)} test threads with correct counts'))
                return True
            self.test_results.append((test_name, False, 'Not all threads present in list'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_get_statistics(self) -> bool:
        """Test statistics resource.

        Returns:
            bool: True if test passed.
        """
        test_name = 'get_statistics'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Get current statistics
            stats = await self.client.call_tool('get_statistics', {})

            stats_data = self._extract_content(stats)

            # Check if we have the statistics fields (no success field needed)
            if 'total_entries' not in stats_data:
                self.test_results.append((test_name, False, f'Failed to get statistics: {stats_data}'))
                return False

            # Store a new context
            result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': f'{self.test_thread_id}_stats',
                    'source': 'user',  # Must be 'user' or 'agent'
                    'text': 'Context for statistics test',
                    'images': [
                        {
                            'data': self._create_test_image(),
                            'mime_type': 'image/png',
                        },
                    ],
                },
            )

            result_data = self._extract_content(result)

            if not result_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to store test context'))
                return False

            # Get updated statistics
            new_stats = await self.client.call_tool('get_statistics', {})

            new_stats_data = self._extract_content(new_stats)

            # Check if we have the statistics fields (no success field needed)
            if 'total_entries' not in new_stats_data:
                self.test_results.append((test_name, False, f'Failed to get updated statistics: {new_stats_data}'))
                return False

            # Verify statistics increased
            old_count = stats_data.get('total_entries', 0)
            new_count = new_stats_data.get('total_entries', 0)
            old_images = stats_data.get('total_images', 0)
            new_images = new_stats_data.get('total_images', 0)

            if new_count > old_count and new_images > old_images:
                self.test_results.append(
                    (test_name, True, f'Stats updated: entries {old_count}->{new_count}, images {old_images}->{new_images}'),
                )
                return True
            self.test_results.append((test_name, False, 'Statistics not updated correctly'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def cleanup(self) -> None:
        """Clean up server and resources."""
        try:
            # Disconnect client (this also stops the server subprocess)
            if self.client:
                await self.client.__aexit__(None, None, None)
                print('[OK] Client disconnected and server stopped')

            # Restore original environment variables
            for key, value in self.original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

            # Clean up temporary database file if it exists
            if self.temp_db_path and self.temp_db_path.exists():
                try:
                    # Remove WAL and SHM files if they exist
                    wal_file = Path(str(self.temp_db_path) + '-wal')
                    shm_file = Path(str(self.temp_db_path) + '-shm')
                    if wal_file.exists():
                        wal_file.unlink()
                    if shm_file.exists():
                        shm_file.unlink()
                    # Remove main database file
                    self.temp_db_path.unlink()
                    print(f'[OK] Temporary database cleaned up: {self.temp_db_path}')
                except Exception as cleanup_err:
                    print(f'[WARNING] Could not clean up temp database: {cleanup_err}')

        except Exception as e:
            print(f'[WARNING] Cleanup error: {e}')

    async def run_all_tests(self) -> bool:
        """Run all tests and report results.

        Returns:
            bool: True if all tests passed.
        """
        print('\n' + '=' * 50)
        print('MCP SERVER INTEGRATION TEST')
        print('=' * 50)

        # Start server
        if not await self.start_server():
            print('[ERROR] Failed to start server')
            await self.cleanup()
            return False

        # Connect client
        if not await self.connect_client():
            print('[ERROR] Failed to connect client')
            await self.cleanup()
            return False

        # Run all tests
        tests = [
            ('Store Context', self.test_store_context),
            ('Search Context', self.test_search_context),
            ('Get Context by IDs', self.test_get_context_by_ids),
            ('Delete Context', self.test_delete_context),
            ('List Threads', self.test_list_threads),
            ('Get Statistics', self.test_get_statistics),
        ]

        print('\nRunning tests...\n')

        for test_name, test_func in tests:
            print(f'Testing: {test_name}...')
            try:
                success = await test_func()
                if success:
                    print(f'  [OK] {test_name} passed')
                else:
                    print(f'  [FAIL] {test_name} failed')
            except Exception as e:
                print(f'  [ERROR] {test_name} error: {e}')
                self.test_results.append((test_name, False, f'Exception: {e}'))

        # Display results
        print('\n' + '=' * 50)
        print('TEST RESULTS')
        print('=' * 50)

        passed = 0
        failed = 0

        for test_name, result, details in self.test_results:
            status = '[PASS]' if result else '[FAIL]'
            print(f'{status}: {test_name}')
            if details:
                print(f'   Details: {details}')
            if result:
                passed += 1
            else:
                failed += 1

        total = passed + failed
        print(f'\nTotal: {passed}/{total} tests passed')

        # Cleanup
        await self.cleanup()

        return failed == 0


# Pytest integration
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_server(tmp_path: Path) -> None:
    """Run integration tests against real server with temporary database.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Raises:
        RuntimeError: If MCP_TEST_MODE is not set or if attempting to use default database.
    """
    # Verify we're in test mode from the global fixture
    if not os.environ.get('MCP_TEST_MODE'):
        raise RuntimeError(
            'MCP_TEST_MODE not set! Global test fixture may have failed.\n'
            'This could lead to pollution of the default database!',
        )

    # Create a unique database path in the temp directory
    temp_db = tmp_path / 'test_real_server.db'

    # Double-check we're not using the default database
    default_db = Path.home() / '.mcp' / 'context_storage.db'
    if temp_db.resolve() == default_db.resolve():
        raise RuntimeError(
            f'Test attempting to use default database!\n'
            f'Default: {default_db}\n'
            f'Test DB: {temp_db}',
        )

    print(f'[TEST] Running with temp database: {temp_db}')
    print(f'[TEST] MCP_TEST_MODE: {os.environ.get("MCP_TEST_MODE")}')

    test = MCPServerIntegrationTest(temp_db_path=temp_db)
    success = await test.run_all_tests()
    assert success, 'Integration tests failed'


if __name__ == '__main__':
    # Allow running directly
    async def main() -> None:
        # Set test mode when running directly
        os.environ['MCP_TEST_MODE'] = '1'

        # Create a temporary directory for the database when running directly
        with tempfile.TemporaryDirectory(prefix='mcp_test_direct_') as tmpdir:
            temp_db_path = Path(tmpdir) / 'test_direct.db'

            # Set DB_PATH for the subprocess
            os.environ['DB_PATH'] = str(temp_db_path)

            print('[INFO] Running directly with test mode enabled')
            print(f'[INFO] Using temporary directory: {tmpdir}')
            print(f'[INFO] DB_PATH set to: {temp_db_path}')
            print(f'[INFO] MCP_TEST_MODE: {os.environ.get("MCP_TEST_MODE")}')

            test = MCPServerIntegrationTest(temp_db_path=temp_db_path)
            success = await test.run_all_tests()
            sys.exit(0 if success else 1)

    asyncio.run(main())
