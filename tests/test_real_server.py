"""Integration tests for the real running MCP Context Storage Server.

Tests the actual server running via subprocess with uvx command,
verifying all 8 tools work correctly via FastMCP client.
"""

import asyncio
import base64
import os
import sqlite3
import sys
import tempfile
import time
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
from anyio import Path as AsyncPath
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
                self.original_env['ENABLE_SEMANTIC_SEARCH'] = os.environ.get('ENABLE_SEMANTIC_SEARCH')
                self.original_env['ENABLE_FTS'] = os.environ.get('ENABLE_FTS')
                self.original_env['ENABLE_HYBRID_SEARCH'] = os.environ.get('ENABLE_HYBRID_SEARCH')

                # Keep FTS and hybrid search enabled - hybrid search has graceful degradation
                # These MUST be set before Client() is called
                os.environ['DB_PATH'] = str(self.temp_db_path)
                os.environ['MCP_TEST_MODE'] = '1'  # THIS IS CRITICAL!
                os.environ['ENABLE_SEMANTIC_SEARCH'] = 'true'
                os.environ['ENABLE_FTS'] = 'true'
                os.environ['ENABLE_HYBRID_SEARCH'] = 'true'

                print('[INFO] Environment set BEFORE Client creation:')
                print(f"[INFO] DB_PATH={os.environ.get('DB_PATH')}")
                print(f"[INFO] MCP_TEST_MODE={os.environ.get('MCP_TEST_MODE')}")
                print(f"[INFO] ENABLE_SEMANTIC_SEARCH={os.environ.get('ENABLE_SEMANTIC_SEARCH')}")
                print(f"[INFO] ENABLE_FTS={os.environ.get('ENABLE_FTS')}")
                print(f"[INFO] ENABLE_HYBRID_SEARCH={os.environ.get('ENABLE_HYBRID_SEARCH')}")
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
            from app.schemas import load_schema

            schema_sql = load_schema('sqlite')
            with sqlite3.connect(str(self.temp_db_path)) as conn:
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
                # Special handling for search responses - return full content as-is
                # (search_context, semantic_search, fts_search, hybrid_search all return 'results' and 'count')
                if 'results' in content and 'count' in content:
                    # Add success=True if not present, preserve all other fields (error, stats, etc.)
                    if 'success' not in content:
                        return {'success': True, **content}
                    return content
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
                {'limit': 50, 'thread_id': self.test_thread_id},
            )

            thread_data = self._extract_content(thread_results)

            # search_context returns success with results
            if not thread_data.get('success'):
                self.test_results.append((test_name, False, f'Thread search failed: {thread_data}'))
                return False

            # Test search by source
            source_results = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'source': 'user'},
            )

            source_data = self._extract_content(source_results)

            # search_context returns success with results
            if not source_data.get('success'):
                self.test_results.append((test_name, False, f'Source search failed: {source_data}'))
                return False

            # Test search by tags
            tag_results = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'tags': ['searchable']},
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

    async def test_metadata_filtering(self) -> bool:
        """Test advanced metadata filtering functionality.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Metadata Filtering'
        print('Testing metadata filtering...')

        # Store test context entries with various metadata
        test_entries = [
            {
                'thread_id': f'{self.test_thread_id}_metadata',
                'source': 'agent',
                'text': 'High priority task',
                'metadata': {'status': 'active', 'priority': 10, 'agent_name': 'analyzer'},
            },
            {
                'thread_id': f'{self.test_thread_id}_metadata',
                'source': 'agent',
                'text': 'Medium priority task',
                'metadata': {'status': 'active', 'priority': 5, 'agent_name': 'coordinator'},
            },
            {
                'thread_id': f'{self.test_thread_id}_metadata',
                'source': 'agent',
                'text': 'Low priority completed',
                'metadata': {'status': 'completed', 'priority': 1, 'completed': True},
            },
            {
                'thread_id': f'{self.test_thread_id}_metadata',
                'source': 'agent',
                'text': 'Failed task',
                'metadata': {'status': 'failed', 'priority': 8},
            },
        ]

        try:
            # Store all test entries
            assert self.client is not None  # Type guard for Pyright
            for entry in test_entries:
                result = await self.client.call_tool('store_context', entry)
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    print(f'Failed to store test entry: {result_data}')
                    self.test_results.append((test_name, False, 'Failed to store test entries'))
                    return False

            # Test 1: Simple metadata filtering
            result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': f'{self.test_thread_id}_metadata',
                    'metadata': {'status': 'active'},
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 2:
                print(f"Simple filter failed: expected 2, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'Simple metadata filter failed'))
                return False

            # Test 2: Advanced metadata filtering with gte operator
            result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': f'{self.test_thread_id}_metadata',
                    'metadata_filters': [{'key': 'priority', 'operator': 'gte', 'value': 5}],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 3:
                print(f"Advanced gte filter failed: expected 3, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'Advanced gte filter failed'))
                return False

            # Test 3: Combined metadata filters
            result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': f'{self.test_thread_id}_metadata',
                    'metadata': {'status': 'active'},
                    'metadata_filters': [{'key': 'priority', 'operator': 'gt', 'value': 7}],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 1:
                print(f"Combined filter failed: expected 1, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'Combined filter failed'))
                return False

            # Test 4: Exists operator
            result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': f'{self.test_thread_id}_metadata',
                    'metadata_filters': [{'key': 'completed', 'operator': 'exists', 'value': None}],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 1:
                print(f"Exists filter failed: expected 1, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'Exists operator filter failed'))
                return False

            # Test 5: In operator
            result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': f'{self.test_thread_id}_metadata',
                    'metadata_filters': [{'key': 'agent_name', 'operator': 'in', 'value': ['analyzer', 'coordinator']}],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 2:
                print(f"In operator filter failed: expected 2, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'In operator filter failed'))
                return False

            print('[OK] All metadata filtering tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
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
                {'limit': 50, 'thread_id': delete_thread},
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
                        error_msg = f"Thread {thread_info['thread_id']} has wrong count: {thread_info.get('entry_count', 0)}"
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

    async def test_update_context(self) -> bool:
        """Test updating existing context entries.

        Returns:
            bool: True if test passed.
        """
        test_name = 'update_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create a separate thread for update tests
            update_thread = f'{self.test_thread_id}_update'

            # Store initial context
            initial_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': update_thread,
                    'source': 'agent',
                    'text': 'Initial text content',
                    'metadata': {'status': 'draft', 'priority': 1},
                    'tags': ['initial', 'test'],
                },
            )

            initial_data = self._extract_content(initial_result)

            if not initial_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store initial context: {initial_data}'))
                return False

            context_id = initial_data.get('context_id')

            # Test 1: Update text only
            update_text_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'text': 'Updated text content',
                },
            )

            update_text_data = self._extract_content(update_text_result)

            if not update_text_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to update text: {update_text_data}'))
                return False

            # Verify text was updated
            verify_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_data = self._extract_content(verify_result)

            if not verify_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify text update'))
                return False

            updated_entry = verify_data['results'][0]

            if updated_entry.get('text_content') != 'Updated text content':
                self.test_results.append((test_name, False, 'Text not updated correctly'))
                return False

            # Test 2: Update metadata only
            update_metadata_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata': {'status': 'completed', 'priority': 10, 'reviewed': True},
                },
            )

            update_metadata_data = self._extract_content(update_metadata_result)

            if not update_metadata_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to update metadata: {update_metadata_data}'))
                return False

            # Test 3: Update tags (replacement)
            update_tags_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'tags': ['updated', 'final'],
                },
            )

            update_tags_data = self._extract_content(update_tags_result)

            if not update_tags_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to update tags: {update_tags_data}'))
                return False

            # Test 4: Add images (verify content_type changes to multimodal)
            update_images_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'images': [
                        {
                            'data': self._create_test_image(),
                            'mime_type': 'image/png',
                        },
                    ],
                },
            )

            update_images_data = self._extract_content(update_images_result)

            if not update_images_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to add images: {update_images_data}'))
                return False

            # Verify content_type changed to multimodal
            verify_multimodal = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id], 'include_images': True},
            )

            verify_multimodal_data = self._extract_content(verify_multimodal)

            if not verify_multimodal_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify multimodal update'))
                return False

            multimodal_entry = verify_multimodal_data['results'][0]

            if multimodal_entry.get('content_type') != 'multimodal':
                self.test_results.append((test_name, False, 'Content type not changed to multimodal'))
                return False

            # Test 5: Remove images (verify content_type changes back to text)
            remove_images_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'images': [],
                },
            )

            remove_images_data = self._extract_content(remove_images_result)

            if not remove_images_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to remove images: {remove_images_data}'))
                return False

            # Verify content_type changed back to text
            verify_text_type = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_text_type_data = self._extract_content(verify_text_type)

            if not verify_text_type_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify text-only update'))
                return False

            text_only_entry = verify_text_type_data['results'][0]

            if text_only_entry.get('content_type') != 'text':
                self.test_results.append((test_name, False, 'Content type not changed back to text'))
                return False

            # Test 6: Metadata patch - add new field to existing metadata
            # First restore metadata for patch testing
            restore_metadata_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata': {'status': 'active', 'priority': 5},
                },
            )

            restore_metadata_data = self._extract_content(restore_metadata_result)

            if not restore_metadata_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to restore metadata: {restore_metadata_data}'))
                return False

            # Now patch to add new field
            patch_add_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata_patch': {'new_field': 'added_value'},
                },
            )

            patch_add_data = self._extract_content(patch_add_result)

            if not patch_add_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to patch-add new field: {patch_add_data}'))
                return False

            # Verify patch added new field while preserving existing ones
            verify_patch_add = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_patch_add_data = self._extract_content(verify_patch_add)

            if not verify_patch_add_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify patch-add'))
                return False

            patched_metadata = verify_patch_add_data['results'][0].get('metadata', {})

            # Check that existing fields are preserved and new field was added
            if patched_metadata.get('status') != 'active':
                self.test_results.append((test_name, False, 'Patch-add did not preserve existing status field'))
                return False

            if patched_metadata.get('priority') != 5:
                self.test_results.append((test_name, False, 'Patch-add did not preserve existing priority field'))
                return False

            if patched_metadata.get('new_field') != 'added_value':
                self.test_results.append((test_name, False, 'Patch-add did not add new field'))
                return False

            # Test 7: Metadata patch - update existing field
            patch_update_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata_patch': {'priority': 10},
                },
            )

            patch_update_data = self._extract_content(patch_update_result)

            if not patch_update_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to patch-update existing field: {patch_update_data}'))
                return False

            # Verify patch updated field while preserving others
            verify_patch_update = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_patch_update_data = self._extract_content(verify_patch_update)

            if not verify_patch_update_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify patch-update'))
                return False

            updated_metadata = verify_patch_update_data['results'][0].get('metadata', {})

            if updated_metadata.get('priority') != 10:
                self.test_results.append((test_name, False, 'Patch-update did not change priority'))
                return False

            if updated_metadata.get('status') != 'active':
                self.test_results.append((test_name, False, 'Patch-update did not preserve status field'))
                return False

            if updated_metadata.get('new_field') != 'added_value':
                self.test_results.append((test_name, False, 'Patch-update did not preserve new_field'))
                return False

            # Test 8: Metadata patch - delete field with null value (RFC 7396 semantics)
            patch_delete_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata_patch': {'new_field': None},
                },
            )

            patch_delete_data = self._extract_content(patch_delete_result)

            if not patch_delete_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to patch-delete field: {patch_delete_data}'))
                return False

            # Verify patch deleted the field
            verify_patch_delete = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_patch_delete_data = self._extract_content(verify_patch_delete)

            if not verify_patch_delete_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify patch-delete'))
                return False

            deleted_metadata = verify_patch_delete_data['results'][0].get('metadata', {})

            if 'new_field' in deleted_metadata:
                self.test_results.append((test_name, False, 'Patch-delete did not remove field (RFC 7396 null semantics)'))
                return False

            if deleted_metadata.get('status') != 'active' or deleted_metadata.get('priority') != 10:
                self.test_results.append((test_name, False, 'Patch-delete modified other fields'))
                return False

            # Test 9: Mutual exclusivity - providing both metadata and metadata_patch should fail
            # The server raises ToolError which may propagate as an exception to the client
            mutual_exclusivity_validated = False
            try:
                mutual_exclusion_result = await self.client.call_tool(
                    'update_context',
                    {
                        'context_id': context_id,
                        'metadata': {'full': 'replacement'},
                        'metadata_patch': {'partial': 'update'},
                    },
                )

                mutual_exclusion_data = self._extract_content(mutual_exclusion_result)

                # If we get here without exception, check the response
                if mutual_exclusion_data.get('success'):
                    self.test_results.append(
                        (test_name, False, 'Mutual exclusivity check failed - both metadata and metadata_patch accepted'),
                    )
                    return False

                # Check error message in response
                error_msg = mutual_exclusion_data.get('error', '')
                error_mentions_mutual_exclusivity = 'mutual' in error_msg.lower() or 'exclusive' in error_msg.lower()
                error_mentions_metadata_params = 'metadata' in error_msg.lower() and 'patch' in error_msg.lower()
                if error_mentions_mutual_exclusivity or error_mentions_metadata_params:
                    mutual_exclusivity_validated = True
                else:
                    self.test_results.append(
                        (test_name, False, f'Mutual exclusivity error message unclear: {error_msg}'),
                    )
                    return False

            except Exception as mutual_exc:
                # ToolError is expected - verify the error message mentions mutual exclusivity
                error_msg = str(mutual_exc)
                error_mentions_mutual_exclusivity = 'mutual' in error_msg.lower() or 'exclusive' in error_msg.lower()
                error_mentions_metadata_params = 'metadata' in error_msg.lower() and 'patch' in error_msg.lower()
                if error_mentions_mutual_exclusivity or error_mentions_metadata_params:
                    mutual_exclusivity_validated = True
                else:
                    self.test_results.append(
                        (test_name, False, f'Unexpected exception during mutual exclusivity test: {mutual_exc}'),
                    )
                    return False

            if not mutual_exclusivity_validated:
                self.test_results.append((test_name, False, 'Mutual exclusivity validation did not complete'))
                return False

            # Test 10: Metadata patch on context with no existing metadata
            # Create new context without metadata
            no_metadata_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': update_thread,
                    'source': 'agent',
                    'text': 'Context without initial metadata',
                },
            )

            no_metadata_data = self._extract_content(no_metadata_result)

            if not no_metadata_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to create context without metadata: {no_metadata_data}'))
                return False

            no_metadata_context_id = no_metadata_data.get('context_id')

            # Apply patch to context with no metadata
            patch_empty_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': no_metadata_context_id,
                    'metadata_patch': {'created_via': 'patch', 'version': 1},
                },
            )

            patch_empty_data = self._extract_content(patch_empty_result)

            if not patch_empty_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to patch empty metadata: {patch_empty_data}'))
                return False

            # Verify metadata was created from scratch
            verify_patch_empty = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [no_metadata_context_id]},
            )

            verify_patch_empty_data = self._extract_content(verify_patch_empty)

            if not verify_patch_empty_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify patch on empty metadata'))
                return False

            created_metadata = verify_patch_empty_data['results'][0].get('metadata', {})

            if created_metadata.get('created_via') != 'patch' or created_metadata.get('version') != 1:
                self.test_results.append((test_name, False, 'Patch on empty metadata did not create expected fields'))
                return False

            # Verify immutable fields remain unchanged
            if text_only_entry.get('thread_id') != update_thread or text_only_entry.get('source') != 'agent':
                self.test_results.append((test_name, False, 'Immutable fields were modified'))
                return False

            self.test_results.append((test_name, True, 'All update operations passed'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_patch_deep_merge(self) -> bool:
        """Test RFC 7396 deep merge semantics for metadata_patch.

        This test verifies that nested objects are correctly merged according to
        RFC 7396 JSON Merge Patch specification, including deep merge and nested
        null deletion.

        RFC 7396 Specification: https://datatracker.ietf.org/doc/html/rfc7396

        Returns:
            bool: True if all deep merge tests passed.
        """
        test_name = 'metadata_patch_deep_merge'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create a separate thread for deep merge tests
            deep_merge_thread = f'{self.test_thread_id}_deep_merge'

            # Test 1: Setup - Create context with nested metadata
            setup_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': deep_merge_thread,
                    'source': 'agent',
                    'text': 'Context for RFC 7396 deep merge testing',
                    'metadata': {
                        'a': {
                            'b': 'original_b',
                            'd': 'original_d',
                        },
                        'top_level': 'preserved',
                    },
                },
            )

            setup_data = self._extract_content(setup_result)
            if not setup_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to create test context: {setup_data}'))
                return False

            context_id = setup_data.get('context_id')

            # Test 2: RFC 7396 Case #7 - Deep merge with nested update (preserves siblings)
            # Patch: {"a": {"b": "updated"}} should preserve "d" in nested object
            patch_deep_merge = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata_patch': {'a': {'b': 'updated_b'}},
                },
            )

            patch_deep_data = self._extract_content(patch_deep_merge)
            if not patch_deep_data.get('success'):
                self.test_results.append((test_name, False, f'Failed deep merge patch: {patch_deep_data}'))
                return False

            # Verify deep merge preserved sibling key
            verify_deep = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_deep_data = self._extract_content(verify_deep)
            if not verify_deep_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify deep merge'))
                return False

            deep_metadata = verify_deep_data['results'][0].get('metadata', {})

            # RFC 7396: Nested sibling key "d" MUST be preserved
            if deep_metadata.get('a', {}).get('b') != 'updated_b':
                self.test_results.append((test_name, False, 'Deep merge did not update nested key "b"'))
                return False

            if deep_metadata.get('a', {}).get('d') != 'original_d':
                error_msg = f'RFC 7396 VIOLATION: Deep merge did not preserve sibling key "d". Got: {deep_metadata}'
                self.test_results.append((test_name, False, error_msg))
                return False

            if deep_metadata.get('top_level') != 'preserved':
                self.test_results.append((test_name, False, 'Deep merge did not preserve top-level key'))
                return False

            # Test 3: Nested null deletion (RFC 7396 Case #7 variant)
            # Patch: {"a": {"b": null}} should delete "b" but preserve "d"
            patch_nested_delete = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata_patch': {'a': {'b': None}},  # RFC 7396: null means delete
                },
            )

            patch_delete_data = self._extract_content(patch_nested_delete)
            if not patch_delete_data.get('success'):
                self.test_results.append((test_name, False, f'Failed nested deletion patch: {patch_delete_data}'))
                return False

            # Verify nested deletion preserved sibling
            verify_delete = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_delete_data = self._extract_content(verify_delete)
            if not verify_delete_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify nested deletion'))
                return False

            delete_metadata = verify_delete_data['results'][0].get('metadata', {})

            # RFC 7396: Key "b" should be deleted
            if 'b' in delete_metadata.get('a', {}):
                self.test_results.append(
                    (test_name, False, f'RFC 7396 VIOLATION: Nested null did not delete key "b". Got: {delete_metadata}'),
                )
                return False

            # RFC 7396: Key "d" MUST be preserved
            if delete_metadata.get('a', {}).get('d') != 'original_d':
                error_msg = f'RFC 7396 VIOLATION: Nested deletion did not preserve "d". Got: {delete_metadata}'
                self.test_results.append((test_name, False, error_msg))
                return False

            # Test 4: Deeply nested null deletion (RFC 7396 Case #15)
            # Create new context for this test
            deep_nested_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': deep_merge_thread,
                    'source': 'agent',
                    'text': 'Context for deeply nested null test',
                    'metadata': {},  # Start empty
                },
            )

            deep_nested_data = self._extract_content(deep_nested_result)
            if not deep_nested_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to create deeply nested test context'))
                return False

            deep_context_id = deep_nested_data.get('context_id')

            # RFC 7396 Case #15: {"a": {"bb": {"ccc": null}}} should result in {"a": {"bb": {}}}
            patch_deep_null = await self.client.call_tool(
                'update_context',
                {
                    'context_id': deep_context_id,
                    'metadata_patch': {'a': {'bb': {'ccc': None}}},
                },
            )

            patch_deep_null_data = self._extract_content(patch_deep_null)
            if not patch_deep_null_data.get('success'):
                self.test_results.append((test_name, False, f'Failed deeply nested null patch: {patch_deep_null_data}'))
                return False

            # Verify deeply nested structure
            verify_deep_null = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [deep_context_id]},
            )

            verify_deep_null_data = self._extract_content(verify_deep_null)
            if not verify_deep_null_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify deeply nested null'))
                return False

            deep_null_metadata = verify_deep_null_data['results'][0].get('metadata', {})

            # RFC 7396 Case #15: Expected {"a": {"bb": {}}}
            # The deeply nested null should create empty nested objects, not include null
            if 'a' not in deep_null_metadata:
                self.test_results.append(
                    (test_name, False, f'RFC 7396 Case #15 VIOLATION: Missing top-level "a". Got: {deep_null_metadata}'),
                )
                return False

            if 'bb' not in deep_null_metadata.get('a', {}):
                self.test_results.append(
                    (test_name, False, f'RFC 7396 Case #15 VIOLATION: Missing nested "bb". Got: {deep_null_metadata}'),
                )
                return False

            # The key "ccc" should NOT exist (deleted by null)
            if 'ccc' in deep_null_metadata.get('a', {}).get('bb', {}):
                self.test_results.append(
                    (test_name, False, f'RFC 7396 Case #15 VIOLATION: Key "ccc" should be deleted. Got: {deep_null_metadata}'),
                )
                return False

            self.test_results.append((test_name, True, 'All RFC 7396 deep merge tests passed'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_context_with_date_filtering(self) -> bool:
        """Test search_context with start_date and end_date parameters.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context_date_filtering'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create a separate thread for date filtering tests
            date_filter_thread = f'{self.test_thread_id}_date_filter'

            # Store a test entry (will be created at current time)
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': date_filter_thread,
                    'source': 'user',
                    'text': 'Entry for date filtering test',
                    'tags': ['date-filter', 'test'],
                },
            )

            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store test entry: {store_data}'))
                return False

            # Get current date information for testing
            from datetime import UTC
            from datetime import datetime
            from datetime import timedelta

            today = datetime.now(UTC).strftime('%Y-%m-%d')
            tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
            yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
            future_date = (datetime.now(UTC) + timedelta(days=30)).strftime('%Y-%m-%d')
            past_date = (datetime.now(UTC) - timedelta(days=30)).strftime('%Y-%m-%d')

            # Test 1: Search with valid date range (today to tomorrow) - should find entry
            valid_range_result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': date_filter_thread,
                    'start_date': today,
                    'end_date': tomorrow,
                },
            )
            valid_range_data = self._extract_content(valid_range_result)
            if not valid_range_data.get('success') or len(valid_range_data.get('results', [])) != 1:
                self.test_results.append(
                    (test_name, False, f'Valid date range search failed: {valid_range_data}'),
                )
                return False

            # Test 2: Search with future start_date - should NOT find entry
            future_start_result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': date_filter_thread,
                    'start_date': future_date,
                },
            )
            future_start_data = self._extract_content(future_start_result)
            if not future_start_data.get('success') or len(future_start_data.get('results', [])) != 0:
                self.test_results.append(
                    (test_name, False, f'Future start_date returned results unexpectedly: {future_start_data}'),
                )
                return False

            # Test 3: Search with past end_date - should NOT find entry
            past_end_result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': date_filter_thread,
                    'end_date': past_date,
                },
            )
            past_end_data = self._extract_content(past_end_result)
            if not past_end_data.get('success') or len(past_end_data.get('results', [])) != 0:
                self.test_results.append(
                    (test_name, False, f'Past end_date returned results unexpectedly: {past_end_data}'),
                )
                return False

            # Test 4: Search with date-only end_date for today - should find entry
            # This verifies the UX fix where date-only end_date is expanded to end-of-day
            today_end_result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': date_filter_thread,
                    'end_date': today,
                },
            )
            today_end_data = self._extract_content(today_end_result)
            if not today_end_data.get('success') or len(today_end_data.get('results', [])) != 1:
                self.test_results.append(
                    (test_name, False, f'Date-only end_date failed to find today entry: {today_end_data}'),
                )
                return False

            # Test 5: Combined filters (date + source)
            combined_result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': date_filter_thread,
                    'source': 'user',
                    'start_date': yesterday,
                    'end_date': tomorrow,
                },
            )
            combined_data = self._extract_content(combined_result)
            if not combined_data.get('success') or len(combined_data.get('results', [])) != 1:
                self.test_results.append(
                    (test_name, False, f'Combined date+source filter failed: {combined_data}'),
                )
                return False

            self.test_results.append((test_name, True, 'All date filtering tests passed'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_semantic_search_context_with_date_filtering(self) -> bool:
        """Test semantic_search_context with date filtering parameters.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'semantic_search_date_filtering'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if semantic search is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            is_enabled = semantic_info.get('enabled', False)
            is_available = semantic_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for semantic search date filtering tests
            semantic_date_thread = f'{self.test_thread_id}_semantic_date'

            # Store semantically meaningful test content
            test_contexts = [
                'Machine learning algorithms process data to make predictions',
                'Database indexing improves query performance significantly',
            ]

            for text in test_contexts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': semantic_date_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Allow time for embedding generation
            import asyncio

            await asyncio.sleep(0.5)

            # Get date information for filtering
            from datetime import UTC
            from datetime import datetime
            from datetime import timedelta

            today = datetime.now(UTC).strftime('%Y-%m-%d')
            tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
            future_date = (datetime.now(UTC) + timedelta(days=30)).strftime('%Y-%m-%d')
            past_date = (datetime.now(UTC) - timedelta(days=30)).strftime('%Y-%m-%d')

            # Test 1: Semantic search with valid date range - should find results
            valid_range_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'machine learning artificial intelligence',
                    'thread_id': semantic_date_thread,
                    'start_date': today,
                    'end_date': tomorrow,
                    'limit': 5,
                },
            )
            valid_range_data = self._extract_content(valid_range_result)
            if 'results' not in valid_range_data or len(valid_range_data.get('results', [])) == 0:
                self.test_results.append(
                    (test_name, False, f'Valid date range semantic search failed: {valid_range_data}'),
                )
                return False

            # Test 2: Semantic search with future start_date - should return empty
            future_start_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'machine learning',
                    'thread_id': semantic_date_thread,
                    'start_date': future_date,
                    'limit': 5,
                },
            )
            future_start_data = self._extract_content(future_start_result)
            if 'results' not in future_start_data or len(future_start_data.get('results', [])) != 0:
                self.test_results.append(
                    (test_name, False, f'Future start_date returned results unexpectedly: {future_start_data}'),
                )
                return False

            # Test 3: Semantic search with past end_date - should return empty
            past_end_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'database indexing',
                    'thread_id': semantic_date_thread,
                    'end_date': past_date,
                    'limit': 5,
                },
            )
            past_end_data = self._extract_content(past_end_result)
            if 'results' not in past_end_data or len(past_end_data.get('results', [])) != 0:
                self.test_results.append(
                    (test_name, False, f'Past end_date returned results unexpectedly: {past_end_data}'),
                )
                return False

            # Test 4: Combined filters (date + source)
            combined_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'algorithms data processing',
                    'thread_id': semantic_date_thread,
                    'source': 'agent',
                    'start_date': today,
                    'end_date': tomorrow,
                    'limit': 5,
                },
            )
            combined_data = self._extract_content(combined_result)
            if 'results' not in combined_data or len(combined_data.get('results', [])) == 0:
                self.test_results.append(
                    (test_name, False, f'Combined date+source filter failed: {combined_data}'),
                )
                return False

            self.test_results.append((test_name, True, 'All semantic search date filtering tests passed'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_store_context_batch(self) -> bool:
        """Test bulk store context operations.

        Tests atomic and non-atomic modes for batch storing multiple entries.

        Returns:
            bool: True if test passed.
        """
        test_name = 'store_context_batch'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create a separate thread for bulk store tests
            bulk_store_thread = f'{self.test_thread_id}_bulk_store'

            # Test 1: Store multiple entries successfully (atomic=True)
            entries = [
                {
                    'thread_id': bulk_store_thread,
                    'source': 'user',
                    'text': 'First bulk entry',
                    'metadata': {'priority': 1, 'type': 'test'},
                    'tags': ['bulk', 'first'],
                },
                {
                    'thread_id': bulk_store_thread,
                    'source': 'agent',
                    'text': 'Second bulk entry',
                    'metadata': {'priority': 2, 'type': 'test'},
                    'tags': ['bulk', 'second'],
                },
                {
                    'thread_id': bulk_store_thread,
                    'source': 'user',
                    'text': 'Third bulk entry',
                    'tags': ['bulk', 'third'],
                },
            ]

            result = await self.client.call_tool(
                'store_context_batch',
                {'entries': entries, 'atomic': True},
            )

            result_data = self._extract_content(result)

            if not result_data.get('success'):
                self.test_results.append((test_name, False, f'Atomic batch store failed: {result_data}'))
                return False

            if result_data.get('total') != 3 or result_data.get('succeeded') != 3:
                self.test_results.append(
                    (test_name, False, f"Expected 3 stored, got {result_data.get('succeeded')}/{result_data.get('total')}"),
                )
                return False

            # Verify all entries have context_ids
            results = result_data.get('results', [])
            if len(results) != 3 or not all(r.get('context_id') for r in results):
                self.test_results.append((test_name, False, 'Missing context_ids in results'))
                return False

            # Test 2: Verify entries stored correctly via search
            search_result = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': bulk_store_thread},
            )

            search_data = self._extract_content(search_result)

            if len(search_data.get('results', [])) != 3:
                self.test_results.append(
                    (test_name, False, f"Expected 3 entries, found {len(search_data.get('results', []))}"),
                )
                return False

            # Test 3: Non-atomic mode (atomic=False)
            non_atomic_thread = f'{self.test_thread_id}_bulk_nonatomic'
            non_atomic_entries = [
                {
                    'thread_id': non_atomic_thread,
                    'source': 'agent',
                    'text': 'Non-atomic entry 1',
                },
                {
                    'thread_id': non_atomic_thread,
                    'source': 'user',
                    'text': 'Non-atomic entry 2',
                    'metadata': {'processed': True},
                },
            ]

            non_atomic_result = await self.client.call_tool(
                'store_context_batch',
                {'entries': non_atomic_entries, 'atomic': False},
            )

            non_atomic_data = self._extract_content(non_atomic_result)

            if not non_atomic_data.get('success') or non_atomic_data.get('succeeded') != 2:
                self.test_results.append((test_name, False, f'Non-atomic batch store failed: {non_atomic_data}'))
                return False

            stored_count = result_data.get('succeeded', 0) + non_atomic_data.get('succeeded', 0)
            self.test_results.append((test_name, True, f'Stored {stored_count} entries in batch'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_update_context_batch(self) -> bool:
        """Test bulk update context operations.

        Tests batch updating multiple entries with various field combinations.

        Returns:
            bool: True if test passed.
        """
        test_name = 'update_context_batch'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create a separate thread for bulk update tests
            bulk_update_thread = f'{self.test_thread_id}_bulk_update'

            # First, create entries to update
            setup_entries = [
                {
                    'thread_id': bulk_update_thread,
                    'source': 'user',
                    'text': 'Original text 1',
                    'metadata': {'status': 'draft', 'version': 1},
                    'tags': ['original'],
                },
                {
                    'thread_id': bulk_update_thread,
                    'source': 'agent',
                    'text': 'Original text 2',
                    'metadata': {'status': 'pending', 'version': 1},
                    'tags': ['original'],
                },
                {
                    'thread_id': bulk_update_thread,
                    'source': 'user',
                    'text': 'Original text 3',
                    'tags': ['original'],
                },
            ]

            setup_result = await self.client.call_tool(
                'store_context_batch',
                {'entries': setup_entries, 'atomic': True},
            )

            setup_data = self._extract_content(setup_result)

            if not setup_data.get('success') or setup_data.get('succeeded') != 3:
                self.test_results.append((test_name, False, f'Failed to setup test entries: {setup_data}'))
                return False

            # Get the context IDs
            context_ids = [r['context_id'] for r in setup_data['results']]

            # Test 1: Batch update text for multiple entries
            updates = [
                {'context_id': context_ids[0], 'text': 'Updated text 1'},
                {'context_id': context_ids[1], 'text': 'Updated text 2'},
                {'context_id': context_ids[2], 'text': 'Updated text 3'},
            ]

            update_result = await self.client.call_tool(
                'update_context_batch',
                {'updates': updates, 'atomic': True},
            )

            update_data = self._extract_content(update_result)

            if not update_data.get('success') or update_data.get('succeeded') != 3:
                self.test_results.append((test_name, False, f'Batch text update failed: {update_data}'))
                return False

            # Verify updated_fields contains text_content
            for item in update_data.get('results', []):
                if 'text_content' not in item.get('updated_fields', []):
                    self.test_results.append((test_name, False, 'text_content not in updated_fields'))
                    return False

            # Test 2: Batch update metadata
            metadata_updates = [
                {
                    'context_id': context_ids[0],
                    'metadata': {'status': 'completed', 'version': 2},
                },
                {
                    'context_id': context_ids[1],
                    'metadata_patch': {'version': 2, 'reviewed': True},
                },
            ]

            metadata_result = await self.client.call_tool(
                'update_context_batch',
                {'updates': metadata_updates, 'atomic': True},
            )

            metadata_data = self._extract_content(metadata_result)

            if not metadata_data.get('success') or metadata_data.get('succeeded') != 2:
                self.test_results.append((test_name, False, f'Batch metadata update failed: {metadata_data}'))
                return False

            # Test 3: Verify updates via get_context_by_ids
            verify_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': context_ids},
            )

            verify_data = self._extract_content(verify_result)

            if not verify_data.get('success') or len(verify_data.get('results', [])) != 3:
                self.test_results.append((test_name, False, 'Failed to verify updates'))
                return False

            # Check that text was updated
            for entry in verify_data['results']:
                if not entry.get('text_content', '').startswith('Updated text'):
                    self.test_results.append((test_name, False, 'Text not updated correctly'))
                    return False

            self.test_results.append((test_name, True, f'Updated {update_data.get("succeeded", 0)} entries in batch'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_delete_context_batch(self) -> bool:
        """Test bulk delete context operations.

        Tests deletion by various criteria: context_ids, thread_ids, and combined filters.

        Returns:
            bool: True if test passed.
        """
        test_name = 'delete_context_batch'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create separate threads for bulk delete tests
            delete_by_ids_thread = f'{self.test_thread_id}_bulk_del_ids'
            delete_by_thread_thread = f'{self.test_thread_id}_bulk_del_thread'
            delete_combined_thread = f'{self.test_thread_id}_bulk_del_combined'

            # Setup: Create entries in different threads for deletion tests
            setup_entries = [
                # Entries for delete by IDs test
                {'thread_id': delete_by_ids_thread, 'source': 'user', 'text': 'Delete by ID 1'},
                {'thread_id': delete_by_ids_thread, 'source': 'agent', 'text': 'Delete by ID 2'},
                # Entries for delete by thread test
                {'thread_id': delete_by_thread_thread, 'source': 'user', 'text': 'Delete by thread 1'},
                {'thread_id': delete_by_thread_thread, 'source': 'agent', 'text': 'Delete by thread 2'},
                {'thread_id': delete_by_thread_thread, 'source': 'user', 'text': 'Delete by thread 3'},
                # Entries for combined criteria test
                {'thread_id': delete_combined_thread, 'source': 'user', 'text': 'Combined user 1'},
                {'thread_id': delete_combined_thread, 'source': 'user', 'text': 'Combined user 2'},
                {'thread_id': delete_combined_thread, 'source': 'agent', 'text': 'Combined agent 1'},
            ]

            setup_result = await self.client.call_tool(
                'store_context_batch',
                {'entries': setup_entries, 'atomic': True},
            )

            setup_data = self._extract_content(setup_result)

            if not setup_data.get('success') or setup_data.get('succeeded') != 8:
                self.test_results.append((test_name, False, f'Failed to setup delete test entries: {setup_data}'))
                return False

            # Get context IDs for the first two entries (delete by IDs test)
            ids_to_delete = [setup_data['results'][0]['context_id'], setup_data['results'][1]['context_id']]

            # Test 1: Delete by context_ids
            delete_by_ids_result = await self.client.call_tool(
                'delete_context_batch',
                {'context_ids': ids_to_delete},
            )

            delete_by_ids_data = self._extract_content(delete_by_ids_result)

            if not delete_by_ids_data.get('success') or delete_by_ids_data.get('deleted_count') != 2:
                self.test_results.append(
                    (test_name, False, f"Delete by IDs failed: expected 2, got {delete_by_ids_data.get('deleted_count')}"),
                )
                return False

            # Verify criteria_used contains context_ids
            if 'context_ids' not in str(delete_by_ids_data.get('criteria_used', [])):
                self.test_results.append((test_name, False, 'context_ids not in criteria_used'))
                return False

            # Verify entries are deleted
            verify_deleted = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': ids_to_delete},
            )

            verify_deleted_data = self._extract_content(verify_deleted)

            if len(verify_deleted_data.get('results', [])) > 0:
                self.test_results.append((test_name, False, 'Entries not deleted by IDs'))
                return False

            # Test 2: Delete by thread_ids
            delete_by_thread_result = await self.client.call_tool(
                'delete_context_batch',
                {'thread_ids': [delete_by_thread_thread]},
            )

            delete_by_thread_data = self._extract_content(delete_by_thread_result)

            if not delete_by_thread_data.get('success') or delete_by_thread_data.get('deleted_count') != 3:
                deleted = delete_by_thread_data.get('deleted_count')
                self.test_results.append(
                    (test_name, False, f'Delete by thread failed: expected 3, got {deleted}'),
                )
                return False

            # Verify thread is empty
            verify_thread = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': delete_by_thread_thread},
            )

            verify_thread_data = self._extract_content(verify_thread)

            if len(verify_thread_data.get('results', [])) > 0:
                self.test_results.append((test_name, False, 'Thread entries not deleted'))
                return False

            # Test 3: Delete by combined criteria (thread + source)
            delete_combined_result = await self.client.call_tool(
                'delete_context_batch',
                {'thread_ids': [delete_combined_thread], 'source': 'user'},
            )

            delete_combined_data = self._extract_content(delete_combined_result)

            if not delete_combined_data.get('success') or delete_combined_data.get('deleted_count') != 2:
                self.test_results.append(
                    (test_name, False, f"Combined delete failed: expected 2, got {delete_combined_data.get('deleted_count')}"),
                )
                return False

            # Verify only agent entry remains
            verify_combined = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': delete_combined_thread},
            )

            verify_combined_data = self._extract_content(verify_combined)

            remaining = verify_combined_data.get('results', [])
            if len(remaining) != 1 or remaining[0].get('source') != 'agent':
                self.test_results.append((test_name, False, 'Combined criteria did not filter correctly'))
                return False

            total_deleted = (
                delete_by_ids_data.get('deleted_count', 0)
                + delete_by_thread_data.get('deleted_count', 0)
                + delete_combined_data.get('deleted_count', 0)
            )
            self.test_results.append((test_name, True, f'Deleted {total_deleted} entries with various criteria'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_semantic_search_context(self) -> bool:
        """Test semantic search functionality (conditional on availability).

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'semantic_search_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if semantic search is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            is_enabled = semantic_info.get('enabled', False)
            is_available = semantic_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for semantic search tests
            semantic_thread = f'{self.test_thread_id}_semantic'

            # Store semantically diverse test contexts
            test_contexts = [
                'Machine learning models require large datasets for training and validation',
                'Python is a popular programming language for data science and AI applications',
                'The weather today is sunny with a high of 25 degrees celsius',
            ]

            for text in test_contexts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': semantic_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Allow time for embedding generation (non-blocking operation)
            await asyncio.sleep(0.5)

            # Test 1: Semantic search for ML-related content
            ml_search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'artificial intelligence and deep learning',
                    'limit': 5,
                },
            )

            ml_search_data = self._extract_content(ml_search_result)

            # semantic_search_context returns results directly without 'success' field
            # Check for 'results' key instead
            if 'results' not in ml_search_data:
                self.test_results.append((test_name, False, f'ML semantic search failed: {ml_search_data}'))
                return False

            # Verify results contain distance/similarity scores
            ml_results = ml_search_data.get('results', [])
            if not ml_results or 'distance' not in ml_results[0]:
                self.test_results.append((test_name, False, 'Missing distance scores in results'))
                return False

            # Test 2: Search with thread_id filter
            thread_search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'programming languages',
                    'thread_id': semantic_thread,
                    'limit': 3,
                },
            )

            thread_search_data = self._extract_content(thread_search_result)

            if 'results' not in thread_search_data:
                self.test_results.append((test_name, False, f'Thread-filtered search failed: {thread_search_data}'))
                return False

            # Test 3: Search with source filter
            source_search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'data science',
                    'source': 'agent',
                    'limit': 5,
                },
            )

            source_search_data = self._extract_content(source_search_result)

            if 'results' not in source_search_data:
                self.test_results.append((test_name, False, f'Source-filtered search failed: {source_search_data}'))
                return False

            # Get model name for success message
            model_name = semantic_info.get('model', 'unknown')

            self.test_results.append((test_name, True, f'Semantic search working (model: {model_name})'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_semantic_search_context_with_metadata_filters(self) -> bool:
        """Test semantic search with metadata filtering (conditional on availability).

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'semantic_search_context_with_metadata_filters'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if semantic search is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            is_enabled = semantic_info.get('enabled', False)
            is_available = semantic_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for metadata filter tests
            metadata_thread = f'{self.test_thread_id}_semantic_metadata'

            # Store test contexts with different metadata
            test_entries = [
                {'text': 'High priority backend task for API development', 'metadata': {'priority': 9, 'category': 'backend'}},
                {'text': 'Low priority frontend task for UI updates', 'metadata': {'priority': 3, 'category': 'frontend'}},
                {'text': 'High priority database optimization task', 'metadata': {'priority': 8, 'category': 'backend'}},
                {'text': 'Medium priority testing task', 'metadata': {'priority': 5, 'category': 'testing'}},
            ]

            for entry in test_entries:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': metadata_thread,
                        'source': 'agent',
                        'text': entry['text'],
                        'metadata': entry['metadata'],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: Semantic search with simple metadata filter (category=backend)
            metadata_search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'development tasks',
                    'thread_id': metadata_thread,
                    'metadata': {'category': 'backend'},
                    'limit': 10,
                },
            )

            metadata_search_data = self._extract_content(metadata_search_result)

            if 'results' not in metadata_search_data:
                self.test_results.append((test_name, False, f'Metadata filter search failed: {metadata_search_data}'))
                return False

            # Should return only backend entries (2 entries)
            metadata_results = metadata_search_data.get('results', [])
            if len(metadata_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 backend entries, got {len(metadata_results)}'),
                )
                return False

            # Test 2: Semantic search with advanced metadata filter (priority > 5)
            advanced_search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'tasks',
                    'thread_id': metadata_thread,
                    'metadata_filters': [{'key': 'priority', 'operator': 'gt', 'value': 5}],
                    'limit': 10,
                },
            )

            advanced_search_data = self._extract_content(advanced_search_result)

            if 'results' not in advanced_search_data:
                self.test_results.append((test_name, False, f'Advanced filter search failed: {advanced_search_data}'))
                return False

            # Should return entries with priority > 5 (priority 8 and 9)
            advanced_results = advanced_search_data.get('results', [])
            if len(advanced_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 high priority entries, got {len(advanced_results)}'),
                )
                return False

            # Test 3: Combined metadata + other filters
            combined_search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'backend',
                    'thread_id': metadata_thread,
                    'source': 'agent',
                    'metadata': {'category': 'backend'},
                    'metadata_filters': [{'key': 'priority', 'operator': 'gte', 'value': 8}],
                    'limit': 10,
                },
            )

            combined_search_data = self._extract_content(combined_search_result)

            if 'results' not in combined_search_data:
                self.test_results.append((test_name, False, f'Combined filter search failed: {combined_search_data}'))
                return False

            # Should return only high priority backend entries (priority >= 8 and category=backend)
            combined_results = combined_search_data.get('results', [])
            if len(combined_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 combined filter entries, got {len(combined_results)}'),
                )
                return False

            self.test_results.append((test_name, True, 'Semantic search with metadata filtering working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_context_invalid_filter_returns_error(self) -> bool:
        """Test that search_context returns explicit error for invalid metadata filter.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context_invalid_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Test with invalid operator
            result = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'metadata_filters': [{'key': 'status', 'operator': 'invalid_operator', 'value': 'test'}]},
            )

            result_data = self._extract_content(result)

            # Should return error response
            if 'error' not in result_data:
                self.test_results.append((test_name, False, f'Expected error response, got: {result_data}'))
                return False

            if result_data['error'] != 'Metadata filter validation failed':
                self.test_results.append((test_name, False, f"Wrong error message: {result_data['error']}"))
                return False

            if 'validation_errors' not in result_data:
                self.test_results.append((test_name, False, 'Missing validation_errors in response'))
                return False

            self.test_results.append((test_name, True, 'Invalid filter returns error as expected'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_semantic_search_invalid_filter_returns_error(self) -> bool:
        """Test that semantic_search_context returns explicit error for invalid metadata filter.

        This test verifies unified error handling between search_context and semantic_search_context.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'semantic_search_invalid_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if semantic search is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            is_enabled = semantic_info.get('enabled', False)
            is_available = semantic_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Test with invalid operator
            result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'test query',
                    'metadata_filters': [{'key': 'status', 'operator': 'invalid_operator', 'value': 'test'}],
                },
            )

            result_data = self._extract_content(result)

            # Should return error response (unified with search_context behavior)
            if 'error' not in result_data:
                self.test_results.append((test_name, False, f'Expected error response, got: {result_data}'))
                return False

            if result_data['error'] != 'Metadata filter validation failed':
                self.test_results.append((test_name, False, f"Wrong error message: {result_data['error']}"))
                return False

            if 'validation_errors' not in result_data:
                self.test_results.append((test_name, False, 'Missing validation_errors in response'))
                return False

            # Verify response structure includes expected fields
            if result_data.get('count') != 0:
                self.test_results.append((test_name, False, 'Expected count=0 on error'))
                return False

            if result_data.get('results') != []:
                self.test_results.append((test_name, False, 'Expected empty results on error'))
                return False

            self.test_results.append((test_name, True, 'Invalid filter returns error (unified with search_context)'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_search_context(self) -> bool:
        """Test full-text search functionality (conditional on availability).

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_search_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if FTS is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for FTS tests
            fts_thread = f'{self.test_thread_id}_fts'

            # Store test contexts for full-text search
            test_contexts = [
                'Python programming language tutorial for beginners',
                'Advanced machine learning with Python frameworks',
                'JavaScript and TypeScript web development guide',
                'Database indexing and query optimization techniques',
            ]

            for text in test_contexts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': fts_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: Basic match mode search for 'Python'
            match_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'python',
                    'mode': 'match',
                    'thread_id': fts_thread,
                    'limit': 10,
                },
            )

            match_data = self._extract_content(match_result)

            # Check for results
            if 'results' not in match_data:
                self.test_results.append((test_name, False, f'Match mode search failed: {match_data}'))
                return False

            match_results = match_data.get('results', [])
            if len(match_results) != 2:  # Should find 2 Python entries
                self.test_results.append(
                    (test_name, False, f'Expected 2 Python results, got {len(match_results)}'),
                )
                return False

            # Verify results have scores
            if not all('score' in r for r in match_results):
                self.test_results.append((test_name, False, 'Missing scores in results'))
                return False

            # Test 2: Prefix mode search for 'prog*'
            prefix_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'prog',
                    'mode': 'prefix',
                    'thread_id': fts_thread,
                    'limit': 10,
                },
            )

            prefix_data = self._extract_content(prefix_result)

            if 'results' not in prefix_data:
                self.test_results.append((test_name, False, f'Prefix mode search failed: {prefix_data}'))
                return False

            prefix_results = prefix_data.get('results', [])
            if len(prefix_results) < 1:  # Should find at least 1 entry with 'programming'
                self.test_results.append(
                    (test_name, False, f'Expected results for prefix "prog*", got {len(prefix_results)}'),
                )
                return False

            # Test 3: Phrase mode search
            phrase_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'machine learning',
                    'mode': 'phrase',
                    'thread_id': fts_thread,
                    'limit': 10,
                },
            )

            phrase_data = self._extract_content(phrase_result)

            if 'results' not in phrase_data:
                self.test_results.append((test_name, False, f'Phrase mode search failed: {phrase_data}'))
                return False

            phrase_results = phrase_data.get('results', [])
            if len(phrase_results) != 1:  # Should find exactly 1 entry with 'machine learning'
                self.test_results.append(
                    (test_name, False, f'Expected 1 phrase result, got {len(phrase_results)}'),
                )
                return False

            # Test 4: Search with source filter
            source_filter_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'python',
                    'mode': 'match',
                    'thread_id': fts_thread,
                    'source': 'agent',
                    'limit': 10,
                },
            )

            source_data = self._extract_content(source_filter_result)

            if 'results' not in source_data:
                self.test_results.append((test_name, False, f'Source filter search failed: {source_data}'))
                return False

            # All our test entries are from 'agent', should still find 2
            source_results = source_data.get('results', [])
            if len(source_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 agent results, got {len(source_results)}'),
                )
                return False

            # Test 5: Verify response structure includes required fields
            if match_data.get('mode') != 'match':
                self.test_results.append((test_name, False, 'Response missing mode field'))
                return False

            if 'count' not in match_data:
                self.test_results.append((test_name, False, 'Response missing count field'))
                return False

            if 'language' not in match_data:
                self.test_results.append((test_name, False, 'Response missing language field'))
                return False

            self.test_results.append((test_name, True, 'FTS search modes and filters working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_search_invalid_filter_returns_error(self) -> bool:
        """Test that fts_search_context returns explicit error for invalid metadata filter.

        This test verifies unified error handling between fts_search_context and other search tools.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_search_invalid_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if FTS is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Test with invalid operator
            result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'test query',
                    'metadata_filters': [{'key': 'status', 'operator': 'invalid_operator', 'value': 'test'}],
                },
            )

            result_data = self._extract_content(result)

            # Should return error response (unified with search_context behavior)
            if 'error' not in result_data:
                self.test_results.append((test_name, False, f'Expected error response, got: {result_data}'))
                return False

            if result_data['error'] != 'Metadata filter validation failed':
                self.test_results.append((test_name, False, f"Wrong error message: {result_data['error']}"))
                return False

            if 'validation_errors' not in result_data:
                self.test_results.append((test_name, False, 'Missing validation_errors in response'))
                return False

            # Verify response structure includes expected fields
            if result_data.get('count') != 0:
                self.test_results.append((test_name, False, 'Expected count=0 on error'))
                return False

            if result_data.get('results') != []:
                self.test_results.append((test_name, False, 'Expected empty results on error'))
                return False

            self.test_results.append((test_name, True, 'Invalid filter returns error (unified with search_context)'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_boolean_mode(self) -> bool:
        """Test FTS boolean mode with AND/OR/NOT operators.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_boolean_mode'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if FTS is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for boolean mode tests
            bool_thread = f'{self.test_thread_id}_fts_boolean'

            # Store test contexts for boolean search
            test_contexts = [
                {'text': 'Python is great for data science and machine learning', 'source': 'agent'},
                {'text': 'JavaScript and TypeScript are popular for web development', 'source': 'agent'},
                {'text': 'Python and JavaScript can both handle backend development', 'source': 'user'},
                {'text': 'Rust is known for memory safety without garbage collection', 'source': 'agent'},
            ]

            for ctx in test_contexts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': bool_thread,
                        'source': ctx['source'],
                        'text': ctx['text'],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: OR operator - should find entries with Python OR JavaScript
            or_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'Python OR JavaScript',
                    'mode': 'boolean',
                    'thread_id': bool_thread,
                    'limit': 10,
                },
            )

            or_data = self._extract_content(or_result)

            if 'results' not in or_data:
                self.test_results.append((test_name, False, f'OR search failed: {or_data}'))
                return False

            or_results = or_data.get('results', [])
            # Should find at least 3 entries (2 with Python, 2 with JavaScript, 1 with both)
            if len(or_results) < 3:
                self.test_results.append(
                    (test_name, False, f'Expected at least 3 results for OR query, got {len(or_results)}'),
                )
                return False

            # Test 2: AND operator - should find entries with both Python AND data
            and_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'Python AND data',
                    'mode': 'boolean',
                    'thread_id': bool_thread,
                    'limit': 10,
                },
            )

            and_data = self._extract_content(and_result)

            if 'results' not in and_data:
                self.test_results.append((test_name, False, f'AND search failed: {and_data}'))
                return False

            and_results = and_data.get('results', [])
            # Should find exactly 1 entry with both Python AND data
            if len(and_results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result for AND query, got {len(and_results)}'),
                )
                return False

            # Verify response mode field
            if or_data.get('mode') != 'boolean':
                self.test_results.append((test_name, False, 'Response mode field incorrect'))
                return False

            self.test_results.append((test_name, True, 'Boolean mode OR/AND operators working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_date_range_filter(self) -> bool:
        """Test FTS date range filtering with start_date and end_date.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_date_range_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if FTS is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for date filter tests
            date_thread = f'{self.test_thread_id}_fts_date'

            # Store test contexts
            test_texts = [
                'Database optimization techniques for large datasets',
                'Query performance tuning and indexing strategies',
            ]

            for text in test_texts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': date_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Get dates in ISO format for filtering using UTC timezone
            from datetime import UTC
            from datetime import datetime
            from datetime import timedelta

            now = datetime.now(tz=UTC)
            yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')
            tomorrow = (now + timedelta(days=1)).strftime('%Y-%m-%d')

            # Test 1: Search with start_date (should include today's entries)
            start_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'database',
                    'mode': 'match',
                    'thread_id': date_thread,
                    'start_date': yesterday,
                    'limit': 10,
                },
            )

            start_data = self._extract_content(start_result)

            if 'results' not in start_data:
                self.test_results.append((test_name, False, f'Start date filter search failed: {start_data}'))
                return False

            start_results = start_data.get('results', [])
            if len(start_results) < 1:
                self.test_results.append(
                    (test_name, False, f'Expected at least 1 result with start_date filter, got {len(start_results)}'),
                )
                return False

            # Test 2: Search with both start_date and end_date
            range_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'database',
                    'mode': 'match',
                    'thread_id': date_thread,
                    'start_date': yesterday,
                    'end_date': tomorrow,
                    'limit': 10,
                },
            )

            range_data = self._extract_content(range_result)

            if 'results' not in range_data:
                self.test_results.append((test_name, False, f'Date range filter search failed: {range_data}'))
                return False

            range_results = range_data.get('results', [])
            if len(range_results) < 1:
                self.test_results.append(
                    (test_name, False, f'Expected at least 1 result with date range filter, got {len(range_results)}'),
                )
                return False

            # Test 3: Search with future start_date (should return no results)
            future_start = (datetime.now(tz=UTC) + timedelta(days=10)).strftime('%Y-%m-%d')
            future_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'database',
                    'mode': 'match',
                    'thread_id': date_thread,
                    'start_date': future_start,
                    'limit': 10,
                },
            )

            future_data = self._extract_content(future_result)

            if 'results' not in future_data:
                self.test_results.append((test_name, False, f'Future date filter search failed: {future_data}'))
                return False

            future_results = future_data.get('results', [])
            if len(future_results) != 0:
                self.test_results.append(
                    (test_name, False, f'Expected 0 results for future start_date, got {len(future_results)}'),
                )
                return False

            self.test_results.append((test_name, True, 'Date range filtering working (start_date, end_date)'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_metadata_filter(self) -> bool:
        """Test FTS simple metadata equality filtering.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_metadata_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if FTS is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for metadata filter tests
            meta_thread = f'{self.test_thread_id}_fts_meta'

            # Store test contexts with different metadata
            test_entries = [
                {
                    'text': 'API design patterns for RESTful services',
                    'metadata': {'category': 'backend', 'priority': 1},
                },
                {
                    'text': 'Frontend component design with React',
                    'metadata': {'category': 'frontend', 'priority': 2},
                },
                {
                    'text': 'Backend database design principles',
                    'metadata': {'category': 'backend', 'priority': 3},
                },
            ]

            for entry in test_entries:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': meta_thread,
                        'source': 'agent',
                        'text': entry['text'],
                        'metadata': entry['metadata'],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: Filter by category='backend'
            backend_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'design',
                    'mode': 'match',
                    'thread_id': meta_thread,
                    'metadata': {'category': 'backend'},
                    'limit': 10,
                },
            )

            backend_data = self._extract_content(backend_result)

            if 'results' not in backend_data:
                self.test_results.append((test_name, False, f'Metadata filter search failed: {backend_data}'))
                return False

            backend_results = backend_data.get('results', [])
            # Should find 2 entries with category='backend'
            if len(backend_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 backend results, got {len(backend_results)}'),
                )
                return False

            # Verify all results have the correct metadata
            for r in backend_results:
                meta = r.get('metadata', {})
                if meta.get('category') != 'backend':
                    self.test_results.append(
                        (test_name, False, f"Result has wrong category: {meta.get('category')}"),
                    )
                    return False

            # Test 2: Filter by category='frontend'
            frontend_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'design',
                    'mode': 'match',
                    'thread_id': meta_thread,
                    'metadata': {'category': 'frontend'},
                    'limit': 10,
                },
            )

            frontend_data = self._extract_content(frontend_result)

            if 'results' not in frontend_data:
                self.test_results.append((test_name, False, f'Frontend filter search failed: {frontend_data}'))
                return False

            frontend_results = frontend_data.get('results', [])
            # Should find exactly 1 entry with category='frontend'
            if len(frontend_results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 frontend result, got {len(frontend_results)}'),
                )
                return False

            self.test_results.append((test_name, True, 'Simple metadata filtering working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_advanced_metadata_filters(self) -> bool:
        """Test FTS advanced metadata filters with operators (gt, lt, contains, etc.).

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_advanced_metadata_filters'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if FTS is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for advanced metadata filter tests
            adv_thread = f'{self.test_thread_id}_fts_adv_meta'

            # Store test contexts with priority metadata for numeric comparison
            test_entries = [
                {
                    'text': 'Critical security vulnerability fix',
                    'metadata': {'priority': 1, 'status': 'resolved'},
                },
                {
                    'text': 'Performance optimization for security module',
                    'metadata': {'priority': 5, 'status': 'pending'},
                },
                {
                    'text': 'Security audit documentation update',
                    'metadata': {'priority': 10, 'status': 'completed'},
                },
            ]

            for entry in test_entries:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': adv_thread,
                        'source': 'agent',
                        'text': entry['text'],
                        'metadata': entry['metadata'],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: Filter with 'gt' (greater than) operator - priority > 3
            gt_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'security',
                    'mode': 'match',
                    'thread_id': adv_thread,
                    'metadata_filters': [{'key': 'priority', 'operator': 'gt', 'value': 3}],
                    'limit': 10,
                },
            )

            gt_data = self._extract_content(gt_result)

            if 'results' not in gt_data:
                self.test_results.append((test_name, False, f'gt operator search failed: {gt_data}'))
                return False

            gt_results = gt_data.get('results', [])
            # Should find 2 entries with priority > 3 (priority 5 and 10)
            if len(gt_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 results for priority > 3, got {len(gt_results)}'),
                )
                return False

            # Verify all results have priority > 3
            for r in gt_results:
                meta = r.get('metadata', {})
                if meta.get('priority', 0) <= 3:
                    self.test_results.append(
                        (test_name, False, f"Result has priority <= 3: {meta.get('priority')}"),
                    )
                    return False

            # Test 2: Filter with 'lt' (less than) operator - priority < 5
            lt_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'security',
                    'mode': 'match',
                    'thread_id': adv_thread,
                    'metadata_filters': [{'key': 'priority', 'operator': 'lt', 'value': 5}],
                    'limit': 10,
                },
            )

            lt_data = self._extract_content(lt_result)

            if 'results' not in lt_data:
                self.test_results.append((test_name, False, f'lt operator search failed: {lt_data}'))
                return False

            lt_results = lt_data.get('results', [])
            # Should find 1 entry with priority < 5 (priority 1)
            if len(lt_results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result for priority < 5, got {len(lt_results)}'),
                )
                return False

            # Test 3: Filter with 'eq' (equals) operator - status = 'pending'
            eq_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'security',
                    'mode': 'match',
                    'thread_id': adv_thread,
                    'metadata_filters': [{'key': 'status', 'operator': 'eq', 'value': 'pending'}],
                    'limit': 10,
                },
            )

            eq_data = self._extract_content(eq_result)

            if 'results' not in eq_data:
                self.test_results.append((test_name, False, f'eq operator search failed: {eq_data}'))
                return False

            eq_results = eq_data.get('results', [])
            # Should find 1 entry with status='pending'
            if len(eq_results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result for status=pending, got {len(eq_results)}'),
                )
                return False

            self.test_results.append((test_name, True, 'Advanced metadata filters (gt, lt, eq) working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_pagination_offset(self) -> bool:
        """Test FTS pagination with offset parameter.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_pagination_offset'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if FTS is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for pagination tests
            page_thread = f'{self.test_thread_id}_fts_page'

            # Store multiple test contexts for pagination
            test_texts = [
                'Testing pagination feature one',
                'Testing pagination feature two',
                'Testing pagination feature three',
                'Testing pagination feature four',
                'Testing pagination feature five',
            ]

            for text in test_texts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': page_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: Get first page (offset=0, limit=2)
            page1_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'pagination',
                    'mode': 'match',
                    'thread_id': page_thread,
                    'offset': 0,
                    'limit': 2,
                },
            )

            page1_data = self._extract_content(page1_result)

            if 'results' not in page1_data:
                self.test_results.append((test_name, False, f'First page search failed: {page1_data}'))
                return False

            page1_results = page1_data.get('results', [])
            if len(page1_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 results on first page, got {len(page1_results)}'),
                )
                return False

            # Get IDs from first page
            page1_ids = {r.get('id') for r in page1_results}

            # Test 2: Get second page (offset=2, limit=2)
            page2_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'pagination',
                    'mode': 'match',
                    'thread_id': page_thread,
                    'offset': 2,
                    'limit': 2,
                },
            )

            page2_data = self._extract_content(page2_result)

            if 'results' not in page2_data:
                self.test_results.append((test_name, False, f'Second page search failed: {page2_data}'))
                return False

            page2_results = page2_data.get('results', [])
            if len(page2_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 results on second page, got {len(page2_results)}'),
                )
                return False

            # Get IDs from second page
            page2_ids = {r.get('id') for r in page2_results}

            # Verify no overlap between pages
            if page1_ids & page2_ids:
                self.test_results.append(
                    (test_name, False, f'Pages overlap: {page1_ids & page2_ids}'),
                )
                return False

            # Test 3: Get third page (offset=4, limit=2) - should get 1 result
            page3_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'pagination',
                    'mode': 'match',
                    'thread_id': page_thread,
                    'offset': 4,
                    'limit': 2,
                },
            )

            page3_data = self._extract_content(page3_result)

            if 'results' not in page3_data:
                self.test_results.append((test_name, False, f'Third page search failed: {page3_data}'))
                return False

            page3_results = page3_data.get('results', [])
            if len(page3_results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result on third page, got {len(page3_results)}'),
                )
                return False

            self.test_results.append((test_name, True, 'Pagination with offset working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_highlight_snippets(self) -> bool:
        """Test FTS highlight parameter returns highlighted text with markers.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_highlight_snippets'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if FTS is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for highlight tests
            hl_thread = f'{self.test_thread_id}_fts_highlight'

            # Store test context with specific searchable terms
            test_text = 'Advanced algorithms for sorting and searching in databases'
            result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': hl_thread,
                    'source': 'agent',
                    'text': test_text,
                },
            )
            result_data = self._extract_content(result)
            if not result_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                return False

            # Test 1: Search without highlight (default)
            no_hl_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'algorithms',
                    'mode': 'match',
                    'thread_id': hl_thread,
                    'limit': 10,
                },
            )

            no_hl_data = self._extract_content(no_hl_result)

            if 'results' not in no_hl_data:
                self.test_results.append((test_name, False, f'No-highlight search failed: {no_hl_data}'))
                return False

            no_hl_results = no_hl_data.get('results', [])
            if len(no_hl_results) < 1:
                self.test_results.append((test_name, False, 'No results found'))
                return False

            # Verify 'highlighted' value is None when highlight=False (default)
            # The field is always present in results but should be None when not requested
            if no_hl_results[0].get('highlighted') is not None:
                self.test_results.append((test_name, False, 'Highlighted value should be None when highlight=False'))
                return False

            # Test 2: Search with highlight=True
            hl_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'algorithms',
                    'mode': 'match',
                    'thread_id': hl_thread,
                    'highlight': True,
                    'limit': 10,
                },
            )

            hl_data = self._extract_content(hl_result)

            if 'results' not in hl_data:
                self.test_results.append((test_name, False, f'Highlight search failed: {hl_data}'))
                return False

            hl_results = hl_data.get('results', [])
            if len(hl_results) < 1:
                self.test_results.append((test_name, False, 'No results found with highlight=True'))
                return False

            # Verify 'highlighted' value is not None when highlight=True
            if hl_results[0].get('highlighted') is None:
                self.test_results.append((test_name, False, 'Highlighted value should not be None when highlight=True'))
                return False

            highlighted_text = hl_results[0].get('highlighted', '')

            # Verify <mark> tags are present in highlighted text
            if '<mark>' not in highlighted_text or '</mark>' not in highlighted_text:
                self.test_results.append(
                    (test_name, False, f'Highlighted text missing <mark> tags: {highlighted_text}'),
                )
                return False

            self.test_results.append((test_name, True, 'Highlight snippets with <mark> tags working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_hybrid_search_context(self) -> bool:
        """Test hybrid search combining FTS and semantic search with RRF fusion.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'hybrid_search_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if ENABLE_HYBRID_SEARCH environment variable is set
            if os.environ.get('ENABLE_HYBRID_SEARCH', '').lower() != 'true':
                self.test_results.append(
                    (test_name, True, 'Skipped (ENABLE_HYBRID_SEARCH not enabled)'),
                )
                return True

            # Check if hybrid search is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            # Check both FTS and semantic search availability
            fts_info = stats_data.get('fts', {})
            semantic_info = stats_data.get('semantic_search', {})

            fts_enabled = fts_info.get('enabled', False)
            fts_available = fts_info.get('available', False)
            semantic_enabled = semantic_info.get('enabled', False)
            semantic_available = semantic_info.get('available', False)

            # Hybrid search requires at least one of FTS or semantic to be available
            has_fts = fts_enabled and fts_available
            has_semantic = semantic_enabled and semantic_available

            # Skip gracefully if neither search type is available
            if not has_fts and not has_semantic:
                self.test_results.append(
                    (
                        test_name,
                        True,
                        f'Skipped (fts={has_fts}, semantic={has_semantic})',
                    ),
                )
                return True

            # Create a separate thread for hybrid search tests
            hybrid_thread = f'{self.test_thread_id}_hybrid'

            # Store test contexts with diverse content
            test_contexts = [
                'Python machine learning algorithms for data science applications',
                'Advanced database indexing and query optimization techniques',
                'Neural networks and deep learning frameworks in Python',
                'JavaScript frontend development with modern frameworks',
            ]

            for text in test_contexts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': hybrid_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: Basic hybrid search with default settings
            hybrid_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'python machine learning',
                    'thread_id': hybrid_thread,
                    'limit': 10,
                },
            )

            hybrid_data = self._extract_content(hybrid_result)

            # Check for results
            if 'results' not in hybrid_data:
                self.test_results.append((test_name, False, f'Hybrid search failed: {hybrid_data}'))
                return False

            # Verify response structure
            if 'fusion_method' not in hybrid_data:
                self.test_results.append((test_name, False, 'Response missing fusion_method field'))
                return False

            if 'search_modes_used' not in hybrid_data:
                self.test_results.append((test_name, False, 'Response missing search_modes_used field'))
                return False

            if 'fts_count' not in hybrid_data or 'semantic_count' not in hybrid_data:
                self.test_results.append((test_name, False, 'Response missing source counts'))
                return False

            hybrid_results = hybrid_data.get('results', [])
            if len(hybrid_results) < 1:
                self.test_results.append((test_name, False, 'No results from hybrid search'))
                return False

            # Test 2: Verify results have RRF scores structure
            first_result = hybrid_results[0]
            if 'scores' not in first_result:
                self.test_results.append((test_name, False, 'Result missing scores field'))
                return False

            scores = first_result.get('scores', {})
            if 'rrf' not in scores:
                self.test_results.append((test_name, False, 'Scores missing rrf field'))
                return False

            # Test 3: Test with specific search modes
            fts_only_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'database indexing',
                    'search_modes': ['fts'],
                    'thread_id': hybrid_thread,
                    'limit': 10,
                },
            )

            fts_only_data = self._extract_content(fts_only_result)

            # If FTS is available, verify search_modes_used reflects the request
            if has_fts:
                search_modes_used = fts_only_data.get('search_modes_used', [])
                if 'fts' not in search_modes_used:
                    self.test_results.append((test_name, False, 'FTS mode not used when requested'))
                    return False

            # Test 4: Test source filtering
            source_filter_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'python',
                    'thread_id': hybrid_thread,
                    'source': 'agent',
                    'limit': 10,
                },
            )

            source_data = self._extract_content(source_filter_result)
            if 'results' not in source_data:
                self.test_results.append((test_name, False, f'Source filter search failed: {source_data}'))
                return False

            # All our test entries are from 'agent', should find results
            source_results = source_data.get('results', [])
            # Verify all results have source='agent'
            for r in source_results:
                if r.get('source') != 'agent':
                    self.test_results.append(
                        (test_name, False, f"Expected source='agent', got '{r.get('source')}'"),
                    )
                    return False

            # Test 5: Verify fusion method in response
            if hybrid_data.get('fusion_method') != 'rrf':
                self.test_results.append(
                    (test_name, False, f"Expected fusion_method='rrf', got '{hybrid_data.get('fusion_method')}'"),
                )
                return False

            self.test_results.append((test_name, True, 'Hybrid search with RRF fusion working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_tools_content_type_filter(self) -> bool:
        """Test content_type parameter across all 4 search tools.

        Verifies that content_type='text' and content_type='multimodal' filters
        work correctly for search_context, semantic_search, fts_search, and hybrid_search.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'search_tools_content_type_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check feature availability
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            fts_info = stats_data.get('fts', {})

            has_semantic = semantic_info.get('enabled', False) and semantic_info.get('available', False)
            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_hybrid = (has_semantic or has_fts) and os.environ.get('ENABLE_HYBRID_SEARCH', '').lower() == 'true'

            # Create a separate thread for content_type tests
            ct_thread = f'{self.test_thread_id}_content_type'

            # Store text-only entries
            for i in range(2):
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': ct_thread,
                        'source': 'agent',
                        'text': f'Text-only content for content type filtering test {i}',
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store text context: {result_data}'))
                    return False

            # Store multimodal entries with images
            for i in range(2):
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': ct_thread,
                        'source': 'agent',
                        'text': f'Multimodal content with image for filtering test {i}',
                        'images': [{'data': self._create_test_image(), 'mime_type': 'image/png'}],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store multimodal context: {result_data}'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: search_context with content_type='text'
            text_result = await self.client.call_tool(
                'search_context',
                {'thread_id': ct_thread, 'content_type': 'text', 'limit': 10},
            )
            text_data = self._extract_content(text_result)
            if not text_data.get('success'):
                self.test_results.append((test_name, False, f'search_context text filter failed: {text_data}'))
                return False

            text_results = text_data.get('results', [])
            if len(text_results) != 2:
                self.test_results.append((test_name, False, f'Expected 2 text entries, got {len(text_results)}'))
                return False

            # Verify all results have content_type='text'
            for r in text_results:
                if r.get('content_type') != 'text':
                    ct = r.get('content_type')
                    self.test_results.append((test_name, False, f"Expected content_type='text', got '{ct}'"))
                    return False

            # Test 2: search_context with content_type='multimodal'
            mm_result = await self.client.call_tool(
                'search_context',
                {'thread_id': ct_thread, 'content_type': 'multimodal', 'limit': 10},
            )
            mm_data = self._extract_content(mm_result)
            if not mm_data.get('success'):
                self.test_results.append((test_name, False, f'search_context multimodal filter failed: {mm_data}'))
                return False

            mm_results = mm_data.get('results', [])
            if len(mm_results) != 2:
                self.test_results.append((test_name, False, f'Expected 2 multimodal entries, got {len(mm_results)}'))
                return False

            # Verify all results have content_type='multimodal'
            for r in mm_results:
                if r.get('content_type') != 'multimodal':
                    ct = r.get('content_type')
                    self.test_results.append((test_name, False, f"Expected content_type='multimodal', got '{ct}'"))
                    return False

            # Test 3: semantic_search with content_type filter (if available)
            if has_semantic:
                sem_text_result = await self.client.call_tool(
                    'semantic_search_context',
                    {'query': 'content filtering', 'thread_id': ct_thread, 'content_type': 'text', 'limit': 10},
                )
                sem_text_data = self._extract_content(sem_text_result)
                if 'results' not in sem_text_data:
                    self.test_results.append((test_name, False, f'semantic_search text filter failed: {sem_text_data}'))
                    return False

                # All results should be text type
                for r in sem_text_data.get('results', []):
                    if r.get('content_type') != 'text':
                        ct = r.get('content_type')
                        self.test_results.append((test_name, False, f"semantic: Expected 'text', got '{ct}'"))
                        return False

            # Test 4: fts_search with content_type filter (if available)
            if has_fts:
                fts_mm_result = await self.client.call_tool(
                    'fts_search_context',
                    {
                        'query': 'content',
                        'mode': 'match',
                        'thread_id': ct_thread,
                        'content_type': 'multimodal',
                        'limit': 10,
                    },
                )
                fts_mm_data = self._extract_content(fts_mm_result)
                if 'results' not in fts_mm_data:
                    self.test_results.append((test_name, False, f'fts multimodal filter failed: {fts_mm_data}'))
                    return False

                # All results should be multimodal type
                for r in fts_mm_data.get('results', []):
                    if r.get('content_type') != 'multimodal':
                        ct = r.get('content_type')
                        self.test_results.append((test_name, False, f"fts: Expected 'multimodal', got '{ct}'"))
                        return False

            # Test 5: hybrid_search with content_type filter (if available)
            if has_hybrid:
                hyb_text_result = await self.client.call_tool(
                    'hybrid_search_context',
                    {
                        'query': 'content filtering',
                        'thread_id': ct_thread,
                        'content_type': 'text',
                        'limit': 10,
                    },
                )
                hyb_text_data = self._extract_content(hyb_text_result)
                if 'results' not in hyb_text_data:
                    self.test_results.append((test_name, False, f'hybrid text filter failed: {hyb_text_data}'))
                    return False

                # All results should be text type
                for r in hyb_text_data.get('results', []):
                    if r.get('content_type') != 'text':
                        ct = r.get('content_type')
                        self.test_results.append((test_name, False, f"hybrid: Expected 'text', got '{ct}'"))
                        return False

            msg = f'content_type filter working (semantic={has_semantic}, fts={has_fts}, hybrid={has_hybrid})'
            self.test_results.append((test_name, True, msg))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_tools_include_images(self) -> bool:
        """Test include_images parameter across all 4 search tools.

        Verifies that include_images=True returns image data and include_images=False excludes it.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'search_tools_include_images'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check feature availability
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            fts_info = stats_data.get('fts', {})

            has_semantic = semantic_info.get('enabled', False) and semantic_info.get('available', False)
            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_hybrid = (has_semantic or has_fts) and os.environ.get('ENABLE_HYBRID_SEARCH', '').lower() == 'true'

            # Create a separate thread for include_images tests
            img_thread = f'{self.test_thread_id}_include_images'

            # Store multimodal entry with image
            result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': img_thread,
                    'source': 'agent',
                    'text': 'Multimodal content for include images test with Python code',
                    'images': [{'data': self._create_test_image(), 'mime_type': 'image/png'}],
                },
            )
            result_data = self._extract_content(result)
            if not result_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store multimodal context: {result_data}'))
                return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: search_context with include_images=True
            with_images_result = await self.client.call_tool(
                'search_context',
                {'thread_id': img_thread, 'include_images': True, 'limit': 10},
            )
            with_images_data = self._extract_content(with_images_result)
            if not with_images_data.get('success'):
                self.test_results.append((test_name, False, f'search_context include_images=True failed: {with_images_data}'))
                return False

            with_img_results = with_images_data.get('results', [])
            if len(with_img_results) < 1:
                self.test_results.append((test_name, False, 'No results found'))
                return False

            # Verify images are included
            first_result = with_img_results[0]
            images = first_result.get('images', [])
            if len(images) < 1:
                self.test_results.append((test_name, False, 'Expected images in result with include_images=True'))
                return False

            # Verify image has data
            if 'data' not in images[0] or not images[0]['data']:
                self.test_results.append((test_name, False, 'Image data missing with include_images=True'))
                return False

            # Test 2: search_context with include_images=False
            without_images_result = await self.client.call_tool(
                'search_context',
                {'thread_id': img_thread, 'include_images': False, 'limit': 10},
            )
            without_images_data = self._extract_content(without_images_result)
            if not without_images_data.get('success'):
                msg = f'search_context include_images=False failed: {without_images_data}'
                self.test_results.append((test_name, False, msg))
                return False

            without_img_results = without_images_data.get('results', [])
            if len(without_img_results) < 1:
                self.test_results.append((test_name, False, 'No results found with include_images=False'))
                return False

            # Verify images are excluded or empty
            first_wo_img = without_img_results[0]
            wo_images = first_wo_img.get('images', [])
            # Images should be empty list or not contain data
            if wo_images:
                for img in wo_images:
                    if img.get('data'):
                        self.test_results.append((test_name, False, 'Image data should be excluded with include_images=False'))
                        return False

            # Test 3: semantic_search with include_images (if available)
            if has_semantic:
                sem_result = await self.client.call_tool(
                    'semantic_search_context',
                    {
                        'query': 'multimodal content',
                        'thread_id': img_thread,
                        'include_images': True,
                        'limit': 10,
                    },
                )
                sem_data = self._extract_content(sem_result)
                if 'results' in sem_data and len(sem_data['results']) > 0:
                    sem_images = sem_data['results'][0].get('images', [])
                    if len(sem_images) < 1 or not sem_images[0].get('data'):
                        self.test_results.append((test_name, False, 'semantic: Expected images'))
                        return False

            # Test 4: fts_search with include_images (if available)
            if has_fts:
                fts_result = await self.client.call_tool(
                    'fts_search_context',
                    {
                        'query': 'multimodal',
                        'mode': 'match',
                        'thread_id': img_thread,
                        'include_images': True,
                        'limit': 10,
                    },
                )
                fts_data = self._extract_content(fts_result)
                if 'results' in fts_data and len(fts_data['results']) > 0:
                    fts_images = fts_data['results'][0].get('images', [])
                    if len(fts_images) < 1 or not fts_images[0].get('data'):
                        self.test_results.append((test_name, False, 'fts: Expected images'))
                        return False

            # Test 5: hybrid_search with include_images (if available)
            if has_hybrid:
                hyb_result = await self.client.call_tool(
                    'hybrid_search_context',
                    {
                        'query': 'multimodal content',
                        'thread_id': img_thread,
                        'include_images': True,
                        'limit': 10,
                    },
                )
                hyb_data = self._extract_content(hyb_result)
                if 'results' in hyb_data and len(hyb_data['results']) > 0:
                    hyb_images = hyb_data['results'][0].get('images', [])
                    if len(hyb_images) < 1 or not hyb_images[0].get('data'):
                        self.test_results.append((test_name, False, 'hybrid: Expected images'))
                        return False

            msg = f'include_images working (semantic={has_semantic}, fts={has_fts}, hybrid={has_hybrid})'
            self.test_results.append((test_name, True, msg))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_tools_tags_filter(self) -> bool:
        """Test tags parameter for semantic_search, fts_search, and hybrid_search.

        Note: search_context already tests tags. This tests the 3 other search tools.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'search_tools_tags_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check feature availability
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            fts_info = stats_data.get('fts', {})

            has_semantic = semantic_info.get('enabled', False) and semantic_info.get('available', False)
            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_hybrid = (has_semantic or has_fts) and os.environ.get('ENABLE_HYBRID_SEARCH', '').lower() == 'true'

            # Skip if no advanced search features are available
            if not has_semantic and not has_fts:
                self.test_results.append((test_name, True, 'Skipped (no advanced search available)'))
                return True

            # Create a separate thread for tags tests
            tags_thread = f'{self.test_thread_id}_tags_filter'

            # Store entries with different tags
            test_entries = [
                {'text': 'Python backend development with Flask', 'tags': ['backend', 'python']},
                {'text': 'JavaScript frontend development with React', 'tags': ['frontend', 'javascript']},
                {'text': 'Full stack development combining both', 'tags': ['fullstack', 'backend', 'frontend']},
                {'text': 'Database design and SQL optimization', 'tags': ['database', 'backend']},
            ]

            for entry in test_entries:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': tags_thread,
                        'source': 'agent',
                        'text': entry['text'],
                        'tags': entry['tags'],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store: {result_data}'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: semantic_search with tags filter (if available)
            if has_semantic:
                sem_result = await self.client.call_tool(
                    'semantic_search_context',
                    {
                        'query': 'development frameworks',
                        'thread_id': tags_thread,
                        'tags': ['backend'],
                        'limit': 10,
                    },
                )
                sem_data = self._extract_content(sem_result)
                if 'results' not in sem_data:
                    self.test_results.append((test_name, False, f'semantic tags failed: {sem_data}'))
                    return False

                sem_results = sem_data.get('results', [])
                # Should find entries with 'backend' tag (Python, Full stack, Database = 3)
                if len(sem_results) < 1:
                    self.test_results.append((test_name, False, 'semantic: No results with backend tag'))
                    return False

                # Verify all results have 'backend' tag
                for r in sem_results:
                    result_tags = r.get('tags', [])
                    if 'backend' not in result_tags:
                        self.test_results.append((test_name, False, f"semantic: Expected 'backend', got {result_tags}"))
                        return False

            # Test 2: fts_search with tags filter (if available)
            if has_fts:
                fts_result = await self.client.call_tool(
                    'fts_search_context',
                    {
                        'query': 'development',
                        'mode': 'match',
                        'thread_id': tags_thread,
                        'tags': ['frontend'],
                        'limit': 10,
                    },
                )
                fts_data = self._extract_content(fts_result)
                if 'results' not in fts_data:
                    self.test_results.append((test_name, False, f'fts tags failed: {fts_data}'))
                    return False

                fts_results = fts_data.get('results', [])
                # Should find entries with 'frontend' tag (JavaScript, Full stack = 2)
                if len(fts_results) < 1:
                    self.test_results.append((test_name, False, 'fts: No results with frontend tag'))
                    return False

                # Verify all results have 'frontend' tag
                for r in fts_results:
                    result_tags = r.get('tags', [])
                    if 'frontend' not in result_tags:
                        self.test_results.append((test_name, False, f"fts: Expected 'frontend', got {result_tags}"))
                        return False

            # Test 3: hybrid_search with tags filter (if available)
            if has_hybrid:
                hyb_result = await self.client.call_tool(
                    'hybrid_search_context',
                    {
                        'query': 'development',
                        'thread_id': tags_thread,
                        'tags': ['python'],
                        'limit': 10,
                    },
                )
                hyb_data = self._extract_content(hyb_result)
                if 'results' not in hyb_data:
                    self.test_results.append((test_name, False, f'hybrid tags failed: {hyb_data}'))
                    return False

                hyb_results = hyb_data.get('results', [])
                # Should find entries with 'python' tag (Python backend = 1)
                if len(hyb_results) < 1:
                    self.test_results.append((test_name, False, 'hybrid: No results with python tag'))
                    return False

                # Verify all results have 'python' tag
                for r in hyb_results:
                    result_tags = r.get('tags', [])
                    if 'python' not in result_tags:
                        self.test_results.append((test_name, False, f"hybrid: Expected 'python', got {result_tags}"))
                        return False

            # Test 4: Multiple tags (OR logic)
            if has_semantic:
                multi_tag_result = await self.client.call_tool(
                    'semantic_search_context',
                    {
                        'query': 'development',
                        'thread_id': tags_thread,
                        'tags': ['python', 'javascript'],
                        'limit': 10,
                    },
                )
                multi_tag_data = self._extract_content(multi_tag_result)
                if 'results' in multi_tag_data:
                    multi_results = multi_tag_data.get('results', [])
                    # Should find at least 2 entries (Python and JavaScript)
                    if len(multi_results) < 2:
                        msg = f'Expected 2+ results with python OR javascript, got {len(multi_results)}'
                        self.test_results.append((test_name, False, msg))
                        return False

            msg = f'tags filter working (semantic={has_semantic}, fts={has_fts}, hybrid={has_hybrid})'
            self.test_results.append((test_name, True, msg))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_semantic_search_offset_pagination(self) -> bool:
        """Test offset pagination for semantic_search_context.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'semantic_search_offset_pagination'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if semantic search is enabled
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            is_enabled = semantic_info.get('enabled', False)
            is_available = semantic_info.get('available', False)

            if not is_enabled or not is_available:
                self.test_results.append((test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'))
                return True

            # Create a separate thread for pagination tests
            page_thread = f'{self.test_thread_id}_semantic_offset'

            # Store 5 entries for pagination testing
            test_texts = [
                'First Python programming tutorial for beginners',
                'Second Python advanced programming concepts',
                'Third Python web development with Django',
                'Fourth Python data science and machine learning',
                'Fifth Python automation and scripting guide',
            ]

            stored_ids = []
            for text in test_texts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': page_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store context: {result_data}'))
                    return False
                stored_ids.append(result_data.get('context_id'))

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: First page (offset=0, limit=2)
            page1_result = await self.client.call_tool(
                'semantic_search_context',
                {'query': 'Python programming', 'thread_id': page_thread, 'offset': 0, 'limit': 2},
            )
            page1_data = self._extract_content(page1_result)
            if 'results' not in page1_data:
                self.test_results.append((test_name, False, f'Page 1 search failed: {page1_data}'))
                return False

            page1_results = page1_data.get('results', [])
            if len(page1_results) != 2:
                self.test_results.append((test_name, False, f'Expected 2 results for page 1, got {len(page1_results)}'))
                return False

            page1_ids = [r.get('id') for r in page1_results]

            # Test 2: Second page (offset=2, limit=2)
            page2_result = await self.client.call_tool(
                'semantic_search_context',
                {'query': 'Python programming', 'thread_id': page_thread, 'offset': 2, 'limit': 2},
            )
            page2_data = self._extract_content(page2_result)
            if 'results' not in page2_data:
                self.test_results.append((test_name, False, f'Page 2 search failed: {page2_data}'))
                return False

            page2_results = page2_data.get('results', [])
            if len(page2_results) != 2:
                self.test_results.append((test_name, False, f'Expected 2 results for page 2, got {len(page2_results)}'))
                return False

            page2_ids = [r.get('id') for r in page2_results]

            # Verify no overlap between pages
            overlap = set(page1_ids) & set(page2_ids)
            if overlap:
                self.test_results.append((test_name, False, f'Overlap found between pages: {overlap}'))
                return False

            # Test 3: Third page (offset=4, limit=2) - should get 1 result
            page3_result = await self.client.call_tool(
                'semantic_search_context',
                {'query': 'Python programming', 'thread_id': page_thread, 'offset': 4, 'limit': 2},
            )
            page3_data = self._extract_content(page3_result)
            if 'results' not in page3_data:
                self.test_results.append((test_name, False, f'Page 3 search failed: {page3_data}'))
                return False

            page3_results = page3_data.get('results', [])
            if len(page3_results) != 1:
                self.test_results.append((test_name, False, f'Expected 1 result for page 3, got {len(page3_results)}'))
                return False

            self.test_results.append((test_name, True, 'Offset pagination working for semantic_search'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_hybrid_search_metadata_filtering(self) -> bool:
        """Test metadata and metadata_filters parameters for hybrid_search_context.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'hybrid_search_metadata_filtering'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if hybrid search is available
            if os.environ.get('ENABLE_HYBRID_SEARCH', '').lower() != 'true':
                self.test_results.append((test_name, True, 'Skipped (ENABLE_HYBRID_SEARCH not enabled)'))
                return True

            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            semantic_info = stats_data.get('semantic_search', {})

            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_semantic = semantic_info.get('enabled', False) and semantic_info.get('available', False)

            if not has_fts and not has_semantic:
                self.test_results.append((test_name, True, f'Skipped (fts={has_fts}, semantic={has_semantic})'))
                return True

            # Create a separate thread for metadata tests
            meta_thread = f'{self.test_thread_id}_hybrid_metadata'

            # Store entries with different metadata
            test_entries = [
                {'text': 'High priority backend task for API development', 'metadata': {'priority': 9, 'category': 'backend'}},
                {'text': 'Low priority frontend task for UI updates', 'metadata': {'priority': 3, 'category': 'frontend'}},
                {'text': 'High priority database optimization task', 'metadata': {'priority': 8, 'category': 'backend'}},
                {'text': 'Medium priority testing task', 'metadata': {'priority': 5, 'category': 'testing'}},
            ]

            for entry in test_entries:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': meta_thread,
                        'source': 'agent',
                        'text': entry['text'],
                        'metadata': entry['metadata'],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store context: {result_data}'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: Simple metadata filter (category=backend)
            simple_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'development task',
                    'thread_id': meta_thread,
                    'metadata': {'category': 'backend'},
                    'limit': 10,
                },
            )
            simple_data = self._extract_content(simple_result)
            if 'results' not in simple_data:
                self.test_results.append((test_name, False, f'Simple metadata filter failed: {simple_data}'))
                return False

            simple_results = simple_data.get('results', [])
            # Should find 2 backend entries
            if len(simple_results) < 1:
                self.test_results.append((test_name, False, 'No results with category=backend'))
                return False

            # Helper to get metadata (may be dict or JSON string)
            def get_meta(result: dict[str, Any]) -> dict[str, Any]:
                meta = result.get('metadata', {})
                if isinstance(meta, str):
                    import json
                    try:
                        return json.loads(meta)
                    except (json.JSONDecodeError, TypeError):
                        return {}
                return meta if isinstance(meta, dict) else {}

            # Verify all results have category=backend
            for r in simple_results:
                meta = get_meta(r)
                if meta.get('category') != 'backend':
                    cat = meta.get('category')
                    self.test_results.append((test_name, False, f"Expected category='backend', got '{cat}'"))
                    return False

            # Test 2: Advanced metadata filter (priority > 5)
            adv_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'task',
                    'thread_id': meta_thread,
                    'metadata_filters': [{'key': 'priority', 'operator': 'gt', 'value': 5}],
                    'limit': 10,
                },
            )
            adv_data = self._extract_content(adv_result)
            if 'results' not in adv_data:
                self.test_results.append((test_name, False, f'Advanced metadata filter failed: {adv_data}'))
                return False

            adv_results = adv_data.get('results', [])
            # Should find entries with priority > 5 (9, 8 = 2)
            if len(adv_results) < 1:
                self.test_results.append((test_name, False, 'No results with priority > 5'))
                return False

            # Verify all results have priority > 5
            for r in adv_results:
                meta = get_meta(r)
                if meta.get('priority', 0) <= 5:
                    self.test_results.append((test_name, False, f"Expected priority > 5, got {meta.get('priority')}"))
                    return False

            # Test 3: Combined metadata filter (category=backend AND priority >= 8)
            combined_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'task',
                    'thread_id': meta_thread,
                    'metadata': {'category': 'backend'},
                    'metadata_filters': [{'key': 'priority', 'operator': 'gte', 'value': 8}],
                    'limit': 10,
                },
            )
            combined_data = self._extract_content(combined_result)
            if 'results' not in combined_data:
                self.test_results.append((test_name, False, f'Combined metadata filter failed: {combined_data}'))
                return False

            combined_results = combined_data.get('results', [])
            # Should find entries with category=backend AND priority >= 8 (9, 8 = 2)
            for r in combined_results:
                meta = get_meta(r)
                if meta.get('category') != 'backend' or meta.get('priority', 0) < 8:
                    self.test_results.append((test_name, False, f'Expected backend+priority>=8, got {meta}'))
                    return False

            self.test_results.append((test_name, True, 'metadata and metadata_filters working for hybrid_search'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_hybrid_search_date_range_filtering(self) -> bool:
        """Test start_date and end_date parameters for hybrid_search_context.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'hybrid_search_date_range_filtering'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if hybrid search is available
            if os.environ.get('ENABLE_HYBRID_SEARCH', '').lower() != 'true':
                self.test_results.append((test_name, True, 'Skipped (ENABLE_HYBRID_SEARCH not enabled)'))
                return True

            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            semantic_info = stats_data.get('semantic_search', {})

            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_semantic = semantic_info.get('enabled', False) and semantic_info.get('available', False)

            if not has_fts and not has_semantic:
                self.test_results.append((test_name, True, f'Skipped (fts={has_fts}, semantic={has_semantic})'))
                return True

            # Create a separate thread for date range tests
            date_thread = f'{self.test_thread_id}_hybrid_date'

            # Store test entries (will all have current timestamp)
            test_texts = [
                'Python machine learning algorithms for AI development',
                'Database query optimization techniques',
                'Frontend web development with modern frameworks',
            ]

            for text in test_texts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': date_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store context: {result_data}'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Get current date for testing
            from datetime import datetime
            from datetime import timedelta

            now = datetime.now(UTC)
            today = now.strftime('%Y-%m-%d')
            yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')
            tomorrow = (now + timedelta(days=1)).strftime('%Y-%m-%d')

            # Test 1: Filter with start_date (should find entries from today)
            start_date_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'development',
                    'thread_id': date_thread,
                    'start_date': today,
                    'limit': 10,
                },
            )
            start_date_data = self._extract_content(start_date_result)
            if 'results' not in start_date_data:
                self.test_results.append((test_name, False, f'start_date filter failed: {start_date_data}'))
                return False

            start_results = start_date_data.get('results', [])
            # Should find all entries (created today)
            if len(start_results) < 1:
                self.test_results.append((test_name, False, 'No results with start_date filter'))
                return False

            # Test 2: Filter with end_date (should find entries up to today)
            end_date_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'development',
                    'thread_id': date_thread,
                    'end_date': today,
                    'limit': 10,
                },
            )
            end_date_data = self._extract_content(end_date_result)
            if 'results' not in end_date_data:
                self.test_results.append((test_name, False, f'end_date filter failed: {end_date_data}'))
                return False

            end_results = end_date_data.get('results', [])
            if len(end_results) < 1:
                self.test_results.append((test_name, False, 'No results with end_date filter'))
                return False

            # Test 3: Filter with date range (yesterday to tomorrow)
            range_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'development',
                    'thread_id': date_thread,
                    'start_date': yesterday,
                    'end_date': tomorrow,
                    'limit': 10,
                },
            )
            range_data = self._extract_content(range_result)
            if 'results' not in range_data:
                self.test_results.append((test_name, False, f'Date range filter failed: {range_data}'))
                return False

            range_results = range_data.get('results', [])
            if len(range_results) < 1:
                self.test_results.append((test_name, False, 'No results with date range filter'))
                return False

            # Test 4: Filter with future start_date (should find no entries)
            future_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'development',
                    'thread_id': date_thread,
                    'start_date': tomorrow,
                    'limit': 10,
                },
            )
            future_data = self._extract_content(future_result)
            if 'results' not in future_data:
                self.test_results.append((test_name, False, f'Future date filter failed: {future_data}'))
                return False

            future_results = future_data.get('results', [])
            if len(future_results) > 0:
                self.test_results.append((test_name, False, f'Expected 0 results for future date, got {len(future_results)}'))
                return False

            self.test_results.append((test_name, True, 'start_date and end_date working for hybrid_search'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_hybrid_search_offset_pagination(self) -> bool:
        """Test offset pagination for hybrid_search_context.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'hybrid_search_offset_pagination'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if hybrid search is available
            if os.environ.get('ENABLE_HYBRID_SEARCH', '').lower() != 'true':
                self.test_results.append((test_name, True, 'Skipped (ENABLE_HYBRID_SEARCH not enabled)'))
                return True

            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            semantic_info = stats_data.get('semantic_search', {})

            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_semantic = semantic_info.get('enabled', False) and semantic_info.get('available', False)

            if not has_fts and not has_semantic:
                self.test_results.append((test_name, True, f'Skipped (fts={has_fts}, semantic={has_semantic})'))
                return True

            # Create a separate thread for pagination tests
            page_thread = f'{self.test_thread_id}_hybrid_offset'

            # Store 5 entries for pagination testing
            # NOTE: All entries MUST contain both 'Python' AND 'programming' because
            # FTS 'match' mode interprets "Python programming" as "Python AND programming"
            test_texts = [
                'First Python programming tutorial for beginners learning to code',
                'Second Python programming advanced concepts for software experts',
                'Third Python programming web development with Django framework',
                'Fourth Python programming data science and machine learning apps',
                'Fifth Python programming automation and scripting guide for DevOps',
            ]

            for text in test_texts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': page_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store context: {result_data}'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: First page (offset=0, limit=2)
            page1_result = await self.client.call_tool(
                'hybrid_search_context',
                {'query': 'Python programming', 'thread_id': page_thread, 'offset': 0, 'limit': 2},
            )
            page1_data = self._extract_content(page1_result)
            if 'results' not in page1_data:
                self.test_results.append((test_name, False, f'Page 1 search failed: {page1_data}'))
                return False

            page1_results = page1_data.get('results', [])
            if len(page1_results) != 2:
                self.test_results.append((test_name, False, f'Expected 2 results for page 1, got {len(page1_results)}'))
                return False

            page1_ids = [r.get('id') for r in page1_results]

            # Test 2: Second page (offset=2, limit=2)
            page2_result = await self.client.call_tool(
                'hybrid_search_context',
                {'query': 'Python programming', 'thread_id': page_thread, 'offset': 2, 'limit': 2},
            )
            page2_data = self._extract_content(page2_result)
            if 'results' not in page2_data:
                self.test_results.append((test_name, False, f'Page 2 search failed: {page2_data}'))
                return False

            page2_results = page2_data.get('results', [])
            if len(page2_results) != 2:
                self.test_results.append((test_name, False, f'Expected 2 results for page 2, got {len(page2_results)}'))
                return False

            page2_ids = [r.get('id') for r in page2_results]

            # Verify no overlap between pages
            overlap = set(page1_ids) & set(page2_ids)
            if overlap:
                self.test_results.append((test_name, False, f'Overlap found between pages: {overlap}'))
                return False

            # Test 3: Third page (offset=4, limit=2) - should get remaining results
            page3_result = await self.client.call_tool(
                'hybrid_search_context',
                {'query': 'Python programming', 'thread_id': page_thread, 'offset': 4, 'limit': 2},
            )
            page3_data = self._extract_content(page3_result)
            if 'results' not in page3_data:
                self.test_results.append((test_name, False, f'Page 3 search failed: {page3_data}'))
                return False

            page3_results = page3_data.get('results', [])
            if len(page3_results) != 1:
                self.test_results.append((test_name, False, f'Expected 1 result for page 3, got {len(page3_results)}'))
                return False

            page3_ids = [r.get('id') for r in page3_results]

            # Verify no overlap with previous pages
            overlap23 = set(page2_ids) & set(page3_ids)
            overlap13 = set(page1_ids) & set(page3_ids)
            if overlap23 or overlap13:
                self.test_results.append((test_name, False, f'Overlap found with page 3: {overlap23 | overlap13}'))
                return False

            # Verify pagination worked (different IDs across pages)
            all_ids = set(page1_ids) | set(page2_ids) | set(page3_ids)
            msg = f'Offset pagination working - {len(all_ids)} unique results across 3 pages'
            self.test_results.append((test_name, True, msg))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_explain_query_statistics(self) -> bool:
        """Test explain_query parameter for search_context, fts_search, and hybrid_search.

        Verifies that explain_query=True returns execution statistics.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'explain_query_statistics'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check feature availability
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_semantic = stats_data.get('semantic_search', {}).get('available', False)
            hybrid_enabled = os.environ.get('ENABLE_HYBRID_SEARCH', '').lower() == 'true'
            has_hybrid = (has_fts or has_semantic) and hybrid_enabled

            # Create a separate thread for explain_query tests
            explain_thread = f'{self.test_thread_id}_explain_query'

            # Store test entry
            result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': explain_thread,
                    'source': 'agent',
                    'text': 'Test content for explain query statistics verification',
                },
            )
            result_data = self._extract_content(result)
            if not result_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store context: {result_data}'))
                return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: search_context with explain_query=True
            search_result = await self.client.call_tool(
                'search_context',
                {'thread_id': explain_thread, 'explain_query': True, 'limit': 10},
            )
            search_data = self._extract_content(search_result)
            if not search_data.get('success'):
                self.test_results.append((test_name, False, f'search_context with explain_query failed: {search_data}'))
                return False

            # Verify stats are included
            if 'stats' not in search_data:
                self.test_results.append((test_name, False, 'search_context: Missing stats with explain_query=True'))
                return False

            search_stats = search_data.get('stats', {})
            if 'execution_time_ms' not in search_stats:
                self.test_results.append((test_name, False, 'search_context: Missing execution_time_ms in stats'))
                return False

            # Test 2: search_context with explain_query=False (default)
            no_explain_result = await self.client.call_tool(
                'search_context',
                {'thread_id': explain_thread, 'explain_query': False, 'limit': 10},
            )
            # With explain_query=False, stats should NOT be included
            no_explain_data = self._extract_content(no_explain_result)

            # Stats should NOT exist when explain_query=False
            if 'stats' in no_explain_data:
                self.test_results.append(
                    (test_name, False, 'search_context: stats should not be included when explain_query=False'),
                )
                return False

            # Test 3: fts_search with explain_query=True (if available)
            if has_fts:
                fts_result = await self.client.call_tool(
                    'fts_search_context',
                    {
                        'query': 'test content',
                        'mode': 'match',
                        'thread_id': explain_thread,
                        'explain_query': True,
                        'limit': 10,
                    },
                )
                fts_data = self._extract_content(fts_result)
                if 'results' not in fts_data:
                    self.test_results.append((test_name, False, f'fts_search with explain_query failed: {fts_data}'))
                    return False

                # Verify stats are included for FTS
                if 'stats' not in fts_data:
                    self.test_results.append((test_name, False, 'fts_search: Missing stats with explain_query=True'))
                    return False

                fts_stats = fts_data.get('stats', {})
                if 'execution_time_ms' not in fts_stats:
                    self.test_results.append((test_name, False, 'fts_search: Missing execution_time_ms in stats'))
                    return False

            # Test 4: hybrid_search with explain_query=True (if available)
            if has_hybrid:
                hybrid_result = await self.client.call_tool(
                    'hybrid_search_context',
                    {'query': 'test content', 'thread_id': explain_thread, 'explain_query': True, 'limit': 10},
                )
                hybrid_data = self._extract_content(hybrid_result)
                if 'results' not in hybrid_data:
                    self.test_results.append((test_name, False, f'hybrid_search with explain_query failed: {hybrid_data}'))
                    return False

                # Verify stats are included for hybrid
                if 'stats' not in hybrid_data:
                    self.test_results.append((test_name, False, 'hybrid_search: Missing stats with explain_query=True'))
                    return False

                hybrid_stats = hybrid_data.get('stats', {})
                if 'execution_time_ms' not in hybrid_stats:
                    self.test_results.append((test_name, False, 'hybrid_search: Missing execution_time_ms in stats'))
                    return False

            self.test_results.append((test_name, True, f'explain_query working (fts={has_fts}, hybrid={has_hybrid})'))
            return True

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
            if self.temp_db_path:
                async_temp_db_path = AsyncPath(self.temp_db_path)
                if await async_temp_db_path.exists():
                    try:
                        # Remove WAL and SHM files if they exist
                        wal_file = AsyncPath(str(self.temp_db_path) + '-wal')
                        shm_file = AsyncPath(str(self.temp_db_path) + '-shm')
                        if await wal_file.exists():
                            await wal_file.unlink()
                        if await shm_file.exists():
                            await shm_file.unlink()
                        # Remove main database file
                        await async_temp_db_path.unlink()
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
            ('Search Context Date Filtering', self.test_search_context_with_date_filtering),
            ('Metadata Filtering', self.test_metadata_filtering),
            ('Get Context by IDs', self.test_get_context_by_ids),
            ('Delete Context', self.test_delete_context),
            ('Update Context', self.test_update_context),
            ('Metadata Patch Deep Merge', self.test_metadata_patch_deep_merge),
            ('List Threads', self.test_list_threads),
            ('Get Statistics', self.test_get_statistics),
            ('Store Context Batch', self.test_store_context_batch),
            ('Update Context Batch', self.test_update_context_batch),
            ('Delete Context Batch', self.test_delete_context_batch),
            ('Semantic Search', self.test_semantic_search_context),
            ('Semantic Search Date Filtering', self.test_semantic_search_context_with_date_filtering),
            ('Semantic Search Metadata Filtering', self.test_semantic_search_context_with_metadata_filters),
            ('Search Context Invalid Filter Error', self.test_search_context_invalid_filter_returns_error),
            ('Semantic Search Invalid Filter Error', self.test_semantic_search_invalid_filter_returns_error),
            ('FTS Search', self.test_fts_search_context),
            ('FTS Search Invalid Filter Error', self.test_fts_search_invalid_filter_returns_error),
            ('FTS Boolean Mode', self.test_fts_boolean_mode),
            ('FTS Date Range Filter', self.test_fts_date_range_filter),
            ('FTS Metadata Filter', self.test_fts_metadata_filter),
            ('FTS Advanced Metadata Filters', self.test_fts_advanced_metadata_filters),
            ('FTS Pagination Offset', self.test_fts_pagination_offset),
            ('FTS Highlight Snippets', self.test_fts_highlight_snippets),
            ('Hybrid Search', self.test_hybrid_search_context),
            ('Search Tools Content Type Filter', self.test_search_tools_content_type_filter),
            ('Search Tools Include Images', self.test_search_tools_include_images),
            ('Search Tools Tags Filter', self.test_search_tools_tags_filter),
            ('Semantic Search Offset Pagination', self.test_semantic_search_offset_pagination),
            ('Hybrid Search Metadata Filtering', self.test_hybrid_search_metadata_filtering),
            ('Hybrid Search Date Range Filtering', self.test_hybrid_search_date_range_filtering),
            ('Hybrid Search Offset Pagination', self.test_hybrid_search_offset_pagination),
            ('Explain Query Statistics', self.test_explain_query_statistics),
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
            f'Test attempting to use default database!\nDefault: {default_db}\nTest DB: {temp_db}',
        )

    print(f'[TEST] Running with temp database: {temp_db}')
    print(f"[TEST] MCP_TEST_MODE: {os.environ.get('MCP_TEST_MODE')}")

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
            print(f"[INFO] MCP_TEST_MODE: {os.environ.get('MCP_TEST_MODE')}")

            test = MCPServerIntegrationTest(temp_db_path=temp_db_path)
            success = await test.run_all_tests()
            sys.exit(0 if success else 1)

    asyncio.run(main())
