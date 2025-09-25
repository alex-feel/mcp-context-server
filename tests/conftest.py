"""
Pytest configuration and shared fixtures for MCP Context Storage Server tests.

Provides reusable test fixtures including database setup, server instances,
mock contexts, and sample test data for comprehensive testing.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
import tempfile
from collections.abc import AsyncGenerator
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import app.server
import pytest
import pytest_asyncio
from app.db_manager import DatabaseConnectionManager
from app.settings import AppSettings
from fastmcp import Context


# Global fixture to ensure NO test uses the default database
@pytest.fixture(autouse=True, scope='session')
def prevent_default_db_pollution():
    """
    Prevents ALL tests from using the default database.

    This fixture runs automatically for ALL tests in the session and ensures:
    1. DB_PATH is set to a temporary location
    2. MCP_TEST_MODE is enabled to indicate testing
    3. Default database path is NEVER used

    Raises:
        RuntimeError: If configuration attempts to use the default database.
    """
    # Store original environment
    original_db_path = os.environ.get('DB_PATH')
    original_test_mode = os.environ.get('MCP_TEST_MODE')

    # Create a session-wide temp directory
    with tempfile.TemporaryDirectory(prefix='mcp_test_session_') as temp_dir:
        # Set test environment variables
        temp_db_path = Path(temp_dir) / 'test_session.db'
        os.environ['DB_PATH'] = str(temp_db_path)
        os.environ['MCP_TEST_MODE'] = '1'

        # Verify we're not using default database
        default_db = Path.home() / '.mcp' / 'context_storage.db'
        if temp_db_path.resolve() == default_db.resolve():
            raise RuntimeError(
                f'CRITICAL: Test configuration error - attempting to use default database!\n'
                f'Default: {default_db}\n'
                f'Current: {temp_db_path}',
            )

        print(f'\n[TEST SAFETY] Session-wide temp DB: {temp_db_path}')
        print('[TEST SAFETY] MCP_TEST_MODE enabled')
        print(f'[TEST SAFETY] Default DB protected: {default_db}\n')

        try:
            yield
        finally:
            # Restore original environment
            if original_db_path is None:
                os.environ.pop('DB_PATH', None)
            else:
                os.environ['DB_PATH'] = original_db_path

            if original_test_mode is None:
                os.environ.pop('MCP_TEST_MODE', None)
            else:
                os.environ['MCP_TEST_MODE'] = original_test_mode


# Test configuration
@pytest.fixture
def test_settings(tmp_path: Path) -> AppSettings:
    """Create test settings with temporary database path."""
    # Create settings with correct storage configuration
    # Use temporary_env_vars context manager to set environment variables
    with temporary_env_vars(
        MAX_IMAGE_SIZE_MB='5',
        MAX_TOTAL_SIZE_MB='20',
        DB_PATH=str(tmp_path / 'test_context.db'),
        LOG_LEVEL='DEBUG',
    ):
        # AppSettings will automatically create StorageSettings
        # with the environment variables
        return AppSettings()


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provide temporary database path."""
    db_path = tmp_path / 'test_context.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@pytest.fixture
def test_db(temp_db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Create and initialize a test database."""
    # Create connection with increased timeout
    conn = sqlite3.connect(str(temp_db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Check if tables already exist (from initialized_server fixture)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='context_entries'")
    if not cursor.fetchone():
        # Tables don't exist, create them
        schema_path = Path(__file__).parent.parent / 'app' / 'schema.sql'
        schema_sql = schema_path.read_text(encoding='utf-8')
        conn.executescript(schema_sql)

    # Apply optimizations
    conn.execute('PRAGMA foreign_keys = ON')
    conn.execute('PRAGMA journal_mode = WAL')
    conn.execute('PRAGMA synchronous = NORMAL')
    conn.execute('PRAGMA temp_store = MEMORY')
    conn.execute('PRAGMA busy_timeout = 30000')  # 30 second busy timeout

    yield conn
    conn.close()


@pytest_asyncio.fixture
async def async_test_db(temp_db_path: Path) -> AsyncGenerator[sqlite3.Connection, None]:
    """Create async test database connection."""
    loop = asyncio.get_event_loop()

    def _create_db():
        schema_path = Path(__file__).parent.parent / 'app' / 'schema.sql'
        schema_sql = schema_path.read_text(encoding='utf-8')

        # Use check_same_thread=False for async tests to avoid thread safety issues
        conn = sqlite3.connect(str(temp_db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.executescript(schema_sql)

        # Apply optimizations
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA synchronous = NORMAL')
        conn.execute('PRAGMA temp_store = MEMORY')
        conn.execute('PRAGMA busy_timeout = 30000')  # 30 second busy timeout
        return conn

    conn = await loop.run_in_executor(None, _create_db)
    yield conn
    await loop.run_in_executor(None, conn.close)


@pytest.fixture
def mock_context() -> Context:
    """Create a mock MCP context for testing."""
    ctx = MagicMock(spec=Context)
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.error = AsyncMock()
    return ctx


@pytest.fixture
def sample_image_data() -> dict[str, str]:
    """Generate sample base64 encoded image data."""
    # Create a simple 1x1 PNG image
    png_data = bytes([
        0x89,
        0x50,
        0x4E,
        0x47,
        0x0D,
        0x0A,
        0x1A,
        0x0A,  # PNG signature
        0x00,
        0x00,
        0x00,
        0x0D,
        0x49,
        0x48,
        0x44,
        0x52,  # IHDR chunk
        0x00,
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x01,  # 1x1 dimensions
        0x08,
        0x02,
        0x00,
        0x00,
        0x00,
        0x90,
        0x77,
        0x53,  # Color type, etc
        0xDE,
        0x00,
        0x00,
        0x00,
        0x0C,
        0x49,
        0x44,
        0x41,  # IDAT chunk
        0x54,
        0x08,
        0x99,
        0x01,
        0x01,
        0x00,
        0x00,
        0x00,
        0x01,
        0x00,
        0x01,
        0x7B,
        0xDB,
        0x56,
        0x61,
        0x00,  # Image data
        0x00,
        0x00,
        0x00,
        0x49,
        0x45,
        0x4E,
        0x44,
        0xAE,  # IEND chunk
        0x42,
        0x60,
        0x82,
    ])
    return {
        'data': base64.b64encode(png_data).decode('utf-8'),
        'mime_type': 'image/png',
    }


@pytest.fixture
def sample_context_data() -> dict[str, Any]:
    """Generate sample context entry data."""
    return {
        'thread_id': 'test_thread_123',
        'source': 'user',
        'text': 'This is a test context entry',
        'metadata': {'key': 'value', 'priority': 'high'},
        'tags': ['test', 'sample', 'fixture'],
    }


@pytest.fixture
def sample_multimodal_data(sample_image_data: dict[str, str]) -> dict[str, Any]:
    """Generate sample multimodal context data."""
    return {
        'thread_id': 'test_multimodal_456',
        'source': 'agent',
        'text': 'Analysis of the attached image',
        'images': [sample_image_data],
        'metadata': {'analysis_type': 'visual'},
        'tags': ['image', 'analysis'],
    }


@pytest.fixture
def multiple_context_entries(test_db: sqlite3.Connection) -> list[int]:
    """Insert multiple test context entries and return their IDs."""
    cursor = test_db.cursor()
    entries = [
        ('thread_1', 'user', 'text', 'First test entry', None),
        ('thread_1', 'agent', 'text', 'Response to first', None),
        ('thread_2', 'user', 'multimodal', 'Second thread entry', json.dumps({'key': 'value'})),
        ('thread_2', 'agent', 'text', 'Agent analysis', None),
        ('thread_3', 'user', 'text', 'Third thread start', json.dumps({'priority': 'low'})),
    ]

    ids = []
    for thread_id, source, content_type, text, metadata in entries:
        cursor.execute(
            '''
            INSERT INTO context_entries
            (thread_id, source, content_type, text_content, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''',
            (thread_id, source, content_type, text, metadata),
        )
        ids.append(cursor.lastrowid)

    # Add tags to some entries
    tags_data = [
        (ids[0], 'important'),
        (ids[0], 'user-input'),
        (ids[1], 'response'),
        (ids[2], 'analysis'),
        (ids[3], 'ai-generated'),
    ]
    for entry_id, tag in tags_data:
        cursor.execute('INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)', (entry_id, tag))

    test_db.commit()
    valid_ids: list[int] = [id_ for id_ in ids if id_ is not None]
    return valid_ids


@pytest.fixture
def mock_server_dependencies(test_settings: AppSettings, temp_db_path: Path) -> Generator[None, None, None]:
    """Mock server dependencies for unit testing."""
    with (
        patch('app.server.get_settings', return_value=test_settings),
        patch('app.server.DB_PATH', temp_db_path),
        patch('app.server.MAX_IMAGE_SIZE_MB', test_settings.storage.max_image_size_mb),
        patch('app.server.MAX_TOTAL_SIZE_MB', test_settings.storage.max_total_size_mb),
    ):
        yield


@pytest.fixture
def large_image_data() -> dict[str, str]:
    """Generate a large image that exceeds size limits."""
    # Create 10MB of data (exceeds default 5MB limit in test settings)
    large_data = b'x' * (10 * 1024 * 1024)
    return {
        'data': base64.b64encode(large_data).decode('utf-8'),
        'mime_type': 'image/png',
    }


@pytest_asyncio.fixture
async def async_db_initialized(temp_db_path: Path) -> AsyncGenerator[DatabaseConnectionManager, None]:
    """Initialize async database for all tests."""
    from app.db_manager import get_connection_manager
    from app.db_manager import reset_connection_manager
    from app.repositories import RepositoryContainer

    # Reset and create new manager
    reset_connection_manager()
    manager = get_connection_manager(str(temp_db_path), force_new=True)
    await manager.initialize()

    # Set in server module
    app.server._db_manager = manager
    app.server.DB_PATH = temp_db_path

    # Initialize repositories
    app.server._repositories = RepositoryContainer(manager)

    # Initialize schema
    await app.server.init_database()

    try:
        yield manager
    finally:
        # Proper cleanup
        if app.server._db_manager is not None:
            await app.server._db_manager.shutdown()
            await app.server._db_manager._shutdown_complete.wait()
            app.server._db_manager = None
        # Reset repositories
        if hasattr(app.server, '_repositories'):
            app.server._repositories = None
        reset_connection_manager()


@pytest_asyncio.fixture
async def initialized_server(mock_server_dependencies: None, temp_db_path: Path) -> AsyncGenerator[None, None]:
    """Initialize server with test database and proper async cleanup.

    Note: mock_server_dependencies fixture is required to patch server settings,
    even though it's not directly used in the function body.

    Yields:
        None: Yields control after initialization, performs cleanup on teardown
    """
    from app.server import init_database

    # Remove existing database if it exists (DB_PATH is patched by mock_server_dependencies)
    if temp_db_path.exists():
        temp_db_path.unlink()

    # Initialize fresh database - now properly await the async function
    await init_database()

    # Ensure the fixture dependency is satisfied
    assert mock_server_dependencies is None

    try:
        yield
    finally:
        # Proper async cleanup with shutdown complete wait
        if hasattr(app.server, '_db_manager') and app.server._db_manager:
            await app.server._db_manager.shutdown()
            # Wait for shutdown to complete
            await app.server._db_manager._shutdown_complete.wait()
            app.server._db_manager = None

        # Reset the repositories to ensure clean state
        if hasattr(app.server, '_repositories'):
            app.server._repositories = None

        # Reset the singleton for next test
        from app.db_manager import reset_connection_manager
        reset_connection_manager()


@contextmanager
def temporary_env_vars(**kwargs):
    """Context manager for temporarily setting environment variables."""

    old_values = {}
    for key, value in kwargs.items():
        old_values[key] = os.environ.get(key)
        if value is not None:
            os.environ[key] = str(value)
        elif key in os.environ:
            del os.environ[key]
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
