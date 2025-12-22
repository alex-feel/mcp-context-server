"""
Pytest configuration and shared fixtures for MCP Context Storage Server tests.

Provides reusable test fixtures including database setup, server instances,
mock contexts, and sample test data for comprehensive testing.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
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

import pytest
import pytest_asyncio
from anyio import Path as AsyncPath
from dotenv import load_dotenv
from fastmcp import Context

import app.server
from app.backends import StorageBackend
from app.backends import create_backend
from app.settings import AppSettings

# ============================================================================
# Conditional Skip Helpers for Optional Dependencies
# ============================================================================


def is_ollama_available() -> bool:
    """Check if ollama package is installed."""
    return importlib.util.find_spec('ollama') is not None


def is_sqlite_vec_available() -> bool:
    """Check if sqlite-vec package is installed."""
    return importlib.util.find_spec('sqlite_vec') is not None


def is_numpy_available() -> bool:
    """Check if numpy package is installed."""
    return importlib.util.find_spec('numpy') is not None


def are_semantic_search_deps_available() -> bool:
    """Check if all semantic search dependencies are available."""
    return is_ollama_available() and is_sqlite_vec_available() and is_numpy_available()


# Pytest markers for conditional skipping
requires_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason='ollama package not installed',
)

requires_sqlite_vec = pytest.mark.skipif(
    not is_sqlite_vec_available(),
    reason='sqlite-vec package not installed',
)

requires_numpy = pytest.mark.skipif(
    not is_numpy_available(),
    reason='numpy package not installed',
)

requires_semantic_search = pytest.mark.skipif(
    not are_semantic_search_deps_available(),
    reason='Semantic search dependencies not available (ollama, sqlite_vec, numpy)',
)


def is_fts_enabled() -> bool:
    """Check if FTS is enabled in environment."""
    from app.settings import get_settings
    return get_settings().enable_fts


requires_fts = pytest.mark.skipif(
    not is_fts_enabled(),
    reason='FTS is not enabled (ENABLE_FTS=true not set)',
)

requires_hybrid_search = pytest.mark.skipif(
    not (is_fts_enabled() or are_semantic_search_deps_available()),
    reason='Neither FTS nor semantic search is available for hybrid search',
)

# Load .env file to make environment variables available for PostgreSQL tests
load_dotenv()

# CRITICAL: Force SQLite backend for ALL tests (override .env STORAGE_BACKEND=postgresql)
# PostgreSQL tests will explicitly create PostgreSQL backends when needed
os.environ['STORAGE_BACKEND'] = 'sqlite'


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
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
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
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')

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
        'metadata': {'key': 'value', 'priority': 10},
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
    """Insert multiple test context entries and return their IDs.

    Returns:
        list[int]: List of context entry IDs created in the database
    """
    cursor = test_db.cursor()
    entries = [
        ('thread_1', 'user', 'text', 'First test entry', None),
        ('thread_1', 'agent', 'text', 'Response to first', None),
        ('thread_2', 'user', 'multimodal', 'Second thread entry', json.dumps({'key': 'value'})),
        ('thread_2', 'agent', 'text', 'Agent analysis', None),
        ('thread_3', 'user', 'text', 'Third thread start', json.dumps({'priority': 1})),
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
    # Initialize the database schema synchronously before patching
    if not temp_db_path.exists():
        temp_db_path.parent.mkdir(parents=True, exist_ok=True)
        # Create database with schema
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        conn = sqlite3.connect(str(temp_db_path))
        try:
            conn.executescript(schema_sql)
            conn.execute('PRAGMA foreign_keys = ON')
            conn.execute('PRAGMA journal_mode = WAL')
            conn.commit()
        finally:
            conn.close()

    # Store original STORAGE_BACKEND environment variable
    original_storage_backend = os.environ.get('STORAGE_BACKEND')

    try:
        # Force SQLite backend for tests (override .env setting)
        os.environ['STORAGE_BACKEND'] = 'sqlite'

        with (
            patch('app.server.get_settings', return_value=test_settings),
            # CRITICAL: Patch factory.get_settings to prevent lazy backend creation from reading environment
            patch('app.backends.factory.get_settings', return_value=test_settings),
            patch('app.server.DB_PATH', temp_db_path),
            patch('app.server.MAX_IMAGE_SIZE_MB', test_settings.storage.max_image_size_mb),
            patch('app.server.MAX_TOTAL_SIZE_MB', test_settings.storage.max_total_size_mb),
        ):
            yield
    finally:
        # Restore original STORAGE_BACKEND
        if original_storage_backend is None:
            os.environ.pop('STORAGE_BACKEND', None)
        else:
            os.environ['STORAGE_BACKEND'] = original_storage_backend


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
async def async_db_initialized(temp_db_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Initialize async database for all tests."""
    from app.repositories import RepositoryContainer

    # Create storage backend
    backend = create_backend(backend_type='sqlite', db_path=str(temp_db_path))
    await backend.initialize()

    # Set in server module
    app.server._backend = backend
    app.server.DB_PATH = temp_db_path

    # Initialize repositories
    app.server._repositories = RepositoryContainer(backend)

    # Initialize the database schema using the backend
    # NOTE: We initialize schema directly instead of calling init_database() to avoid
    # reading STORAGE_BACKEND from environment (user may have postgresql in .env)
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')

    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(schema_sql)

    await backend.execute_write(_init_schema)

    try:
        yield backend
    finally:
        # Proper cleanup
        if hasattr(app.server, '_backend') and app.server._backend is not None:
            try:
                # Shutdown the storage backend
                await app.server._backend.shutdown()
            except Exception as e:
                # Log error but continue cleanup to prevent test suite hang
                import logging
                logging.getLogger(__name__).error(f'Error during backend shutdown: {e}')
            finally:
                # Always clear the reference, even if shutdown failed
                app.server._backend = None

        # Reset repositories
        if hasattr(app.server, '_repositories'):
            app.server._repositories = None


@pytest_asyncio.fixture
async def initialized_server(mock_server_dependencies: None, temp_db_path: Path) -> AsyncGenerator[None, None]:
    """Initialize server with test database and proper async cleanup.

    Note: mock_server_dependencies fixture is required to patch server settings,
    even though it's not directly used in the function body.

    Yields:
        None: Yields control after initialization, performs cleanup on teardown
    """
    from app.repositories import RepositoryContainer

    # CRITICAL: Aggressive pre-cleanup to prevent interference from previous tests
    # Shut down any existing backend from previous tests
    if hasattr(app.server, '_backend') and app.server._backend is not None:
        try:
            await app.server._backend.shutdown()
        except Exception:
            pass
        finally:
            app.server._backend = None

    # Reset repositories
    if hasattr(app.server, '_repositories'):
        app.server._repositories = None

    # Small delay to let background tasks fully terminate
    await asyncio.sleep(0.05)

    # Remove existing database if it exists (DB_PATH is patched by mock_server_dependencies)
    async_temp_db_path = AsyncPath(temp_db_path)
    if await async_temp_db_path.exists():
        await async_temp_db_path.unlink()

    # Create persistent backend before yielding (prevents lazy initialization in tests)
    # NOTE: We create SQLite backend directly instead of calling init_database() to avoid
    # reading STORAGE_BACKEND from environment (user may have postgresql in .env)
    backend = create_backend(backend_type='sqlite', db_path=str(temp_db_path))
    await backend.initialize()
    app.server._backend = backend

    # Initialize repositories with the backend
    app.server._repositories = RepositoryContainer(backend)

    # Initialize the database schema using the backend
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')

    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(schema_sql)

    await backend.execute_write(_init_schema)

    # Ensure the fixture dependency is satisfied
    assert mock_server_dependencies is None

    try:
        yield
    finally:
        # Proper async cleanup with timeout protection
        if hasattr(app.server, '_backend') and app.server._backend is not None:
            try:
                # Shutdown the storage backend with timeout to prevent hangs
                await asyncio.wait_for(
                    app.server._backend.shutdown(),
                    timeout=5.0,
                )
            except TimeoutError:
                # Log timeout but continue cleanup to prevent test suite hang
                import logging
                logging.getLogger(__name__).warning('Backend shutdown timed out after 5 seconds')
            except Exception as e:
                # Log error but continue cleanup to prevent test suite hang
                import logging
                logging.getLogger(__name__).error(f'Error during backend shutdown: {e}')
            finally:
                # Always clear the reference, even if shutdown failed
                app.server._backend = None

        # Reset the repositories to ensure clean state
        if hasattr(app.server, '_repositories'):
            app.server._repositories = None


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


@pytest_asyncio.fixture
async def async_db_with_embeddings(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Initialize async database with semantic search migration applied.

    This fixture creates a database with the full schema AND the semantic search
    migration (vec_context_embeddings table). Use this fixture for tests that
    require the EmbeddingRepository.

    Yields:
        StorageBackend: Initialized SQLite backend with semantic search tables.
    """
    from app.repositories import RepositoryContainer
    from app.schemas import load_schema
    from app.settings import get_settings

    settings = get_settings()
    db_path = tmp_path / 'test_context.db'

    # Create storage backend
    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    # Set in server module
    app.server._backend = backend
    app.server.DB_PATH = db_path

    # Initialize repositories
    app.server._repositories = RepositoryContainer(backend)

    # Initialize the base database schema
    schema_sql = load_schema('sqlite')

    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(schema_sql)

    await backend.execute_write(_init_schema)

    # Apply semantic search migration with correct dimension
    migration_path = Path(__file__).parent.parent / 'app' / 'migrations' / 'add_semantic_search_sqlite.sql'
    migration_sql = migration_path.read_text(encoding='utf-8')
    # Replace the template with actual dimension
    migration_sql = migration_sql.replace('{EMBEDDING_DIM}', str(settings.embedding_dim))

    def _apply_migration(conn: sqlite3.Connection) -> None:
        # Load sqlite-vec extension before executing migration
        # The migration SQL uses vec0 module which requires sqlite-vec extension
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except ImportError as e:
            raise RuntimeError(
                'sqlite_vec package is required for embedding tests. '
                'Install with: uv sync --extra semantic-search',
            ) from e

        conn.executescript(migration_sql)

    await backend.execute_write(_apply_migration)

    try:
        yield backend
    finally:
        # Proper cleanup
        if hasattr(app.server, '_backend') and app.server._backend is not None:
            try:
                await app.server._backend.shutdown()
            except Exception:
                pass
            finally:
                app.server._backend = None

        if hasattr(app.server, '_repositories'):
            app.server._repositories = None


@pytest.fixture
def embedding_dim() -> int:
    """Provide embedding dimension from settings.

    The configured embedding dimension defaults to 768 but can be overridden
    via EMBEDDING_DIM environment variable (e.g., 384 in CI).

    Returns:
        int: The embedding dimension from settings.
    """
    from app.settings import get_settings
    return get_settings().embedding_dim
