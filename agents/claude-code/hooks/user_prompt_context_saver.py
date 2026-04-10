#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "fastmcp>=2.10.5",
#   "pyyaml",
# ]
# ///
"""
User Prompt Context Saver Hook for Claude Code.

This hook captures user prompts from UserPromptSubmit events and stores them
in the mcp-context-server for enhanced conversation context management.

Note: Currently, the UserPromptSubmit event does not provide images from user
requests, so this hook cannot save image content to the context server. Only
text prompts are captured and stored.

Trigger: UserPromptSubmit
"""

import asyncio
import ctypes
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Coroutine
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import cast

from fastmcp import Client


def _load_config_loader() -> ModuleType:
    """Dynamically load hook_config_loader from the same directory."""
    loader_path = Path(__file__).parent / 'hook_config_loader.py'
    spec = importlib.util.spec_from_file_location('hook_config_loader', loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load hook_config_loader from {loader_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Check if logging is enabled (module-level check, executed once)
_LOGGING_ENABLED = os.environ.get('CLAUDE_HOOK_DEBUG_ENABLED', '').lower() in ('1', 'true', 'yes')

# Message size limits for Windows subprocess pipe buffers
# Windows pipes typically have 64KB buffer, use conservative limits
_MAX_MESSAGE_SIZE = int(os.environ.get('CLAUDE_HOOK_MAX_MESSAGE_SIZE', '32768'))  # 32KB default
_CHUNK_SIZE = int(os.environ.get('CLAUDE_HOOK_CHUNK_SIZE', '30000'))  # 30KB default for chunks
_JSON_OVERHEAD = 500  # Estimated bytes for JSON structure (thread_id, source, etc.)

# Default configuration - used when no config file provided
# Maintains backward compatibility with original behavior
DEFAULT_CONFIG: dict[str, Any] = {
    'enabled': True,
    'output_context_id': True,  # Output stored context_id via hookSpecificOutput.additionalContext
    'skip_patterns': [],  # Regex patterns to skip (for internal hook prompts)
    'prebuilt_commands': [
        'add-dir',
        'agents',
        'bug',
        'clear',
        'compact',
        'config',
        'cost',
        'doctor',
        'help',
        'init',
        'login',
        'logout',
        'mcp',
        'memory',
        'model',
        'permissions',
        'pr_comments',
        'review',
        'status',
        'terminal-setup',
        'vim',
    ],
    'message_limits': {
        'max_message_size': 32768,
        'chunk_size': 30000,
        'json_overhead': 500,
    },
    'thread_id': {
        'max_retries': 3,
    },
    'mcp_client': {
        'max_retries': 3,
        'max_connection_retries': 3,  # Retry attempts for actual MCP connection
        'connection_timeout': 30.0,  # Timeout per connection attempt (seconds)
        'timeout_first_run': 240.0,
        'timeout_normal': 240.0,
    },
    'mcp_server': {
        # Transport type: 'stdio' (default, existing behavior) or 'http'
        'transport': 'stdio',
        # For stdio transport (existing behavior)
        'command': 'uvx',
        'python_version': '3.12',
        'package': 'mcp-context-server[embeddings-ollama]<2.0.0',
        'entry_point': 'mcp-context-server',
        'prewarm_cache': True,  # Pre-warm uvx cache at module load
        # For http transport (used when transport: http)
        # 'url': 'https://mcp-context-server.example.com/mcp',
        # 'headers': {},  # Optional custom headers for authentication
        # 'timeout': 30.0,  # Request timeout in seconds
    },
    'chunking': {
        'max_chunk_retries': 3,  # Retries per chunk within single connection
        'chunk_retry_delay': 0.5,  # Initial delay between chunk retries (seconds)
        'fail_mode': 'warn',  # 'silent', 'warn', or 'error'
    },
}


def _get_log_file() -> Path:
    """
    Get log file location with multiple fallbacks and diagnostic reporting.

    Fallback chain:
    1. CLAUDE_HOOK_DEBUG_FILE environment variable
    2. {CLAUDE_PROJECT_DIR}/.claude/.hook_debug.log
    3. {HOME}/.claude/hook_logs/user_prompt_context_saver.log
    4. {TEMP}/claude_hook_user_prompt_context_saver.log

    Returns:
        Path to the log file (guaranteed to return a valid path)
    """
    import sys
    from contextlib import suppress

    def _diagnostic(msg: str) -> None:
        """Write diagnostic to stderr (unconditionally)."""
        with suppress(Exception):
            print(f'[LOG PATH DIAGNOSTIC] {msg}', file=sys.stderr, flush=True)

    # Fallback 1: Explicit debug file location
    debug_file = os.environ.get('CLAUDE_HOOK_DEBUG_FILE')
    if debug_file:
        _diagnostic(f'Using CLAUDE_HOOK_DEBUG_FILE: {debug_file}')
        return Path(debug_file)

    # Fallback 2: Project directory
    project_dir = os.environ.get('CLAUDE_PROJECT_DIR')
    if project_dir:
        claude_dir = Path(project_dir) / '.claude'
        try:
            claude_dir.mkdir(parents=True, exist_ok=True)
            log_path = claude_dir / '.hook_debug.log'
            _diagnostic(f'Using project dir: {log_path}')
            return log_path
        except Exception as e:
            _diagnostic(f'Project dir fallback failed: {e}')

    # Fallback 3: User home directory
    try:
        home = Path.home()
        log_dir = home / '.claude' / 'hook_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'user_prompt_context_saver.log'
        _diagnostic(f'Using home dir: {log_path}')
        return log_path
    except Exception as e:
        _diagnostic(f'Home dir fallback failed: {e}')

    # Fallback 4: System temp directory (always works)
    temp_dir = Path(tempfile.gettempdir())
    log_path = temp_dir / 'claude_hook_user_prompt_context_saver.log'
    _diagnostic(f'Using temp dir: {log_path}')
    return log_path


# Initialize log file IMMEDIATELY
_LOG_FILE = _get_log_file()


def log_always(message: str, level: str = 'INFO') -> None:
    """
    Log message with guaranteed write when logging is enabled (never raises exceptions).

    Logging is CONDITIONAL based on CLAUDE_HOOK_DEBUG_ENABLED environment variable.
    If not set or set to values other than "1", "true", "yes" → NO logs written.

    This function provides conditional logging that:
    - Only writes logs when CLAUDE_HOOK_DEBUG_ENABLED is set to "1", "true", or "yes"
    - Never depends on CLAUDE_PROJECT_DIR environment variable
    - Never breaks the hook (all exceptions caught silently)
    - Uses multiple fallback locations for reliability when enabled
    - Provides timestamp and log level for each message

    Args:
        message: The message to log
        level: Log level (INFO, ERROR, DEBUG, etc.)
    """
    # Early exit if logging not enabled
    if not _LOGGING_ENABLED:
        return

    try:
        with _LOG_FILE.open('a', encoding='utf-8') as f:
            timestamp = datetime.now(tz=UTC).isoformat()
            f.write(f'{timestamp} [{level}] {message}\n')
    except Exception:
        # Even logging failures are silent - never break the hook
        pass


# Log script start IMMEDIATELY
log_always('=' * 80)
log_always('SCRIPT START')
log_always(f'sys.argv: {sys.argv}')
log_always(f'cwd: {os.getcwd()}')
log_always(f'Python version: {sys.version}')
log_always(f'Python executable: {sys.executable}')
log_always(f"CLAUDE_PROJECT_DIR: {os.environ.get('CLAUDE_PROJECT_DIR', 'NOT SET')}")
log_always(f"CLAUDE_HOOK_DEBUG_FILE: {os.environ.get('CLAUDE_HOOK_DEBUG_FILE', 'NOT SET')}")
log_always(f'Log file location: {_LOG_FILE}')
log_always(f'stdin isatty: {sys.stdin.isatty()}')

log_always('FastMCP Client imported successfully')


def _warmup_uvx_cache() -> None:
    """
    Pre-warm uvx cache to avoid cold start delays.

    Runs a lightweight uvx command to ensure the package is cached.
    This helps prevent "Connection closed" errors on first run due to
    uvx downloading packages from PyPI.

    This function:
    - Runs synchronously at module load
    - Silent failure (never breaks the hook)
    - Uses --help flag for minimal execution time
    """
    import subprocess

    server_config = DEFAULT_CONFIG.get('mcp_server', {})
    if not server_config.get('prewarm_cache', True):
        log_always('uvx cache pre-warm disabled via config')
        return

    command = server_config.get('command', 'uvx')
    python_version = server_config.get('python_version', '3.12')
    package = server_config.get('package', 'mcp-context-server[embeddings-ollama]<2.0.0')

    try:
        log_always('Pre-warming uvx cache...')
        # Use --help to trigger package download without starting server
        # Command list is built from trusted config values
        warmup_cmd = [str(command), '--python', str(python_version), '--with', str(package), '--help']
        result = subprocess.run(
            warmup_cmd,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            log_always('uvx cache pre-warm successful')
        else:
            log_always(f'uvx cache pre-warm returned code {result.returncode}', level='WARN')
    except subprocess.TimeoutExpired:
        log_always('uvx cache pre-warm timed out (continuing anyway)', level='WARN')
    except FileNotFoundError:
        log_always('uvx command not found (continuing anyway)', level='WARN')
    except Exception as e:
        # Silent failure for pre-warm
        log_always(f'uvx cache pre-warm failed: {type(e).__name__}: {e}', level='WARN')


# Pre-warm uvx cache at module load (non-blocking, silent failure)
_warmup_uvx_cache()


def setup_windows_utf8() -> None:
    """
    Configure Windows console for UTF-8 encoding.

    This ensures that subprocess communication uses UTF-8 instead of
    Windows codepage (CP1252, Windows-1251) which corrupts non-ASCII text.

    CRITICAL for handling Cyrillic, Chinese, Arabic, and other non-ASCII text.
    Without this, non-ASCII characters stored via the hook appear as
    garbled text (mojibake) due to Windows default codepage encoding.

    This function:
    1. Sets PYTHONUTF8=1 environment variable (Python 3.7+)
    2. Configures Windows console codepage to UTF-8 (65001)

    It is applied before any subprocess operations to ensure proper
    encoding for FastMCP StdioTransport stdin/stdout communication.
    """
    log_always('Configuring Windows UTF-8 encoding')
    if sys.platform != 'win32':
        log_always('Not Windows platform, skipping UTF-8 setup')
        return

    try:
        # Set Python to UTF-8 mode (Python 3.7+)
        # This ensures all text I/O uses UTF-8 by default
        os.environ['PYTHONUTF8'] = '1'

        # Set Windows console codepage to UTF-8 (65001)
        # This affects stdin/stdout/stderr of spawned subprocesses
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)  # Input codepage
        kernel32.SetConsoleOutputCP(65001)  # Output codepage

        log_always('UTF-8 mode configured for Windows console')
    except Exception as e:
        # Non-fatal: log error but continue
        # Hook should still work even if UTF-8 setup fails
        error_msg = f'Failed to set Windows UTF-8 mode: {e}'
        log_always(error_msg, level='ERROR')


def log_error(message: str) -> None:
    """
    Log errors to a debug file with default location for better diagnostics.

    Uses CLAUDE_HOOK_DEBUG_FILE environment variable to specify log location.
    If not set, defaults to .claude/.hook_debug.log in the project directory.

    This function is kept for backward compatibility with existing code that
    uses it, but internally delegates to log_always for guaranteed logging.

    Logging is CONDITIONAL based on CLAUDE_HOOK_DEBUG_ENABLED environment variable.

    Args:
        message: The error message to log
    """
    # Use log_always for guaranteed logging
    log_always(message, level='INFO')

    # Early exit if logging not enabled
    if not _LOGGING_ENABLED:
        return

    # Also try old logging path for compatibility
    debug_file = os.environ.get('CLAUDE_HOOK_DEBUG_FILE')

    # Default to project-local debug log if not specified
    if not debug_file:
        project_dir = os.environ.get('CLAUDE_PROJECT_DIR')
        if project_dir:
            debug_file = str(Path(project_dir) / '.claude' / '.hook_debug.log')

    if debug_file and debug_file != str(_LOG_FILE):
        try:
            with Path(debug_file).open('a', encoding='utf-8') as f:
                timestamp = datetime.now(tz=UTC).isoformat()
                f.write(f'{timestamp}: {message}\n')
        except Exception:
            # Silent failure for logging - don't break the hook
            pass


def report_error(error_type: str, error_msg: str) -> None:
    """
    Report error to both debug log and stats file for better diagnostics.

    Creates an error tracking file at .claude/.hook_errors with structured
    error information for troubleshooting intermittent failures.

    Logging is CONDITIONAL based on CLAUDE_HOOK_DEBUG_ENABLED environment variable.

    Args:
        error_type: Category of error (e.g., 'UVX_FAILURE', 'THREAD_ID_READ')
        error_msg: Detailed error message
    """
    full_msg = f'{error_type}: {error_msg}'
    log_always(full_msg, level='ERROR')

    # Early exit if logging not enabled
    if not _LOGGING_ENABLED:
        return

    # Track error statistics
    project_dir = os.environ.get('CLAUDE_PROJECT_DIR')
    if project_dir:
        stats_file = Path(project_dir) / '.claude' / '.hook_errors'
        try:
            stats = {
                'timestamp': datetime.now(tz=UTC).isoformat(),
                'error': error_type,
                'message': error_msg,
            }
            with stats_file.open('a', encoding='utf-8') as f:
                f.write(json.dumps(stats) + '\n')
        except Exception:
            # Silent failure for stats tracking
            pass


class SyncMCPClient:
    """
    Synchronous wrapper for the async FastMCP client.

    This wrapper allows us to use the async FastMCP client in a synchronous
    context, which is required for Claude Code hooks.
    """

    def __init__(
        self,
        server_command: list[str] | str,
        timeout: float = 120.0,
        max_connection_retries: int = 3,
        connection_timeout: float = 30.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the synchronous MCP client wrapper.

        Args:
            server_command: Command to start the MCP server (list or string)
            timeout: Timeout in seconds for MCP operations
            max_connection_retries: Number of retry attempts for MCP connection
            connection_timeout: Timeout in seconds per connection attempt
            config: Optional configuration dictionary with chunking settings
        """
        self.server_command = server_command
        self.timeout = timeout
        self.max_connection_retries = max_connection_retries
        self.connection_timeout = connection_timeout
        self.config = config or DEFAULT_CONFIG
        log_always(
            f'SyncMCPClient initialized: command={server_command}, timeout={timeout}, '
            f'max_conn_retries={max_connection_retries}, conn_timeout={connection_timeout}',
        )

    def _run_async(self, coro: Coroutine[Any, Any, dict[str, Any]]) -> dict[str, Any]:
        """
        Run an async coroutine in a sync context.

        Args:
            coro: The async coroutine to run

        Returns:
            The result of the coroutine

        Raises:
            RuntimeError: If called from an existing async context
        """
        try:
            asyncio.get_running_loop()
            # If we get here, we're already in an async context
            raise RuntimeError('Cannot run from async context')
        except RuntimeError as e:
            if 'no running event loop' in str(e).lower():
                # No event loop, safe to create one
                return asyncio.run(coro)
            # Re-raise if it's the "already in async" error
            raise

    def _calculate_message_size(self, thread_id: str, source: str, text: str) -> int:
        """
        Calculate the JSON message size in bytes.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context
            text: The text content to store

        Returns:
            Size of the JSON message in bytes
        """
        test_json = json.dumps({'thread_id': thread_id, 'source': source, 'text': text})
        return len(test_json.encode('utf-8'))

    @staticmethod
    def _chunk_text_by_bytes(text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks based on byte size, respecting UTF-8 boundaries.

        This method ensures that:
        1. Each chunk is at most `chunk_size` bytes when UTF-8 encoded
        2. No multi-byte UTF-8 characters are split mid-sequence
        3. All chunks decode back to valid UTF-8 strings

        UTF-8 encoding rules used for boundary detection:
        - 1-byte chars (ASCII): 0xxxxxxx (0x00-0x7F)
        - 2-byte chars: 110xxxxx 10xxxxxx
        - 3-byte chars: 1110xxxx 10xxxxxx 10xxxxxx
        - 4-byte chars: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        - Continuation bytes: 10xxxxxx (0x80-0xBF)

        Args:
            text: The text to split into chunks
            chunk_size: Maximum size in bytes for each chunk

        Returns:
            List of text chunks, each at most `chunk_size` bytes when encoded
        """
        if not text:
            return []

        text_bytes = text.encode('utf-8')
        chunks: list[str] = []

        i = 0
        while i < len(text_bytes):
            # Take up to chunk_size bytes
            end = min(i + chunk_size, len(text_bytes))
            chunk_bytes = text_bytes[i:end]

            # If we're not at the end, ensure we don't split a multi-byte character
            if end < len(text_bytes):
                # Walk back if we're in the middle of a UTF-8 sequence
                # Continuation bytes have pattern 10xxxxxx (0x80-0xBF)
                while chunk_bytes and (chunk_bytes[-1] & 0xC0) == 0x80:
                    chunk_bytes = chunk_bytes[:-1]

                # Safety check: if we walked back to nothing, something is wrong
                # This should never happen with valid UTF-8, but handle gracefully
                if not chunk_bytes:
                    log_always(
                        f'UTF-8 boundary detection failed at position {i}, '
                        f'forcing single character chunk',
                        level='WARN',
                    )
                    # Find the start byte and include the full character
                    start_byte = text_bytes[i]
                    if (start_byte & 0x80) == 0x00:
                        char_len = 1  # ASCII
                    elif (start_byte & 0xE0) == 0xC0:
                        char_len = 2  # 2-byte char
                    elif (start_byte & 0xF0) == 0xE0:
                        char_len = 3  # 3-byte char
                    elif (start_byte & 0xF8) == 0xF0:
                        char_len = 4  # 4-byte char
                    else:
                        char_len = 1  # Fallback for malformed UTF-8
                    chunk_bytes = text_bytes[i : i + char_len]

            # Decode the chunk and add to list
            try:
                chunk_str = chunk_bytes.decode('utf-8')
                chunks.append(chunk_str)
            except UnicodeDecodeError as e:
                # This should never happen if our boundary logic is correct
                log_always(
                    f'UTF-8 decode error at position {i}: {e}. '
                    f'Chunk bytes (first 20): {chunk_bytes[:20]!r}',
                    level='ERROR',
                )
                # Skip problematic bytes and continue
                i += 1
                continue

            i += len(chunk_bytes)

        return chunks

    async def _store_single_context_async(
        self,
        thread_id: str,
        source: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a single context message asynchronously using the MCP server.

        Includes retry logic with exponential backoff for connection reliability.
        The connection is wrapped in asyncio.timeout() to prevent hanging.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context (always "user" for this hook)
            text: The prompt text to store
            metadata: Optional metadata to include with the context

        Returns:
            The server response as a dictionary

        Raises:
            RuntimeError: If all connection retries are exhausted
        """
        from fastmcp.client.transports import StdioTransport

        log_always(f'_store_single_context_async called: thread_id={thread_id}, source={source}, text_len={len(text)}')

        if isinstance(self.server_command, list):
            cmd = self.server_command[0]
            args = self.server_command[1:] if len(self.server_command) > 1 else []
        else:
            parts = self.server_command.split()
            cmd = parts[0]
            args = parts[1:] if len(parts) > 1 else []

        log_always(f'MCP server command: {cmd} {args}')

        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'

        last_error: Exception | None = None
        max_retries = self.max_connection_retries
        operation_timeout = self.timeout

        for attempt in range(max_retries):
            try:
                log_always(f'Connection attempt {attempt + 1}/{max_retries} (timeout={operation_timeout}s)')

                transport = cast(Any, StdioTransport(cmd, args, env=env))
                log_always('StdioTransport created')

                # Wrap connection in asyncio.timeout for reliability
                async with asyncio.timeout(operation_timeout):
                    async with cast(Any, Client(transport)) as client:
                        # Connection health check - log successful connection
                        log_always('MCP client connected successfully (health check passed)')

                        # Normalize Windows line endings for NDJSON format
                        normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')

                        # Build call parameters
                        params: dict[str, Any] = {
                            'thread_id': thread_id,
                            'source': source,
                            'text': normalized_text,
                        }
                        if metadata:
                            params['metadata'] = metadata

                        log_always(f'Calling store_context with thread_id={thread_id}, source={source}')
                        result = await client.call_tool('store_context', params)
                        log_always(f'store_context returned: {type(result).__name__}')

                        # Use structured_content if available (canonical FastMCP approach)
                        if hasattr(result, 'structured_content') and result.structured_content is not None:
                            log_always('Using structured_content for stdio response')
                            return cast(dict[str, Any], result.structured_content)

                        return cast(dict[str, Any], result)

            except TimeoutError:
                last_error = TimeoutError(f'Connection timed out after {operation_timeout}s')
                log_always(
                    f'Connection attempt {attempt + 1}/{max_retries} timed out after {operation_timeout}s',
                    level='WARN',
                )
            except Exception as e:
                last_error = e
                log_always(
                    f'Connection attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}',
                    level='WARN',
                )

            # Exponential backoff before retry (1s, 2s, 4s)
            if attempt < max_retries - 1:
                backoff_delay = 1.0 * (2**attempt)
                log_always(f'Waiting {backoff_delay}s before retry (exponential backoff)')
                await asyncio.sleep(backoff_delay)

        # All retries exhausted
        error_msg = f'All {max_retries} connection attempts failed'
        if last_error:
            error_msg = f'{error_msg}: {type(last_error).__name__}: {last_error}'
        log_always(error_msg, level='ERROR')
        report_error('MCP_CONNECTION_EXHAUSTED', error_msg)

        if last_error:
            raise last_error
        raise RuntimeError(error_msg)

    async def _store_context_async(
        self,
        thread_id: str,
        source: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store context asynchronously using the MCP server with automatic chunking.

        This method handles large messages by splitting them into chunks when they
        exceed the Windows subprocess pipe buffer limit (~64KB). Messages are split
        at safe boundaries to avoid breaking UTF-8 encoding.

        CRITICAL: Uses a SINGLE MCP connection for ALL chunks to prevent connection
        churn and ensure reliable chunked storage. Per-chunk retry with exponential
        backoff is implemented within the single connection.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context (always "user" for this hook)
            text: The prompt text to store
            metadata: Optional metadata to include with the context

        Returns:
            The server response as a dictionary
        """
        # Calculate message size to check if chunking is needed
        message_size = self._calculate_message_size(thread_id, source, text)
        log_always(f'Message size: {len(text)} chars, {message_size} bytes (limit: {_MAX_MESSAGE_SIZE} bytes)')

        # If message is small enough, send directly
        if message_size <= _MAX_MESSAGE_SIZE:
            log_always('Message size within limits, sending directly')
            return await self._store_single_context_async(thread_id, source, text, metadata)

        # Message is too large, need to chunk it
        log_always(
            f'Message too large ({message_size} bytes > {_MAX_MESSAGE_SIZE} bytes), chunking required',
            level='WARN',
        )

        # Use byte-based chunking with UTF-8 boundary handling
        # This ensures each chunk respects the byte limit even for multi-byte characters
        chunks = self._chunk_text_by_bytes(text, _CHUNK_SIZE)

        # Log chunk statistics for debugging
        text_bytes = len(text.encode('utf-8'))
        chunk_byte_sizes = [len(c.encode('utf-8')) for c in chunks]
        log_always(
            f'Split {text_bytes} bytes into {len(chunks)} chunks: '
            f'sizes={chunk_byte_sizes}, avg={sum(chunk_byte_sizes) / len(chunks):.0f} bytes',
        )

        # Get chunking config settings
        chunking_config = self.config.get('chunking', DEFAULT_CONFIG['chunking'])
        max_chunk_retries = chunking_config.get('max_chunk_retries', 3)
        chunk_retry_delay = chunking_config.get('chunk_retry_delay', 0.5)

        # Use SINGLE connection for ALL chunks to prevent connection churn
        return await self._store_chunks_single_connection(
            thread_id=thread_id,
            source=source,
            chunks=chunks,
            text_bytes=text_bytes,
            max_chunk_retries=max_chunk_retries,
            chunk_retry_delay=chunk_retry_delay,
        )

    async def _store_chunks_single_connection(
        self,
        thread_id: str,
        source: str,
        chunks: list[str],
        text_bytes: int,
        max_chunk_retries: int,
        chunk_retry_delay: float,
    ) -> dict[str, Any]:
        """
        Store all chunks using a SINGLE MCP connection.

        This method establishes ONE connection and stores ALL chunks through it,
        with per-chunk retry logic using exponential backoff. This prevents the
        connection churn that was causing chunk 2+ failures.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context
            chunks: List of text chunks to store
            text_bytes: Total size in bytes for logging
            max_chunk_retries: Maximum retry attempts per chunk
            chunk_retry_delay: Initial delay between chunk retries (seconds)

        Returns:
            Summary of chunked storage results

        Raises:
            RuntimeError: If all connection retries are exhausted
        """
        from fastmcp.client.transports import StdioTransport

        log_always(f'_store_chunks_single_connection: storing {len(chunks)} chunks through SINGLE connection')

        if isinstance(self.server_command, list):
            cmd = self.server_command[0]
            args = self.server_command[1:] if len(self.server_command) > 1 else []
        else:
            parts = self.server_command.split()
            cmd = parts[0]
            args = parts[1:] if len(parts) > 1 else []

        log_always(f'MCP server command: {cmd} {args}')

        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'

        last_error: Exception | None = None
        max_retries = self.max_connection_retries
        # Extended timeout for chunked storage: base timeout * number of chunks
        extended_timeout = self.timeout * max(len(chunks), 2)

        for conn_attempt in range(max_retries):
            try:
                log_always(f'Connection attempt {conn_attempt + 1}/{max_retries} (timeout={extended_timeout}s)')

                transport = cast(Any, StdioTransport(cmd, args, env=env))
                log_always('StdioTransport created for chunked storage')

                # Wrap connection in asyncio.timeout for reliability
                async with asyncio.timeout(extended_timeout):
                    async with cast(Any, Client(transport)) as client:
                        log_always('MCP client connected for chunked storage (SINGLE connection for all chunks)')

                        results: list[dict[str, Any]] = []

                        # Store ALL chunks through the SAME connection
                        for idx, chunk in enumerate(chunks):
                            chunk_num = idx + 1
                            chunk_byte_size = len(chunk.encode('utf-8'))
                            log_always(
                                f'Storing chunk {chunk_num}/{len(chunks)} '
                                f'({len(chunk)} chars, {chunk_byte_size} bytes)',
                            )

                            # Add metadata to track chunks
                            metadata = {
                                'chunk': chunk_num,
                                'total_chunks': len(chunks),
                                'chunk_size_chars': len(chunk),
                                'chunk_size_bytes': chunk_byte_size,
                                'is_chunked': True,
                            }

                            # Per-chunk retry with exponential backoff (within same connection)
                            chunk_stored = False
                            chunk_last_error: Exception | None = None

                            for chunk_attempt in range(max_chunk_retries):
                                try:
                                    # Normalize Windows line endings for NDJSON format
                                    normalized_chunk = chunk.replace('\r\n', '\n').replace('\r', '\n')

                                    params: dict[str, Any] = {
                                        'thread_id': thread_id,
                                        'source': source,
                                        'text': normalized_chunk,
                                        'metadata': metadata,
                                    }

                                    result = await client.call_tool('store_context', params)
                                    results.append(cast(dict[str, Any], result))
                                    log_always(f'Chunk {chunk_num}/{len(chunks)} stored successfully')
                                    chunk_stored = True
                                    break  # Success, move to next chunk

                                except Exception as chunk_error:
                                    chunk_last_error = chunk_error
                                    if chunk_attempt < max_chunk_retries - 1:
                                        # Exponential backoff: delay * 2^attempt
                                        backoff = chunk_retry_delay * (2**chunk_attempt)
                                        log_always(
                                            f'Chunk {chunk_num} attempt {chunk_attempt + 1}/{max_chunk_retries} '
                                            f'failed: {chunk_error}. Retrying in {backoff}s',
                                            level='WARN',
                                        )
                                        await asyncio.sleep(backoff)
                                    else:
                                        log_always(
                                            f'Chunk {chunk_num} failed after {max_chunk_retries} attempts: '
                                            f'{chunk_error}',
                                            level='ERROR',
                                        )

                            if not chunk_stored:
                                # All retries exhausted for this chunk
                                error_msg = str(chunk_last_error) if chunk_last_error else 'Unknown error'
                                error_result: dict[str, Any] = {'error': error_msg, 'chunk': chunk_num}
                                results.append(error_result)

                        # All chunks processed - return results
                        successful_chunks = [r for r in results if 'error' not in r]
                        failed_chunks = [r for r in results if 'error' in r]

                        log_always(
                            f'Chunked storage complete: {len(successful_chunks)}/{len(chunks)} succeeded, '
                            f'{len(failed_chunks)} failed',
                        )

                        return {
                            'success': len(failed_chunks) == 0,
                            'chunked': True,
                            'total_chunks': len(chunks),
                            'chunks_stored': len(successful_chunks),
                            'chunks_failed': len(failed_chunks),
                            'total_bytes': text_bytes,
                            'results': results,
                            'failed_chunk_numbers': [r['chunk'] for r in failed_chunks],
                        }

            except TimeoutError:
                last_error = TimeoutError(f'Connection timed out after {extended_timeout}s')
                log_always(
                    f'Connection attempt {conn_attempt + 1}/{max_retries} timed out after {extended_timeout}s',
                    level='WARN',
                )
            except Exception as e:
                last_error = e
                log_always(
                    f'Connection attempt {conn_attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}',
                    level='WARN',
                )

            # Exponential backoff before retry (1s, 2s, 4s)
            if conn_attempt < max_retries - 1:
                backoff_delay = 1.0 * (2**conn_attempt)
                log_always(f'Waiting {backoff_delay}s before connection retry (exponential backoff)')
                await asyncio.sleep(backoff_delay)

        # All connection retries exhausted
        error_msg = f'All {max_retries} connection attempts failed for chunked storage'
        if last_error:
            error_msg = f'{error_msg}: {type(last_error).__name__}: {last_error}'
        log_always(error_msg, level='ERROR')
        report_error('MCP_CHUNKED_CONNECTION_EXHAUSTED', error_msg)

        if last_error:
            raise last_error
        raise RuntimeError(error_msg)

    def store_context(
        self,
        thread_id: str,
        source: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store context synchronously.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context (always "user" for this hook)
            text: The prompt text to store
            metadata: Optional metadata to include with the context

        Returns:
            The server response as a dictionary
        """
        log_always(f'store_context called: thread_id={thread_id}, source={source}, text_len={len(text)}')
        result = self._run_async(self._store_context_async(thread_id, source, text, metadata))
        log_always('store_context completed successfully')
        return result


class FastMCPHttpClient:
    """
    HTTP-based MCP client using FastMCP's StreamableHttpTransport.

    This client uses the official FastMCP Client library for HTTP communication,
    providing robust protocol handling, automatic retries, and proper error management.

    The FastMCP Client is async-only, so this class wraps operations with asyncio.run()
    for synchronous hook execution.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        max_retries: int = 3,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the FastMCP HTTP client.

        Args:
            url: The MCP server endpoint URL
            timeout: Request timeout in seconds
            headers: Optional custom headers for authentication
            max_retries: Number of retry attempts for failed requests
            config: Optional configuration dictionary
        """
        self.url = url
        self.timeout = timeout
        self.headers = headers or {}
        self.max_retries = max_retries
        self.config = config or DEFAULT_CONFIG
        log_always(
            f'FastMCPHttpClient initialized: url={url}, timeout={timeout}, '
            f'max_retries={max_retries}',
        )

    def _calculate_message_size(self, thread_id: str, source: str, text: str) -> int:
        """Calculate the JSON message size in bytes."""
        test_json = json.dumps({'thread_id': thread_id, 'source': source, 'text': text})
        return len(test_json.encode('utf-8'))

    async def _store_context_async(
        self,
        thread_id: str,
        source: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store context asynchronously using FastMCP Client.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context (always "user" for this hook)
            text: The prompt text to store
            metadata: Optional metadata to include with the context

        Returns:
            The server response as a dictionary

        Raises:
            RuntimeError: If all retry attempts are exhausted
        """
        from fastmcp.client.transports import StreamableHttpTransport
        from fastmcp.exceptions import ToolError as FastMCPToolError

        log_always(f'_store_context_async: thread_id={thread_id}, source={source}, text_len={len(text)}')

        # Normalize Windows line endings
        normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                log_always(f'FastMCP HTTP attempt {attempt + 1}/{self.max_retries}')

                # Create transport with custom headers
                transport = StreamableHttpTransport(
                    url=self.url,
                    headers=self.headers or None,
                )

                # Use async context manager for proper connection lifecycle
                async with asyncio.timeout(self.timeout):
                    async with Client(transport, timeout=self.timeout) as client:
                        log_always('FastMCP client connected successfully')

                        # Build call parameters
                        call_params: dict[str, Any] = {
                            'thread_id': thread_id,
                            'source': source,
                            'text': normalized_text,
                        }
                        if metadata:
                            call_params['metadata'] = metadata

                        # Call store_context tool
                        result = await client.call_tool(
                            'store_context',
                            call_params,
                            timeout=self.timeout,
                            raise_on_error=True,
                        )

                        log_always(f'FastMCP call_tool returned: {type(result).__name__}')

                        # PRIORITY 1: Use structured_content (canonical FastMCP approach)
                        # FastMCP docs: "Raw structured JSON is also available via result.structured_content"
                        if hasattr(result, 'structured_content') and result.structured_content is not None:
                            log_always(
                                f'Using structured_content (dict) for response: '
                                f'{type(result.structured_content).__name__}',
                            )
                            # Explicit type annotation to satisfy both mypy and pyright
                            structured_dict: dict[str, Any] = result.structured_content
                            return structured_dict

                        # PRIORITY 2: Handle result.data with comprehensive fallback
                        if hasattr(result, 'data') and result.data is not None:
                            data = result.data

                            # Pydantic v2
                            if hasattr(data, 'model_dump') and callable(data.model_dump):
                                log_always(f'Using model_dump() for: {type(data).__name__}')
                                return cast(dict[str, Any], data.model_dump())

                            # Pydantic v1
                            if hasattr(data, 'dict') and callable(data.dict):
                                log_always(f'Using dict() for: {type(data).__name__}')
                                return cast(dict[str, Any], data.dict())

                            # RootModel with .root attribute
                            if hasattr(data, 'root'):
                                root_data = data.root
                                if isinstance(root_data, dict):
                                    log_always(f'Using .root for: {type(data).__name__}')
                                    return cast(dict[str, Any], root_data)

                            # Direct dict
                            if isinstance(data, dict):
                                log_always('Using direct dict')
                                return cast(dict[str, Any], data)

                            # Final fallback - log as INFO not WARN (this path is acceptable)
                            log_always(f'Using string fallback for: {type(data).__name__}')
                            return {'success': True, 'raw_data': str(data)}

                        # PRIORITY 3: Fallback to content block parsing
                        if hasattr(result, 'content') and result.content:
                            first_content = result.content[0]
                            content_text = getattr(first_content, 'text', None)
                            if content_text is not None:
                                try:
                                    log_always('Using content block parsing')
                                    return cast(dict[str, Any], json.loads(str(content_text)))
                                except json.JSONDecodeError:
                                    return {'success': True, 'raw_content': str(content_text)}

                        # No usable data
                        log_always('[WARN] No structured_content, data, or content available')
                        return {'success': False, 'error': 'No response data available'}

            except FastMCPToolError as e:
                last_error = e
                log_always(
                    f'FastMCP attempt {attempt + 1}/{self.max_retries} tool error: {e}',
                    level='WARN',
                )
            except TimeoutError:
                last_error = TimeoutError(f'Request timed out after {self.timeout}s')
                log_always(
                    f'FastMCP attempt {attempt + 1}/{self.max_retries} timed out',
                    level='WARN',
                )
            except Exception as e:
                last_error = e
                log_always(
                    f'FastMCP attempt {attempt + 1}/{self.max_retries} failed: '
                    f'{type(e).__name__}: {e}',
                    level='WARN',
                )

            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                backoff_delay = 1.0 * (2 ** attempt)
                log_always(f'Waiting {backoff_delay}s before retry')
                await asyncio.sleep(backoff_delay)

        # All retries exhausted
        error_msg = f'All {self.max_retries} FastMCP HTTP attempts failed'
        if last_error:
            error_msg = f'{error_msg}: {type(last_error).__name__}: {last_error}'
        log_always(error_msg, level='ERROR')
        report_error('FASTMCP_HTTP_CONNECTION_EXHAUSTED', error_msg)

        if last_error:
            raise last_error
        raise RuntimeError(error_msg)

    def store_context(
        self,
        thread_id: str,
        source: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store context via FastMCP HTTP client synchronously.

        This method wraps the async FastMCP client with asyncio.run() for
        synchronous hook execution.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context (always "user" for this hook)
            text: The prompt text to store
            metadata: Optional metadata to include with the context

        Returns:
            The server response as a dictionary
        """
        log_always(f'FastMCPHttpClient.store_context: thread_id={thread_id}, source={source}, text_len={len(text)}')
        result = asyncio.run(self._store_context_async(thread_id, source, text, metadata))
        log_always('FastMCPHttpClient.store_context completed successfully')
        return result


def get_worktree_info(project_dir: str) -> dict[str, Any]:
    """Get worktree information for context metadata.

    Provides canonical project name from git remote URL and worktree metadata
    for proper context isolation across parallel worktree sessions.

    Fallback chain for project name:
    1. Parse repo name from git remote URL (origin -> upstream)
    2. Basename of git toplevel directory
    3. Current directory basename

    Args:
        project_dir: Project directory path.

    Returns:
        Dictionary with:
        - project: Canonical project name
        - worktree_id: Worktree directory name (None if not git repo)
        - worktree_path: Absolute path to worktree (None if not git repo)
        - is_linked_worktree: Boolean for linked worktree (None if not git repo)
    """

    def run_git(args: list[str]) -> str | None:
        """Run git command and return output, or None on failure."""
        try:
            result = subprocess.run(
                ['git', *args],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (subprocess.SubprocessError, OSError):
            return None

    # Get git common dir and toplevel
    git_common_dir = run_git(['rev-parse', '--git-common-dir'])
    git_toplevel = run_git(['rev-parse', '--show-toplevel'])

    if not git_toplevel:
        # Not a git repository - use directory name as project name
        return {
            'project': Path(project_dir).name,
            'worktree_id': None,
            'worktree_path': None,
            'is_linked_worktree': None,
        }

    toplevel_path = Path(git_toplevel).resolve()

    # Detect linked worktree
    # For main worktree: git_common_dir is relative '.git' or absolute path to .git
    # For linked worktree: git_common_dir points to main repo's .git directory
    is_linked = False
    if git_common_dir:
        common_path = Path(git_common_dir)
        if common_path.is_absolute():
            expected_git_dir = toplevel_path / '.git'
            is_linked = common_path.resolve() != expected_git_dir.resolve()
        else:
            is_linked = git_common_dir not in ['.git', './.git']

    # Get canonical project name from remote URL with fallback chain
    project_name = toplevel_path.name  # Default fallback to toplevel dir name
    for remote in ['origin', 'upstream']:
        url = run_git(['remote', 'get-url', remote])
        if url:
            # Parse repo name from URL using generic pattern
            # Supports: https://github.com/user/repo.git, git@github.com:user/repo.git, etc.
            match = re.search(r'.*/([^/]+?)(?:\.git)?$', url.strip())
            if match:
                project_name = match.group(1)
                break

    return {
        'project': project_name,
        'worktree_id': toplevel_path.name,
        'worktree_path': str(toplevel_path),
        'is_linked_worktree': is_linked,
    }


def is_prebuilt_slash_command(prompt: str, config: dict[str, Any]) -> bool:
    """
    Check if a prompt is a pre-built slash command that should be skipped.

    Pre-built slash commands are built-in Claude Code commands that don't need
    context saving as they are system-level operations.

    Args:
        prompt: The user prompt to check
        config: Configuration dictionary with prebuilt_commands list

    Returns:
        True if the prompt is a pre-built slash command, False otherwise
    """
    # Get prebuilt commands from config
    prebuilt_commands = set(config.get('prebuilt_commands', DEFAULT_CONFIG['prebuilt_commands']))

    # Pattern to match slash commands at the start of a prompt
    # Matches: /command or /command with args
    # Uses \S+ to match any non-whitespace characters (handles underscores, hyphens, numbers, etc.)
    slash_command_pattern = re.compile(r'^/(\S+)(?:\s|$)')

    match = slash_command_pattern.match(prompt.strip())
    if not match:
        return False

    command_name = match.group(1).lower()
    return command_name in prebuilt_commands


def matches_skip_pattern(prompt: str, config: dict[str, Any]) -> bool:
    """
    Check if a prompt matches any skip pattern (for filtering internal hook prompts).

    This function filters out internal hook prompts from being saved to the context
    server. When hooks of `type: prompt` are evaluated, they trigger UserPromptSubmit
    events that would otherwise be captured and saved as user messages.

    Args:
        prompt: The prompt text to check
        config: Configuration dictionary with skip_patterns list

    Returns:
        True if the prompt matches any skip pattern, False otherwise
    """
    skip_patterns = config.get('skip_patterns', DEFAULT_CONFIG['skip_patterns'])

    if not skip_patterns:
        return False

    for pattern in skip_patterns:
        try:
            if re.match(pattern, prompt):
                log_always(f'Prompt matches skip pattern: {pattern[:50]}...')
                return True
        except re.error as e:
            log_always(f'Invalid regex pattern "{pattern}": {e}', level='ERROR')
            continue

    return False


def resolve_thread_id(project_dir: str, config: dict[str, Any], worktree_info: dict[str, Any]) -> str:
    """
    Resolve the thread ID for context server storage.

    Discovery chain:
    1. Read .context_server/.thread_id file (with retry for Windows file-locking)
    2. Fall back to canonical project name from worktree_info

    Args:
        project_dir: The Claude project directory path
        config: Configuration dictionary with thread_id settings
        worktree_info: Pre-computed worktree information from get_worktree_info()

    Returns:
        The thread ID string (always non-empty)
    """
    thread_id_config = config.get('thread_id', DEFAULT_CONFIG['thread_id'])
    max_retries: int = thread_id_config.get('max_retries', 3)

    thread_id_file = Path(project_dir) / '.context_server' / '.thread_id'
    log_always(f'Reading thread ID from: {thread_id_file}')

    if not thread_id_file.exists():
        project_name: str = str(worktree_info['project'])
        log_always(f'Thread ID file does not exist, using project name: {project_name}')
        return project_name

    for attempt in range(max_retries):
        try:
            thread_id = thread_id_file.read_text(encoding='utf-8').strip()
            if thread_id:
                if attempt > 0:
                    log_always(f'Thread ID read succeeded on attempt {attempt + 1}/{max_retries}')
                log_always(f'Thread ID: {thread_id}')
                return thread_id
            report_error(
                'THREAD_ID_READ',
                f'Thread ID file empty (attempt {attempt + 1}/{max_retries})',
            )
        except OSError as e:
            error_msg = f'Failed to read thread ID (attempt {attempt + 1}/{max_retries}): {e}'
            if attempt < max_retries - 1:
                log_always(error_msg, level='ERROR')
                time.sleep(0.1 * (2**attempt))  # Exponential backoff: 100ms, 200ms, 400ms
            else:
                report_error('THREAD_ID_READ', error_msg)

    fallback_name: str = str(worktree_info['project'])
    log_always(f'All thread ID read attempts failed, using project name: {fallback_name}')
    return fallback_name


def create_mcp_client_with_retry(config: dict[str, Any]) -> SyncMCPClient:
    """
    Create MCP client with retry logic for uvx reliability.

    Args:
        config: Configuration dictionary with mcp_client and mcp_server settings

    Returns:
        Configured SyncMCPClient instance

    Raises:
        RuntimeError: If all retry attempts fail
    """
    mcp_config = config.get('mcp_client', DEFAULT_CONFIG['mcp_client'])
    server_config = config.get('mcp_server', DEFAULT_CONFIG['mcp_server'])

    max_retries: int = mcp_config.get('max_retries', 3)
    max_connection_retries: int = mcp_config.get('max_connection_retries', 3)
    connection_timeout: float = mcp_config.get('connection_timeout', 30.0)
    timeout_first_run: float = mcp_config.get('timeout_first_run', 120.0)
    timeout_normal: float = mcp_config.get('timeout_normal', 120.0)

    server_command: str = server_config.get('command', 'uvx')
    python_version: str = server_config.get('python_version', '3.12')
    package: str = server_config.get('package', 'mcp-context-server[embeddings-ollama]<2.0.0')
    entry_point: str = server_config.get('entry_point', 'mcp-context-server')

    log_always('Creating MCP client with retry logic')
    log_always(
        f'Config: max_retries={max_retries}, max_conn_retries={max_connection_retries}, '
        f'conn_timeout={connection_timeout}, timeout_first={timeout_first_run}, timeout_normal={timeout_normal}',
    )
    log_always(f'Server: command={server_command}, python={python_version}, package={package}, entry={entry_point}')

    last_error: Exception | None = None
    attempts = 0

    while attempts < max_retries:
        try:
            mcp_server_command = [
                server_command,
                '--python',
                python_version,
                '--with',
                package,
                entry_point,
            ]
            timeout = timeout_first_run if attempts == 0 else timeout_normal
            log_always(f'Attempt {attempts + 1}/{max_retries}: Creating MCP client (timeout={timeout}s)')
            client = SyncMCPClient(
                mcp_server_command,
                timeout=timeout,
                max_connection_retries=max_connection_retries,
                connection_timeout=connection_timeout,
                config=config,
            )

            if attempts > 0:
                log_always(f'MCP client created successfully on attempt {attempts + 1}/{max_retries}')
            else:
                log_always('MCP client created successfully')

            return client

        except Exception as e:
            last_error = e
            attempts += 1
            error_msg = f'Failed to create MCP client (attempt {attempts}/{max_retries}): {type(e).__name__}: {e}'
            log_always(error_msg, level='ERROR')

            if attempts < max_retries:
                if attempts > 1:
                    try:
                        log_always(f'{error_msg}, trying offline mode')
                        mcp_server_command = [
                            server_command,
                            '--python',
                            python_version,
                            '--with',
                            package,
                            '--offline',
                            entry_point,
                        ]
                        client = SyncMCPClient(
                            mcp_server_command,
                            timeout=timeout_normal,
                            max_connection_retries=max_connection_retries,
                            connection_timeout=connection_timeout,
                            config=config,
                        )
                        log_always('MCP client created successfully in offline mode')
                        return client
                    except Exception as offline_error:
                        log_always(
                            f'Offline mode failed: {type(offline_error).__name__}: {offline_error}',
                            level='ERROR',
                        )

                sleep_time = 0.5 * (2 ** (attempts - 1))
                log_always(f'Sleeping {sleep_time}s before retry')
                time.sleep(sleep_time)
            else:
                report_error('UVX_FAILURE', error_msg)

    if last_error:
        raise last_error
    raise RuntimeError('Failed to create MCP client after all retries')


def create_mcp_client(config: dict[str, Any]) -> SyncMCPClient | FastMCPHttpClient:
    """
    Create appropriate MCP client based on transport configuration.

    This factory function selects between stdio transport (existing behavior via uvx)
    and HTTP transport (using FastMCP Client for remote MCP servers).

    Args:
        config: Configuration dictionary with mcp_server settings

    Returns:
        Either SyncMCPClient (stdio) or FastMCPHttpClient (http)

    Raises:
        ValueError: If transport type is invalid or required fields missing
    """
    server_config = config.get('mcp_server', DEFAULT_CONFIG['mcp_server'])
    transport = server_config.get('transport', 'stdio')

    log_always(f'Creating MCP client with transport: {transport}')

    if transport == 'http':
        # HTTP transport using FastMCP Client
        url = server_config.get('url')
        if not url:
            raise ValueError("mcp_server.url is required when transport is 'http'")

        mcp_config = config.get('mcp_client', DEFAULT_CONFIG['mcp_client'])
        timeout = server_config.get('timeout', mcp_config.get('connection_timeout', 30.0))
        headers = server_config.get('headers', {})
        max_retries = mcp_config.get('max_retries', 3)

        log_always(f'HTTP transport (FastMCP): url={url}, timeout={timeout}, max_retries={max_retries}')

        return FastMCPHttpClient(
            url=url,
            timeout=timeout,
            headers=headers,
            max_retries=max_retries,
            config=config,
        )
    if transport == 'stdio':
        # Existing stdio transport via uvx subprocess
        log_always('stdio transport: using create_mcp_client_with_retry')
        return create_mcp_client_with_retry(config)
    raise ValueError(f"Invalid mcp_server.transport: {transport}. Must be 'stdio' or 'http'")


def main() -> None:
    """Main hook execution function."""
    log_always('Entering main() function')
    start_time = datetime.now(tz=UTC)

    try:
        # Load configuration (defaults merged with config file if provided)
        try:
            config_loader = _load_config_loader()
            config: dict[str, Any] = config_loader.get_config_from_argv(DEFAULT_CONFIG)
        except Exception:
            # If config loading fails, use defaults
            config = DEFAULT_CONFIG.copy()

        # Check if hook is enabled
        if not config.get('enabled', True):
            log_always('Hook disabled via config, exiting')
            sys.exit(0)

        # CRITICAL: Configure UTF-8 for Windows BEFORE any subprocess operations
        # This prevents non-ASCII text corruption (mojibake) in MCP communication
        setup_windows_utf8()

        # Reconfigure stdin to UTF-8 for Git Bash compatibility
        # Git Bash on Windows uses MinGW64/MSYS2 runtime with Unix-style locale system.
        # Without LANG/LC_ALL exports, it defaults to Windows codepage (CP1252/Windows-1251),
        # causing stdin pipes to corrupt UTF-8 data BEFORE Python reads it.
        # This reconfigures the already-open stdin stream to UTF-8 encoding.
        #
        # Use getattr() for type-safe access to reconfigure() method (Python 3.7+)
        # sys.stdin is typed as TextIO in stubs but is TextIOWrapper at runtime
        log_always('Reconfiguring stdin for UTF-8')
        reconfigure_method = getattr(sys.stdin, 'reconfigure', None)
        if reconfigure_method is not None:
            # Python 3.7+ has reconfigure() method on TextIOWrapper
            try:
                reconfigure_method(encoding='utf-8')
                log_always('stdin reconfigured to UTF-8 via reconfigure()')
                log_error('Git Bash compatibility: stdin reconfigured to UTF-8')
            except OSError as e:
                error_msg = f'stdin reconfigure failed: {e}'
                log_always(error_msg, level='ERROR')
                log_error(f'Git Bash compatibility: {error_msg}')
        else:
            # Fallback for Python < 3.7 or if reconfigure() not available
            try:
                sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                log_always('stdin wrapped with UTF-8 TextIOWrapper')
                log_error('Git Bash compatibility: stdin wrapped with UTF-8 TextIOWrapper')
            except Exception as e:
                error_msg = f'stdin UTF-8 fix failed: {e}'
                log_always(error_msg, level='ERROR')
                log_error(f'Git Bash compatibility: {error_msg}')

        # Read input from stdin
        log_always('Reading stdin data')
        try:
            input_data = json.load(sys.stdin)
            log_always(f'stdin data keys: {list(input_data.keys())}')
        except json.JSONDecodeError as e:
            log_always(f'JSON decode error: {e}', level='ERROR')
            log_always('Exiting: Invalid JSON from stdin')
            sys.exit(0)
        except Exception as e:
            log_always(f'Error reading stdin: {type(e).__name__}: {e}', level='ERROR')
            log_always('Exiting: Failed to read stdin')
            sys.exit(0)

        # Extract key fields
        hook_event_name = input_data.get('hook_event_name', '')
        log_always(f'hook_event_name: {hook_event_name}')

        # Validate this is a UserPromptSubmit event
        if hook_event_name != 'UserPromptSubmit':
            log_always(f'Skipping: Event type is {hook_event_name}, not UserPromptSubmit')
            sys.exit(0)

        # Extract prompt from input data (UserPromptSubmit has prompt directly)
        prompt = input_data.get('prompt', '')
        log_always(f'Prompt length: {len(prompt)} characters')
        if not prompt:
            # No prompt to save
            log_always('Skipping: Empty prompt')
            sys.exit(0)

        # Check if this is a pre-built slash command that should be skipped
        if is_prebuilt_slash_command(prompt, config):
            # Skip pre-built slash commands
            log_always('Skipping: Pre-built slash command detected')
            sys.exit(0)

        # Check if this prompt matches any skip pattern (internal hook prompts)
        if matches_skip_pattern(prompt, config):
            # Skip internal hook prompts
            log_always('Skipping: Internal hook prompt detected')
            sys.exit(0)

        # Get Claude project directory
        claude_project_dir = os.environ.get('CLAUDE_PROJECT_DIR')
        log_always(f'CLAUDE_PROJECT_DIR for context save: {claude_project_dir}')
        if not claude_project_dir:
            # No project directory, can't proceed
            log_always('Exiting: CLAUDE_PROJECT_DIR not set', level='ERROR')
            sys.exit(0)

        # Get worktree information for context metadata and thread ID resolution
        # Provides canonical project name from git remote URL for cross-worktree consistency
        worktree_info = get_worktree_info(claude_project_dir)
        log_always(f'Worktree info: project={worktree_info["project"]}')

        # Resolve thread ID (reads .thread_id file, falls back to project name)
        thread_id = resolve_thread_id(claude_project_dir, config, worktree_info)

        # Create MCP client and store context
        try:
            # Any failure in MCP communication should be silent
            #
            # CRITICAL UTF-8 REQUIREMENT:
            # The setup_windows_utf8() function MUST be called before this point
            # to ensure subprocess stdin/stdout use UTF-8 encoding.
            # Without this, non-ASCII text (Cyrillic, Chinese, Arabic, etc.) gets
            # corrupted on Windows due to codepage defaults (CP1252/Windows-1251).
            #
            # FastMCP's StdioTransport does not explicitly set encoding='utf-8'
            # on subprocess pipes, so we configure it via environment variable
            # (PYTHONUTF8=1) and console codepage (65001) instead.

            # Create client based on transport configuration (stdio or http)
            client = create_mcp_client(config)

            # Build metadata with worktree fields for context isolation
            metadata: dict[str, Any] = {
                'project': worktree_info['project'],
            }
            # Add optional worktree fields if available (git repository detected)
            if worktree_info.get('worktree_id') is not None:
                metadata['worktree_id'] = worktree_info['worktree_id']
            if worktree_info.get('worktree_path') is not None:
                metadata['worktree_path'] = worktree_info['worktree_path']
            if worktree_info.get('is_linked_worktree') is not None:
                metadata['is_linked_worktree'] = worktree_info['is_linked_worktree']

            log_always('Storing context in MCP server')
            result = client.store_context(
                thread_id=thread_id,
                source='user',
                text=prompt,
                metadata=metadata,
            )

            # Check for chunked storage with partial failures and provide user feedback
            if result.get('chunked', False):
                chunks_failed = result.get('chunks_failed', 0)
                total_chunks = result.get('total_chunks', 0)
                chunks_stored = result.get('chunks_stored', 0)

                if chunks_failed > 0:
                    # Get fail_mode from config
                    chunking_config = config.get('chunking', DEFAULT_CONFIG['chunking'])
                    fail_mode = chunking_config.get('fail_mode', 'warn')

                    failed_numbers = result.get('failed_chunk_numbers', [])
                    feedback_msg = (
                        f'Context storage partial failure: {chunks_failed}/{total_chunks} chunks failed '
                        f'(chunks {failed_numbers}). {chunks_stored} chunks stored successfully.'
                    )

                    # Always log the failure (existing logging continues)
                    log_always(feedback_msg, level='WARN')
                    report_error('CHUNK_STORAGE_PARTIAL', feedback_msg)

                    # User feedback based on fail_mode (displays in TUI)
                    if fail_mode == 'warn':
                        print(f'[WARN] {feedback_msg}', file=sys.stderr)
                    elif fail_mode == 'error':
                        print(f'[ERROR] {feedback_msg}', file=sys.stderr)
                    # 'silent' mode: no stderr output, only logging

                    log_always(f'User feedback mode: {fail_mode}')
                else:
                    log_always(f'SUCCESS: All {total_chunks} chunks stored successfully')

                # Output chunk IDs via additionalContext for orchestrator reference
                if config.get('output_context_id', True):
                    chunk_ids = result.get('chunk_ids', [])
                    if chunk_ids:
                        hook_output: dict[str, Any] = {
                            'hookSpecificOutput': {
                                'hookEventName': 'UserPromptSubmit',
                                'additionalContext': (
                                    f'[Hook: user message stored to context-server (chunked).'
                                    f' context_ids={chunk_ids}]'
                                ),
                            },
                        }
                        sys.stdout.write(json.dumps(hook_output))
                        sys.stdout.flush()
                        log_always(f'Output chunk context_ids={chunk_ids} via additionalContext')
            else:
                log_always('SUCCESS: Context stored successfully')

                # Output context_id via additionalContext for orchestrator reference
                if config.get('output_context_id', True):
                    context_id = result.get('context_id')
                    if context_id is not None:
                        hook_output = {
                            'hookSpecificOutput': {
                                'hookEventName': 'UserPromptSubmit',
                                'additionalContext': f'[Hook: user message stored to context-server. context_id={context_id}]',
                            },
                        }
                        sys.stdout.write(json.dumps(hook_output))
                        sys.stdout.flush()
                        log_always(f'Output context_id={context_id} via additionalContext')

        except Exception as e:
            # Log the error for debugging with full traceback, then suppress as designed
            error_msg = f'{type(e).__name__}: {e}'
            full_traceback = traceback.format_exc()

            # Check for specific error patterns related to pipe buffer issues
            error_str = str(e).lower()
            error_context = ''

            if 'broken pipe' in error_str:
                error_context = ' (Message likely too large for subprocess pipe buffer)'
            elif 'timeout' in error_str:
                error_context = ' (Consider increasing timeout for large messages via CLAUDE_HOOK_MCP_TIMEOUT)'
            elif '[errno 32]' in error_str or 'epipe' in error_str:
                error_context = ' (Subprocess pipe broken - message size may exceed buffer capacity)'
            elif 'buffer' in error_str:
                error_context = ' (Buffer-related error - message may be too large)'

            log_always(f'MCP store failure: {error_msg}{error_context}', level='ERROR')
            log_always(f'Traceback:\n{full_traceback}', level='ERROR')
            report_error('MCP_STORE_FAILURE', f'{error_msg}{error_context}\n{full_traceback}')
            # Silent failure - don't break Claude Code workflow

        # Always exit successfully
        end_time = datetime.now(tz=UTC)
        duration = (end_time - start_time).total_seconds()
        log_always(f'Execution completed in {duration:.3f} seconds')
        sys.exit(0)

    except Exception as e:
        # Handle all errors silently but log them
        error_msg = f'Unexpected error in main(): {type(e).__name__}: {e}'
        full_traceback = traceback.format_exc()
        log_always(error_msg, level='ERROR')
        log_always(f'Traceback:\n{full_traceback}', level='ERROR')
        sys.exit(0)


if __name__ == '__main__':
    main()
