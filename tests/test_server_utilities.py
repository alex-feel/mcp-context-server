"""Tests for server utility functions.

This module tests utility functions in app/server.py including
text truncation and helper functions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.server import truncate_text


class TestTruncateText:
    """Test the truncate_text utility function."""

    def test_truncate_none_text(self) -> None:
        """Test that None text returns (None, False)."""
        result, is_truncated = truncate_text(None)
        assert result is None
        assert is_truncated is False

    def test_truncate_empty_text(self) -> None:
        """Test that empty text returns ('', False)."""
        result, is_truncated = truncate_text('')
        assert result == ''
        assert is_truncated is False

    def test_truncate_short_text(self) -> None:
        """Test that short text is not truncated."""
        short_text = 'This is a short text.'
        result, is_truncated = truncate_text(short_text)
        assert result == short_text
        assert is_truncated is False

    def test_truncate_text_at_default_length(self) -> None:
        """Test text at exactly 300 characters is not truncated."""
        text_300 = 'x' * 300
        result, is_truncated = truncate_text(text_300)
        assert result == text_300
        assert is_truncated is False

    def test_truncate_long_text(self) -> None:
        """Test that long text is truncated with ellipsis."""
        long_text = 'x' * 400
        result, is_truncated = truncate_text(long_text)
        assert is_truncated is True
        assert result is not None
        assert result.endswith('...')
        assert len(result) <= 303  # 300 + '...'

    def test_truncate_custom_max_length(self) -> None:
        """Test truncation with custom max_length."""
        text = 'This is a test text that should be truncated.'
        result, is_truncated = truncate_text(text, max_length=20)
        assert is_truncated is True
        assert result is not None
        assert result.endswith('...')
        assert len(result) <= 23  # 20 + '...'

    def test_truncate_at_word_boundary(self) -> None:
        """Test that truncation happens at word boundaries when possible."""
        # Text with a space after position 210 (70% of 300)
        text = 'This is a sentence with several words ' + 'x' * 265 + ' end'
        result, is_truncated = truncate_text(text)
        assert is_truncated is True
        assert result is not None
        assert result.endswith('...')

    def test_truncate_no_word_boundary(self) -> None:
        """Test truncation when no good word boundary exists."""
        # Text without spaces within the 70% threshold
        text = 'x' * 400  # No spaces at all
        result, is_truncated = truncate_text(text)
        assert is_truncated is True
        assert result is not None
        assert result.endswith('...')
        assert len(result) == 303  # Exact truncation at 300 + '...'

    def test_truncate_word_boundary_threshold(self) -> None:
        """Test word boundary threshold (70% of max_length)."""
        # max_length=300, 70% = 210
        # Place a space at position 200 (below threshold) and another at 220 (above)
        text = 'x' * 200 + ' ' + 'y' * 19 + ' ' + 'z' * 200
        result, is_truncated = truncate_text(text)
        assert is_truncated is True
        # Should truncate at position 110 (the space above threshold)
        assert result is not None
        assert result.endswith('...')


class TestServerEnsureFunctions:
    """Test _ensure_backend and _ensure_repositories functions."""

    @pytest.mark.asyncio
    async def test_ensure_backend_creates_new(self) -> None:
        """Test that ensure_backend creates a new backend when none exists."""
        import app.startup
        from app.startup import ensure_backend

        # Store original
        original_backend = app.startup._backend

        try:
            # Clear the backend
            app.startup._backend = None

            # Mock create_backend and initialize
            mock_backend = MagicMock()
            mock_backend.initialize = AsyncMock()

            with patch('app.startup.create_backend', return_value=mock_backend) as mock_create:
                backend = await ensure_backend()

                # Verify backend was created and initialized
                mock_create.assert_called_once()
                mock_backend.initialize.assert_awaited_once()
                assert backend == mock_backend
        finally:
            # Restore original
            app.startup._backend = original_backend

    @pytest.mark.asyncio
    async def test_ensure_backend_returns_existing(self) -> None:
        """Test that ensure_backend returns existing backend."""
        import app.startup
        from app.startup import ensure_backend

        # Store original
        original_backend = app.startup._backend

        try:
            # Set a mock backend
            mock_backend = MagicMock()
            app.startup._backend = mock_backend

            with patch('app.startup.create_backend') as mock_create:
                backend = await ensure_backend()

                # Should return existing, not create new
                mock_create.assert_not_called()
                assert backend == mock_backend
        finally:
            # Restore original
            app.startup._backend = original_backend

    @pytest.mark.asyncio
    async def test_ensure_repositories_creates_new(self) -> None:
        """Test that ensure_repositories creates new when none exists."""
        import app.startup
        from app.repositories import RepositoryContainer
        from app.startup import ensure_repositories

        # Store originals
        original_backend = app.startup._backend
        original_repos = app.startup._repositories

        try:
            # Set up a mock backend
            mock_backend = MagicMock()
            mock_backend.backend_type = 'sqlite'
            app.startup._backend = mock_backend
            app.startup._repositories = None

            repos = await ensure_repositories()

            # Verify repositories were created
            assert isinstance(repos, RepositoryContainer)
            assert app.startup._repositories is not None
        finally:
            # Restore originals
            app.startup._backend = original_backend
            app.startup._repositories = original_repos

    @pytest.mark.asyncio
    async def test_ensure_repositories_returns_existing(self) -> None:
        """Test that ensure_repositories returns existing."""
        import app.startup
        from app.startup import ensure_repositories

        # Store originals
        original_repos = app.startup._repositories

        try:
            # Set a mock repository container
            mock_repos = MagicMock()
            app.startup._repositories = mock_repos

            repos = await ensure_repositories()

            # Should return existing
            assert repos == mock_repos
        finally:
            # Restore original
            app.startup._repositories = original_repos


class TestStoreContextEdgeCases:
    """Test edge cases for store_context function."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_whitespace_thread_id(self) -> None:
        """Test that whitespace-only thread_id is rejected."""
        from fastmcp.exceptions import ToolError

        from app.server import store_context

        with pytest.raises(ToolError, match='thread_id cannot be empty or whitespace'):
            await store_context(
                thread_id='   ',  # Whitespace only
                source='user',
                text='Some text',
            )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_whitespace_text(self) -> None:
        """Test that whitespace-only text is rejected."""
        from fastmcp.exceptions import ToolError

        from app.server import store_context

        with pytest.raises(ToolError, match='text cannot be empty or whitespace'):
            await store_context(
                thread_id='test_thread',
                source='user',
                text='   ',  # Whitespace only
            )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_missing_image_data(self) -> None:
        """Test that images without 'data' field are rejected."""
        from fastmcp.exceptions import ToolError

        from app.server import store_context

        with pytest.raises(ToolError, match='missing required "data" field'):
            await store_context(
                thread_id='test_thread',
                source='user',
                text='Some text',
                images=[{'mime_type': 'image/png'}],  # Missing 'data'
            )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_empty_image_data(self) -> None:
        """Test that images with empty 'data' field are rejected."""
        from fastmcp.exceptions import ToolError

        from app.server import store_context

        with pytest.raises(ToolError, match='empty "data" field'):
            await store_context(
                thread_id='test_thread',
                source='user',
                text='Some text',
                images=[{'data': '', 'mime_type': 'image/png'}],
            )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_default_mime_type(self) -> None:
        """Test that missing mime_type defaults to image/png."""
        import base64

        from app.server import get_context_by_ids
        from app.server import store_context

        image_data = base64.b64encode(b'test_image').decode('utf-8')
        result = await store_context(
            thread_id='test_default_mime',
            source='user',
            text='Test image',
            images=[{'data': image_data}],  # No mime_type specified
        )

        assert result['success'] is True

        # Verify the image was stored
        context = await get_context_by_ids(
            context_ids=[result['context_id']],
            include_images=True,
        )
        assert len(context) == 1
        # Convert TypedDict to regular dict for test assertion
        entry = dict(context[0])
        assert entry['content_type'] == 'multimodal'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_failed_store_returns_null_id(self) -> None:
        """Test that failed store with null context_id raises ToolError."""
        from unittest.mock import AsyncMock

        from fastmcp.exceptions import ToolError

        from app.server import store_context

        # Mock the repository to return (None, False)
        with patch('app.tools.context.ensure_repositories') as mock_ensure:
            mock_repos = MagicMock()
            mock_repos.context.store_with_deduplication = AsyncMock(return_value=(None, False))
            mock_ensure.return_value = mock_repos

            with pytest.raises(ToolError, match='Failed to store context'):
                await store_context(
                    thread_id='test_thread',
                    source='user',
                    text='Some text',
                )


class TestUpdateContextEdgeCases:
    """Test edge cases for update_context function."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_no_fields_provided(self) -> None:
        """Test that update with no fields raises ToolError."""
        from fastmcp.exceptions import ToolError

        from app.server import store_context
        from app.server import update_context

        # First create a context
        result = await store_context(
            thread_id='test_update',
            source='user',
            text='Original text',
        )
        context_id = result['context_id']

        # Try to update with no fields
        with pytest.raises(ToolError, match='At least one field must be provided'):
            await update_context(context_id=context_id)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_whitespace_text(self) -> None:
        """Test that update with whitespace-only text is rejected."""
        from fastmcp.exceptions import ToolError

        from app.server import store_context
        from app.server import update_context

        # First create a context
        result = await store_context(
            thread_id='test_update_ws',
            source='user',
            text='Original text',
        )
        context_id = result['context_id']

        # Try to update with whitespace text
        with pytest.raises(ToolError, match='text cannot be empty'):
            await update_context(context_id=context_id, text='   ')

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_nonexistent_id(self) -> None:
        """Test that update of nonexistent context raises ToolError."""
        from fastmcp.exceptions import ToolError

        from app.server import update_context

        with pytest.raises(ToolError, match='not found'):
            await update_context(context_id=999999, text='New text')

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_clear_images(self) -> None:
        """Test that setting images to empty list removes all images."""
        import base64

        from app.server import get_context_by_ids
        from app.server import store_context
        from app.server import update_context

        image_data = base64.b64encode(b'test').decode('utf-8')

        # Create with image
        result = await store_context(
            thread_id='test_clear_images',
            source='user',
            text='With image',
            images=[{'data': image_data, 'mime_type': 'image/png'}],
        )
        context_id = result['context_id']

        # Update with empty images list
        update_result = await update_context(context_id=context_id, images=[])

        assert update_result['success'] is True
        assert 'images' in update_result['updated_fields']
        assert 'content_type' in update_result['updated_fields']

        # Verify images were removed
        context = await get_context_by_ids(
            context_ids=[context_id],
            include_images=True,
        )
        # Convert TypedDict to regular dict for test assertion
        entry = dict(context[0])
        assert entry['content_type'] == 'text'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_context_image_size_limit(self) -> None:
        """Test that oversized images in update are rejected."""
        import base64

        from fastmcp.exceptions import ToolError

        from app.server import store_context
        from app.server import update_context

        # Create a context
        result = await store_context(
            thread_id='test_image_limit',
            source='user',
            text='Original',
        )
        context_id = result['context_id']

        # Create oversized image data (6MB)
        large_data = base64.b64encode(b'x' * (6 * 1024 * 1024)).decode('utf-8')

        with pytest.raises(ToolError, match='exceeds size limit'):
            await update_context(
                context_id=context_id,
                images=[{'data': large_data, 'mime_type': 'image/png'}],
            )


class TestDeleteContextEdgeCases:
    """Test edge cases for delete_context function."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_delete_empty_thread(self) -> None:
        """Test deleting from nonexistent thread returns 0."""
        from app.server import delete_context

        result = await delete_context(thread_id='nonexistent_thread_xyz')

        assert result['success'] is True
        assert result['deleted_count'] == 0


class TestSemanticSearchNotAvailable:
    """Test semantic search tool when not available."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_semantic_search_not_available(self) -> None:
        """Test that semantic_search_context raises error when service not available."""
        from fastmcp.exceptions import ToolError

        import app.startup
        from app.server import semantic_search_context

        # Store original
        original_service = app.startup._embedding_provider

        try:
            # Clear the embedding service
            app.startup._embedding_provider = None

            with pytest.raises(ToolError, match='Semantic search is not available'):
                await semantic_search_context(query='test query', limit=20)
        finally:
            # Restore original
            app.startup._embedding_provider = original_service


class TestFormatExceptionMessage:
    """Tests for format_exception_message()."""

    def test_formats_exception_with_message(self) -> None:
        """Test exception with non-empty str()."""
        from app.migrations import format_exception_message

        error = ValueError('Something went wrong')
        result = format_exception_message(error)

        assert result == 'Something went wrong'

    def test_handles_empty_str_exception(self) -> None:
        """Test exception with empty str() falls back to repr."""
        from typing import override

        from app.migrations import format_exception_message

        # Create an exception subclass that returns empty string
        class EmptyStrError(Exception):
            @override
            def __str__(self) -> str:
                return ''

        error = EmptyStrError()
        result = format_exception_message(error)

        # Should fall back to repr
        assert 'EmptyStrError' in result

    def test_handles_exception_with_repr(self) -> None:
        """Test exception uses repr when str is empty."""
        from typing import override

        from app.migrations import format_exception_message

        class CustomError(Exception):
            @override
            def __str__(self) -> str:
                return ''

            @override
            def __repr__(self) -> str:
                return 'CustomError(custom repr)'

        error = CustomError()
        result = format_exception_message(error)

        assert result == 'CustomError(custom repr)'

    def test_standard_exceptions(self) -> None:
        """Test formatting of standard Python exceptions."""
        from app.migrations import format_exception_message

        # Test various standard exceptions
        exceptions = [
            (ValueError('value error'), 'value error'),
            (TypeError('type error'), 'type error'),
            (RuntimeError('runtime error'), 'runtime error'),
            (KeyError('missing_key'), "'missing_key'"),
        ]

        for error, expected in exceptions:
            result = format_exception_message(error)
            assert expected in result


class TestEstimateMigrationTime:
    """Tests for estimate_migration_time()."""

    def test_small_dataset(self) -> None:
        """Test <= 1000 records returns 2 seconds."""
        from app.migrations import estimate_migration_time

        assert estimate_migration_time(0) == 2
        assert estimate_migration_time(500) == 2
        assert estimate_migration_time(1000) == 2

    def test_medium_dataset(self) -> None:
        """Test 1000-10000 records returns 15 seconds."""
        from app.migrations import estimate_migration_time

        assert estimate_migration_time(1001) == 15
        assert estimate_migration_time(5000) == 15
        assert estimate_migration_time(10000) == 15

    def test_large_dataset(self) -> None:
        """Test 10000-100000 records returns 120 seconds."""
        from app.migrations import estimate_migration_time

        assert estimate_migration_time(10001) == 120
        assert estimate_migration_time(50000) == 120
        assert estimate_migration_time(100000) == 120

    def test_very_large_dataset(self) -> None:
        """Test 100000-1000000 records returns 600 seconds."""
        from app.migrations import estimate_migration_time

        assert estimate_migration_time(100001) == 600
        assert estimate_migration_time(500000) == 600
        assert estimate_migration_time(1000000) == 600

    def test_massive_dataset(self) -> None:
        """Test > 1000000 records returns 1200 seconds."""
        from app.migrations import estimate_migration_time

        assert estimate_migration_time(1000001) == 1200
        assert estimate_migration_time(5000000) == 1200
        assert estimate_migration_time(10000000) == 1200


class TestValidatePoolTimeoutForEmbedding:
    """Tests for validate_pool_timeout_for_embedding()."""

    def test_logs_warning_when_timeout_too_short(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning logged when pool timeout insufficient."""
        import logging

        from app.startup.validation import validate_pool_timeout_for_embedding

        caplog.set_level(logging.INFO)  # Function logs at INFO level

        # Create settings where pool timeout < embedding timeout + retry delays
        mock_storage_settings = MagicMock()
        mock_storage_settings.postgresql_pool_timeout_s = 5.0  # Short timeout

        mock_embedding_settings = MagicMock()
        mock_embedding_settings.timeout_s = 60.0  # Long embedding timeout
        mock_embedding_settings.retry_max_attempts = 5
        mock_embedding_settings.retry_base_delay_s = 1.0

        mock_settings = MagicMock()
        mock_settings.storage = mock_storage_settings
        mock_settings.embedding = mock_embedding_settings
        mock_settings.semantic_search.enabled = True

        with patch('app.startup.validation.settings', mock_settings):
            validate_pool_timeout_for_embedding()

        # Should have logged a message about timeout
        assert any('timeout' in record.message.lower() for record in caplog.records)

    def test_no_warning_when_timeout_sufficient(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test no warning when pool timeout adequate."""
        import logging

        from app.startup.validation import validate_pool_timeout_for_embedding

        caplog.set_level(logging.INFO)

        # Create settings where pool timeout > embedding timeout + retry delays
        mock_storage_settings = MagicMock()
        mock_storage_settings.postgresql_pool_timeout_s = 400.0  # Long timeout

        mock_embedding_settings = MagicMock()
        mock_embedding_settings.timeout_s = 60.0
        mock_embedding_settings.retry_max_attempts = 5
        mock_embedding_settings.retry_base_delay_s = 1.0

        mock_settings = MagicMock()
        mock_settings.storage = mock_storage_settings
        mock_settings.embedding = mock_embedding_settings
        mock_settings.semantic_search.enabled = True

        with patch('app.startup.validation.settings', mock_settings):
            validate_pool_timeout_for_embedding()

        # Should not have logged message about timeout being too short
        timeout_messages = [r for r in caplog.records if 'below recommended' in r.message.lower()]
        assert len(timeout_messages) == 0

    def test_skipped_when_semantic_search_disabled(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test no-op when ENABLE_SEMANTIC_SEARCH=false."""
        import logging

        from app.startup.validation import validate_pool_timeout_for_embedding

        caplog.set_level(logging.WARNING)

        mock_settings = MagicMock()
        mock_settings.semantic_search.enabled = False

        with patch('app.startup.validation.settings', mock_settings):
            validate_pool_timeout_for_embedding()

        # Should not have logged anything about timeout
        timeout_warnings = [r for r in caplog.records if 'timeout' in r.message.lower()]
        assert len(timeout_warnings) == 0


class TestGetServerVersion:
    """Tests for _get_server_version()."""

    def test_returns_version_string(self) -> None:
        """Test version is returned as a string."""
        from app.server import _get_server_version

        version = _get_server_version()

        assert isinstance(version, str)
        assert len(version) > 0

    def test_returns_version_from_package(self) -> None:
        """Test version from installed package metadata."""
        from app.server import SERVER_VERSION
        from app.server import _get_server_version

        # The cached version should match a fresh call
        version = _get_server_version()

        # Both should be valid version strings or 'unknown'
        assert isinstance(version, str)
        assert isinstance(SERVER_VERSION, str)

    def test_fallback_to_pyproject_toml(self) -> None:
        """Test fallback when package not installed."""
        from importlib.metadata import PackageNotFoundError

        from app.server import _get_server_version

        # Raise PackageNotFoundError to force fallback to pyproject.toml
        with patch('app.server.pkg_version', side_effect=PackageNotFoundError('mcp-context-server')):
            version = _get_server_version()

            # Should return a version string or 'unknown'
            assert isinstance(version, str)
            # Either a valid semver pattern or 'unknown'
            assert version == 'unknown' or '.' in version

    def test_returns_unknown_on_failure(self) -> None:
        """Test 'unknown' when all methods fail."""
        from importlib.metadata import PackageNotFoundError

        from app.server import _get_server_version

        # Mock both pkg_version and Path operations to fail
        with (
            patch('app.server.pkg_version', side_effect=PackageNotFoundError('mcp-context-server')),
            patch('pathlib.Path.exists', return_value=False),
        ):
            version = _get_server_version()

            assert version == 'unknown'

    def test_server_version_passed_to_fastmcp(self) -> None:
        """Test SERVER_VERSION is passed to FastMCP constructor for MCP protocol handshake."""
        from app.server import SERVER_VERSION

        mock_fastmcp_cls = MagicMock()
        mock_instance = mock_fastmcp_cls.return_value
        mock_instance.run = MagicMock(side_effect=SystemExit(0))
        mock_instance.custom_route = MagicMock(return_value=lambda f: f)

        with (
            patch('app.server.FastMCP', mock_fastmcp_cls),
            patch('app.server.create_auth_provider', return_value=None),
        ):
            from app.server import main

            with pytest.raises(SystemExit):
                main()

        mock_fastmcp_cls.assert_called_once()
        call_kwargs = mock_fastmcp_cls.call_args.kwargs
        assert 'version' in call_kwargs, 'version parameter not passed to FastMCP constructor'
        assert call_kwargs['version'] == SERVER_VERSION, (
            f'Expected version={SERVER_VERSION!r}, got {call_kwargs["version"]!r}'
        )
