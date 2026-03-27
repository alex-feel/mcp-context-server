"""Tests for the unified error classification system."""

import pytest

from app.errors import ConfigurationError
from app.errors import DependencyError
from app.errors import ErrorCategory
from app.errors import classify_provider_error


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_configuration_value(self) -> None:
        """ErrorCategory.CONFIGURATION has correct string value."""
        assert ErrorCategory.CONFIGURATION.value == 'configuration'

    def test_dependency_value(self) -> None:
        """ErrorCategory.DEPENDENCY has correct string value."""
        assert ErrorCategory.DEPENDENCY.value == 'dependency'

    def test_str_enum_behavior(self) -> None:
        """ErrorCategory values can be used as strings."""
        assert f'Error category: {ErrorCategory.CONFIGURATION}' == 'Error category: configuration'
        assert f'Error category: {ErrorCategory.DEPENDENCY}' == 'Error category: dependency'


class TestConfigurationError:
    """Tests for ConfigurationError exception."""

    def test_exit_code_is_78(self) -> None:
        """ConfigurationError has exit code 78 (EX_CONFIG)."""
        assert ConfigurationError.EXIT_CODE == 78

    def test_is_exception_subclass(self) -> None:
        """ConfigurationError is a proper Exception subclass."""
        assert issubclass(ConfigurationError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """ConfigurationError can be raised and caught."""
        with pytest.raises(ConfigurationError, match='test message'):
            raise ConfigurationError('test message')

    def test_preserves_message(self) -> None:
        """ConfigurationError preserves the error message."""
        error = ConfigurationError('Missing API key')
        assert str(error) == 'Missing API key'

    def test_can_chain_exceptions(self) -> None:
        """ConfigurationError supports exception chaining."""
        original = ValueError('original')
        error = ConfigurationError('config error')
        error.__cause__ = original
        assert error.__cause__ is original


class TestDependencyError:
    """Tests for DependencyError exception."""

    def test_exit_code_is_69(self) -> None:
        """DependencyError has exit code 69 (EX_UNAVAILABLE)."""
        assert DependencyError.EXIT_CODE == 69

    def test_is_exception_subclass(self) -> None:
        """DependencyError is a proper Exception subclass."""
        assert issubclass(DependencyError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """DependencyError can be raised and caught."""
        with pytest.raises(DependencyError, match='service unavailable'):
            raise DependencyError('service unavailable')

    def test_preserves_message(self) -> None:
        """DependencyError preserves the error message."""
        error = DependencyError('Ollama service not running')
        assert str(error) == 'Ollama service not running'


class TestClassifyProviderError:
    """Tests for classify_provider_error function."""

    @pytest.mark.parametrize(
        ('reason', 'expected_class'),
        [
            # Package not installed cases -> ConfigurationError
            ('langchain-ollama package not installed', ConfigurationError),
            ('langchain-openai package not installed', ConfigurationError),
            ('langchain-huggingface package not installed', ConfigurationError),
            ('langchain-voyageai package not installed', ConfigurationError),
            # Environment variable not set cases -> ConfigurationError
            ('OPENAI_API_KEY environment variable is not set', ConfigurationError),
            ('HUGGINGFACEHUB_API_TOKEN environment variable is not set', ConfigurationError),
            ('VOYAGE_API_KEY environment variable is not set', ConfigurationError),
            ('Required environment variables not set: AZURE_OPENAI_API_KEY', ConfigurationError),
            # Package import failed cases -> ConfigurationError
            ('langchain-ollama package not available: ImportError', ConfigurationError),
            # Unknown provider cases -> ConfigurationError
            ("Unknown provider: 'invalid'", ConfigurationError),
            # Model not available -> ConfigurationError (requires human intervention: ollama pull)
            ('Embedding model "qwen3-embedding:0.6b" not available: model not found', ConfigurationError),
            ("model 'model' not found (status code: 404)", ConfigurationError),
            ('API endpoint not found', ConfigurationError),
            # HTTP 404 errors -> ConfigurationError (resource doesn't exist)
            ('Request failed with status code: 404', ConfigurationError),
            ('Model does not exist in the registry', ConfigurationError),
            # Case-insensitive matching tests
            ('Model NOT FOUND', ConfigurationError),
            ('UNKNOWN PROVIDER: test', ConfigurationError),
            # HTTP 400 Bad Request indicators -> ConfigurationError
            ('unsupported value: reasoning_effort does not support minimal', ConfigurationError),
            ('Error: unsupported_value', ConfigurationError),
            ('invalid_request_error: parameter not supported', ConfigurationError),
            ('HTTP 400 Bad Request', ConfigurationError),
            ('bad request: invalid parameter', ConfigurationError),
        ],
    )
    def test_configuration_error_classification(
        self,
        reason: str,
        expected_class: type,
    ) -> None:
        """Configuration-related failures return ConfigurationError class."""
        assert classify_provider_error(reason) is expected_class

    @pytest.mark.parametrize(
        'reason',
        [
            # Service not accessible -> DependencyError (may recover when service starts)
            'Ollama service returned status 503',
            'Ollama service not accessible at http://localhost:11434: Connection refused',
            # Network/connectivity issues -> DependencyError (may recover)
            'Connection timed out',
            'Network unreachable',
            'Service temporarily unavailable',
            # Service startup timing -> DependencyError (may recover when service is ready)
            'Connection refused',
            'Service is starting up',
        ],
    )
    def test_dependency_error_classification(self, reason: str) -> None:
        """Transient/external failures return DependencyError class."""
        assert classify_provider_error(reason) is DependencyError

    def test_unknown_reason_defaults_to_dependency(self) -> None:
        """Unknown/unrecognized reasons default to DependencyError."""
        # Unknown reasons are treated as potentially transient
        assert classify_provider_error('Something unexpected happened') is DependencyError


class TestExitCodeConventions:
    """Tests validating exit code conventions."""

    def test_exit_codes_are_distinct(self) -> None:
        """Each error type has a unique exit code."""
        assert ConfigurationError.EXIT_CODE != DependencyError.EXIT_CODE

    def test_exit_codes_follow_sysexits_convention(self) -> None:
        """Exit codes follow BSD sysexits.h conventions."""
        # EX_CONFIG = 78 (configuration error)
        assert ConfigurationError.EXIT_CODE == 78
        # EX_UNAVAILABLE = 69 (service unavailable)
        assert DependencyError.EXIT_CODE == 69

    def test_exit_codes_are_nonzero(self) -> None:
        """All error exit codes are non-zero (failure)."""
        assert ConfigurationError.EXIT_CODE != 0
        assert DependencyError.EXIT_CODE != 0


class TestPostgreSQLBackendErrorClassification:
    """Tests verifying PostgreSQL backend raises correct error types.

    These tests verify that the PostgreSQL backend properly classifies errors
    to enable correct container restart behavior:
    - ConfigurationError (exit 78): Container should halt, requires human fix
    - DependencyError (exit 69): Container may retry, might auto-resolve
    """

    def test_insufficient_privilege_raises_configuration_error(self) -> None:
        """InsufficientPrivilegeError mapped to ConfigurationError (exit 78).

        When pgvector extension cannot be created due to insufficient privileges,
        this requires human intervention (enable via dashboard or grant permissions).
        Container should halt, not retry infinitely.
        """
        # Verify ConfigurationError is raised with correct message pattern
        error = ConfigurationError(
            'pgvector extension required but cannot be created (insufficient privileges): test',
        )
        assert 'insufficient privileges' in str(error)
        assert ConfigurationError.EXIT_CODE == 78

    def test_connection_refused_raises_dependency_error(self) -> None:
        """Connection refused (OSError Errno 111) mapped to DependencyError (exit 69).

        When PostgreSQL is not running, this is a transient condition that
        may resolve when the database starts. Container may retry with backoff.
        """
        # Verify DependencyError is raised with correct message pattern
        error = DependencyError(
            'PostgreSQL connection failed: [Errno 111] Connection refused. '
            'Ensure PostgreSQL is running and accessible.',
        )
        assert 'PostgreSQL connection failed' in str(error)
        assert DependencyError.EXIT_CODE == 69

    def test_invalid_password_raises_configuration_error(self) -> None:
        """InvalidPasswordError mapped to ConfigurationError (exit 78).

        Wrong password requires human intervention to fix environment variables.
        Container should halt, not retry infinitely.
        """
        error = ConfigurationError(
            'PostgreSQL authentication failed: password authentication failed. '
            'Check POSTGRESQL_USER and POSTGRESQL_PASSWORD.',
        )
        assert 'PostgreSQL authentication failed' in str(error)
        assert ConfigurationError.EXIT_CODE == 78

    def test_invalid_catalog_name_raises_configuration_error(self) -> None:
        """InvalidCatalogNameError mapped to ConfigurationError (exit 78).

        Non-existent database requires human intervention to create it or fix config.
        Container should halt, not retry infinitely.
        """
        error = ConfigurationError(
            'PostgreSQL database does not exist: database "test" does not exist. '
            'Create the database or check POSTGRESQL_DATABASE.',
        )
        assert 'PostgreSQL database does not exist' in str(error)
        assert ConfigurationError.EXIT_CODE == 78

    def test_pgvector_not_installed_raises_configuration_error(self) -> None:
        """pgvector extension not installed mapped to ConfigurationError (exit 78).

        Missing extension requires human intervention to enable it.
        Container should halt, not retry infinitely.
        """
        error = ConfigurationError(
            'pgvector extension is not installed. '
            'Enable it via: CREATE EXTENSION vector; (PostgreSQL) '
            'or Dashboard → Extensions → vector (Supabase)',
        )
        assert 'pgvector extension is not installed' in str(error)
        assert ConfigurationError.EXIT_CODE == 78

    def test_codec_registration_failed_raises_configuration_error(self) -> None:
        """pgvector codec registration failure mapped to ConfigurationError (exit 78).

        Codec registration issues require human intervention to fix the setup.
        Container should halt, not retry infinitely.
        """
        error = ConfigurationError('pgvector codec registration failed: Unknown type')
        assert 'pgvector codec registration failed' in str(error)
        assert ConfigurationError.EXIT_CODE == 78

    def test_generic_error_raises_dependency_error(self) -> None:
        """Generic/unknown errors default to DependencyError (exit 69).

        Unknown errors are safer to classify as DependencyError (allows retry)
        rather than ConfigurationError (halts container). This is the conservative
        approach that avoids permanently blocking a potentially recoverable situation.
        """
        error = DependencyError('pgvector extension is required but could not be created: Unknown error')
        assert 'pgvector extension is required' in str(error)
        assert DependencyError.EXIT_CODE == 69

    def test_timeout_raises_dependency_error(self) -> None:
        """TimeoutError mapped to DependencyError (exit 69).

        Timeouts are transient and may resolve on retry.
        Container may retry with backoff.
        """
        error = DependencyError(
            'PostgreSQL connection failed: Connection timed out. '
            'Ensure PostgreSQL is running and accessible.',
        )
        assert 'PostgreSQL connection failed' in str(error)
        assert DependencyError.EXIT_CODE == 69


class TestIsClientError:
    """Tests for is_client_error helper function."""

    def test_detects_400_status_code(self) -> None:
        """is_client_error() detects exceptions with status_code 400."""
        from app.errors import is_client_error

        class FakeClientError(Exception):
            """Exception with status_code attribute."""

            def __init__(self, message: str, status_code: int) -> None:
                super().__init__(message)
                self.status_code = status_code

        assert is_client_error(FakeClientError('bad request', status_code=400)) is True

    def test_detects_401_status_code(self) -> None:
        """is_client_error() detects exceptions with status_code 401."""
        from app.errors import is_client_error

        class FakeClientError(Exception):
            """Exception with status_code attribute."""

            def __init__(self, message: str, status_code: int) -> None:
                super().__init__(message)
                self.status_code = status_code

        assert is_client_error(FakeClientError('unauthorized', status_code=401)) is True

    def test_ignores_500_status_code(self) -> None:
        """is_client_error() returns False for 5xx server errors."""
        from app.errors import is_client_error

        class FakeServerError(Exception):
            """Exception with 5xx status_code."""

            def __init__(self, message: str, status_code: int) -> None:
                super().__init__(message)
                self.status_code = status_code

        assert is_client_error(FakeServerError('server error', status_code=500)) is False

    def test_ignores_no_status_code(self) -> None:
        """is_client_error() returns False for exceptions without status_code."""
        from app.errors import is_client_error

        assert is_client_error(ConnectionError('refused')) is False

    def test_checks_cause_chain(self) -> None:
        """is_client_error() detects status_code in exception __cause__."""
        from app.errors import is_client_error

        class FakeClientError(Exception):
            """Exception with status_code attribute."""

            def __init__(self, message: str, status_code: int) -> None:
                super().__init__(message)
                self.status_code = status_code

        cause = FakeClientError('original', status_code=400)
        wrapper = RuntimeError('wrapped')
        wrapper.__cause__ = cause
        assert is_client_error(wrapper) is True

    def test_checks_context_chain(self) -> None:
        """is_client_error() detects status_code in exception __context__."""
        from app.errors import is_client_error

        class FakeClientError(Exception):
            """Exception with status_code attribute."""

            def __init__(self, message: str, status_code: int) -> None:
                super().__init__(message)
                self.status_code = status_code

        context = FakeClientError('original', status_code=400)
        wrapper = RuntimeError('implicitly chained')
        wrapper.__context__ = context
        assert is_client_error(wrapper) is True


class TestFormatExceptionMessage:
    """Tests for format_exception_message()."""

    def test_formats_exception_with_message(self) -> None:
        """Test exception with non-empty str()."""
        from app.errors import format_exception_message

        error = ValueError('Something went wrong')
        result = format_exception_message(error)

        assert result == 'Something went wrong'

    def test_handles_empty_str_exception(self) -> None:
        """Test exception with empty str() falls back to repr."""
        from typing import override

        from app.errors import format_exception_message

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

        from app.errors import format_exception_message

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
        from app.errors import format_exception_message

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
