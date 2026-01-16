"""
Unified error classification system for server startup and runtime.

This module provides a structured approach to error handling that enables
external supervisors (Docker, Kubernetes, systemd) to make informed restart
decisions based on error categories.

Exit codes follow BSD sysexits.h conventions:
- 0: Success
- 1: General error (unknown, may retry)
- 69 (EX_UNAVAILABLE): Service unavailable (may retry with backoff)
- 78 (EX_CONFIG): Configuration error (do NOT retry)
"""

from enum import StrEnum


class ErrorCategory(StrEnum):
    """Classification of error severity and recoverability.

    Used to determine appropriate exit codes and supervisor behavior.
    """

    CONFIGURATION = 'configuration'  # Missing config, wrong env vars - NEVER retry
    DEPENDENCY = 'dependency'  # External service unavailable - MAY retry with limits


class ConfigurationError(Exception):
    """Raised for unrecoverable configuration errors.

    These errors indicate problems that require human intervention:
    - Missing or invalid environment variables
    - Missing required packages/dependencies
    - Invalid configuration values

    Supervisors should NOT automatically restart on this error.
    Exit code: 78 (EX_CONFIG from BSD sysexits.h)
    """

    EXIT_CODE = 78  # BSD sysexits.h EX_CONFIG


class DependencyError(Exception):
    """Raised when required external dependency is unavailable.

    These errors indicate problems that MAY resolve with time:
    - Service not running but may start later
    - Temporary network issues
    - Resource temporarily unavailable

    Supervisors MAY retry with exponential backoff.
    Exit code: 69 (EX_UNAVAILABLE from BSD sysexits.h)
    """

    EXIT_CODE = 69  # BSD sysexits.h EX_UNAVAILABLE


def classify_provider_error(reason: str) -> type[ConfigurationError] | type[DependencyError]:
    """Classify a provider check failure into the appropriate error type.

    ConfigurationError (exit 78 - never retry):
    - Package/dependency installation issues
    - Environment variable configuration
    - Invalid provider configuration
    - Model not found (requires ollama pull or config fix - human intervention)

    DependencyError (exit 69 - may retry):
    - Service temporarily unavailable
    - Network connectivity issues
    - Service startup timing issues

    Args:
        reason: The failure reason string from ProviderCheckResult

    Returns:
        ConfigurationError class for configuration issues,
        DependencyError class for transient/external issues.
    """
    reason_lower = reason.lower()

    # Configuration errors: require human intervention to fix
    config_indicators = [
        # Package installation issues
        'not installed',
        'package not available',
        # Environment variable issues
        'not set',
        'environment variable',
        # Provider configuration
        'unknown provider',
        # MODEL AVAILABILITY - requires human action: ollama pull or fix EMBEDDING_MODEL
        'not found',  # Catches "model 'x' not found", "API not found", etc.
        'does not exist',
        'model not available',  # Explicit model availability
        # HTTP 404 errors indicate resource doesn't exist (won't auto-appear)
        '404',
        'status code: 404',
    ]

    for indicator in config_indicators:
        if indicator in reason_lower:
            return ConfigurationError

    # Dependency errors: MAY resolve through retry (service startup, network)
    # Examples: "Connection refused", "timeout", "503 Service Unavailable"
    return DependencyError
