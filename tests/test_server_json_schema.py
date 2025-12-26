"""
Tests for validating server.json against the official MCP Registry JSON Schema.

This module fetches the MCP Registry JSON Schema and validates the local server.json
file to ensure compliance with the official specification. This helps catch configuration
errors before deployment and ensures the server can be properly discovered by MCP clients.

Schema URL: https://raw.githubusercontent.com/modelcontextprotocol/registry/refs/heads/main/docs/reference/server-json/server.schema.json
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError

# Schema URL for the official MCP Registry JSON Schema
MCP_SCHEMA_URL = (
    'https://raw.githubusercontent.com/modelcontextprotocol/registry/'
    'refs/heads/main/docs/reference/server-json/server.schema.json'
)

# Path to local server.json file (relative to repository root)
PROJECT_ROOT = Path(__file__).parent.parent
SERVER_JSON_PATH = PROJECT_ROOT / 'server.json'


def fetch_schema_from_url(url: str, timeout: float = 30.0) -> dict[str, Any]:
    """
    Fetch JSON Schema from a URL.

    Uses urllib to fetch the schema, avoiding additional dependencies.
    Network errors are raised to be handled by the caller.

    Args:
        url: The URL to fetch the schema from.
        timeout: Request timeout in seconds.

    Returns:
        The parsed JSON schema as a dictionary.
    """
    request = urllib.request.Request(
        url,
        headers={'User-Agent': 'mcp-context-server-tests/1.0'},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def load_server_json() -> dict[str, Any]:
    """
    Load the local server.json file.

    Returns:
        The parsed server.json content as a dictionary.
    """
    return json.loads(SERVER_JSON_PATH.read_text(encoding='utf-8'))


def format_validation_error(error: ValidationError) -> str:
    """
    Format a jsonschema ValidationError into a human-readable string.

    Provides clear information about what failed and where in the document.

    Args:
        error: The ValidationError to format.

    Returns:
        A formatted error message string.
    """
    path = ' -> '.join(str(p) for p in error.absolute_path) if error.absolute_path else '(root)'
    return f'Path: {path}\n  Error: {error.message}'


class TestServerJsonExists:
    """Tests to verify server.json file exists and is valid JSON."""

    def test_server_json_file_exists(self) -> None:
        """Verify that server.json exists in the project root."""
        assert SERVER_JSON_PATH.exists(), (
            f'server.json not found at expected location: {SERVER_JSON_PATH}\n'
            'The server.json file is required for MCP client discovery.'
        )

    def test_server_json_is_valid_json(self) -> None:
        """Verify that server.json contains valid JSON syntax."""
        assert SERVER_JSON_PATH.exists(), f'server.json not found: {SERVER_JSON_PATH}'

        try:
            load_server_json()
        except json.JSONDecodeError as e:
            pytest.fail(f'server.json contains invalid JSON syntax:\n  Line {e.lineno}, Column {e.colno}: {e.msg}')


class TestServerJsonSchemaValidation:
    """Tests to validate server.json against the official MCP Registry JSON Schema."""

    @pytest.fixture
    def mcp_schema(self) -> dict[str, Any]:
        """Fetch the MCP Registry JSON Schema (skips test if network unavailable)."""
        try:
            return fetch_schema_from_url(MCP_SCHEMA_URL)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            pytest.skip(f'Could not fetch MCP schema (network unavailable): {e}')

    @pytest.fixture
    def server_json_content(self) -> dict[str, Any]:
        """Load the local server.json content."""
        if not SERVER_JSON_PATH.exists():
            pytest.fail(f'server.json not found: {SERVER_JSON_PATH}')
        return load_server_json()

    def test_server_json_validates_against_mcp_schema(
        self,
        mcp_schema: dict[str, Any],
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Validate server.json against the official MCP Registry JSON Schema.

        This is the primary validation test that ensures the server.json file
        complies with the MCP specification.
        """
        # Use Draft7Validator as the MCP schema uses draft-07
        validator = Draft7Validator(mcp_schema)

        # Collect all validation errors for comprehensive reporting
        errors = list(validator.iter_errors(server_json_content))

        if errors:
            error_messages = [format_validation_error(e) for e in errors]
            pytest.fail(
                f'server.json failed validation against MCP Registry schema.\n'
                f'Found {len(errors)} error(s):\n\n' + '\n\n'.join(error_messages),
            )

    def test_server_json_transport_is_stdio(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Verify transport type is 'stdio' for this server.

        The MCP Context Server uses stdio transport for communication.
        """
        packages = server_json_content.get('packages', [])

        for i, package in enumerate(packages):
            transport = package.get('transport', {})
            transport_type = transport.get('type')

            if transport_type != 'stdio':
                pytest.fail(
                    f'Package at index {i} has unexpected transport type: {transport_type}\n'
                    f'Expected: stdio',
                )

    def test_server_json_environment_variables_match_settings(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Verify all env vars from settings.py are in server.json.

        Uses recursive detection to automatically find ALL environment variable aliases
        from AppSettings and all nested settings classes (StorageSettings, TransportSettings,
        and any future additions). This ensures server.json stays in sync with application
        configuration without manual maintenance.
        """
        import types
        from typing import ClassVar
        from typing import Union
        from typing import get_args
        from typing import get_origin
        from typing import get_type_hints

        from pydantic import BaseModel

        from app.settings import AppSettings

        def _extract_model_types(annotation: type | object) -> list[type[BaseModel]]:
            """Extract all BaseModel subclasses from a type annotation.

            Handles:
            - Direct types: SomeSettings -> [SomeSettings]
            - Union types: SomeSettings | None -> [SomeSettings]
            - typing.Union: Union[SomeSettings, Other] -> [SomeSettings, Other] (if BaseModel)
            - Nested generics: list[SomeSettings] -> [SomeSettings]
            - ClassVar: ClassVar[...] -> [] (skipped)

            Returns:
                List of BaseModel subclasses found in the annotation.
            """
            origin = get_origin(annotation)

            # Skip ClassVar fields (like model_config)
            if origin is ClassVar:
                return []

            # Handle Union types (including Python 3.10+ X | Y syntax)
            if origin is Union or origin is types.UnionType:
                result: list[type[BaseModel]] = []
                for arg in get_args(annotation):
                    if arg is type(None):
                        continue  # Skip NoneType
                    result.extend(_extract_model_types(arg))
                return result

            # Handle other generic types (list[T], dict[K,V], etc.)
            if origin is not None:
                result = []
                for arg in get_args(annotation):
                    result.extend(_extract_model_types(arg))
                return result

            # Direct type - check if it's a BaseModel subclass
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                return [annotation]

            return []

        def collect_all_env_aliases(
            settings_cls: type[BaseModel],
            visited: set[type[BaseModel]] | None = None,
        ) -> set[str]:
            """Recursively collect all environment variable aliases from a settings class.

            This function traverses the settings class and all nested BaseModel subclasses
            to find all fields with environment variable aliases. It handles:
            - Direct class fields with aliases
            - Nested settings classes (fields typed as BaseModel subclasses)
            - Optional/Union types (e.g., SomeSettings | None)
            - Circular references (via visited set)

            Returns:
                Set of environment variable alias names found in the settings class hierarchy.
            """
            if visited is None:
                visited = set()

            # Prevent infinite recursion on circular references
            if settings_cls in visited:
                return set()
            visited.add(settings_cls)

            aliases: set[str] = set()

            # Collect aliases from this class's fields
            for field_info in settings_cls.model_fields.values():
                if field_info.alias:
                    aliases.add(field_info.alias)

            # Resolve type annotations (handles 'from __future__ import annotations')
            try:
                hints = get_type_hints(settings_cls)
            except Exception:
                # If type hints can't be resolved, fall back to just this class
                return aliases

            # Check each field's type for nested settings classes
            for annotation in hints.values():
                nested_types = _extract_model_types(annotation)
                for nested_type in nested_types:
                    if nested_type not in visited:
                        aliases.update(collect_all_env_aliases(nested_type, visited))

            return aliases

        # Automatically collect all env vars from AppSettings and ALL nested classes
        settings_env_vars = collect_all_env_aliases(AppSettings)

        # Get env vars from server.json
        server_json_env_vars = {
            env['name']
            for pkg in server_json_content.get('packages', [])
            for env in pkg.get('environmentVariables', [])
        }

        # Check for missing
        missing = settings_env_vars - server_json_env_vars
        if missing:
            pytest.fail(
                f'Environment variables defined in settings.py but missing from server.json:\n'
                f'{sorted(missing)}',
            )

    def test_server_json_schema_matches_schema_id(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Verify $schema URL is valid and self-consistent.

        Fetches the schema from the URL specified in server.json's $schema field,
        then verifies that the fetched schema's $id matches the URL we used.
        This ensures:
        1. The $schema URL actually exists and is fetchable
        2. The schema is self-consistent ($id matches its hosting URL)
        3. No hardcoded versions required - test is future-proof
        """
        schema_url = server_json_content.get('$schema')
        if not schema_url:
            pytest.fail('server.json missing $schema field')

        try:
            # Fetch from the URL in server.json, NOT from GitHub
            actual_schema = fetch_schema_from_url(schema_url)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            pytest.fail(
                f'Cannot fetch schema from $schema URL.\n'
                f'  URL: {schema_url}\n'
                f'  Error: {e}\n\n'
                f'The URL may be deprecated or invalid.\n'
                f'Check MCP Registry changelog for current schema URL:\n'
                f'https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/CHANGELOG.md',
            )

        schema_id = actual_schema.get('$id')

        if schema_url != schema_id:
            pytest.fail(
                f'$schema in server.json does not match $id in fetched schema.\n'
                f'  server.json $schema: {schema_url}\n'
                f'  Fetched schema $id: {schema_id}\n\n'
                f'This indicates a schema hosting issue or incorrect URL.',
            )

    def test_server_json_repository_source_is_valid(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Verify repository.source is a valid value for MCP Registry.

        The MCP schema defines repository.source as a string without enum constraint,
        but the MCP Registry only accepts certain source control providers. This test
        ensures we use a known valid value.
        """
        valid_sources = {'github', 'gitlab', 'bitbucket'}
        source = server_json_content.get('repository', {}).get('source')

        if source not in valid_sources:
            pytest.fail(
                f'repository.source "{source}" is not a known valid value.\n'
                f'  Valid values: {sorted(valid_sources)}',
            )

    def test_server_json_registry_type_is_valid(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Verify packages[].registryType is a valid value.

        The MCP schema defines registryType as a string without enum constraint,
        but the MCP Registry only accepts certain package registries. This test
        ensures we use a known valid value.
        """
        valid_types = {'pypi', 'npm', 'cargo', 'go', 'oci', 'nuget', 'mcpb'}

        for i, pkg in enumerate(server_json_content.get('packages', [])):
            registry_type = pkg.get('registryType')
            if registry_type not in valid_types:
                pytest.fail(
                    f'Package {i} has invalid registryType "{registry_type}".\n'
                    f'  Valid values: {sorted(valid_types)}',
                )


class TestServerJsonVersionSync:
    """Tests to verify version synchronization between server.json and pyproject.toml."""

    def test_server_json_version_matches_pyproject(self) -> None:
        """
        Verify server.json version matches pyproject.toml version.

        Both files must have identical version strings to ensure consistency
        in package distribution and registry metadata.
        """
        # Load server.json
        server_json = load_server_json()
        server_version = server_json.get('version')

        # Load pyproject.toml and extract version
        pyproject_path = PROJECT_ROOT / 'pyproject.toml'
        if not pyproject_path.exists():
            pytest.skip('pyproject.toml not found')

        pyproject_content = pyproject_path.read_text(encoding='utf-8')

        # Simple extraction of version from pyproject.toml
        # Looks for: version = "x.y.z"
        version_match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject_content, re.MULTILINE)
        if not version_match:
            pytest.skip('Could not extract version from pyproject.toml')

        pyproject_version = version_match.group(1)

        assert server_version == pyproject_version, (
            f'Version mismatch between server.json and pyproject.toml:\n'
            f'  server.json version: {server_version}\n'
            f'  pyproject.toml version: {pyproject_version}\n'
            'Update server.json to match pyproject.toml version.'
        )

    def test_server_json_package_version_matches_top_level(self) -> None:
        """
        Verify package version matches top-level version in server.json.

        Both the top-level version and package-level version must be identical.
        """
        server_json = load_server_json()

        top_level_version = server_json.get('version')
        packages = server_json.get('packages', [])

        for i, package in enumerate(packages):
            package_version = package.get('version')

            if package_version and package_version != top_level_version:
                pytest.fail(
                    f'Version mismatch in server.json:\n'
                    f'  Top-level version: {top_level_version}\n'
                    f'  Package[{i}] version: {package_version}\n'
                    'Both versions must be identical.',
                )
