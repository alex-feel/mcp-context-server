"""
Tests for validating server.json against the official MCP Registry JSON Schema.

This module fetches the MCP Registry JSON Schema and validates the local server.json
file to ensure compliance with the official specification. This helps catch configuration
errors before deployment and ensures the server can be properly discovered by MCP clients.

Schema URL: Dynamically read from server.json's $schema field (currently hosted at static.modelcontextprotocol.io).
"""


import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from typing import get_args
from typing import get_origin

import pytest
from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError
from pydantic import BaseModel
from pydantic import SecretStr
from pydantic.fields import FieldInfo

from app.settings import AppSettings

# Path to local server.json file (relative to repository root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
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


def _extract_model_types(annotation: object) -> list[type[BaseModel]]:
    """Extract all BaseModel subclasses from a type annotation.

    Handles direct types, ``X | None`` and ``Union`` types, and nested generics such
    as ``list[SomeSettings]``.

    Returns:
        List of BaseModel subclasses found in the annotation.
    """
    if get_origin(annotation) is not None:
        result: list[type[BaseModel]] = []
        for arg in get_args(annotation):
            if arg is not type(None):
                result.extend(_extract_model_types(arg))
        return result
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return []


def _holds_secret(annotation: object) -> bool:
    """Report whether a field annotation is ``SecretStr`` or a union that contains it.

    Returns:
        True when the annotation is SecretStr or contains it as a union member.
    """
    return annotation is SecretStr or SecretStr in get_args(annotation)


def collect_settings_env_fields(
    settings_cls: type[BaseModel],
    visited: set[type[BaseModel]] | None = None,
) -> dict[str, FieldInfo]:
    """Map every environment-variable alias in a settings class hierarchy to its field.

    Traverses the settings class and every nested BaseModel subclass reachable through
    its field annotations (optional and union types included), so a new nested
    settings class is covered without maintenance. The visited set stops recursion on
    circular references.

    Returns:
        Mapping of environment variable alias to the pydantic field that declares it.
    """
    if visited is None:
        visited = set()
    if settings_cls in visited:
        return {}
    visited.add(settings_cls)

    fields: dict[str, FieldInfo] = {}
    for field_info in settings_cls.model_fields.values():
        if field_info.alias:
            fields[field_info.alias] = field_info
        for nested_type in _extract_model_types(field_info.annotation):
            fields.update(collect_settings_env_fields(nested_type, visited))
    return fields


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
        """Fetch the MCP Registry JSON Schema from the URL declared in server.json."""
        server_json = load_server_json()
        schema_url = server_json.get('$schema')
        if not schema_url:
            pytest.fail('server.json missing $schema field')

        try:
            return fetch_schema_from_url(schema_url)
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

    def test_server_json_transport_types_match_registry_types(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Verify transport types are appropriate for each package registry type.

        - PyPI packages (registryType: pypi) should use stdio transport
        - OCI packages (registryType: oci) should use streamable-http transport
        """
        packages = server_json_content.get('packages', [])

        expected_transport_by_registry: dict[str, str] = {
            'pypi': 'stdio',
            'npm': 'stdio',
            'cargo': 'stdio',
            'go': 'stdio',
            'nuget': 'stdio',
            'oci': 'streamable-http',
            'mcpb': 'streamable-http',
        }

        for i, package in enumerate(packages):
            registry_type = package.get('registryType')
            transport = package.get('transport', {})
            transport_type = transport.get('type')

            expected = expected_transport_by_registry.get(registry_type)
            if expected and transport_type != expected:
                pytest.fail(
                    f'Package at index {i} (registryType: {registry_type}) has '
                    f'unexpected transport type: {transport_type}\n'
                    f'Expected: {expected}',
                )

    def test_server_json_oci_package_has_valid_transport_url(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Verify OCI packages have valid streamable-http transport URLs.

        OCI packages with streamable-http transport must have a valid URL
        that ends with '/mcp' for the MCP endpoint.
        """
        for i, pkg in enumerate(server_json_content.get('packages', [])):
            if pkg.get('registryType') != 'oci':
                continue

            transport = pkg.get('transport', {})
            if transport.get('type') != 'streamable-http':
                continue

            url = transport.get('url')
            if not url:
                pytest.fail(
                    f'OCI package at index {i} with streamable-http transport '
                    f'is missing required "url" field.',
                )

            if not url.endswith('/mcp'):
                pytest.fail(
                    f'OCI package at index {i} has non-standard MCP endpoint URL: {url}\n'
                    f'Expected URL to end with "/mcp".',
                )

    def test_server_json_environment_variables_match_settings(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Verify all env vars from app.settings are in server.json.

        Uses recursive detection to automatically find ALL environment variable aliases
        from AppSettings and all nested settings classes (StorageSettings, TransportSettings,
        and any future additions). This ensures server.json stays in sync with application
        configuration without manual maintenance.
        """
        settings_env_vars = set(collect_settings_env_fields(AppSettings))

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
                f'Environment variables defined in app.settings but missing from server.json:\n'
                f'{sorted(missing)}',
            )

    def test_server_json_secret_flags_match_settings(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """Verify every package marks exactly the SecretStr-typed settings variables as secret.

        The MCP Registry schema asks clients to handle a variable flagged ``isSecret`` as a
        secret, so a credential field added to app.settings without the flag reaches clients
        as an ordinary value, and a flag on a plain setting hides a value that is not secret.
        """
        secret_env_vars = {
            alias
            for alias, field_info in collect_settings_env_fields(AppSettings).items()
            if _holds_secret(field_info.annotation)
        }
        for i, pkg in enumerate(server_json_content.get('packages', [])):
            registry_type = pkg.get('registryType', f'package-{i}')
            flagged = {env['name'] for env in pkg.get('environmentVariables', []) if env.get('isSecret')}
            if flagged != secret_env_vars:
                pytest.fail(
                    f'isSecret flags in the {registry_type} package differ from the SecretStr settings:\n'
                    f'  missing isSecret: {sorted(secret_env_vars - flagged)}\n'
                    f'  isSecret on a non-secret setting: {sorted(flagged - secret_env_vars)}',
                )

    def test_all_packages_have_identical_environment_variables(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Verify all packages have identical environment variable sets.

        All packages in server.json should expose the same configuration options
        to ensure consistent behavior regardless of deployment method (PyPI vs OCI).
        This prevents silent failures where one package is missing variables that
        exist in another.
        """
        packages = server_json_content.get('packages', [])

        if len(packages) < 2:
            pytest.skip('Only one package present, nothing to compare')

        # Collect environment variable sets per package
        package_env_sets: list[tuple[str, set[str]]] = []
        for i, pkg in enumerate(packages):
            registry_type = pkg.get('registryType', f'package-{i}')
            env_vars = {env['name'] for env in pkg.get('environmentVariables', [])}
            package_env_sets.append((registry_type, env_vars))

        # Compare all packages against the first one (reference)
        reference_name, reference_vars = package_env_sets[0]

        for pkg_name, pkg_vars in package_env_sets[1:]:
            missing_from_pkg = reference_vars - pkg_vars
            extra_in_pkg = pkg_vars - reference_vars

            if missing_from_pkg or extra_in_pkg:
                errors: list[str] = []
                if missing_from_pkg:
                    errors.append(
                        f'  Missing from {pkg_name} (present in {reference_name}):\n'
                        f'    {sorted(missing_from_pkg)}',
                    )
                if extra_in_pkg:
                    errors.append(
                        f'  Extra in {pkg_name} (not in {reference_name}):\n'
                        f'    {sorted(extra_in_pkg)}',
                    )
                pytest.fail(
                    'Environment variable mismatch between packages:\n'
                    + '\n'.join(errors),
                )

    def test_all_packages_have_identical_environment_variable_descriptions(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """Verify every package describes each environment variable identically.

        server.json duplicates the environment-variable list across the PyPI and OCI package
        blocks. The name-set check above catches a variable missing from one block, but not a
        DESCRIPTION corrected in one block and left stale in the other -- a recurring drift when a
        toggle's registration semantics change (e.g. a tri-state toggle whose 'true' no longer
        forces registration). Asserting description equality across packages closes that gap so
        the two blocks cannot silently diverge in wording.
        """
        packages = server_json_content.get('packages', [])
        if len(packages) < 2:
            pytest.skip('Only one package present, nothing to compare')

        def _descriptions(pkg: dict[str, Any]) -> dict[str, str]:
            return {env['name']: env.get('description', '') for env in pkg.get('environmentVariables', [])}

        reference_type = packages[0].get('registryType', 'package-0')
        reference = _descriptions(packages[0])
        for i, pkg in enumerate(packages[1:], start=1):
            pkg_type = pkg.get('registryType', f'package-{i}')
            current = _descriptions(pkg)
            mismatches = sorted(name for name in reference.keys() & current.keys() if reference[name] != current[name])
            if mismatches:
                detail = '\n'.join(
                    f'  {name}:\n    {reference_type}: {reference[name]!r}\n    {pkg_type}: {current[name]!r}'
                    for name in mismatches
                )
                pytest.fail(
                    f'Environment variable descriptions differ between {reference_type} and '
                    f'{pkg_type} (they must be identical across packages):\n{detail}',
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

    def test_oci_packages_do_not_have_registry_base_url(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Verify OCI packages don't have registryBaseUrl field (MCP Registry business rule).

        While the JSON Schema technically allows registryBaseUrl for OCI packages,
        the MCP Registry has a runtime business rule that rejects OCI packages with
        this field. OCI packages must use a canonical reference in the identifier
        instead (e.g., 'ghcr.io/owner/image:1.0.0').
        """
        for i, pkg in enumerate(server_json_content.get('packages', [])):
            if pkg.get('registryType') == 'oci':
                assert 'registryBaseUrl' not in pkg, (
                    f'OCI package at index {i} has invalid registryBaseUrl field. '
                    f'OCI packages must use canonical reference in identifier instead '
                    f'(e.g., "ghcr.io/owner/image:1.0.0").'
                )

    def test_oci_packages_do_not_have_version_field(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Verify OCI packages don't have version field (MCP Registry business rule).

        OCI packages must include the version in the identifier field instead
        (e.g., 'ghcr.io/owner/image:1.0.0'), not as a separate 'version' field.
        """
        for i, pkg in enumerate(server_json_content.get('packages', [])):
            if pkg.get('registryType') == 'oci':
                assert 'version' not in pkg, (
                    f'OCI package at index {i} has invalid version field. '
                    f'OCI packages must include version in identifier instead '
                    f'(e.g., "ghcr.io/owner/image:1.0.0").'
                )

    def test_oci_packages_have_version_in_identifier(
        self,
        server_json_content: dict[str, Any],
    ) -> None:
        """
        Verify OCI package identifiers include version tag.

        OCI identifiers must be in format: registry/namespace/image:tag
        or include a digest reference (@sha256:...).
        """
        # Pattern matches :tag or @sha256:digest at end of string
        version_pattern = re.compile(r'(:[a-zA-Z0-9][a-zA-Z0-9._-]*|@sha256:[a-f0-9]{64})$')

        for i, pkg in enumerate(server_json_content.get('packages', [])):
            if pkg.get('registryType') == 'oci':
                identifier = pkg.get('identifier', '')
                assert version_pattern.search(identifier), (
                    f'OCI package at index {i} has identifier without version tag: {identifier}. '
                    f'OCI identifiers must include version tag (e.g., ":1.0.0") or digest.'
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
