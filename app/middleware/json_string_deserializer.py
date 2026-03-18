"""Schema-aware middleware to fix MCP client parameter serialization.

Some MCP clients (including Claude Code) intermittently serialize list and dict
parameters as JSON strings instead of native types. This middleware intercepts
tool calls and deserializes string-typed arguments that the tool's JSON Schema
expects as array or object types.

Upstream issues:
- Claude Code #22394 (closed NOT_PLANNED, 2026-02-01)
- FastMCP #932
- Claude Code #5504, #4192, #3084, #26094

This middleware can be removed when upstream clients fix their serialization.
"""

import contextlib
import json
import logging
from collections.abc import Sequence
from typing import Any
from typing import cast
from typing import override

import mcp.types as mt
from fastmcp.server.middleware.middleware import CallNext
from fastmcp.server.middleware.middleware import Middleware
from fastmcp.server.middleware.middleware import MiddlewareContext
from fastmcp.tools.tool import Tool
from fastmcp.tools.tool import ToolResult

logger = logging.getLogger(__name__)

_COMPLEX_TYPES = frozenset(('array', 'object'))


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a JSON Schema $ref to its definition.

    Only handles local references in the form '#/$defs/<name>'.

    Returns:
        Resolved schema dict, or empty dict for unresolvable references.
    """
    prefix = '#/$defs/'
    if ref.startswith(prefix):
        return cast(dict[str, Any], defs.get(ref[len(prefix):], {}))
    return {}


def _is_complex_type(schema: dict[str, Any], defs: dict[str, Any]) -> bool:
    """Determine if a JSON Schema property expects an array or object type.

    Handles direct type declarations, $ref to $defs, and anyOf variants
    (used by Optional[list[...]] and Optional[dict[...]] parameters).

    Returns:
        True if the schema describes an array or object type.
    """
    if schema.get('type') in _COMPLEX_TYPES:
        return True
    if '$ref' in schema:
        resolved = _resolve_ref(schema['$ref'], defs)
        if resolved.get('type') in _COMPLEX_TYPES:
            return True
    for variant in schema.get('anyOf', []):
        if variant.get('type') in _COMPLEX_TYPES:
            return True
        if '$ref' in variant:
            resolved = _resolve_ref(variant['$ref'], defs)
            if resolved.get('type') in _COMPLEX_TYPES:
                return True
    return False


def build_schema_map(tools: Sequence[Tool]) -> dict[str, set[str]]:
    """Build a static map of tool parameters that expect complex types.

    Inspects each tool's JSON Schema to identify parameters with type
    'array' or 'object' (including Optional variants via anyOf and
    $ref definitions).

    Args:
        tools: Tool objects from FastMCP.list_tools()

    Returns:
        Mapping of tool_name -> set of parameter names expecting array/object
    """
    schema_map: dict[str, set[str]] = {}
    for tool in tools:
        complex_params: set[str] = set()
        defs = tool.parameters.get('$defs', {})
        properties = tool.parameters.get('properties', {})
        for param_name, param_schema in properties.items():
            if _is_complex_type(param_schema, defs):
                complex_params.add(param_name)
        if complex_params:
            schema_map[tool.name] = complex_params
    return schema_map


class JsonStringDeserializerMiddleware(Middleware):
    """Deserialize stringified JSON parameters based on tool schema.

    Only attempts deserialization for parameters that the tool's JSON Schema
    declares as array or object types. String parameters (like text, query)
    are never touched, preventing over-deserialization.

    Handles double-encoding: if json.loads() returns a string that is itself
    valid JSON for the expected type, it deserializes again.

    Args:
        schema_map: Mapping from tool name to set of parameter names
                    that expect complex types (array/object)
    """

    def __init__(self, schema_map: dict[str, set[str]]) -> None:
        self._schema_map = schema_map

    @override
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        tool_name = context.message.name
        arguments = context.message.arguments

        complex_params = self._schema_map.get(tool_name)
        if complex_params and arguments:
            modified = False
            for param_name in complex_params:
                if param_name not in arguments:
                    continue
                value = arguments[param_name]
                if not isinstance(value, str):
                    continue
                try:
                    parsed = json.loads(value)
                    # Handle double-encoding: json.loads returns string -> try again
                    if isinstance(parsed, str):
                        with contextlib.suppress(json.JSONDecodeError, ValueError):
                            parsed = json.loads(parsed)
                    # Only accept if the result is actually a list or dict
                    if isinstance(parsed, (list, dict)):
                        arguments[param_name] = parsed
                        modified = True
                except (json.JSONDecodeError, ValueError):
                    pass  # Not valid JSON, leave as-is

            if modified:
                logger.debug(
                    'Deserialized stringified parameters for tool %s',
                    tool_name,
                )

        return await call_next(context)
