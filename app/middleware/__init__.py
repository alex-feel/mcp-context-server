"""Schema-aware middleware for MCP client compatibility."""

from app.middleware.json_string_deserializer import JsonStringDeserializerMiddleware
from app.middleware.json_string_deserializer import build_schema_map

__all__ = [
    'JsonStringDeserializerMiddleware',
    'build_schema_map',
]
