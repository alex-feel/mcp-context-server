"""Every metadata-filter shape the query builder renders, for the SQL-shape tests.

The cast guard renders each path's PostgreSQL SQL and the totality matrix
evaluates it against stored values, so both cover the same set: the flat and the
nested key path, crossed with every operator and every value kind the filter
model accepts for it (int, float, bool and string scalars; IN/NOT_IN lists of one
type and of mixed types), in the case-insensitive default and, where it renders
differently, the case-sensitive variant, plus the simple ``metadata={}``
equality path for each scalar kind.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder

type FilterValue = str | int | float | bool | list[str | int | float | bool] | None

# Flat and nested key paths; the nested one renders the #>/#>> array-literal accessors.
KEY_PATHS: dict[str, str] = {'flat': 'k', 'nested': 'a.b'}

SCALAR_VALUES: dict[str, str | int | float | bool] = {
    'int': 42,
    'float': 3.5,
    'bool': True,
    'string': 'hidden-SECRET-4711',
}

LIST_VALUES: dict[str, list[str | int | float | bool]] = {
    'int_list': [42, -7],
    'float_list': [3.5, 1.5],
    'bool_list': [True, False],
    'string_list': ['42', 'hidden-SECRET-4711'],
    'int_float_list': [42, 3.5],
    'mixed_list': [42, 3.5, True, 'hidden-SECRET-4711'],
}


@dataclass(frozen=True)
class MetadataPath:
    """One filter the builder renders: an advanced filter, or the simple equality path.

    Attributes:
        case_id: Stable test id naming the key path, operator, value kind and case variant.
        key: The dotted metadata key path.
        operator: The advanced-filter operator, or None for ``add_simple_filter``.
        value: The filter value as the filter model normalized it.
        case_sensitive: The advanced filter's case sensitivity (ignored by the simple path).
    """

    case_id: str
    key: str
    operator: MetadataOperator | None
    value: FilterValue
    case_sensitive: bool

    def render(self, backend_type: str, *, param_offset: int = 0) -> tuple[str, list[Any]]:
        """Build this filter's WHERE clause and bound parameters for one backend.

        Args:
            backend_type: 'sqlite' or 'postgresql'.
            param_offset: Number of PostgreSQL placeholders that precede the clause.

        Returns:
            The clause SQL and its parameters in placeholder order.

        Raises:
            TypeError: If the simple equality path carries a list value.
        """
        builder = MetadataQueryBuilder(backend_type=backend_type, param_offset=param_offset)
        if self.operator is None:
            if isinstance(self.value, list):
                raise TypeError('the simple equality path takes a scalar value')
            builder.add_simple_filter(self.key, self.value)
        else:
            builder.add_advanced_filter(
                MetadataFilter(
                    key=self.key,
                    operator=self.operator,
                    value=self.value,
                    case_sensitive=self.case_sensitive,
                ),
            )
        return builder.build_where_clause()


def _advanced_paths(key_id: str, key: str) -> list[MetadataPath]:
    """Return every distinct advanced filter the model accepts for one key path."""
    values: list[tuple[str, FilterValue]] = [*SCALAR_VALUES.items(), *LIST_VALUES.items()]
    paths: list[MetadataPath] = []
    for operator in MetadataOperator:
        renderings: set[tuple[str, str]] = set()
        for value_id, value in values:
            for case_sensitive in (False, True):
                try:
                    spec = MetadataFilter(key=key, operator=operator, value=value, case_sensitive=case_sensitive)
                except ValidationError:
                    continue
                value_part = '' if spec.value is None else f'-{value_id}'
                case_part = '-cs' if case_sensitive else ''
                path = MetadataPath(
                    case_id=f'{key_id}-{operator.value}{value_part}{case_part}',
                    key=key,
                    operator=operator,
                    value=spec.value,
                    case_sensitive=case_sensitive,
                )
                rendering = (path.render('postgresql')[0], path.render('sqlite')[0])
                if rendering not in renderings:
                    renderings.add(rendering)
                    paths.append(path)
    return paths


def _simple_paths(key_id: str, key: str) -> list[MetadataPath]:
    """Return the simple ``metadata={}`` equality path for each scalar kind."""
    return [
        MetadataPath(
            case_id=f'{key_id}-simple-{value_id}',
            key=key,
            operator=None,
            value=value,
            case_sensitive=False,
        )
        for value_id, value in SCALAR_VALUES.items()
    ]


METADATA_PATHS: tuple[MetadataPath, ...] = tuple(
    path
    for key_id, key in KEY_PATHS.items()
    for path in (*_advanced_paths(key_id, key), *_simple_paths(key_id, key))
)
