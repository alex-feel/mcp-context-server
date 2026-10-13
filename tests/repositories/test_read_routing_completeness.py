"""Structural completeness of read routing, read from the source of ``app/``.

A plain read names every routable table it selects from through the repository seam
``BaseRepository._readable``, which asks the backend for the relation to read. Three
rules hold:

1. No guarded method and no id-keyed child reader names a routable table after ``FROM``
   or ``JOIN`` in a string constant: each such item comes from ``_readable``. A delete
   names its target table, and the write gate ``entry_exists`` keeps the base name
   because it locks the row it reads.
2. Every ``_readable`` call passes string literals for a routable table and its alias,
   so the relation a statement reads is visible in its source.
3. Child writers and system methods never call ``_readable``: they write, or run outside
   any request, and name their tables directly.
"""

import ast
import functools
import re
from collections.abc import Iterator

import pytest

from app.backends.base import ROUTABLE_TABLES
from tests.repositories._access_scope_completeness_model import GUARDED
from tests.repositories._access_scope_completeness_model import READ_SEAM
from tests.repositories._access_scope_completeness_model import REGISTRY
from tests.repositories._access_scope_completeness_model import FunctionNode
from tests.repositories._access_scope_completeness_model import ParsedModule
from tests.repositories._access_scope_completeness_model import Seam
from tests.repositories._access_scope_completeness_model import app_modules
from tests.repositories._access_scope_completeness_model import callee_name
from tests.repositories._access_scope_completeness_model import direct_closures
from tests.repositories._access_scope_completeness_model import registry_keys
from tests.repositories._access_scope_completeness_model import repository_methods

# A FROM or JOIN item over a routable table; the optional DELETE marks a delete target.
ROUTABLE_ITEM = re.compile(
    rf'\b(DELETE\s+)?(?:FROM|JOIN)\s+({"|".join(sorted(ROUTABLE_TABLES))})\b',
    re.IGNORECASE,
)

# The write gate reads under a row lock on PostgreSQL, so it keeps the base name.
LOCKING_READ = 'app.repositories.context_repository.reads:ContextReadMixin.entry_exists'

ROUTED = frozenset({*GUARDED, Seam.CHILD_BY_TRUSTED_ID})
UNROUTED = frozenset({Seam.CHILD_WRITER, Seam.SYSTEM})

ALIAS = re.compile(r'[a-z_][a-z0-9_]*')


def _routed_constants(key: str) -> Iterator[tuple[int, str]]:
    """Yield the line and value of every string constant of a method whose plain reads must route.

    The whole method counts, except the classified closures of a ranked search whose class
    does not route.

    Yields:
        The line and value of each non-docstring string constant.
    """
    node, module = repository_methods()[key]
    closures = direct_closures(node)
    excluded = {
        id(child)
        for name, seam in REGISTRY[key].closures.items() if seam not in ROUTED
        for child in ast.walk(closures[name])
    }
    for line, value, constant in _constants(node, module):
        if id(constant) not in excluded:
            yield line, value


def _constants(node: ast.AST, module: ParsedModule) -> Iterator[tuple[int, str, ast.Constant]]:
    """Yield the line, value and node of every non-docstring string constant of ``node``'s subtree."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str) and id(child) not in module.docstrings:
            yield child.lineno, child.value, child


def _seam_calls(node: ast.AST) -> list[ast.Call]:
    """Return every call of the read-routing seam in ``node``'s subtree."""
    return [child for child in ast.walk(node) if isinstance(child, ast.Call) and callee_name(child) == READ_SEAM]


@pytest.mark.parametrize('key', [key for key in registry_keys(*ROUTED) if key != LOCKING_READ])
def test_routed_method_names_no_routable_table(key: str) -> None:
    """A plain read of a guarded method or child reader names its relations through the seam."""
    hits = [
        f'{line}: {match.group(0)!r}'
        for line, value in _routed_constants(key)
        for match in ROUTABLE_ITEM.finditer(value)
        if match.group(1) is None
    ]

    assert hits == []


@functools.cache
def _seam_calls_by_location() -> dict[str, ast.Call]:
    """Map ``<path under app>:<line>:<column>`` to every call of the seam under ``app/``."""
    return {
        f'{module.path}:{call.lineno}:{call.col_offset}': call
        for module in app_modules()
        for call in _seam_calls(module.tree)
    }


def test_the_seam_is_called() -> None:
    """The literal-argument rule inspects real calls, so the seam has callers."""
    assert _seam_calls_by_location()


@pytest.mark.parametrize('location', sorted(_seam_calls_by_location()))
def test_seam_call_passes_literal_table_and_alias(location: str) -> None:
    """Each call names a routable table and an alias as string literals and nothing else."""
    call = _seam_calls_by_location()[location]
    assert len(call.args) == 1
    assert [keyword.arg for keyword in call.keywords] == ['alias']
    table = call.args[0]
    alias = call.keywords[0].value
    assert isinstance(table, ast.Constant)
    assert table.value in ROUTABLE_TABLES
    assert isinstance(alias, ast.Constant)
    assert isinstance(alias.value, str)
    assert ALIAS.fullmatch(alias.value) is not None


@pytest.mark.parametrize('key', registry_keys(*UNROUTED))
def test_child_writers_and_system_methods_never_route(key: str) -> None:
    """A writer or a system method names its tables directly, never through the seam."""
    node: FunctionNode = repository_methods()[key][0]

    assert _seam_calls(node) == []
