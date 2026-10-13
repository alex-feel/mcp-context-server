"""Structural completeness of read routing, read from the source of ``app/``.

A plain read names every routable table it selects from through the repository seam
``BaseRepository._readable``, which asks the backend for the relation to read. Four
rules hold:

1. No guarded method and no id-keyed child reader names a routable table after ``FROM``
   or ``JOIN`` in a string constant: each such item comes from ``_readable``. A delete
   names its target table, and the write gate ``entry_exists`` keeps the base name
   because it locks the row it reads.
2. Every ``_readable`` call passes string literals for a routable table and its alias,
   so the relation a statement reads is visible in its source.
3. Child writers and system methods never call ``_readable``: they write, or run outside
   any request, and name their tables directly.
4. On a request backend under row-level security each call over a mapped table, the
   context table or tags, names that table's read view under the call's alias, and on
   every other backend each call names the table itself.
"""

import ast
import functools
import re
from collections.abc import Iterator

import pytest

from app.backends.base import ROUTABLE_TABLES
from app.backends.base import base_relation
from app.backends.postgresql_backend.session_profiles import RLS_READ_VIEWS
from app.backends.postgresql_backend.session_profiles import RlsRequestProfile
from app.backends.postgresql_backend.session_profiles import UnboundProfile
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


def _literals(call: ast.Call) -> tuple[str, str]:
    """Return the table and the alias a seam call passes as string literals."""
    table = call.args[0]
    alias = call.keywords[0].value
    assert isinstance(table, ast.Constant)
    assert isinstance(table.value, str)
    assert isinstance(alias, ast.Constant)
    assert isinstance(alias.value, str)
    return table.value, alias.value


@pytest.mark.parametrize('location', sorted(_seam_calls_by_location()))
def test_seam_call_reads_a_mapped_table_through_its_view_under_row_level_security_alone(location: str) -> None:
    """A request backend under row-level security reads a mapped table through its view; any other names the table."""
    table, alias = _literals(_seam_calls_by_location()[location])
    base = base_relation(table, alias=alias)
    expected = f'{RLS_READ_VIEWS[table]} AS {alias}' if table in RLS_READ_VIEWS else base

    assert RlsRequestProfile(read_views=RLS_READ_VIEWS).readable_relation(table, alias=alias) == expected
    assert UnboundProfile().readable_relation(table, alias=alias) == base


def test_every_plain_read_of_tags_names_tags_readable_under_row_level_security() -> None:
    """Every plain read of tags, the tag filters, the tags of an entry and the tag statistics, goes through tags_readable."""
    profile = RlsRequestProfile(read_views=RLS_READ_VIEWS)
    aliases = [alias for table, alias in map(_literals, _seam_calls_by_location().values()) if table == 'tags']

    assert aliases
    assert [profile.readable_relation('tags', alias=alias) for alias in aliases] == [
        f'tags_readable AS {alias}' for alias in aliases
    ]
