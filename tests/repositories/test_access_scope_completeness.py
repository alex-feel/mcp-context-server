"""Structural completeness of access scoping, read from the source of ``app/``.

The unit of classification is the method: a function defined directly under a class or a
module of ``app/repositories``, with every closure nested inside it counted as part of it.
The classification and the source parser live in
:mod:`tests.repositories._access_scope_completeness_model`, which the backend-access rules
share. Six rules hold:

1. Detection: every method whose non-docstring string constants carry a statement against
   ``context_entries`` or one of its child tables is a key of :data:`REGISTRY`, and every
   key names a method of the package.
2. Classification: :data:`REGISTRY` assigns each method a :class:`Seam`.
3. Predicate reference: every scoped, write-authorized and owner-keyed method builds the
   access predicate, hands its scope to a predicate fragment helper, or is a dispatcher
   that forwards its scope to every executor it names.
4. Closure list: the closures of the two ranked searches are classified one by one.
5. Signatures and cases: a guarded method takes a required keyword-only ``scope``; every
   guarded entry names two-principal cases that exist in the case registry and run on
   both backends, with a filtered variant when the method accepts client filters; each
   dispatcher names the spy tests that prove it forwards the scope on every branch.
6. Call sites and sweeps: every attribute reference to a child reader or writer is a
   pinned site; no context-table SQL exists outside the repositories, migrations, CLI and
   schemas; only the FTS migration uses the system scope; no tool exposes ``scope`` or
   ``owner_id``; every tool resolves its caller exactly once; no string constant names
   the ``shared`` visibility.

Rule 3 proves that no method lost its predicate wholesale. Whether every per-backend
executor inside a method applies it is behavioral: every registered case runs on both
backends, so an executor that drops the predicate fails its case there.
"""

import ast
import inspect
import textwrap
from collections import Counter
from collections.abc import Iterator

import pytest

import app.tools
from app.access_scope import AccessScope
from app.access_scope import Scope
from tests.repositories._access_scope_completeness_model import CHILD
from tests.repositories._access_scope_completeness_model import CONTEXT_TABLE_SQL
from tests.repositories._access_scope_completeness_model import GUARDED
from tests.repositories._access_scope_completeness_model import REGISTRY
from tests.repositories._access_scope_completeness_model import ROOT
from tests.repositories._access_scope_completeness_model import ParsedModule
from tests.repositories._access_scope_completeness_model import Seam
from tests.repositories._access_scope_completeness_model import app_modules
from tests.repositories._access_scope_completeness_model import callee_name
from tests.repositories._access_scope_completeness_model import direct_closures
from tests.repositories._access_scope_completeness_model import hands_over_scope
from tests.repositories._access_scope_completeness_model import registry_keys
from tests.repositories._access_scope_completeness_model import repository_methods
from tests.repositories._access_scope_completeness_model import resolve_target
from tests.repositories._access_scope_completeness_model import runs_context_sql
from tests.repositories._access_scope_completeness_model import scope_parameter
from tests.repositories._access_scope_completeness_model import string_constants
from tests.repositories._access_scope_registry import CASES

# Top-level directories of app/ whose SQL is reviewed by classification or is system-level by
# construction, and the one module outside them that emits SQL: the predicate builder.
SQL_HOMES = frozenset({'repositories', 'migrations', 'cli', 'schemas'})
PREDICATE_MODULE = 'access_scope.py'

# The access-scope functions that turn a scope into a predicate.
PREDICATE_BUILDERS = frozenset({'build_access_predicate', 'build_readable_parent_predicate'})

# Helpers that build the access predicate for the method that hands them its scope.
PREDICATE_HELPERS = frozenset({'_candidate_sql', '_interleave_sql', '_dedup_update_sql', '_build_context_filter_clause'})

RESOLVERS = frozenset({'resolve_access_scope', 'resolve_effective_principal'})
SYSTEM_SCOPE_NAMES = frozenset({'SYSTEM_SCOPE', 'SystemScope'})

# The two-principal case ids of the repository seams.
KNOWN_CASE_IDS = frozenset({
    *(f'S{n}' for n in range(1, 8)), *(f'R{n}' for n in range(1, 5)), *(f'K{n}' for n in range(1, 5)),
    *(f'W{n}' for n in range(1, 6)), 'X1', 'X2', *(f'A{n}' for n in range(1, 7)),
})

# Tool-layer helpers that carry the caller's scope to the repositories.
SCOPE_CARRYING_HELPERS = (
    'app.tools.search.legs:semantic_search_raw',
    'app.tools.search.legs:fts_search_raw',
    'app.tools._transactions:execute_store_in_transaction',
    'app.tools._transactions:execute_update_in_transaction',
    'app.tools._transactions:reread_entry_version',
    'app.tools._delete_cleanup:delete_entries_with_cleanup',
    'app.tools._delete_cleanup:cleanup_embeddings_for_delete',
    'app.ids:resolve_prefix',
    'app.ids:resolve_or_normalize_id',
    'app.ids:resolve_or_normalize_ids',
)


# Part 1: detection.


def test_every_context_sql_method_is_classified() -> None:
    """A repository method that runs SQL against a context table fails until it is classified."""
    detected = {key for key, (node, module) in repository_methods().items() if runs_context_sql(node, module)}

    assert sorted(detected - REGISTRY.keys()) == []


def test_every_registry_key_names_a_repository_method() -> None:
    """A renamed or deleted method leaves no stale classification behind."""
    assert sorted(REGISTRY.keys() - repository_methods().keys()) == []


# Part 3: predicate reference.


@pytest.mark.parametrize('key', registry_keys(*GUARDED))
def test_guarded_method_applies_the_access_predicate(key: str) -> None:
    """A guarded method builds the predicate, hands its scope to a fragment helper, or forwards it."""
    entry = REGISTRY[key]
    node, _module = repository_methods()[key]
    calls = [call for call in ast.walk(node) if isinstance(call, ast.Call)]
    if entry.forwards_to:
        forwarded = [call for call in calls if callee_name(call) in entry.forwards_to]
        assert {callee_name(call) for call in forwarded} == set(entry.forwards_to)
        assert all(hands_over_scope(call) for call in forwarded)
        return
    builds = any(
        isinstance(child, ast.Name | ast.Attribute)
        and (child.id if isinstance(child, ast.Name) else child.attr) in PREDICATE_BUILDERS
        for child in ast.walk(node)
    )
    helper_calls = [call for call in calls if callee_name(call) in PREDICATE_HELPERS]

    assert builds or helper_calls, f'{key} neither builds the access predicate nor calls a fragment helper'
    assert all(hands_over_scope(call) for call in helper_calls)


# Part 4: closure list.


@pytest.mark.parametrize('key', sorted(key for key, entry in REGISTRY.items() if entry.closures))
def test_ranked_search_closures_are_classified(key: str) -> None:
    """Every closure of a ranked search is listed, and only the compute closures run no SQL."""
    entry = REGISTRY[key]
    node, module = repository_methods()[key]
    closures = direct_closures(node)

    assert sorted(closures) == sorted(entry.closures)
    for name, closure in closures.items():
        assert runs_context_sql(closure, module) is (entry.closures[name] is not Seam.COMPUTE), name


# Part 5: signatures, dispatchers and cases.


@pytest.mark.parametrize('key', registry_keys(*GUARDED))
def test_guarded_method_requires_a_scope(key: str) -> None:
    """A guarded method takes a keyword-only ``scope`` with no default, typed per its class."""
    parameter, annotation = scope_parameter(key)
    expected = AccessScope if REGISTRY[key].seam is Seam.OWNER_KEYED else Scope

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    assert annotation is expected


@pytest.mark.parametrize('target', SCOPE_CARRYING_HELPERS)
def test_scope_carrying_helper_requires_a_scope(target: str) -> None:
    """A tool-layer helper that reaches a repository takes a required keyword-only scope."""
    parameter, annotation = scope_parameter(target)

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    assert annotation in {Scope, AccessScope}


@pytest.mark.parametrize('key', sorted(key for key, entry in REGISTRY.items() if entry.forwards_to))
def test_dispatcher_names_its_spy_tests(key: str) -> None:
    """Each dispatcher names the tests that spy on its branches, and they exist."""
    spies = REGISTRY[key].spies
    assert spies
    for spy in spies:
        path, name = spy.split('::')
        tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
        assert name in {node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}, spy


@pytest.mark.parametrize('key', registry_keys(*GUARDED))
def test_guarded_method_names_registered_cases(key: str) -> None:
    """A guarded method names cases that run on both backends, filtered when it accepts filters."""
    names = REGISTRY[key].cases
    accepts_filters = 'metadata_filters' in inspect.signature(resolve_target(key)[1]).parameters

    assert names
    assert set(names) <= KNOWN_CASE_IDS
    for name in names:
        assert any(case.case_id == name and case.backend is None for case in CASES), f'{name} has no case on both backends'
    if accepts_filters:
        assert any(case.case_id in names and case.filtered and case.backend is None for case in CASES)


def test_every_registered_case_proves_a_classified_method() -> None:
    """Every case id in the case registry is claimed by a guarded method."""
    claimed = {name for entry in REGISTRY.values() for name in entry.cases}

    assert {case.case_id for case in CASES} == claimed


# Part 6: call sites and sweeps.


def _qualified_references(module: ParsedModule) -> Iterator[tuple[str, ast.Attribute]]:
    """Yield every attribute read of ``module`` with the qualified name of its enclosing function."""

    def visit(node: ast.AST, scope: tuple[str, ...]) -> Iterator[tuple[str, ast.Attribute]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                yield from visit(child, (*scope, child.name))
                continue
            if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
                yield '.'.join(scope), child
            yield from visit(child, scope)

    yield from visit(module.tree, ())


def _receiver_name(node: ast.expr) -> str | None:
    """The last name of the object an attribute is read from: ``repos.embeddings`` gives ``embeddings``."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


@pytest.mark.parametrize('key', registry_keys(*CHILD))
def test_child_reader_and_writer_sites_are_pinned(key: str) -> None:
    """A new reference to a child reader or writer fails until its id source is reviewed."""
    entry = REGISTRY[key]
    name = key.rsplit('.', 1)[1]
    sites = Counter(
        f'{module.path}:{function}'
        for module in app_modules()
        for function, attribute in _qualified_references(module)
        if attribute.attr == name and (entry.receiver is None or _receiver_name(attribute.value) == entry.receiver)
    )

    assert sites == Counter(entry.sites)


def test_no_context_sql_outside_the_repositories() -> None:
    """Context-table SQL lives only in the repositories, migrations, CLI, schemas and the predicate builder."""
    stray = [
        f'{module.path}:{line}'
        for module in app_modules()
        if module.path.split('/')[0] not in SQL_HOMES and module.path != PREDICATE_MODULE
        for line, value in string_constants(module.tree, module)
        if CONTEXT_TABLE_SQL.search(value)
    ]

    assert stray == []


def test_system_scope_is_used_only_by_the_fts_migration() -> None:
    """No request path can reach the system scope: its one consumer is the FTS migration."""
    users = {
        module.path
        for module in app_modules()
        if module.path != PREDICATE_MODULE
        and any(
            (isinstance(node, ast.Name) and node.id in SYSTEM_SCOPE_NAMES)
            or (isinstance(node, ast.Attribute) and node.attr in SYSTEM_SCOPE_NAMES)
            or (isinstance(node, ast.alias) and node.name in SYSTEM_SCOPE_NAMES)
            for node in ast.walk(module.tree)
        )
    }

    assert users == {'migrations/fts.py'}


@pytest.mark.parametrize('name', sorted(app.tools.TOOL_ANNOTATIONS))
def test_tool_signature_hides_scope_and_owner(name: str) -> None:
    """The server resolves the caller; no tool lets a client pass a scope or an owner."""
    parameters = inspect.signature(getattr(app.tools, name)).parameters

    assert 'scope' not in parameters
    assert 'owner_id' not in parameters


@pytest.mark.parametrize('name', sorted(app.tools.TOOL_ANNOTATIONS))
def test_tool_resolves_its_caller_once(name: str) -> None:
    """Every tool resolves the calling principal exactly once."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(getattr(app.tools, name)))))
    resolutions = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and callee_name(node) in RESOLVERS]

    assert len(resolutions) == 1


def test_no_string_constant_names_the_shared_visibility() -> None:
    """The visibility vocabulary is exactly ``private`` and ``public``."""
    hits = [
        f'{module.path}:{line}'
        for module in app_modules()
        for line, value in string_constants(module.tree, module)
        if value == 'shared'
    ]

    assert hits == []
