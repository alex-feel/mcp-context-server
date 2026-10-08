"""Structural completeness of the backend access seam, read from the source of ``app/``.

Every write entry point of a storage backend (``execute_write`` and ``begin_transaction``)
takes the identity it runs as through a required keyword-only ``access_scope``. Five rules
hold:

1. Every call of a write entry point under ``app/`` passes ``access_scope``.
2. A repository method that calls a write entry point takes a required keyword-only
   ``scope`` and hands that same ``scope`` to every such call.
3. Such a method is typed by its class: a guarded method keeps the scope type its
   predicate needs, and a child or system method takes the wider ``BackendAccess``.
4. No repository method carries a ``scope`` it neither hands to a call nor reads, and no
   guarded ``scope`` admits the maintenance identity.
5. The maintenance identity never appears under ``app/tools`` or ``app/repositories``:
   only migrations, startup, the CLI and the compression provenance run as it.
"""

import ast
import inspect
import types
import typing

import pytest

from app.access_scope import AccessScope
from app.access_scope import SystemScope
from app.backends.base import BackendAccess
from app.backends.base import MaintenanceAccess
from tests.repositories._access_scope_completeness_model import GUARDED
from tests.repositories._access_scope_completeness_model import FunctionNode
from tests.repositories._access_scope_completeness_model import app_modules
from tests.repositories._access_scope_completeness_model import registry_keys
from tests.repositories._access_scope_completeness_model import repository_methods
from tests.repositories._access_scope_completeness_model import scope_parameter

# The backend methods that bind the caller's identity before running anything.
BACKEND_ENTRY_POINTS = frozenset({'execute_write', 'begin_transaction'})

# The names of the maintenance identity and the directories it must never reach.
MAINTENANCE_NAMES = frozenset({'MAINTENANCE', 'MaintenanceAccess'})
REQUEST_PATH_DIRECTORIES = frozenset({'tools', 'repositories'})


def _entry_point_calls(node: ast.AST) -> list[ast.Call]:
    """Return every call of a backend entry point in ``node``'s subtree."""
    return [
        child for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        and child.func.attr in BACKEND_ENTRY_POINTS
    ]


def _access_scope_argument(call: ast.Call) -> ast.expr | None:
    """Return the value a call passes as ``access_scope``, or None when it passes none."""
    return next((keyword.value for keyword in call.keywords if keyword.arg == 'access_scope'), None)


def _backend_callers() -> list[str]:
    """The repository methods whose subtree calls a backend entry point, sorted."""
    return sorted(key for key, (node, _module) in repository_methods().items() if _entry_point_calls(node))


def _scope_carriers() -> list[str]:
    """The repository methods that declare a ``scope`` parameter, sorted."""
    return sorted(
        key for key, (node, _module) in repository_methods().items()
        if any(argument.arg == 'scope' for argument in (*node.args.args, *node.args.kwonlyargs))
    )


def _consumes_scope(node: FunctionNode) -> bool:
    """Whether ``node``'s subtree hands the bare name ``scope`` to a call or reads an attribute of it.

    Binding it to a throwaway name does not count: that discards the identity.

    Returns:
        True when the scope reaches a call or an attribute read.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            arguments = (*child.args, *(keyword.value for keyword in child.keywords))
            if any(isinstance(argument, ast.Name) and argument.id == 'scope' for argument in arguments):
                return True
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id == 'scope':
            return True
    return False


def _union_members(annotation: object) -> set[object]:
    """Expand type aliases and unions into the set of types an annotation admits."""
    if isinstance(annotation, typing.TypeAliasType):
        return _union_members(annotation.__value__)
    if typing.get_origin(annotation) in {types.UnionType, typing.Union}:
        return set().union(*(_union_members(member) for member in typing.get_args(annotation)))
    return {annotation}


def test_backend_access_admits_every_identity_and_nothing_else() -> None:
    """``BackendAccess`` is exactly the request scopes plus the maintenance identity."""
    assert _union_members(BackendAccess) == {AccessScope, SystemScope, MaintenanceAccess}


def test_every_backend_entry_point_call_passes_the_access_scope() -> None:
    """A write that reaches the backend without naming its identity fails until it does."""
    missing = [
        f'{module.path}:{call.lineno}'
        for module in app_modules()
        for call in _entry_point_calls(module.tree)
        if _access_scope_argument(call) is None
    ]

    assert missing == []


@pytest.mark.parametrize('key', _backend_callers())
def test_backend_calling_method_forwards_its_scope(key: str) -> None:
    """A repository method that reaches the backend requires a scope and hands it to the backend."""
    parameter, _annotation = scope_parameter(key)
    node, _module = repository_methods()[key]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    for call in _entry_point_calls(node):
        argument = _access_scope_argument(call)
        assert isinstance(argument, ast.Name), f'{key}:{call.lineno}'
        assert argument.id == 'scope', f'{key}:{call.lineno}'


@pytest.mark.parametrize('key', [key for key in _backend_callers() if key not in registry_keys(*GUARDED)])
def test_child_and_system_backend_callers_take_the_backend_access(key: str) -> None:
    """A child or system method renders no predicate, so it accepts every identity the backend binds."""
    _parameter, annotation = scope_parameter(key)

    assert annotation is BackendAccess


@pytest.mark.parametrize('key', _scope_carriers())
def test_repository_scope_is_consumed(key: str) -> None:
    """A repository method never carries a scope it neither hands on nor reads."""
    node, _module = repository_methods()[key]

    assert _consumes_scope(node), key


@pytest.mark.parametrize('key', registry_keys(*GUARDED))
def test_guarded_scope_never_admits_the_maintenance_identity(key: str) -> None:
    """A method that renders the access predicate cannot be handed the maintenance identity."""
    _parameter, annotation = scope_parameter(key)

    assert MaintenanceAccess not in _union_members(annotation)


def test_maintenance_identity_stays_off_the_request_paths() -> None:
    """The tools and repositories run as the caller; only non-request paths name the maintenance identity."""
    users = sorted(
        f'{module.path}:{getattr(node, "lineno", 0)}'
        for module in app_modules()
        if module.path.split('/')[0] in REQUEST_PATH_DIRECTORIES
        for node in ast.walk(module.tree)
        if (isinstance(node, ast.Name) and node.id in MAINTENANCE_NAMES)
        or (isinstance(node, ast.Attribute) and node.attr in MAINTENANCE_NAMES)
        or (isinstance(node, ast.alias) and node.name in MAINTENANCE_NAMES)
    )

    assert users == []
