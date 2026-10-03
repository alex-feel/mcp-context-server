"""Structural completeness of access scoping, read from the source of ``app/``.

The unit of classification is the method: a function defined directly under a class or a
module of ``app/repositories``, with every closure nested inside it counted as part of it.
Six rules hold:

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
import functools
import importlib
import inspect
import re
import textwrap
import typing
from collections import Counter
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from pathlib import Path
from types import ModuleType

import pytest

import app.tools
from app.access_scope import AccessScope
from app.access_scope import Scope
from tests.repositories._access_scope_registry import CASES

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / 'app'

CONTEXT_TABLE_SQL = re.compile(
    r'\b(FROM|JOIN|INTO|UPDATE|TABLE)\s+(context_entries|context_entries_fts|context_entry_grants|tags'
    r'|image_attachments|context_index_nodes|embedding_chunks|embedding_metadata|vec_context_embeddings'
    r'|vec_context_embeddings_compressed)\b',
    re.IGNORECASE,
)

# Top-level directories of app/ whose SQL is reviewed by classification or is system-level by
# construction, and the one module outside them that emits SQL: the predicate builder.
SQL_HOMES = frozenset({'repositories', 'migrations', 'cli', 'schemas'})
PREDICATE_MODULE = 'access_scope.py'

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
    'app.ids:resolve_prefix',
    'app.ids:resolve_or_normalize_id',
    'app.ids:resolve_or_normalize_ids',
)


class Seam(StrEnum):
    """How a method or a ranked-search closure is guarded."""

    SCOPED = 'scoped'  # the read predicate sits in the statement
    WRITE_AUTHORIZED = 'write_authorized'  # the write or owner predicate sits in the statement
    OWNER_KEYED = 'owner_keyed'  # the dedup candidate, interleave and update fragments
    CHILD_BY_TRUSTED_ID = 'child_by_trusted_id'  # id-keyed child read; its call sites are pinned
    CHILD_WRITER = 'child_writer'  # child write behind the write gate or the owned dedup; sites pinned
    SYSTEM = 'system'  # schema maintenance or a deployment-level figure
    COMPUTE = 'compute'  # a ranked-search closure that runs no SQL


GUARDED = frozenset({Seam.SCOPED, Seam.WRITE_AUTHORIZED, Seam.OWNER_KEYED})
CHILD = frozenset({Seam.CHILD_BY_TRUSTED_ID, Seam.CHILD_WRITER})


@dataclass(frozen=True)
class Entry:
    """The classification of one repository method.

    Attributes:
        seam: How the method is guarded.
        cases: The two-principal case ids that prove a guarded method on both backends.
        closures: For the two ranked searches, the class of every closure defined directly
            inside the method.
        forwards_to: For a dispatcher, the executors it hands its scope to.
        spies: For a dispatcher, the tests (``path::name``) that spy on every branch.
        sites: For a child reader or writer, every function of ``app/`` that references it,
            as ``<path under app>:<qualified function name>``, once per reference.
        receiver: Counts only references through this attribute or name, for a method whose
            name other objects share (``Path.exists``).
    """

    seam: Seam
    cases: tuple[str, ...] = ()
    closures: Mapping[str, Seam] = field(default_factory=dict[str, Seam])
    forwards_to: tuple[str, ...] = ()
    spies: tuple[str, ...] = ()
    sites: tuple[str, ...] = ()
    receiver: str | None = None


_REPO = 'app.repositories.'
_READS = _REPO + 'context_repository.reads:ContextReadMixin.'
_SEARCH = _REPO + 'context_repository.search:ContextSearchMixin.'
_UPDATES = _REPO + 'context_repository.updates:ContextUpdateMixin.'
_DELETES = _REPO + 'context_repository.deletes:ContextDeleteMixin.'
_DEDUP = _REPO + 'context_repository.dedup:'
_EMBEDDINGS = _REPO + 'embedding_repository:EmbeddingRepository.'
_FP32 = _REPO + 'embedding_repository.fp32_search:Fp32SearchMixin.'
_COMPRESSED = _REPO + 'embedding_repository.compressed_search:CompressedSearchMixin.'
_INVENTORY = _REPO + 'embedding_repository.inventory:EmbeddingInventoryMixin.'
_CHUNKS = _REPO + 'embedding_repository.chunk_writes:ChunkWriteMixin.'
_FTS_SEARCH = _REPO + 'fts_repository.search:FtsSearchMixin.'
_FTS_MAINTENANCE = _REPO + 'fts_repository.maintenance:FtsMaintenanceMixin.'
_STATISTICS = _REPO + 'statistics_repository:StatisticsRepository.'
_NODES = _REPO + 'index_node_repository:IndexNodeRepository.'
_TAGS = _REPO + 'tag_repository:TagRepository.'
_IMAGES = _REPO + 'image_repository:ImageRepository.'
_GRANTS = _REPO + 'grant_repository:GrantRepository.'

_STORE_TX = 'tools/_transactions.py:execute_store_in_transaction'
_UPDATE_TX = 'tools/_transactions.py:execute_update_in_transaction'
_CHUNK_WRITES = 'repositories/embedding_repository/chunk_writes.py:ChunkWriteMixin.'
_REEMBED = 'cli/migrate_reembed.py:_reembed_async'
_RESULT_HYDRATORS = (
    'tools/context/retrieve.py:get_context_by_ids',
    'tools/search/browse.py:search_context',
    'tools/search/fts.py:fts_search_context',
    'tools/search/semantic.py:semantic_search_context',
    'tools/search/hybrid.py:hybrid_search_context',
)
_EMBEDDING_DISPATCH_SPIES = 'tests/repositories/embedding_repository/test_search_dispatch.py::'
_FTS_DISPATCH_SPIES = 'tests/repositories/fts_repository/test_search_dispatch.py::'


def _guarded(
    seam: Seam,
    *cases: str,
    closures: Mapping[str, Seam] | None = None,
    forwards_to: tuple[str, ...] = (),
    spies: tuple[str, ...] = (),
) -> Entry:
    """Build the entry of a guarded method proven by ``cases``."""
    return Entry(seam, cases=cases, closures=closures or {}, forwards_to=forwards_to, spies=spies)


def _child(seam: Seam, *sites: str, receiver: str | None = None) -> Entry:
    """Build the entry of a child reader or writer referenced from ``sites``."""
    return Entry(seam, sites=sites, receiver=receiver)


REGISTRY: dict[str, Entry] = {
    # Scoped: the read predicate in the statement.
    _READS + 'get_by_ids': _guarded(Seam.SCOPED, 'S3'),
    _READS + 'find_ids_by_prefix': _guarded(Seam.SCOPED, 'S4'),
    _READS + 'check_entry_exists': _guarded(Seam.SCOPED, 'S5'),
    _READS + 'probe_ids': _guarded(Seam.SCOPED, 'S7'),
    _SEARCH + 'search_contexts': _guarded(Seam.SCOPED, 'S1'),
    _SEARCH + 'grep_scan_text_contents': _guarded(Seam.SCOPED, 'S2'),
    _SEARCH + '_build_context_filter_clause': _guarded(Seam.SCOPED, 'S1', 'S2'),
    _EMBEDDINGS + 'search': _guarded(
        Seam.SCOPED, 'R1', 'R2', forwards_to=('search_fp32', 'search_compressed'),
        spies=(
            _EMBEDDING_DISPATCH_SPIES + 'test_compressed_branch_receives_the_scope',
            _EMBEDDING_DISPATCH_SPIES + 'test_fp32_branch_receives_the_scope',
        ),
    ),
    _FP32 + 'search_fp32': _guarded(
        Seam.SCOPED, 'R1', closures={'_search_sqlite': Seam.SCOPED, '_search_postgresql': Seam.SCOPED},
    ),
    _COMPRESSED + 'search_compressed': _guarded(
        Seam.SCOPED, 'R2', 'R3', closures={
            '_candidates_sqlite': Seam.SCOPED,
            '_candidates_pg': Seam.SCOPED,
            '_hydrate_sqlite': Seam.SCOPED,
            '_hydrate_pg': Seam.SCOPED,
            '_read_compressed_sqlite': Seam.CHILD_BY_TRUSTED_ID,
            '_read_compressed_pg': Seam.CHILD_BY_TRUSTED_ID,
            '_decode_and_concat': Seam.COMPUTE,
            '_rank_from_gemm': Seam.COMPUTE,
        },
    ),
    _FTS_SEARCH + 'search': _guarded(
        Seam.SCOPED, 'R4', forwards_to=('_search_sqlite', '_search_postgresql'),
        spies=(_FTS_DISPATCH_SPIES + 'test_executor_receives_the_scope',),
    ),
    _FTS_SEARCH + '_search_sqlite': _guarded(Seam.SCOPED, 'R4'),
    _FTS_SEARCH + '_search_postgresql': _guarded(Seam.SCOPED, 'R4'),
    _FTS_MAINTENANCE + 'get_statistics': _guarded(Seam.SCOPED, 'A5'),
    _INVENTORY + 'get_statistics': _guarded(Seam.SCOPED, 'A4'),
    _STATISTICS + 'get_thread_list': _guarded(Seam.SCOPED, 'A1'),
    _STATISTICS + 'get_database_statistics': _guarded(Seam.SCOPED, 'A2'),
    _STATISTICS + 'get_summary_statistics': _guarded(Seam.SCOPED, 'A3'),
    _NODES + 'count_all_nodes': _guarded(Seam.SCOPED, 'A6'),
    # Write-authorized: the write or owner predicate in the statement.
    _READS + 'entry_exists': _guarded(Seam.WRITE_AUTHORIZED, 'S6'),
    _READS + 'get_content_type': _guarded(Seam.WRITE_AUTHORIZED, 'W5'),
    _UPDATES + 'update_context_entry': _guarded(Seam.WRITE_AUTHORIZED, 'W1', 'W2'),
    _UPDATES + 'patch_metadata': _guarded(Seam.WRITE_AUTHORIZED, 'W3'),
    _UPDATES + 'touch_updated_at': _guarded(Seam.WRITE_AUTHORIZED, 'W4'),
    _UPDATES + 'update_content_type': _guarded(Seam.WRITE_AUTHORIZED, 'W4'),
    _DELETES + 'delete_by_ids': _guarded(Seam.WRITE_AUTHORIZED, 'X1'),
    _DELETES + 'get_ids_matching_batch_criteria': _guarded(Seam.WRITE_AUTHORIZED, 'X2'),
    # Owner-keyed: the dedup candidate, the read-scoped interleave check and the owned update.
    _DEDUP + 'ContextDedupMixin.store_with_deduplication': _guarded(Seam.OWNER_KEYED, 'K1', 'K2', 'K4'),
    _DEDUP + 'ContextDedupMixin.check_latest_is_duplicate': _guarded(Seam.OWNER_KEYED, 'K3'),
    _DEDUP + '_candidate_sql': _guarded(Seam.OWNER_KEYED, 'K1', 'K3', 'K4'),
    _DEDUP + '_interleave_sql': _guarded(Seam.OWNER_KEYED, 'K2', 'K3'),
    _DEDUP + '_dedup_update_sql': _guarded(Seam.OWNER_KEYED, 'K1'),
    # Child readers keyed by ids a scoped read produced in the same request.
    _TAGS + 'get_tags_for_context': _child(Seam.CHILD_BY_TRUSTED_ID, *_RESULT_HYDRATORS),
    _IMAGES + 'get_images_for_context': _child(Seam.CHILD_BY_TRUSTED_ID, *_RESULT_HYDRATORS),
    _IMAGES + 'count_images_for_context': _child(Seam.CHILD_BY_TRUSTED_ID, _UPDATE_TX),
    _NODES + 'get_nodes_for_context': _child(Seam.CHILD_BY_TRUSTED_ID, 'tools/navigation.py:navigate_context'),
    _INVENTORY + 'exists': _child(
        Seam.CHILD_BY_TRUSTED_ID,
        _STORE_TX, 'tools/batch/store.py:store_context_batch', 'tools/context/store.py:store_context',
        receiver='embeddings',
    ),
    # Child writers behind the write gate or the owned dedup.
    _TAGS + 'store_tags': _child(Seam.CHILD_WRITER, _STORE_TX),
    _TAGS + 'replace_tags_for_context': _child(Seam.CHILD_WRITER, _STORE_TX, _UPDATE_TX),
    _TAGS + '_insert_tag_sql': _child(
        Seam.CHILD_WRITER,
        'repositories/tag_repository.py:TagRepository.store_tags._store_tags_sqlite',
        'repositories/tag_repository.py:TagRepository.store_tags._store_tags_postgresql',
        'repositories/tag_repository.py:TagRepository.replace_tags_for_context._replace_tags_sqlite',
        'repositories/tag_repository.py:TagRepository.replace_tags_for_context._replace_tags_postgresql',
    ),
    _IMAGES + 'store_image': _child(Seam.CHILD_WRITER),
    _IMAGES + 'store_images': _child(Seam.CHILD_WRITER, _STORE_TX),
    _IMAGES + 'replace_images_for_context': _child(Seam.CHILD_WRITER, _STORE_TX, _UPDATE_TX, _UPDATE_TX),
    _NODES + 'replace_nodes_for_context': _child(Seam.CHILD_WRITER, _STORE_TX, _UPDATE_TX),
    _GRANTS + 'store_group_read_grants': _child(Seam.CHILD_WRITER, _STORE_TX),
    _GRANTS + '_insert_grant_sql': _child(
        Seam.CHILD_WRITER,
        'repositories/grant_repository.py:GrantRepository.store_group_read_grants._store_sqlite',
        'repositories/grant_repository.py:GrantRepository.store_group_read_grants._store_postgresql',
    ),
    # The embedding CLI commands run outside any request and write by the ids they read.
    _CHUNKS + 'store_chunked': _child(
        Seam.CHILD_WRITER, _STORE_TX, _UPDATE_TX, 'cli/migrate_embeddings.py:_embed_missing_async', _REEMBED,
    ),
    _CHUNKS + '_store_chunked_compressed': _child(Seam.CHILD_WRITER, _CHUNK_WRITES + 'store_chunked'),
    _CHUNKS + 'delete_all_chunks': _child(
        Seam.CHILD_WRITER,
        _CHUNK_WRITES + 'store_chunked', _CHUNK_WRITES + 'delete_all_chunks_bulk', _UPDATE_TX, _UPDATE_TX, _REEMBED,
    ),
    _CHUNKS + '_delete_all_chunks_compressed': _child(
        Seam.CHILD_WRITER, _CHUNK_WRITES + '_store_chunked_compressed', _CHUNK_WRITES + 'delete_all_chunks',
    ),
    # The delete path hands over only the ids its in-transaction owner probe cleared.
    _CHUNKS + 'delete_all_chunks_bulk': _child(Seam.CHILD_WRITER, 'tools/_delete_cleanup.py:cleanup_embeddings_for_delete'),
    _CHUNKS + '_delete_all_chunks_compressed_bulk': _child(Seam.CHILD_WRITER, _CHUNK_WRITES + 'delete_all_chunks_bulk'),
    # System: schema maintenance and deployment-level figures.
    _FTS_MAINTENANCE + 'is_available': Entry(Seam.SYSTEM),
    _FTS_MAINTENANCE + 'get_current_tokenizer': Entry(Seam.SYSTEM),
    _FTS_MAINTENANCE + 'get_desired_tokenizer': Entry(Seam.SYSTEM),
    _FTS_MAINTENANCE + 'get_current_language': Entry(Seam.SYSTEM),
    _FTS_MAINTENANCE + 'migrate_tokenizer': Entry(Seam.SYSTEM),
    _FTS_MAINTENANCE + 'migrate_language': Entry(Seam.SYSTEM),
    _INVENTORY + 'embedding_tables_exist': Entry(Seam.SYSTEM),
    _INVENTORY + 'get_embeddings_size': Entry(Seam.SYSTEM),
    _INVENTORY + '_get_embeddings_size_sqlite': Entry(Seam.SYSTEM),
}


@dataclass(frozen=True)
class ParsedModule:
    """One parsed module of ``app/`` and the ids of its docstring constants."""

    path: str  # relative to app/, POSIX separators
    dotted: str
    tree: ast.Module
    docstrings: frozenset[int]


type FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _docstring_ids(tree: ast.Module) -> frozenset[int]:
    """Return the node ids of the module, class and function docstrings of ``tree``."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return frozenset(ids)


@functools.cache
def _app_modules() -> tuple[ParsedModule, ...]:
    """Parse every module of ``app/`` once."""
    modules = []
    for file in sorted(APP.rglob('*.py')):
        parts = file.relative_to(ROOT).with_suffix('').parts
        dotted = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
        tree = ast.parse(file.read_text(encoding='utf-8'))
        modules.append(ParsedModule(file.relative_to(APP).as_posix(), dotted, tree, _docstring_ids(tree)))
    return tuple(modules)


def _strings(node: ast.AST, module: ParsedModule) -> Iterator[tuple[int, str]]:
    """Yield the line and value of every non-docstring string constant of ``node``'s subtree, f-string parts included."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str) and id(child) not in module.docstrings:
            yield child.lineno, child.value


def _runs_context_sql(node: ast.AST, module: ParsedModule) -> bool:
    """Whether ``node``'s subtree carries a statement against a context table."""
    return any(CONTEXT_TABLE_SQL.search(value) for _line, value in _strings(node, module))


@functools.cache
def _repository_methods() -> dict[str, tuple[FunctionNode, ParsedModule]]:
    """Map ``module:Class.method`` (or ``module:function``) to every method of ``app/repositories``."""
    methods: dict[str, tuple[FunctionNode, ParsedModule]] = {}
    for module in _app_modules():
        if not module.path.startswith('repositories/'):
            continue
        for node in module.tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                methods[f'{module.dotted}:{node.name}'] = (node, module)
            elif isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                        methods[f'{module.dotted}:{node.name}.{member.name}'] = (member, module)
    return methods


def _callee(call: ast.Call) -> str | None:
    """The bare name a call invokes: ``f(...)`` and ``x.f(...)`` both give ``f``."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _hands_over_scope(call: ast.Call) -> bool:
    """Whether a call passes the caller's own ``scope``, positionally or by keyword."""
    passed = [*call.args, *(keyword.value for keyword in call.keywords if keyword.arg == 'scope')]
    return any(isinstance(argument, ast.Name) and argument.id == 'scope' for argument in passed)


def _direct_closures(node: ast.AST) -> dict[str, FunctionNode]:
    """Return the functions defined inside ``node`` but outside any function nested in it."""
    closures: dict[str, FunctionNode] = {}
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            closures[child.name] = child
        elif not isinstance(child, ast.Lambda | ast.ClassDef):
            closures.update(_direct_closures(child))
    return closures


def _resolve(target: str) -> tuple[ModuleType, Callable[..., object]]:
    """Import ``module:qualname`` and return the module and the function it names."""
    module_name, qualname = target.split(':')
    module = importlib.import_module(module_name)
    found: object = module
    for part in qualname.split('.'):
        found = getattr(found, part)
    assert callable(found), target
    return module, found


def _type_checking_names(module: ModuleType) -> dict[str, object]:
    """Import the names ``module`` binds only under ``if TYPE_CHECKING:``, for resolving its hints."""
    names: dict[str, object] = {}
    for node in ast.parse(inspect.getsource(module)).body:
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'TYPE_CHECKING'):
            continue
        for statement in node.body:
            if isinstance(statement, ast.ImportFrom) and statement.module:
                source = importlib.import_module(statement.module)
                names.update({alias.asname or alias.name: getattr(source, alias.name) for alias in statement.names})
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    bound = alias.asname or alias.name.split('.')[0]
                    names[bound] = importlib.import_module(alias.name if alias.asname else bound)
    return names


def _scope_parameter(target: str) -> tuple[inspect.Parameter, object]:
    """Return the ``scope`` parameter of ``target`` and its resolved annotation."""
    module, function = _resolve(target)
    parameters = inspect.signature(function).parameters
    assert 'scope' in parameters, f'{target} takes no scope'
    hints = typing.get_type_hints(function, localns=_type_checking_names(module))
    return parameters['scope'], hints['scope']


def _keys(*seams: Seam) -> list[str]:
    """The registry keys of the given classes, sorted."""
    return sorted(key for key, entry in REGISTRY.items() if entry.seam in seams)


# Part 1: detection.


def test_every_context_sql_method_is_classified() -> None:
    """A repository method that runs SQL against a context table fails until it is classified."""
    detected = {key for key, (node, module) in _repository_methods().items() if _runs_context_sql(node, module)}

    assert sorted(detected - REGISTRY.keys()) == []


def test_every_registry_key_names_a_repository_method() -> None:
    """A renamed or deleted method leaves no stale classification behind."""
    assert sorted(REGISTRY.keys() - _repository_methods().keys()) == []


# Part 3: predicate reference.


@pytest.mark.parametrize('key', _keys(*GUARDED))
def test_guarded_method_applies_the_access_predicate(key: str) -> None:
    """A guarded method builds the predicate, hands its scope to a fragment helper, or forwards it."""
    entry = REGISTRY[key]
    node, _module = _repository_methods()[key]
    calls = [call for call in ast.walk(node) if isinstance(call, ast.Call)]
    if entry.forwards_to:
        forwarded = [call for call in calls if _callee(call) in entry.forwards_to]
        assert {_callee(call) for call in forwarded} == set(entry.forwards_to)
        assert all(_hands_over_scope(call) for call in forwarded)
        return
    builds = any(
        isinstance(child, ast.Name | ast.Attribute)
        and (child.id if isinstance(child, ast.Name) else child.attr) == 'build_access_predicate'
        for child in ast.walk(node)
    )
    helper_calls = [call for call in calls if _callee(call) in PREDICATE_HELPERS]

    assert builds or helper_calls, f'{key} neither builds the access predicate nor calls a fragment helper'
    assert all(_hands_over_scope(call) for call in helper_calls)


# Part 4: closure list.


@pytest.mark.parametrize('key', sorted(key for key, entry in REGISTRY.items() if entry.closures))
def test_ranked_search_closures_are_classified(key: str) -> None:
    """Every closure of a ranked search is listed, and only the compute closures run no SQL."""
    entry = REGISTRY[key]
    node, module = _repository_methods()[key]
    closures = _direct_closures(node)

    assert sorted(closures) == sorted(entry.closures)
    for name, closure in closures.items():
        assert _runs_context_sql(closure, module) is (entry.closures[name] is not Seam.COMPUTE), name


# Part 5: signatures, dispatchers and cases.


@pytest.mark.parametrize('key', _keys(*GUARDED))
def test_guarded_method_requires_a_scope(key: str) -> None:
    """A guarded method takes a keyword-only ``scope`` with no default, typed per its class."""
    parameter, annotation = _scope_parameter(key)
    expected = AccessScope if REGISTRY[key].seam is Seam.OWNER_KEYED else Scope

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    assert annotation is expected


@pytest.mark.parametrize('target', SCOPE_CARRYING_HELPERS)
def test_scope_carrying_helper_requires_a_scope(target: str) -> None:
    """A tool-layer helper that reaches a repository takes a required keyword-only scope."""
    parameter, annotation = _scope_parameter(target)

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


@pytest.mark.parametrize('key', _keys(*GUARDED))
def test_guarded_method_names_registered_cases(key: str) -> None:
    """A guarded method names cases that run on both backends, filtered when it accepts filters."""
    names = REGISTRY[key].cases
    accepts_filters = 'metadata_filters' in inspect.signature(_resolve(key)[1]).parameters

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


@pytest.mark.parametrize('key', _keys(*CHILD))
def test_child_reader_and_writer_sites_are_pinned(key: str) -> None:
    """A new reference to a child reader or writer fails until its id source is reviewed."""
    entry = REGISTRY[key]
    name = key.rsplit('.', 1)[1]
    sites = Counter(
        f'{module.path}:{function}'
        for module in _app_modules()
        for function, attribute in _qualified_references(module)
        if attribute.attr == name and (entry.receiver is None or _receiver_name(attribute.value) == entry.receiver)
    )

    assert sites == Counter(entry.sites)


def test_no_context_sql_outside_the_repositories() -> None:
    """Context-table SQL lives only in the repositories, migrations, CLI, schemas and the predicate builder."""
    stray = [
        f'{module.path}:{line}'
        for module in _app_modules()
        if module.path.split('/')[0] not in SQL_HOMES and module.path != PREDICATE_MODULE
        for line, value in _strings(module.tree, module)
        if CONTEXT_TABLE_SQL.search(value)
    ]

    assert stray == []


def test_system_scope_is_used_only_by_the_fts_migration() -> None:
    """No request path can reach the system scope: its one consumer is the FTS migration."""
    users = {
        module.path
        for module in _app_modules()
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
    resolutions = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _callee(node) in RESOLVERS]

    assert len(resolutions) == 1


def test_no_string_constant_names_the_shared_visibility() -> None:
    """The visibility vocabulary is exactly ``private`` and ``public``."""
    hits = [
        f'{module.path}:{line}'
        for module in _app_modules()
        for line, value in _strings(module.tree, module)
        if value == 'shared'
    ]

    assert hits == []
