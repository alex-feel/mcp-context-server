"""Structural test: every tool that can write refuses the read-only principal before it does anything else.

``reject_read_only_principal`` refuses the principal named by
``ACCESS_CONTROL_ANONYMOUS_PRINCIPAL``. For every tool whose annotations do not mark
it read-only, the tool function must call the gate exactly once, naming itself, and
the call must precede, by source position, every repository access
(``ensure_repositories()`` and any ``repos.<attribute>``), every id or prefix
resolution and every generation call, so a refused caller never reaches the
database, an embedding or summary provider, or an argument check that could reveal
whether an entry exists. A read-only tool calls no gate: the read-only principal
reads the entries it may read like any other caller.
"""

import ast
import inspect
import textwrap

import pytest

import app.tools

GATE = 'reject_read_only_principal'
GUARDED_CALLS = frozenset({
    'ensure_repositories',
    'resolve_or_normalize_id',
    'resolve_or_normalize_ids',
    'resolve_prefix',
    'run_generation',
})
GENERATION_PREFIX = 'generate_'
REPOSITORIES = 'repos'

WRITE_TOOLS = sorted(
    name for name, annotations in app.tools.TOOL_ANNOTATIONS.items() if annotations.get('readOnlyHint') is not True
)
READ_TOOLS = sorted(set(app.tools.TOOL_ANNOTATIONS) - set(WRITE_TOOLS))


def _callee(call: ast.Call) -> str | None:
    """Return the bare name a call invokes: ``f(...)`` and ``x.f(...)`` both give ``f``."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _guarded_action(node: ast.AST) -> str | None:
    """Name the repository access, id resolution or generation a node performs, if any."""
    if isinstance(node, ast.Call):
        callee = _callee(node)
        if callee is not None and (callee in GUARDED_CALLS or callee.startswith(GENERATION_PREFIX)):
            return f'{callee}()'
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == REPOSITORIES:
        return f'{REPOSITORIES}.{node.attr}'
    return None


def _position(node: ast.expr) -> tuple[int, int]:
    """Return a node's source position, comparable within one parsed function."""
    return node.lineno, node.col_offset


def _tool_function(source: str, tool: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    """Parse a tool's source and return the function that implements it."""
    tree = ast.parse(textwrap.dedent(source))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == tool
    )


def _gate_calls(function: ast.AsyncFunctionDef | ast.FunctionDef) -> list[ast.Call]:
    """Return every call of the read-only gate inside a function."""
    return [node for node in ast.walk(function) if isinstance(node, ast.Call) and _callee(node) == GATE]


def gate_problems(source: str, tool: str) -> list[str]:
    """Return every way a write tool's source breaks the read-only gate rule.

    Args:
        source: The source of the module or function that defines the tool.
        tool: The tool's name, which is also its function's name.

    Returns:
        One message per violation; empty when the tool calls the gate exactly once,
        naming itself, before any repository access, id resolution or generation.
    """
    function = _tool_function(source, tool)
    gates = _gate_calls(function)
    if len(gates) != 1:
        return [f'{tool} calls {GATE} {len(gates)} times instead of once']
    gate = gates[0]
    problems: list[str] = []
    named = [keyword.value for keyword in gate.keywords if keyword.arg == 'tool']
    if not (len(named) == 1 and isinstance(named[0], ast.Constant) and named[0].value == tool):
        problems.append(f"{tool} calls {GATE} without tool='{tool}'")
    for node in ast.walk(function):
        action = _guarded_action(node)
        if action is not None and isinstance(node, ast.expr) and _position(node) < _position(gate):
            problems.append(f'{tool} reaches {action} at line {node.lineno} before {GATE}')
    return problems


def _source(tool: str) -> str:
    """Return the source of a registered tool's function."""
    return inspect.getsource(inspect.unwrap(getattr(app.tools, tool)))


def test_the_annotations_name_write_tools() -> None:
    """The rule has tools to check: the annotations mark some tools as able to write."""
    assert WRITE_TOOLS
    assert READ_TOOLS


@pytest.mark.parametrize('tool', WRITE_TOOLS)
def test_write_tool_refuses_the_read_only_principal_first(tool: str) -> None:
    """A write tool calls the gate once, naming itself, before it touches anything."""
    assert gate_problems(_source(tool), tool) == []


@pytest.mark.parametrize('tool', READ_TOOLS)
def test_read_tool_calls_no_gate(tool: str) -> None:
    """A read-only tool lets the read-only principal read like any other caller."""
    assert _gate_calls(_tool_function(_source(tool), tool)) == []


_GATE_AFTER_REPOSITORIES = '''
async def store_context(text):
    repos = await ensure_repositories()
    principal = resolve_effective_principal()
    reject_read_only_principal(principal, tool='store_context')
    await repos.context.insert_entry(text)
'''

_GATE_AFTER_GENERATION = '''
async def update_context(context_id, text):
    principal = resolve_effective_principal()
    embeddings = await generate_embeddings_with_timeout(text)
    reject_read_only_principal(principal, tool='update_context')
    return embeddings
'''

_GATE_AFTER_ID_RESOLUTION = '''
async def delete_context(context_ids):
    principal = resolve_effective_principal()
    context_ids = await resolve_or_normalize_ids(context_ids)
    reject_read_only_principal(principal, tool='delete_context')
    return context_ids
'''

_NO_GATE = '''
async def store_context_batch(entries):
    principal = resolve_effective_principal()
    repos = await ensure_repositories()
    return repos, principal, entries
'''

_REPOSITORIES_READ_BEFORE_THE_GATE = '''
async def delete_context_batch(context_ids, repos):
    principal = resolve_effective_principal()
    context = repos.context
    reject_read_only_principal(principal, tool='delete_context_batch')
    return context, context_ids
'''

_GATE_NAMING_ANOTHER_TOOL = '''
async def update_context_batch(updates):
    principal = resolve_effective_principal()
    reject_read_only_principal(principal, tool='update_context')
    repos = await ensure_repositories()
    return repos, updates
'''


@pytest.mark.parametrize(
    ('source', 'tool', 'expected'),
    [
        (_GATE_AFTER_REPOSITORIES, 'store_context', 'store_context reaches ensure_repositories() at line 3'),
        (_GATE_AFTER_GENERATION, 'update_context', 'update_context reaches generate_embeddings_with_timeout()'),
        (_GATE_AFTER_ID_RESOLUTION, 'delete_context', 'delete_context reaches resolve_or_normalize_ids()'),
        (_NO_GATE, 'store_context_batch', 'store_context_batch calls reject_read_only_principal 0 times'),
        (_GATE_NAMING_ANOTHER_TOOL, 'update_context_batch', "without tool='update_context_batch'"),
    ],
    ids=['after-repositories', 'after-generation', 'after-id-resolution', 'missing', 'wrong-tool-name'],
)
def test_the_rule_rejects_a_misplaced_or_missing_gate(source: str, tool: str, expected: str) -> None:
    """The rule is not vacuous: a gate after any guarded action, missing or misnamed fails it."""
    problems = gate_problems(source, tool)

    assert any(expected in problem for problem in problems), problems


def test_the_rule_flags_a_repository_attribute_read_before_the_gate() -> None:
    """Reading the repository container before the gate is a violation even without a call."""
    problems = gate_problems(_REPOSITORIES_READ_BEFORE_THE_GATE, 'delete_context_batch')

    assert problems == ['delete_context_batch reaches repos.context at line 4 before reject_read_only_principal']
