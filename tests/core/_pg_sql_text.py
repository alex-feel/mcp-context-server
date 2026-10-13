"""Quote- and parenthesis-aware readers of rendered PostgreSQL metadata-filter SQL.

The cast guard and the totality matrix inspect the SQL the metadata builder
renders: which operand each cast applies to, and which operands each AND/OR
joins. Both read it structurally, so a string literal, a quoted identifier or a
nested parenthesis never changes the answer. Every reader works on a masked copy
of the text in which the bodies of single-quoted literals and double-quoted
identifiers are blanked, so their characters never count as structure.
"""

import re
from dataclasses import dataclass

# Target types a cast may convert a column-derived operand to without being able to
# raise on any stored value: text input accepts every text.
TEXT_TYPES = frozenset({'text'})

_WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_CAST_OPERATOR = re.compile(r'::\s*([A-Za-z_][A-Za-z0-9_]*(?:\s+precision)?)\s*(\[\])?', re.IGNORECASE)
_CAST_FUNCTION = re.compile(r'\bCAST\s*\(', re.IGNORECASE)
_COLUMN_WORD = re.compile(r'\b(?:metadata|elem)\b')
_OPERAND_TAIL = re.compile(r'[A-Za-z0-9_$.]')


@dataclass(frozen=True)
class Cast:
    """One cast in rendered SQL: the operand text and the target type it converts to."""

    operand: str
    target_type: str


def mask_quoted(sql: str) -> str:
    """Blank the bodies of single-quoted literals and double-quoted identifiers.

    The delimiting quotes stay in place and every body character (a doubled
    quote included) becomes a space, so the masked text has the same length and
    the same positions as ``sql``. Backslashes are ordinary characters, as under
    ``standard_conforming_strings``.

    Args:
        sql: The SQL text to mask.

    Returns:
        The masked text.
    """
    masked: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(sql):
        char = sql[index]
        if quote is None:
            masked.append(char)
            if char in ('"', "'"):
                quote = char
        elif char == quote and sql[index + 1:index + 2] == quote:
            masked.append('  ')
            index += 1
        elif char == quote:
            masked.append(char)
            quote = None
        else:
            masked.append(' ')
        index += 1
    return ''.join(masked)


def _matching_open(masked: str, close: int) -> int:
    """Return the index of the parenthesis that ``masked[close]`` closes."""
    depth = 0
    for index in range(close, -1, -1):
        if masked[index] == ')':
            depth += 1
        elif masked[index] == '(':
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f'unbalanced parentheses before position {close}')


def _matching_close(masked: str, open_: int) -> int:
    """Return the index of the parenthesis that closes ``masked[open_]``."""
    depth = 0
    for index in range(open_, len(masked)):
        if masked[index] == '(':
            depth += 1
        elif masked[index] == ')':
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f'unbalanced parentheses after position {open_}')


def _operand_start(masked: str, end: int) -> int:
    """Return where the primary expression ending just before ``end`` begins.

    The operand of a ``::`` cast is the primary expression immediately to its
    left: a parenthesized group (together with the function name in front of
    it), a quoted literal or identifier, or a bare token such as a column, a
    parameter or a number. When that token is the type name of a preceding cast
    (``x::numeric::float8``), the operand is the whole preceding cast.

    Args:
        masked: The masked SQL text.
        end: The position just past the operand.

    Returns:
        The position of the operand's first character.
    """
    last = end - 1
    if masked[last] == ')':
        start = _matching_open(masked, last)
        while start > 0 and (masked[start - 1].isalnum() or masked[start - 1] == '_'):
            start -= 1
        return start
    if masked[last] in ('"', "'"):
        return masked.rindex(masked[last], 0, last)
    start = end
    while start > 0 and _OPERAND_TAIL.match(masked[start - 1]):
        start -= 1
    before = masked[:start].rstrip()
    if before.endswith('::'):
        return _operand_start(masked, len(before[:-2].rstrip()))
    return start


def find_casts(sql: str) -> list[Cast]:
    """Return every ``::type`` and ``CAST(... AS type)`` cast in ``sql``.

    Args:
        sql: Rendered SQL text.

    Returns:
        The casts in order of appearance, each with its operand text and target type.

    Raises:
        ValueError: If a ``CAST(`` call has no top-level ``AS``.
    """
    masked = mask_quoted(sql)
    casts: list[Cast] = []
    for match in _CAST_OPERATOR.finditer(masked):
        end = match.start()
        while end > 0 and masked[end - 1].isspace():
            end -= 1
        start = _operand_start(masked, end)
        target = match.group(1) + (match.group(2) or '')
        casts.append(Cast(operand=sql[start:end], target_type=' '.join(target.split())))
    for match in _CAST_FUNCTION.finditer(masked):
        open_ = match.end() - 1
        close = _matching_close(masked, open_)
        body = masked[open_ + 1:close]
        separators = [m for m in re.finditer(r'\bAS\b', body, re.IGNORECASE) if _depth_at(body, m.start()) == 0]
        if not separators:
            raise ValueError(f'CAST without AS at position {match.start()}')
        separator = separators[-1]
        operand = sql[open_ + 1:open_ + 1 + separator.start()].strip()
        target = sql[open_ + 1 + separator.end():close].strip()
        casts.append(Cast(operand=operand, target_type=' '.join(target.split())))
    return casts


def _depth_at(masked: str, position: int) -> int:
    """Return the parenthesis depth of ``masked`` at ``position``."""
    return masked[:position].count('(') - masked[:position].count(')')


def is_column_derived(operand: str) -> bool:
    """Return True when an operand reads stored data.

    An operand is column-derived when, outside its quoted literals, it names the
    ``metadata`` column or an ``elem`` array element, or applies a ``#>``/``#>>``
    path accessor.

    Args:
        operand: The operand text of a cast.

    Returns:
        Whether the operand's value comes from stored metadata.
    """
    masked = mask_quoted(operand)
    return bool(_COLUMN_WORD.search(masked)) or '#>' in masked


def column_derived_casts(sql: str) -> list[Cast]:
    """Return the casts in ``sql`` that convert stored data to a non-text type.

    Casts of bound parameters and literals are not column-derived and are never
    returned, whatever their target type.

    Args:
        sql: Rendered SQL text.

    Returns:
        The offending casts, in order of appearance.
    """
    return [
        cast
        for cast in find_casts(sql)
        if cast.target_type.lower() not in TEXT_TYPES and is_column_derived(cast.operand)
    ]


def _top_level_words(masked: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, WORD)`` for every word outside parentheses and CASE bodies.

    The AND of a ``BETWEEN x AND y`` belongs to the BETWEEN and is not returned.

    Args:
        masked: The masked SQL text.

    Returns:
        The top-level words in order of appearance, upper-cased.
    """
    words: list[tuple[int, int, str]] = []
    depth = 0
    case_depth = 0
    between_pending = False
    index = 0
    while index < len(masked):
        char = masked[index]
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif (char.isalpha() or char == '_') and not _continues_token(masked, index):
            match = _WORD.match(masked, index)
            if match is not None:
                word = match.group().upper()
                if depth == 0:
                    if word == 'CASE':
                        case_depth += 1
                    elif word == 'END' and case_depth > 0:
                        case_depth -= 1
                    elif case_depth == 0:
                        if word == 'BETWEEN':
                            between_pending = True
                        elif word == 'AND' and between_pending:
                            between_pending = False
                        else:
                            words.append((match.start(), match.end(), word))
                index = match.end()
                continue
        index += 1
    return words


def _continues_token(masked: str, index: int) -> bool:
    """Return True when ``masked[index]`` continues the token before it."""
    return index > 0 and (masked[index - 1].isalnum() or masked[index - 1] in '_$')


def _unwrap(expr: str) -> str:
    """Strip leading NOTs and parentheses that enclose the whole expression."""
    while True:
        expr = expr.strip()
        masked = mask_quoted(expr)
        prefix = re.match(r'NOT\b', masked, re.IGNORECASE)
        if prefix is not None:
            expr = expr[prefix.end():]
            continue
        if masked.startswith('(') and _matching_close(masked, 0) == len(masked) - 1:
            expr = expr[1:-1]
            continue
        return expr


def _split(expr: str, connective: str) -> list[str]:
    """Split ``expr`` at its top-level ``connective`` (AND or OR)."""
    masked = mask_quoted(expr)
    parts: list[str] = []
    start = 0
    for word_start, word_end, word in _top_level_words(masked):
        if word == connective:
            parts.append(expr[start:word_start].strip())
            start = word_end
    parts.append(expr[start:].strip())
    return parts


def boolean_operands(expr: str) -> list[str]:
    """Return every maximal operand of every AND and OR in ``expr``, at any depth.

    OR binds looser than AND, so an expression is split at its top-level ORs when
    it has any and at its top-level ANDs otherwise; each operand is then split the
    same way after any enclosing NOT and parentheses are removed. A CASE body is
    never split, because CASE fixes the order in which its parts are evaluated.

    Args:
        expr: A rendered boolean SQL expression.

    Returns:
        The operands in order of appearance, outer operands before their own operands.
    """
    operands: list[str] = []
    inner = _unwrap(expr)
    for connective in ('OR', 'AND'):
        parts = _split(inner, connective)
        if len(parts) > 1:
            for part in parts:
                operands.append(part)
                operands.extend(boolean_operands(part))
            break
    return operands
