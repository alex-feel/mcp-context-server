"""The documentation sections that server messages and deployment files point operators to exist.

Startup refusals, command-line remediations and deployment-file comments name a
documentation file and, often, a section heading in it (``docs/database-backends.md,
'Row-level security (PostgreSQL)'`` or ``"Enabling row-level security" in
docs/deployment/docker.md``). An operator who reads such a message follows that pointer,
so every named file must exist and every named heading must be a heading of that file.

References are read from the string constants of ``app/`` (implicitly concatenated
literals joined, as the message prints them) and from the text of the Compose, Helm and
shell files under ``deploy/``.
"""

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DEPLOY_SUFFIXES = frozenset({'.yml', '.yaml', '.tpl', '.sh'})
_DOC_PATH = r'docs/(?P<path>[\w./-]+\.md)'
_QUOTED = r'(?P<quote>[\'"])(?P<heading>[^\'"\n]+)(?P=quote)'
# "docs/<file>.md, 'Heading'" and "docs/<file>.md "Heading"".
_PATH_THEN_HEADING = re.compile(_DOC_PATH + r',?[ \t]+' + _QUOTED)
# '"Heading" in docs/<file>.md'.
_HEADING_THEN_PATH = re.compile(_QUOTED + r'[ \t]+in[ \t]+' + _DOC_PATH)
_ANY_DOC_PATH = re.compile(_DOC_PATH)
_HEADING_LINE = re.compile(r'^#{1,6}[ \t]+(?P<text>.+?)[ \t]*$')
_FENCE = re.compile(r'^[ \t]*(```|~~~)')


def _string_constants(source: str) -> list[str]:
    """Return every string literal of a module, an f-string's placeholders rendered as ``{}``."""
    texts: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            texts.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            texts.append(''.join(
                part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else '{}'
                for part in node.values
            ))
    return texts


def _texts() -> list[tuple[str, str]]:
    """Return ``(origin, text)`` for every text that can carry a documentation reference."""
    texts = [
        (path.relative_to(_ROOT).as_posix(), text)
        for path in sorted((_ROOT / 'app').rglob('*.py'))
        for text in _string_constants(path.read_text(encoding='utf-8'))
    ]
    texts.extend(
        (path.relative_to(_ROOT).as_posix(), path.read_text(encoding='utf-8'))
        for path in sorted((_ROOT / 'deploy').rglob('*'))
        if path.is_file() and path.suffix in _DEPLOY_SUFFIXES
    )
    return texts


def _section_references() -> list[tuple[str, str, str]]:
    """Return every ``(origin, document, heading)`` reference, without repeats."""
    found: dict[tuple[str, str, str], None] = {}
    for origin, text in _texts():
        for pattern in (_PATH_THEN_HEADING, _HEADING_THEN_PATH):
            for match in pattern.finditer(text):
                found[(origin, match['path'], match['heading'])] = None
    return list(found)


def _headings(document: Path) -> set[str]:
    """Return the text of every Markdown heading of ``document`` outside code fences."""
    headings: set[str] = set()
    in_fence = False
    for line in document.read_text(encoding='utf-8').splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and (match := _HEADING_LINE.match(line)):
            headings.add(match['text'])
    return headings


SECTION_REFERENCES = _section_references()


def test_the_references_operators_meet_are_collected() -> None:
    """The collector finds the section pointers of the startup, removal and deployment messages."""
    pairs = {(document, heading) for _, document, heading in SECTION_REFERENCES}
    assert ('database-backends.md', 'Row-level security (PostgreSQL)') in pairs
    assert ('database-backends.md', 'Disabling row-level security') in pairs
    assert ('database-backends.md', 'Session Pooler Connection Limits') in pairs
    assert ('deployment/docker.md', 'Enabling row-level security') in pairs


@pytest.mark.parametrize(
    ('origin', 'document', 'heading'),
    SECTION_REFERENCES,
    ids=[f'{origin}->{document}#{heading}' for origin, document, heading in SECTION_REFERENCES],
)
def test_referenced_section_is_a_heading_of_its_document(origin: str, document: str, heading: str) -> None:
    """Each named heading is a heading of the named document."""
    path = _ROOT / 'docs' / document
    assert path.is_file(), f'{origin} names docs/{document}, which does not exist'
    assert heading in _headings(path), f"{origin} names the section '{heading}', which docs/{document} lacks"


def test_referenced_documents_exist() -> None:
    """Every documentation file named in app/ or deploy/ exists."""
    missing = sorted({
        f'{origin}: docs/{match["path"]}'
        for origin, text in _texts()
        for match in _ANY_DOC_PATH.finditer(text)
        if not (_ROOT / 'docs' / match['path']).is_file()
    })
    assert not missing
