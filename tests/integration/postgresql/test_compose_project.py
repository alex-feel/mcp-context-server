"""Each checkout runs the PostgreSQL test container under its own compose project.

The project name is derived from the repository root, so a test session in one
checkout (a separate clone or a linked working tree) never brings down the
container of a session running in another. Within one checkout, a session that
never brings the container up leaves it alone at its end, so a unit-test session
can finish beside a running integration session. These checks need no Docker:
they exercise the name helper, the argv every compose call is built from, and
the session-end cleanup.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.integration.postgresql.conftest import COMPOSE_FILE
from tests.integration.postgresql.conftest import PROJECT_NAME
from tests.integration.postgresql.conftest import SESSION_CONTAINER
from tests.integration.postgresql.conftest import _compose_command
from tests.integration.postgresql.conftest import _force_cleanup
from tests.integration.postgresql.conftest import _project_name_for
from tests.integration.postgresql.conftest import _wait_for_health
from tests.integration.postgresql.conftest import pytest_sessionfinish

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_PROJECT_NAME = re.compile(r'[a-z0-9][a-z0-9-]*')


def test_different_roots_get_different_names(tmp_path: Path) -> None:
    assert _project_name_for(tmp_path / 'main') != _project_name_for(tmp_path / 'second-checkout')


def test_the_same_root_gets_the_same_name(tmp_path: Path) -> None:
    assert _project_name_for(tmp_path / 'main') == _project_name_for(tmp_path / 'main')


def test_the_name_is_a_valid_compose_project_name(tmp_path: Path) -> None:
    name = _project_name_for(tmp_path / 'Mixed Case_Checkout')

    assert COMPOSE_PROJECT_NAME.fullmatch(name)
    assert re.fullmatch(r'mcp-context-server-test-pg-[0-9a-f]{8}', name)


def test_the_session_project_is_named_for_this_checkout() -> None:
    assert _project_name_for(REPOSITORY_ROOT) == PROJECT_NAME


def test_compose_commands_name_the_checkout_project_and_the_test_file() -> None:
    assert _compose_command('ps') == [
        'docker', 'compose', '-p', PROJECT_NAME, '-f', str(COMPOSE_FILE), 'ps',
    ]


def _record_compose_calls(
    monkeypatch: pytest.MonkeyPatch, stdout: str,
) -> list[list[str]]:
    """Replace ``subprocess.run`` with a recorder that answers every call with ``stdout``.

    Returns:
        The list each recorded argv is appended to.
    """
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout, '')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    return calls


def test_cleanup_brings_down_only_the_checkout_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, 'which', lambda _name: 'docker')
    calls = _record_compose_calls(monkeypatch, stdout='')

    _force_cleanup()

    assert calls == [_compose_command('down', '-v', '--remove-orphans')]


def test_health_polling_reads_the_checkout_project(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_compose_calls(monkeypatch, stdout='healthy')

    _wait_for_health(timeout_s=5.0)

    assert calls == [_compose_command('ps', '--format', '{{.Health}}')]


def test_session_end_leaves_a_container_the_session_did_not_bring_up(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
) -> None:
    monkeypatch.setattr(shutil, 'which', lambda _name: 'docker')
    monkeypatch.setattr(SESSION_CONTAINER, 'brought_up', False)
    calls = _record_compose_calls(monkeypatch, stdout='')

    pytest_sessionfinish(request.session, 0)

    assert calls == []


def test_session_end_brings_down_a_container_the_session_brought_up(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
) -> None:
    monkeypatch.setattr(shutil, 'which', lambda _name: 'docker')
    monkeypatch.setattr(SESSION_CONTAINER, 'brought_up', True)
    calls = _record_compose_calls(monkeypatch, stdout='')

    pytest_sessionfinish(request.session, 0)

    assert calls == [_compose_command('down', '-v', '--remove-orphans')]
