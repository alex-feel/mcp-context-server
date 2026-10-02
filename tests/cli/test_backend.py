"""Tests for the backend construction shared by the maintenance CLI modes."""

import pytest

from app.cli import _backend
from app.cli._backend import make_backend


class _StubBackend:
    """Stand-in for the backend the faked factory returns; ``make_backend`` only constructs it."""


def test_make_backend_forwards_provision_vector_for_postgresql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``make_backend`` threads ``provision_vector`` into the PostgreSQL factory."""
    captured: dict[str, object] = {}

    def _fake_create_backend(**kwargs: object) -> _StubBackend:
        captured.update(kwargs)
        return _StubBackend()

    monkeypatch.setattr(_backend, 'create_backend', _fake_create_backend)

    make_backend('postgresql://u:p@h:5432/db', provision_vector=False)

    assert captured['backend_type'] == 'postgresql'
    assert captured['connection_string'] == 'postgresql://u:p@h:5432/db'
    assert captured['provision_vector'] is False


def test_make_backend_ignores_provision_vector_for_sqlite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLite construction never receives ``provision_vector`` (its vec load is unconditional)."""
    captured: dict[str, object] = {}

    def _fake_create_backend(**kwargs: object) -> _StubBackend:
        captured.update(kwargs)
        return _StubBackend()

    monkeypatch.setattr(_backend, 'create_backend', _fake_create_backend)

    make_backend('sqlite:///tmp/context.db', provision_vector=True)

    assert captured['backend_type'] == 'sqlite'
    assert 'provision_vector' not in captured
