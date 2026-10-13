"""Tests for ``mcp-context-server-migrate --disable-rls``.

The mode builds the maintenance backend from ``--source-url``, refuses a role that
row-level security filters, then removes every row-level security object the server
installed, refusing while members of the policy role are connected unless ``--force``.
These tests drive it with recording stand-ins for the backend, the precondition and the
removal, plus a real SQLite database, which carries nothing to remove. The PostgreSQL
behavior end to end lives in ``tests/integration/postgresql/test_rls_cli_postgresql.py``.
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.cli import migrate_disable_rls
from app.cli._backend import MaintenanceConnectionError
from app.cli.migrate import build_parser
from app.cli.migrate import main as cli_main
from app.cli.migrate_disable_rls import run_disable_rls
from app.migrations.rls_removal import MemberSession
from app.migrations.rls_removal import MemberSessionsConnectedError
from app.migrations.rls_removal import RlsRemovalReport
from app.settings import get_settings

SOURCE = 'postgresql://mcp_owner:owner-secret@db.internal:5432/mcp_context'
REPORT = RlsRemovalReport(
    statements=[
        'DROP VIEW IF EXISTS "public".context_entries_readable',
        'DROP POLICY context_entries_read ON "public".context_entries',
    ],
    disabled=['context_entries'],
    left_enabled={'tags': 'preenabled=true'},
)
REFUSAL = (
    "This command must connect with the table owner's credentials: row-level security filters mcp_app on "
    "context_entries. Rerun with the owner's connection string (docs/migration-v2-to-v3.md)."
)


class _Backend:
    """A PostgreSQL backend stand-in that records its lifecycle."""

    backend_type = 'postgresql'

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.initialize = AsyncMock(side_effect=lambda: events.append('initialize'))
        self.shutdown = AsyncMock(side_effect=lambda: events.append('shutdown'))


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    report: RlsRemovalReport = REPORT,
    removal_error: BaseException | None = None,
    precondition_error: BaseException | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Replace the backend factory, the precondition and the removal with recording stand-ins.

    Returns:
        The ordered events and the keyword arguments the removal was called with.
    """
    events: list[str] = []
    removal: dict[str, Any] = {}
    backend = _Backend(events)

    def _make_backend(source_url: str, *, provision_vector: bool | None = None) -> _Backend:
        assert source_url == SOURCE
        assert provision_vector is False
        events.append('make_backend')
        return backend

    async def _assert(target: object, schema: str) -> None:
        assert target is backend
        assert schema == get_settings().storage.postgresql_schema
        events.append('precondition')
        if precondition_error is not None:
            raise precondition_error

    async def _remove(target: object, _settings: object, *, dry_run: bool, force: bool) -> RlsRemovalReport:
        assert target is backend
        events.append('remove')
        removal.update(dry_run=dry_run, force=force)
        if removal_error is not None:
            raise removal_error
        return report

    monkeypatch.setattr(migrate_disable_rls, 'make_backend', _make_backend)
    monkeypatch.setattr(migrate_disable_rls, 'assert_maintenance_connection', _assert)
    monkeypatch.setattr(migrate_disable_rls, 'remove_rls', _remove)
    return events, removal


class TestRunDisableRls:
    """The mode checks the owner first, then removes or reports, and maps each outcome to an exit code."""

    def test_a_real_run_removes_and_prints_the_report(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        events, removal = _install(monkeypatch)

        rc = run_disable_rls(SOURCE, dry_run=False, force=False)

        assert rc == 0
        assert events == ['make_backend', 'initialize', 'precondition', 'remove', 'shutdown']
        assert removal == {'dry_run': False, 'force': False}
        err = capsys.readouterr().err
        assert 'Issued 2 statements:' in err
        assert '  DROP VIEW IF EXISTS "public".context_entries_readable' in err
        assert 'Disabled row-level security on: context_entries' in err
        assert 'Left row-level security enabled on: tags (preenabled=true)' in err
        assert 'mcp_owner:***@' in err
        assert 'owner-secret' not in err

    def test_a_dry_run_prints_what_a_real_run_would_do(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, removal = _install(monkeypatch)

        rc = run_disable_rls(SOURCE, dry_run=True, force=False)

        assert rc == 0
        assert removal == {'dry_run': True, 'force': False}
        err = capsys.readouterr().err
        assert '[DRY-RUN] Would issue 2 statements:' in err
        assert '[DRY-RUN] Would disable row-level security on: context_entries' in err
        assert '[DRY-RUN] Would leave row-level security enabled on: tags (preenabled=true)' in err

    def test_nothing_installed_is_a_success(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        _install(monkeypatch, report=RlsRemovalReport(statements=[], disabled=[], left_enabled={}))

        rc = run_disable_rls(SOURCE, dry_run=False, force=False)

        assert rc == 0
        assert 'No row-level security object of this server is installed; nothing to remove.' in capsys.readouterr().err

    def test_a_table_enabled_before_the_server_is_reported_even_when_nothing_else_is_installed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _install(
            monkeypatch, report=RlsRemovalReport(statements=[], disabled=[], left_enabled={'tags': 'no readable comment'}),
        )

        rc = run_disable_rls(SOURCE, dry_run=False, force=False)

        assert rc == 0
        err = capsys.readouterr().err
        assert 'No row-level security object of this server is installed; nothing to remove.' in err
        assert 'Left row-level security enabled on: tags (no readable comment)' in err

    def test_connected_members_refuse_naming_them_and_the_force_flag(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        sessions = (MemberSession(pid=4242, user='mcp_app', application_name='mcp-context-server'),)
        events, _ = _install(monkeypatch, removal_error=MemberSessionsConnectedError(sessions))

        rc = run_disable_rls(SOURCE, dry_run=False, force=False)

        assert rc == 1
        assert events[-1] == 'shutdown'
        assert capsys.readouterr().err.splitlines()[-1] == (
            '[ERROR] Refusing to remove row-level security: 1 session of a policy-role member is connected: '
            "pid 4242 (mcp_app, application 'mcp-context-server'). Stop every server instance first, "
            'or rerun with --force to remove it while they stay connected.'
        )

    def test_force_reaches_the_removal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, removal = _install(monkeypatch)

        assert run_disable_rls(SOURCE, dry_run=False, force=True) == 0
        assert removal == {'dry_run': False, 'force': True}

    def test_the_owner_refusal_reaches_the_dispatcher_before_any_removal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events, _ = _install(monkeypatch, precondition_error=MaintenanceConnectionError(REFUSAL))

        with pytest.raises(MaintenanceConnectionError):
            run_disable_rls(SOURCE, dry_run=False, force=False)

        assert 'remove' not in events
        assert events[-1] == 'shutdown'

    def test_an_unexpected_failure_exits_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, removal_error=RuntimeError('Row-level security removal failed: connection reset'))

        assert run_disable_rls(SOURCE, dry_run=False, force=False) == 2

    def test_sqlite_has_nothing_to_remove(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = run_disable_rls(f'sqlite:///{tmp_path / "context.db"}', dry_run=False, force=False)

        assert rc == 0
        assert 'No row-level security object of this server is installed; nothing to remove.' in capsys.readouterr().err


class TestDispatch:
    """``--disable-rls`` is an in-place mode of its own, and ``--force`` belongs to it alone."""

    @pytest.mark.parametrize('other', [['--compress'], ['--decompress'], ['--re-embed'], ['--reassign-owner', 'a', 'b']])
    def test_disable_rls_excludes_the_other_modes(self, other: list[str]) -> None:
        with pytest.raises(SystemExit) as exited:
            build_parser().parse_args(['--source-url', SOURCE, '--disable-rls', *other])

        assert exited.value.code == 2

    def test_force_without_disable_rls_is_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exited:
            cli_main(['--source-url', SOURCE, '--reassign-owner', 'a', 'b', '--force'])

        assert exited.value.code == 2
        assert '--force is valid only with --disable-rls' in capsys.readouterr().err

    def test_the_flags_reach_the_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, bool, bool]] = []

        def _run(source_url: str, *, dry_run: bool, force: bool) -> int:
            calls.append((source_url, dry_run, force))
            return 0

        monkeypatch.setattr(migrate_disable_rls, 'run_disable_rls', _run)

        rc = cli_main(['--source-url', SOURCE, '--disable-rls', '--dry-run', '--force'])

        assert rc == 0
        assert calls == [(SOURCE, True, True)]

    def test_the_owner_refusal_exits_1_with_the_remediation(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _install(monkeypatch, precondition_error=MaintenanceConnectionError(REFUSAL))

        rc = cli_main(['--source-url', SOURCE, '--disable-rls'])

        assert rc == 1
        assert capsys.readouterr().err.splitlines()[-1] == REFUSAL
