"""Tests for the mcp-context-server-migrate dispatcher: argument parsing and configuration errors."""

import sqlite3
from pathlib import Path

import pytest

from app.cli.migrate import build_parser
from app.cli.migrate import main as cli_main


class TestCliArgs:
    """Argparse plumbing for the migrate CLI."""

    def test_help_runs_and_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``--help`` exits with code 0 and prints usage."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main(['--help'])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert 'mcp-context-server-migrate' in captured.out
        assert '--source-url' in captured.out

    def test_missing_source_url_errors(self) -> None:
        """Calling without ``--source-url`` triggers a non-zero exit."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main(['--target-url', 'sqlite:///dummy.db'])
        assert exc_info.value.code != 0

    def test_build_parser_has_required_flags(self) -> None:
        """The argparse parser declares all expected options."""
        parser = build_parser()
        actions = {action.dest for action in parser._actions}
        assert {'source_url', 'target_url', 'dry_run', 'report', 'reassign_owner'}.issubset(actions)


class TestSettingsValidationExitCode:
    """A settings ValidationError exits EX_CONFIG (78) on every CLI path.

    The server's guarded import (app/server.py) classifies an import-time
    settings ValidationError as a permanent misconfiguration and exits 78,
    but mcp-context-server-migrate never imports app.server: its in-place
    flows reach get_settings() through the lazily imported backend modules.
    Without its own mapping, a value like RETRY_MAX_RETRIES=0 would surface
    there as a raw multi-frame pydantic traceback with the generic exit 1 --
    the failure mode the server guard eliminates, on a second first-class
    entry point.
    """

    def test_in_place_dispatch_maps_validation_error_to_exit_78(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--compress under an invalid env exits 78 with the pydantic detail."""
        from app.errors import ConfigurationError
        from app.settings import get_settings

        db = tmp_path / 'src.db'
        sqlite3.connect(str(db)).close()
        monkeypatch.setenv('RETRY_MAX_RETRIES', '0')
        get_settings.cache_clear()
        try:
            rc = cli_main(['--source-url', f'sqlite:///{db}', '--compress', '--dry-run'])
        finally:
            monkeypatch.delenv('RETRY_MAX_RETRIES', raising=False)
            get_settings.cache_clear()

        assert rc == ConfigurationError.EXIT_CODE
        err = capsys.readouterr().err
        assert 'Configuration invalid' in err
        assert 'RETRY_MAX_RETRIES' in err or 'retry_max_retries' in err


def test_build_parser_accepts_compress_flag() -> None:
    """``--compress`` is parsed as a boolean toggle."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--compress',
    ])
    assert args.compress is True
    assert args.decompress is False


def test_build_parser_accepts_decompress_flag() -> None:
    """``--decompress`` is parsed as a boolean toggle."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--decompress',
    ])
    assert args.decompress is True
    assert args.compress is False


def test_build_parser_rejects_both_flags() -> None:
    """``--compress`` and ``--decompress`` are mutually exclusive."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            '--source-url', 'sqlite:///fake.db',
            '--compress', '--decompress',
        ])


def test_build_parser_target_url_optional_when_compress() -> None:
    """``--target-url`` is optional when --compress is set."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--compress',
    ])
    assert args.target_url is None
