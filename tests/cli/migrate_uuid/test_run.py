"""Tests for the migration report file and the stdout summary."""

import json
from pathlib import Path

import pytest

from app.cli.migrate import main as cli_main


class TestReportOutput:
    """JSON report path and stdout summary behavior."""

    def test_report_written_to_path(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
        tmp_path: Path,
    ) -> None:
        """``--report`` writes a JSON file matching MigrationStats.to_dict() shape."""
        report = tmp_path / 'report.json'
        exit_code = cli_main(
            [
                '--source-url',
                f'sqlite:///{legacy_source_db.as_posix()}',
                '--target-url',
                f'sqlite:///{new_target_db_path.as_posix()}',
                '--report',
                str(report),
            ],
        )
        assert exit_code == 0
        assert report.exists()
        loaded = json.loads(report.read_text(encoding='utf-8'))
        assert 'rows_migrated' in loaded
        assert 'references_rewritten' in loaded
        assert 'warnings' in loaded
        assert 'errors' in loaded

    def test_summary_printed_to_stdout(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without ``--report``, the summary appears on stdout."""
        cli_main(
            [
                '--source-url',
                f'sqlite:///{legacy_source_db.as_posix()}',
                '--target-url',
                f'sqlite:///{new_target_db_path.as_posix()}',
            ],
        )
        captured = capsys.readouterr()
        assert 'Migration summary' in captured.out
        assert 'rows migrated' in captured.out
