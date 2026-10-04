"""Unit tests for the --reassign-owner mode of mcp-context-server-migrate.

Every test drives the dispatcher ``main()`` with an explicit SQLite database
under ``tmp_path`` built from the base schema, so no test reads the routing
variables of the surrounding environment. The PostgreSQL counterpart lives in
``tests/integration/postgresql/test_migrate_reassign_owner_postgresql.py``.
"""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from app.cli.migrate import build_parser
from app.cli.migrate import main as cli_main
from app.errors import ConfigurationError
from app.schemas import load_schema
from app.settings import get_settings
from app.settings.auth import SAFE_PRINCIPAL_ID_PATTERN

_SEEDED_UPDATED_AT = '2000-01-01 00:00:00'

# entry id -> (owner_id, version)
_SEED: dict[str, tuple[str, int]] = {
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa': ('local', 3),
    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb': ('local', 0),
    'cccccccccccccccccccccccccccccccc': ('bob', 5),
}

# (context_entry_id, principal_type, principal_id, permission, granted_by)
_GRANTS: list[tuple[str, str, str, str, str]] = [
    ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'user', 'bob', 'read', 'local'),
    ('cccccccccccccccccccccccccccccccc', 'user', 'local', 'write', 'bob'),
]


def _bootstrap(path: Path) -> None:
    """Create the SQLite base schema and seed two owners plus two grant rows."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(load_schema('sqlite'))
        for entry_id, (owner_id, version) in _SEED.items():
            conn.execute(
                'INSERT INTO context_entries '
                '(id, thread_id, source, content_type, text_content, owner_id, version, updated_at) '
                "VALUES (?, 'thread-a', 'user', 'text', 'doc', ?, ?, ?)",
                (entry_id, owner_id, version, _SEEDED_UPDATED_AT),
            )
        conn.executemany(
            'INSERT INTO context_entry_grants '
            '(context_entry_id, principal_type, principal_id, permission, granted_by) '
            'VALUES (?, ?, ?, ?, ?)',
            _GRANTS,
        )
        conn.commit()
    finally:
        conn.close()


def _entries(path: Path) -> dict[str, tuple[str, int, str]]:
    """Return ``id -> (owner_id, version, updated_at)`` for every context entry."""
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute('SELECT id, owner_id, version, updated_at FROM context_entries').fetchall()
    finally:
        conn.close()
    return {str(r[0]): (str(r[1]), int(r[2]), str(r[3])) for r in rows}


def _grants(path: Path) -> list[tuple[str, str, str, str, str]]:
    """Return every grant row as a sorted list of tuples."""
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute(
            'SELECT context_entry_id, principal_type, principal_id, permission, granted_by '
            'FROM context_entry_grants',
        ).fetchall()
    finally:
        conn.close()
    return sorted((str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4])) for r in rows)


def _unchanged_entries() -> dict[str, tuple[str, int, str]]:
    """Return the seeded entry state as :func:`_entries` reports it."""
    return {entry_id: (owner, version, _SEEDED_UPDATED_AT) for entry_id, (owner, version) in _SEED.items()}


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """SQLite database with entries owned by 'local' and 'bob'."""
    db = tmp_path / 'reassign.db'
    _bootstrap(db)
    return db


class TestParser:
    """Argparse plumbing for --reassign-owner."""

    def test_accepts_two_values(self) -> None:
        """``--reassign-owner FROM TO`` stores both values under ``reassign_owner``."""
        args = build_parser().parse_args(['--source-url', 'sqlite:///fake.db', '--reassign-owner', 'local', 'alice'])
        assert args.reassign_owner == ['local', 'alice']
        assert args.target_url is None

    def test_defaults_to_none(self) -> None:
        """Without the flag the destination is None."""
        args = build_parser().parse_args(['--source-url', 'sqlite:///fake.db', '--compress'])
        assert args.reassign_owner is None

    def test_rejects_a_single_value(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The flag needs exactly two values."""
        with pytest.raises(SystemExit) as exc_info:
            build_parser().parse_args(['--source-url', 'sqlite:///fake.db', '--reassign-owner', 'local'])

        assert exc_info.value.code == 2
        assert 'argument --reassign-owner: expected 2 arguments' in capsys.readouterr().err

    @pytest.mark.parametrize('other_mode', ['--compress', '--decompress', '--re-embed'])
    def test_mutually_exclusive_with_other_modes(self, other_mode: str, capsys: pytest.CaptureFixture[str]) -> None:
        """--reassign-owner joins the exclusive mode group."""
        with pytest.raises(SystemExit) as exc_info:
            build_parser().parse_args([
                '--source-url', 'sqlite:///fake.db', '--reassign-owner', 'local', 'alice', other_mode,
            ])

        assert exc_info.value.code == 2
        assert f'argument {other_mode}: not allowed with argument --reassign-owner' in capsys.readouterr().err


class TestDispatch:
    """main() routes --reassign-owner to run_reassign_owner."""

    def test_routes_without_target_url(self) -> None:
        """The mode runs on --source-url alone and forwards both principals and --dry-run."""
        with (
            patch('app.cli.migrate_reassign_owner.run_reassign_owner', return_value=0) as mock_reassign,
            patch('app.cli.migrate.run_uuid_migration') as mock_uuid,
        ):
            rc = cli_main(['--source-url', 'sqlite:///fake.db', '--reassign-owner', 'local', 'alice', '--dry-run'])

        assert rc == 0
        mock_reassign.assert_called_once_with('sqlite:///fake.db', 'local', 'alice', dry_run=True)
        mock_uuid.assert_not_called()

    def test_co_passed_embed_missing_is_not_run(self) -> None:
        """--reassign-owner returns before the --embed-missing branch."""
        with (
            patch('app.cli.migrate_reassign_owner.run_reassign_owner', return_value=0) as mock_reassign,
            patch('app.cli.migrate_embeddings.run_embed_missing') as mock_embed,
        ):
            rc = cli_main(['--source-url', 'sqlite:///fake.db', '--reassign-owner', 'local', 'alice', '--embed-missing'])

        assert rc == 0
        mock_reassign.assert_called_once()
        mock_embed.assert_not_called()

    def test_invalid_environment_exits_ex_config(
        self,
        seeded_db: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A settings ValidationError exits 78 and changes nothing."""
        monkeypatch.setenv('RETRY_MAX_RETRIES', '0')
        get_settings.cache_clear()

        rc = cli_main(['--source-url', f'sqlite:///{seeded_db}', '--reassign-owner', 'local', 'alice'])

        assert rc == ConfigurationError.EXIT_CODE
        assert 'Configuration invalid' in capsys.readouterr().err
        assert _entries(seeded_db) == _unchanged_entries()


class TestValidation:
    """Invalid principal arguments exit 1 before touching the database."""

    def test_same_principal_exits_one(self, seeded_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """FROM equal to TO is refused and writes nothing."""
        rc = cli_main(['--source-url', f'sqlite:///{seeded_db}', '--reassign-owner', 'local', 'local'])

        assert rc == 1
        assert 'FROM and TO are the same principal' in capsys.readouterr().err
        assert _entries(seeded_db) == _unchanged_entries()

    @pytest.mark.parametrize(
        ('from_principal', 'to_principal'),
        [('', 'alice'), ('local', ''), ('', '')],
        ids=['empty-from', 'empty-to', 'both-empty'],
    )
    def test_empty_value_exits_one(
        self,
        seeded_db: Path,
        capsys: pytest.CaptureFixture[str],
        from_principal: str,
        to_principal: str,
    ) -> None:
        """An empty FROM or TO is refused and writes nothing."""
        rc = cli_main(['--source-url', f'sqlite:///{seeded_db}', '--reassign-owner', from_principal, to_principal])

        assert rc == 1
        assert 'must not be empty' in capsys.readouterr().err
        assert _entries(seeded_db) == _unchanged_entries()


class TestReassignment:
    """End-to-end runs against a seeded SQLite database."""

    def test_dry_run_prints_count_and_changes_nothing(
        self, seeded_db: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--dry-run reports the matching row count and leaves every row as it was."""
        rc = cli_main(['--source-url', f'sqlite:///{seeded_db}', '--reassign-owner', 'local', 'alice', '--dry-run'])

        assert rc == 0
        err = capsys.readouterr().err
        assert '[DRY-RUN] 2 context entries' in err
        assert _entries(seeded_db) == _unchanged_entries()
        assert _grants(seeded_db) == sorted(_GRANTS)

    def test_reassigns_exactly_the_from_rows(
        self, seeded_db: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Only FROM's rows change owner; grants and versions stay; updated_at moves."""
        rc = cli_main(['--source-url', f'sqlite:///{seeded_db}', '--reassign-owner', 'local', 'alice'])

        assert rc == 0
        err = capsys.readouterr().err
        assert "Reassigned 2 context entries from 'local' to 'alice'" in err

        after = _entries(seeded_db)
        for entry_id, (owner, version) in _SEED.items():
            new_owner, new_version, new_updated_at = after[entry_id]
            assert new_version == version
            if owner == 'local':
                assert new_owner == 'alice'
                assert new_updated_at > _SEEDED_UPDATED_AT
            else:
                assert new_owner == owner
                assert new_updated_at == _SEEDED_UPDATED_AT
        assert _grants(seeded_db) == sorted(_GRANTS)

    def test_to_value_outside_the_default_principal_charset(
        self, seeded_db: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A subject the environment-variable route cannot hold is bound as given."""
        subject = 'auth0|abc'
        assert SAFE_PRINCIPAL_ID_PATTERN.fullmatch(subject) is None

        rc = cli_main(['--source-url', f'sqlite:///{seeded_db}', '--reassign-owner', 'local', subject])

        assert rc == 0
        capsys.readouterr()
        owners = {entry_id: owner for entry_id, (owner, _, _) in _entries(seeded_db).items()}
        assert owners == {
            'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa': subject,
            'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb': subject,
            'cccccccccccccccccccccccccccccccc': 'bob',
        }

    def test_no_matching_rows_is_a_success_no_op(
        self, seeded_db: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A FROM that owns nothing exits 0 with a message and changes nothing."""
        rc = cli_main(['--source-url', f'sqlite:///{seeded_db}', '--reassign-owner', 'mcp-client', 'local'])

        assert rc == 0
        assert "No context entries are owned by 'mcp-client'" in capsys.readouterr().err
        assert _entries(seeded_db) == _unchanged_entries()

    def test_database_without_schema_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A database lacking context_entries fails with exit 2."""
        db = tmp_path / 'empty.db'
        sqlite3.connect(str(db)).close()

        rc = cli_main(['--source-url', f'sqlite:///{db}', '--reassign-owner', 'local', 'alice'])

        assert rc == 2
        capsys.readouterr()
