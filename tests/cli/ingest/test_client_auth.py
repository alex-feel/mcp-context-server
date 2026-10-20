"""Tests for the ingest client's bearer-token auth: the token file, the named variable, the 401 retry and secrets."""

import errno
import logging
import os
from pathlib import Path

import httpx2
import pytest
from fastmcp.client.auth import BearerAuth

from app.cli.ingest.client import EXIT_CONFIG
from app.cli.ingest.client import IngestExitError
from app.cli.ingest.client import build_auth
from app.cli.ingest.report import RunReport
from app.cli.ingest.report import render_summary
from app.cli.ingest.report import write_json
from tests.cli.ingest._stand_in import PLAN_ARGUMENTS
from tests.cli.ingest._stand_in import PLAN_RESULT
from tests.cli.ingest._stand_in import PLAN_TOOL
from tests.cli.ingest._stand_in import TOKEN_A
from tests.cli.ingest._stand_in import TOKEN_B
from tests.cli.ingest._stand_in import TOKEN_VARIABLE
from tests.cli.ingest._stand_in import Exchange
from tests.cli.ingest._stand_in import context_server
from tests.cli.ingest._stand_in import exit_of_plan_call
from tests.cli.ingest._stand_in import file_auth
from tests.cli.ingest._stand_in import ingest_client
from tests.cli.ingest._stand_in import rotating
from tests.cli.ingest._stand_in import serving
from tests.cli.ingest._stand_in import status
from tests.cli.ingest._stand_in import write_token_file


class TestTokenFile:
    """A token file is read on every request, so a refresher can rotate it during a run."""

    @pytest.mark.asyncio
    async def test_file_is_read_on_every_request(self, exchange: Exchange, tmp_path: Path) -> None:
        """A token written between two calls is the one the second call sends."""
        path = write_token_file(tmp_path, TOKEN_A)

        async with ingest_client(exchange, auth=file_auth(path)) as client:
            await client.call_tool(PLAN_TOOL, PLAN_ARGUMENTS)
            path.write_text(TOKEN_B + '\n', encoding='utf-8')
            await client.call_tool(PLAN_TOOL, PLAN_ARGUMENTS)

        assert [seen.authorization for seen in exchange.calls_of(PLAN_TOOL)] == [f'Bearer {TOKEN_A}', f'Bearer {TOKEN_B}']
        assert all(seen.authorization is not None for seen in exchange.seen)

    def test_empty_token_file_is_refused(self, tmp_path: Path) -> None:
        """A file holding only whitespace is no token."""
        path = write_token_file(tmp_path, '  ')

        with pytest.raises(IngestExitError) as raised:
            build_auth(token_file=path, token_env=None)

        assert (raised.value.code, raised.value.message) == (EXIT_CONFIG, f'the token file is empty: {path}')

    def test_missing_token_file_is_refused(self, tmp_path: Path) -> None:
        """A token file that cannot be read names the file and the reason."""
        path = tmp_path / 'absent'

        with pytest.raises(IngestExitError) as raised:
            build_auth(token_file=path, token_env=None)

        assert (raised.value.code, raised.value.message) == (
            EXIT_CONFIG,
            f'the token file cannot be read: {path}: {os.strerror(errno.ENOENT)}',
        )

    @pytest.mark.asyncio
    async def test_token_file_emptied_during_the_run(self, exchange: Exchange, tmp_path: Path) -> None:
        """A request that finds the file empty ends the run with the same refusal."""
        path = write_token_file(tmp_path, TOKEN_A)

        async with ingest_client(exchange, auth=file_auth(path)) as client:
            path.write_text('', encoding='utf-8')
            ended = await exit_of_plan_call(client)

        assert (ended.code, ended.message) == (EXIT_CONFIG, f'the token file is empty: {path}')
        assert exchange.calls_of(PLAN_TOOL) == []


class TestTokenEnvironmentVariable:
    """A token named by --token-env is read once from the process environment."""

    @pytest.mark.asyncio
    async def test_token_from_a_named_variable(self, exchange: Exchange, monkeypatch: pytest.MonkeyPatch) -> None:
        """The named variable's value is sent as the bearer token."""
        monkeypatch.setenv(TOKEN_VARIABLE, TOKEN_A)

        auth = build_auth(token_file=None, token_env=TOKEN_VARIABLE)
        async with ingest_client(exchange, auth=auth) as client:
            await client.call_tool(PLAN_TOOL, PLAN_ARGUMENTS)

        assert isinstance(auth, BearerAuth)
        assert [seen.authorization for seen in exchange.calls_of(PLAN_TOOL)] == [f'Bearer {TOKEN_A}']

    def test_unset_variable_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A variable that is not set is named in the refusal."""
        monkeypatch.delenv(TOKEN_VARIABLE, raising=False)

        with pytest.raises(IngestExitError) as raised:
            build_auth(token_file=None, token_env=TOKEN_VARIABLE)

        assert (raised.value.code, raised.value.message) == (
            EXIT_CONFIG,
            f'the environment variable {TOKEN_VARIABLE} named by --token-env is not set',
        )

    def test_empty_variable_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A variable holding only whitespace is no token."""
        monkeypatch.setenv(TOKEN_VARIABLE, ' ')

        with pytest.raises(IngestExitError) as raised:
            build_auth(token_file=None, token_env=TOKEN_VARIABLE)

        assert (raised.value.code, raised.value.message) == (
            EXIT_CONFIG,
            f'the environment variable {TOKEN_VARIABLE} named by --token-env is empty',
        )

    def test_no_token_source_means_no_auth(self) -> None:
        """Without either flag the client sends no Authorization header."""
        assert build_auth(token_file=None, token_env=None) is None

    def test_both_token_sources_are_refused(self, tmp_path: Path) -> None:
        """The two token sources are exclusive."""
        with pytest.raises(IngestExitError) as raised:
            build_auth(token_file=write_token_file(tmp_path, TOKEN_A), token_env=TOKEN_VARIABLE)

        assert (raised.value.code, raised.value.message) == (
            EXIT_CONFIG,
            '--token-file and --token-env cannot be combined',
        )


class TestUnauthorizedRetry:
    """A 401 after the server accepted the token is retried once, and only when the token file changed."""

    @pytest.mark.asyncio
    async def test_retried_once_after_the_file_was_rotated(self, exchange: Exchange, tmp_path: Path) -> None:
        """The retry sends the new token and the call succeeds."""
        path = write_token_file(tmp_path, TOKEN_A)
        exchange.script(PLAN_TOOL, rotating(path, TOKEN_B))

        async with ingest_client(exchange, auth=file_auth(path)) as client:
            result = await client.call_tool(PLAN_TOOL, PLAN_ARGUMENTS)

        assert result == PLAN_RESULT
        assert [seen.authorization for seen in exchange.calls_of(PLAN_TOOL)] == [f'Bearer {TOKEN_A}', f'Bearer {TOKEN_B}']

    @pytest.mark.asyncio
    async def test_not_retried_when_the_token_is_unchanged(self, exchange: Exchange, tmp_path: Path) -> None:
        """A refused token that the file still holds has expired: exit 2 without a retry."""
        path = write_token_file(tmp_path, TOKEN_A)
        exchange.script(PLAN_TOOL, status(401))

        async with ingest_client(exchange, auth=file_auth(path)) as client:
            ended = await exit_of_plan_call(client)

        assert (ended.code, ended.message) == (
            EXIT_CONFIG,
            (
                'the server refused the bearer token (HTTP 401) after accepting it earlier in this run: '
                f'the token has expired; write a fresh token to {path} and rerun'
            ),
        )
        assert len(exchange.calls_of(PLAN_TOOL)) == 1

    @pytest.mark.asyncio
    async def test_retried_only_once(self, exchange: Exchange, tmp_path: Path) -> None:
        """A second refusal after the retry ends the run naming expiry."""
        path = write_token_file(tmp_path, TOKEN_A)
        exchange.script(PLAN_TOOL, rotating(path, TOKEN_B), rotating(path, TOKEN_A))

        async with ingest_client(exchange, auth=file_auth(path)) as client:
            ended = await exit_of_plan_call(client)

        assert ended.code == EXIT_CONFIG
        assert 'the token has expired' in ended.message
        assert len(exchange.calls_of(PLAN_TOOL)) == 2

    @pytest.mark.asyncio
    async def test_a_token_refused_before_any_success_is_wrong(self, exchange: Exchange, tmp_path: Path) -> None:
        """A 401 to the very first request names a wrong token and is not retried, even after a rotation."""
        unrotated_directory = tmp_path / 'unrotated'
        unrotated_directory.mkdir()
        unrotated = write_token_file(unrotated_directory, TOKEN_A)
        async with serving(context_server()) as baseline:
            baseline.every_request = status(401)
            with pytest.raises(IngestExitError):
                async with ingest_client(baseline, auth=file_auth(unrotated)):
                    pass
        path = write_token_file(tmp_path, TOKEN_A)
        refusals = 0

        async def refuse_and_rotate(request: httpx2.Request) -> httpx2.Response:
            nonlocal refusals
            refusals += 1
            return await rotating(path, f'{TOKEN_B}-{refusals}')(request)

        exchange.every_request = refuse_and_rotate

        with pytest.raises(IngestExitError) as raised:
            async with ingest_client(exchange, auth=file_auth(path)):
                pass

        assert (raised.value.code, raised.value.message) == (
            EXIT_CONFIG,
            'the server refused the bearer token (HTTP 401): the token is wrong or not valid for this server',
        )
        # The negotiation sends exactly the requests it sends when the token never changes.
        assert len(exchange.seen) == len(baseline.seen)
        assert exchange.calls_of(PLAN_TOOL) == []

    @pytest.mark.asyncio
    async def test_a_token_from_the_environment_is_never_retried(
        self, exchange: Exchange, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A process environment does not rotate, so its expired token ends the run at once."""
        monkeypatch.setenv(TOKEN_VARIABLE, TOKEN_A)
        exchange.script(PLAN_TOOL, status(401))

        async with ingest_client(exchange, auth=build_auth(token_file=None, token_env=TOKEN_VARIABLE)) as client:
            ended = await exit_of_plan_call(client)

        assert (ended.code, ended.message) == (
            EXIT_CONFIG,
            (
                'the server refused the bearer token (HTTP 401) after accepting it earlier in this run: '
                'the token has expired; rerun with a fresh token'
            ),
        )
        assert len(exchange.calls_of(PLAN_TOOL)) == 1

    @pytest.mark.asyncio
    async def test_403_is_not_retried(self, exchange: Exchange, tmp_path: Path) -> None:
        """Only a 401 is retried; a 403 after a rotation still ends the run."""
        path = write_token_file(tmp_path, TOKEN_A)
        exchange.script(PLAN_TOOL, rotating(path, TOKEN_B, code=403))

        async with ingest_client(exchange, auth=file_auth(path)) as client:
            ended = await exit_of_plan_call(client)

        assert ended.code == EXIT_CONFIG
        assert ended.message.startswith('the server refused the bearer token (HTTP 403) after accepting it earlier')
        assert len(exchange.calls_of(PLAN_TOOL)) == 1


class TestSecrets:
    """No token and no URL userinfo reach a message, a log record, the report file or the summary."""

    @pytest.mark.asyncio
    async def test_secrets_stay_out_of_every_output(
        self,
        exchange: Exchange,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A run that connects, calls, is refused and reports leaks neither the token nor the URL's userinfo."""
        caplog.set_level(logging.DEBUG)
        path = write_token_file(tmp_path, TOKEN_A)
        server = 'http://alice:s3cret-pass@127.0.0.1:8765/mcp'
        auth = file_auth(path)
        report = RunReport(server=server, thread_id='kb:handbook', dry_run=False)
        async with ingest_client(exchange, auth=auth, server=server) as client:
            await client.preflight()
            await client.call_tool(PLAN_TOOL, PLAN_ARGUMENTS)
            exchange.script(PLAN_TOOL, status(404))
            ended = await exit_of_plan_call(client)
        report.fail(ended.message, exit_code=ended.code)
        report_path = tmp_path / 'report.json'
        write_json(report, report_path)
        print(render_summary(report))

        outputs = {
            'message': ended.message,
            'logs': caplog.text,
            'report': report_path.read_text(encoding='utf-8'),
            'summary': capsys.readouterr().out,
            'auth': repr(auth),
        }
        assert 'http://127.0.0.1:8765/mcp' in ended.message
        for name, text in outputs.items():
            assert TOKEN_A not in text, name
            if name != 'logs':
                assert 's3cret-pass' not in text, name
                assert 'alice' not in text, name
