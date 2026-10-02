"""Tests for app/backends/postgresql_backend/pool_callbacks.py: the asyncpg pool callbacks.

Covers the typed-timeout connect wrapper, the setup callback that applies the
session GUCs on every acquire, the init callback's pgvector fault
classification, and the reset callback that runs on release.
"""

import unittest.mock
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.postgresql_backend.acquire_faults import ConnectionEstablishmentTimeoutError


class TestSetupPoolConnectionTimeout:
    """The pool setup callback re-raises its TimeoutError as a distinguishable fault.

    asyncpg re-raises setup-callback failures from pool.acquire() verbatim, where a
    bare TimeoutError (a dead pooled connection whose SET statement exceeded the
    pool command_timeout) is indistinguishable from the saturation TimeoutError of
    a full pool -- which the backend deliberately leaves uncharged on the circuit
    breaker. Re-raising as ConnectionDoesNotExistError routes the fault into the
    charged retryable connection-error arm instead.
    """

    @pytest.mark.asyncio
    async def test_setup_timeout_reraised_as_connection_does_not_exist(self) -> None:
        """A TimeoutError in the setup statement surfaces as ConnectionDoesNotExistError."""
        from app.backends.postgresql_backend.pool_callbacks import setup_pool_connection

        conn = MagicMock()
        conn.execute = AsyncMock(side_effect=TimeoutError('dead pooled connection'))

        with pytest.raises(asyncpg.exceptions.ConnectionDoesNotExistError, match='setup timed out'):
            await setup_pool_connection(cast(asyncpg.Connection, conn))

    @pytest.mark.asyncio
    async def test_setup_success_sets_statement_timeout(self) -> None:
        """The happy path issues exactly one SET statement_timeout statement."""
        from app.backends.postgresql_backend.pool_callbacks import setup_pool_connection

        conn = MagicMock()
        conn.execute = AsyncMock()

        await setup_pool_connection(cast(asyncpg.Connection, conn))

        conn.execute.assert_awaited_once()
        assert conn.execute.await_args is not None
        sql = conn.execute.await_args.args[0]
        assert sql.startswith('SET statement_timeout = ')

    @pytest.mark.asyncio
    async def test_setup_generic_failure_propagates_unwrapped(self) -> None:
        """A non-timeout setup failure propagates as-is (asyncpg closes the connection)."""
        from app.backends.postgresql_backend.pool_callbacks import setup_pool_connection

        conn = MagicMock()
        conn.execute = AsyncMock(side_effect=RuntimeError('protocol desync'))

        with pytest.raises(RuntimeError, match='protocol desync'):
            await setup_pool_connection(cast(asyncpg.Connection, conn))


class TestPoolConnectTypedTimeout:
    """The pool's connect callable types establishment timeouts distinguishably.

    asyncpg's pool re-raises connect failures from pool.acquire() verbatim,
    where a bare establishment TimeoutError is indistinguishable from the
    saturation TimeoutError of a full pool. The wrapper re-raises it as
    ConnectionEstablishmentTimeoutError (still a TimeoutError) so the
    acquire-phase arms can charge an unreachable database without inferring the
    fault class from elapsed wall time; every other failure propagates
    unchanged.
    """

    @pytest.mark.asyncio
    async def test_establishment_timeout_is_typed_with_cause(self) -> None:
        """A TimeoutError from asyncpg.connect surfaces typed, keeping its cause."""
        from app.backends.postgresql_backend.pool_callbacks import connect_pool_connection

        original = TimeoutError('connect timed out')
        with (
            unittest.mock.patch(
                'asyncpg.connect',
                new_callable=AsyncMock,
                side_effect=original,
            ),
            pytest.raises(ConnectionEstablishmentTimeoutError, match='timed out establishing') as exc_info,
        ):
            await connect_pool_connection('postgresql://localhost:5432/testdb')

        assert exc_info.value.__cause__ is original
        # Still a member of the broad timeout family for existing handlers.
        assert isinstance(exc_info.value, TimeoutError)

    @pytest.mark.asyncio
    async def test_non_timeout_failure_propagates_unchanged(self) -> None:
        """A refused connection propagates as-is, never as the typed timeout."""
        from app.backends.postgresql_backend.pool_callbacks import connect_pool_connection

        with (
            unittest.mock.patch(
                'asyncpg.connect',
                new_callable=AsyncMock,
                side_effect=ConnectionRefusedError('connection refused'),
            ),
            pytest.raises(ConnectionRefusedError),
        ):
            await connect_pool_connection('postgresql://localhost:5432/testdb')

    @pytest.mark.asyncio
    async def test_success_returns_connection_and_forwards_arguments(self) -> None:
        """The wrapper forwards the pool's arguments verbatim and returns the result."""
        from app.backends.postgresql_backend.pool_callbacks import connect_pool_connection

        fake_conn = MagicMock()
        connect_mock = AsyncMock(return_value=fake_conn)

        with unittest.mock.patch('asyncpg.connect', connect_mock):
            result = await connect_pool_connection(
                'postgresql://localhost:5432/testdb',
                timeout=12.5,
                statement_cache_size=0,
            )

        assert result is fake_conn
        connect_mock.assert_awaited_once_with(
            'postgresql://localhost:5432/testdb',
            timeout=12.5,
            statement_cache_size=0,
        )


class TestPgvectorInitCallbackClassification:
    """The pool's pgvector init callback must not relabel transient faults.

    asyncpg re-raises an ``init`` failure VERBATIM out of ``pool.acquire()``, so
    whatever type this callback raises IS the type the caller classifies. Its two
    awaits are real server round-trips, so it sees the whole transient transport
    family; turning one of those into ConfigurationError makes a self-clearing
    blip permanent -- no retry at runtime, and exit 78 (never retried by the
    supervisor) at boot, where the identical fault one line later is retried five
    times or exits 69.
    """

    @staticmethod
    async def _init_callback(monkeypatch: pytest.MonkeyPatch) -> 'Callable[[Any], Awaitable[None]]':
        """Boot a backend against a fake pool and return the pool's init callback.

        Args:
            monkeypatch: Fixture used to stub the boot steps around pool creation.

        Returns:
            The ``init`` callable asyncpg would run for every new connection.
        """
        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )
        monkeypatch.setattr(backend, '_resolve_provision_vector', AsyncMock(return_value=True))
        monkeypatch.setattr(backend, '_ensure_pgvector_extension', AsyncMock())
        monkeypatch.setattr(backend, '_verify_connectivity', AsyncMock())
        monkeypatch.setattr(backend, '_detect_pgpool_ii', AsyncMock())
        monkeypatch.setattr(backend, '_detect_session_mode_pooler', MagicMock())
        create_pool = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr('asyncpg.create_pool', create_pool)

        await backend.initialize()

        await_args = create_pool.await_args
        assert await_args is not None
        init_callback = await_args.kwargs['init']
        assert callable(init_callback)
        return cast('Callable[[Any], Awaitable[None]]', init_callback)

    @staticmethod
    def _connection_failing_the_extension_probe(error: BaseException) -> MagicMock:
        """Build a connection whose pg_extension probe raises.

        Args:
            error: The exception the probe raises.

        Returns:
            The configured connection mock.
        """
        conn = MagicMock()
        conn.fetchrow = AsyncMock(side_effect=error)
        conn.set_type_codec = AsyncMock()
        return conn

    @pytest.mark.asyncio
    async def test_lost_connection_stays_retryable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A backend that disappears mid-registration keeps its retryable type."""
        init_callback = await self._init_callback(monkeypatch)
        conn = self._connection_failing_the_extension_probe(
            asyncpg.exceptions.ConnectionDoesNotExistError('connection was closed'),
        )

        with pytest.raises(asyncpg.exceptions.ConnectionDoesNotExistError):
            await init_callback(conn)

    @pytest.mark.asyncio
    async def test_timeout_stays_retryable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A command timeout during registration keeps its retryable type."""
        init_callback = await self._init_callback(monkeypatch)
        conn = self._connection_failing_the_extension_probe(TimeoutError())

        with pytest.raises(TimeoutError):
            await init_callback(conn)

    @pytest.mark.asyncio
    async def test_socket_fault_stays_retryable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reset socket during registration keeps its retryable type."""
        init_callback = await self._init_callback(monkeypatch)
        conn = self._connection_failing_the_extension_probe(
            ConnectionResetError('connection reset by peer'),
        )

        with pytest.raises(ConnectionResetError):
            await init_callback(conn)

    @pytest.mark.asyncio
    async def test_permanent_fault_is_still_a_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A genuinely permanent registration fault still reaches exit 78."""
        from app.errors import ConfigurationError

        init_callback = await self._init_callback(monkeypatch)
        conn = self._connection_failing_the_extension_probe(RuntimeError('codec table corrupted'))

        with pytest.raises(ConfigurationError, match='RuntimeError: codec table corrupted'):
            await init_callback(conn)

    @pytest.mark.asyncio
    async def test_missing_extension_is_still_a_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A database without the pgvector extension still fails fast at boot."""
        from app.errors import ConfigurationError

        init_callback = await self._init_callback(monkeypatch)
        conn = MagicMock()
        conn.fetchrow = AsyncMock(return_value=None)
        conn.set_type_codec = AsyncMock()

        with pytest.raises(ConfigurationError, match='pgvector extension is not installed'):
            await init_callback(conn)


class TestResetPoolConnection:
    """The reset callback cleans a released connection before the pool reuses it."""

    @pytest.mark.asyncio
    async def test_rolls_back_then_validates_then_resets(self) -> None:
        """ROLLBACK runs first, then the SELECT 1 health check, then RESET ALL.

        ROLLBACK aborts work a cancelled request left open, so it must precede
        RESET ALL, which clears the session GUCs the setup callback re-applies on
        the next acquire.
        """
        from app.backends.postgresql_backend.pool_callbacks import reset_pool_connection

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.fetchval = AsyncMock(return_value=1)
        calls = MagicMock()
        calls.attach_mock(conn.execute, 'execute')
        calls.attach_mock(conn.fetchval, 'fetchval')

        await reset_pool_connection(conn)

        assert calls.mock_calls == [
            unittest.mock.call.execute('ROLLBACK'),
            unittest.mock.call.fetchval('SELECT 1'),
            unittest.mock.call.execute('RESET ALL'),
        ]

    @pytest.mark.asyncio
    async def test_failed_step_is_reraised(self) -> None:
        """A failing step propagates, so asyncpg terminates the connection instead of pooling it."""
        from app.backends.postgresql_backend.pool_callbacks import reset_pool_connection

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.fetchval = AsyncMock(side_effect=asyncpg.exceptions.ConnectionDoesNotExistError('connection was closed'))

        with pytest.raises(asyncpg.exceptions.ConnectionDoesNotExistError):
            await reset_pool_connection(conn)

        conn.execute.assert_awaited_once_with('ROLLBACK')
