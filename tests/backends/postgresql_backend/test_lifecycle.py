"""Tests for app/backends/postgresql_backend/lifecycle.py: initialize().

Covers the error classification of pool creation and pgvector provisioning,
the pool wiring of the acquire and establishment timeouts, and the
provision_vector constructor override.
"""

import unittest.mock
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.backends.postgresql_backend import PostgreSQLBackend


class TestInitializeErrorClassification:
    """Test error classification in PostgreSQLBackend.initialize().

    When initialize() fails, errors must be classified as either
    DependencyError (exit code 69, retryable) or ConfigurationError
    (exit code 78, non-retryable) to enable proper Docker/Kubernetes
    restart policy behavior.

    Each case targets the fault raised by POOL CREATION, so the vector-provision
    probe that runs before it is stubbed out. The probe opens a real connection of
    its own and, because it propagates a connect fault rather than swallowing it (a
    probe must not answer a question it never got to ask), an unstubbed probe would
    fail first with its own error and mask the fault under test.
    """

    @pytest.fixture(autouse=True)
    def _skip_vector_provision_probe(self) -> Iterator[None]:
        """Stub the boot-time vector-provision probe for every case in this class."""
        with unittest.mock.patch.object(
            PostgreSQLBackend, '_resolve_provision_vector', AsyncMock(return_value=False),
        ):
            yield

    @pytest.mark.asyncio
    async def test_connection_refused_raises_dependency_error(self) -> None:
        """ConnectionRefusedError during pool creation raises DependencyError."""

        from app.errors import DependencyError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=ConnectionRefusedError('Connection refused'),
            ),
            pytest.raises(DependencyError, match='PostgreSQL connection failed'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_os_error_raises_dependency_error(self) -> None:
        """OSError (network unreachable, timeout) during pool creation raises DependencyError."""

        from app.errors import DependencyError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=OSError('Network is unreachable'),
            ),
            pytest.raises(DependencyError, match='PostgreSQL connection failed'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_too_many_connections_raises_dependency_error(self) -> None:
        """TooManyConnectionsError during pool creation raises DependencyError."""
        from app.errors import DependencyError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=asyncpg.exceptions.TooManyConnectionsError('too many connections'),
            ),
            pytest.raises(DependencyError, match='PostgreSQL connection failed'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_invalid_password_raises_configuration_error(self) -> None:
        """InvalidPasswordError during pool creation raises ConfigurationError."""
        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:wrong@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=asyncpg.exceptions.InvalidPasswordError('password authentication failed'),
            ),
            pytest.raises(ConfigurationError, match='PostgreSQL authentication failed'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_invalid_catalog_name_raises_configuration_error(self) -> None:
        """InvalidCatalogNameError during pool creation raises ConfigurationError."""
        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/nonexistent',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=asyncpg.exceptions.InvalidCatalogNameError('database "nonexistent" does not exist'),
            ),
            pytest.raises(ConfigurationError, match='PostgreSQL database does not exist'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_value_error_raises_configuration_error(self) -> None:
        """ValueError during pool creation raises ConfigurationError.

        asyncpg raises plain ValueError synchronously for invalid construction
        inputs (pool size combinations, a non-positive command_timeout) before
        any network I/O; these are permanent misconfigurations that must exit
        78 instead of restart-looping as a retryable dependency failure. DSN
        option errors are NOT this shape: asyncpg raises those as
        ClientConfigurationError, whose InterfaceError base would shadow the
        ValueError clause, so the backend classifies them in a dedicated
        earlier clause covered by
        test_client_configuration_error_raises_configuration_error.
        """
        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=ValueError('min_size is greater than max_size'),
            ),
            pytest.raises(ConfigurationError, match='PostgreSQL configuration invalid'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_client_configuration_error_raises_configuration_error(self) -> None:
        """ClientConfigurationError classifies as ConfigurationError, not DependencyError.

        asyncpg raises ClientConfigurationError for permanent client-side
        misconfigurations (invalid sslmode/target_session_attrs/gsslib values,
        unresolvable DSN options). The class subclasses BOTH InterfaceError and
        ValueError, so a broad InterfaceError tuple listed first would shadow
        it into a retryable DependencyError (exit 69) and restart-loop the
        supervisor on a permanent misconfiguration; the backend must classify
        it as ConfigurationError (exit 78) in a clause preceding the
        InterfaceError tuple.
        """
        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb?sslmode=bogus',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=asyncpg.exceptions.ClientConfigurationError(
                    "sslmode is invalid, valid values are: 'disable', 'prefer', 'require'",
                ),
            ),
            pytest.raises(ConfigurationError, match='PostgreSQL client configuration invalid'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_client_configuration_error_in_pgvector_precheck(self) -> None:
        """The pgvector pre-check classifies ClientConfigurationError the same way.

        _ensure_pgvector_extension opens its own connection BEFORE pool
        creation and carries the same InterfaceError tuple, so it needs the
        same preceding ClientConfigurationError clause.
        """
        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb?sslmode=bogus',
        )

        with (
            unittest.mock.patch(
                'asyncpg.connect',
                side_effect=asyncpg.exceptions.ClientConfigurationError(
                    "sslmode is invalid, valid values are: 'disable', 'prefer', 'require'",
                ),
            ),
            pytest.raises(ConfigurationError, match='PostgreSQL client configuration invalid'),
        ):
            await backend._ensure_pgvector_extension()

    @pytest.mark.asyncio
    async def test_value_error_in_pgvector_precheck_raises_configuration_error(self) -> None:
        """A malformed-DSN ValueError in the pgvector pre-check classifies as exit 78.

        asyncpg raises a plain builtins.ValueError (NOT ClientConfigurationError)
        for a malformed DSN authority -- e.g. a double-bracketed host literal --
        before any network I/O. _ensure_pgvector_extension runs BEFORE pool
        creation on the compression-off generation-on path, so without a
        dedicated ValueError clause its catch-all would wrap this permanent
        client-side misconfiguration as a retryable DependencyError (exit 69)
        and supervisors would restart-loop on it; initialize()'s own ValueError
        clause would not run because its 'except DependencyError' re-raise wins
        first.
        """
        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@[[::1]]:5432/testdb',
        )

        with (
            unittest.mock.patch(
                'asyncpg.connect',
                side_effect=ValueError("invalid IPv6 address: '[::1'"),
            ),
            pytest.raises(ConfigurationError, match='PostgreSQL configuration invalid'),
        ):
            await backend._ensure_pgvector_extension()

    @pytest.mark.asyncio
    async def test_acquire_timeout_and_connect_timeout_wiring(self) -> None:
        """The acquire-wait and establishment timeouts reach their real asyncpg knobs.

        asyncpg.create_pool has NO acquire-timeout parameter -- an unknown
        'timeout' kwarg falls through connect_kwargs to asyncpg.connect() as
        the connection ESTABLISHMENT timeout. The documented acquire-wait
        bound (POSTGRESQL_POOL_TIMEOUT_S) therefore must be passed per-call
        at pool.acquire(timeout=...); wiring it into create_pool instead
        silently leaves every acquire waiting unbounded under pool
        exhaustion.
        """
        from app.settings import get_settings

        captured: dict[str, object] = {}

        class _FakeAcquireContext:
            async def __aenter__(self) -> AsyncMock:
                return AsyncMock()

            async def __aexit__(self, *exc_info: object) -> bool:
                return False

        fake_pool = unittest.mock.MagicMock()

        def _acquire(*, timeout: float | None = None) -> _FakeAcquireContext:
            captured['acquire_timeout'] = timeout
            return _FakeAcquireContext()

        fake_pool.acquire = _acquire

        create_kwargs: dict[str, object] = {}

        async def _fake_create_pool(dsn: str, **kwargs: object) -> unittest.mock.MagicMock:
            _ = dsn
            create_kwargs.update(kwargs)
            return fake_pool

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch('asyncpg.create_pool', side_effect=_fake_create_pool),
        ):
            await backend.initialize()

        settings = get_settings()
        # create_pool's 'timeout' is the ESTABLISHMENT timeout, sourced from
        # the dedicated connect knob -- never from the acquire knob.
        assert create_kwargs['timeout'] == settings.storage.postgresql_connect_timeout_s
        # New connections must dial through the typed-connect wrapper so an
        # establishment timeout surfaces as ConnectionEstablishmentTimeoutError,
        # distinguishable from the bare TimeoutError of the acquire deadline.
        from app.backends.postgresql_backend.pool_callbacks import connect_pool_connection

        assert create_kwargs['connect'] is connect_pool_connection
        # The Pgpool-II detection probe runs during initialize() and must have
        # acquired with the acquire-wait bound.
        assert captured['acquire_timeout'] == settings.storage.postgresql_pool_timeout_s

    @pytest.mark.asyncio
    async def test_unknown_exception_raises_dependency_error(self) -> None:
        """Unknown exceptions during pool creation default to DependencyError."""

        from app.errors import DependencyError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=RuntimeError('unexpected internal error'),
            ),
            pytest.raises(DependencyError, match='PostgreSQL initialization failed'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_configuration_error_from_init_connection_reraised(self) -> None:
        """ConfigurationError from init_pool_connection is re-raised without wrapping."""

        from app.errors import ConfigurationError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(backend, '_ensure_pgvector_extension', new_callable=AsyncMock),
            unittest.mock.patch(
                'asyncpg.create_pool',
                side_effect=ConfigurationError('pgvector codec registration failed'),
            ),
            pytest.raises(ConfigurationError, match='pgvector codec registration failed'),
        ):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_dependency_error_from_ensure_pgvector_reraised(self) -> None:
        """DependencyError from _ensure_pgvector_extension is re-raised without wrapping."""
        from app.errors import DependencyError

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:postgres@localhost:5432/testdb',
        )

        with (
            unittest.mock.patch.object(
                backend,
                '_ensure_pgvector_extension',
                new_callable=AsyncMock,
                side_effect=DependencyError('PostgreSQL connection failed: Connection refused'),
            ),
            pytest.raises(DependencyError, match='PostgreSQL connection failed'),
        ):
            await backend.initialize()


class TestProvisionVectorOverride:
    """The explicit provision_vector constructor override bypasses the boot gate.

    The migration CLI knows up front whether its target carries the fp32 vector
    layout (with_semantic), so it must not depend on the CLI process's env-driven
    settings gate: a vector-free target initialized under a compression-off env
    would otherwise force CREATE EXTENSION vector and crash on a pgvector-less host.
    """

    @staticmethod
    def _initialize_ready_backend(
        monkeypatch: pytest.MonkeyPatch,
        provision_vector: bool | None,
    ) -> tuple[PostgreSQLBackend, AsyncMock, AsyncMock]:
        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@localhost:5432/db',
            provision_vector=provision_vector,
        )
        resolve = AsyncMock(return_value=True)
        ensure = AsyncMock()
        monkeypatch.setattr(backend, '_resolve_provision_vector', resolve)
        monkeypatch.setattr(backend, '_ensure_pgvector_extension', ensure)
        monkeypatch.setattr(backend, '_detect_pgpool_ii', AsyncMock())
        monkeypatch.setattr(backend, '_detect_session_mode_pooler', MagicMock())
        return backend, resolve, ensure

    @pytest.mark.asyncio
    async def test_explicit_false_skips_resolution_and_extension(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """provision_vector=False initializes without probing or touching pgvector."""
        backend, resolve, ensure = self._initialize_ready_backend(monkeypatch, provision_vector=False)
        with unittest.mock.patch(
            'asyncpg.create_pool',
            new=AsyncMock(return_value=MagicMock()),
        ):
            await backend.initialize()
        assert backend._provision_vector is False
        resolve.assert_not_called()
        ensure.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_true_provisions_without_probe(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """provision_vector=True pre-creates the extension without the settings gate."""
        backend, resolve, ensure = self._initialize_ready_backend(monkeypatch, provision_vector=True)
        with unittest.mock.patch(
            'asyncpg.create_pool',
            new=AsyncMock(return_value=MagicMock()),
        ):
            await backend.initialize()
        assert backend._provision_vector is True
        resolve.assert_not_called()
        ensure.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_default_none_resolves_via_gate(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without the override, initialize() resolves via _resolve_provision_vector."""
        backend, resolve, ensure = self._initialize_ready_backend(monkeypatch, provision_vector=None)
        with unittest.mock.patch(
            'asyncpg.create_pool',
            new=AsyncMock(return_value=MagicMock()),
        ):
            await backend.initialize()
        assert backend._provision_vector is True
        resolve.assert_awaited_once()
        ensure.assert_awaited_once()
