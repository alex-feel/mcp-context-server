"""Tests for app/backends/postgresql_backend/provisioning.py: the pgvector provisioning gate and its probe."""

import unittest.mock
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import asyncpg
import pytest

from app.backends.postgresql_backend import PostgreSQLBackend
from tests.helpers import rebind_package_settings


class TestResolveProvisionVector:
    """The pgvector-provisioning gate keys on the ACTIVE payload format.

    The vector type is used ONLY by the fp32 vec_context_embeddings layout, so a
    compressed (BYTEA-only) database -- including a generation-off archive restored on a
    host without pgvector -- must NOT be forced to create the unused extension, while a
    compression-off database that will provision or already carries the fp32 layout must.
    """

    @staticmethod
    def _backend() -> PostgreSQLBackend:
        return PostgreSQLBackend(connection_string='postgresql://u:p@localhost:5432/db')

    @staticmethod
    def _settings(*, compression: bool, generation: bool) -> MagicMock:
        s = MagicMock()
        s.compression.enabled = compression
        s.embedding.generation_enabled = generation
        s.storage.postgresql_connect_timeout_s = 5.0
        return s

    @staticmethod
    def _conn(*, fp32: bool, embedding_metadata: bool) -> AsyncMock:
        async def _fetchval(sql: str) -> bool:
            if 'vec_context_embeddings' in sql:
                return fp32
            if 'embedding_metadata' in sql:
                return embedding_metadata
            raise AssertionError(f'unexpected probe SQL: {sql}')

        conn = AsyncMock()
        conn.fetchval = AsyncMock(side_effect=_fetchval)
        conn.close = AsyncMock()
        return conn

    @pytest.mark.asyncio
    async def test_generation_on_compression_off_provisions_without_probe(self) -> None:
        """generation on + compression off -> always provision the fp32 layout; no probe.

        Covers the compression-off server (the migration CLI's target init does not rely on
        this gate: it passes the explicit provision_vector constructor override keyed on
        with_semantic).
        """
        connect = AsyncMock()
        with unittest.mock.patch(
            'app.backends.postgresql_backend.provisioning.settings', self._settings(compression=False, generation=True),
        ), unittest.mock.patch('asyncpg.connect', new=connect):
            assert await self._backend()._resolve_provision_vector() is True
            connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_generation_on_compression_on_no_fp32_skips_pgvector(self) -> None:
        """generation on + compression on (the v3.0.0 default) + no fp32 table -> skip pgvector.

        The compressed server stores BYTEA and never binds the vector type, so forcing CREATE
        EXTENSION vector would crash boot on a pgvector-less host. The gate falls through to the
        fp32-table probe and skips when no stray fp32 table exists.
        """
        conn = self._conn(fp32=False, embedding_metadata=True)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.provisioning.settings', self._settings(compression=True, generation=True),
        ), unittest.mock.patch('asyncpg.connect', new=AsyncMock(return_value=conn)):
            assert await self._backend()._resolve_provision_vector() is False
        conn.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_generation_on_compression_on_fp32_present_provisions(self) -> None:
        """generation on + compression on + a stray fp32 table present -> provision the codec.

        A leftover fp32 table still needs the vector codec to read it, so the probe returns True
        even though the compressed write path itself binds no vector.
        """
        conn = self._conn(fp32=True, embedding_metadata=True)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.provisioning.settings', self._settings(compression=True, generation=True),
        ), unittest.mock.patch('asyncpg.connect', new=AsyncMock(return_value=conn)):
            assert await self._backend()._resolve_provision_vector() is True

    @pytest.mark.asyncio
    async def test_compressed_generation_off_no_fp32_skips_pgvector(self) -> None:
        """A compressed archive (embedding_metadata, no fp32) skips pgvector."""
        conn = self._conn(fp32=False, embedding_metadata=True)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.provisioning.settings', self._settings(compression=True, generation=False),
        ), unittest.mock.patch('asyncpg.connect', new=AsyncMock(return_value=conn)):
            assert await self._backend()._resolve_provision_vector() is False
        conn.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_fp32_table_present_provisions(self) -> None:
        """An fp32 vec table present (fp32 archive, or the --compress CLI reading it) -> provision."""
        conn = self._conn(fp32=True, embedding_metadata=True)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.provisioning.settings', self._settings(compression=True, generation=False),
        ), unittest.mock.patch('asyncpg.connect', new=AsyncMock(return_value=conn)):
            assert await self._backend()._resolve_provision_vector() is True

    @pytest.mark.asyncio
    async def test_compression_off_gen_off_infra_present_reprovisions(self) -> None:
        """compression off + generation off + embedding_metadata present -> fp32 reprovisioned."""
        conn = self._conn(fp32=False, embedding_metadata=True)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.provisioning.settings', self._settings(compression=False, generation=False),
        ), unittest.mock.patch('asyncpg.connect', new=AsyncMock(return_value=conn)):
            assert await self._backend()._resolve_provision_vector() is True

    @pytest.mark.asyncio
    async def test_fresh_generation_off_skips(self) -> None:
        """A fresh generation-off database (no fp32, no embedding_metadata) skips pgvector."""
        conn = self._conn(fp32=False, embedding_metadata=False)
        with unittest.mock.patch(
            'app.backends.postgresql_backend.provisioning.settings', self._settings(compression=False, generation=False),
        ), unittest.mock.patch('asyncpg.connect', new=AsyncMock(return_value=conn)):
            assert await self._backend()._resolve_provision_vector() is False

    @pytest.mark.asyncio
    async def test_probe_connection_failure_propagates(self) -> None:
        """A probe that never reached the database must not answer the question.

        Returning False here means "the fp32 layout is not needed", so the boot skips
        CREATE EXTENSION vector and the vector codec -- while a later
        `CREATE TABLE ... vector(dim)` still runs and fails unclassified. The fault is
        raised instead, and initialize()'s classification ladder turns it into the
        right exit code.
        """
        with unittest.mock.patch(
            'app.backends.postgresql_backend.provisioning.settings', self._settings(compression=True, generation=False),
        ), unittest.mock.patch(
            'asyncpg.connect', new=AsyncMock(side_effect=OSError('unreachable')),
        ), pytest.raises(OSError, match='unreachable'):
            await self._backend()._resolve_provision_vector()


class TestVectorProvisionProbeFaultScope:
    """The vector-provision probe answers a question; it never invents an answer.

    The probe decides whether the pgvector extension and vector codec must be
    provisioned. A connect fault means it never got to ask, so swallowing it and
    returning False would skip CREATE EXTENSION and the codec while a later
    ``CREATE TABLE ... vector(dim)`` still runs, failing unclassified. Letting the
    fault propagate puts it in front of initialize()'s classification ladder.
    """

    @pytest.mark.asyncio
    async def test_connect_fault_propagates_to_the_classifier(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A wrong password at probe time becomes exit 78, not a False answer."""
        import app.backends.postgresql_backend as pg_module
        from app.errors import ConfigurationError
        from app.settings import get_settings

        backend = PostgreSQLBackend(
            connection_string='postgresql://postgres:wrong@localhost:5432/testdb',
        )
        monkeypatch.setattr(
            'asyncpg.connect',
            AsyncMock(side_effect=asyncpg.exceptions.InvalidPasswordError(
                'password authentication failed',
            )),
        )
        # Force the probe path: it is skipped outright when generation is on and
        # compression is off. The settings models are frozen, so a copy is installed
        # on the module bindings the backend reads.
        get_settings.cache_clear()
        base = get_settings()
        rebind_package_settings(
            monkeypatch,
            pg_module,
            base.model_copy(
                update={'compression': base.compression.model_copy(update={'enabled': True})},
            ),
        )

        with pytest.raises(ConfigurationError, match='authentication failed'):
            await backend.initialize()
