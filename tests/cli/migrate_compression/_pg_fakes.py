"""Recording PostgreSQL fakes shared by the compression migration CLI tests.

The ``--compress`` / ``--decompress`` PostgreSQL paths borrow the server ``PostgreSQLBackend`` pool, whose
``command_timeout`` (~60s) asyncpg applies as the default client-side deadline. Their heaviest operations -- a full
HNSW index build, a whole-table DROP, the streamed batch reads -- can exceed that on a large corpus and be cancelled
client-side as a non-retryable ``asyncio.TimeoutError`` before the longer migration budget applies. The fakes record
every statement with its per-call timeout, so the tests can assert that each transaction raises its statement budget
via ``SET LOCAL statement_timeout`` and that every heavy statement carries an explicit deadline matching the
migration budget (so it is NOT capped at the pool ``command_timeout``).
"""

from typing import Any

from app.compression.types import CompressionMetadata
from app.settings import get_settings

# The client-side margin _pg_ddl adds on top of the server-side budget so an overrun
# is cancelled server-side (retryable) before the client gives up (non-retryable).
_CLIENT_MARGIN_S = 5.0


class RecordingConn:
    """An asyncpg-shaped connection that records statements and their timeouts."""

    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, float | None]] = []
        self.fetch_calls: list[tuple[str, float | None]] = []
        self.fetchval_calls: list[tuple[str, float | None]] = []

    async def execute(self, statement: str, *_args: Any, **kwargs: Any) -> str:
        self.execute_calls.append((statement, kwargs.get('timeout')))
        return 'OK'

    async def fetch(self, statement: str, *_args: Any, **kwargs: Any) -> list[Any]:
        self.fetch_calls.append((statement, kwargs.get('timeout')))
        # An empty batch makes the streaming loop exit after the first read.
        return []

    async def fetchval(self, statement: str, *_args: Any, **kwargs: Any) -> int:
        self.fetchval_calls.append((statement, kwargs.get('timeout')))
        # Idempotency probes: compress wants 0 (provenance absent -> proceed);
        # decompress wants truthy (compressed source table reachable via the
        # to_regclass search_path probe -> proceed). COUNT(*) recounts return 0
        # (empty, so the recount-vs-processed guard passes).
        return 1 if 'to_regclass' in statement else 0

    def transaction(self) -> 'ReadTxnCM':
        return ReadTxnCM()


class ReadTxnCM:
    """An asyncpg-shaped transaction context manager (for the pre-lock read count)."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class FakeTxn:
    def __init__(self, conn: RecordingConn) -> None:
        self.connection = conn
        self.backend_type = 'postgresql'


class TxnContext:
    def __init__(self, conn: RecordingConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> FakeTxn:
        return FakeTxn(self._conn)

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class FakeBackend:
    backend_type = 'postgresql'

    def __init__(self, conn: RecordingConn) -> None:
        self._conn = conn

    def begin_transaction(self) -> TxnContext:
        return TxnContext(self._conn)


def sample_provenance() -> CompressionMetadata:
    return CompressionMetadata(
        provider='turboquant',
        bits=4,
        variant='ip',
        seed=42,
        dim=128,
        codebook_fingerprint='ab' * 32,
    )


def budget_set_local() -> str:
    budget = get_settings().storage.postgresql_migration_timeout_s
    return f'SET LOCAL statement_timeout = {int(budget * 1000)}'


def expected_client_timeout() -> float:
    return get_settings().storage.postgresql_migration_timeout_s + _CLIENT_MARGIN_S


def timeouts_for(conn: RecordingConn, needle: str) -> list[float | None]:
    return [t for stmt, t in conn.execute_calls if needle in stmt]
