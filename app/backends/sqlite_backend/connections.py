"""Connection creation, tracking and teardown of the SQLite backend.

The finalizer-safe connection class, the shared writer and per-use reader connections with
their PRAGMAs and sqlite-vec load, the breaker charge for connection-establishment faults,
and closing every tracked connection.
"""

import asyncio
import contextlib
import logging
import sqlite3
import time
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import suppress
from typing import Any
from typing import cast
from typing import override
from urllib.parse import quote

from app.backends.sqlite_backend.contention import is_sqlite_locked_error
from app.backends.sqlite_backend.core import SQLiteBackendCore
from app.errors import ControlFlowError
from app.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# A connection subclass that guarantees best-effort finalization.
# If a connection object reaches GC while still open, this class
# will close it in __del__, which prevents ResourceWarning.
class ManagedConnection(sqlite3.Connection):
    _closed: bool = False

    @override
    def close(self) -> None:
        # Idempotent close, safe for __del__
        if self._closed:
            return
        try:
            super().close()
        finally:
            self._closed = True

    def __del__(self) -> None:
        # Never raise from __del__
        with contextlib.suppress(Exception):
            self.close()


class SQLiteConnectionsMixin(SQLiteBackendCore):
    """Create, track and close the shared writer and the per-use reader connections."""

    def _safe_close_connection(self, conn: sqlite3.Connection) -> None:
        """Close and untrack a connection, idempotent and exception-safe.

        Always use this helper instead of calling conn.close() directly.
        """
        try:
            if conn is self._writer_conn:
                with contextlib.suppress(Exception):
                    conn.execute('PRAGMA optimize')
            conn.close()
        except Exception:
            pass
        finally:
            with self._connection_lock:
                self._all_connections.discard(conn)
                self._temporary_connections.discard(conn)
                self._connection_ids.pop(id(conn), None)
            self.metrics.active_connections = max(0, self.metrics.active_connections - 1)

    def _load_sqlite_vec_extension(self, conn: sqlite3.Connection) -> None:
        """Load the sqlite-vec extension on a connection whenever it is installed.

        Args:
            conn: SQLite connection

        Note:
            This method is safe to call even if sqlite_vec is not installed; it
            gracefully skips loading when the package is unavailable. The load is
            NOT gated on embedding generation: the fp32 vec0 virtual table can
            physically persist from an earlier session that had generation
            enabled, and the delete/update stale-embedding cleanup paths gate on
            the durable ``embedding_tables_exist()`` table-presence signal rather
            than the runtime generation toggle. The vec0 module must therefore be
            available on every connection whenever that table could exist; if it
            is absent, any access to the table raises ``no such module: vec0`` --
            silently orphaning the FK-less vec0 rows on delete and rolling back
            text updates. Loading is ImportError-guarded and idempotent, so
            attempting it unconditionally is harmless when no vec table exists or
            when compression replaced it with the BLOB layout.
        """
        # Check if already loaded to avoid duplicate loading
        if hasattr(conn, '_vec_loaded') and getattr(conn, '_vec_loaded', False):
            return

        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            try:
                # sqlite_vec.load() is a plain conn.load_extension() dlopen: it
                # raises sqlite3.OperationalError whenever the bundled shared
                # object cannot be loaded (a noexec mount, a musl base image, a
                # distroless image missing its runtime deps). The finally pairing
                # is what guarantees the capability is withdrawn again on that
                # path -- otherwise SQL-level load_extension() stays enabled on
                # the writer and on every reader connection for the process
                # lifetime, silently removing the hardening layer that turns any
                # future SQL-injection defect into 'not authorized' instead of
                # arbitrary native code execution.
                cast(Any, sqlite_vec).load(conn)
                cast(Any, conn)._vec_loaded = True
            finally:
                with suppress(Exception):
                    conn.enable_load_extension(False)
            logger.debug('sqlite-vec extension loaded successfully')
        except ImportError:
            logger.debug('sqlite-vec package not installed, skipping extension loading')
        except Exception as e:
            logger.warning(f'Failed to load sqlite-vec extension: {e}')

    def _create_connection(self, readonly: bool = False) -> sqlite3.Connection:
        """Create a new SQLite connection with optimal settings."""
        # Do not create new connections during shutdown
        if self._shutdown:
            logger.error(f'Attempted to create connection during shutdown! readonly={readonly}')
            raise RuntimeError('Cannot create new connections during shutdown')

        # Create database file if it does not exist, for write connections
        if not readonly and not self.db_path.exists():
            # Create an initial connection to create the database file
            # Use with statement to ensure closure
            with sqlite3.connect(str(self.db_path)) as init_conn:
                # Set UTF-8 encoding BEFORE any data operations
                init_conn.execute('PRAGMA encoding = "UTF-8"')
                init_conn.commit()

        # Use URI mode for better control. SQLite percent-decodes the URI
        # path before use, so the filesystem path must be percent-encoded:
        # a raw path containing '%', '?', or '#' would otherwise be misread
        # as an escape sequence, query string, or fragment. A POSIX path
        # with a leading double slash is collapsed first -- 'file://tmp/db'
        # would parse 'tmp' as the URI authority and be rejected.
        path_str = str(self.db_path)
        if path_str.startswith('//'):
            path_str = '/' + path_str.lstrip('/')
        uri = f"file:{quote(path_str, safe='/:')}?mode={'ro' if readonly else 'rw'}"
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=self.pool_config.connection_timeout,
            check_same_thread=False,  # Thread safety handled at a higher level
            isolation_level='DEFERRED',  # Better for concurrent access
            factory=ManagedConnection,  # Ensure finalizer-based safety on GC
        )

        # ENSURE proper UTF-8 handling by verifying encoding (read-only check)
        if not readonly:
            # For write connections, ensure the database is using UTF-8
            cursor = conn.execute('PRAGMA encoding')
            encoding = cursor.fetchone()[0]
            if encoding != 'UTF-8':
                logger.warning(f'Database encoding is {encoding}, expected UTF-8. This may cause issues with non-ASCII text.')

        try:
            conn.row_factory = sqlite3.Row

            # Apply optimized PRAGMAs for production.
            #
            # DATABASE-FILE properties, applied on WRITABLE connections ONLY. A
            # mode=ro connection cannot establish them, so issuing them there can
            # only be a no-op or an error: when the on-disk journal mode differs
            # from SQLITE_JOURNAL_MODE -- another process (a second server
            # instance, the migration CLI, a bare sqlite3 shell) moved the shared
            # file into WAL, or the setting was changed -- SQLite answers
            # `PRAGMA journal_mode` on a read-only connection with SQLITE_IOERR
            # ('disk I/O error') or SQLITE_READONLY ('attempt to write a readonly
            # database'). Neither is in the self-clearing BUSY/LOCKED family the
            # creation-fault wrapper exempts, so EVERY reader creation would
            # charge the process-global circuit breaker until it opened and
            # rejected healthy writes too. The journal mode is established by the
            # writer at initialize(); a reader re-asserting it can never help.
            #
            # page_size MUST precede journal_mode. Switching to WAL (like any write)
            # finalizes the database header's page size; a later `PRAGMA page_size` is
            # then silently ignored on the existing file (it would need a VACUUM).
            # Applying it first lets a FRESH database honor SQLITE_PAGE_SIZE; on an
            # already-initialized database the pragma is a harmless no-op.
            file_pragmas: list[tuple[str, str]] = []
            if not readonly:
                file_pragmas = [
                    ('page_size', str(settings.storage.sqlite_page_size)),
                    ('journal_mode', settings.storage.sqlite_journal_mode),
                ]

            # Connection-scoped properties: safe and meaningful on readers and
            # writers alike, since each only configures the open handle.
            connection_pragmas: list[tuple[str, str]] = [
                ('foreign_keys', 'ON' if settings.storage.sqlite_foreign_keys else 'OFF'),
                ('synchronous', settings.storage.sqlite_synchronous),
                ('temp_store', settings.storage.sqlite_temp_store),
                ('mmap_size', str(settings.storage.sqlite_mmap_size)),
                ('cache_size', str(settings.storage.sqlite_cache_size)),
                ('wal_autocheckpoint', str(settings.storage.sqlite_wal_autocheckpoint)),
                ('busy_timeout', str(settings.storage.resolved_busy_timeout_ms)),
            ]

            # Writer-specific optimizations
            writer_pragmas: list[tuple[str, str]] = []
            if not readonly:
                writer_pragmas = [('wal_checkpoint', settings.storage.sqlite_wal_checkpoint)]

            for pragma, value in (*file_pragmas, *connection_pragmas, *writer_pragmas):
                conn.execute(f'PRAGMA {pragma} = {value}')

            # Load the sqlite-vec extension whenever the package is installed
            self._load_sqlite_vec_extension(conn)

            self.metrics.total_connections += 1
            self.metrics.active_connections += 1

            # Track all created connections for cleanup
            with self._connection_lock:
                self._all_connections.add(conn)
                self._connection_ids[id(conn)] = f'readonly={readonly}'

            logger.debug(f'Created connection: {id(conn)} (readonly={readonly}), total: {len(self._all_connections)}')

            return conn
        except Exception:
            # Close connection on any error during setup
            with contextlib.suppress(Exception):
                conn.close()
            with self._connection_lock:
                self._all_connections.discard(conn)
            raise

    async def _ensure_writer_connection(self) -> sqlite3.Connection:
        """Ensure writer connection exists and is healthy.

        Returns:
            The shared writer connection, created on demand when the previous one
            was closed (unhealthy, or recycled after POOL_IDLE_TIMEOUT_S).
        """
        loop = asyncio.get_running_loop()

        def _get_writer() -> sqlite3.Connection:
            with self._pool_lock:
                if not self._writer_conn:
                    self._writer_conn = self._create_connection(readonly=False)
                    logger.debug('Created new writer connection')
                # Every writer user (the write queue, begin_transaction and the
                # allow_write scope) acquires through here, so this is the single
                # site that can date the writer for idle recycling.
                self._writer_last_used = time.monotonic()
                return self._writer_conn

        return await loop.run_in_executor(None, _get_writer)

    async def _get_reader_connection(self) -> sqlite3.Connection:
        """Get a reader connection from the pool.

        The worker registers the new connection in ``_temporary_connections``
        BEFORE returning it, so a cancellation landing on the creation await
        would otherwise orphan an open connection: the caller's finally-cleanup
        never runs (its ``conn`` is left unbound) and nothing else reclaims
        temporary connections during normal operation. On any unwind the
        future is drained and the created connection is closed and untracked so
        a cancelled read leaks nothing.

        Returns:
            An isolated read-only SQLite connection tracked for later cleanup.
        """
        loop = asyncio.get_running_loop()

        def _get_reader() -> sqlite3.Connection:
            # For concurrent operations, always create isolated connections
            # SQLite doesn't handle connection sharing well across threads
            # Always create a temporary connection to ensure thread safety
            temp_conn = self._create_connection(readonly=True)
            with self._connection_lock:
                self._temporary_connections.add(temp_conn)
            logger.debug('Created isolated reader connection for thread safety')
            return temp_conn

        future = loop.run_in_executor(None, _get_reader)
        try:
            return await asyncio.shield(future)
        except BaseException:
            while not future.done():
                try:
                    await asyncio.wait([future])
                except asyncio.CancelledError:
                    continue
            if not future.cancelled() and future.exception() is None:
                orphan = future.result()
                with self._connection_lock:
                    self._temporary_connections.discard(orphan)
                self._safe_close_connection(orphan)
            raise

    async def _acquire_connection_charging_faults(
        self,
        create: Callable[[], Awaitable[sqlite3.Connection]],
    ) -> sqlite3.Connection:
        """Establish a connection, charging the breaker on genuine creation faults.

        Connection ESTABLISHMENT (creating a reader, recreating the writer) sits
        above the breaker-recording try blocks that wrap connection USE, so a
        creation failure -- e.g. SQLITE_CANTOPEN 'unable to open database file'
        when the volume holding the database detaches -- would otherwise propagate
        with the breaker still reporting healthy for the whole outage. This wrapper
        records exactly one breaker failure for a genuine creation fault and then
        re-raises, so the read and store paths can open the breaker during a
        blackholed-storage outage the same way the queued write path already does.

        A genuine (non-contention) creation Exception is re-raised after charging
        exactly one breaker failure. The SQLITE_BUSY / SQLITE_LOCKED handshake-
        contention family stays UNCHARGED (self-clearing cross-process contention,
        not a database fault), and a cancellation (a non-Exception BaseException)
        unwinding the establishment await propagates without being charged -- both
        mirror the lock-retry and cancellation exemptions elsewhere on this backend.

        Args:
            create: Zero-argument coroutine factory that establishes and returns
                the connection (reader creation or writer recreation).

        Returns:
            The established SQLite connection.

        Raises:
            ControlFlowError: Re-raised unchanged without charging the breaker
                (normal control flow, not a database fault).
        """
        try:
            return await create()
        except ControlFlowError:
            raise
        except Exception as e:
            if not is_sqlite_locked_error(e):
                self._record_charged_failure(e)
            raise

    async def _close_all_connections(self) -> None:
        """Close all database connections with comprehensive tracking."""
        # Close connections synchronously to avoid race conditions with garbage collection
        # Need both locks to access pools and tracking sets
        with self._pool_lock, self._connection_lock:
            # Create master list of ALL connections to close
            all_conns_to_close: set[sqlite3.Connection] = set()

            # Add all tracked connections
            logger.debug(f'Tracked connections: {len(self._all_connections)}')
            all_conns_to_close.update(self._all_connections)

            logger.debug(f'Temporary connections: {len(self._temporary_connections)}')
            all_conns_to_close.update(self._temporary_connections)

            # Add writer connection
            if self._writer_conn:
                logger.debug(f'Writer connection: {id(self._writer_conn)}')
                all_conns_to_close.add(self._writer_conn)

            logger.debug(f'Total connections to close: {len(all_conns_to_close)}')
            for conn in all_conns_to_close:
                logger.debug(f'  Connection to close: {id(conn)}')

            # Close ALL connections - don't check if already closed, just close them
            closed_count = 0
            for conn in all_conns_to_close:
                self._safe_close_connection(conn)
                closed_count += 1
                logger.debug(f'Closed connection: {id(conn)}')

            logger.debug(f'Closed {closed_count} connections out of {len(all_conns_to_close)} total')

            # Log any connection IDs that weren't closed
            remaining_ids = set(self._connection_ids.keys())
            remaining_ids.difference_update(id(conn) for conn in all_conns_to_close)
            if remaining_ids:
                logger.warning(f'Connection IDs not in close list: {remaining_ids}')

            # Clear all connection references completely
            self._all_connections.clear()
            self._temporary_connections.clear()
            self._writer_conn = None

    def _close_all_connections_sync(self) -> None:
        """Synchronous cleanup used by __del__, safe in interpreter shutdown."""
        with self._pool_lock, self._connection_lock:
            conns: set[sqlite3.Connection] = set(self._all_connections)
            if self._writer_conn:
                conns.add(self._writer_conn)
            conns.update(self._temporary_connections)

            for conn in conns:
                self._safe_close_connection(conn)

            self._all_connections.clear()
            self._temporary_connections.clear()
            self._writer_conn = None
