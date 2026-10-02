"""Lifecycle of the SQLite backend.

Initialization and background-task startup, graceful shutdown, and the best-effort cleanup
when a backend is garbage-collected without a shutdown.
"""

import asyncio
import contextlib
import logging

from app.backends.sqlite_backend.config import is_test_environment
from app.backends.sqlite_backend.health import SQLiteHealthMixin
from app.backends.sqlite_backend.write_queue import SQLiteWriteQueueMixin
from app.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class SQLiteLifecycleMixin(SQLiteWriteQueueMixin, SQLiteHealthMixin):
    """Start the background tasks, shut down gracefully, and close leftover connections on collection."""

    async def initialize(self) -> None:
        """Initialize the connection manager and start background tasks."""
        logger.info(f'Initializing connection manager for {self.db_path}')

        # Create asyncio primitives in proper async context
        # This MUST happen here, not in __init__, to ensure they bind to the correct event loop
        if self._reader_semaphore is None:
            self._reader_semaphore = asyncio.Semaphore(self.pool_config.max_readers)
        if self._write_queue is None:
            self._write_queue = asyncio.Queue()
        if self._writer_lock is None:
            self._writer_lock = asyncio.Lock()
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()
        if self._shutdown_complete is None:
            self._shutdown_complete = asyncio.Event()

        # Create database directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize writer connection
        await self._ensure_writer_connection()

        # Start background tasks with proper tracking
        if not self._write_processor_task:
            task = asyncio.create_task(self._process_write_queue())
            self._write_processor_task = task
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        if not self._health_check_task:
            task = asyncio.create_task(self._health_check_loop())
            self._health_check_task = task
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        logger.info('Connection manager initialized successfully')

    async def wait_for_shutdown_complete(self, timeout_seconds: float | None = None) -> bool:
        """Wait for shutdown to complete with optional timeout.

        Args:
            timeout_seconds: Maximum time to wait in seconds, None for no timeout

        Returns:
            bool: True if shutdown completed, False if timed out
        """
        assert self._shutdown_complete is not None, 'Backend not initialized, call initialize() first'
        try:
            if timeout_seconds is None:
                await self._shutdown_complete.wait()
                return True
            await asyncio.wait_for(self._shutdown_complete.wait(), timeout=timeout_seconds)
            return True
        except TimeoutError:
            return False

    async def shutdown(self) -> None:
        """Gracefully shutdown the connection manager with enhanced task cleanup."""
        logger.info('Shutting down connection manager')

        assert self._shutdown_event is not None, 'Backend not initialized, call initialize() first'
        assert self._write_queue is not None, 'Backend not initialized, call initialize() first'
        assert self._shutdown_complete is not None, 'Backend not initialized, call initialize() first'

        try:
            # Signal shutdown to all background tasks
            self._shutdown = True
            self._shutdown_event.set()

            # Drain write queue, cancel pending futures
            while not self._write_queue.empty():
                try:
                    request = self._write_queue.get_nowait()
                    if not request.future.done():
                        request.future.cancel()
                except asyncio.QueueEmpty:
                    break
                except Exception:
                    pass

            # Determine timeout based on environment
            shutdown_timeout = (
                settings.storage.shutdown_timeout_test_s
                if is_test_environment()
                else settings.storage.shutdown_timeout_s
            )

            # Give the write processor a chance to exit on the shutdown signal
            # BEFORE cancelling it: task.cancel() cannot interrupt a write
            # already running in the executor thread (the write may still
            # commit), so cancelling mid-write forfeits the true outcome the
            # caller is awaiting. The processor finishes the in-flight
            # request, resolves its future with the real result, and exits on
            # the already-set shutdown event; only a write exceeding the
            # timeout falls through to the forceful cancellation below.
            if self._write_processor_task and not self._write_processor_task.done():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        asyncio.shield(self._write_processor_task),
                        timeout=shutdown_timeout,
                    )

            # Cancel and await all background tasks
            tasks_to_cancel = list(self._background_tasks)

            # Also include specific task references if they exist
            if self._write_processor_task and not self._write_processor_task.done():
                tasks_to_cancel.append(self._write_processor_task)
            if self._health_check_task and not self._health_check_task.done():
                tasks_to_cancel.append(self._health_check_task)

            if tasks_to_cancel:
                logger.debug(f'Cancelling {len(tasks_to_cancel)} background tasks')
                for task in tasks_to_cancel:
                    if not task.done():
                        task.cancel()

                # Wait for all tasks to complete with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=shutdown_timeout,
                    )
                except TimeoutError:
                    logger.warning('Some tasks did not complete within timeout')
                    # Force-cancel any remaining tasks
                    for task in tasks_to_cancel:
                        if not task.done():
                            task.cancel()
                            # Give tasks a brief moment to handle cancellation
                            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                                await asyncio.wait_for(task, timeout=0.1)

            # Clear task references before closing connections
            self._background_tasks.clear()
            self._write_processor_task = None
            self._health_check_task = None

            # Small delay to ensure all async operations are settled
            await asyncio.sleep(0.01)

            # Close all connections
            await self._close_all_connections()

            logger.info('Connection manager shutdown complete')
        except Exception as e:
            logger.error(f'Error during connection manager shutdown: {e}')
            raise
        finally:
            # Always signal shutdown complete, even on error
            # This prevents infinite hangs in cleanup code waiting for this event
            self._shutdown_complete.set()

    def __del__(self) -> None:
        # Last safety net, if user code forgot to call shutdown
        with contextlib.suppress(Exception):
            shutdown_complete = getattr(self, '_shutdown_complete', None)
            if not shutdown_complete or not shutdown_complete.is_set():
                self._close_all_connections_sync()
