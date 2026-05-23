"""Regression test verifying the compression semaphore bounds per-call CPU
concurrency, not just the outer gather wrapper.

Before the fix the ``async with _compression_semaphore`` block wrapped
the outer ``asyncio.gather(...)`` call, meaning an N-chunk batch ran all
N encodes under a single permit -- completely bypassing the documented
``COMPRESSION_MAX_CONCURRENT`` bound. After the fix the semaphore wraps
the inner ``encode_sync`` call, so per-call concurrency is correctly
bounded.
"""

import time
from collections.abc import Generator

import pytest

pytest.importorskip('numpy')

import app.tools._shared as shared_module
from app.repositories.embedding_repository import ChunkEmbedding
from app.settings import get_settings


@pytest.fixture
def reset_compression_state() -> Generator[None, None, None]:
    """Restore the settings singleton + module bindings after the test.

    monkeypatch restores env vars and individual setattr targets, but the
    @lru_cache singleton inside ``get_settings`` retains the polluted
    AppSettings instance unless explicitly cleared. Downstream tests that
    assume the default ``compression.enabled = False`` state break
    without this teardown step. The cached compression provider in
    :mod:`app.compression.factory` is also reset so the patched factory
    does not leak its FakeProvider into subsequent tests.
    """
    yield
    from app.compression import reset_cached_compression_provider
    get_settings.cache_clear()
    shared_module.settings = get_settings()
    shared_module._reset_compression_semaphore()
    reset_cached_compression_provider()


@pytest.mark.asyncio
@pytest.mark.usefixtures('reset_compression_state')
async def test_semaphore_serializes_per_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With max_concurrent=2 and 4 chunks, observe at most 2 concurrent encodes."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_MAX_CONCURRENT', '2')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '16')
    get_settings.cache_clear()
    monkeypatch.setattr(shared_module, 'settings', get_settings())

    # Rebind the module-level semaphore so the new max_concurrent takes effect.
    shared_module._reset_compression_semaphore()

    concurrent_now = 0
    max_observed = 0

    class FakeProvider:
        def encode_sync(self, _vector: object) -> bytes:
            nonlocal concurrent_now, max_observed
            concurrent_now += 1
            max_observed = max(max_observed, concurrent_now)
            # Simulate real work to widen the concurrency window
            time.sleep(0.05)
            concurrent_now -= 1
            return b'encoded-payload'

    # Reset the cached provider so the patched factory takes effect on
    # the next call. The cached helper in app/compression/factory.py
    # memoizes the constructed provider; pre-test pollution would mask
    # the FakeProvider.
    from app.compression import reset_cached_compression_provider
    reset_cached_compression_provider()

    async def _fake_async_factory() -> FakeProvider:
        return FakeProvider()

    monkeypatch.setattr(
        'app.compression.get_cached_compression_provider',
        _fake_async_factory,
    )

    chunks = [
        ChunkEmbedding(embedding=[0.1] * 16, start_index=0, end_index=10)
        for _ in range(4)
    ]

    result = await shared_module.generate_compression_with_timeout(chunks)

    assert result is not None
    assert all(c.payload == b'encoded-payload' for c in result)
    # The semaphore MUST cap concurrency at 2; if it wrapped only gather
    # we would observe up to 4 concurrent encodes.
    assert max_observed <= 2, (
        f'Expected semaphore to bound concurrency to 2, observed {max_observed}'
    )
