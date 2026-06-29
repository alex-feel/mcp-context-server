"""Regression tests for the SHARED summary-model semaphore in run_generation.

The store/update latency hardening unified summary-model concurrency: the SINGLE
``_summary_model_semaphore`` (sized ``SUMMARY_MAX_CONCURRENT``) is acquired by BOTH
the flat document summary (``generate_summary_with_timeout``) AND every per-node
index_tree summary (``_summarize_node``). These tests assert three properties of
that unification through the real ``run_generation`` orchestrator:

1. SHARED budget: across one flat summary + N node summaries, the peak number of
   concurrent summary-model calls never exceeds ``SUMMARY_MAX_CONCURRENT`` (set to
   1 here), proving both paths gate on the same object.
2. NO PERMIT LEAK: after both a SUCCESS path and an ABORT path (embedding leg
   raises) through ``run_generation``, the semaphore's internal ``_value`` returns
   to its full count -- cancellation releases permits.
3. ABORT cancels node summaries: when the embedding leg raises while node
   summaries are in flight, no node-summary provider call is left running after
   ``run_generation`` raises (the fake provider distinguishes "completed" from
   "cancelled").

Settings/semaphore reset pattern (per CLAUDE.md): ``monkeypatch.setenv(...)`` ->
``get_settings.cache_clear()`` -> rebind ``shared_tools.settings`` ->
``_reset_summary_model_semaphore()`` / ``_reset_node_summary_semaphore()``, then
restore on teardown via an autouse fixture so module-level state does not leak.
"""

import asyncio
from collections.abc import Generator

import pytest

import app.tools._shared as shared_tools
from app.settings import get_settings

# Three headings, each section >= 500 chars (INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH),
# so generate_index_nodes_with_timeout attempts a summary for every section.
MULTI_HEADING_TEXT = (
    '# Heading One\n' + ('a' * 600) + '\n\n'
    '## Heading Two\n' + ('b' * 600) + '\n\n'
    '## Heading Three\n' + ('c' * 600) + '\n'
)


class _ConcurrencyTrackingProvider:
    """Fake summary provider that records peak concurrent in-flight calls and
    distinguishes completed calls from cancelled ones.

    Both ``summarize`` (flat) and ``summarize_with_prompt`` (per-node) route
    through the same accounting so the test can observe the COMBINED concurrency
    governed by the shared summary-model semaphore. ``node_current`` tracks only
    the per-node calls, so a test can wait for a NODE summary specifically to be
    in flight; ``node_hold`` blocks ONLY the per-node calls, letting the flat
    summary complete promptly.
    """

    def __init__(
        self,
        *,
        hold: asyncio.Event | None = None,
        node_hold: asyncio.Event | None = None,
    ) -> None:
        self.current = 0
        self.peak = 0
        self.completed = 0
        self.cancelled = 0
        # In-flight per-node (summarize_with_prompt) calls only.
        self.node_current = 0
        # When set, a call blocks on this event until released, so the test can
        # deterministically pin calls "in flight" while it triggers an abort.
        self._hold = hold
        # When set, ONLY the per-node calls block on this event; the flat summary
        # then completes promptly (its own ``hold`` is typically None), so a test
        # can pin node summaries in flight without stalling the flat-summary leg.
        self._node_hold = node_hold

    async def _run(self, hold_event: asyncio.Event | None, *, is_node: bool) -> str:
        self.current += 1
        if is_node:
            self.node_current += 1
        self.peak = max(self.peak, self.current)
        try:
            if hold_event is not None:
                await hold_event.wait()
            else:
                # Yield so genuinely-concurrent calls overlap when the budget allows.
                await asyncio.sleep(0.01)
            self.completed += 1
            return 'fake summary'
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.current -= 1
            if is_node:
                self.node_current -= 1

    async def summarize(self, _text: str, _source: str) -> str:
        return await self._run(self._hold, is_node=False)

    async def summarize_with_prompt(self, _text: str, _system_prompt: str) -> str:
        node_event = self._node_hold if self._node_hold is not None else self._hold
        return await self._run(node_event, is_node=True)


@pytest.fixture(autouse=True)
def restore_summary_settings() -> Generator[None, None, None]:
    """Snapshot and restore the module-level settings + semaphores so the
    per-test ``SUMMARY_MAX_CONCURRENT`` override never leaks across tests.
    """
    original_settings = shared_tools.settings
    yield
    shared_tools.settings = original_settings
    get_settings.cache_clear()
    shared_tools._reset_summary_model_semaphore()
    shared_tools._reset_node_summary_semaphore()


def _apply_low_summary_budget(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    """Force SUMMARY_MAX_CONCURRENT=value and rebind the shared semaphore.

    Also forces ENABLE_INDEX_TREE_NODE_SUMMARIES on and a low node min-content
    length so the multi-heading text reliably produces node summaries.
    """
    monkeypatch.setenv('SUMMARY_MAX_CONCURRENT', str(value))
    monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
    monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '100')
    monkeypatch.setenv('SUMMARY_MIN_CONTENT_LENGTH', '0')
    get_settings.cache_clear()
    shared_tools.settings = get_settings()
    shared_tools._reset_summary_model_semaphore()
    shared_tools._reset_node_summary_semaphore()


@pytest.mark.usefixtures('mock_server_dependencies')
class TestSharedSummarySemaphore:
    """The flat-summary and node-summary paths share ONE summary-model budget."""

    @pytest.mark.asyncio
    async def test_peak_concurrency_bounded_by_shared_budget(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With SUMMARY_MAX_CONCURRENT=1, the flat summary + 3 node summaries
        never exceed 1 concurrent summary-model call -- proving both paths gate on
        the SAME ``_summary_model_semaphore``.
        """
        _apply_low_summary_budget(monkeypatch, 1)
        provider = _ConcurrencyTrackingProvider()

        # No embedding leg (provider None) so the only summary-model load is the
        # flat summary + the node summaries, all on the shared budget.
        monkeypatch.setattr(shared_tools, 'get_embedding_provider', lambda: None)
        monkeypatch.setattr(shared_tools, 'get_summary_provider', lambda: provider)
        monkeypatch.setattr(shared_tools, 'compute_summary_total_timeout', lambda: 5.0)

        _emb, summary_text, index_nodes = await shared_tools.run_generation(
            MULTI_HEADING_TEXT, 'agent',
            run_embedding=False,
            run_summary=True,
            run_nodes=True,
        )

        assert summary_text == 'fake summary'
        # All three heading sections produced node rows.
        assert index_nodes is not None
        assert len(index_nodes) == 3
        # The flat summary (1) plus 3 node summaries (4 total) were observed, but
        # peak concurrency stayed at the shared cap of 1.
        assert provider.completed == 4
        assert provider.peak <= 1

    @pytest.mark.asyncio
    async def test_higher_budget_allows_more_concurrency(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sanity counterpart: with SUMMARY_MAX_CONCURRENT=3 the same workload
        overlaps (peak > 1), confirming the cap in the prior test is the budget
        doing the bounding, not accidental serialization.
        """
        _apply_low_summary_budget(monkeypatch, 3)
        provider = _ConcurrencyTrackingProvider()

        monkeypatch.setattr(shared_tools, 'get_embedding_provider', lambda: None)
        monkeypatch.setattr(shared_tools, 'get_summary_provider', lambda: provider)
        monkeypatch.setattr(shared_tools, 'compute_summary_total_timeout', lambda: 5.0)

        await shared_tools.run_generation(
            MULTI_HEADING_TEXT, 'agent',
            run_embedding=False,
            run_summary=True,
            run_nodes=True,
        )

        assert provider.completed == 4
        # With budget 3 and 4 calls, at least 2 ran concurrently at some point.
        assert provider.peak >= 2
        assert provider.peak <= 3

    @pytest.mark.asyncio
    async def test_success_path_releases_all_permits(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After a SUCCESS run_generation, the shared semaphore's ``_value``
        returns to the full SUMMARY_MAX_CONCURRENT count (no permit leak).
        """
        _apply_low_summary_budget(monkeypatch, 2)
        provider = _ConcurrencyTrackingProvider()

        monkeypatch.setattr(shared_tools, 'get_embedding_provider', lambda: None)
        monkeypatch.setattr(shared_tools, 'get_summary_provider', lambda: provider)
        monkeypatch.setattr(shared_tools, 'compute_summary_total_timeout', lambda: 5.0)

        await shared_tools.run_generation(
            MULTI_HEADING_TEXT, 'agent',
            run_embedding=False,
            run_summary=True,
            run_nodes=True,
        )

        assert shared_tools._summary_model_semaphore._value == 2

    @pytest.mark.asyncio
    async def test_abort_releases_permits_and_cancels_node_summaries(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ABORT path: the embedding leg raises while a node summary is in
        flight. run_generation must (a) raise the combined error, (b) cancel the
        in-flight NODE summary so none is left running, and (c) leave the shared
        semaphore's ``_value`` back at full count (no leaked permits).

        Only the per-node calls block on ``hold``; the flat summary completes
        promptly. This is deliberate: if the flat summary also blocked, the
        abort-mandatory flat-summary leg would stall the whole abort-legs gather
        until its timeout fired (a needless multi-second wait), and the node leg --
        which awaits the flat summary first -- would never start, so the
        cancellation assertion would be satisfied by the flat summary instead of a
        genuine in-flight node summary.
        """
        from fastmcp.exceptions import ToolError

        _apply_low_summary_budget(monkeypatch, 2)

        # Block ONLY the per-node calls so a node summary is genuinely in flight
        # (holding the shared permit) when the embedding leg fails, while the flat
        # summary completes immediately.
        hold = asyncio.Event()
        provider = _ConcurrencyTrackingProvider(node_hold=hold)

        async def failing_embed(_text: str) -> object:
            # Fail the abort-mandatory embedding leg only AFTER a node summary is
            # genuinely in flight, so the abort cancels a real in-flight node call.
            for _ in range(400):
                if provider.node_current >= 1:
                    break
                await asyncio.sleep(0.005)
            raise RuntimeError('embedding provider exploded')

        monkeypatch.setattr(shared_tools, 'get_summary_provider', lambda: provider)
        monkeypatch.setattr(shared_tools, 'compute_summary_total_timeout', lambda: 5.0)
        # Replace the whole embed->compress leg with a failing coroutine.
        monkeypatch.setattr(shared_tools, 'embed_then_compress', failing_embed)

        with pytest.raises(ToolError, match='Generation failed after exhausting configured retries'):
            await shared_tools.run_generation(
                MULTI_HEADING_TEXT, 'agent',
                run_embedding=True,
                run_summary=True,
                run_nodes=True,
            )

        # The shared budget is fully restored: every acquired permit (the flat
        # summary + the in-flight node summary) was released on cancellation.
        assert shared_tools._summary_model_semaphore._value == 2

        # No node-summary provider call is left running after the abort: the
        # in-flight node call was cancelled, and nothing remains active.
        assert provider.current == 0
        assert provider.node_current == 0
        # The abort genuinely cancelled an in-flight NODE summary rather than
        # letting it complete (the hold event was never released).
        assert provider.cancelled >= 1
        assert not hold.is_set()

    @pytest.mark.asyncio
    async def test_outer_cancellation_releases_node_leg_when_summary_disabled(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OUTER cancellation with ``run_summary=False`` must not orphan the node leg.

        This is the leak window the abort-ERROR test does NOT cover. With
        ``run_summary=False`` (the common short-text path -- ``update_context``
        hardcodes it) the node leg does not transitively cancel via the flat
        summary, so when an outer cancellation (MCP client disconnect / request
        timeout) lands on the abort-legs gather -- kept suspended here by a slow
        embedding leg -- ``run_generation`` must still cancel and await the node leg
        in its ``finally``. Otherwise the node task is orphaned holding the shared
        summary-model permit, and repeated cancellations starve all summary
        generation. The test asserts (a) the shared semaphore is restored, (b) the
        in-flight node call was genuinely cancelled, and (c) no node task is leaked.
        """
        _apply_low_summary_budget(monkeypatch, 2)

        # Node summaries block in flight on ``hold`` (never released here) so they
        # are pinned holding the shared summary-model permit when we cancel.
        hold = asyncio.Event()
        provider = _ConcurrencyTrackingProvider(hold=hold)

        async def slow_embed(_text: str) -> object:
            # Keep the abort-legs gather suspended so the outer cancel lands THERE
            # (the exact leak window), not on the final ``await node_task``.
            await asyncio.sleep(5)
            return None

        monkeypatch.setattr(shared_tools, 'get_summary_provider', lambda: provider)
        monkeypatch.setattr(shared_tools, 'compute_summary_total_timeout', lambda: 5.0)
        monkeypatch.setattr(shared_tools, 'embed_then_compress', slow_embed)

        task = asyncio.create_task(
            shared_tools.run_generation(
                MULTI_HEADING_TEXT, 'agent',
                run_embedding=True,
                run_summary=False,
                run_nodes=True,
            ),
        )

        # Wait until at least one per-node summary is in flight holding a permit.
        for _ in range(400):
            if provider.current >= 1:
                break
            await asyncio.sleep(0.005)
        assert provider.current >= 1, 'node summary never reached its in-flight hold'
        assert shared_tools._summary_model_semaphore._value < 2  # a permit is held

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The finally cancelled and awaited the node leg: shared budget fully
        # restored, no node-summary call left running, the in-flight call genuinely
        # cancelled (its hold was never released), and no node task leaked.
        assert shared_tools._summary_model_semaphore._value == 2
        assert provider.current == 0
        assert provider.cancelled >= 1
        assert not hold.is_set()
        leaked_node_tasks = [
            t for t in asyncio.all_tasks()
            if not t.done() and (
                '_nodes_after_summary' in repr(t.get_coro())
                or 'generate_index_nodes_with_timeout' in repr(t.get_coro())
                or '_summarize_node' in repr(t.get_coro())
            )
        ]
        assert leaked_node_tasks == []


class _PrecedenceTrackingProvider:
    """Fake summary provider that records, in order, the lifecycle events of the
    flat document summary and every per-node summary, plus whether the flat
    summary had COMPLETED at the moment each per-node summary STARTED.

    ``summarize`` (the flat document summary) is made deliberately slow so that, if
    the per-node leg did NOT wait for it, a per-node ``summarize_with_prompt`` would
    start while the flat summary is still in flight (overlapping it). Each per-node
    call records ``flat_done`` at its start, so the test can assert strict
    flat-summary precedence rather than mere acquisition order (which, under
    budget 1, would be preserved by task-creation order even without the
    precedence await and so would NOT discriminate the regression).
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.flat_done = False
        # Records flat_done at the START of each per-node summary, in call order.
        self.node_start_flat_done: list[bool] = []

    async def summarize(self, _text: str, _source: str) -> str:
        self.events.append('flat_start')
        # Slow enough that a non-waiting node leg would demonstrably overlap it
        # (budget 2 leaves a free permit a racing node could grab).
        await asyncio.sleep(0.05)
        self.flat_done = True
        self.events.append('flat_end')
        return 'fake summary'

    async def summarize_with_prompt(self, _text: str, _system_prompt: str) -> str:
        self.node_start_flat_done.append(self.flat_done)
        self.events.append(f'node_start(flat_done={self.flat_done})')
        await asyncio.sleep(0.001)
        return 'fake summary'


@pytest.mark.usefixtures('mock_server_dependencies')
class TestFlatSummaryPrecedence:
    """Per-node summaries start only AFTER the flat document summary finishes
    (the abort-mandatory flat summary keeps strict precedence -- no latency
    inversion)."""

    @pytest.mark.asyncio
    async def test_flat_summary_precedes_node_summaries(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With SUMMARY_MAX_CONCURRENT=2, no per-node ``summarize_with_prompt`` may
        START until the flat ``summarize`` has fully COMPLETED.

        ``_nodes_after_summary`` awaits the flat-summary task before launching any
        node coroutine, so the abort-mandatory flat summary always runs to
        completion first. Budget 2 (not 1) is the discriminator: with a free second
        permit, a node summary that did NOT wait for the flat summary would overlap
        it (``flat_done == False`` at the node's start). This test therefore FAILS
        if ``_nodes_after_summary`` stops awaiting the flat summary first -- a
        budget-1 acquisition-order check would NOT, because task-creation order
        alone preserves "flat first" even when the precedence await is removed.
        """
        # Budget 2 leaves a free permit a racing (non-waiting) node summary could
        # grab, making any precedence violation observable. Low node min-content
        # length guarantees the heading sections actually produce node summaries.
        _apply_low_summary_budget(monkeypatch, 2)
        provider = _PrecedenceTrackingProvider()

        # No embedding leg (provider None) so the only model load is the flat
        # summary + the node summaries, all on the shared budget.
        monkeypatch.setattr(shared_tools, 'get_embedding_provider', lambda: None)
        monkeypatch.setattr(shared_tools, 'get_summary_provider', lambda: provider)
        monkeypatch.setattr(shared_tools, 'compute_summary_total_timeout', lambda: 5.0)

        _emb, summary_text, index_nodes = await shared_tools.run_generation(
            MULTI_HEADING_TEXT, 'agent',
            run_embedding=False,
            run_summary=True,
            run_nodes=True,
        )

        assert summary_text == 'fake summary'
        # Node summaries actually ran (otherwise the assertions below are vacuous).
        assert index_nodes is not None
        assert len(index_nodes) == 3
        assert provider.node_start_flat_done  # at least one node summary ran
        # The discriminating assertion: EVERY per-node summary started only after
        # the flat summary had completed -- none overlapped it.
        assert all(provider.node_start_flat_done)
        # And in the raw event log, the flat summary's end precedes the first
        # per-node start (no node event appears before 'flat_end').
        flat_end_idx = provider.events.index('flat_end')
        first_node_idx = next(
            i for i, e in enumerate(provider.events) if e.startswith('node_start')
        )
        assert flat_end_idx < first_node_idx
