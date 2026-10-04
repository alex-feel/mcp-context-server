"""Tests for app/tools/batch/update_access.py.

Covers the batch update's pre-generation write-access check (an entry the
caller may not read is not found, one it may read but not modify is not
authorized, and a visibility change is owner-only; non-atomic mode collects
the refusals per entry and atomic mode raises on the first) and the
in-transaction re-probe that turns a compare-and-set matching zero rows into a
not-found error or a version conflict.
"""

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastmcp.exceptions import ToolError

from app.auth.principal import RequestPrincipal
from app.repositories.context_repository.records import EntryProbe
from app.repositories.context_repository.records import VersionConflictError
from app.tools._transactions import EntryNotFoundError
from app.tools.batch.update_access import authorize_updates
from app.tools.batch.update_access import reraise_disambiguated_cas_conflict
from tests.helpers import LOCAL_SCOPE

_ALICE = RequestPrincipal(principal_id='alice', groups=frozenset(), roles=frozenset())
_MISSING = EntryProbe(exists=False, source=None, version=None, owner_id=None, can_write=False)
_READ_ONLY = EntryProbe(exists=True, source='agent', version=3, owner_id='bob', can_write=False)
_GRANTED = EntryProbe(exists=True, source='user', version=5, owner_id='bob', can_write=True)
_OWNED = EntryProbe(exists=True, source='agent', version=7, owner_id='alice', can_write=True)


def _repos_probing(*probes: EntryProbe) -> MagicMock:
    repos = MagicMock()
    repos.context.check_entry_exists = AsyncMock(side_effect=list(probes))
    return repos


def _update(index: int, context_id: str, **fields: Any) -> dict[str, Any]:
    return {'index': index, 'context_id': context_id, **fields}


class TestAuthorizeUpdates:
    """authorize_updates admits only the updates the caller may apply."""

    @pytest.mark.asyncio
    async def test_writable_entries_record_source_and_version(self) -> None:
        """Every admitted update contributes its entry's source and pre-generation version."""
        repos = _repos_probing(_OWNED, _GRANTED)

        sources, versions, errors = await authorize_updates(
            repos, [_update(0, 'owned', text='a'), _update(1, 'granted', text='b')], _ALICE,
            scope=LOCAL_SCOPE, atomic=True,
        )

        assert sources == {'owned': 'agent', 'granted': 'user'}
        assert versions == {'owned': 7, 'granted': 5}
        assert errors == []
        repos.context.check_entry_exists.assert_any_await('owned', scope=LOCAL_SCOPE)
        repos.context.check_entry_exists.assert_any_await('granted', scope=LOCAL_SCOPE)

    @pytest.mark.asyncio
    async def test_non_atomic_refusals_are_collected_per_entry(self) -> None:
        """Non-atomic mode records each refusal by its original index and keeps going."""
        repos = _repos_probing(_MISSING, _READ_ONLY, _GRANTED, _OWNED)
        updates = [
            _update(0, 'gone', text='a'),
            _update(1, 'read-only', text='b'),
            _update(2, 'granted', visibility='public'),
            _update(3, 'owned', visibility='public'),
        ]

        sources, versions, errors = await authorize_updates(repos, updates, _ALICE, scope=LOCAL_SCOPE, atomic=False)

        assert errors == [
            (0, 'gone', 'Context entry gone not found'),
            (1, 'read-only', 'Not authorized to modify context entry read-only'),
            (2, 'granted', 'Only the owner may change the visibility of context granted'),
        ]
        assert sources == {'granted': 'user', 'owned': 'agent'}
        assert versions == {'granted': 5, 'owned': 7}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('probe', 'fields', 'message'),
        [
            (_MISSING, {'text': 'a'}, 'Context entry e1 not found at index 4'),
            (_READ_ONLY, {'text': 'a'}, 'Not authorized to modify context entry e1 at index 4'),
            (_GRANTED, {'visibility': 'public'}, 'Only the owner may change the visibility of context e1 (index 4)'),
        ],
    )
    async def test_atomic_mode_raises_on_the_first_refusal(
        self, probe: EntryProbe, fields: dict[str, Any], message: str,
    ) -> None:
        """Atomic mode aborts with the refusal and the update's original index."""
        repos = _repos_probing(probe)

        with pytest.raises(ToolError) as error:
            await authorize_updates(repos, [_update(4, 'e1', **fields)], _ALICE, scope=LOCAL_SCOPE, atomic=True)

        assert str(error.value) == message


class TestReraiseDisambiguatedCasConflict:
    """A compare-and-set that matched zero rows re-probes the row on the open transaction."""

    @pytest.mark.asyncio
    async def test_row_the_caller_may_no_longer_modify_is_not_found(self) -> None:
        """A gone or no-longer-writable row raises the not-found signal."""
        repos = MagicMock()
        repos.context.entry_exists = AsyncMock(return_value=False)
        txn = MagicMock()

        with pytest.raises(EntryNotFoundError):
            await reraise_disambiguated_cas_conflict(repos, txn, 'e1', scope=LOCAL_SCOPE)

        repos.context.entry_exists.assert_awaited_once_with('e1', scope=LOCAL_SCOPE, txn=txn)

    @pytest.mark.asyncio
    async def test_row_still_writable_is_a_version_conflict(self) -> None:
        """A row the caller may still modify had its version changed by a concurrent writer."""
        repos = MagicMock()
        repos.context.entry_exists = AsyncMock(return_value=True)

        with pytest.raises(VersionConflictError):
            await reraise_disambiguated_cas_conflict(repos, MagicMock(), 'e1', scope=LOCAL_SCOPE)
