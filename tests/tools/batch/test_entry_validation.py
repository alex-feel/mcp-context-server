"""Tests for app/tools/batch/entry_validation.py.

Covers the visibility vocabulary both batch validators share with the
single-entry write tools: 'private' and 'public' pass, and any other value fails
the entry with one fixed message.
"""

from unittest.mock import MagicMock

import pytest

from app.auth.principal import RequestPrincipal
from app.ids import generate_id
from app.repositories.context_repository import ContextRepository
from app.tools.batch.entry_validation import validate_store_entry
from app.tools.batch.entry_validation import validate_update_entry

_VISIBILITY_ERROR = "visibility must be one of 'private', 'public'"
_PRINCIPAL = RequestPrincipal(principal_id='alice', groups=frozenset(), roles=frozenset())


def _store_entry(visibility: str) -> dict[str, object]:
    return {'thread_id': 'validation-thread', 'source': 'agent', 'text': 'entry body', 'visibility': visibility}


class TestValidateStoreEntryVisibility:
    """validate_store_entry accepts exactly 'private' and 'public'."""

    @pytest.mark.parametrize('visibility', ['private', 'public'])
    def test_private_and_public_accepted(self, visibility: str) -> None:
        """Both values pass and become the entry's effective visibility."""
        validated, error = validate_store_entry(_store_entry(visibility), 0, _PRINCIPAL)
        assert error is None
        assert validated is not None
        assert validated['visibility'] == visibility

    @pytest.mark.parametrize('visibility', ['shared', 'everyone'])
    def test_other_values_rejected(self, visibility: str) -> None:
        """Any other value fails the entry with the vocabulary message."""
        assert validate_store_entry(_store_entry(visibility), 0, _PRINCIPAL) == (None, _VISIBILITY_ERROR)


class TestValidateUpdateEntryVisibility:
    """validate_update_entry accepts exactly 'private' and 'public'."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('visibility', ['private', 'public'])
    async def test_private_and_public_accepted(self, visibility: str) -> None:
        """Both values pass and reach the validated update."""
        context_id = generate_id()
        repo = MagicMock(spec=ContextRepository)
        validated, resolved_id, error = await validate_update_entry(
            {'context_id': context_id, 'visibility': visibility}, 0, repo,
        )
        assert error is None
        assert resolved_id == context_id
        assert validated is not None
        assert validated['visibility'] == visibility

    @pytest.mark.asyncio
    @pytest.mark.parametrize('visibility', ['shared', 'everyone'])
    async def test_other_values_rejected(self, visibility: str) -> None:
        """Any other value fails the update with the vocabulary message."""
        context_id = generate_id()
        repo = MagicMock(spec=ContextRepository)
        result = await validate_update_entry({'context_id': context_id, 'visibility': visibility}, 0, repo)
        assert result == (None, context_id, _VISIBILITY_ERROR)
