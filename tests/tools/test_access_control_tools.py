"""Tests for access-control behavior at the MCP tool boundary.

Covers owner stamping through store_context / store_context_batch (default
principal fallback, verified-principal stamping, and a new entry for text that
matches another principal's entry), the publish gate on both
store and update, the owner-only visibility change on update_context /
update_context_batch (reachable only for entries the caller may read; another
principal's private entry is not found), the not-authorized denial for an entry
the caller may read but not modify, author-group grant stamping, and the
invariant that owner_id is never a tool parameter.
"""

import inspect
import sqlite3
from typing import get_args
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.startup
from app.auth.principal import RequestPrincipal
from app.settings import get_settings
from tests.helpers import as_principal
from tests.helpers import insert_grant
from tests.helpers import read_grants


def _principal(
    principal_id: str,
    groups: frozenset[str] = frozenset(),
    roles: frozenset[str] = frozenset(),
) -> RequestPrincipal:
    return RequestPrincipal(principal_id=principal_id, groups=groups, roles=roles)


async def _read_owner_visibility(context_id: str) -> tuple[str, str]:
    backend = app.startup.get_backend()
    assert backend is not None

    def _read(conn: sqlite3.Connection) -> tuple[str, str]:
        cursor = conn.execute(
            'SELECT owner_id, visibility FROM context_entries WHERE id = ?', (context_id,),
        )
        row = cursor.fetchone()
        assert row is not None
        return row[0], row[1]

    return await backend.execute_read(_read)


async def _read_text(context_id: str) -> str:
    backend = app.startup.get_backend()
    assert backend is not None

    def _read(conn: sqlite3.Connection) -> str:
        row = conn.execute('SELECT text_content FROM context_entries WHERE id = ?', (context_id,)).fetchone()
        assert row is not None
        return str(row[0])

    return await backend.execute_read(_read)


@pytest.mark.usefixtures('initialized_server')
class TestOwnerStamping:
    """store paths stamp the server-resolved owner, never a caller value."""

    @pytest.mark.asyncio
    async def test_store_without_token_stamps_default_principal(self) -> None:
        """With no verified token the configured default principal owns the row."""
        from app.tools.context.store import store_context

        result = await store_context(
            thread_id='access-tools', source='agent', text='default-principal entry',
        )
        owner, visibility = await _read_owner_visibility(result['context_id'])
        assert owner == get_settings().access_control.default_principal
        assert visibility == 'private'

    @pytest.mark.asyncio
    async def test_store_stamps_verified_principal(self) -> None:
        """A verified principal becomes the row owner."""
        from app.tools.context.store import store_context

        with patch('app.tools.context.store.resolve_effective_principal', return_value=_principal('alice')):
            result = await store_context(
                thread_id='access-tools', source='agent', text='alice-owned entry',
            )
        owner, _ = await _read_owner_visibility(result['context_id'])
        assert owner == 'alice'

    @pytest.mark.asyncio
    async def test_batch_entry_owner_id_key_is_ignored(self) -> None:
        """A caller-supplied owner_id key in a batch entry never reaches the row."""
        from app.tools.batch.store import store_context_batch

        with patch('app.tools.batch.store.resolve_effective_principal', return_value=_principal('alice')):
            result = await store_context_batch(entries=[{
                'thread_id': 'access-tools',
                'source': 'agent',
                'text': 'owner-injection attempt',
                'owner_id': 'mallory',
            }])
        cid = result['results'][0]['context_id']
        assert cid is not None
        owner, _ = await _read_owner_visibility(cid)
        assert owner == 'alice'

    @pytest.mark.asyncio
    async def test_identical_text_of_another_principal_is_a_new_entry(self) -> None:
        """Text matching another principal's readable latest entry is stored as the sender's own
        entry, never merged into the other principal's entry, which stays unchanged."""
        from app.tools.context.store import store_context

        with as_principal('alice'):
            alice = await store_context(
                thread_id='access-dedup', source='agent', text='shared wording', visibility='public',
            )
        with as_principal('bob'):
            bob = await store_context(thread_id='access-dedup', source='agent', text='shared wording')

        assert bob['context_id'] != alice['context_id']
        assert await _read_owner_visibility(bob['context_id']) == ('bob', 'private')
        assert await _read_owner_visibility(alice['context_id']) == ('alice', 'public')

    def test_owner_id_is_never_a_tool_parameter(self) -> None:
        """No write tool exposes owner_id in its signature (wire schema source)."""
        from app.tools.batch.store import store_context_batch
        from app.tools.batch.update import update_context_batch
        from app.tools.context.store import store_context
        from app.tools.context.update import update_context

        for tool in (store_context, update_context, store_context_batch, update_context_batch):
            assert 'owner_id' not in inspect.signature(tool).parameters

    def test_visibility_parameter_accepts_private_and_public(self) -> None:
        """The single-entry write tools declare exactly 'private' and 'public' (wire schema source)."""
        from app.tools.context.store import store_context
        from app.tools.context.update import update_context

        for tool in (store_context, update_context):
            annotation = inspect.signature(tool).parameters['visibility'].annotation
            optional_type = get_args(annotation)[0]
            literal_type = next(arg for arg in get_args(optional_type) if arg is not type(None))
            assert get_args(literal_type) == ('private', 'public')


@pytest.mark.usefixtures('initialized_server')
class TestPublishGate:
    """ACCESS_CONTROL_PUBLISH_ROLE gates visibility 'public' on store and update."""

    @pytest.mark.asyncio
    async def test_store_public_denied_without_role(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Publishing without the configured role fails before any storage."""
        from app.tools.context.store import store_context

        monkeypatch.setenv('ACCESS_CONTROL_PUBLISH_ROLE', 'publisher')
        get_settings.cache_clear()
        try:
            with pytest.raises(ToolError, match='publisher'):
                await store_context(
                    thread_id='access-tools', source='agent',
                    text='denied publish', visibility='public',
                )
        finally:
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_store_public_allowed_with_role(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A caller carrying the publish role stores a public entry."""
        from app.tools.context.store import store_context

        monkeypatch.setenv('ACCESS_CONTROL_PUBLISH_ROLE', 'publisher')
        get_settings.cache_clear()
        try:
            publisher = _principal('alice', roles=frozenset({'publisher'}))
            with patch('app.tools.context.store.resolve_effective_principal', return_value=publisher):
                result = await store_context(
                    thread_id='access-tools', source='agent',
                    text='allowed publish', visibility='public',
                )
        finally:
            get_settings.cache_clear()
        _, visibility = await _read_owner_visibility(result['context_id'])
        assert visibility == 'public'

    @pytest.mark.asyncio
    async def test_store_public_allowed_when_role_unset(self) -> None:
        """With no publish role configured, any owner may publish."""
        from app.tools.context.store import store_context

        result = await store_context(
            thread_id='access-tools', source='agent',
            text='ungated publish', visibility='public',
        )
        _, visibility = await _read_owner_visibility(result['context_id'])
        assert visibility == 'public'

    @pytest.mark.asyncio
    async def test_update_public_denied_without_role(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Updating visibility to public is gated exactly like storing it."""
        from app.tools.context.store import store_context
        from app.tools.context.update import update_context

        stored = await store_context(
            thread_id='access-tools', source='agent', text='to be published later',
        )
        monkeypatch.setenv('ACCESS_CONTROL_PUBLISH_ROLE', 'publisher')
        get_settings.cache_clear()
        try:
            with pytest.raises(ToolError, match='publisher'):
                await update_context(context_id=stored['context_id'], visibility='public')
        finally:
            get_settings.cache_clear()


@pytest.mark.usefixtures('initialized_server')
class TestOwnerOnlyVisibilityChange:
    """Only the entry owner may change visibility."""

    @pytest.mark.asyncio
    async def test_owner_changes_visibility(self) -> None:
        """The owner flips visibility and the field is reported."""
        from app.tools.context.store import store_context
        from app.tools.context.update import update_context

        result = await store_context(
            thread_id='access-tools', source='agent', text='owner visibility change',
        )
        updated = await update_context(context_id=result['context_id'], visibility='public')
        assert 'visibility' in updated['updated_fields']
        _, visibility = await _read_owner_visibility(result['context_id'])
        assert visibility == 'public'

    @staticmethod
    async def _store_alice_private(
        text: str, *, write_grant_to: str | None = None, read_grant_to: str | None = None,
    ) -> str:
        """Store a private entry as alice, optionally granted to other principals, and return its id."""
        from app.tools.context.store import store_context

        with as_principal('alice'):
            result = await store_context(thread_id='access-tools', source='agent', text=text)
        context_id = result['context_id']
        backend = app.startup.get_backend()
        assert backend is not None
        if write_grant_to is not None:
            await insert_grant(backend, context_id, 'user', write_grant_to, 'write', 'alice')
        if read_grant_to is not None:
            await insert_grant(backend, context_id, 'user', read_grant_to, 'read', 'alice')
        return context_id

    @pytest.mark.asyncio
    async def test_non_owner_visibility_change_on_private_entry_is_not_found(self) -> None:
        """Another principal's private entry is not found, so its visibility gate is never reached."""
        from app.tools.context.update import update_context

        context_id = await self._store_alice_private('alice-only visibility')
        with as_principal('bob'), pytest.raises(ToolError) as error:
            await update_context(context_id=context_id, visibility='public')
        assert str(error.value) == f'Context entry with ID {context_id} not found'

    @pytest.mark.asyncio
    async def test_write_grantee_visibility_change_rejected(self) -> None:
        """A write grantee reads the entry but may not change its visibility."""
        from app.tools.context.update import update_context

        context_id = await self._store_alice_private('write-granted visibility', write_grant_to='bob')
        with as_principal('bob'), pytest.raises(ToolError, match='Only the owner'):
            await update_context(context_id=context_id, visibility='public')
        _, visibility = await _read_owner_visibility(context_id)
        assert visibility == 'private'

    @pytest.mark.asyncio
    async def test_non_owner_text_update_on_private_entry_is_not_found(self) -> None:
        """Another principal's private entry is not found for a text update either."""
        from app.tools.context.update import update_context

        context_id = await self._store_alice_private('text update target')
        with as_principal('bob'), pytest.raises(ToolError) as error:
            await update_context(context_id=context_id, text='new body')
        assert str(error.value) == f'Context entry with ID {context_id} not found'

    @pytest.mark.asyncio
    async def test_write_grantee_text_update_succeeds(self) -> None:
        """A write grantee updates the text of an entry it does not own."""
        from app.tools.context.update import update_context

        context_id = await self._store_alice_private('write-granted text target', write_grant_to='bob')
        with as_principal('bob'):
            updated = await update_context(context_id=context_id, text='new body')
        assert 'text_content' in updated['updated_fields']

    @pytest.mark.asyncio
    async def test_read_grantee_text_update_is_not_authorized(self) -> None:
        """A read grantee sees the entry but may not modify it, and the denial comes before any generation."""
        from app.tools.context.update import update_context

        context_id = await self._store_alice_private('read-granted text target', read_grant_to='bob')
        with (
            as_principal('bob'),
            patch('app.tools.context.update.run_generation', new_callable=AsyncMock) as generation,
            pytest.raises(ToolError) as error,
        ):
            await update_context(context_id=context_id, text='new body')
        assert str(error.value) == f'Not authorized to modify context entry with ID {context_id}'
        generation.assert_not_awaited()
        assert await _read_text(context_id) == 'read-granted text target'

    @pytest.mark.asyncio
    async def test_read_grantee_visibility_change_is_not_authorized(self) -> None:
        """The write check runs before the owner-only visibility check, so a read grantee is not authorized."""
        from app.tools.context.update import update_context

        context_id = await self._store_alice_private('read-granted visibility target', read_grant_to='bob')
        with as_principal('bob'), pytest.raises(ToolError) as error:
            await update_context(context_id=context_id, visibility='public')
        assert str(error.value) == f'Not authorized to modify context entry with ID {context_id}'
        _, visibility = await _read_owner_visibility(context_id)
        assert visibility == 'private'

    @pytest.mark.asyncio
    async def test_public_entry_update_by_non_owner_is_not_authorized(self) -> None:
        """Everyone reads a public entry, but only its owner and its write grantees modify it."""
        from app.tools.context.store import store_context
        from app.tools.context.update import update_context

        with as_principal('alice'):
            stored = await store_context(
                thread_id='access-tools', source='agent', text='public patch target', visibility='public',
            )
        context_id = stored['context_id']
        with as_principal('bob'), pytest.raises(ToolError) as error:
            await update_context(context_id=context_id, metadata_patch={'reviewed': True})
        assert str(error.value) == f'Not authorized to modify context entry with ID {context_id}'

    @pytest.mark.asyncio
    async def test_batch_read_grantee_update_records_not_authorized(self) -> None:
        """Non-atomic batch: a read grantee's update fails only that entry, as not authorized."""
        from app.tools.batch.update import update_context_batch

        context_id = await self._store_alice_private('batch read-granted target', read_grant_to='bob')
        with as_principal('bob'):
            batch_result = await update_context_batch(
                updates=[{'context_id': context_id, 'text': 'new body'}],
                atomic=False,
            )
        assert batch_result['failed'] == 1
        assert batch_result['results'][0]['error'] == f'Not authorized to modify context entry {context_id}'
        assert await _read_text(context_id) == 'batch read-granted target'

    @pytest.mark.asyncio
    async def test_atomic_batch_read_grantee_update_aborts_as_not_authorized(self) -> None:
        """Atomic batch: a read grantee's update aborts the whole batch, naming the entry and its index."""
        from app.tools.batch.update import update_context_batch

        context_id = await self._store_alice_private('atomic batch read-granted target', read_grant_to='bob')
        with as_principal('bob'), pytest.raises(ToolError) as error:
            await update_context_batch(
                updates=[{'context_id': context_id, 'text': 'new body'}],
                atomic=True,
            )
        assert str(error.value) == f'Not authorized to modify context entry {context_id} at index 0'
        assert await _read_text(context_id) == 'atomic batch read-granted target'

    @pytest.mark.asyncio
    async def test_batch_non_owner_update_of_private_entry_records_not_found(self) -> None:
        """Non-atomic batch: another principal's private entry fails only that entry, as not found."""
        from app.tools.batch.update import update_context_batch

        context_id = await self._store_alice_private('batch visibility target')
        with as_principal('bob'):
            batch_result = await update_context_batch(
                updates=[{'context_id': context_id, 'visibility': 'public'}],
                atomic=False,
            )
        assert batch_result['failed'] == 1
        assert batch_result['results'][0]['error'] == f'Context entry {context_id} not found'
        _, visibility = await _read_owner_visibility(context_id)
        assert visibility == 'private'

    @pytest.mark.asyncio
    async def test_batch_write_grantee_visibility_change_records_per_entry_error(self) -> None:
        """Non-atomic batch: a write grantee's visibility change fails only that entry."""
        from app.tools.batch.update import update_context_batch

        context_id = await self._store_alice_private('batch write-granted visibility target', write_grant_to='bob')
        with as_principal('bob'):
            batch_result = await update_context_batch(
                updates=[{'context_id': context_id, 'visibility': 'public'}],
                atomic=False,
            )
        assert batch_result['failed'] == 1
        assert 'Only the owner' in (batch_result['results'][0]['error'] or '')
        _, visibility = await _read_owner_visibility(context_id)
        assert visibility == 'private'

    @pytest.mark.asyncio
    async def test_atomic_batch_non_owner_update_of_private_entry_aborts_as_not_found(self) -> None:
        """Atomic batch: another principal's private entry aborts the whole batch as not found."""
        from app.tools.batch.update import update_context_batch

        context_id = await self._store_alice_private('atomic batch visibility target')
        with as_principal('bob'), pytest.raises(ToolError) as error:
            await update_context_batch(
                updates=[{'context_id': context_id, 'visibility': 'public'}],
                atomic=True,
            )
        assert str(error.value) == f'Context entry {context_id} not found at index 0'

    @pytest.mark.asyncio
    async def test_atomic_batch_write_grantee_visibility_change_aborts(self) -> None:
        """Atomic batch: a write grantee's visibility change aborts the whole batch."""
        from app.tools.batch.update import update_context_batch

        context_id = await self._store_alice_private('atomic write-granted visibility target', write_grant_to='bob')
        with as_principal('bob'), pytest.raises(ToolError, match='Only the owner'):
            await update_context_batch(
                updates=[{'context_id': context_id, 'visibility': 'public'}],
                atomic=True,
            )


@pytest.mark.usefixtures('initialized_server')
class TestBatchVisibilityVersionTracking:
    """A visibility update bumps the row version; same-batch trackers must follow."""

    @pytest.mark.asyncio
    async def test_atomic_batch_visibility_then_text_on_same_entry(self) -> None:
        """A visibility-only update followed by a same-id text update succeeds atomically.

        The visibility write rides the compare-and-set and bumps the row version,
        so the batch's local version tracker must advance too -- otherwise the
        second update presents a stale token and the whole atomic batch aborts on
        a self-inflicted version conflict.
        """
        from app.tools.batch.update import update_context_batch
        from app.tools.context.store import store_context

        stored = await store_context(
            thread_id='access-tools', source='agent', text='visibility-then-text target',
        )
        result = await update_context_batch(
            updates=[
                {'context_id': stored['context_id'], 'visibility': 'public'},
                {'context_id': stored['context_id'], 'text': 'updated after visibility change'},
            ],
            atomic=True,
        )
        assert result['succeeded'] == 2, result
        _, visibility = await _read_owner_visibility(stored['context_id'])
        assert visibility == 'public'

    @pytest.mark.asyncio
    async def test_non_atomic_batch_visibility_then_text_on_same_entry(self) -> None:
        """The non-atomic loop tracks the visibility version bump identically."""
        from app.tools.batch.update import update_context_batch
        from app.tools.context.store import store_context

        stored = await store_context(
            thread_id='access-tools', source='agent', text='non-atomic visibility-then-text target',
        )
        result = await update_context_batch(
            updates=[
                {'context_id': stored['context_id'], 'visibility': 'public'},
                {'context_id': stored['context_id'], 'text': 'updated after publish'},
            ],
            atomic=False,
        )
        assert result['succeeded'] == 2, result
        _, visibility = await _read_owner_visibility(stored['context_id'])
        assert visibility == 'public'


@pytest.mark.usefixtures('initialized_server')
class TestBatchVisibilityValidation:
    """Batch entries validate the visibility enum per entry."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('visibility', ['everyone', 'shared'])
    async def test_invalid_visibility_fails_only_that_entry(self, visibility: str) -> None:
        """Non-atomic: a visibility value other than private or public records a per-entry error."""
        from app.tools.batch.store import store_context_batch

        result = await store_context_batch(
            entries=[
                {'thread_id': 'access-tools', 'source': 'agent', 'text': 'good entry'},
                {
                    'thread_id': 'access-tools', 'source': 'agent',
                    'text': f'bad visibility entry {visibility}', 'visibility': visibility,
                },
            ],
            atomic=False,
        )
        assert result['succeeded'] == 1
        assert result['failed'] == 1
        errors = [r['error'] for r in result['results'] if r['error']]
        assert errors == ["visibility must be one of 'private', 'public'"]

    @pytest.mark.asyncio
    async def test_batch_per_entry_visibility_is_stamped(self) -> None:
        """A per-entry visibility value lands on that entry's row."""
        from app.tools.batch.store import store_context_batch

        result = await store_context_batch(entries=[{
            'thread_id': 'access-tools', 'source': 'agent',
            'text': 'public batch entry', 'visibility': 'public',
        }])
        cid = result['results'][0]['context_id']
        assert cid is not None
        _, visibility = await _read_owner_visibility(cid)
        assert visibility == 'public'


@pytest.mark.usefixtures('initialized_server')
class TestAuthorGroupGrants:
    """ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups writes read grants."""

    @pytest.mark.asyncio
    async def test_author_groups_write_read_grants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each author group receives a read grant in the store transaction."""
        import app.tools.context.store as context_store_module
        from app.tools.context.store import store_context

        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_GROUP_GRANTS', 'author_groups')
        get_settings.cache_clear()
        monkeypatch.setattr(context_store_module, 'settings', get_settings())
        try:
            author = _principal('alice', groups=frozenset({'team-b', 'team-a'}))
            with patch('app.tools.context.store.resolve_effective_principal', return_value=author):
                result = await store_context(
                    thread_id='access-tools', source='agent', text='group-granted entry',
                )
        finally:
            get_settings.cache_clear()

        backend = app.startup.get_backend()
        assert backend is not None
        grants = await read_grants(backend, result['context_id'])
        assert grants == [
            ('group', 'team-a', 'read', 'alice'),
            ('group', 'team-b', 'read', 'alice'),
        ]

    @pytest.mark.asyncio
    async def test_default_none_writes_no_grants(self) -> None:
        """With the default policy no grant rows are written."""
        from app.tools.context.store import store_context

        author = _principal('alice', groups=frozenset({'team-a'}))
        with patch('app.tools.context.store.resolve_effective_principal', return_value=author):
            result = await store_context(
                thread_id='access-tools', source='agent', text='ungranted entry',
            )

        backend = app.startup.get_backend()
        assert backend is not None
        grants = await read_grants(backend, result['context_id'])
        assert grants == []
