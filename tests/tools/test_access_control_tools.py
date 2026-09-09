"""Tests for access-control behavior at the MCP tool boundary.

Covers owner stamping through store_context / store_context_batch (default
principal fallback and verified-principal stamping), the publish gate on both
store and update, the owner-only visibility change on update_context /
update_context_batch, author-group grant stamping, and the invariant that
owner_id is never a tool parameter.
"""

import inspect
import sqlite3
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.startup
from app.auth.principal import RequestPrincipal
from app.repositories.grant_repository import GrantRepository
from app.repositories.grant_repository import GrantRow
from app.settings import get_settings


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


@pytest.mark.usefixtures('initialized_server')
class TestOwnerStamping:
    """store paths stamp the server-resolved owner, never a caller value."""

    @pytest.mark.asyncio
    async def test_store_without_token_stamps_default_principal(self) -> None:
        """With no verified token the configured default principal owns the row."""
        from app.tools.context import store_context

        result = await store_context(
            thread_id='access-tools', source='agent', text='default-principal entry',
        )
        owner, visibility = await _read_owner_visibility(result['context_id'])
        assert owner == get_settings().access_control.default_principal
        assert visibility == 'private'

    @pytest.mark.asyncio
    async def test_store_stamps_verified_principal(self) -> None:
        """A verified principal becomes the row owner."""
        from app.tools.context import store_context

        with patch('app.tools.context.resolve_effective_principal', return_value=_principal('alice')):
            result = await store_context(
                thread_id='access-tools', source='agent', text='alice-owned entry',
            )
        owner, _ = await _read_owner_visibility(result['context_id'])
        assert owner == 'alice'

    @pytest.mark.asyncio
    async def test_batch_entry_owner_id_key_is_ignored(self) -> None:
        """A caller-supplied owner_id key in a batch entry never reaches the row."""
        from app.tools.batch import store_context_batch

        with patch('app.tools.batch.resolve_effective_principal', return_value=_principal('alice')):
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

    def test_owner_id_is_never_a_tool_parameter(self) -> None:
        """No write tool exposes owner_id in its signature (wire schema source)."""
        from app.tools.batch import store_context_batch
        from app.tools.batch import update_context_batch
        from app.tools.context import store_context
        from app.tools.context import update_context

        for tool in (store_context, update_context, store_context_batch, update_context_batch):
            assert 'owner_id' not in inspect.signature(tool).parameters


@pytest.mark.usefixtures('initialized_server')
class TestPublishGate:
    """ACCESS_CONTROL_PUBLISH_ROLE gates visibility 'public' on store and update."""

    @pytest.mark.asyncio
    async def test_store_public_denied_without_role(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Publishing without the configured role fails before any storage."""
        from app.tools.context import store_context

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
        from app.tools.context import store_context

        monkeypatch.setenv('ACCESS_CONTROL_PUBLISH_ROLE', 'publisher')
        get_settings.cache_clear()
        try:
            publisher = _principal('alice', roles=frozenset({'publisher'}))
            with patch('app.tools.context.resolve_effective_principal', return_value=publisher):
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
        from app.tools.context import store_context

        result = await store_context(
            thread_id='access-tools', source='agent',
            text='ungated publish', visibility='public',
        )
        _, visibility = await _read_owner_visibility(result['context_id'])
        assert visibility == 'public'

    @pytest.mark.asyncio
    async def test_update_public_denied_without_role(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Updating visibility to public is gated exactly like storing it."""
        from app.tools.context import store_context
        from app.tools.context import update_context

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
        from app.tools.context import store_context
        from app.tools.context import update_context

        result = await store_context(
            thread_id='access-tools', source='agent', text='owner visibility change',
        )
        updated = await update_context(context_id=result['context_id'], visibility='shared')
        assert 'visibility' in updated['updated_fields']
        _, visibility = await _read_owner_visibility(result['context_id'])
        assert visibility == 'shared'

    @pytest.mark.asyncio
    async def test_non_owner_visibility_change_rejected(self) -> None:
        """A different principal cannot change visibility."""
        from app.tools.context import store_context
        from app.tools.context import update_context

        with patch('app.tools.context.resolve_effective_principal', return_value=_principal('alice')):
            result = await store_context(
                thread_id='access-tools', source='agent', text='alice-only visibility',
            )
        with (
            patch('app.tools.context.resolve_effective_principal', return_value=_principal('bob')),
            pytest.raises(ToolError, match='Only the owner'),
        ):
            await update_context(context_id=result['context_id'], visibility='public')

    @pytest.mark.asyncio
    async def test_non_owner_text_update_still_allowed(self) -> None:
        """A text-only update carries no visibility change and is not owner-gated
        (read/write scoping arrives with read-path enforcement)."""
        from app.tools.context import store_context
        from app.tools.context import update_context

        with patch('app.tools.context.resolve_effective_principal', return_value=_principal('alice')):
            result = await store_context(
                thread_id='access-tools', source='agent', text='text update target',
            )
        with patch('app.tools.context.resolve_effective_principal', return_value=_principal('bob')):
            updated = await update_context(context_id=result['context_id'], text='new body')
        assert 'text_content' in updated['updated_fields']

    @pytest.mark.asyncio
    async def test_batch_non_owner_visibility_change_records_per_entry_error(self) -> None:
        """Non-atomic batch: an unauthorized visibility change fails only that entry."""
        from app.tools.batch import update_context_batch
        from app.tools.context import store_context

        with patch('app.tools.context.resolve_effective_principal', return_value=_principal('alice')):
            result = await store_context(
                thread_id='access-tools', source='agent', text='batch visibility target',
            )
        with patch('app.tools.batch.resolve_effective_principal', return_value=_principal('bob')):
            batch_result = await update_context_batch(
                updates=[{'context_id': result['context_id'], 'visibility': 'public'}],
                atomic=False,
            )
        assert batch_result['failed'] == 1
        assert 'Only the owner' in (batch_result['results'][0]['error'] or '')
        _, visibility = await _read_owner_visibility(result['context_id'])
        assert visibility == 'private'

    @pytest.mark.asyncio
    async def test_atomic_batch_non_owner_visibility_change_aborts(self) -> None:
        """Atomic batch: an unauthorized visibility change aborts the whole batch."""
        from app.tools.batch import update_context_batch
        from app.tools.context import store_context

        with patch('app.tools.context.resolve_effective_principal', return_value=_principal('alice')):
            result = await store_context(
                thread_id='access-tools', source='agent', text='atomic batch visibility target',
            )
        with (
            patch('app.tools.batch.resolve_effective_principal', return_value=_principal('bob')),
            pytest.raises(ToolError, match='Only the owner'),
        ):
            await update_context_batch(
                updates=[{'context_id': result['context_id'], 'visibility': 'public'}],
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
        from app.tools.batch import update_context_batch
        from app.tools.context import store_context

        stored = await store_context(
            thread_id='access-tools', source='agent', text='visibility-then-text target',
        )
        result = await update_context_batch(
            updates=[
                {'context_id': stored['context_id'], 'visibility': 'shared'},
                {'context_id': stored['context_id'], 'text': 'updated after visibility change'},
            ],
            atomic=True,
        )
        assert result['succeeded'] == 2, result
        _, visibility = await _read_owner_visibility(stored['context_id'])
        assert visibility == 'shared'

    @pytest.mark.asyncio
    async def test_non_atomic_batch_visibility_then_text_on_same_entry(self) -> None:
        """The non-atomic loop tracks the visibility version bump identically."""
        from app.tools.batch import update_context_batch
        from app.tools.context import store_context

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
    async def test_invalid_visibility_fails_only_that_entry(self) -> None:
        """Non-atomic: an invalid visibility value records a per-entry error."""
        from app.tools.batch import store_context_batch

        result = await store_context_batch(
            entries=[
                {'thread_id': 'access-tools', 'source': 'agent', 'text': 'good entry'},
                {
                    'thread_id': 'access-tools', 'source': 'agent',
                    'text': 'bad visibility entry', 'visibility': 'everyone',
                },
            ],
            atomic=False,
        )
        assert result['succeeded'] == 1
        assert result['failed'] == 1
        errors = [r['error'] for r in result['results'] if r['error']]
        assert any('visibility' in e for e in errors)

    @pytest.mark.asyncio
    async def test_batch_per_entry_visibility_is_stamped(self) -> None:
        """A per-entry visibility value lands on that entry's row."""
        from app.tools.batch import store_context_batch

        result = await store_context_batch(entries=[{
            'thread_id': 'access-tools', 'source': 'agent',
            'text': 'shared batch entry', 'visibility': 'shared',
        }])
        cid = result['results'][0]['context_id']
        assert cid is not None
        _, visibility = await _read_owner_visibility(cid)
        assert visibility == 'shared'


@pytest.mark.usefixtures('initialized_server')
class TestAuthorGroupGrants:
    """ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups writes read grants."""

    @pytest.mark.asyncio
    async def test_author_groups_write_read_grants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each author group receives a read grant in the store transaction."""
        import app.tools.context as context_module
        from app.tools.context import store_context

        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_GROUP_GRANTS', 'author_groups')
        get_settings.cache_clear()
        monkeypatch.setattr(context_module, 'settings', get_settings())
        try:
            author = _principal('alice', groups=frozenset({'team-b', 'team-a'}))
            with patch('app.tools.context.resolve_effective_principal', return_value=author):
                result = await store_context(
                    thread_id='access-tools', source='agent', text='group-granted entry',
                )
        finally:
            get_settings.cache_clear()

        backend = app.startup.get_backend()
        assert backend is not None
        grants = await GrantRepository(backend).get_grants_for_context(result['context_id'])
        assert grants == [
            GrantRow('group', 'team-a', 'read', 'alice'),
            GrantRow('group', 'team-b', 'read', 'alice'),
        ]

    @pytest.mark.asyncio
    async def test_default_none_writes_no_grants(self) -> None:
        """With the default policy no grant rows are written."""
        from app.tools.context import store_context

        author = _principal('alice', groups=frozenset({'team-a'}))
        with patch('app.tools.context.resolve_effective_principal', return_value=author):
            result = await store_context(
                thread_id='access-tools', source='agent', text='ungranted entry',
            )

        backend = app.startup.get_backend()
        assert backend is not None
        grants = await GrantRepository(backend).get_grants_for_context(result['context_id'])
        assert grants == []
