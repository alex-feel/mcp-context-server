"""Tool-level tests for share_context, unshare_context and list_context_grants (SQLite backend).

Only an entry's owner shares it, removes a grant from it and lists its grants: an entry
the caller may not read is reported exactly like an absent one, and an entry the caller
may read but does not own is refused explicitly. Each permission is its own grant row,
grants stay outside the entry's version check, and they go away with their entry.
"""

import os
import re
import sqlite3
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from typing import Literal
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.startup
import app.tools.context.store as context_store_module
from app.access_scope import Scope
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.repositories import RepositoryContainer
from app.repositories.context_repository.records import EntryProbe
from app.settings import get_settings
from app.tools.context.delete import delete_context
from app.tools.context.retrieve import get_context_by_ids
from app.tools.context.store import store_context
from app.tools.context.update import update_context
from app.tools.sharing import list_context_grants
from app.tools.sharing import share_context
from app.tools.sharing import unshare_context
from tests.helpers import as_principal
from tests.helpers import insert_grant
from tests.helpers import read_grants
from tests.tools._read_only import read_only_caller

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('initialized_server')]

ABSENT_ID = '0190abcdef1234567890abcdef123456'

type Visibility = Literal['private', 'public']
type PrincipalType = Literal['user', 'group']
type Permission = Literal['read', 'write']
type GrantChangeTool = Callable[..., Awaitable[object]]


def _backend() -> StorageBackend:
    backend = app.startup.get_backend()
    assert backend is not None
    return backend


def _repositories() -> RepositoryContainer:
    repos = app.startup.get_repositories()
    assert repos is not None
    return repos


async def _store(owner: str, text: str, *, visibility: Visibility = 'private') -> str:
    """Store one entry as ``owner`` and return its id."""
    with as_principal(owner):
        result = await store_context(thread_id='sharing', source='agent', text=text, visibility=visibility)
    return result['context_id']


async def _version_and_updated_at(context_id: str) -> tuple[int, str]:
    """Read an entry's version token and update time through raw SQL."""

    def _read(conn: sqlite3.Connection) -> tuple[int, str]:
        row = conn.execute('SELECT version, updated_at FROM context_entries WHERE id = ?', (context_id,)).fetchone()
        assert row is not None
        return int(row[0]), str(row[1])

    return await _backend().execute_read(_read, access_scope=MAINTENANCE)


@contextmanager
def _anonymous_principal_named() -> Generator[None, None, None]:
    """Name ``anonymous`` as the read-only principal in freshly built settings inside the block."""
    with patch.dict(os.environ, {'ACCESS_CONTROL_ANONYMOUS_PRINCIPAL': 'anonymous'}):
        get_settings.cache_clear()
        try:
            yield
        finally:
            get_settings.cache_clear()


class TestShareContext:
    """share_context grants one permission to one user or group, once."""

    async def test_share_creates_the_grant_then_reports_it_exists(self) -> None:
        cid = await _store('alice', 'shared with bob')

        with as_principal('alice'):
            first = await share_context(context_id=cid, principal_type='user', principal_id='bob', permission='read')
            second = await share_context(context_id=cid, principal_type='user', principal_id='bob', permission='read')

        assert first == {
            'success': True,
            'context_id': cid,
            'principal_type': 'user',
            'principal_id': 'bob',
            'permission': 'read',
            'created': True,
            'message': f'Granted read on context entry {cid} to user bob',
        }
        assert second == {**first, 'created': False, 'message': f'user bob already holds read on context entry {cid}'}
        assert await read_grants(_backend(), cid) == [('user', 'bob', 'read', 'alice')]

    async def test_share_resolves_an_id_prefix(self) -> None:
        cid = await _store('alice', 'shared by prefix')

        with as_principal('alice'):
            result = await share_context(
                context_id=cid[:10], principal_type='group', principal_id='team-x', permission='write',
            )

        assert result['context_id'] == cid
        assert await read_grants(_backend(), cid) == [('group', 'team-x', 'write', 'alice')]

    async def test_owner_may_grant_a_group_it_belongs_to(self) -> None:
        cid = await _store('alice', 'shared with my own group')

        with as_principal('alice', groups=['team-x']):
            result = await share_context(
                context_id=cid, principal_type='group', principal_id='team-x', permission='read',
            )

        assert result['created'] is True
        assert await read_grants(_backend(), cid) == [('group', 'team-x', 'read', 'alice')]

    async def test_read_and_write_are_separate_grants(self) -> None:
        cid = await _store('alice', 'read beside write')

        with as_principal('alice'):
            await share_context(context_id=cid, principal_type='user', principal_id='bob', permission='write')
            read = await share_context(context_id=cid, principal_type='user', principal_id='bob', permission='read')

        assert read['created'] is True
        assert await read_grants(_backend(), cid) == [('user', 'bob', 'read', 'alice'), ('user', 'bob', 'write', 'alice')]

    async def test_reshare_of_an_author_group_grant_reports_it_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_GROUP_GRANTS', 'author_groups')
        get_settings.cache_clear()
        monkeypatch.setattr(context_store_module, 'settings', get_settings())
        try:
            with as_principal('alice', groups=['team-a']):
                stored = await store_context(thread_id='sharing', source='agent', text='author group grant')
                result = await share_context(
                    context_id=stored['context_id'], principal_type='group', principal_id='team-a', permission='read',
                )
        finally:
            get_settings.cache_clear()

        assert result['created'] is False
        assert await read_grants(_backend(), stored['context_id']) == [('group', 'team-a', 'read', 'alice')]

    async def test_owner_naming_itself_is_refused_and_nothing_is_written(self) -> None:
        cid = await _store('alice', 'no self share')

        with as_principal('alice'), pytest.raises(ToolError) as error:
            await share_context(context_id=cid, principal_type='user', principal_id='alice', permission='write')

        assert str(error.value) == f'Cannot share context entry with ID {cid} with its owner'
        assert await read_grants(_backend(), cid) == []

    async def test_non_owner_naming_itself_is_not_authorized(self) -> None:
        cid = await _store('alice', 'bob may only read')
        await insert_grant(_backend(), cid, 'user', 'bob', 'read', 'alice')

        with as_principal('bob'), pytest.raises(ToolError) as error:
            await share_context(context_id=cid, principal_type='user', principal_id='bob', permission='write')

        assert str(error.value) == f'Not authorized to share context entry with ID {cid}'

    async def test_user_grant_to_the_anonymous_principal_is_refused_before_id_resolution(self) -> None:
        hidden = await _store('bob', 'bob keeps this private')

        with _anonymous_principal_named(), as_principal('alice'):
            with (
                patch('app.tools.sharing.ensure_repositories', AsyncMock()) as repositories,
                pytest.raises(ToolError) as by_id,
            ):
                await share_context(context_id=hidden, principal_type='user', principal_id='anonymous', permission='read')
            with pytest.raises(ToolError) as by_prefix:
                await share_context(context_id='deadbeef', principal_type='user', principal_id='anonymous', permission='read')

        repositories.assert_not_awaited()
        assert str(by_id.value) == (
            f"Cannot share context entry with ID {hidden} with the anonymous principal; "
            "set its visibility to 'public' to publish it"
        )
        assert str(by_prefix.value) == (
            "Cannot share context entry with ID deadbeef with the anonymous principal; "
            "set its visibility to 'public' to publish it"
        )

    async def test_group_named_like_the_anonymous_principal_is_an_ordinary_group(self) -> None:
        cid = await _store('alice', 'a group called anonymous')

        with _anonymous_principal_named(), as_principal('alice'):
            result = await share_context(
                context_id=cid, principal_type='group', principal_id='anonymous', permission='read',
            )

        assert result['created'] is True


class TestPrincipalIdValidation:
    """The grantee id is checked before anything else, with an exact error per breach."""

    @pytest.mark.parametrize(
        ('principal_id', 'message'),
        [
            ('', 'principal_id cannot be empty or whitespace'),
            ('   ', 'principal_id cannot be empty or whitespace'),
            (' bob', 'principal_id cannot have leading or trailing whitespace'),
            ('bob\t', 'principal_id cannot have leading or trailing whitespace'),
            ('x' * 257, 'principal_id exceeds 256 characters'),
        ],
        ids=['empty', 'whitespace-only', 'leading-space', 'trailing-tab', 'too-long'],
    )
    @pytest.mark.parametrize('tool', [share_context, unshare_context], ids=['share', 'unshare'])
    async def test_invalid_principal_id_is_refused_before_id_resolution(
        self, tool: GrantChangeTool, principal_id: str, message: str,
    ) -> None:
        with (
            as_principal('alice'),
            patch('app.tools.sharing.ensure_repositories', AsyncMock()) as repositories,
            pytest.raises(ToolError) as error,
        ):
            await tool(context_id=ABSENT_ID, principal_type='user', principal_id=principal_id, permission='read')

        assert str(error.value) == message
        repositories.assert_not_awaited()

    async def test_unstorable_principal_id_is_refused(self) -> None:
        with as_principal('alice'), pytest.raises(ToolError) as error:
            await share_context(context_id=ABSENT_ID, principal_type='user', principal_id='bob\x00', permission='read')

        assert str(error.value).startswith('principal_id: ')

    @pytest.mark.parametrize('principal_id', ['é' * 256, 'auth0|x', 'Team-X'], ids=['256-non-ascii', 'auth0', 'case'])
    async def test_any_charset_up_to_256_code_points_is_accepted(self, principal_id: str) -> None:
        cid = await _store('alice', f'accepts {principal_id[:8]}')

        with as_principal('alice'):
            result = await share_context(context_id=cid, principal_type='user', principal_id=principal_id, permission='read')

        assert result['created'] is True
        assert await read_grants(_backend(), cid) == [('user', principal_id, 'read', 'alice')]


class TestOwnerOnly:
    """Unreadable equals absent; readable but not owned is an explicit refusal."""

    async def test_unreadable_entry_is_reported_like_an_absent_one(self) -> None:
        hidden = await _store('bob', 'bob only')

        for context_id in (hidden, ABSENT_ID):
            not_found = f'Context entry with ID {context_id} not found'
            with as_principal('alice'):
                with pytest.raises(ToolError) as shared:
                    await share_context(context_id=context_id, principal_type='user', principal_id='carol', permission='read')
                with pytest.raises(ToolError) as unshared:
                    await unshare_context(
                        context_id=context_id, principal_type='user', principal_id='carol', permission='read',
                    )
                with pytest.raises(ToolError) as listed:
                    await list_context_grants(context_id=context_id)
            assert (str(shared.value), str(unshared.value), str(listed.value)) == (not_found, not_found, not_found)
        assert await read_grants(_backend(), hidden) == []

    async def test_unreadable_prefix_matches_nothing(self) -> None:
        hidden = await _store('bob', 'bob only, by prefix')

        with as_principal('alice'), pytest.raises(ToolError) as error:
            await share_context(context_id=hidden[:12], principal_type='user', principal_id='carol', permission='read')

        assert str(error.value) == f"Invalid context ID: No context entry matches prefix '{hidden[:12]}'"

    @pytest.mark.parametrize('arrangement', ['read grant', 'write grant', 'public entry'])
    async def test_readable_entry_the_caller_does_not_own_is_refused(self, arrangement: str) -> None:
        visibility: Visibility = 'public' if arrangement == 'public entry' else 'private'
        cid = await _store('alice', f'bob reads through a {arrangement}', visibility=visibility)
        if arrangement != 'public entry':
            await insert_grant(_backend(), cid, 'user', 'bob', arrangement.split()[0], 'alice')
        await insert_grant(_backend(), cid, 'user', 'carol', 'read', 'alice')
        before = await read_grants(_backend(), cid)

        with as_principal('bob'):
            with pytest.raises(ToolError) as shared:
                await share_context(context_id=cid, principal_type='user', principal_id='dave', permission='read')
            with pytest.raises(ToolError) as unshared:
                await unshare_context(context_id=cid, principal_type='user', principal_id='carol', permission='read')
            with pytest.raises(ToolError) as listed:
                await list_context_grants(context_id=cid)

        assert str(shared.value) == f'Not authorized to share context entry with ID {cid}'
        assert str(unshared.value) == f'Not authorized to unshare context entry with ID {cid}'
        assert str(listed.value) == f'Not authorized to list the grants of context entry with ID {cid}'
        assert await read_grants(_backend(), cid) == before

    async def test_read_only_principal_may_neither_share_nor_unshare(self) -> None:
        with read_only_caller(), patch('app.tools.sharing.ensure_repositories', AsyncMock()) as repositories:
            with pytest.raises(ToolError) as shared:
                await share_context(context_id=ABSENT_ID, principal_type='user', principal_id='bob', permission='read')
            with pytest.raises(ToolError) as unshared:
                await unshare_context(context_id=ABSENT_ID, principal_type='user', principal_id='bob', permission='read')

        assert str(shared.value) == 'The anonymous principal is read-only: share_context is not permitted'
        assert str(unshared.value) == 'The anonymous principal is read-only: unshare_context is not permitted'
        repositories.assert_not_awaited()

    async def test_read_only_principal_lists_nothing_it_does_not_own(self) -> None:
        cid = await _store('alice', 'public, owned by alice', visibility='public')

        with read_only_caller(), pytest.raises(ToolError) as error:
            await list_context_grants(context_id=cid)

        assert str(error.value) == f'Not authorized to list the grants of context entry with ID {cid}'

    async def test_refusals_are_not_retried_and_never_charge_the_breaker(self) -> None:
        cid = await _store('alice', 'refusals stay cheap')
        hidden = await _store('bob', 'hidden from alice')
        breaker = cast(Any, _backend()).circuit_breaker

        with (
            patch.object(breaker, 'record_failure', wraps=breaker.record_failure) as failures,
            patch('app.tools.sharing.asyncio.sleep', AsyncMock()) as backoff,
        ):
            with as_principal('alice'):
                for context_id, grantee in ((hidden, 'carol'), (cid, 'alice')):
                    with pytest.raises(ToolError):
                        await share_context(
                            context_id=context_id, principal_type='user', principal_id=grantee, permission='read',
                        )
                with pytest.raises(ToolError):
                    await unshare_context(context_id=hidden, principal_type='user', principal_id='carol', permission='read')
            with as_principal('bob'), pytest.raises(ToolError):
                await share_context(context_id=cid, principal_type='user', principal_id='carol', permission='read')

        failures.assert_not_called()
        backoff.assert_not_awaited()

    async def test_transient_lock_is_retried(self) -> None:
        cid = await _store('alice', 'retried share')
        repos = _repositories()
        real_store_grant = repos.grants.store_grant
        attempts = 0

        async def locked_once(*args: Any, **kwargs: Any) -> bool:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise sqlite3.OperationalError('database is locked')
            return await real_store_grant(*args, **kwargs)

        with (
            as_principal('alice'),
            patch.object(repos.grants, 'store_grant', locked_once),
            patch('app.tools.sharing.asyncio.sleep', AsyncMock()) as backoff,
        ):
            result = await share_context(context_id=cid, principal_type='user', principal_id='bob', permission='read')

        assert result['created'] is True
        assert attempts == 2
        backoff.assert_awaited_once_with(0.5)
        assert await read_grants(_backend(), cid) == [('user', 'bob', 'read', 'alice')]


class TestUnshareContext:
    """unshare_context removes exactly the named grant and reports what the same key still holds."""

    async def test_unshare_removes_then_reports_nothing_to_remove(self) -> None:
        cid = await _store('alice', 'unshared twice')
        with as_principal('alice'):
            await share_context(context_id=cid, principal_type='user', principal_id='bob', permission='read')
            first = await unshare_context(context_id=cid, principal_type='user', principal_id='bob', permission='read')
            second = await unshare_context(context_id=cid, principal_type='user', principal_id='bob', permission='read')

        assert first == {
            'success': True,
            'context_id': cid,
            'principal_type': 'user',
            'principal_id': 'bob',
            'permission': 'read',
            'removed': True,
            'remaining_permissions': [],
            'message': f'Removed read on context entry {cid} from user bob',
        }
        assert second == {**first, 'removed': False, 'message': f'No read grant on context entry {cid} for user bob'}
        assert await read_grants(_backend(), cid) == []

    @pytest.mark.parametrize(
        ('granted', 'removed_permission', 'removed', 'remaining'),
        [
            (('read', 'write'), 'read', True, ['write']),
            (('read', 'write'), 'write', True, ['read']),
            (('write',), 'read', False, ['write']),
            (('read',), 'write', False, ['read']),
        ],
        ids=['read-under-write', 'write-over-read', 'absent-read-under-write', 'absent-write-over-read'],
    )
    async def test_remaining_permissions_report_the_same_key(
        self, granted: tuple[Permission, ...], removed_permission: Permission, removed: bool, remaining: list[Permission],
    ) -> None:
        cid = await _store('alice', f'remaining after {removed_permission} of {"+".join(granted)}')
        with as_principal('alice'):
            for permission in granted:
                await share_context(context_id=cid, principal_type='user', principal_id='bob', permission=permission)
            result = await unshare_context(
                context_id=cid, principal_type='user', principal_id='bob', permission=removed_permission,
            )

        assert (result['removed'], result['remaining_permissions']) == (removed, remaining)

    async def test_remaining_permissions_ignore_other_keys(self) -> None:
        cid = await _store('alice', 'a group and a user of the same name')
        with as_principal('alice'):
            await share_context(context_id=cid, principal_type='group', principal_id='bob', permission='write')
            await share_context(context_id=cid, principal_type='user', principal_id='carol', permission='read')
            await share_context(context_id=cid, principal_type='user', principal_id='bob', permission='read')
            result = await unshare_context(context_id=cid, principal_type='user', principal_id='bob', permission='read')

        assert result['remaining_permissions'] == []
        assert await read_grants(_backend(), cid) == [('group', 'bob', 'write', 'alice'), ('user', 'carol', 'read', 'alice')]

    async def test_owner_removes_grants_no_share_could_create(self) -> None:
        cid = await _store('alice', 'grants stored by other means')
        await insert_grant(_backend(), cid, 'user', 'alice', 'write', 'alice')
        await insert_grant(_backend(), cid, 'user', 'anonymous', 'read', 'alice')

        with _anonymous_principal_named(), as_principal('alice'):
            to_owner = await unshare_context(context_id=cid, principal_type='user', principal_id='alice', permission='write')
            to_anonymous = await unshare_context(
                context_id=cid, principal_type='user', principal_id='anonymous', permission='read',
            )

        assert (to_owner['removed'], to_anonymous['removed']) == (True, True)
        assert await read_grants(_backend(), cid) == []


class TestListContextGrants:
    """list_context_grants returns every grant of an owned entry in key order."""

    async def test_grants_are_listed_in_key_order_with_their_grantor(self) -> None:
        cid = await _store('alice', 'four grants')
        grants: tuple[tuple[PrincipalType, str, Permission], ...] = (
            ('user', 'carol', 'read'), ('user', 'bob', 'write'), ('group', 'team-x', 'read'), ('user', 'bob', 'read'),
        )
        with as_principal('alice'):
            for principal_type, principal_id, permission in grants:
                await share_context(
                    context_id=cid, principal_type=principal_type, principal_id=principal_id, permission=permission,
                )
            listed = await list_context_grants(context_id=cid[:9])

        assert listed['context_id'] == cid
        assert listed['count'] == 4
        assert [(g['principal_type'], g['principal_id'], g['permission'], g['granted_by']) for g in listed['grants']] == [
            ('group', 'team-x', 'read', 'alice'),
            ('user', 'bob', 'read', 'alice'),
            ('user', 'bob', 'write', 'alice'),
            ('user', 'carol', 'read', 'alice'),
        ]
        for grant in listed['grants']:
            assert set(grant) == {'principal_type', 'principal_id', 'permission', 'granted_by', 'created_at'}
            assert re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', grant['created_at'])

    async def test_entry_without_grants_lists_none(self) -> None:
        cid = await _store('alice', 'no grants yet')

        with as_principal('alice'):
            listed = await list_context_grants(context_id=cid)

        assert listed == {'context_id': cid, 'grants': [], 'count': 0}

    async def test_deleting_the_entry_deletes_its_grants(self) -> None:
        cid = await _store('alice', 'deleted with its grants')
        with as_principal('alice'):
            await share_context(context_id=cid, principal_type='user', principal_id='bob', permission='read')
            await delete_context(context_ids=[cid])
            with pytest.raises(ToolError) as error:
                await list_context_grants(context_id=cid)

        assert str(error.value) == f'Context entry with ID {cid} not found'
        assert await read_grants(_backend(), cid) == []


class TestGrantsOutsideTheVersionCheck:
    """Grants change neither the entry's version nor its update time, so they never conflict with an update."""

    async def test_share_and_unshare_leave_version_and_updated_at_unchanged(self) -> None:
        cid = await _store('alice', 'version stays put')

        # The stamps have whole-second precision, so the entry is backdated: a grant change that
        # stamped updated_at would then show even when it runs within the second of the store.
        def _backdate(conn: sqlite3.Connection) -> None:
            conn.execute("UPDATE context_entries SET updated_at = '2000-01-01 00:00:00' WHERE id = ?", (cid,))

        await _backend().execute_write(_backdate, access_scope=MAINTENANCE)

        async def observed() -> tuple[int, str, str]:
            version, raw_updated_at = await _version_and_updated_at(cid)
            with as_principal('alice'):
                entries = await get_context_by_ids(context_ids=[cid])
            return version, raw_updated_at, cast(dict[str, Any], entries[0])['updated_at']

        before = await observed()
        assert before[1] == '2000-01-01 00:00:00'
        assert before[2] == '2000-01-01T00:00:00Z'
        with as_principal('alice'):
            await share_context(context_id=cid, principal_type='user', principal_id='bob', permission='write')
        after_share = await observed()
        with as_principal('alice'):
            await unshare_context(context_id=cid, principal_type='user', principal_id='bob', permission='write')
        after_unshare = await observed()

        assert before == after_share == after_unshare

    async def test_update_that_read_its_version_before_a_share_commits_without_a_conflict(self) -> None:
        cid = await _store('alice', 'updated while shared')
        repos = _repositories()
        real_probe = repos.context.check_entry_exists

        async def probe_then_share(context_id: str, *, scope: Scope) -> EntryProbe:
            probe = await real_probe(context_id, scope=scope)
            await share_context(context_id=context_id, principal_type='user', principal_id='bob', permission='read')
            return probe

        with (
            as_principal('alice'),
            patch.object(repos.context, 'check_entry_exists', probe_then_share),
            patch('app.tools.context.update.reread_entry_version', AsyncMock()) as reread,
        ):
            result = await update_context(context_id=cid, metadata={'state': 'after share'})

        reread.assert_not_awaited()
        assert result['updated_fields'] == ['metadata']
        assert await read_grants(_backend(), cid) == [('user', 'bob', 'read', 'alice')]
        with as_principal('alice'):
            entries = await get_context_by_ids(context_ids=[cid])
        assert cast(dict[str, Any], entries[0])['metadata'] == {'state': 'after share'}
