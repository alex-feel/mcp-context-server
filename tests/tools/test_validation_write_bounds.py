"""Tests for the write-path input bounds enforced through ``app.tools._validation``: the tag count and length
caps, the thread_id and indexed metadata value length caps, and the cast compatibility of typed indexed
metadata fields, on the single-entry and batch write tools.
"""

from collections.abc import Callable
from collections.abc import Iterator
from typing import get_args
from typing import get_type_hints
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from annotated_types import MaxLen
from fastmcp.exceptions import ToolError
from pydantic.fields import FieldInfo

import app.tools


class TestWritePathTagCaps:
    """The write path bounds tags in BOTH dimensions: list length and per-tag length.

    Without the caps, a single over-long tag stores happily on SQLite and
    hard-fails on PostgreSQL, where ``idx_tags_tag`` is a btree whose index-tuple
    ceiling (~2704 bytes) rejects it -- inside the store transaction, after the whole
    generation pass, and while charging the circuit breaker. An unbounded tag COUNT
    is a second hazard: tags are inserted one row per tag inside that same
    transaction, and every later search response re-hydrates the entry's full tag
    list, so one poisoned entry inflates every subsequent read forever.

    The caps are advertised in the wire schema of the typed single-entry tools and
    enforced in the shared chokepoint that the untyped batch tools also use.
    """

    @staticmethod
    def _tags_annotation(fn: Callable[..., object]) -> object:
        """Return the annotation attached to a tool's ``tags`` parameter.

        Returns:
            The raw (Annotated) annotation object for the parameter.
        """
        return get_type_hints(fn, include_extras=True)['tags']

    @pytest.mark.parametrize('tool_name', ['store_context', 'update_context'])
    def test_tags_declares_both_caps_in_wire_schema(self, tool_name: str) -> None:
        """Both write tools advertise maxItems and per-item maxLength."""
        from app.models import MAX_TAG_LENGTH
        from app.models import MAX_TAGS_PER_ENTRY

        annotation = self._tags_annotation(getattr(app.tools, tool_name))
        outer_meta = [m for m in get_args(annotation)[1:] if isinstance(m, FieldInfo)]
        assert outer_meta, f'{tool_name} tags carries no FieldInfo'
        outer_caps = [c.max_length for c in outer_meta[0].metadata if isinstance(c, MaxLen)]
        assert outer_caps == [MAX_TAGS_PER_ENTRY]

        # The list ITEM type carries the per-tag length cap. The declared type is
        # ``list[Annotated[str, Field(...)]] | None``, so unwrap the optional union,
        # then the list, then the item annotation.
        optional_union = get_args(annotation)[0]
        list_type = get_args(optional_union)[0]
        item_annotation = get_args(list_type)[0]
        assert get_args(item_annotation)[0] is str
        item_meta = [m for m in get_args(item_annotation)[1:] if isinstance(m, FieldInfo)]
        assert item_meta, f'{tool_name} tags items carry no FieldInfo'
        item_caps = [c.max_length for c in item_meta[0].metadata if isinstance(c, MaxLen)]
        assert item_caps == [MAX_TAG_LENGTH]

    @pytest.mark.asyncio
    async def test_store_context_rejects_over_long_tag_before_generation(self) -> None:
        """An over-long tag is refused in the validation phase, before any DB work."""
        from app.models import MAX_TAG_LENGTH

        ensure_repos = AsyncMock()
        with (
            patch('app.tools.context.store.ensure_repositories', ensure_repos),
            pytest.raises(ToolError, match='Tag 0 is too long'),
        ):
            await app.tools.store_context(
                thread_id='tag-cap-thread',
                source='user',
                text='body',
                tags=['t' * (MAX_TAG_LENGTH + 1)],
            )
        ensure_repos.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_context_rejects_too_many_tags_before_generation(self) -> None:
        """A tag list above the count cap is refused before any DB work."""
        from app.models import MAX_TAGS_PER_ENTRY

        ensure_repos = AsyncMock()
        with (
            patch('app.tools.context.store.ensure_repositories', ensure_repos),
            pytest.raises(ToolError, match='Too many tags'),
        ):
            await app.tools.store_context(
                thread_id='tag-cap-thread',
                source='user',
                text='body',
                tags=[f'tag-{i}' for i in range(MAX_TAGS_PER_ENTRY + 1)],
            )
        ensure_repos.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_context_rejects_over_long_tag_before_generation(self) -> None:
        """update_context applies the identical caps."""
        from app.models import MAX_TAG_LENGTH

        ensure_repos = AsyncMock()
        with (
            patch('app.tools.context.update.ensure_repositories', ensure_repos),
            pytest.raises(ToolError, match='Tag 0 is too long'),
        ):
            await app.tools.update_context(
                context_id='0190abcdef1234567890abcdef123456',
                tags=['t' * (MAX_TAG_LENGTH + 1)],
            )
        ensure_repos.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_batch_rejects_over_long_tag_per_entry(self) -> None:
        """The untyped batch path enforces the same caps, as a per-entry error.

        Without this the batch path is a bypass: its per-entry dicts never pass
        through the wire schema that bounds the single-entry tools.
        """
        from app.models import MAX_TAG_LENGTH
        from app.tools.batch.store import store_context_batch

        result = await store_context_batch(
            entries=[{
                'thread_id': 'tag-cap-batch',
                'source': 'user',
                'text': 'body',
                'tags': ['t' * (MAX_TAG_LENGTH + 1)],
            }],
            atomic=False,
        )

        assert result['succeeded'] == 0
        assert result['failed'] == 1
        error = result['results'][0]['error']
        assert error is not None
        assert 'Tag 0 is too long' in error

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_batch_rejects_too_many_tags_per_entry(self) -> None:
        """The batch update path enforces the count cap as a per-entry error."""
        from app.models import MAX_TAGS_PER_ENTRY
        from app.tools.batch.update import update_context_batch

        result = await update_context_batch(
            updates=[{
                'context_id': '0190abcdef1234567890abcdef123456',
                'tags': [f'tag-{i}' for i in range(MAX_TAGS_PER_ENTRY + 1)],
            }],
            atomic=False,
        )

        assert result['succeeded'] == 0
        error = result['results'][0]['error']
        assert error is not None
        assert 'Too many tags' in error

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_accepts_tags_at_both_caps(self) -> None:
        """Exactly at the caps is accepted (the bounds are inclusive)."""
        from app.models import MAX_TAG_LENGTH
        from app.models import MAX_TAGS_PER_ENTRY

        at_cap = [f'{i:03d}'.ljust(MAX_TAG_LENGTH, 'x') for i in range(MAX_TAGS_PER_ENTRY)]
        result = await app.tools.store_context(
            thread_id='tag-cap-accepted',
            source='user',
            text='body at cap',
            tags=at_cap,
        )
        assert result['success'] is True


class TestIndexedWriteValueUpperBound:
    """thread_id and indexed metadata values are bounded on the write path.

    ``tags`` is not the only client value that lands in a PostgreSQL btree index.
    ``thread_id`` feeds four of them (idx_thread_id, idx_thread_source,
    idx_context_entries_dedup_hash, idx_thread_created) and every value stored under
    a METADATA_INDEXED_FIELDS key feeds that field's expression index
    (idx_metadata_<field> on metadata->>'<field>'). Unbounded, an oversized value
    stores on SQLite and aborts the PostgreSQL INSERT with 'index row size ...
    exceeds btree version 4 maximum 2704' -- inside the store transaction, after the
    whole generation pass, and while charging the circuit breaker.

    Values under NON-indexed metadata keys stay uncapped: jsonb has no such limit and
    the always-present GIN index uses jsonb_path_ops, which hashes its entries.
    """

    @pytest.mark.asyncio
    async def test_store_context_rejects_over_long_thread_id(self) -> None:
        """An over-long thread_id is refused in the validation phase, before any DB work."""
        from app.models import MAX_THREAD_ID_LENGTH

        ensure_repos = AsyncMock()
        with (
            patch('app.tools.context.store.ensure_repositories', ensure_repos),
            pytest.raises(ToolError, match='thread_id is too long'),
        ):
            await app.tools.store_context(
                thread_id='t' * (MAX_THREAD_ID_LENGTH + 1),
                source='user',
                text='body',
            )
        ensure_repos.assert_not_awaited()

    def test_store_context_advertises_the_thread_id_cap_in_the_wire_schema(self) -> None:
        """The MCP schema carries maxLength so a client sees the bound up front."""
        from app.models import MAX_THREAD_ID_LENGTH

        annotation = get_type_hints(app.tools.store_context, include_extras=True)['thread_id']
        meta = [m for m in get_args(annotation)[1:] if isinstance(m, FieldInfo)]
        assert meta, 'store_context thread_id carries no FieldInfo'
        caps = [c.max_length for c in meta[0].metadata if isinstance(c, MaxLen)]
        assert caps == [MAX_THREAD_ID_LENGTH]

    @pytest.mark.asyncio
    async def test_store_context_rejects_over_long_indexed_metadata_value(self) -> None:
        """An indexed metadata field's value is bounded like the tag and thread_id."""
        from app.models import MAX_INDEXED_METADATA_VALUE_LENGTH

        ensure_repos = AsyncMock()
        with (
            patch('app.tools.context.store.ensure_repositories', ensure_repos),
            pytest.raises(ToolError, match="metadata field 'status' is indexed"),
        ):
            await app.tools.store_context(
                thread_id='indexed-cap-thread',
                source='user',
                text='body',
                metadata={'status': 'x' * (MAX_INDEXED_METADATA_VALUE_LENGTH + 1)},
            )
        ensure_repos.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_accepts_a_long_unindexed_metadata_value(self) -> None:
        """A non-indexed key is NOT capped: jsonb and the GIN index both accept it."""
        from app.models import MAX_INDEXED_METADATA_VALUE_LENGTH

        result = await app.tools.store_context(
            thread_id='unindexed-cap-thread',
            source='user',
            text='body',
            metadata={'free_form_note': 'x' * (MAX_INDEXED_METADATA_VALUE_LENGTH + 1)},
        )
        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_update_context_rejects_over_long_indexed_metadata_patch_value(self) -> None:
        """The merge-patch form reaches the same expression index, so it is capped too."""
        from app.models import MAX_INDEXED_METADATA_VALUE_LENGTH

        ensure_repos = AsyncMock()
        with (
            patch('app.tools.context.update.ensure_repositories', ensure_repos),
            pytest.raises(ToolError, match="metadata field 'project' is indexed"),
        ):
            await app.tools.update_context(
                context_id='0190abcdef1234567890abcdef123456',
                metadata_patch={'project': 'p' * (MAX_INDEXED_METADATA_VALUE_LENGTH + 1)},
            )
        ensure_repos.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_batch_rejects_over_long_thread_id_per_entry(self) -> None:
        """The untyped batch path enforces the same bounds, as a per-entry error."""
        from app.models import MAX_THREAD_ID_LENGTH
        from app.tools.batch.store import store_context_batch

        result = await store_context_batch(
            entries=[{
                'thread_id': 't' * (MAX_THREAD_ID_LENGTH + 1),
                'source': 'user',
                'text': 'body',
            }],
            atomic=False,
        )

        assert result['succeeded'] == 0
        error = result['results'][0]['error']
        assert error is not None
        assert 'thread_id is too long' in error

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_batch_rejects_over_long_indexed_metadata_per_entry(self) -> None:
        """The batch store path bounds indexed metadata values per entry."""
        from app.models import MAX_INDEXED_METADATA_VALUE_LENGTH
        from app.tools.batch.store import store_context_batch

        result = await store_context_batch(
            entries=[{
                'thread_id': 'indexed-cap-batch',
                'source': 'user',
                'text': 'body',
                'metadata': {'agent_name': 'a' * (MAX_INDEXED_METADATA_VALUE_LENGTH + 1)},
            }],
            atomic=False,
        )

        assert result['succeeded'] == 0
        error = result['results'][0]['error']
        assert error is not None
        assert "metadata field 'agent_name' is indexed" in error

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_batch_rejects_over_long_indexed_metadata_per_entry(self) -> None:
        """The batch update path bounds indexed metadata values per entry."""
        from app.models import MAX_INDEXED_METADATA_VALUE_LENGTH
        from app.tools.batch.update import update_context_batch

        result = await update_context_batch(
            updates=[{
                'context_id': '0190abcdef1234567890abcdef123456',
                'metadata': {'task_name': 'n' * (MAX_INDEXED_METADATA_VALUE_LENGTH + 1)},
            }],
            atomic=False,
        )

        assert result['succeeded'] == 0
        error = result['results'][0]['error']
        assert error is not None
        assert "metadata field 'task_name' is indexed" in error


class TestTypedIndexedMetadataCastCompatibility:
    """A typed indexed metadata field is validated before any generation runs.

    METADATA_INDEXED_FIELDS supports integer/boolean/float type hints, and on
    PostgreSQL each becomes a hard cast inside the field's expression index
    (``((metadata->>'priority')::INTEGER)``), evaluated on every INSERT. Without a
    boundary check, a cast-incompatible value pays the full embedding + summary +
    compression pass and then aborts the transaction with a raw driver error that
    charges the circuit breaker for what is purely client input -- while the
    identical request stores fine on SQLite, whose json_extract index applies no
    cast. The four write tools refuse it at the boundary, so both backends behave
    identically.
    """

    @pytest.fixture(autouse=True)
    def typed_indexed_fields(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """Configure an integer-typed indexed metadata field for the whole class.

        Yields:
            Control to the test body, then drops the patched settings singleton.
        """
        import app.tools._validation as validation_module
        from app.settings import get_settings

        monkeypatch.setenv('METADATA_INDEXED_FIELDS', 'status,priority:integer')
        get_settings.cache_clear()
        monkeypatch.setattr(validation_module, 'settings', get_settings())
        yield
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_store_context_rejects_uncastable_indexed_value(self) -> None:
        """The rejection happens in the validation phase, before any DB or model work."""
        ensure_repos = AsyncMock()
        with (
            patch('app.tools.context.store.ensure_repositories', ensure_repos),
            pytest.raises(ToolError, match="metadata field 'priority'"),
        ):
            await app.tools.store_context(
                thread_id='typed-index-thread',
                source='user',
                text='body',
                metadata={'priority': 'high'},
            )
        ensure_repos.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_context_rejects_out_of_range_indexed_value(self) -> None:
        """A number too large for a 32-bit INTEGER is refused with a range message."""
        ensure_repos = AsyncMock()
        with (
            patch('app.tools.context.store.ensure_repositories', ensure_repos),
            pytest.raises(ToolError, match='out of range'),
        ):
            await app.tools.store_context(
                thread_id='typed-index-thread',
                source='user',
                text='body',
                metadata={'priority': 99999999999},
            )
        ensure_repos.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_context_accepts_castable_indexed_value(self) -> None:
        """A value the cast accepts still stores, including its string spelling."""
        result = await app.tools.store_context(
            thread_id='typed-index-thread-ok',
            source='user',
            text='body',
            metadata={'priority': '7'},
        )
        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_update_context_rejects_uncastable_indexed_patch_value(self) -> None:
        """The merge-patch form reaches the same expression index, so it is checked too."""
        ensure_repos = AsyncMock()
        with (
            patch('app.tools.context.update.ensure_repositories', ensure_repos),
            pytest.raises(ToolError, match="metadata field 'priority'"),
        ):
            await app.tools.update_context(
                context_id='0190abcdef1234567890abcdef123456',
                metadata_patch={'priority': 'urgent'},
            )
        ensure_repos.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_store_batch_rejects_uncastable_indexed_value_per_entry(self) -> None:
        """The untyped batch path records it as that entry's error, not a batch abort."""
        from app.tools.batch.store import store_context_batch

        result = await store_context_batch(
            entries=[{
                'thread_id': 'typed-index-batch',
                'source': 'user',
                'text': 'body',
                'metadata': {'priority': 'high'},
            }],
            atomic=False,
        )

        assert result['succeeded'] == 0
        error = result['results'][0]['error']
        assert error is not None
        assert "metadata field 'priority'" in error

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('initialized_server')
    async def test_update_batch_rejects_uncastable_indexed_value_per_entry(self) -> None:
        """Same per-entry boundary error on the batch update path."""
        from app.tools.batch.update import update_context_batch

        result = await update_context_batch(
            updates=[{
                'context_id': '0190abcdef1234567890abcdef123456',
                'metadata': {'priority': 'high'},
            }],
            atomic=False,
        )

        assert result['succeeded'] == 0
        error = result['results'][0]['error']
        assert error is not None
        assert "metadata field 'priority'" in error
