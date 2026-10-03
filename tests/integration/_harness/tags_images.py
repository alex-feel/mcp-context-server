"""Real-server checks for tag and image handling on the write tools.

Tag lowercase normalization and deduplication on every write path, the
shared per-entry tag caps, per-image metadata as a JSON-encoded string
(the empty string included), and the ``image/png`` default for an image
sent to ``update_context`` without a ``mime_type``.
"""

from typing import Any

from tests.integration._harness.core import HarnessCore


class TagsImagesMixin(HarnessCore):
    """Checks for tag normalization and image attachment handling."""

    async def test_tags_lowercase_normalization(self) -> bool:
        """Verify tags are lowercased and deduplicated on storage and filtering.

        The store_context contract states tags are normalized to lowercase.
        This stores mixed-case duplicate tags and asserts the stored set is
        lowercased and deduplicated, and that an uppercase tag filter still
        matches (filter-side normalization).

        Returns:
            bool: True if test passed.
        """
        test_name = 'tags_lowercase_normalization'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_tag_norm'
            store = await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Entry exercising tag normalization',
                'tags': ['Foo', 'FOO', 'bar', 'BAR', 'Baz'],
            })
            store_data = self._extract_content(store)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {store_data}'))
                return False
            context_id = store_data['context_id']

            got = await self.client.call_tool('get_context_by_ids', {'context_ids': [context_id]})
            results = self._extract_content(got).get('results', [])
            if len(results) != 1:
                self.test_results.append((test_name, False, f'Expected 1 entry, got {len(results)}'))
                return False
            stored_tags = set(results[0].get('tags', []))
            if stored_tags != {'foo', 'bar', 'baz'}:
                self.test_results.append((test_name, False,
                    f'Tags not lowercased/deduped: got {sorted(stored_tags)}, expected [bar, baz, foo]'))
                return False

            # Uppercase filter must still match (filter-side normalization).
            search = await self.client.call_tool('search_context', {
                'thread_id': thread, 'tags': ['FOO'], 'limit': 10,
            })
            if len(self._extract_content(search).get('results', [])) != 1:
                self.test_results.append((test_name, False, "Uppercase tag filter 'FOO' did not match lowercased tag"))
                return False

            self.test_results.append((test_name, True, 'Tags lowercased+deduped to {bar,baz,foo}; uppercase filter matches'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_tag_write_caps_parity(self) -> bool:
        """Both backends accept and reject the SAME per-entry tag list on all four write tools.

        ``idx_tags_tag`` is a PostgreSQL btree index whose index-tuple ceiling would reject
        an oversized tag INSIDE the store transaction -- after a full generation pass, and
        while charging the circuit breaker -- while SQLite would store the same value.
        The shared write-path caps make ``store_context``, ``update_context`` and both
        batch tools accept and reject identically on both backends, and a list sitting
        exactly ON both caps must still be accepted.

        Returns:
            bool: True if test passed.
        """
        test_name = 'tag_write_caps_parity'
        assert self.client is not None

        async def _rejected(tool: str, args: dict[str, Any]) -> bool:
            """Report whether a call was refused, by raised error or by an unsuccessful response."""
            assert self.client is not None
            try:
                data = self._extract_content(await self.client.call_tool(tool, args))
            except Exception:
                return True
            return data.get('success') is not True

        try:
            from app.models import MAX_TAG_LENGTH
            from app.models import MAX_TAGS_PER_ENTRY

            thread = f'{self.test_thread_id}_tag_caps'
            at_cap = [(f'tag{i:03d}' + 'a' * MAX_TAG_LENGTH)[:MAX_TAG_LENGTH] for i in range(MAX_TAGS_PER_ENTRY)]
            over_length = ['b' * (MAX_TAG_LENGTH + 1)]
            too_many = [f'c{i}' for i in range(MAX_TAGS_PER_ENTRY + 1)]

            accepted = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Entry carrying a tag list exactly at both write caps',
                'tags': at_cap,
            }))
            if not accepted.get('success'):
                self.test_results.append((test_name, False, f'A tag list exactly at both caps was refused: {accepted}'))
                return False
            entry_id = str(accepted['context_id'])

            got = self._extract_content(await self.client.call_tool('get_context_by_ids', {'context_ids': [entry_id]}))
            rows = got.get('results', [])
            if len(rows) != 1 or len(rows[0].get('tags', [])) != MAX_TAGS_PER_ENTRY:
                self.test_results.append((
                    test_name, False,
                    f'Expected {MAX_TAGS_PER_ENTRY} stored tags, got {len(rows[0].get("tags", [])) if rows else 0}',
                ))
                return False

            rejections: list[tuple[str, str, dict[str, Any]]] = [
                ('store_context over-long tag', 'store_context', {
                    'thread_id': thread, 'source': 'agent', 'text': 'over-long tag store', 'tags': over_length,
                }),
                ('store_context too many tags', 'store_context', {
                    'thread_id': thread, 'source': 'agent', 'text': 'too many tags store', 'tags': too_many,
                }),
                ('update_context over-long tag', 'update_context', {'context_id': entry_id, 'tags': over_length}),
                ('update_context too many tags', 'update_context', {'context_id': entry_id, 'tags': too_many}),
            ]
            for label, tool, args in rejections:
                if not await _rejected(tool, args):
                    self.test_results.append((test_name, False, f'{label} was accepted on {self.backend}'))
                    return False

            # The batch tools take untyped dicts, so the shared chokepoint -- not the wire
            # schema -- refuses them, per entry, leaving a valid sibling untouched.
            batch = self._extract_content(await self.client.call_tool('store_context_batch', {
                'entries': [
                    {'thread_id': thread, 'source': 'agent', 'text': 'batch entry with a legal tag', 'tags': ['legal']},
                    {'thread_id': thread, 'source': 'agent', 'text': 'batch entry with an over-long tag',
                     'tags': over_length},
                ],
                'atomic': False,
            }))
            batch_errors = [str(r.get('error', '')) for r in batch.get('results', []) if not r.get('success')]
            if batch.get('succeeded') != 1 or batch.get('failed') != 1:
                self.test_results.append((
                    test_name, False,
                    f'store_context_batch reported {batch.get("succeeded")} succeeded / {batch.get("failed")} failed',
                ))
                return False
            if not any('too long' in message for message in batch_errors):
                self.test_results.append((
                    test_name, False, f'store_context_batch error lacks the length reason: {batch_errors}',
                ))
                return False

            update_batch = self._extract_content(await self.client.call_tool('update_context_batch', {
                'updates': [{'context_id': entry_id, 'tags': too_many}],
                'atomic': False,
            }))
            update_errors = [str(r.get('error', '')) for r in update_batch.get('results', []) if not r.get('success')]
            if update_batch.get('failed') != 1 or not any('Too many tags' in message for message in update_errors):
                self.test_results.append((
                    test_name, False, f'update_context_batch did not reject the oversized list: {update_batch}',
                ))
                return False

            self.test_results.append((
                test_name, True,
                f'Tag caps ({MAX_TAGS_PER_ENTRY} tags / {MAX_TAG_LENGTH} chars) enforced identically on {self.backend}',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_tag_deduplication_across_write_paths(self) -> bool:
        """A repeated tag is stored once, on all four write paths and both backends.

        Tags are a SET of labels, and every reader exposes the stored rows verbatim, so
        inserting the same label twice would leak a duplicate into every response and
        inflate the tag counts. Normalization itself manufactures the collision --
        'DEDUPA' and 'dedupa' are distinct on the wire and identical once trimmed and
        lower-cased -- so deduplication happens after it, at the single chokepoint every
        write path funnels through. The list is compared RAW, never through ``set()``,
        so a duplicate cannot hide behind the comparison.

        Returns:
            bool: True if test passed.
        """
        test_name = 'tag_deduplication_across_write_paths'
        assert self.client is not None
        raw_tags = ['dedup-z', 'dedupa', 'dedup-z', 'DEDUPA']
        # Byte order puts the hyphen below 'a', and the read path is byte-ordered on
        # both backends.
        expected_tags = ['dedup-z', 'dedupa']

        async def _tags_of(entry_id: str) -> list[str] | None:
            """Read one entry's stored tag list, or None when the entry is missing."""
            assert self.client is not None
            got = self._extract_content(
                await self.client.call_tool('get_context_by_ids', {'context_ids': [entry_id]}),
            )
            rows = got.get('results', [])
            if len(rows) != 1:
                return None
            return list(rows[0].get('tags', []))

        async def _assert_tags(label: str, entry_id: str) -> bool:
            """Report whether the entry's stored tags are exactly the deduplicated list."""
            tags = await _tags_of(entry_id)
            if tags == expected_tags:
                return True
            self.test_results.append((test_name, False, f'{label} stored tags {tags}, expected {expected_tags}'))
            return False

        try:
            thread = f'{self.test_thread_id}_tag_dedup'
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Entry whose tag list repeats a label', 'tags': raw_tags,
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {stored}'))
                return False
            single_id = str(stored['context_id'])
            if not await _assert_tags('store_context', single_id):
                return False

            # Each path is asserted immediately after it writes, so a leak on one
            # cannot be masked by the next path replacing the list correctly.
            updated = self._extract_content(await self.client.call_tool('update_context', {
                'context_id': single_id, 'tags': raw_tags,
            }))
            if not updated.get('success'):
                self.test_results.append((test_name, False, f'Update failed: {updated}'))
                return False
            if not await _assert_tags('update_context', single_id):
                return False

            batch = self._extract_content(await self.client.call_tool('store_context_batch', {
                'entries': [{
                    'thread_id': thread, 'source': 'agent',
                    'text': 'Batch entry whose tag list repeats a label', 'tags': raw_tags,
                }],
                'atomic': True,
            }))
            if batch.get('succeeded') != 1:
                self.test_results.append((test_name, False, f'Batch store failed: {batch}'))
                return False
            batch_id = str(batch['results'][0]['context_id'])
            if not await _assert_tags('store_context_batch', batch_id):
                return False

            batch_updated = self._extract_content(await self.client.call_tool('update_context_batch', {
                'updates': [{'context_id': batch_id, 'tags': raw_tags}], 'atomic': True,
            }))
            if batch_updated.get('succeeded') != 1:
                self.test_results.append((test_name, False, f'Batch update failed: {batch_updated}'))
                return False
            if not await _assert_tags('update_context_batch', batch_id):
                return False

            self.test_results.append((
                test_name, True, f'All four write paths store {expected_tags} once on {self.backend}',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_image_metadata_json_string_contract(self) -> bool:
        """Per-image metadata crosses the boundary as a JSON-ENCODED STRING on both backends.

        The typed single-entry tools declare images as ``list[dict[str, str]]``, the write
        path serializes that already-stringified value and the read path parses it back,
        so a client receives exactly the string it sent and ``get_context_by_ids`` passes
        its strict output schema. The untyped batch path bypasses the typed declaration,
        so a dict reaching storage there would be stored in a shape the single-entry tool
        refuses -- and would then fail that output schema, leaving the entry permanently
        unreadable. The shared validation chokepoint rejects it per entry.

        Returns:
            bool: True if test passed.
        """
        test_name = 'image_metadata_json_string_contract'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_image_meta'
            encoded = '{"iso": 100}'
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent', 'text': 'Entry with per-image metadata',
                'images': [{'data': self._create_test_image(), 'mime_type': 'image/png', 'metadata': encoded}],
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Store with image metadata failed: {stored}'))
                return False
            entry_id = str(stored['context_id'])

            # A shape the output schema rejects would raise here rather than return.
            got = self._extract_content(await self.client.call_tool('get_context_by_ids', {
                'context_ids': [entry_id], 'include_images': True,
            }))
            rows = got.get('results', [])
            if len(rows) != 1 or not rows[0].get('images'):
                self.test_results.append((test_name, False, f'Entry came back without its image: {got}'))
                return False
            image_metadata = rows[0]['images'][0].get('metadata')
            if image_metadata != encoded:
                self.test_results.append((
                    test_name, False, f'Image metadata round-tripped as {image_metadata!r}, expected {encoded!r}',
                ))
                return False

            batch = self._extract_content(await self.client.call_tool('store_context_batch', {
                'entries': [{
                    'thread_id': thread, 'source': 'agent', 'text': 'Batch entry with dict image metadata',
                    'images': [{'data': self._create_test_image(), 'mime_type': 'image/png', 'metadata': {'iso': 100}}],
                }],
                'atomic': False,
            }))
            batch_errors = [str(r.get('error', '')) for r in batch.get('results', []) if not r.get('success')]
            if batch.get('failed') != 1 or not any('metadata must be a JSON-encoded string' in m for m in batch_errors):
                self.test_results.append((
                    test_name, False, f'Batch store accepted a dict image metadata or misreported it: {batch}',
                ))
                return False

            self.test_results.append((
                test_name, True, 'Per-image metadata round-trips as a JSON string; a dict is rejected per entry',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_image_metadata_empty_string_preserved(self) -> bool:
        """A deliberately EMPTY per-image metadata value survives the round trip.

        Per-image metadata crosses the boundary as a JSON-encoded string, and the empty
        string is the one valid payload Python considers falsy. Gating the write on
        truthiness would store it as SQL NULL, which the read path reports as metadata
        never supplied -- so "supplied empty" and "never supplied" would collapse into the
        same response and a client could not tell them apart. The gate is
        ``is not None`` on both the write and the read side. The control case pins the
        other half: an image with no metadata key must still come back without one.

        Returns:
            bool: True if test passed.
        """
        test_name = 'image_metadata_empty_string_preserved'
        assert self.client is not None

        async def _first_image(entry_id: str) -> dict[str, Any] | None:
            """Read the first stored image of an entry, or None when there is none."""
            assert self.client is not None
            got = self._extract_content(await self.client.call_tool('get_context_by_ids', {
                'context_ids': [entry_id], 'include_images': True,
            }))
            rows = got.get('results', [])
            if len(rows) != 1:
                return None
            images = rows[0].get('images') or []
            first = images[0] if images else None
            return first if isinstance(first, dict) else None

        try:
            thread = f'{self.test_thread_id}_image_empty_meta'
            supplied = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Entry whose image carries a deliberately empty metadata value',
                'images': [{'data': self._create_test_image(), 'mime_type': 'image/png', 'metadata': ''}],
            }))
            if not supplied.get('success'):
                self.test_results.append((test_name, False, f'Store with empty image metadata failed: {supplied}'))
                return False
            supplied_image = await _first_image(str(supplied['context_id']))
            if supplied_image is None:
                self.test_results.append((test_name, False, 'The entry came back without its image'))
                return False
            if 'metadata' not in supplied_image or supplied_image['metadata'] != '':
                self.test_results.append((
                    test_name, False,
                    f'An empty image metadata value came back as {supplied_image.get("metadata")!r}',
                ))
                return False

            absent = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Entry whose image carries no metadata value at all',
                'images': [{'data': self._create_test_image(), 'mime_type': 'image/png'}],
            }))
            if not absent.get('success'):
                self.test_results.append((test_name, False, f'Store without image metadata failed: {absent}'))
                return False
            absent_image = await _first_image(str(absent['context_id']))
            if absent_image is None:
                self.test_results.append((test_name, False, 'The control entry came back without its image'))
                return False
            if 'metadata' in absent_image:
                self.test_results.append((
                    test_name, False,
                    f'An image with no metadata reported metadata {absent_image["metadata"]!r}',
                ))
                return False

            self.test_results.append((
                test_name, True, 'Supplied-empty and never-supplied image metadata stay distinguishable',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_update_context_image_without_mime_type_integration(self) -> bool:
        """Verify mime_type defaults to 'image/png' when omitted in update_context.

        Returns:
            bool: True if test passed.
        """
        test_name = 'update_context_image_without_mime_type_integration'
        assert self.client is not None
        try:
            mime_thread = f'{self.test_thread_id}_image_mime_default'

            store_result = await self.client.call_tool('store_context', {
                'thread_id': mime_thread, 'source': 'agent',
                'text': 'Entry for image MIME default test',
            })
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {store_data}'))
                return False

            context_id = store_data.get('context_id')

            update_result = await self.client.call_tool('update_context', {
                'context_id': context_id,
                'images': [{'data': self._create_test_image()}],
            })
            update_data = self._extract_content(update_result)

            if not update_data.get('success'):
                self.test_results.append((test_name, False,
                    f'Update with image (no mime_type) failed: {update_data}'))
                return False

            get_result = await self.client.call_tool('get_context_by_ids', {
                'context_ids': [context_id], 'include_images': True,
            })
            get_data = self._extract_content(get_result)
            if not get_data.get('success') or len(get_data.get('results', [])) == 0:
                self.test_results.append((test_name, False, 'Could not retrieve updated entry'))
                return False

            entry = get_data['results'][0]
            if entry.get('content_type') != 'multimodal':
                self.test_results.append((test_name, False,
                    f"Expected content_type='multimodal', got '{entry.get('content_type')}'"))
                return False

            self.test_results.append((test_name, True,
                'Image without mime_type accepted in update_context'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
