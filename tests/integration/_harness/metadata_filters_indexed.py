"""Real-server checks for the write-boundary caps on indexed metadata fields.

A list or object under an indexed field is capped by the width the index
stores, and a value a typed indexed field cannot cast is rejected before the
write, identically on both backends.
"""

from typing import Any

from tests.integration._harness.core import HarnessCore


class MetadataFiltersIndexedMixin(HarnessCore):
    """Checks for indexed metadata field width caps and typed casts."""

    async def test_indexed_metadata_container_length_parity(self) -> bool:
        """A container under an indexed metadata field is capped by its INDEXED width.

        PostgreSQL's ``->>`` renders a list or object as its whole serialized JSON, and
        that text is what ``idx_metadata_<field>`` stores -- under a btree index-tuple
        ceiling. A cap that inspected only string values would let an oversized container
        reach the INSERT, which PostgreSQL aborts inside the store transaction (after a full
        generation pass, charging the circuit breaker) while SQLite stores it. The write
        boundary therefore measures the text the index would hold, so both backends refuse
        the same value up front -- and a small container still stores, so the cap is not a
        blanket ban on containers.

        Returns:
            bool: True if test passed.
        """
        test_name = 'indexed_metadata_container_length_parity'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_indexed_container'
            oversized = ['x' * 100] * 40

            refused = False
            try:
                response = self._extract_content(await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent',
                    'text': 'Oversized container under an indexed metadata field',
                    'metadata': {'project': oversized},
                }))
            except Exception:
                refused = True
            else:
                refused = response.get('success') is not True
            if not refused:
                self.test_results.append((
                    test_name, False, 'An oversized container under an indexed field was accepted',
                ))
                return False

            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Small container under an indexed metadata field',
                'metadata': {'project': ['alpha', 'beta']},
            }))
            if not stored.get('success'):
                self.test_results.append((
                    test_name, False, f'A small container under an indexed field was refused: {stored}',
                ))
                return False

            self.test_results.append((
                test_name, True, 'Indexed metadata containers are capped by their indexed width on both backends',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_typed_indexed_metadata_cast_parity(self) -> bool:
        """A typed indexed metadata field accepts and rejects identically on both backends.

        A ``METADATA_INDEXED_FIELDS`` entry carrying an ``integer`` type hint becomes a
        hard SQL cast inside the PostgreSQL expression index, which PostgreSQL evaluates
        on every INSERT. Unchecked, a non-numeric value would abort the write with a raw
        driver error -- after a full generation pass, inside the transaction, charging the
        circuit breaker -- while SQLite's uncast ``json_extract`` index stores the same
        value. The write boundary therefore rejects it up front on both backends, and a
        castable value still stores, which on PostgreSQL also proves the real expression
        index accepts it.

        The field must exist in the server's configuration at startup, so this runs
        against a second server whose ``METADATA_INDEXED_FIELDS`` declares it.

        Returns:
            bool: True if test passed.
        """
        test_name = 'typed_indexed_metadata_cast_parity'
        field = 'castprobe'
        try:
            async with self._second_server({'METADATA_INDEXED_FIELDS': f'{field}:integer'}) as client:
                thread = f'{self.test_thread_id}_typed_metadata'

                async def _refused(tool: str, args: dict[str, Any]) -> bool:
                    """Report whether a call was refused, by raised error or unsuccessful response."""
                    try:
                        data = self._extract_content(await client.call_tool(tool, args))
                    except Exception:
                        return True
                    return data.get('success') is not True

                if not await _refused('store_context', {
                    'thread_id': thread, 'source': 'agent',
                    'text': 'Store carrying a non-numeric value under an integer-indexed field',
                    'metadata': {field: 'not-a-number'},
                }):
                    self.test_results.append((
                        test_name, False, f'A non-numeric {field} value was accepted on {self.backend}',
                    ))
                    return False

                stored = self._extract_content(await client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent',
                    'text': 'Store carrying a castable value under an integer-indexed field',
                    'metadata': {field: 42},
                }))
                if not stored.get('success'):
                    self.test_results.append((test_name, False, f'A castable {field} value was refused: {stored}'))
                    return False
                entry_id = str(stored['context_id'])

                got = self._extract_content(await client.call_tool('get_context_by_ids', {'context_ids': [entry_id]}))
                rows = got.get('results', [])
                if len(rows) != 1 or rows[0].get('metadata', {}).get(field) != 42:
                    self.test_results.append((test_name, False, f'The castable value did not round-trip: {got}'))
                    return False

                if not await _refused('update_context', {
                    'context_id': entry_id, 'metadata_patch': {field: 'still-not-a-number'},
                }):
                    self.test_results.append((
                        test_name, False, f'A non-numeric {field} patch was accepted on {self.backend}',
                    ))
                    return False

                batch = self._extract_content(await client.call_tool('store_context_batch', {
                    'entries': [
                        {'thread_id': thread, 'source': 'agent', 'text': 'batch entry with a castable value',
                         'metadata': {field: 7}},
                        {'thread_id': thread, 'source': 'agent', 'text': 'batch entry with a non-numeric value',
                         'metadata': {field: 'nope'}},
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
                if not any('indexed as integer' in message for message in batch_errors):
                    self.test_results.append((
                        test_name, False, f'The batch rejection lacks the index-type reason: {batch_errors}',
                    ))
                    return False

            self.test_results.append((
                test_name, True, f'A {field}:integer field accepts and rejects identically on {self.backend}',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
