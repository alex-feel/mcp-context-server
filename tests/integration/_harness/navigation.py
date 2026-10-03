"""Real-server checks for the navigation tools.

``grep_context`` literal, regex and Unicode-case matching, its exhaustive
keyset scan, the scan-cap boundary and request clamping;
``read_context_range`` addressing, clamping and composition with grep;
``navigate_context`` outlines resolved through node reads; and the
index_tree per-node summaries with their ``get_statistics`` block.
"""

from tests.integration._harness.core import HarnessCore


class NavigationMixin(HarnessCore):
    """Checks for grep_context, read_context_range, navigate_context and index_tree statistics."""

    async def test_grep_context_literal_regex_unicode(self) -> bool:
        """Verify grep_context matches literal, regex, and Unicode-case patterns identically on both backends.

        The Cyrillic upper-vs-lower case match is the parity-critical check: it
        forces Python re.IGNORECASE (the ASCII-only SQL substring pre-narrow is
        skipped for non-ASCII), so SQLite and PostgreSQL must agree.

        Returns:
            bool: True if test passed.
        """
        test_name = 'grep_context_literal_regex_unicode'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_grep'
            cyr_lower = ''.join(chr(c) for c in (0x043F, 0x0440, 0x0438, 0x0432, 0x0435, 0x0442))
            store = await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': f'alpha NEEDLE line\nbeta line\n{cyr_lower} tail',
            })
            if not self._extract_content(store).get('success'):
                self.test_results.append((test_name, False, 'store failed'))
                return False

            literal = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': 'NEEDLE', 'thread_id': thread,
            }))
            if len(literal.get('results', [])) != 1:
                self.test_results.append((test_name, False, f'literal grep expected 1 entry, got {literal}'))
                return False

            unicode_ci = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': cyr_lower.upper(), 'thread_id': thread,
            }))
            if len(unicode_ci.get('results', [])) != 1:
                self.test_results.append((test_name, False, 'Cyrillic case-insensitive grep failed (backend parity)'))
                return False

            regex = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': 'N.+E', 'thread_id': thread, 'is_regex': True,
            }))
            if len(regex.get('results', [])) != 1:
                self.test_results.append((test_name, False, 'regex grep failed'))
                return False

            self.test_results.append((test_name, True, f'grep literal+regex+unicode on {self.backend}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_grep_keyset_scan_exhaustive_beyond_limit(self) -> bool:
        """grep_context's keyset scan must cover ALL matching entries, not cap at
        the ``search_contexts`` LIMIT 50, on both backends.

        Stores 60 entries (> 50) each carrying a shared token in a dedicated
        thread, then greps for the token and asserts every entry comes back. The
        scan is ``grep_scan_text_contents``'s exhaustive id-DESC keyset
        pagination; a scan routed through the capped search path would return at
        most 50 of them.

        Returns:
            bool: True if test passed.
        """
        test_name = 'grep_keyset_scan_exhaustive_beyond_limit'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_grepkeyset'
            count = 60
            for i in range(count):
                store = self._extract_content(await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent',
                    'text': f'entry number {i} carries KEYSETTOKEN inline',
                }))
                if not store.get('success'):
                    self.test_results.append((test_name, False, f'store {i} failed: {store}'))
                    return False

            result = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': 'KEYSETTOKEN', 'thread_id': thread,
                'output_mode': 'files_with_matches', 'max_matches': 1000, 'max_entries_scanned': 1000,
            }))
            rows = result.get('results', [])
            if len(rows) != count:
                self.test_results.append((
                    test_name, False,
                    (
                        f'keyset scan returned {len(rows)} of {count} entries '
                        f'(capped at search LIMIT?): truncated={result.get("truncated")}'
                    ),
                ))
                return False

            self.test_results.append((test_name, True, f'grep keyset scanned all {count} entries on {self.backend}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_grep_scan_cap_boundary_truncation(self) -> bool:
        """grep_context's cap+lookahead must distinguish exhaustion (exactly the
        cap) from overflow (one more row) on BOTH backends.

        Exercises ``grep_scan_text_contents``'s per-backend single-row lookahead at
        the EXACT ``max_entries_scanned`` boundary: with N matching entries and
        ``max_entries_scanned=N`` the scan is exhausted -> ``truncated`` False;
        adding one more matching entry -> ``truncated`` True. Running on both
        backends keeps the structurally duplicated ``_scan_sqlite`` /
        ``_scan_postgresql`` lookahead in agreement, the PostgreSQL boundary
        branch included.

        Returns:
            bool: True if test passed.
        """
        test_name = 'grep_scan_cap_boundary_truncation'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_grepcap'
            token = 'GREPCAPBOUNDARY'
            cap = 5
            for i in range(cap):
                store = self._extract_content(await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent',
                    'text': f'entry {i} holds {token} inline',
                }))
                if not store.get('success'):
                    self.test_results.append((test_name, False, f'store {i} failed: {store}'))
                    return False

            exact = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': token, 'thread_id': thread,
                'output_mode': 'files_with_matches', 'max_matches': 1000, 'max_entries_scanned': cap,
            }))
            if exact.get('truncated') is not False:
                self.test_results.append((
                    test_name, False,
                    f'exact-fit scan (N==cap=={cap}) should be truncated=False, got {exact.get("truncated")}',
                ))
                return False

            # One more matching entry -> the scan caps at `cap` and the lookahead
            # finds the overflow row -> truncated True.
            store = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': f'entry overflow holds {token} inline',
            }))
            if not store.get('success'):
                self.test_results.append((test_name, False, f'overflow store failed: {store}'))
                return False

            over = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': token, 'thread_id': thread,
                'output_mode': 'files_with_matches', 'max_matches': 1000, 'max_entries_scanned': cap,
            }))
            if over.get('truncated') is not True:
                self.test_results.append((
                    test_name, False,
                    f'one-over-cap scan (N==cap+1) should be truncated=True, got {over.get("truncated")}',
                ))
                return False

            self.test_results.append((test_name, True, f'grep cap boundary correct on {self.backend}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_grep_context_request_caps_clamped(self) -> bool:
        """grep_context clamps an oversized request to the server caps on both backends.

        The wire schema deliberately admits values far above the server bounds
        (``max_matches`` to 10000, ``context_lines`` to 100, ``max_entries_scanned`` to
        1000000) so a client is never rejected for asking; the server clamps each one to
        its configured cap. Without the clamp a single call can return every match in a
        dense corpus with a hundred context lines apiece, flooding the caller's context
        window and the event loop. The corpus here carries more matches than the default
        1000 cap and more surrounding lines than the default 20, so a missing clamp
        shows up as an over-cap response rather than as a silent pass.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'grep_context_request_caps_clamped'
        assert self.client is not None
        try:
            tool_names = {t.name for t in await self.client.list_tools()}
            if 'grep_context' not in tool_names:
                self.test_results.append((test_name, True, 'Skipped (grep_context not registered)'))
                return True

            dense_thread = f'{self.test_thread_id}_grep_caps_dense'
            token = 'GREPCLAMPTOKEN'
            # 30 lines x 40 occurrences = 1200 matches, past the default 1000 cap.
            dense_line = ' '.join([token] * 40)
            dense_text = '\n'.join(f'{index:03d} {dense_line}' for index in range(30))
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': dense_thread, 'source': 'agent', 'text': dense_text,
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Failed to store the dense corpus: {stored}'))
                return False

            counted = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': token, 'thread_id': dense_thread, 'output_mode': 'count',
                'max_matches': 10000, 'context_lines': 100, 'max_entries_scanned': 1000000,
            }))
            total = counted.get('total_matches')
            if not isinstance(total, int) or not (0 < total <= 1000):
                self.test_results.append((
                    test_name, False, f'total_matches {total!r} is not clamped into (0, 1000]',
                ))
                return False
            if counted.get('truncated') is not True:
                self.test_results.append((
                    test_name, False, f'A capped scan must report truncated=True, got {counted.get("truncated")!r}',
                ))
                return False

            # A separate small entry keeps the content-mode response tiny while still
            # offering more surrounding lines than the context cap allows.
            context_thread = f'{self.test_thread_id}_grep_caps_context'
            marker = 'GREPCLAMPMARKER'
            lines = [f'context line {index}' for index in range(61)]
            lines[30] = f'context line 30 holds {marker}'
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': context_thread, 'source': 'agent', 'text': '\n'.join(lines),
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Failed to store the context corpus: {stored}'))
                return False

            content = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': marker, 'thread_id': context_thread, 'output_mode': 'content',
                'max_matches': 10000, 'context_lines': 100, 'max_entries_scanned': 1000000,
            }))
            rows = content.get('results', [])
            if len(rows) != 1:
                self.test_results.append((test_name, False, f'Expected exactly one content match, got {len(rows)}'))
                return False
            before = rows[0].get('before', [])
            after = rows[0].get('after', [])
            if not (0 < len(before) <= 20) or not (0 < len(after) <= 20):
                self.test_results.append((
                    test_name, False,
                    f'context_lines was not clamped: before={len(before)}, after={len(after)} (30 available each side)',
                ))
                return False

            self.test_results.append((
                test_name, True,
                f'Oversized request clamped: total_matches={total}, before={len(before)}, after={len(after)}',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_read_context_range_clamp_and_composition(self) -> bool:
        """Verify read_context_range char/line addressing, clamp+echo, and grep->read composition.

        Proves the shared code-point offset contract end to end on both backends:
        a grep content match's offsets feed read_context_range to extract exactly
        the matched span, and an over-range request is clamped to the document end.

        Returns:
            bool: True if test passed.
        """
        test_name = 'read_context_range_clamp_and_composition'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_read'
            full_text = 'line one\nfind TARGET here\nline three'
            store = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent', 'text': full_text,
            }))
            if not store.get('success'):
                self.test_results.append((test_name, False, 'store failed'))
                return False
            cid = store['context_id']

            char_read = self._extract_content(await self.client.call_tool('read_context_range', {
                'context_id': cid, 'start_char': 0, 'end_char': 8,
            }))
            if char_read.get('text') != 'line one':
                self.test_results.append((test_name, False, f'char range wrong: {char_read}'))
                return False

            clamped = self._extract_content(await self.client.call_tool('read_context_range', {
                'context_id': cid, 'start_char': 0, 'end_char': 100000,
            }))
            if clamped.get('end_char') != len(full_text):
                self.test_results.append((test_name, False, f'clamp not applied: {clamped}'))
                return False

            grep = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': 'TARGET', 'thread_id': thread, 'output_mode': 'content', 'case_sensitive': True,
            }))
            matches = grep.get('results', [])
            if not matches:
                self.test_results.append((test_name, False, 'grep content found no TARGET match'))
                return False
            match = matches[0]
            extracted = self._extract_content(await self.client.call_tool('read_context_range', {
                'context_id': match['context_id'],
                'start_char': match['match_start'],
                'end_char': match['match_end'],
            }))
            if extracted.get('text') != 'TARGET':
                self.test_results.append((test_name, False, f'composition extracted wrong span: {extracted}'))
                return False

            # Multibyte composition: a Cyrillic prefix makes code-point and UTF-8
            # byte offsets diverge, so this proves grep's match offsets are
            # code-point indices that compose with read_context_range identically
            # on SQLite and PostgreSQL.
            mb_thread = f'{self.test_thread_id}_readmb'
            cyr = ''.join(chr(c) for c in (0x0451, 0x0451, 0x0451))  # 3 two-byte chars
            mb_store = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': mb_thread, 'source': 'agent', 'text': f'{cyr} MULTIBYTE tail',
            }))
            mb_cid = mb_store['context_id']
            mb_grep = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': 'MULTIBYTE', 'thread_id': mb_thread, 'output_mode': 'content', 'case_sensitive': True,
            }))
            mb_matches = mb_grep.get('results', [])
            if not mb_matches:
                self.test_results.append((test_name, False, 'multibyte grep found no match'))
                return False
            mb_match = mb_matches[0]
            # Code-point offset is 4 (3 Cyrillic + 1 space), not 7 UTF-8 bytes.
            if mb_match['match_start'] != 4:
                self.test_results.append((
                    test_name, False,
                    f'multibyte match_start is not a code-point offset (got {mb_match["match_start"]}, want 4)',
                ))
                return False
            mb_extracted = self._extract_content(await self.client.call_tool('read_context_range', {
                'context_id': mb_cid, 'start_char': mb_match['match_start'], 'end_char': mb_match['match_end'],
            }))
            if mb_extracted.get('text') != 'MULTIBYTE':
                self.test_results.append((test_name, False, f'multibyte composition wrong span: {mb_extracted}'))
                return False

            self.test_results.append((test_name, True, f'read_context_range clamp+composition (+multibyte) on {self.backend}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_navigate_context_outline_and_node_read(self) -> bool:
        """Verify navigate_context builds a Markdown outline and node_id reads its section, on both backends.

        Stores a multi-section Markdown entry, asserts the on-demand heading tree
        (root + nested sections with code-point offsets), then resolves a node_id
        through read_context_range to extract exactly that section -- proving the
        navigate->extract path and the shared offset contract across backends.

        Returns:
            bool: True if test passed.
        """
        test_name = 'navigate_context_outline_and_node_read'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_nav'
            text = '# Intro\nintro body\n## Details\ndetail body here\n'
            store = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent', 'text': text,
            }))
            if not store.get('success'):
                self.test_results.append((test_name, False, 'store failed'))
                return False
            cid = store['context_id']

            nav = self._extract_content(await self.client.call_tool('navigate_context', {'context_id': cid}))
            if nav.get('node_count') != 2:
                self.test_results.append((test_name, False, f'expected 2 nodes, got {nav}'))
                return False
            root = nav.get('root', {})
            intro = root.get('children', [{}])[0]
            details = intro.get('children', [{}])[0]
            if details.get('node_id') != 'intro/details':
                self.test_results.append((test_name, False, f'node_id wrong: {details}'))
                return False

            section = self._extract_content(await self.client.call_tool('read_context_range', {
                'context_id': cid, 'node_id': 'intro/details',
            }))
            if not section.get('text', '').startswith('## Details'):
                self.test_results.append((test_name, False, f'node read wrong span: {section}'))
                return False

            self.test_results.append((test_name, True, f'navigate_context outline+node read on {self.backend}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_index_tree_node_summaries_and_statistics(self) -> bool:
        """Verify node summaries stay additive (never abort a store) and get_statistics exposes index_tree.

        Stores a multi-section Markdown entry with per-node summaries ON (default).
        Whether or not a summary provider is configured, the store must succeed
        (a missing/failed node summary never aborts), and get_statistics must
        carry an ``index_tree`` block with enabled + a non-negative node_count.

        Returns:
            bool: True if test passed.
        """
        test_name = 'index_tree_node_summaries_and_statistics'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_idxtree'
            store = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': '# Alpha\nalpha body\n## Beta\nbeta body here\n',
            }))
            if not store.get('success'):
                self.test_results.append((test_name, False, f'store failed: {store}'))
                return False

            stats = self._extract_content(await self.client.call_tool('get_statistics', {}))
            index_tree = stats.get('index_tree')
            if not isinstance(index_tree, dict):
                self.test_results.append((test_name, False, f'index_tree block missing from statistics: {stats.keys()}'))
                return False
            if 'enabled' not in index_tree or not isinstance(index_tree.get('node_count'), int):
                self.test_results.append((test_name, False, f'index_tree block malformed: {index_tree}'))
                return False
            if index_tree['node_count'] < 0:
                self.test_results.append((test_name, False, f'negative node_count: {index_tree}'))
                return False

            self.test_results.append((
                test_name, True,
                (
                    f'index_tree additive store + statistics on {self.backend} '
                    f'(node_count={index_tree["node_count"]})'
                ),
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
