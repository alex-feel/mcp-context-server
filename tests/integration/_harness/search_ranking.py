"""Real-server checks for the ordering of ranked search results.

The semantic, full-text and hybrid search tools serve every page from one
fixed-depth ordering: the union of the pages equals a single call, a window
past the depth carries a hint, and tied scores page without skipping or
repeating a row. Ordered text comes back byte-ordered on both backends, and
literal ``<mark>`` markup in a document survives ranked search.
"""

from typing import Any

from tests.integration._harness.core import HarnessCore


class SearchRankingMixin(HarnessCore):
    """Checks for ranked pagination, tie ordering, byte-wise collation and literal markup."""

    async def _ranked_search_legs(self, thread: str, query: str) -> list[tuple[str, dict[str, Any]]]:
        """Build the call arguments for every ranked search tool currently available.

        Args:
            thread: Thread the ranked query is scoped to.
            query: Free-text query passed to each tool.

        Returns:
            A (tool_name, arguments) pair per available ranked tool; empty when none is.
        """
        assert self.client is not None
        tool_names = {t.name for t in await self.client.list_tools()}
        stats_data = self._extract_content(await self.client.call_tool('get_statistics', {}))
        fts_info = stats_data.get('fts', {})
        fts_ok = bool(fts_info.get('enabled')) and bool(fts_info.get('available'))
        semantic_ok = bool(stats_data.get('semantic_search', {}).get('available'))

        legs: list[tuple[str, dict[str, Any]]] = []
        if fts_ok and 'fts_search_context' in tool_names:
            legs.append(('fts_search_context', {'query': query, 'mode': 'match', 'thread_id': thread}))
        if semantic_ok and 'semantic_search_context' in tool_names:
            legs.append(('semantic_search_context', {'query': query, 'thread_id': thread}))
        if (fts_ok or semantic_ok) and 'hybrid_search_context' in tool_names:
            legs.append(('hybrid_search_context', {'query': query, 'thread_id': thread}))
        return legs

    async def test_ranked_pagination_union_matches_single_page(self) -> bool:
        """Paging a ranked result set yields exactly the rows the single call yields.

        Semantic, FTS and hybrid search decide their FINAL order after the database
        returns rows -- cross-encoder reranking, RRF fusion, or both. A candidate window
        sized from the requested page would build page N and page N+1 from DIFFERENT
        candidate pools: a document that only entered the larger pool could outrank rows
        already served, pushing them onto a later page a second time while other rows
        were never returned by any page. The candidate depth is therefore fixed and
        page-independent (``RANKED_SEARCH_DEPTH``), so one query has ONE ordering and
        limit/offset merely select a window inside it.

        The assertion is the union property that guarantees: four two-row pages
        concatenated must equal the first eight rows of a single eight-row call, in the
        same order and with no id repeated. The corpus is deliberately larger than the
        window, so candidate pools sized from each page would differ from page to page.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'ranked_pagination_union_matches_single_page'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_rank_pages'
            corpus_size = 25
            entries = [
                {
                    'thread_id': thread, 'source': 'agent',
                    'text': (
                        f'Ranked pagination corpus document {index:02d} discussing paginated '
                        f'retrieval windows, ordering stability and page offsets'
                    ),
                }
                for index in range(corpus_size)
            ]
            batch = self._extract_content(await self.client.call_tool('store_context_batch', {
                'entries': entries, 'atomic': True,
            }))
            if batch.get('succeeded') != corpus_size:
                self.test_results.append((test_name, False, f'Corpus store failed: {batch}'))
                return False

            legs = await self._ranked_search_legs(thread, 'paginated retrieval windows')
            if not legs:
                self.test_results.append((test_name, True, 'Skipped (no ranked search tool available)'))
                return True

            page_size = 2
            window = 8
            for tool, args in legs:
                single = self._extract_content(
                    await self.client.call_tool(tool, {**args, 'limit': window, 'offset': 0}),
                )
                single_ids = [str(row.get('id')) for row in single.get('results', [])]
                if len(single_ids) != window:
                    self.test_results.append((
                        test_name, False, f'{tool} returned {len(single_ids)} rows for a {window}-row page',
                    ))
                    return False

                paged_ids: list[str] = []
                for offset in range(0, window, page_size):
                    page = self._extract_content(
                        await self.client.call_tool(tool, {**args, 'limit': page_size, 'offset': offset}),
                    )
                    rows = page.get('results', [])
                    if len(rows) != page_size:
                        self.test_results.append((
                            test_name, False,
                            f'{tool} page at offset {offset} returned {len(rows)} rows, expected {page_size}',
                        ))
                        return False
                    paged_ids.extend(str(row.get('id')) for row in rows)

                if len(set(paged_ids)) != len(paged_ids):
                    self.test_results.append((
                        test_name, False, f'{tool} returned the same id on two pages: {paged_ids}',
                    ))
                    return False
                if paged_ids != single_ids:
                    self.test_results.append((
                        test_name, False,
                        f'{tool} paged ids {paged_ids} differ from the single-call ids {single_ids}',
                    ))
                    return False

            self.test_results.append((
                test_name, True,
                f'Paged and unpaged ids agree across {len(legs)} ranked tool(s) over {corpus_size} documents',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_ranked_depth_limit_hint(self) -> bool:
        """A ranked page reaching past the fixed depth is reported, not silently empty.

        Ranked search serves every page from one ordering at most 100 rows deep (the
        depth the tool documentation advertises), so a window past that depth comes back
        short -- empty when the offset alone is past it. Without the hint a client cannot
        tell that from an exhausted result set and pages forever. The hint therefore
        appears exactly when ``offset + limit`` exceeds the depth, echoing the window it
        describes, and is ABSENT for an ordinary page.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'ranked_depth_limit_hint'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_rank_depth'
            entries = [
                {
                    'thread_id': thread, 'source': 'agent',
                    'text': f'Depth hint probe document {index} about paginated ranking depth',
                }
                for index in range(3)
            ]
            batch = self._extract_content(await self.client.call_tool('store_context_batch', {
                'entries': entries, 'atomic': True,
            }))
            if batch.get('succeeded') != len(entries):
                self.test_results.append((test_name, False, f'Probe store failed: {batch}'))
                return False

            legs = await self._ranked_search_legs(thread, 'paginated ranking depth')
            if not legs:
                self.test_results.append((test_name, True, 'Skipped (no ranked search tool available)'))
                return True

            expected_hint = {'requested_offset': 99, 'requested_limit': 5, 'rank_depth': 100}
            for tool, args in legs:
                deep = self._extract_content(
                    await self.client.call_tool(tool, {**args, 'limit': 5, 'offset': 99}),
                )
                if deep.get('rank_depth_limit') != expected_hint:
                    self.test_results.append((
                        test_name, False,
                        f'{tool} reported rank_depth_limit {deep.get("rank_depth_limit")!r}, expected {expected_hint}',
                    ))
                    return False
                if deep.get('results'):
                    self.test_results.append((
                        test_name, False, f'{tool} returned rows for a window past the ranked depth: {deep}',
                    ))
                    return False

                ordinary = self._extract_content(
                    await self.client.call_tool(tool, {**args, 'limit': 5, 'offset': 0}),
                )
                if 'rank_depth_limit' in ordinary:
                    self.test_results.append((
                        test_name, False, f'{tool} reported rank_depth_limit for an ordinary page: {ordinary}',
                    ))
                    return False

                # A window that STARTS at the depth is empty by arithmetic alone, so the
                # tool answers it without retrieving or scoring anything -- and must still
                # report the same shape rather than an error or a bare empty page.
                past = self._extract_content(
                    await self.client.call_tool(tool, {**args, 'limit': 10, 'offset': 100}),
                )
                if past.get('rank_depth_limit') != {
                    'requested_offset': 100, 'requested_limit': 10, 'rank_depth': 100,
                } or past.get('results') or past.get('count') != 0:
                    self.test_results.append((
                        test_name, False, f'{tool} mis-reported a page starting past the ranked depth: {past}',
                    ))
                    return False

            self.test_results.append((
                test_name, True, f'rank_depth_limit reported only past the ranked depth on {len(legs)} tool(s)',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_tied_score_pagination_parity(self) -> bool:
        """Paging a TIED ranked result set neither skips nor duplicates a row.

        Three byte-identical documents in one thread score identically in every ranked
        search, and a score-only ORDER BY would leave their relative order to the scan:
        on PostgreSQL an unrelated UPDATE rewrites the physical tuple (MVCC), so the heap
        order behind a tied LIMIT/OFFSET window changes under churn, and a client paging
        one row at a time would silently lose one document and see another twice. Each
        ranked tool therefore carries an explicit UNIQUE secondary key (the context id),
        so the union of the single-row pages must equal the unpaginated result --
        asserted before AND after a metadata-only update, the churn that reorders the
        heap.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'tied_score_pagination_parity'
        assert self.client is not None
        try:
            tool_names = {t.name for t in await self.client.list_tools()}
            stats_data = self._extract_content(await self.client.call_tool('get_statistics', {}))
            fts_info = stats_data.get('fts', {})
            fts_ok = bool(fts_info.get('enabled')) and bool(fts_info.get('available'))
            semantic_ok = bool(stats_data.get('semantic_search', {}).get('available'))

            thread = f'{self.test_thread_id}_tied_page'
            text = 'Tied ranking parity document about zebra quokka narwhal ordering'
            # Byte-identical text in ONE thread. The opposite-source store between the
            # repeats trips the deduplication interleaving check, so each repeat is a new
            # turn and INSERTS instead of updating the previous entry.
            expected_ids: list[str] = []
            for source in ('agent', 'user', 'agent'):
                stored = self._extract_content(await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': source, 'text': text,
                }))
                if not stored.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store tied entry: {stored}'))
                    return False
                expected_ids.append(str(stored['context_id']))
            if len(set(expected_ids)) != 3:
                self.test_results.append((
                    test_name, False, f'Deduplication collapsed the identical-text entries: {expected_ids}',
                ))
                return False

            legs: list[tuple[str, dict[str, Any]]] = []
            if fts_ok and 'fts_search_context' in tool_names:
                legs.append(('fts_search_context', {'query': 'zebra quokka', 'mode': 'match', 'thread_id': thread}))
            if semantic_ok and 'semantic_search_context' in tool_names:
                legs.append(('semantic_search_context', {'query': text, 'thread_id': thread}))
            if (fts_ok or semantic_ok) and 'hybrid_search_context' in tool_names:
                legs.append(('hybrid_search_context', {'query': text, 'thread_id': thread}))
            if not legs:
                self.test_results.append((test_name, True, 'Skipped (no ranked search tool available)'))
                return True

            async def _paging_error(label: str) -> str | None:
                """Return an error message when a leg's pages disagree with its full result."""
                assert self.client is not None
                for tool, args in legs:
                    full = self._extract_content(
                        await self.client.call_tool(tool, {**args, 'limit': 10, 'offset': 0}),
                    )
                    full_ids = [str(row.get('id')) for row in full.get('results', [])]
                    if sorted(full_ids) != sorted(expected_ids):
                        return f'{tool} {label}: unpaginated result {full_ids} != stored {expected_ids}'
                    paged: list[str] = []
                    for offset in range(3):
                        page = self._extract_content(
                            await self.client.call_tool(tool, {**args, 'limit': 1, 'offset': offset}),
                        )
                        rows = page.get('results', [])
                        if len(rows) != 1:
                            return f'{tool} {label}: page at offset {offset} returned {len(rows)} rows, expected 1'
                        paged.append(str(rows[0].get('id')))
                    if sorted(paged) != sorted(expected_ids):
                        return f'{tool} {label}: pages {paged} are not a partition of {expected_ids}'
                return None

            error = await _paging_error('before churn')
            if error:
                self.test_results.append((test_name, False, error))
                return False

            churn = self._extract_content(await self.client.call_tool('update_context', {
                'context_id': expected_ids[0], 'metadata_patch': {'churn': 'tied-pagination'},
            }))
            if not churn.get('success'):
                self.test_results.append((test_name, False, f'Metadata-only update failed: {churn}'))
                return False

            error = await _paging_error('after metadata-only update')
            if error:
                self.test_results.append((test_name, False, error))
                return False

            self.test_results.append((
                test_name, True,
                f'Tied pagination stable across {len(legs)} ranked tool(s) before and after churn',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_collation_ordering_parity(self) -> bool:
        """Ordered text comes back byte-ordered on BOTH backends, not locale-ordered.

        SQLite compares TEXT with its BINARY (byte) collation while PostgreSQL uses the
        database locale, which ranks punctuation and case differently, so without an
        explicit collation byte-identical data would serialize in a DIFFERENT order on
        the two backends: an entry's public ``tags`` array, and the tiebreak deciding
        which rows survive the statistics LIMIT. Every observable ordering site renders
        an explicit byte-wise collation, so two expectations hold on both backends:

        * an entry's tags come back byte-ordered ('t-z' before 'ta', because '-' sorts
          below 'a' by byte while the locale ranks it after), identically from
          get_context_by_ids and from every search tool;
        * inside the statistics top-N lists, rows sharing a count are byte-ordered --
          asserted over deliberately collation-sensitive tags seeded at a count that
          places them inside the top_tags window, and over whatever else the window
          holds.

        Returns:
            bool: True if test passed.
        """
        test_name = 'collation_ordering_parity'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_collation'
            entry_tags = ['ta', 'tb', 't-z']
            expected_tags = sorted(entry_tags)
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Collation ordering probe entry mentioning ferroniobium alloys',
                'tags': entry_tags,
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {stored}'))
                return False
            entry_id = str(stored['context_id'])

            got = self._extract_content(await self.client.call_tool('get_context_by_ids', {'context_ids': [entry_id]}))
            rows = got.get('results', [])
            if len(rows) != 1 or rows[0].get('tags') != expected_tags:
                self.test_results.append((
                    test_name, False,
                    f'get_context_by_ids returned tags {rows[0].get("tags") if rows else None}, expected {expected_tags}',
                ))
                return False

            legs: list[tuple[str, dict[str, Any]]] = [('search_context', {'thread_id': thread, 'limit': 10})]
            legs.extend(await self._ranked_search_legs(thread, 'ferroniobium alloys'))
            for tool, args in legs:
                data = self._extract_content(await self.client.call_tool(tool, {**args, 'limit': 10}))
                row = next((r for r in data.get('results', []) if str(r.get('id')) == entry_id), None)
                if row is None:
                    self.test_results.append((test_name, False, f'{tool} did not return the tagged entry'))
                    return False
                if row.get('tags') != expected_tags:
                    self.test_results.append((
                        test_name, False, f'{tool} returned tags {row.get("tags")}, expected {expected_tags}',
                    ))
                    return False

            # Six collation-sensitive labels whose byte order ('-' below 'a') differs
            # from the locale order, seeded across six equally-sized threads so both
            # statistics tiebreaks see them.
            collation_names = ['coll-a', 'coll-b', 'coll-c', 'colla', 'collb', 'collc']
            seed_entries = [
                {
                    'thread_id': f'{thread}_{collation_names[index % len(collation_names)]}',
                    'source': 'agent',
                    'text': f'Collation tie seed {index}',
                    'tags': collation_names,
                }
                for index in range(len(collation_names) * 2)
            ]
            seeded = self._extract_content(await self.client.call_tool('store_context_batch', {
                'entries': seed_entries, 'atomic': True,
            }))
            if seeded.get('succeeded') != len(seed_entries):
                self.test_results.append((test_name, False, f'Collation seed store failed: {seeded}'))
                return False

            stats = self._extract_content(await self.client.call_tool('get_statistics', {}))
            top_tags = stats.get('top_tags', [])
            most_active = stats.get('most_active_threads', [])
            seeded_order = [str(row.get('tag')) for row in top_tags if str(row.get('tag')) in set(collation_names)]
            if len(seeded_order) < 2:
                self.test_results.append((
                    test_name, False,
                    f'The seeded collation tags did not reach the top_tags window: {top_tags}',
                ))
                return False
            if seeded_order != sorted(seeded_order):
                self.test_results.append((
                    test_name, False, f'Tied seeded tags came back as {seeded_order}, expected byte order',
                ))
                return False

            for label, listed, key in (('top_tags', top_tags, 'tag'), ('most_active_threads', most_active, 'thread_id')):
                previous_key: str | None = None
                previous_count: object = None
                for row in listed:
                    current_key = str(row.get(key))
                    current_count = row.get('count')
                    if previous_key is not None and current_count == previous_count and current_key < previous_key:
                        self.test_results.append((
                            test_name, False,
                            f'{label} ties are not byte-ordered: {previous_key!r} precedes {current_key!r}',
                        ))
                        return False
                    previous_key, previous_count = current_key, current_count

            self.test_results.append((
                test_name, True, f'Tags and tied statistics rows are byte-ordered on {self.backend}',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_literal_markup_survives_ranked_search(self) -> bool:
        """A document containing literal <mark> markup stays searchable on both backends.

        Neither SQLite's ``highlight()`` nor PostgreSQL's ``ts_headline()`` escapes
        markup already present in a document, so a literal '<mark>' in the stored text is
        indistinguishable by shape from a marker the engine inserted. The passage
        extractor that feeds the cross-encoder therefore aligns the highlight against
        the source text; counting every tag as inserted would subtract a phantom offset
        and point the extracted passage at unrelated text. The observable
        contract at the tool boundary is what this pins: the document is still returned
        by the ranked tools, and its literal markup is stored and read back verbatim
        rather than being consumed as a highlight marker.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'literal_markup_survives_ranked_search'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_literal_markup'
            term = 'quarkbeacon'
            text = (
                f'Passage alignment probe: this document literally contains <mark>{term}</mark> '
                f'markup around the term {term}, and continues with further prose so the passage '
                f'extractor has a window of surrounding sentences to work with.'
            )
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent', 'text': text,
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {stored}'))
                return False
            entry_id = str(stored['context_id'])

            got = self._extract_content(await self.client.call_tool('get_context_by_ids', {'context_ids': [entry_id]}))
            rows = got.get('results', [])
            if len(rows) != 1 or f'<mark>{term}</mark>' not in str(rows[0].get('text_content', '')):
                self.test_results.append((test_name, False, 'The literal markup did not round-trip through storage'))
                return False

            legs = await self._ranked_search_legs(thread, term)
            if not legs:
                self.test_results.append((test_name, True, 'Skipped (no ranked search tool available)'))
                return True
            for tool, args in legs:
                data = self._extract_content(await self.client.call_tool(tool, {**args, 'limit': 10}))
                if not any(str(row.get('id')) == entry_id for row in data.get('results', [])):
                    self.test_results.append((
                        test_name, False, f'{tool} did not return the document carrying literal markup: {data}',
                    ))
                    return False

            self.test_results.append((
                test_name, True, f'Literal markup neither breaks nor is consumed by {len(legs)} ranked tool(s)',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
