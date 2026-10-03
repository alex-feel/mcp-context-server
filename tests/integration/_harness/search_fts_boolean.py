"""Real-server checks for boolean-mode full-text search and term parity.

AND/OR/NOT operators and NOT exclusion in boolean mode, malformed and deeply
nested boolean queries that degrade instead of erroring, and a term with an
embedded double quote that matches the same documents on both backends.
"""

from tests.integration._harness.core import HarnessCore


class SearchFtsBooleanMixin(HarnessCore):
    """Checks for boolean-mode FTS queries and their cross-backend parity."""

    async def test_fts_boolean_mode(self) -> bool:
        """Test FTS boolean mode with AND/OR/NOT operators.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_boolean_mode'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if FTS is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for boolean mode tests
            bool_thread = f'{self.test_thread_id}_fts_boolean'

            # Store test contexts for boolean search
            test_contexts = [
                {'text': 'Python is great for data science and machine learning', 'source': 'agent'},
                {'text': 'JavaScript and TypeScript are popular for web development', 'source': 'agent'},
                {'text': 'Python and JavaScript can both handle backend development', 'source': 'user'},
                {'text': 'Rust is known for memory safety without garbage collection', 'source': 'agent'},
            ]

            for ctx in test_contexts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': bool_thread,
                        'source': ctx['source'],
                        'text': ctx['text'],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: OR operator - should find entries with Python OR JavaScript
            or_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'Python OR JavaScript',
                    'mode': 'boolean',
                    'thread_id': bool_thread,
                    'limit': 10,
                },
            )

            or_data = self._extract_content(or_result)

            if 'results' not in or_data:
                self.test_results.append((test_name, False, f'OR search failed: {or_data}'))
                return False

            or_results = or_data.get('results', [])
            # Should find at least 3 entries (2 with Python, 2 with JavaScript, 1 with both)
            if len(or_results) < 3:
                self.test_results.append(
                    (test_name, False, f'Expected at least 3 results for OR query, got {len(or_results)}'),
                )
                return False

            # Test 2: AND operator - should find entries with both Python AND data
            and_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'Python AND data',
                    'mode': 'boolean',
                    'thread_id': bool_thread,
                    'limit': 10,
                },
            )

            and_data = self._extract_content(and_result)

            if 'results' not in and_data:
                self.test_results.append((test_name, False, f'AND search failed: {and_data}'))
                return False

            and_results = and_data.get('results', [])
            # Should find exactly 1 entry with both Python AND data
            if len(and_results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result for AND query, got {len(and_results)}'),
                )
                return False

            # Verify response mode field
            if or_data.get('mode') != 'boolean':
                self.test_results.append((test_name, False, 'Response mode field incorrect'))
                return False

            self.test_results.append((test_name, True, 'Boolean mode OR/AND operators working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_search_not_operator_exclusion(self) -> bool:
        """Verify FTS boolean mode NOT operator excludes entries.

        SQLite FTS5 uses the NOT keyword in boolean mode, while PostgreSQL
        uses the '-' prefix via websearch_to_tsquery.

        Returns:
            bool: True if test passed.
        """
        test_name = 'fts_search_not_operator_exclusion'
        assert self.client is not None
        try:
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)
            fts_info = stats_data.get('fts', {})
            if not (fts_info.get('enabled') and fts_info.get('available')):
                self.test_results.append((test_name, True, 'Skipped (FTS unavailable)'))
                return True

            not_thread = f'{self.test_thread_id}_fts_not'

            await self.client.call_tool('store_context', {
                'thread_id': not_thread, 'source': 'agent',
                'text': 'Python web development with Django framework',
            })
            await self.client.call_tool('store_context', {
                'thread_id': not_thread, 'source': 'agent',
                'text': 'Python data science with pandas and numpy',
            })

            # Boolean NOT uses each backend's documented NATIVE syntax. The
            # server intentionally does NOT unify boolean operators across
            # backends (no native cross-engine syntax exists); the correct
            # per-backend syntax is published to clients via the dynamic tool
            # descriptions in app/tools/descriptions.py:
            #   SQLite FTS5  -> NOT keyword ('Python NOT Django')
            #   PostgreSQL   -> websearch '-' prefix ('Python -Django'); the
            #                   bare word 'NOT' is an English stop word there
            #                   (websearch_to_tsquery), so it is NOT an operator.
            not_query = 'Python -Django' if self.backend == 'postgresql' else 'Python NOT Django'
            result = await self.client.call_tool('fts_search_context', {
                'query': not_query, 'mode': 'boolean',
                'thread_id': not_thread, 'limit': 10,
            })
            data = self._extract_content(result)

            if 'results' not in data:
                self.test_results.append((test_name, False, f'NOT search failed: {data}'))
                return False

            results = data.get('results', [])
            if len(results) != 1:
                self.test_results.append((test_name, False,
                    f'Expected 1 result (Django excluded), got {len(results)}'))
                return False

            found_text = results[0].get('text_content', '')
            if 'Django' in found_text:
                self.test_results.append((test_name, False, 'NOT operator failed: Django entry included'))
                return False

            self.test_results.append((test_name, True, 'FTS NOT operator correctly excludes entries'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_search_malformed_boolean_parity(self) -> bool:
        """Verify malformed boolean FTS queries degrade gracefully on BOTH backends.

        Boolean mode hands the query to SQLite FTS5 unchanged, and FTS5 rejects a
        malformed boolean query (an unbalanced parenthesis) with 'fts5: syntax error',
        while PostgreSQL's tolerant websearch_to_tsquery returns results for the same
        input. Surfacing that error as a ToolError would be a cross-backend MCP-contract
        divergence for byte-identical arguments, so SQLite degrades a malformed boolean
        query to the crash-safe sanitized term match, and both backends return a result
        set (no hard ToolError). A well-formed boolean query works natively on both,
        exercising each backend's documented boolean syntax.

        Returns:
            bool: True if test passed.
        """
        test_name = 'fts_search_malformed_boolean_parity'
        assert self.client is not None
        try:
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)
            fts_info = stats_data.get('fts', {})
            if not (fts_info.get('enabled') and fts_info.get('available')):
                self.test_results.append((test_name, True, 'Skipped (FTS unavailable)'))
                return True

            mb_thread = f'{self.test_thread_id}_fts_malformed_boolean'
            await self.client.call_tool('store_context', {
                'thread_id': mb_thread, 'source': 'agent',
                'text': 'Structured error handling guidance for resilient services',
            })

            # Malformed boolean (unbalanced parenthesis): FTS5 rejects it with
            # 'fts5: syntax error' while PostgreSQL accepts it. Both must return a result
            # set (no hard ToolError) and still find the entry via best-effort term recall.
            malformed_result = await self.client.call_tool('fts_search_context', {
                'query': 'error AND (handling', 'mode': 'boolean',
                'thread_id': mb_thread, 'limit': 10,
            })
            malformed_data = self._extract_content(malformed_result)
            if 'results' not in malformed_data:
                self.test_results.append((test_name, False,
                    f'Malformed boolean did not return a result set: {malformed_data}'))
                return False
            if len(malformed_data.get('results', [])) < 1:
                self.test_results.append((test_name, False,
                    'Malformed boolean degraded to zero results (expected best-effort recall)'))
                return False

            # Well-formed boolean still works (each backend's native syntax: 'AND' is an FTS5
            # operator on SQLite and an ignored stop word on PostgreSQL websearch -- both AND
            # the surviving lexemes, so the entry matches on both).
            valid_result = await self.client.call_tool('fts_search_context', {
                'query': 'error AND handling', 'mode': 'boolean',
                'thread_id': mb_thread, 'limit': 10,
            })
            valid_data = self._extract_content(valid_result)
            if len(valid_data.get('results', [])) < 1:
                self.test_results.append((test_name, False,
                    f'Well-formed boolean returned no results: {valid_data}'))
                return False

            self.test_results.append((test_name, True,
                'Malformed boolean degrades gracefully on both backends; valid boolean unaffected'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_embedded_quote_term_parity(self) -> bool:
        """A bare FTS term carrying an embedded double quote matches identically on both backends.

        Escaping the quote by doubling it does NOT neutralize it: FTS5 re-tokenizes the
        contents of a string literal, the escape decodes back to a literal quote, and the
        tokenizer treats it as a word boundary -- silently turning an ordinary token into
        a strict two-word ADJACENCY phrase, so SQLite would return only the document whose
        words happen to be adjacent while PostgreSQL's plainto_tsquery ANDs the two
        lexemes with no adjacency requirement and returns both. Splitting the token on
        the quote into independently ANDed literals makes the two backends agree.

        The hyphen target is pinned as the complementary invariant: it stays an adjacency
        phrase on SQLite and a compound lexeme on PostgreSQL, so the NON-adjacent document
        must never match it on either backend.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_embedded_quote_term_parity'
        assert self.client is not None
        try:
            stats_data = self._extract_content(await self.client.call_tool('get_statistics', {}))
            fts_info = stats_data.get('fts', {})
            if not (fts_info.get('enabled') and fts_info.get('available')):
                self.test_results.append((test_name, True, 'Skipped (FTS not available)'))
                return True

            thread = f'{self.test_thread_id}_fts_quote'
            seeds = [
                ('alpha zulu beta', 'adjacent'),
                ('alpha somewhere else entirely zulu', 'separated'),
            ]
            ids: dict[str, str] = {}
            for text, label in seeds:
                stored = self._extract_content(await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent', 'text': text,
                }))
                if not stored.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store {label} seed: {stored}'))
                    return False
                ids[label] = str(stored['context_id'])

            quoted = self._extract_content(await self.client.call_tool('fts_search_context', {
                'query': 'alpha"zulu', 'mode': 'match', 'thread_id': thread, 'limit': 10,
            }))
            quoted_ids = {str(row.get('id')) for row in quoted.get('results', [])}
            if quoted_ids != set(ids.values()):
                self.test_results.append((
                    test_name, False,
                    f'Query alpha"zulu returned {len(quoted_ids)} of 2 documents on {self.backend}: {quoted_ids}',
                ))
                return False

            hyphen = self._extract_content(await self.client.call_tool('fts_search_context', {
                'query': 'alpha-zulu', 'mode': 'match', 'thread_id': thread, 'limit': 10,
            }))
            hyphen_ids = {str(row.get('id')) for row in hyphen.get('results', [])}
            if ids['separated'] in hyphen_ids:
                self.test_results.append((
                    test_name, False,
                    'Query alpha-zulu matched the non-adjacent document (the hyphen lost its adjacency meaning)',
                ))
                return False

            self.test_results.append((
                test_name, True, 'An embedded quote ANDs its fragments on both backends; the hyphen stays adjacency-only',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_deeply_nested_boolean_query_degrades(self) -> bool:
        """A deeply nested boolean query degrades instead of erroring, and spares the breaker.

        Boolean mode forwards the client's query to the engine verbatim, and SQLite's
        FTS5 rejects a deeply nested expression with its own parser message, outside any
        short list of grammar messages a classifier could enumerate as client errors.
        Read as a server fault, it would propagate as a hard error where PostgreSQL's
        tolerant websearch parser succeeds on the same input, and it would charge the
        PROCESS-GLOBAL circuit breaker, so a client repeating one malformed query could
        open the breaker and have every other caller's reads and writes rejected.
        Failure attribution is therefore inverted -- the database-fault families are the
        closed set and everything else is attributed to the one client-controlled
        fragment -- so the query degrades to the sanitized term match on SQLite, matches
        on PostgreSQL, and neither backend charges a failure.

        The repetition afterwards is the point of the breaker half: it exceeds the
        consecutive-failure threshold that charged failures would trip, and ordinary
        traffic on the SAME server must still succeed.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_deeply_nested_boolean_query_degrades'
        assert self.client is not None
        try:
            stats_data = self._extract_content(await self.client.call_tool('get_statistics', {}))
            fts_info = stats_data.get('fts', {})
            if not (fts_info.get('enabled') and fts_info.get('available')):
                self.test_results.append((test_name, True, 'Skipped (FTS not available)'))
                return True

            thread = f'{self.test_thread_id}_fts_nested'
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Deeply nested boolean probe describing an error condition in the ingest pipeline',
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {stored}'))
                return False

            nesting = 100
            query = '(' * nesting + 'error' + ')' * nesting
            repeats = 12
            for attempt in range(repeats):
                data = self._extract_content(await self.client.call_tool('fts_search_context', {
                    'query': query, 'mode': 'boolean', 'thread_id': thread, 'limit': 10,
                }))
                if 'results' not in data:
                    self.test_results.append((
                        test_name, False, f'Nested boolean call {attempt} returned no result set: {data}',
                    ))
                    return False
                if not data.get('results'):
                    self.test_results.append((
                        test_name, False, f'Nested boolean call {attempt} degraded to zero results',
                    ))
                    return False

            browse = self._extract_content(await self.client.call_tool('search_context', {
                'thread_id': thread, 'limit': 10,
            }))
            if not browse.get('results'):
                self.test_results.append((
                    test_name, False, f'Ordinary browse failed after {repeats} nested queries: {browse}',
                ))
                return False
            follow_up = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Write issued after the repeated nested boolean queries',
            }))
            if not follow_up.get('success'):
                self.test_results.append((
                    test_name, False, f'Ordinary write failed after {repeats} nested queries: {follow_up}',
                ))
                return False

            after = self._extract_content(await self.client.call_tool('get_statistics', {}))
            circuit_state = str(after.get('connection_metrics', {}).get('circuit_state'))
            if circuit_state != 'healthy':
                self.test_results.append((
                    test_name, False, f'circuit_state is {circuit_state!r} after {repeats} nested queries',
                ))
                return False

            self.test_results.append((
                test_name, True,
                f'{repeats} nested boolean queries returned results and left the breaker healthy on {self.backend}',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
