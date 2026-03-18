# Changelog

## [2.1.0](https://github.com/alex-feel/mcp-context-server/compare/v2.0.0...v2.1.0) (2026-03-18)


### Features

* add automatic model pulling for Ollama providers on startup ([3196c2b](https://github.com/alex-feel/mcp-context-server/commit/3196c2bc2eb236908202ce44dffbdae7913f3ccc))
* add context window validation for summary generation ([7ea909e](https://github.com/alex-feel/mcp-context-server/commit/7ea909ed68a2203022315754bc30aaac8b821fa7))
* add eager model loading and concurrency protection for FlashRank reranking ([f260d27](https://github.com/alex-feel/mcp-context-server/commit/f260d272fb63392350abfb76f856eb771ae8a9ea))
* add Ollama cold-start optimization with model pre-warming ([41e2776](https://github.com/alex-feel/mcp-context-server/commit/41e2776d8d40963d38e2cf339325da822655659d))
* add SHA-256 content hash for deduplication optimization ([76ec45f](https://github.com/alex-feel/mcp-context-server/commit/76ec45f24d805c3055f4b191131528c013a966b8))
* add Skill Integration section to DEFAULT_INSTRUCTIONS ([7b84ae0](https://github.com/alex-feel/mcp-context-server/commit/7b84ae07cac3c72c4c6200a0d79379b5a70bfd23))
* change default summary model to qwen3:0.6b ([c5c7c0e](https://github.com/alex-feel/mcp-context-server/commit/c5c7c0e4ee63a58fd8aee1ead2c6f15f728cb44a))


### Bug Fixes

* classify PostgreSQL initialization errors and limit Docker restart retries ([ef6e58a](https://github.com/alex-feel/mcp-context-server/commit/ef6e58ad10327e8f4cd63318646f7d8542d48548))
* eliminate redundant PostgreSQL schema execution ([976d625](https://github.com/alex-feel/mcp-context-server/commit/976d625d3274e88a13f6c5ed252710cbc5c4673b))
* enforce uniform Generation-First Transactional Integrity across all tools ([a1bb652](https://github.com/alex-feel/mcp-context-server/commit/a1bb652795f76012893cfad8fb82d627868695d1))
* replace per-call-site JSON string deserialization with schema-aware FastMCP middleware ([f706b5b](https://github.com/alex-feel/mcp-context-server/commit/f706b5bae9e9d6b1789276b1cb01682740cd0949))
* replace provider-availability checks with actual generation counters in batch tool messages ([a473919](https://github.com/alex-feel/mcp-context-server/commit/a4739190f3d81662675f11383693d3ab1a20d625))
* resolve &lt;unknown&gt; in tenacity retry log messages ([91e4bb4](https://github.com/alex-feel/mcp-context-server/commit/91e4bb4bbb4f9548053b96fc9f2121838fc604bb))
* resolve Docker and Helm deployment issues with model pull and summary support ([1c893d1](https://github.com/alex-feel/mcp-context-server/commit/1c893d165c57c8c7467539051e852590fa5af9e8))
* use source-aware dynamic prompts for summary generation ([5c80bea](https://github.com/alex-feel/mcp-context-server/commit/5c80bea54497d45c812b0a821ab01f40527089a2))

## [2.0.0](https://github.com/alex-feel/mcp-context-server/compare/v1.7.1...v2.0.0) (2026-03-13)


### ⚠ BREAKING CHANGES

* This release introduces two breaking changes:

### Features

* add LLM-powered summary generation for context entries ([9f4c9e2](https://github.com/alex-feel/mcp-context-server/commit/9f4c9e2706beedbc25946365ce00d73eb9c6a5b9))

## [1.7.1](https://github.com/alex-feel/mcp-context-server/compare/v1.7.0...v1.7.1) (2026-03-09)


### Bug Fixes

* change FASTMCP_STATELESS_HTTP default from false to true ([277baa3](https://github.com/alex-feel/mcp-context-server/commit/277baa3b8cacf82f93d0f5a47abe0857a73c6bcd))
* prevent event loop blocking in FlashRank model loading ([b0076fd](https://github.com/alex-feel/mcp-context-server/commit/b0076fdabc23622d50d6a1b8d243daf2cfc23054))

## [1.7.0](https://github.com/alex-feel/mcp-context-server/compare/v1.6.0...v1.7.0) (2026-03-08)


### Features

* add configurable micro-batching for FlashRank reranking ([43fcd9c](https://github.com/alex-feel/mcp-context-server/commit/43fcd9c12a4654127b756dd89fdb565500c08304))


### Bug Fixes

* apply ts_headline only to LIMIT'd rows in PostgreSQL FTS ([2cdb52e](https://github.com/alex-feel/mcp-context-server/commit/2cdb52e808cef57a35226e6bcdf5bae957a0799a))
* improve tool description clarity and self-sufficiency ([ed522c3](https://github.com/alex-feel/mcp-context-server/commit/ed522c3754a46a7665cfab4eb23ffd3ec470c4af))
* offload FlashRank ONNX inference to thread pool via asyncio.to_thread ([272eb9e](https://github.com/alex-feel/mcp-context-server/commit/272eb9e8b605e837a743c0697b0f818091ecf547))
* report project version in MCP protocol handshake ([0372f58](https://github.com/alex-feel/mcp-context-server/commit/0372f58f4d1e5a2a6cdeb5c1d05762225e9f6a9c))
* skip auth initialization on stdio transport ([7fbc9ef](https://github.com/alex-feel/mcp-context-server/commit/7fbc9ef8ba1a7572cba1d79ae820520ee9000c46))

## [1.6.0](https://github.com/alex-feel/mcp-context-server/compare/v1.5.0...v1.6.0) (2026-03-04)


### Features

* add adaptive FTS mode to hybrid search for improved long-query recall ([54c5f31](https://github.com/alex-feel/mcp-context-server/commit/54c5f31b1d1c875ef72c6b647096a802190d887b))
* add RERANKING_CPU_MEM_ARENA env var to control ONNX memory arena ([5f98cf7](https://github.com/alex-feel/mcp-context-server/commit/5f98cf7e79af9ae321ac3cdce298bb73b4f4f711))


### Bug Fixes

* add temporary monkey-patches for MCP SDK session crash on client disconnect ([d9a9f3d](https://github.com/alex-feel/mcp-context-server/commit/d9a9f3d21ae66b98ae61100a394a6485d0c4f5a3))
* correct deduplication data integrity in store_context and store_context_batch ([a395ede](https://github.com/alex-feel/mcp-context-server/commit/a395ede0378b1342d774846d4adaae11f57f9944))
* correct modes_used to reflect execution rather than results in hybrid search ([c747edf](https://github.com/alex-feel/mcp-context-server/commit/c747edfd46ccbd933285c68aac4d7879249527e0))

## [1.5.0](https://github.com/alex-feel/mcp-context-server/compare/v1.4.0...v1.5.0) (2026-03-02)


### Features

* add configurable ONNX intra-op thread control for FlashRank reranking ([1352988](https://github.com/alex-feel/mcp-context-server/commit/1352988939b92e5c1cb2ad251a09f5e17f8b98e9))
* add embedding concurrency control, dynamic timeout, and search quality improvements ([757d19f](https://github.com/alex-feel/mcp-context-server/commit/757d19fdbf658b1990a039e45e27a2d7ba1e546d))


### Bug Fixes

* sanitize hybrid search warning messages to avoid leaking error details ([d71b0c9](https://github.com/alex-feel/mcp-context-server/commit/d71b0c977af29735008e5d9e8b20149eaa12c615))
* suppress onnxruntime type errors via mypy and pyright config ([e59e9de](https://github.com/alex-feel/mcp-context-server/commit/e59e9de46b101af2516ee0f22905ccb018b93c9c))

## [1.4.0](https://github.com/alex-feel/mcp-context-server/compare/v1.3.3...v1.4.0) (2026-02-27)


### Features

* add FASTMCP_STATELESS_HTTP setting for horizontal scaling ([c94a423](https://github.com/alex-feel/mcp-context-server/commit/c94a42357b735eb417f92c57c3a2851e535cac8e))
* add MCP server instructions support ([e1805a6](https://github.com/alex-feel/mcp-context-server/commit/e1805a6dfb3c1344d596547c9c0af3af8e124fc1))
* remove search_modes parameter from hybrid_search_context ([2335d0c](https://github.com/alex-feel/mcp-context-server/commit/2335d0cc6ab5bd72e5cd76983cef87d385f7cc79))


### Bug Fixes

* remove install hints from inner exceptions in providers and services ([e9708b2](https://github.com/alex-feel/mcp-context-server/commit/e9708b2245097f5d0623bae11cee3210efee145a))
* replace session-scoped advisory locks with transaction-scoped locks in migrations ([4508eb8](https://github.com/alex-feel/mcp-context-server/commit/4508eb81918eca61ce40f5141d79119a3adf6c97))
* update dependencies to resolve Trivy CVE findings ([c21d736](https://github.com/alex-feel/mcp-context-server/commit/c21d7362bb8f0e1e719c58bb984a58a316e099bd))

## [1.3.3](https://github.com/alex-feel/mcp-context-server/compare/v1.3.2...v1.3.3) (2026-02-11)


### Bug Fixes

* add advisory locks to PostgreSQL migrations and schema initialization ([d637cc5](https://github.com/alex-feel/mcp-context-server/commit/d637cc547d32fa5ab565930abe57321f68aa3ec9))
* add PostgreSQL connection resilience defensive hardening ([e0ae6da](https://github.com/alex-feel/mcp-context-server/commit/e0ae6da0becdea7f93da63596287c116fa8f3131))
* add ROLLBACK to PostgreSQL backend ([bb69f45](https://github.com/alex-feel/mcp-context-server/commit/bb69f45dd0129dfcbb8c48d57e60571ab2e3c1d8))

## [1.3.2](https://github.com/alex-feel/mcp-context-server/compare/v1.3.1...v1.3.2) (2026-02-06)


### Bug Fixes

* add python-multipart constraint for CVE-2026-24486 ([9c6d2ea](https://github.com/alex-feel/mcp-context-server/commit/9c6d2ea7d39f7f2f579559ccdbc32c42031afa8f))
* resolve embedding deduplication race condition ([89f6596](https://github.com/alex-feel/mcp-context-server/commit/89f6596d18bbc10e18662ed86f62c46fe80fd5a1))
* resolve PostgreSQL migration timeout and idempotency issues ([d2f8f8e](https://github.com/alex-feel/mcp-context-server/commit/d2f8f8e3787c891ef6bde67652ec9fb3c8522406))

## [1.3.1](https://github.com/alex-feel/mcp-context-server/compare/v1.3.0...v1.3.1) (2026-01-25)


### Bug Fixes

* classify PostgreSQL backend errors to control container restart behavior ([f5ed4c8](https://github.com/alex-feel/mcp-context-server/commit/f5ed4c8ac0cec0d7ead6d85d018dcb7dc44bc1ea))
* upgrade FastMCP to 2.14.4 ([8780345](https://github.com/alex-feel/mcp-context-server/commit/8780345f35fee558af266f9d33c226592ccdf9e0))

## [1.3.0](https://github.com/alex-feel/mcp-context-server/compare/v1.2.1...v1.3.0) (2026-01-20)


### Features

* add Pgpool-II detection for PostgreSQL backend ([e64f082](https://github.com/alex-feel/mcp-context-server/commit/e64f0826238417bfd5624de10d94dd31b3da83cf))

## [1.2.1](https://github.com/alex-feel/mcp-context-server/compare/v1.2.0...v1.2.1) (2026-01-18)


### Bug Fixes

* add backend-specific FTS tool descriptions ([09113cd](https://github.com/alex-feel/mcp-context-server/commit/09113cd9fef0e5aec83c9dbc73be2b77f3b311e0))

## [1.2.0](https://github.com/alex-feel/mcp-context-server/compare/v1.1.0...v1.2.0) (2026-01-17)


### Features

* add configurable asyncpg prepared statement cache settings ([5ebf287](https://github.com/alex-feel/mcp-context-server/commit/5ebf2876bbee7c089f7b3901a54c9eebb8fac281))

## [1.1.0](https://github.com/alex-feel/mcp-context-server/compare/v1.0.0...v1.1.0) (2026-01-17)


### Features

* implement embedding-first transactional integrity ([c9e4c12](https://github.com/alex-feel/mcp-context-server/commit/c9e4c12957018510732a7bf7815c2f98ebe9e88b))

## [1.0.0](https://github.com/alex-feel/mcp-context-server/compare/v0.17.0...v1.0.0) (2026-01-16)


### ⚠ BREAKING CHANGES

* API response structure changed for all search tools. Users must update their code:
    - FTS: result['score'] -> result['scores']['fts_score']
    - Semantic: result['distance'] -> result['scores']['semantic_distance']
    - All tools: result['rerank_score'] -> result['scores']['rerank_score']
* Default embedding model changed from embeddinggemma:latest (768 dim) to qwen3-embedding:0.6b (1024 dim). Existing vector databases will have incompatible embeddings.
* ENABLE_EMBEDDING_GENERATION now defaults to true. Server will NOT start if embedding dependencies are not available when ENABLE_EMBEDDING_GENERATION=true (the default).

### Features

* add chunk-aware reranking for FTS, semantic search, and hybrid search ([ff79859](https://github.com/alex-feel/mcp-context-server/commit/ff79859a65645be5fe83b50054994ee1ac352f48))
* add embedding truncation control with universal validator ([d941b6f](https://github.com/alex-feel/mcp-context-server/commit/d941b6f349ddac787abfee581e5f22e6b8103579))
* add text chunking and cross-encoder reranking ([942bb24](https://github.com/alex-feel/mcp-context-server/commit/942bb24d66330017864d91d1a73d07f9df7920ed))
* implement universal retry wrapper for embedding providers ([4031061](https://github.com/alex-feel/mcp-context-server/commit/40310619dfdf09be9ef1ca24bebbff8160ac5a92))
* prevent Docker infinite restart loops with exit code handling ([91acfa4](https://github.com/alex-feel/mcp-context-server/commit/91acfa4c00cd979577ee2157619a7b495caff8d7))
* replace embeddinggemma with qwen3-embedding:0.6b ([02ce75e](https://github.com/alex-feel/mcp-context-server/commit/02ce75e3833eb7150e873f33e670c8809d26da6c))
* separate embedding generation from semantic search ([3967262](https://github.com/alex-feel/mcp-context-server/commit/396726205d2a681e939d1a0fe36ba96c2822b7aa))
* standardize scores API across all search tools ([0a940c9](https://github.com/alex-feel/mcp-context-server/commit/0a940c9d134b44a8e7d1c3e4fca637aad0b1566a))
* switch PostgreSQL FTS ranking from ts_rank to ts_rank_cd ([a58fd9c](https://github.com/alex-feel/mcp-context-server/commit/a58fd9c9e4d24e91cc26eb53b31732fa4aa4a845))


### Bug Fixes

* enable LangSmith tracing for embedding operations ([ba7bf9f](https://github.com/alex-feel/mcp-context-server/commit/ba7bf9fc12a66e673dac54d9d947970b5f380587))

## [0.17.0](https://github.com/alex-feel/mcp-context-server/compare/v0.16.1...v0.17.0) (2026-01-11)


### Features

* implement LangChain embeddings multi-provider architecture ([7382629](https://github.com/alex-feel/mcp-context-server/commit/7382629a74da6fb39d5e635096d7eea1c619e3c8))


### Bug Fixes

* add schema qualification to recursive jsonb_merge_patch call ([51bc48c](https://github.com/alex-feel/mcp-context-server/commit/51bc48c8ba86e4ae990b7bb3ef5efa15c8e1a6b7))
* resolve critical PostgreSQL backend issues ([8186f90](https://github.com/alex-feel/mcp-context-server/commit/8186f90da0804f5adf858cf857d25e8b81468649))

## [0.16.1](https://github.com/alex-feel/mcp-context-server/compare/v0.16.0...v0.16.1) (2026-01-09)


### Bug Fixes

* add POSTGRESQL_SCHEMA setting and refactor metadata index management ([772f3aa](https://github.com/alex-feel/mcp-context-server/commit/772f3aa6c7d31945e69fd3ed31ddc69ae7da1454))
* improve error handling and add timeout/retry logic ([319e08f](https://github.com/alex-feel/mcp-context-server/commit/319e08fdca103e2e835c9318984c10e931db6e75))
* replace hardcoded public schema with POSTGRESQL_SCHEMA setting ([3e6511f](https://github.com/alex-feel/mcp-context-server/commit/3e6511fc927ff9c0c7c55480c4841cd6b31eddfb))

## [0.16.0](https://github.com/alex-feel/mcp-context-server/compare/v0.15.1...v0.16.0) (2026-01-06)


### Features

* add configurable metadata field indexing with sync modes ([10ab4cb](https://github.com/alex-feel/mcp-context-server/commit/10ab4cb5924ae7230c146e39a8786a755b3f44fa))

## [0.15.1](https://github.com/alex-feel/mcp-context-server/compare/v0.15.0...v0.15.1) (2026-01-05)


### Bug Fixes

* add array_contains to MCP tool metadata_filters descriptions ([45f2357](https://github.com/alex-feel/mcp-context-server/commit/45f23572fe6250c570f0d41f272996311f617857))

## [0.15.0](https://github.com/alex-feel/mcp-context-server/compare/v0.14.1...v0.15.0) (2026-01-05)


### Features

* add array_contains operator with graceful non-array handling ([b23642a](https://github.com/alex-feel/mcp-context-server/commit/b23642a2933ac267fbbb9519062a252bef64d8c8))

## [0.14.1](https://github.com/alex-feel/mcp-context-server/compare/v0.14.0...v0.14.1) (2025-12-30)


### Bug Fixes

* add search_path to functions for CVE-2018-1058 mitigation ([d156235](https://github.com/alex-feel/mcp-context-server/commit/d156235547f0ab5b5837022bb91bbdfdba94edf9))

## [0.14.0](https://github.com/alex-feel/mcp-context-server/compare/v0.13.0...v0.14.0) (2025-12-28)


### Features

* add HTTP authentication support with bearer token and OAuth options ([e625b42](https://github.com/alex-feel/mcp-context-server/commit/e625b42add2360f1bd9212690182b0b8474b03f8))

## [0.13.0](https://github.com/alex-feel/mcp-context-server/compare/v0.12.0...v0.13.0) (2025-12-27)


### Features

* add DISABLED_TOOLS environment variable and MCP tool annotations ([f6bccd7](https://github.com/alex-feel/mcp-context-server/commit/f6bccd7593c0bc2e9220299b5a9325ed76323850))

## [0.12.0](https://github.com/alex-feel/mcp-context-server/compare/v0.11.0...v0.12.0) (2025-12-26)


### Features

* add compose configuration for external PostgreSQL ([61b294f](https://github.com/alex-feel/mcp-context-server/commit/61b294f5c3112ffd46d4472146cac317f09aa245))
* add Docker deployment with HTTP transport support ([77390e9](https://github.com/alex-feel/mcp-context-server/commit/77390e9b552cd7b76d31bd0c89cccd699e65bb68))
* split compose files for independent SQLite and PostgreSQL deployments ([7840291](https://github.com/alex-feel/mcp-context-server/commit/784029108dd798a0dde86ba417dd8007876cda6b))

## [0.11.0](https://github.com/alex-feel/mcp-context-server/compare/v0.10.0...v0.11.0) (2025-12-22)


### Features

* standardize search tools API response structure ([04d7472](https://github.com/alex-feel/mcp-context-server/commit/04d7472dec0cb23e292b5405412c9aec0029f9d6))


### Bug Fixes

* add explain_query support to semantic search for consistent stats structure ([cd5884b](https://github.com/alex-feel/mcp-context-server/commit/cd5884b490f43df22150fa338a786c2af957ec22))
* remove duplicate Args sections from MCP tool docstrings ([9e0df55](https://github.com/alex-feel/mcp-context-server/commit/9e0df55b71ffd33154c30ba829030298730d7169))
* replace blocking pathlib.Path with anyio.Path in async code ([0101b0a](https://github.com/alex-feel/mcp-context-server/commit/0101b0a32566a790a89b62ba5ff42b3c6770713d))
* standardize MCP tool docstrings to use inline JSON structures ([7bef90d](https://github.com/alex-feel/mcp-context-server/commit/7bef90d55c2c08dcc79b94c2b2d2b20b80871cb7))

## [0.10.0](https://github.com/alex-feel/mcp-context-server/compare/v0.9.0...v0.10.0) (2025-12-21)


### Features

* add explain_query parameter to fts_search_context ([7334e9f](https://github.com/alex-feel/mcp-context-server/commit/7334e9fef5ac88f7806cc5ed9754a219fe703d5c))
* add explain_query parameter to hybrid_search_context ([b32129d](https://github.com/alex-feel/mcp-context-server/commit/b32129d6a8b60a45575aefc69a5536044ab2cd09))
* add hybrid search with RRF fusion ([8c6ea15](https://github.com/alex-feel/mcp-context-server/commit/8c6ea15ae7867880183a9ac93e855e586069838f))
* add offset, content_type, include_images to search tools ([bb22012](https://github.com/alex-feel/mcp-context-server/commit/bb2201257c20e4c645cc3f70863d8e8de133f55e))
* add tags parameter to semantic, FTS, and hybrid search ([bcf1332](https://github.com/alex-feel/mcp-context-server/commit/bcf1332f5015fae69be640e062f45fd60534fffd))
* add uniform backend display to all statistics output ([0d7c988](https://github.com/alex-feel/mcp-context-server/commit/0d7c988a25693bdced56cc89cb6235dda5f5d2f0))
* rename top_k to limit in semantic_search_context ([2c27c14](https://github.com/alex-feel/mcp-context-server/commit/2c27c1418e0055fdf835055c1efb87b37d3e4b0a))
* standardize limit parameter across all search tools ([d85d07d](https://github.com/alex-feel/mcp-context-server/commit/d85d07de4aada18d4c90375f4e17589692dc4d82))


### Bug Fixes

* handle hyphenated words in FTS queries ([b2f9ccb](https://github.com/alex-feel/mcp-context-server/commit/b2f9ccbf94ff1fa5ca875843504edff593de0d2c))
* hybrid search pagination and test quality improvements ([2da98fe](https://github.com/alex-feel/mcp-context-server/commit/2da98fe5a31d594832916ef6a968245d821fb6a1))
* make startup checks backend-specific ([3ee1149](https://github.com/alex-feel/mcp-context-server/commit/3ee1149638acdc8869ef8755aadbcb5f14dc9e04))
* register hybrid_search_context tool dynamically ([0ee0d22](https://github.com/alex-feel/mcp-context-server/commit/0ee0d22f9d1743eb51f5516facad6dcd74a7abae))

## [0.9.0](https://github.com/alex-feel/mcp-context-server/compare/v0.8.0...v0.9.0) (2025-12-03)


### Features

* add full-text search with FTS5 and PostgreSQL tsvector ([34ef13f](https://github.com/alex-feel/mcp-context-server/commit/34ef13f5c39f82dae84d41784dfc1d055dd63fa4))


### Bug Fixes

* handle boolean metadata filtering for PostgreSQL JSONB ([1ff407e](https://github.com/alex-feel/mcp-context-server/commit/1ff407e6891e3aec579aef2c531ac33f3e2b2452))

## [0.8.0](https://github.com/alex-feel/mcp-context-server/compare/v0.7.0...v0.8.0) (2025-11-30)


### Features

* add bulk operations for batch context management ([7516a93](https://github.com/alex-feel/mcp-context-server/commit/7516a935cef9fbe7bed5d7bc55dbf0001b41b830))
* add metadata filtering to semantic_search_context ([50346d6](https://github.com/alex-feel/mcp-context-server/commit/50346d64dccc067925e405b9ebbfa24dee94f459))


### Bug Fixes

* handle integer arrays in in/not_in metadata operators ([23076df](https://github.com/alex-feel/mcp-context-server/commit/23076df01b778464a1c4189e698d0782aacdfc28))

## [0.7.0](https://github.com/alex-feel/mcp-context-server/compare/v0.6.0...v0.7.0) (2025-11-29)


### Features

* support date filtering in search_context and semantic_search_context ([e433f5e](https://github.com/alex-feel/mcp-context-server/commit/e433f5e195b8bd3278f55ef4685d7be0844b3c2b))

## [0.6.0](https://github.com/alex-feel/mcp-context-server/compare/v0.5.1...v0.6.0) (2025-11-29)


### Features

* add metadata_patch parameter for partial metadata updates ([42f7a5f](https://github.com/alex-feel/mcp-context-server/commit/42f7a5fc33cd0b544db97b5de71406bbe78c8beb))

## [0.5.1](https://github.com/alex-feel/mcp-context-server/compare/v0.5.0...v0.5.1) (2025-11-26)


### Bug Fixes

* resolve Supabase and PostgreSQL issues ([3d5c450](https://github.com/alex-feel/mcp-context-server/commit/3d5c450b19311fbf2e25cc73339b5b9f2dcb2d4e))

## [0.5.0](https://github.com/alex-feel/mcp-context-server/compare/v0.4.1...v0.5.0) (2025-11-23)


### Features

* add PostgreSQL backend with pgvector semantic search ([03ad9b0](https://github.com/alex-feel/mcp-context-server/commit/03ad9b023ead730805bc448cf3d3ee7a7ea0f58f))
* add storage backend abstraction for multi-database support ([4e7744a](https://github.com/alex-feel/mcp-context-server/commit/4e7744a6924be6c1f4cd0448b54d944e7c6d6848))


### Bug Fixes

* eliminate duplicate tool registration and improve naming ([d46cb7c](https://github.com/alex-feel/mcp-context-server/commit/d46cb7c3b524a3609b9f78801a01c09f27429f79))
* eliminate redundant backend initializations during startup ([dcd14d7](https://github.com/alex-feel/mcp-context-server/commit/dcd14d73c2ab3b8854edb7e269aac5f58577470a))
* load sqlite-vec extension before semantic search migration ([3069460](https://github.com/alex-feel/mcp-context-server/commit/306946028d2a6a6ae86a1d6d8368036ebb19d2c9))
* resolve asyncio primitives event loop binding issue ([705fd9d](https://github.com/alex-feel/mcp-context-server/commit/705fd9d37af693298d482b787f7feb46b79beafe))
* resolve integration test hang with persistent backend ([35219db](https://github.com/alex-feel/mcp-context-server/commit/35219dbc79dc2d8bc823d00032773ccc02e2c345))

## [0.4.1](https://github.com/alex-feel/mcp-context-server/compare/v0.4.0...v0.4.1) (2025-10-10)


### Bug Fixes

* allow nested JSON structures in metadata ([7f624ee](https://github.com/alex-feel/mcp-context-server/commit/7f624ee82dd3ad6292f583bf04e0cd815d6e1ecf))

## [0.4.0](https://github.com/alex-feel/mcp-context-server/compare/v0.3.0...v0.4.0) (2025-10-06)


### Features

* add semantic search with EmbeddingGemma and sqlite-vec ([2e0d3db](https://github.com/alex-feel/mcp-context-server/commit/2e0d3db3616da98e8da418f7391c452a722aa3fa))
* enable configurable embedding dimensions for Ollama models ([2d68963](https://github.com/alex-feel/mcp-context-server/commit/2d68963de47d10c54442c8454e855792f388deae))


### Bug Fixes

* correct semantic search filtering with CTE-based pre-filtering ([66161a3](https://github.com/alex-feel/mcp-context-server/commit/66161a357b1a06e51058939135611063a4c1123f))
* resolve type checking errors for optional dependencies ([be47f9d](https://github.com/alex-feel/mcp-context-server/commit/be47f9d7f33b5634f68e922a05e60154af881091))

## [0.3.0](https://github.com/alex-feel/mcp-context-server/compare/v0.2.0...v0.3.0) (2025-10-04)


### Features

* add update_context tool for modifying existing context entries ([08aed11](https://github.com/alex-feel/mcp-context-server/commit/08aed11af11e4d1e476181a7885e7d90e7ad08a0))


### Bug Fixes

* enforce Pydantic validation and resolve test reliability issues ([6137efc](https://github.com/alex-feel/mcp-context-server/commit/6137efc83af36cf162a941836ede78811d68530b))
* ensure consistent validation patterns across all MCP tools ([7137aca](https://github.com/alex-feel/mcp-context-server/commit/7137acade3f6d1b7af1f98461ddab8cd80bb1e4e))
* move validation to Pydantic models ([1e2e480](https://github.com/alex-feel/mcp-context-server/commit/1e2e4803e94dc7d3e7f0c9965dfcfc3f727af17c))
* resolve all pre-commit issues and test failures ([0a2142d](https://github.com/alex-feel/mcp-context-server/commit/0a2142dc8a3d16b967693038982825af277ae82b))

## [0.2.0](https://github.com/alex-feel/mcp-context-server/compare/v0.1.0...v0.2.0) (2025-09-28)


### Features

* add comprehensive metadata filtering to search_context ([e22cfe0](https://github.com/alex-feel/mcp-context-server/commit/e22cfe0fac6294725d423823bdc2d5ff802f88f5))


### Bug Fixes

* improve metadata filtering error handling and query plan serialization ([faa25b6](https://github.com/alex-feel/mcp-context-server/commit/faa25b6b9ba84d20768a4377e0736ad19b7a8f86))
* remove REGEX operator and fix case sensitivity for string operators ([b6d3534](https://github.com/alex-feel/mcp-context-server/commit/b6d3534d64ec467a498cb4f7fa3c588462d950fe))

## 0.1.0 (2025-09-25)


### ⚠ BREAKING CHANGES

* add initial version

### Features

* add initial version ([ac17f19](https://github.com/alex-feel/mcp-context-server/commit/ac17f19b3cc0d6d23aaf6820c73abe588ac75da4))
