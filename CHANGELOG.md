# Changelog

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
