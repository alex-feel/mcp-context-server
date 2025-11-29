# Changelog

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
