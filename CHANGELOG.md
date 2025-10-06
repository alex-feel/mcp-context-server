# Changelog

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
