# Contributing to MCP Context Server

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Run pre-commit hooks** before committing
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

## Development Workflow

### Local Development Setup

```bash
# 1. Clone and setup
git clone https://github.com/alex-feel/mcp-context-server.git
cd mcp-context-server
uv sync

# 2. Run tests
uv run pytest                         # Unit tests
uv run python run_integration_test.py # Integration tests

# 3. Test server locally
uv run python -m app.server           # Should start without errors
# Press Ctrl+C to stop

# 4. Test published version (optional)
uvx mcp-context-server                # Run from PyPI without cloning
```

### Making Changes

After code changes:
1. Test your changes: `uv run pytest`
2. Run code quality checks: `uv run pre-commit run --all-files`

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run specific test file
uv run pytest tests/test_server.py

# Run integration tests only
uv run pytest -m integration

# Skip slow tests for quick feedback
uv run pytest -m "not integration"
```

### Code Quality

```bash
# Run pre-commit hooks on all files, including Ruff, mypy, and pyright
uv run pre-commit run --all-files
```

## Commit Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for all commit messages. This enables automated versioning and changelog generation.

### Commit Types

#### `feat`: New Features
Introduces new functionality or capabilities to the MCP Context Server.

Examples:
- `feat: add batch context retrieval endpoint`
- `feat: implement context search by date range`
- `feat: support WebP image format in multimodal storage`

#### `fix`: Bug Fixes
Fixes issues or bugs in the existing codebase.

Examples:
- `fix: resolve database lock on concurrent writes`
- `fix: handle invalid base64 image data gracefully`
- `fix: prevent memory leak in connection pool`

#### `chore`: Maintenance Tasks
Updates dependencies, refactors code, or performs housekeeping tasks.

Examples:
- `chore: update FastMCP to version 2.13`
- `chore: reorganize repository module structure`
- `chore: clean up unused test fixtures`

#### `docs`: Documentation
Improves or updates documentation, including README, API docs, or code comments.

Examples:
- `docs: add examples for multimodal context storage`
- `docs: update MCP client configuration guide`
- `docs: clarify thread-based context scoping in architecture`

#### `ci`: CI/CD Changes
Modifies continuous integration, deployment pipelines, or automation workflows.

Examples:
- `ci: add automated PyPI release workflow`
- `ci: configure pre-commit hooks for type checking`
- `ci: enable coverage reporting in GitHub Actions`

#### `test`: Testing
Adds or modifies tests, including unit, integration, or end-to-end tests.

Examples:
- `test: add integration tests for thread isolation`
- `test: implement concurrent write test scenarios`
- `test: validate multimodal context deduplication`

### Version Bump Rules

The commit type determines how the version number is incremented:

- **`feat:`** → Minor version bump (`0.x.0`)
- **`fix:`** → Patch version bump (`0.0.x`)
- **`feat!`**, **`fix!`**, or **`BREAKING CHANGE`** → Major version bump (`x.0.0`)

### Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Examples:**

```
feat: add support for compressed image storage

Implements automatic image compression for attachments larger than 1MB, reducing database size by up to 70% while maintaining visual quality.

Closes #42
```

```
fix: resolve race condition in repository initialization

The database connection manager now uses a lock to prevent concurrent initialization attempts during high load.
```

## Architecture Overview

### MCP Protocol Integration

This server implements the Model Context Protocol (MCP), an open standard for seamless integration between LLM applications and external data sources. MCP provides:

- **Standardized Communication**: JSON-RPC 2.0 based protocol for reliable tool invocation
- **Tool Discovery**: Automatic detection and documentation of available tools
- **Type Safety**: Strong typing with Pydantic models ensures data integrity
- **Universal Compatibility**: Works with any MCP-compliant client

### Repository Pattern Architecture

The server uses a clean repository pattern to separate concerns:

- **RepositoryContainer**: Dependency injection container for all repositories
- **ContextRepository**: Manages context entries with deduplication
- **TagRepository**: Handles tag normalization and relationships
- **ImageRepository**: Manages multimodal attachments
- **StatisticsRepository**: Provides metrics and thread statistics
- **DatabaseConnectionManager**: Thread-safe connection pooling with retry logic

All SQL operations are encapsulated in repository classes, keeping the server layer clean.

### Thread-Based Context Management

The server uses thread IDs to scope context, enabling multiple agents to collaborate:

```
Thread: "analyze-q4-sales"
├── User Context: "Analyze our Q4 sales data"
├── Agent 1 Context: "Fetched sales data from database"
├── Agent 2 Context: "Generated charts showing 15% growth"
└── Agent 3 Context: "Identified top performing products"
```

### Database Architecture

```
context_entries (main table)
├── thread_id (indexed)
├── source (user/agent, indexed)
├── content_type (text/multimodal)
├── text_content
├── metadata (JSON)
└── timestamps

tags (normalized, many-to-many)
├── context_entry_id (foreign key)
└── tag (indexed, lowercase)

image_attachments (binary storage)
├── context_entry_id (foreign key)
├── image_data (BLOB)
├── mime_type
└── position
```

### Key Design Principles

1. **Thread-Based Context Scoping**
   - Single thread_id shared by all agents working on the same task
   - No hierarchical threads - simplicity is key
   - Thread_id is the primary grouping mechanism

2. **Source Attribution**
   - Only two sources: "user" and "agent"
   - Agents can filter to see only user context or all context
   - No agent-specific IDs in source field (use metadata if needed)

3. **Standardized Operations**
   - All operations are stateless and idempotent
   - Clear error messages for agent understanding
   - Consistent JSON response formats
   - Async-first design for non-blocking execution

### Performance Optimization

#### Database Configuration
- **WAL Mode**: Better concurrency for multi-agent access
- **Memory-mapped I/O**: 256MB for faster reads
- **Strategic Indexing**:
  - Primary: `thread_id`, `thread_source`, `created_at`
  - Secondary: `tags`, `image_context`
- **Query Optimization**: Indexed fields filtered first

#### Resource Limits
- **Individual Image**: 10MB maximum
- **Total Request**: 100MB maximum
- **Query Results**: 500 entries maximum
- **Thread Isolation**: Prevents accidental cross-context access

## Need Help?

- Check existing [GitHub Issues](https://github.com/alex-feel/mcp-context-server/issues)
- Read the [README.md](README.md) for usage documentation
- Create a new issue with detailed information if needed
