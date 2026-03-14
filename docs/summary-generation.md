# Summary Generation Guide

## Introduction

Summary generation automatically creates concise, dense summaries for each stored context entry using a local or cloud LLM. Summaries are stored alongside the full text and returned in all search tool results (`search_context`, `semantic_search_context`, `fts_search_context`, `hybrid_search_context`), giving LLM agents more actionable information per token when browsing large context collections.

**Key benefit:** All search tools return truncated `text_content` (configurable via `SEARCH_TRUNCATION_LENGTH`, default 150 characters). With summary generation enabled, the `summary` field is populated with a concise LLM-generated summary of the full entry (token limit controlled by `SUMMARY_MAX_TOKENS`), capturing key topics, decisions, and action items that help an agent determine relevance without fetching the full entry.

This feature is **enabled by default** when the `summary-ollama` extra is installed (included in the recommended setup).

## Summary Providers

The server supports three summary providers via LangChain integration:

| Provider             | Default Model    | Cost            | Best For                     |
|----------------------|------------------|-----------------|------------------------------|
| **Ollama** (default) | qwen3:0.6b       | Free (local)    | Development, privacy-focused |
| **OpenAI**           | gpt-5-nano       | Pay-per-use API | Production, high quality     |
| **Anthropic**        | claude-haiku-4-5 | Pay-per-use API | Production, high quality     |

Select a provider via the `SUMMARY_PROVIDER` environment variable.

## Prerequisites

- **Python**: 3.12+ (already required by MCP Context Server)
- **Ollama** (for default provider): Installed from [ollama.com/download](https://ollama.com/download)
- **RAM**: 2GB minimum for `qwen3:0.6b`; 8GB recommended for `qwen3:4b`

## Installation

### Step 1: Install Provider Dependencies

Each provider has its own optional dependency group:

```bash
# Ollama provider (default)
uv sync --extra summary-ollama

# OpenAI provider
uv sync --extra summary-openai

# Anthropic provider
uv sync --extra summary-anthropic
```

### Step 2: Provider-Specific Setup

See the provider-specific sections below for model and credential requirements.

## Provider Configuration

### Ollama (Default)

Ollama runs summary models locally with no API costs. The default model `qwen3:0.6b` is optimized for fast summaries with minimal resource requirements.

#### Setup

1. **Install Ollama** from [ollama.com/download](https://ollama.com/download)

2. **Pull the summary model**:
   ```bash
   ollama pull qwen3:0.6b
   ```

3. **Verify**:
   ```bash
   ollama list
   ```

#### Configuration

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-ollama,summary-ollama,reranking]", "mcp-context-server"],
      "env": {
        "ENABLE_SUMMARY_GENERATION": "true",
        "SUMMARY_PROVIDER": "ollama",
        "SUMMARY_MODEL": "qwen3:0.6b"
      }
    }
  }
}
```

#### Environment Variables

| Variable                     | Default      | Description                                                                             |
|------------------------------|--------------|-----------------------------------------------------------------------------------------|
| `ENABLE_SUMMARY_GENERATION`  | `true`       | Enable/disable summary generation                                                       |
| `SUMMARY_PROVIDER`           | `ollama`     | Set to `ollama`                                                                         |
| `SUMMARY_MODEL`              | `qwen3:0.6b` | Ollama model name (see model table below)                                               |
| `SUMMARY_MAX_TOKENS`         | `2000`       | Maximum output tokens for summary generation (50-5000)                                  |
| `SUMMARY_TIMEOUT_S`          | `120.0`      | Timeout in seconds for summary generation API calls                                     |
| `SUMMARY_RETRY_MAX_ATTEMPTS` | `3`          | Maximum retry attempts on transient errors                                              |
| `SUMMARY_RETRY_BASE_DELAY_S` | `1.0`        | Base delay in seconds between retries (exponential backoff)                             |
| `SUMMARY_MAX_CONCURRENT`     | `3`          | Maximum concurrent summary generation operations (1-20)                                 |
| `SUMMARY_MIN_CONTENT_LENGTH` | `300`        | Minimum text length (characters) to trigger summary generation. 0 = always generate     |
| `SUMMARY_PROMPT`             | (built-in)   | Custom system prompt. Overrides the default prompt. See [Custom Prompt](#custom-prompt) |

#### Qwen3 Model Options (Ollama)

The Qwen3 family offers a range of sizes for different resource constraints and quality requirements:

| Model        | RAM Required | Quality   | Speed     | Notes                                          |
|--------------|--------------|-----------|-----------|------------------------------------------------|
| `qwen3:0.6b` | ~2GB         | Basic     | Fastest   | **Default**. Lightweight, minimal resources    |
| `qwen3:1.7b` | ~4GB         | Good      | Fast      | Higher quality, good balance for most uses     |
| `qwen3:4b`   | ~8GB         | Better    | Moderate  | Recommended when higher quality is needed      |
| `qwen3:8b`   | ~16GB        | Best      | Slower    | Highest quality, requires dedicated hardware   |

**Recommendation:** Start with `qwen3:0.6b` (default). Upgrade to `qwen3:1.7b` or `qwen3:4b` if summary quality is insufficient for your use case.

Pull any alternative model before use:
```bash
ollama pull qwen3:4b
```

### OpenAI

OpenAI provides high-quality summaries via API with no local hardware requirements.

#### Setup

1. **Get API key** from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

2. **Install dependencies**:
   ```bash
   uv sync --extra summary-openai
   ```

#### Configuration

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-ollama,summary-openai,reranking]", "mcp-context-server"],
      "env": {
        "ENABLE_SUMMARY_GENERATION": "true",
        "SUMMARY_PROVIDER": "openai",
        "SUMMARY_MODEL": "gpt-5-nano",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

#### Environment Variables

| Variable                    | Default       | Description                          |
|-----------------------------|---------------|--------------------------------------|
| `SUMMARY_PROVIDER`          | -             | Set to `openai`                      |
| `SUMMARY_MODEL`             | `gpt-5-nano`  | OpenAI model name                    |
| `OPENAI_API_KEY`            | -             | **Required**: OpenAI API key         |

### Anthropic

Anthropic's Claude models offer high-quality summaries with strong instruction-following.

#### Setup

1. **Get API key** from [console.anthropic.com](https://console.anthropic.com)

2. **Install dependencies**:
   ```bash
   uv sync --extra summary-anthropic
   ```

#### Configuration

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-ollama,summary-anthropic,reranking]", "mcp-context-server"],
      "env": {
        "ENABLE_SUMMARY_GENERATION": "true",
        "SUMMARY_PROVIDER": "anthropic",
        "SUMMARY_MODEL": "claude-haiku-4-5-20251001",
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

#### Environment Variables

| Variable            | Default                     | Description                     |
|---------------------|-----------------------------|---------------------------------|
| `SUMMARY_PROVIDER`  | -                           | Set to `anthropic`              |
| `SUMMARY_MODEL`     | `claude-haiku-4-5-20251001` | Anthropic model name            |
| `ANTHROPIC_API_KEY` | -                           | **Required**: Anthropic API key |

## Custom Prompt

The server ships with a carefully engineered default summarization prompt. For most use cases, the default prompt works well. You can override it with a custom prompt via the `SUMMARY_PROMPT` environment variable.

### Default Prompt

The built-in prompt (from `app/summary/instructions.py`) is:

```text
/no_think
You are an expert summarizer for a context storage system used by AI agents.
Your task is to produce a single, dense paragraph that captures the essential
meaning of the input text.

Requirements:
- Include key topics, named entities, decisions, conclusions, and action items
- Prioritize information that helps determine relevance without reading the full text
- Use specific terms from the original text (do not generalize or abstract away details)
- Write exactly one paragraph with no line breaks
- Do not add any labels, prefixes, headers, or explanations
- Do not start with "This text" or "The author" or similar meta-references
- Output ONLY the summary text, nothing else
```

**Design notes:**
- `/no_think` disables Qwen3 reasoning mode (saves tokens and time for small models)
- Zero-shot format maximizes input token budget on resource-constrained models
- Single-paragraph constraint is easiest for small models to follow consistently
- Negative constraints (`Do not...`) prevent common small-model failure modes

### Overriding the Prompt

Set `SUMMARY_PROMPT` to your custom system message:

```json
{
  "env": {
    "SUMMARY_PROMPT": "You are a technical documentation summarizer. Produce a single sentence capturing the main purpose, key entities, and outcome. Output only the sentence."
  }
}
```

**Important notes:**
- The prompt is used as the **system message**. The text to summarize is passed separately as the user message.
- An empty string or whitespace-only value falls back to the default prompt (unlike `MCP_SERVER_INSTRUCTIONS` where empty string disables the feature).
- For Qwen3 models, include `/no_think` at the start of your prompt to disable the model's reasoning mode and save tokens.

## How It Works

1. **On `store_context`**: Summary generation runs in parallel with embedding generation before the database transaction. Both must complete before data is saved. If summary generation fails, the entire operation fails (transactional integrity).
2. **On `store_context_batch`**: Summary generation runs for each entry within its transaction. All generation completes before the entry is committed.
3. **On `update_context`**: When `text` changes, summary and embedding are regenerated in parallel before the update transaction.
4. **On `update_context_batch`**: Summary regenerated for entries with text changes within each entry's transaction.
5. **Deduplication**: If an entry already has a summary (duplicate detection), summary generation is skipped.

### Minimum Content Length

By default, summary generation is skipped for short text (fewer than 300 characters). Short text already fits within search result truncation limits, so a separate summary adds no value and wastes LLM resources.

| Variable                       | Default | Range      | Description                                                      |
|--------------------------------|---------|------------|------------------------------------------------------------------|
| `SUMMARY_MIN_CONTENT_LENGTH`   | `300`   | 0 - 10000  | Minimum text length (characters) to trigger summary generation   |

**Behavior by operation:**

- **`store_context`** / **`store_context_batch`**: Text shorter than the threshold is stored without a summary (`summary` field is empty).
- **`update_context`** / **`update_context_batch`**: When updated text is shorter than the threshold, any existing summary is cleared (set to NULL) since the summary no longer accurately represents the content.
- **Deduplication**: When a short-text duplicate is stored without generating a summary, any pre-existing summary on the original entry is preserved via SQL `COALESCE`.

**Special values:**

- `0` disables the threshold entirely — summaries are always generated regardless of text length.
- The comparison uses strict `<` (text at exactly the threshold length IS summarized).

### Summary in Search Results

The `summary` field appears in all search tool results when available:

```json
{
  "results": [
    {
      "id": 123,
      "thread_id": "project-abc",
      "source": "agent",
      "text_content": "Agent implemented OAuth2 authentication with JWT tokens for the user management API, resolving rate-limiting issues on the /auth/...",
      "summary": "Agent implemented OAuth2 authentication with JWT tokens for the user management API, resolving rate-limiting issues on the /auth/login endpoint.",
      "is_text_content_truncated": true,
      "metadata": {"agent_name": "developer", "status": "done"},
      "tags": ["implementation", "auth"]
    }
  ]
}
```

All search tools always return truncated `text_content` (configurable via `SEARCH_TRUNCATION_LENGTH`, default 150 characters) with `is_text_content_truncated` flag. The `summary` field provides a dense LLM-generated summary (controlled by `SUMMARY_MAX_TOKENS`, default 2000 tokens) when summary generation is enabled, or an empty string when disabled or not yet generated. Use `get_context_by_ids` to retrieve the full, untruncated text content.

## Disabling Summary Generation

To disable summary generation entirely:

```json
{
  "env": {
    "ENABLE_SUMMARY_GENERATION": "false"
  }
}
```

When disabled:
- No LLM calls are made at store/update time
- The `summary` field in search results is always an empty string
- No summary provider dependencies are required at startup

**Note:** Like `ENABLE_EMBEDDING_GENERATION`, when `ENABLE_SUMMARY_GENERATION=true` (default) and the required provider package is not installed, the server will NOT start. Set to `false` to run without summary generation.

## Installation with Multiple Features

When combining summary generation with semantic search:

```bash
# Ollama for both embeddings and summaries (recommended)
uv sync --extra embeddings-ollama --extra summary-ollama --extra reranking

# Ollama embeddings + OpenAI summaries
uv sync --extra embeddings-ollama --extra summary-openai --extra reranking
```

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-ollama,summary-ollama,reranking]", "mcp-context-server"],
      "env": {
        "ENABLE_SEMANTIC_SEARCH": "true",
        "ENABLE_SUMMARY_GENERATION": "true",
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "qwen3-embedding:0.6b",
        "SUMMARY_PROVIDER": "ollama",
        "SUMMARY_MODEL": "qwen3:0.6b",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

## Troubleshooting

### Summary Not Generated

**Symptom:** `summary` field is an empty string in search results.

**Causes and solutions:**

| Cause                             | Solution                                                  |
|-----------------------------------|-----------------------------------------------------------|
| `ENABLE_SUMMARY_GENERATION=false` | Set to `true` and install provider dependencies           |
| Provider package not installed    | Run `uv sync --extra summary-ollama` or `summary-openai`  |
| Ollama not running                | Start Ollama: `ollama serve`                              |
| Model not pulled                  | Run `ollama pull qwen3:0.6b`                              |
| Generation timed out              | Raise `SUMMARY_TIMEOUT_S` (default 120s) for slow models  |
| API key missing                   | Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`               |

### Server Won't Start

**Error**: `ENABLE_SUMMARY_GENERATION=true but langchain-ollama package not installed`

**Solution**: Install the required extra:
```bash
uv sync --extra summary-ollama
```
Or disable summary generation: `ENABLE_SUMMARY_GENERATION=false`

### Poor Summary Quality

**Solutions:**
- Upgrade to a larger model (`qwen3:4b` or `qwen3:8b`)
- Increase `SUMMARY_MAX_TOKENS` to allow longer, more detailed summaries
- Provide a custom `SUMMARY_PROMPT` tailored to your domain

### Timeout Errors

**Error**: `Summary generation timed out after 120s`

**Solutions:**
- Increase `SUMMARY_TIMEOUT_S` (e.g., `180`)
- Use a smaller/faster model (`qwen3:0.6b`) or upgrade to `qwen3:1.7b` for better quality
- Reduce `SUMMARY_MAX_CONCURRENT` to limit parallel generation load on the model server

## Additional Resources

- **API Reference**: [API Reference](api-reference.md) - complete tool documentation including `summary` field
- **Semantic Search**: [Semantic Search Guide](semantic-search.md) - vector similarity search setup
- **Docker Deployment**: [Docker Deployment Guide](deployment/docker.md) - SUMMARY_EXTRA build argument
- **Main Documentation**: [README.md](../README.md) - overview and quick start
