# Connecting to Your AI Assistant

This guide explains how to connect the MCP Context Server to Claude Code using the one-command Docker bootstrap. One command downloads the Docker Compose stack, starts all services, and fully configures Claude Code with context management hooks, skills, and rules.

## Overview

The MCP Context Server provides persistent multimodal context storage for LLM agents over the Model Context Protocol (MCP). The one-command bootstrap handles everything:

1. **Downloads and starts the Docker stack** -- PostgreSQL (with pgvector), Ollama (embedding + summary models), and the MCP Context Server (HTTP transport on port `8000`).
2. **Fully configures Claude Code** -- registers the MCP server, installs context management hooks, skills, rules, and sets appropriate permissions.

No manual steps required. No local clone of this repository needed. No Git required.

## Prerequisites

- Docker Desktop installed and running
- Windows PowerShell 5.1+ (Windows) or Bash (macOS / Linux)
- Host port `8000` available
- ~2 GB free disk space (Docker images + Ollama models)

## One-Command Setup

Run the command for your operating system. Each command does the same thing: installs `uv`, downloads the Docker Compose stack, starts the services, and fully configures Claude Code.

### Windows

```powershell
powershell -NoProfile -NoExit -ExecutionPolicy Bypass -Command "`$env:CLAUDE_CODE_TOOLBOX_ENV_CONFIG='https://raw.githubusercontent.com/alex-feel/mcp-context-server/refs/heads/main/agents/claude-code/environment-docker.yaml'; `$env:CLAUDE_CODE_TOOLBOX_SKIP_INSTALL='1'; iex (irm 'https://raw.githubusercontent.com/alex-feel/claude-code-toolbox/main/scripts/windows/setup-environment.ps1')"
```

### macOS

```bash
CLAUDE_CODE_TOOLBOX_ENV_CONFIG='https://raw.githubusercontent.com/alex-feel/mcp-context-server/refs/heads/main/agents/claude-code/environment-docker.yaml' \
CLAUDE_CODE_TOOLBOX_SKIP_INSTALL=1 \
bash -c "$(curl -fsSL https://raw.githubusercontent.com/alex-feel/claude-code-toolbox/main/scripts/macos/setup-environment.sh)"
```

### Linux

```bash
CLAUDE_CODE_TOOLBOX_ENV_CONFIG='https://raw.githubusercontent.com/alex-feel/mcp-context-server/refs/heads/main/agents/claude-code/environment-docker.yaml' \
CLAUDE_CODE_TOOLBOX_SKIP_INSTALL=1 \
bash -c "$(curl -fsSL https://raw.githubusercontent.com/alex-feel/claude-code-toolbox/main/scripts/linux/setup-environment.sh)"
```

### What This Does

Each command performs the same steps, in order:

1. Installs `uv` (always installed; required for PEP 723 hooks bundled with the environment).
2. Downloads `docker-compose.postgresql.ollama.yml` to `~/.mcp/`.
3. Runs `docker compose -f ~/.mcp/docker-compose.postgresql.ollama.yml up -d`. This starts:
   - `mcp-context-server` (HTTP transport on port `8000`)
   - `postgres` (with the pgvector extension for semantic search)
   - `ollama` (stock `ollama/ollama:latest` image; pulls `qwen3-embedding:0.6b` and `qwen3:0.6b` on first run)
4. Configures Claude Code to connect to the server via HTTP at `http://localhost:8000/mcp`.
5. Installs the bundled context-server skills, hooks, and rules into `~/.claude/`.

`CLAUDE_CODE_TOOLBOX_SKIP_INSTALL=1` tells the bootstrap to skip Claude Code itself and its IDE extensions if you already have them installed -- `uv` and the environment configuration are still applied.

**First-run note.** Ollama pulls roughly 1.2 GB of model data (`qwen3-embedding:0.6b` + `qwen3:0.6b`) on the first start. Allow 2-5 minutes before the server is ready to serve embedding-backed tools. To follow progress:

```bash
docker compose -f ~/.mcp/docker-compose.postgresql.ollama.yml logs -f ollama
```

To verify the server responds:

```bash
curl http://localhost:8000/health
```

## Troubleshooting

| Symptom                                                                                   | Cause                                                                     | Fix                                                                                                                                                                                           |
|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `docker: command not found` (Windows) or Docker Desktop is not running                    | Docker Desktop is not installed or not started                            | Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and make sure it is running before re-running the bootstrap                                                         |
| `Error response from daemon: ... bind for 0.0.0.0:8000 failed: port is already allocated` | Another process is using port `8000` on the host                          | Stop the conflicting process, OR edit `~/.mcp/docker-compose.postgresql.ollama.yml` and remap the host port (e.g., `"8001:8000"`), then update your client URL to `http://localhost:8001/mcp` |
| Bootstrap reports "Ollama not healthy yet" / `/health` returns connection refused         | First-run model download still in progress                                | Watch progress with `docker compose -f ~/.mcp/docker-compose.postgresql.ollama.yml logs -f ollama`. Wait until both `qwen3-embedding:0.6b` and `qwen3:0.6b` are pulled                        |
| GPU acceleration is not used                                                              | Docker Compose file's GPU section is commented out by default             | See the [GPU Acceleration Guide](deployment/gpu-acceleration.md) for vendor-specific instructions (NVIDIA / AMD ROCm / Intel Vulkan)                                                          |
| Two compose variants conflict over the `ollama-models` volume                             | The `ollama-models` named volume is shared by all Ollama compose variants | Only one Ollama compose variant can run at a time. Stop the other variant before starting a new one: `docker compose -f <other-file> down`                                                    |

## Update and Uninstall

**Update** -- exit Claude Code and re-run the one-command bootstrap for your OS (see [One-Command Setup](#one-command-setup) above). This re-downloads the Compose file and restarts the stack with the latest images.

**Stop the stack (preserve data):**

```bash
docker compose -f ~/.mcp/docker-compose.postgresql.ollama.yml down
```

**Remove the stack and all data (Docker volumes):**

```bash
docker compose -f ~/.mcp/docker-compose.postgresql.ollama.yml down -v
```

This deletes the PostgreSQL data, the SQLite/Postgres backend volume, and the `ollama-models` volume (forcing re-download of models on the next start).

**Remove the downloaded Compose file:**

```powershell
# Windows PowerShell
Remove-Item -Recurse -Force "$env:USERPROFILE\.mcp"
```

```bash
# macOS / Linux
rm -rf ~/.mcp
```

**Unregister the server from Claude Code:**

```bash
claude mcp remove context-server
```

## Related Documentation

- [Docker Deployment Guide](deployment/docker.md) -- full reference for all 15 Docker Compose configurations, environment variables, advanced configuration, and PostgreSQL backend error scenarios.
- [GPU Acceleration Guide](deployment/gpu-acceleration.md) -- NVIDIA, AMD ROCm, and Intel/Vulkan GPU configuration.
- [Database Backends Guide](database-backends.md) -- choosing between SQLite and PostgreSQL.
- [Authentication Guide](authentication.md) -- bearer-token authentication for HTTP transport.
- [Environment Variables Reference](environment-variables.md) -- full reference for every environment variable the server understands.
- [README](../README.md) -- feature overview and entry point.
