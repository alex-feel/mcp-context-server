# Production Dockerfile for MCP Context Server
# Supports HTTP transport for remote client connections

# ============================================
# BUILDER STAGE
# ============================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# uv build optimization settings
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# Install dependencies first (better layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --extra semantic-search --no-dev

# Copy application code
COPY app/ ./app/
COPY pyproject.toml uv.lock README.md ./

# Install project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra semantic-search --no-dev

# ============================================
# RUNTIME STAGE
# ============================================
FROM python:3.12-slim-bookworm AS runtime

# Security: Create non-root user (UID 10001 is conventional for containerized apps)
RUN groupadd --system --gid 10001 appuser \
    && useradd --system --gid 10001 --uid 10001 --create-home appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# Runtime environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create data directory for SQLite
RUN mkdir -p /data && chown appuser:appuser /data

# Switch to non-root user
USER appuser

# Expose HTTP port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

# Run the server
CMD ["python", "-m", "app.server"]
