# =============================================================================
# Artenic AI Platform — Production Dockerfile
# =============================================================================
# Build:  docker build -t artenic-ai-platform .
# Run:    docker run -p 9000:9000 --env-file .env artenic-ai-platform
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: build — install uv and sync dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS build

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (cache-friendly layer)
COPY pyproject.toml uv.lock ./
COPY packages/sdk/pyproject.toml packages/sdk/pyproject.toml
COPY packages/platform/pyproject.toml packages/platform/pyproject.toml
COPY packages/cli/pyproject.toml packages/cli/pyproject.toml
COPY packages/optimizer/pyproject.toml packages/optimizer/pyproject.toml

# Create stub packages so uv/hatchling can resolve workspace members
# README files are required by hatchling (readme field in pyproject.toml)
COPY README.md ./
COPY packages/platform/README.md packages/platform/README.md
RUN mkdir -p packages/sdk/src/artenic_ai_sdk && \
    touch packages/sdk/src/artenic_ai_sdk/__init__.py && \
    mkdir -p packages/platform/src/artenic_ai_platform && \
    touch packages/platform/src/artenic_ai_platform/__init__.py && \
    mkdir -p packages/cli/src/artenic_ai_cli && \
    touch packages/cli/src/artenic_ai_cli/__init__.py && \
    mkdir -p packages/optimizer/src/artenic_optimizer && \
    touch packages/optimizer/src/artenic_optimizer/__init__.py

# Sync platform dependencies (without dev deps)
RUN uv sync --package artenic-ai-platform --no-dev --frozen

# Copy actual source code
COPY packages/sdk/src packages/sdk/src
COPY packages/platform/src packages/platform/src

# Re-sync to install the actual packages
RUN uv sync --package artenic-ai-platform --no-dev --frozen

# ---------------------------------------------------------------------------
# Stage 2: runtime — slim production image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Create non-root user
RUN groupadd --gid 1000 artenic && \
    useradd --uid 1000 --gid artenic --shell /bin/bash --create-home artenic

WORKDIR /app

# Copy the virtual environment from build stage
COPY --from=build /app/.venv /app/.venv

# Copy source code
COPY --from=build /app/packages/sdk/src /app/packages/sdk/src
COPY --from=build /app/packages/platform/src /app/packages/platform/src

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default port (matches PlatformSettings.port)
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health')" || exit 1

# Switch to non-root
USER artenic

# Run the platform
CMD ["python", "-m", "artenic_ai_platform"]
