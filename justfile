# =============================================================================
# Artenic AI — Development Commands
# =============================================================================
# Install just: https://github.com/casey/just
# Usage: just <recipe>
# =============================================================================

# Default recipe: show available commands
default:
    @just --list

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Install all Python dependencies (workspace-wide)
install:
    uv sync --dev

# Install pre-commit hooks
hooks:
    uv run pre-commit install

# Full setup: install deps + hooks
setup: install hooks

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

# Run ruff linter
lint:
    uv run ruff check .

# Run ruff linter with auto-fix
lint-fix:
    uv run ruff check --fix .

# Run ruff formatter
fmt:
    uv run ruff format .

# Check formatting without modifying
fmt-check:
    uv run ruff format --check .

# Run mypy type checker
typecheck:
    uv run mypy packages/

# Run all quality checks (lint + format check + type check)
check: lint fmt-check typecheck

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

# Run all tests
test *ARGS:
    uv run pytest {{ ARGS }}

# Run tests with coverage report
test-cov:
    uv run pytest --cov --cov-report=term-missing --cov-report=html --cov-fail-under=100

# Run all SDK tests (core + ensemble + training + client)
test-sdk:
    uv run pytest packages/sdk/ -v

# Run only SDK core tests
test-sdk-core:
    uv run pytest packages/sdk/core/tests/ -v

# Run only SDK ensemble tests
test-sdk-ensemble:
    uv run pytest packages/sdk/ensemble/tests/ -v

# Run only SDK training tests
test-sdk-training:
    uv run pytest packages/sdk/training/tests/ -v

# Run only SDK client tests
test-sdk-client:
    uv run pytest packages/sdk/client/tests/ -v

# Run all Platform tests (core + providers + training)
test-platform:
    uv run pytest packages/platform/ -v

# Run only Platform core tests
test-platform-core:
    uv run pytest packages/platform/core/tests/ -v

# Run only Platform providers tests
test-platform-providers:
    uv run pytest packages/platform/providers/tests/ -v

# Run only Platform training tests
test-platform-training:
    uv run pytest packages/platform/training/tests/ -v

# Run only CLI tests
test-cli:
    uv run pytest packages/cli/tests/ -v

# Run only Optimizer tests
test-optimizer:
    uv run pytest packages/optimizer/tests/ -v

# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------

# Start the platform server (in-memory SQLite)
run-platform:
    uv run python -m artenic_ai_platform

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

# Start dev dependencies (PostgreSQL + MLflow)
dev-up:
    docker compose -f docker-compose.dev.yml up -d

# Stop dev dependencies
dev-down:
    docker compose -f docker-compose.dev.yml down

# Stop dev dependencies and remove volumes
dev-reset:
    docker compose -f docker-compose.dev.yml down -v

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

# Install dashboard dependencies
dash-install:
    cd dashboard && npm install

# Start dashboard dev server
dash-dev:
    cd dashboard && npm run dev

# Build dashboard for production
dash-build:
    cd dashboard && npm run build

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

# Update all Python dependencies
update:
    uv lock --upgrade

# Run pre-commit on all files
pre-commit:
    uv run pre-commit run --all-files

# Clean build artifacts
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    rm -rf htmlcov/ .coverage coverage.xml
