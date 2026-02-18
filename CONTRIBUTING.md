# Contributing to Artenic AI

Thanks for your interest in contributing! This guide will help you get started.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [just](https://github.com/casey/just) (optional but recommended)
- Docker and Docker Compose (for PostgreSQL + MLflow)
- Git

## Setup

```bash
# Clone the repo
git clone https://github.com/Artenic-Ai/artenic_ai.git
cd artenic_ai

# Install all dependencies
just setup
# or manually:
#   uv sync --dev
#   uv run pre-commit install
```

This installs all Python packages in development mode via uv workspaces, plus
dev tools (pytest, ruff, mypy, pre-commit).

## Development Workflow

### Running Tests

```bash
# All tests
just test

# With coverage
just test-cov

# Specific package
just test-sdk
just test-platform
just test-cli

# Specific test file
just test packages/sdk/tests/test_something.py -v
```

### Code Quality

Pre-commit hooks run automatically on `git commit`. You can also run them manually:

```bash
# All checks
just check

# Individual checks
just lint          # ruff check
just fmt           # ruff format
just typecheck     # mypy strict
```

### Dev Infrastructure

```bash
just dev-up        # Start PostgreSQL + MLflow
just dev-down      # Stop
just dev-reset     # Stop + delete volumes
```

### Running the Platform Locally

```bash
# In-memory SQLite (no Docker required)
uv run python -m artenic_ai_platform

# With PostgreSQL + MLflow (start with: just dev-up)
ARTENIC_DATABASE_URL=postgresql+asyncpg://artenic:artenic@localhost:5432/artenic \
ARTENIC_MLFLOW_TRACKING_URI=http://localhost:5000 \
  uv run python -m artenic_ai_platform
```

## Dashboard Development

The dashboard is a standalone React app in `dashboard/`.

### Prerequisites

- Node.js 22+
- npm

### Quick Start

```bash
cd dashboard
npm install
npm run dev          # Dev server on http://localhost:5173
npm test             # Run tests (Vitest)
npm run build        # Production build
```

### Demo Mode

The dashboard runs in **demo mode** by default (`VITE_DEMO_MODE=true`), serving realistic
mock data without a backend. This is what powers the live demo at **[ai.artenic.ch](https://ai.artenic.ch)**.

To connect to a running platform server, set `VITE_DEMO_MODE=false` and `VITE_API_URL`
to the platform URL.

### Dashboard Patterns

- **TypeScript strict** with `noUncheckedIndexedAccess`
- **Tailwind CSS 4** with `@theme` syntax for semantic design tokens
- **React Router 7** with lazy-loaded pages (`React.lazy()` + Suspense)
- **TanStack React Query 5** with query key factory pattern (`lib/query-keys.ts`)
- **Vitest** + `@testing-library/react` for component and integration tests
- Shared UI components in `components/ui/` (Button, Badge, Card, DataTable, Dialog, etc.)
- Error states with retry on all data-fetching pages
- Demo/real API toggle is transparent — hooks never know which mode is active

## Code Style

- **Python 3.12+** features encouraged (`from __future__ import annotations` in all files)
- **Pydantic v2** for all data models (`ConfigDict`, not `class Config`)
- **async/await** for I/O operations
- **Type hints** on all function signatures (mypy strict enforced)
- **ruff** for linting and formatting (replaces black, isort, flake8)
- Line length: 100 characters
- Follow existing patterns in the codebase

### Platform-specific Patterns

- `TYPE_CHECKING` imports for circular dependency avoidance (services reference each other)
- `# pragma: no cover` for optional SDK import blocks (`except ImportError`)
- aiosqlite in-memory databases for all tests (no Docker in CI)
- `unittest.mock` for cloud provider SDKs (never call real cloud APIs in tests)
- Lazy imports for optional dependencies (cloud SDKs, MLflow, OTel)

## Commit Conventions

```
<type>(<scope>): short description

- Point 1
- Point 2
- ...
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`

Scopes: `sdk`, `platform`, `cli`, `optimizer`, `dashboard`, `infra`

Examples:

```
feat(sdk): add BaseModel contract
feat(platform): add AWS provider with spot preemption handling
fix(platform): handle null provider response
test(platform): add budget enforcement edge cases
ci: add platform type-check and test jobs
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feature/your-feature`
3. **Make your changes** following the code style above
4. **Write tests** — we enforce 100% coverage for SDK, platform, and CLI
5. **Run `just check`** to verify lint, format, and types pass
6. **Run `just test-cov`** to verify tests and coverage
7. **Commit** with a clear message (see conventions above)
8. **Push** and open a Pull Request

## Architecture Principles

- **Local + cloud** — supports local subprocess training and 16+ cloud providers
- **Multi-cloud** — never lock into a single provider
- **SDK contracts** — `BaseModel` ABC is the foundation, don't break it
- **Type safety** — mypy strict mode, no `Any` escape hatches
- **Test coverage** — target 100% for all packages
- **Async first** — all I/O operations use async/await
- **Graceful degradation** — optional services (MLflow, OTel) fail gracefully when unavailable

## Adding a New Cloud Provider

1. Create `packages/platform/src/artenic_ai_platform/providers/your_provider.py`
2. Extend `CloudProvider` from `cloud_base.py`
3. Implement the abstract hooks: `_connect`, `_disconnect`, `_upload_code`, `_provision_and_start`, `_poll_provider`, `_collect_artifacts`, `_cleanup_compute`, `_cancel_provider_job`
4. Add provider settings to `settings.py` (nested `BaseModel` with `enabled: bool = False`)
5. Add optional dependency group in `packages/platform/pyproject.toml`
6. Write tests (mock the provider SDK, test all lifecycle methods)
7. Register the provider in `app.py` lifespan

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
