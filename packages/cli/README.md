# Artenic AI CLI

Command-line interface for the Artenic AI platform — manage models, training, inference, and configuration from your terminal.

The CLI is a **client package** of the monorepo — it depends on `artenic-ai-sdk` and communicates with the platform via its REST API.

## Installation

```bash
pip install artenic-ai-cli
```

The `artenic` command becomes available after installation:

```bash
artenic --version
artenic --help
```

## Quick Start

```bash
# Check platform health
artenic health check

# List registered models
artenic model list

# Register a new model
artenic model register --name my-model --version 1.0 --type lightgbm

# Dispatch a training job
artenic training dispatch --service my-svc --model my-model --provider local

# Run a prediction
artenic predict my-service --data '{"feature_a": 1.5, "feature_b": "cat"}'

# Use JSON output (machine-readable)
artenic --json model list
```

## Global Options

| Option | Env Variable | Description |
|--------|-------------|-------------|
| `--url` | `ARTENIC_URL` | Platform base URL (default: `http://localhost:9000`) |
| `--api-key` | `ARTENIC_API_KEY` | API key for authentication |
| `--profile` | `ARTENIC_PROFILE` | Config profile name (default: `default`) |
| `--timeout` | `ARTENIC_TIMEOUT` | Request timeout in seconds (default: `30`) |
| `--json` | — | Output raw JSON instead of tables |
| `--version` | — | Show CLI version and exit |

## Commands

### Health

| Command | Description |
|---------|-------------|
| `artenic health check` | Basic liveness probe |
| `artenic health ready` | Readiness probe (checks DB connectivity) |
| `artenic health detailed` | Full component health report |

### Model Registry

| Command | Description |
|---------|-------------|
| `artenic model list` | List all registered models |
| `artenic model get MODEL_ID` | Get model details by ID |
| `artenic model register` | Register a new model (`--name`, `--version`, `--type`, `--tag key=value`) |
| `artenic model promote MODEL_ID` | Promote model to production (`--version`) |
| `artenic model retire MODEL_ID` | Retire (archive) a model |

### Inference

| Command | Description |
|---------|-------------|
| `artenic predict SERVICE` | Single prediction (`--data JSON`, `--model-id`) |
| `artenic predict-batch SERVICE` | Batch predictions (`--data JSON_ARRAY`, `--model-id`) |

### Training

| Command | Description |
|---------|-------------|
| `artenic training dispatch` | Dispatch job (`--service`, `--model`, `--provider`, `--spot`, `--config-json`) |
| `artenic training list` | List jobs (`--service`, `--provider`, `--status`, `--limit`) |
| `artenic training status JOB_ID` | Get job status |
| `artenic training cancel JOB_ID` | Cancel a running or pending job |

### Budget

| Command | Description |
|---------|-------------|
| `artenic budget list` | List budget rules (`--scope`, `--all`) |
| `artenic budget create` | Create rule (`--scope`, `--scope-value`, `--period`, `--limit-eur`) |
| `artenic budget update BUDGET_ID` | Update rule (`--limit-eur`, `--enabled/--disabled`) |
| `artenic budget spending` | Current spending (`--scope`, `--scope-value`) |

### Ensembles

| Command | Description |
|---------|-------------|
| `artenic ensemble create` | Create ensemble (`--name`, `--service`, `--strategy`, `--model-ids`) |
| `artenic ensemble list` | List ensembles (`--service`, `--stage`) |
| `artenic ensemble get ENSEMBLE_ID` | Get ensemble details |
| `artenic ensemble update ENSEMBLE_ID` | Update ensemble (`--model-ids`, `--strategy`, `--reason`) |
| `artenic ensemble train ENSEMBLE_ID` | Dispatch training (`--provider`, `--config-json`) |
| `artenic ensemble job-status ENSEMBLE_ID JOB_ID` | Get training job status |
| `artenic ensemble versions ENSEMBLE_ID` | Show version history |

### A/B Testing

| Command | Description |
|---------|-------------|
| `artenic ab-test create` | Create test (`--name`, `--service`, `--variants JSON`, `--primary-metric`) |
| `artenic ab-test list` | List tests (`--service`, `--status`) |
| `artenic ab-test get TEST_ID` | Get test details |
| `artenic ab-test results TEST_ID` | Get aggregated results and metrics |
| `artenic ab-test conclude TEST_ID` | Conclude test (`--winner`, `--reason`) |
| `artenic ab-test pause TEST_ID` | Pause a running test |
| `artenic ab-test resume TEST_ID` | Resume a paused test |

### Runtime Settings

| Command | Description |
|---------|-------------|
| `artenic settings schema` | Show configuration schema |
| `artenic settings get SCOPE [SECTION]` | Get current settings |
| `artenic settings update SCOPE SECTION` | Update settings (`--set KEY=VALUE`, repeatable) |
| `artenic settings audit` | Show settings audit log |

### Local Configuration

| Command | Description |
|---------|-------------|
| `artenic config show` | Show current effective configuration |
| `artenic config set KEY VALUE` | Set a config value (`--profile`) |
| `artenic config use-profile NAME` | Switch active profile |

## Configuration

### Precedence (highest to lowest)

1. **CLI flags** (`--url`, `--api-key`, `--timeout`, `--profile`)
2. **Environment variables** (`ARTENIC_URL`, `ARTENIC_API_KEY`, `ARTENIC_TIMEOUT`, `ARTENIC_PROFILE`)
3. **TOML config file** (`~/.artenic/config.toml`)
4. **Defaults** (url=`http://localhost:9000`, timeout=`30s`, profile=`default`)

### Config File

```toml
[default]
url = "http://localhost:9000"
api_key = "sk-your-key"
timeout = 30

[staging]
url = "https://staging.artenic.cloud"
api_key = "sk-staging-key"
timeout = 60
```

Switch profiles:

```bash
# Via flag
artenic --profile staging model list

# Via env var
export ARTENIC_PROFILE=staging
artenic model list

# Persist active profile
artenic config use-profile staging
```

## Architecture

```
packages/cli/src/artenic_ai_cli/
├── __init__.py                # Package version
├── main.py                    # Root CLI group + error handling
├── _async.py                  # asyncio.run() bridge (Click is sync)
├── _client.py                 # Async HTTP client (httpx)
├── _config.py                 # TOML config loader + saver
├── _context.py                # CliContext dataclass (shared state)
├── _output.py                 # Rich output (tables, dicts, JSON)
│
└── commands/
    ├── health.py              # artenic health *
    ├── models.py              # artenic model *
    ├── inference.py           # artenic predict / predict-batch
    ├── training.py            # artenic training *
    ├── budgets.py             # artenic budget *
    ├── ensembles.py           # artenic ensemble *
    ├── ab_tests.py            # artenic ab-test *
    ├── settings.py            # artenic settings *
    └── config_cmd.py          # artenic config *
```

## Security

- **Credential masking** — API keys are masked in `config show` output (`***...cdef`)
- **Sensitive value protection** — `config set api_key` never prints the full value
- **Error sanitization** — error messages containing credentials (bearer, authorization, etc.) are replaced with a generic message
- **TOML injection prevention** — values are properly escaped before writing to config file
- **Config directory permissions** — `~/.artenic/` is created with `0o700` on Unix systems
- **Input validation** — JSON parameters, `key=value` tags, and `KEY=VALUE` settings pairs are validated before use

## Development

```bash
# From monorepo root
uv sync --dev

# Run CLI tests
uv run pytest packages/cli/tests/ -v

# Quality checks
uv run ruff check packages/cli/
uv run ruff format --check packages/cli/
uv run mypy packages/cli/

# Coverage (100% required)
uv run pytest packages/cli/tests/ --cov=artenic_ai_cli --cov-fail-under=100
```

### Test Architecture

All tests use Click's `CliRunner` for isolated command invocation. API calls are mocked via `unittest.mock.patch` on the async HTTP client — no platform server required.

- **159 tests**, 100% coverage
- mypy strict — 0 errors
- ruff lint + format — clean

## License

Apache License 2.0 — see [LICENSE](../../LICENSE) for details.
