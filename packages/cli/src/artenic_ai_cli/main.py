"""Root CLI entry point — ``artenic`` command group."""

from __future__ import annotations

import functools
from typing import Any

import click
from rich.console import Console

from artenic_ai_cli import __version__
from artenic_ai_cli._client import ApiClient
from artenic_ai_cli._config import load_config
from artenic_ai_cli._context import CliContext
from artenic_ai_sdk.exceptions import (
    ArtenicAIError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
)

_SENSITIVE_KEYWORDS = ("bearer", "authorization", "api_key", "password", "secret", "token")

# ---------------------------------------------------------------------------
# Error-handling decorator
# ---------------------------------------------------------------------------


def handle_errors(fn: Any) -> Any:
    """Catch SDK / HTTP exceptions and render user-friendly messages."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except click.exceptions.Exit:
            raise
        except AuthenticationError:
            _die("Authentication failed. Check your API key (--api-key or ARTENIC_API_KEY).")
        except RateLimitError as exc:
            after = getattr(exc, "retry_after", None)
            msg = f"Rate limited. Retry after {after}s." if after else "Rate limited."
            _die(msg)
        except ServiceUnavailableError as exc:
            _die(str(exc) or "Platform unavailable.")
        except ArtenicAIError as exc:
            _die(str(exc))
        except click.ClickException:
            raise
        except Exception as exc:
            msg = str(exc)
            if any(kw in msg.lower() for kw in _SENSITIVE_KEYWORDS):
                msg = "An unexpected error occurred."
            _die(f"Unexpected error: {msg}")

    return wrapper


def _die(message: str) -> None:
    raise click.ClickException(message)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--url", default=None, help="Platform base URL.")
@click.option("--api-key", default=None, help="API key for authentication.")
@click.option("--profile", default=None, help="Config profile name.")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output raw JSON.")
@click.option("--timeout", type=float, default=None, help="Request timeout in seconds.")
@click.version_option(version=__version__, prog_name="artenic")
@click.pass_context
def cli(
    ctx: click.Context,
    url: str | None,
    api_key: str | None,
    profile: str | None,
    json_mode: bool,
    timeout: float | None,
) -> None:
    """Artenic AI Platform CLI."""
    cfg = load_config(url=url, api_key=api_key, timeout=timeout, profile=profile)
    ctx.obj = CliContext(
        api=ApiClient(cfg.url, cfg.api_key, cfg.timeout),
        console=Console(),
        err_console=Console(stderr=True),
        json_mode=json_mode,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Register command groups (lazy imports to keep startup fast)
# ---------------------------------------------------------------------------


def _register_commands() -> None:
    from artenic_ai_cli.commands.ab_tests import ab_test_group
    from artenic_ai_cli.commands.budgets import budget_group
    from artenic_ai_cli.commands.config_cmd import config_group
    from artenic_ai_cli.commands.datasets import dataset_group
    from artenic_ai_cli.commands.ensembles import ensemble_group
    from artenic_ai_cli.commands.health import health_group
    from artenic_ai_cli.commands.inference import predict_batch_cmd, predict_cmd
    from artenic_ai_cli.commands.models import model_group
    from artenic_ai_cli.commands.settings import settings_group
    from artenic_ai_cli.commands.training import training_group

    cli.add_command(health_group)
    cli.add_command(model_group)
    cli.add_command(training_group)
    cli.add_command(predict_cmd)
    cli.add_command(predict_batch_cmd)
    cli.add_command(ensemble_group)
    cli.add_command(ab_test_group)
    cli.add_command(budget_group)
    cli.add_command(dataset_group)
    cli.add_command(settings_group)
    cli.add_command(config_group)


_register_commands()
