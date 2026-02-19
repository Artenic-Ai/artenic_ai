"""Shared test fixtures for CLI tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from artenic_ai_cli._client import ApiClient
from artenic_ai_cli._config import CliConfig
from artenic_ai_cli._context import CliContext


@pytest.fixture
def runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_response() -> Any:
    """Factory for mock httpx responses."""

    def _make(
        status_code: int = 200,
        json_data: Any = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data if json_data is not None else {}
        resp.text = text
        resp.headers = headers or {}
        return resp

    return _make


@pytest.fixture
def cli_context() -> CliContext:
    """A CliContext with a mock ApiClient for unit tests."""
    api = ApiClient("http://test:9000", api_key="test-key")
    return CliContext(
        api=api,
        console=Console(file=MagicMock(), no_color=True),
        err_console=Console(file=MagicMock(), stderr=True, no_color=True),
        json_mode=False,
        config=CliConfig(),
    )


@pytest.fixture
def patch_run_async() -> Any:
    """Patch run_async to return a predetermined value.

    Usage::

        def test_foo(runner, patch_run_async):
            with patch_run_async({"status": "ok"}) as m:
                result = runner.invoke(cli, ["health", "check"])
                assert result.exit_code == 0
    """
    from contextlib import contextmanager

    @contextmanager
    def _patch(return_value: Any = None, side_effect: Any = None):  # type: ignore[no-untyped-def]
        target = "artenic_ai_cli._async.run_async"

        _real_se = side_effect

        def _close_and_apply(coro: Any) -> Any:
            # Close the unawaited coroutine to suppress RuntimeWarning
            if hasattr(coro, "close"):
                coro.close()
            if _real_se is not None:
                if callable(_real_se) and not isinstance(_real_se, type):
                    return _real_se(coro)
                raise _real_se
            return return_value

        with patch(target, side_effect=_close_and_apply) as m:
            yield m

    return _patch
